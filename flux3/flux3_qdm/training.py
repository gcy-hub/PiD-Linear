"""A1 layer alignment, A2 prediction KD and B KD+LoRA on a single frozen base."""
from __future__ import annotations

import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import uuid

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .checkpoint import atomic_json, capture_rng, restore_rng, read_checkpoint, latest_checkpoint, save_checkpoint
from .runtime import GlobalSampler, lr_factor, training_inputs
from .compute import DIT_COMPUTE, dit_autocast

STAGES={'a1':dict(steps=500,save=50,validate=0,warmup=50,subset=20000),
        'a2':dict(steps=1000,save=100,validate=0,warmup=50,subset=20000),
        'b':dict(steps=10000,save=2000,validate=2000,warmup=200,subset=50000)}


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=(*STAGES,'all'),default='all')
    p.add_argument('--weights-root',default=os.getenv('FLUX3_WEIGHTS_ROOT',str(Path.home()/'huggingface_ckpts')))
    dataset=Path.home()/'dataset/MultiAspect-4K-1M'
    p.add_argument('--subset-root',default=os.getenv('FLUX3_SUBSET_ROOT',str(dataset/'pid_vdn_subsets')))
    p.add_argument('--cache-root',default=os.getenv('FLUX3_CACHE_ROOT',str(dataset/'flux3_qdm_cache')))
    p.add_argument('--output-root',default='outputs/flux3-qdm')
    p.add_argument('--resume',default='auto')
    p.add_argument('--init-from',default='')
    p.add_argument('--a1-init-from',default='',help='Warm-start only A1, then load same-run A1/A2 parents for later stages')
    p.add_argument('--retain-lora',action='store_true',help='Keep inherited B-stage adapters active but frozen during A1/A2')
    p.add_argument('--from-base',action='store_true',help='Start this stage directly from original base weights without a parent delta')
    p.add_argument('--training-subset-size',type=int,choices=(20000,50000),default=0,help='Override the stage cache size while preserving original stage defaults otherwise')
    p.add_argument('--qdm-layers',default=None,help='Comma-separated zero-based joint layers; default replaces all 28')
    p.add_argument('--attention-mode',choices=('hybrid','local-only','memory-only'),default='hybrid')
    p.add_argument('--memory-text-write',action='store_true',help='Write valid text K/V into each QDM bank; image memory remains NoPE')
    p.add_argument('--caption-dropout',type=float,default=.1)
    p.add_argument('--warmup-steps',type=int,default=-1,help='Override stage warmup; -1 keeps original stage defaults')
    for name,default in [('halo',0),('region-size',8),('batch-size',1),('grad-accum',1),('workers',4),('threads',1),
                         ('seed',42),('lora-rank',64),('lora-alpha',64),('steps',0),('max-steps',0),
                         ('save-steps',0),('validation-steps',-1),('max-seconds',0),('keep-checkpoints',3),
                         ('log-steps',1),('sample-limit',0)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--swanlab-mode',choices=('offline','cloud','disabled'),default='offline')
    p.add_argument('--allow-batch-change',action='store_true')
    p.add_argument('--no-activation-checkpointing',action='store_true')
    p.add_argument('--skip-validation',action='store_true',help='Explicitly skip images for short correctness tests')
    p.add_argument('--deterministic',action='store_true',help='Deterministic GPU kernels for strict continuation checks')
    args=p.parse_args(argv)
    from .model import normalize_qdm_layers
    try:args.qdm_layers=list(normalize_qdm_layers(args.qdm_layers))
    except ValueError as exc:p.error(str(exc))
    if not args.qdm_layers:p.error('Training needs at least one QDM layer')
    if min(args.batch_size,args.grad_accum,args.workers,args.threads,args.log_steps,args.region_size)<1:
        p.error('batch, accumulation, workers, threads, log interval and region size must be positive')
    if args.a1_init_from and (args.init_from or args.from_base):p.error('--a1-init-from cannot be combined with --init-from or --from-base')
    if args.from_base and args.init_from:p.error('--from-base and --init-from are mutually exclusive')
    if args.from_base and args.stage=='all':p.error('--from-base requires a single explicit stage')
    if args.halo<0:p.error('--halo cannot be negative')
    if args.memory_text_write and args.attention_mode=='local-only':p.error('--memory-text-write requires a QDM branch')
    if args.steps and args.max_steps and args.steps!=args.max_steps:p.error('--steps and --max-steps disagree')
    if args.steps<0 or args.max_steps<0 or args.max_seconds<0:p.error('Limits cannot be negative')
    if args.warmup_steps < -1:p.error('--warmup-steps must be -1 or nonnegative')
    for stage in STAGES if args.stage=='all' else (args.stage,):
        total=args.steps or args.max_steps or STAGES[stage]['steps']
        if args.warmup_steps>=total:p.error('--warmup-steps must be smaller than stage steps')
    if not 0 <= args.caption_dropout <= 1:p.error('--caption-dropout must lie in [0, 1]')
    return args


def stage_settings(args,stage):
    settings=STAGES[stage].copy()
    if getattr(args,'training_subset_size',0):settings['subset']=args.training_subset_size
    settings['steps']=args.steps or args.max_steps or settings['steps']
    settings['save']=args.save_steps or settings['save']
    settings['validate']=settings['validate'] if args.validation_steps<0 else args.validation_steps
    if getattr(args,'warmup_steps',-1)>=0:settings['warmup']=args.warmup_steps
    return settings


def initial_source(args,stage):
    return args.init_from or (getattr(args,'a1_init_from','') if stage=='a1' else '')


def requires_parent(args,stage):
    return bool(initial_source(args,stage)) or (stage!='a1' and not args.from_base)


def consensus(value,device):
    t=torch.tensor(int(bool(value)),device=device)
    if dist.is_initialized():dist.all_reduce(t,op=dist.ReduceOp.MIN)
    return bool(t.item())


def validate_parent(state,stage,config,metadata,parent_data_fingerprint):
    previous={'a2':'a1','b':'a2'}.get(stage)
    if previous and state['config']['stage']!=previous:raise ValueError('Parent checkpoint belongs to a different stage')
    if state['config'].get('qdm_layers',list(range(28)))!=config.get('qdm_layers',list(range(28))):raise ValueError('Parent QDM layout mismatch')
    if state['config'].get('attention_mode','hybrid')!=config.get('attention_mode','hybrid'):raise ValueError('Parent attention_mode mismatch')
    if state['config'].get('memory_text_write',False)!=config.get('memory_text_write',False):raise ValueError('Parent memory_text_write mismatch')
    if state['config'].get('retain_lora',False)!=config.get('retain_lora',False):raise ValueError('Parent retained LoRA mismatch')
    if state['config'].get('halo',0)!=config.get('halo',0):raise ValueError('Parent halo mismatch')
    for key in ('region_size','lora_rank','lora_alpha','seed'):
        if state['config'][key]!=config[key]:raise ValueError(f'Parent {key} mismatch')
    for key in ('base_fingerprint','architecture','native_dimensions','training_policy'):
        if state['metadata'].get(key)!=metadata.get(key):raise ValueError(f'Parent {key} mismatch')
    if state['metadata']['data_fingerprint']!=parent_data_fingerprint:
        raise ValueError('Parent cache fingerprint differs from the intended parent subset')


def alignment_loss(predicted,target,samples):
    """Only the image contribution is adapted in A1; text is a diagnostic."""
    target=target.detach().float();predicted=predicted.float()
    image_error=target.new_zeros(());text_error=target.new_zeros(())
    image_den=target.new_zeros(());text_den=target.new_zeros(())
    for sample in samples:
        si=slice(sample.image_start,sample.image_end);st=slice(sample.text_start,sample.text_end)
        image_error=image_error+(predicted[si]-target[si]).square().sum()
        image_den=image_den+target[si].square().sum()
        text_error=text_error+(predicted[st]-target[st]).square().sum()
        text_den=text_den+target[st].square().sum()
    image_loss=image_error/image_den.clamp_min(1e-30)
    return image_loss,dict(relative_mse=image_loss.detach(),image_relative_mse=image_loss.detach(),
        text_relative_mse=(text_error/text_den.clamp_min(1e-30)).detach(),
        joint_relative_mse=((image_error+text_error)/(image_den+text_den).clamp_min(1e-30)).detach(),
        image_energy_fraction=(image_den/(image_den+text_den).clamp_min(1e-30)).detach())


class Alignment:
    def __init__(self,scale=1.):
        self.scale=scale;self.values={};self.events=[]

    def __call__(self,block,info,target):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        with torch.enable_grad(),dit_autocast():
            predicted=block.student_attention(info)
            loss,metrics=alignment_loss(predicted,target,info.samples)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite A1 alignment loss')
            (loss*self.scale).backward()
        b.record();self.events.append((a,b))
        self.values[block.layer_index if hasattr(block,'layer_index') else id(block)]={k:v.item() for k,v in metrics.items()}


def _run_stage(args,stage,rank,world,device):
    from .data import CachedDataset
    from .model import build_model, execution_mode, optimizer_groups, iter_qdm_blocks, delta_state_dict, load_delta
    from .resources import resource_paths,sha256_file
    settings=stage_settings(args,stage)
    root=Path(args.output_root).resolve();directory=root/stage;directory.mkdir(parents=True,exist_ok=True)
    lease=None
    if rank==0:
        lease=(directory/'.training.lock').open('a')
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if world>1:dist.barrier()
    stop=directory/'STOP'
    if stop.exists():return False
    dataset=CachedDataset(args.subset_root,args.cache_root,settings['subset'],limit=args.sample_limit)
    sampler=GlobalSampler(dataset.bucket_indices,args.batch_size,world,args.grad_accum,args.seed)
    config={**vars(args),'stage':stage,'effective_batch':sampler.effective,'world_size':world,'settings':settings}
    paths=resource_paths(args.weights_root)
    # Fingerprint original bytes once per stage; no model mutation is accepted on resume.
    digest_cache=root/'base_fingerprint.json'
    stat=Path(paths['dit']).stat();signature=dict(path=paths['dit'],size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    fingerprint=None
    if rank==0:
        old=json.loads(digest_cache.read_text()) if digest_cache.exists() else {}
        fingerprint=old.get('sha256') if old.get('signature')==signature else sha256_file(paths['dit'])
        atomic_json(digest_cache,dict(signature=signature,sha256=fingerprint))
    if world>1:
        obj=[fingerprint];dist.broadcast_object_list(obj,src=0);fingerprint=obj[0]
    metadata=dict(base_fingerprint=fingerprint,data_fingerprint=dataset.fingerprint,
                  architecture='flux3-qdm-single28-fixed4',native_dimensions=[3072,24,128],
                  training_policy=dict(a1_objective='image_relative_mse_v1',caption_dropout=args.caption_dropout,
                                       dit_compute=DIT_COMPUTE))
    if args.attention_mode!='hybrid':metadata['attention_mode']=args.attention_mode
    if args.memory_text_write:metadata['memory_text_write']=True
    if args.retain_lora:metadata['retain_lora']=True
    if args.qdm_layers!=list(range(28)):
        metadata['qdm_layers']=args.qdm_layers
    saved=None
    if args.resume=='auto':saved=latest_checkpoint(directory)
    elif args.resume not in ('none',''):
        saved=Path(args.resume)
        if saved.is_dir() and (saved/'checkpoints').exists():saved=latest_checkpoint(saved)
        if saved is None:raise FileNotFoundError(args.resume)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed(args.seed)
    net=build_model(paths['dit'],stage,device,args.region_size,args.lora_rank,args.lora_alpha,qdm_layers=args.qdm_layers,halo=args.halo,attention_mode=args.attention_mode,memory_text_write=args.memory_text_write,retain_lora=args.retain_lora)
    blocks=list(iter_qdm_blocks(net))
    groups=optimizer_groups(net,stage)
    if not groups:raise ValueError('No trainable parameters: local-only adaptation requires --stage b (LoRA and Softmax gate)')
    optimizer=torch.optim.AdamW(groups,weight_decay=0.,foreach=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:lr_factor(step,settings['steps'],settings['warmup']))
    progress=dict(stage_step=0,total_step=0,cursor=0,samples_seen=0,seen_rows=[],skipped_batches=0,validation_pending=False)
    if saved:
        state=read_checkpoint(saved);previous=state['config']
        if previous.get('qdm_layers',list(range(28)))!=args.qdm_layers:raise ValueError('Resume QDM layout mismatch')
        if previous.get('attention_mode','hybrid')!=args.attention_mode:raise ValueError('Resume attention_mode mismatch')
        if previous.get('memory_text_write',False)!=args.memory_text_write:raise ValueError('Resume memory_text_write mismatch')
        if previous.get('retain_lora',False)!=args.retain_lora:raise ValueError('Resume retained LoRA mismatch')
        if previous.get('halo',0)!=args.halo:raise ValueError('Resume halo mismatch')
        for key in ('region_size','lora_rank','lora_alpha','seed','sample_limit'):
            if previous[key]!=config[key]:raise ValueError(f'Resume {key} mismatch')
        if previous['stage']!=stage:raise ValueError('Cannot resume a checkpoint from a different stage')
        if previous.get('deterministic',False)!=args.deterministic:raise ValueError('Resume deterministic-kernel setting mismatch')
        if any(state['metadata'].get(k)!=v for k,v in metadata.items()):raise ValueError('Base/data fingerprint, architecture or training policy changed; use a separate output directory')
        if previous['settings']!=settings:raise ValueError('Stage schedule changed; use a separate output directory')
        if previous['effective_batch']!=sampler.effective and not args.allow_batch_change:
            raise ValueError('Effective batch changed; explicitly set --allow-batch-change to record a new draw schedule')
        load_delta(net,torch.load(saved/'delta.pt',map_location='cpu',weights_only=True))
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);progress=state['progress']
        restore_rng(state['rng'][rank%len(state['rng'])])
        config['resumed_from']=str(saved)
    elif requires_parent(args,stage):
        previous_stage={'a2':'a1','b':'a2'}.get(stage)
        source=initial_source(args,stage)
        parent=Path(source) if source else latest_checkpoint(root/previous_stage)
        if parent is None:raise FileNotFoundError(f'No parent checkpoint for stage {stage}')
        state=read_checkpoint(parent)
        parent_fingerprint=CachedDataset(args.subset_root,args.cache_root,state['config']['settings']['subset'],limit=args.sample_limit).fingerprint if previous_stage else metadata['data_fingerprint']
        validate_parent(state,stage,config,metadata,parent_fingerprint)
        inherited=torch.load(parent/'delta.pt',map_location='cpu',weights_only=True)
        result=load_delta(net,inherited,strict=False)
        if result.unexpected_keys or any('lora_' not in k for k in result.missing_keys):
            raise ValueError(f'Parent delta mismatch: {result}')
        progress['total_step']=state['progress']['total_step'];config['parent_checkpoint']=str(parent)
    if rank==0:
        atomic_json(directory/'config.json',config)
        print(json.dumps(dict(stage=stage,step=progress['stage_step'],parameters=sum(p.numel() for p in net.parameters()),
                              trainable=sum(p.numel() for p in net.parameters() if p.requires_grad),effective_batch=sampler.effective,
                              qdm_layers=args.qdm_layers,full_attention_layers=[i for i in range(28) if i not in args.qdm_layers],halo=args.halo,attention_mode=args.attention_mode,memory_text_write=args.memory_text_write)),flush=True)
    model=DDP(net,device_ids=[device.index],broadcast_buffers=False,find_unused_parameters=True) if world>1 and stage!='a1' else net
    tracker=None;tracking=json.loads((directory/'swanlab.json').read_text()) if (directory/'swanlab.json').exists() else {}
    if rank==0 and args.swanlab_mode!='disabled':
        import swanlab
        run_id=tracking.get('run_id') or uuid.uuid4().hex[:16]
        tracker=swanlab.init(project='FLUX3-QDM',name=stage,mode='online' if args.swanlab_mode=='cloud' else args.swanlab_mode,
                             log_dir=str(directory/'swanlog'),id=run_id,resume=True,config=config)
        atomic_json(directory/'swanlab.json',dict(run_id=run_id,mode=args.swanlab_mode))
        metadata['swanlab_run_id']=run_id
    seen=set(progress['seen_rows']);interrupted={'value':False}
    def handler(signum,frame):interrupted['value']=True
    old_handlers={s:signal.signal(s,handler) for s in (signal.SIGTERM,signal.SIGINT)}
    start=time.monotonic();last_path=saved
    pool=ThreadPoolExecutor(max_workers=args.workers)

    def save(milestone=False):
        progress['seen_rows']=sorted(seen)
        rng=capture_rng();states=[None]*world
        if world>1:dist.all_gather_object(states,rng)
        else:states[0]=rng
        value=None;error=None
        if rank==0:
            try:
                value=str(save_checkpoint(directory,delta_state_dict(net),optimizer,scheduler,progress,config,metadata,
                                          keep=args.keep_checkpoints,rng=states,milestone=milestone))
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        if world>1:
            obj=[value,error];dist.broadcast_object_list(obj,src=0);value,error=obj
        if error:raise RuntimeError(error)
        return Path(value)

    try:
        while progress['stage_step']<settings['steps']:
            tick=time.monotonic();torch.cuda.reset_peak_memory_stats(device);optimizer.zero_grad(set_to_none=True)
            callback=Alignment(1/args.grad_accum);kd=fms=fmt=0.;teacher_ms=student_ms=0.;skipped=False
            null_samples=conditioned_samples=0
            for micro in range(args.grad_accum):
                samples=None;load_error=''
                try:samples=list(pool.map(dataset.__getitem__,sampler.rank_batch(progress['cursor'],rank,micro)))
                except Exception as exc:load_error=f'{type(exc).__name__}: {exc}'
                if not consensus(not load_error,device):
                    if load_error:
                        with (directory/f'bad_batches_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(dict(cursor=progress['cursor'],error=load_error))+'\n')
                    skipped=True;break
                packed,target=training_inputs(samples,args.seed,stage,progress['cursor'],device,
                    caption_dropout=args.caption_dropout,empty_prompt=dataset.empty_prompt)
                null_samples+=sum(packed.caption_dropout_mask);conditioned_samples+=len(samples)
                events=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
                events[0].record()
                with execution_mode(net,teacher=True),torch.no_grad(),dit_autocast():
                    teacher=net(packed,alignment_callback=callback if stage=='a1' else None).detach()
                events[1].record();events[2].record()
                if stage=='a1':
                    kd+=sum(v['relative_mse'] for v in callback.values.values())/len(blocks)/args.grad_accum
                else:
                    sync=model.no_sync() if world>1 and micro<args.grad_accum-1 else contextlib.nullcontext()
                    with sync,execution_mode(net,teacher=False),dit_autocast():
                        student=model(packed,use_gradient_checkpointing=not args.no_activation_checkpointing)
                        loss=torch.nn.functional.mse_loss(student.float(),teacher.float())
                        if not consensus(bool(torch.isfinite(loss)),device):raise FloatingPointError('nonfinite KD loss')
                        (loss/args.grad_accum).backward()
                    kd+=loss.detach().item()/args.grad_accum
                    fms+=torch.nn.functional.mse_loss(student.detach().float(),target).item()/args.grad_accum
                    del student,loss
                fmt+=torch.nn.functional.mse_loss(teacher.float(),target).item()/args.grad_accum
                events[3].record();events[3].synchronize()
                align_ms=sum(a.elapsed_time(b) for a,b in callback.events) if stage=='a1' else 0.
                callback.events.clear()
                teacher_ms+=events[0].elapsed_time(events[1])-align_ms
                student_ms+=align_ms+events[2].elapsed_time(events[3])
                del packed,target,teacher,samples
            rows=sampler.global_batch(progress['cursor']);progress['cursor']+=1
            if skipped:
                optimizer.zero_grad(set_to_none=True);progress['skipped_batches']+=1
                if progress['skipped_batches']>=10:raise RuntimeError('Too many unreadable global batches; repair cache')
                continue
            if stage=='a1' and world>1:
                for p in net.parameters():
                    if p.requires_grad:
                        if p.grad is None:p.grad=torch.zeros_like(p)
                        dist.all_reduce(p.grad);p.grad.div_(world)
            norm=max((torch.nn.utils.clip_grad_norm_(b.parameters(),.1) for b in blocks),key=float) if stage=='a1' else torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            if not consensus(bool(torch.isfinite(norm)),device):raise FloatingPointError('nonfinite gradient norm')
            optimizer.step();scheduler.step();progress['stage_step']+=1;progress['total_step']+=1
            progress['samples_seen']+=sampler.effective;seen.update(dataset.rows[i]['id'] for i in rows)
            torch.cuda.synchronize(device);elapsed=time.monotonic()-tick
            counts=torch.tensor([null_samples,conditioned_samples],device=device,dtype=torch.int64)
            if world>1:dist.all_reduce(counts)
            progress['unconditional_samples_seen']=progress.get('unconditional_samples_seen',0)+int(counts[0])
            numbers=torch.tensor([kd,fms,fmt,teacher_ms,student_ms],device=device)
            maxima=torch.tensor([elapsed,torch.cuda.max_memory_allocated(device)/2**30],device=device)
            if world>1:
                dist.all_reduce(numbers);numbers/=world;dist.all_reduce(maxima,op=dist.ReduceOp.MAX)
            if progress['stage_step']%args.log_steps==0:
                metrics=dict(zip(['train/kd_loss','train/fm_student','train/fm_teacher','efficiency/teacher_ms','efficiency/student_forward_backward_ms'],numbers.tolist()))
                if stage=='a1':metrics.pop('train/fm_student')
                metrics.update({'efficiency/seconds_per_update':maxima[0].item(),'efficiency/peak_memory_gib':maxima[1].item(),
                                'train/caption_dropout_fraction':float(counts[0]/counts[1]),
                                'stage/unconditional_samples_seen':progress['unconditional_samples_seen'],
                                'efficiency/images_per_second':sampler.effective/maxima[0].item(),'train/grad_norm':float(norm),
                                'stage/step':progress['stage_step'],'stage/samples_seen':progress['samples_seen'],
                                'stage/unique_samples':len(seen),'stage/equivalent_epochs':progress['samples_seen']/len(dataset)})
                for group in optimizer.param_groups:metrics['lr/'+group['group_name']]=group['lr']
                for block in blocks:
                    i=block.layer_index
                    if stage=='a1':
                        vals=callback.values.get(i,{})
                        for k,v in vals.items():
                            t=torch.tensor(v,device=device)
                            if world>1:dist.all_reduce(t);t/=world
                            metrics[f'alignment/layer_{i}/{k}']=float(t)
                if rank==0:
                    with (directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=progress['total_step'],**metrics))+'\n')
                    print(f"{stage} step={progress['stage_step']}/{settings['steps']} KD={numbers[0].item():.6f} {maxima[0].item():.2f}s/update peak={maxima[1].item():.2f}GiB",flush=True)
                    if tracker:tracker.log(metrics,step=progress['total_step'])
            finished=progress['stage_step']>=settings['steps']
            validation=not args.skip_validation and (finished or settings['validate']>0 and progress['stage_step']%settings['validate']==0)
            stopping=not consensus(not(interrupted['value'] or stop.exists() or args.max_seconds and time.monotonic()-start>=args.max_seconds),device)
            progress['validation_pending']=bool(validation)
            if finished or validation or stopping or progress['stage_step']%settings['save']==0:
                last_path=save(milestone=finished or validation)
            if rank==0 and validation:
                atomic_json(directory/'validation_pending.json',dict(checkpoint=str(last_path),stage=stage,final=finished))
            if stopping or validation:break
        finished=progress['stage_step']>=settings['steps']
        if finished:
            if last_path is None:last_path=save(milestone=True)
            if rank==0:
                atomic_json(directory/'training_finished.json',dict(checkpoint=str(last_path),stage=stage))
                if args.skip_validation:atomic_json(directory/'finished.json',dict(checkpoint=str(last_path),stage=stage,validation_skipped=True))
        return finished and args.skip_validation
    finally:
        pool.shutdown();optimizer.zero_grad(set_to_none=True)
        if tracker:tracker.finish()
        for s,h in old_handlers.items():signal.signal(s,h)
        if lease:lease.close()
        del model,net,optimizer,scheduler,blocks
        gc.collect();torch.cuda.empty_cache()


def main(argv=None):
    args=parse_args(argv);torch.set_num_threads(args.threads)
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError('FLUX.3 full training requires a CUDA GPU')
    rank=int(os.getenv('RANK','0'));world=int(os.getenv('WORLD_SIZE','1'));local=int(os.getenv('LOCAL_RANK','0'))
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(minutes=10),device_id=device)
    try:
        for stage in STAGES if args.stage=='all' else (args.stage,):
            completed=_run_stage(args,stage,rank,world,device)
            if not completed:break
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
