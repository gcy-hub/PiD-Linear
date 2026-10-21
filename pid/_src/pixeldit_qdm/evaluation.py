"""Prompt-only fixed galleries: reuse Gemma conditions, discard every image latent."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from PIL import Image
from tqdm import tqdm
from pid._src.linear_pid.data import atomic_json
from pid._src.linear_pid.evaluation import asset_cases,save_views,tensor_image
from pid._src.vdn_pid.attention import execution_mode
from pid._src.vdn_pid.checkpoint import delta_state,load_state,load_delta,verify_base
from pid._src.vdn_pid.config import VDNConfig,local_layout
from pid._src.vdn_pid.training import build_student
from pid._src.linear_pid.training import init_distributed
from .runtime import sample


def render_gallery(net,config,output,rank=0,world=1,tracker=None,step=0,suite='qualitative',include_4k=False,
                   limit=0,model_identity=None,should_stop=None,completion=None,**unused):
    cases,meta=asset_cases(config.gallery_root,'quick' if suite=='quick' else 'full',include_4k)
    if limit:cases=cases[:limit]
    index=json.loads((Path(config.index_root)/'index.json').read_text())
    if meta['data_fingerprint']!=index.get('source_fingerprint',index['fingerprint']):raise ValueError('Gallery/source mismatch')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    baseline=Path(config.output_root)/'original-gallery';baseline.mkdir(parents=True,exist_ok=True)
    device=next(net.parameters()).device;was_training=net.training;net.eval();complete=True;measurements=[]
    null_conditions=getattr(net,'pixeldit_null_conditions',None)
    if null_conditions is None:
        if config.text_cache_root:
            from .conditioning import load_null_cache
            special=load_null_cache(config.text_cache_root)
            null_conditions=(special['null_embs'].to(device),special['null_mask'].to(device))
        else:
            from .conditioning import PixelDiTTextEncoder
            from pid._src.linear_pid.text_encoder import NEGATIVE_PROMPT
            encoder=PixelDiTTextEncoder(config.weights_root,device)
            null_conditions=encoder.encode_raw([NEGATIVE_PROMPT])
            del encoder
    if model_identity is None:
        digest=hashlib.sha256()
        for name,tensor in sorted(delta_state(net).items()):
            digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        model_identity=digest.hexdigest()
    settings=dict(backbone='pixeldit_t2i',conditioning='prompt-only',steps=50,cfg=2.75,shift=config.flow_shift,
                  training_resolution=config.resolution,
                  local_attention=local_layout(config),memory_mode=config.memory_mode,model_identity=model_identity)
    if config.remote_rope:
        settings['remote_rope'] = True
    try:
        with torch.no_grad():
            for case in tqdm(cases[rank::world],desc=f'PixelDiT gallery rank{rank}'):
                if should_stop and should_stop():complete=False;break
                resolution=config.resolution if str(case['resolution'])=='2048' else str(case['resolution'])
                identity=f"{case['kind']}_{case['ordinal']:03d}_{resolution}"
                path=output/(identity+'.png');done=path.with_suffix('.done.json')
                current={**settings,'asset_sha256':case['sha256'],'seed':case['seed']}
                if done.exists():
                    old=json.loads(done.read_text())
                    if old['settings']!=current or not path.exists():raise ValueError('Gallery identity changed')
                    measurements.append(old);continue
                asset=torch.load(Path(config.gallery_root)/case['file'],map_location='cpu',weights_only=True)
                asset.pop('latent',None)
                conditions={k:asset[k].to(device) for k in ('caption_embs','caption_mask','null_embs','null_mask')}
                conditions['null_embs'],conditions['null_mask']=null_conditions
                scale=int(resolution)/int(case['resolution'])
                height,width=(max(16,round(asset[k]*scale/16)*16) for k in ('height','width'))
                sample_args=dict(height=height,width=width,seed=case['seed'],steps=50,cfg=2.75,shift=config.flow_shift)
                ref=baseline/(identity+'.png');ref_done=ref.with_suffix('.done.json')
                reference=dict(base=getattr(net,'base_sha256',None) or str((Path(config.teacher_path).resolve(),Path(config.teacher_path).stat().st_size,Path(config.teacher_path).stat().st_mtime_ns)),asset_sha256=case['sha256'],
                               conditioning='prompt-only',**sample_args)
                if not ref_done.exists():
                    with execution_mode(net,teacher=True,inference=True):teacher=sample(net,**conditions,**sample_args)
                    save_views(tensor_image(teacher),ref);atomic_json(ref_done,reference);del teacher
                elif json.loads(ref_done.read_text())!=reference or not ref.exists():raise ValueError('Baseline identity changed')
                if should_stop and should_stop():complete=False;break
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
                with execution_mode(net,inference=True):student=sample(net,**conditions,**sample_args)
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                save_views(tensor_image(student),path)
                with Image.open(ref) as left,Image.open(path) as right:
                    pair=Image.new('RGB',(left.width+right.width,left.height));pair.paste(left,(0,0));pair.paste(right,(left.width,0))
                    pair.save(output/(identity+'.comparison.png'))
                result=dict(settings=current,caption=asset['caption'],height=height,width=width,
                            decode_seconds=seconds,peak_memory_gib=torch.cuda.max_memory_allocated(device)/2**30,
                            original=str(ref),student=str(path))
                atomic_json(done,result);measurements.append(result)
                if tracker is not None and rank==0:tracker.log({f'gallery/{identity}':tracker.Image(str(path),caption=asset['caption'])},step=step)
                del asset,student,conditions
    finally:net.train(was_training)
    atomic_json(output/f'measurements_rank_{rank}.json',measurements)
    if completion is not None:completion['complete']=complete
    return measurements


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--weights-root');p.add_argument('--teacher-checkpoint');p.add_argument('--gallery-root');p.add_argument('--index-root');p.add_argument('--text-cache-root')
    p.add_argument('--output-dir',default='outputs/pixeldit-t2i-qdm-halo2/inference');p.add_argument('--include-4k',action='store_true')
    p.add_argument('--suite',choices=['quick','qualitative'],default='qualitative');p.add_argument('--limit',type=int,default=0)
    args=p.parse_args(argv);payload=load_state(args.checkpoint);config=VDNConfig(**payload['config'])
    if config.backbone!='pixeldit_t2i':raise ValueError('Expected PixelDiT checkpoint')
    for name in ('weights_root','gallery_root','index_root','text_cache_root'):
        if getattr(args,name):setattr(config,name,getattr(args,name))
    if args.teacher_checkpoint:config.teacher_checkpoint=args.teacher_checkpoint
    elif args.weights_root:config.teacher_checkpoint=str(Path(args.weights_root)/payload['metadata']['base_relative_path'])
    verify_base(config,payload);rank,world,device=init_distributed()
    try:
        net=build_student(config,payload['stage'],device);load_delta(net,payload['delta']);net.requires_grad_(False)
        net.base_sha256=payload['metadata']['base_sha256']
        render_gallery(net,config,args.output_dir,rank,world,suite=args.suite,include_4k=args.include_4k,limit=args.limit)
    finally:
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()

if __name__=='__main__':main()
