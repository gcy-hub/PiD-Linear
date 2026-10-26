"""Original-teacher quadrant masks and trained-QDM boundary interventions; no optimization."""
import argparse
import fcntl
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from .partition_diagnostics import (QuadrantMaskAttention, quadrant_labels, error_metrics,
                                    token_map, spatial_summary, discontinuity)


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.incomplete')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
    os.replace(tmp,path)


def save_tensor(path,value):
    tmp=path.with_suffix('.incomplete');torch.save(value,tmp);os.replace(tmp,path)


def save_image(path,value):
    from PIL import Image
    z=value[0].float().clamp(-1,1).add(1).mul(127.5).round().byte().permute(1,2,0).cpu().numpy()
    Image.fromarray(z).save(path)


def load_base(kind,weights):
    if kind=='pixeldit':
        from .runtime import load_net,checkpoint_path
        net=load_net(checkpoint_path(weights/'PixelDiT-T2I'))
    else:
        from pid._src.linear_pid.runtime import build_net
        from pid._src.vdn_pid.config import BASE_RELATIVE
        with torch.device('meta'):net=build_net()
        state=torch.load(weights/BASE_RELATIVE,map_location='cpu',mmap=True,weights_only=True)
        if not all(k.startswith('net.') for k in state):raise ValueError('Unexpected PiD checkpoint prefixes')
        net.load_state_dict({k[4:]:v for k,v in state.items()},strict=True,assign=True)
    net.requires_grad_(False).eval().to(device='cuda',dtype=torch.bfloat16)
    net.activation_checkpointing=False;net.pit_chunk_size=2048
    return net


def load_student(kind,weights,checkpoint):
    from pid._src.vdn_pid.attention import convert_attention
    from pid._src.vdn_pid.spatial_memory import replace_memory
    from pid._src.vdn_pid.checkpoint import load_state,load_delta
    payload=load_state(checkpoint);cfg=payload['config']
    expected='pixeldit_t2i' if kind=='pixeldit' else 'pid'
    if cfg.get('backbone','pid')!=expected or cfg['memory_mode']!='fixed4':raise ValueError('Wrong backbone/memory checkpoint')
    net=load_base(kind,weights)
    convert_attention(net,cfg['region_size'],local_mode=cfg.get('local_mode','window'),
                      local_halo=cfg.get('local_halo',0),local_shift=cfg.get('local_shift',4),
                      local_chunk=cfg.get('local_chunk',16),remote_rope=cfg.get('remote_rope',False),
                      full_attention_layers=cfg.get('full_attention_layers',''))
    replace_memory(net,mode='fixed',banks=4)
    if payload['stage']=='b':
        from pid._src.vdn_pid.lora import configure_stage
        configure_stage(net,'b',cfg['lora_rank'],cfg['lora_alpha'])
    load_delta(net,payload['delta']);net.to('cuda').requires_grad_(False).eval()
    return net,dict(checkpoint=str(Path(checkpoint).resolve()),stage=payload['stage'],
                    progress=payload['progress'],remote_rope=cfg.get('remote_rope',False),
                    local_mode=cfg.get('local_mode','window'),halo=cfg.get('local_halo',0))


def load_asset(args):
    a=torch.load(args.assets/(args.case+'.pt'),map_location='cpu',weights_only=False)
    if args.model=='pixeldit':
        from .conditioning import load_null_cache
        special=load_null_cache(args.text_cache)
        a['null_embs'],a['null_mask']=special['null_embs'],special['null_mask']
        a['height']=max(16,round(a['height']/2/16)*16);a['width']=max(16,round(a['width']/2/16)*16)
        a.pop('latent',None)
    return {k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in a.items()}


def call(net,a,x,t,conditional=True,kind='pid'):
    y=a['caption_embs' if conditional else 'null_embs'];mask=a['caption_mask' if conditional else 'null_mask']
    kw=dict(text_valid_mask=mask)
    if kind=='pid':kw.update(lq_latent=a['latent'],degrade_sigma=torch.zeros(1,device='cuda'))
    if kind=='pixeldit':x,t=x.bfloat16(),t.bfloat16()
    return net(x,t,y,**kw)


def snapshots(net,a,args,root,teacher_context):
    """Capture conditional inputs on the native teacher trajectory, never student trajectories."""
    path=root/'teacher-trajectory.pt'
    if path.exists():
        data=torch.load(path,map_location='cpu',weights_only=False)
        return [(item['target'],item['x'].cuda(),item['t'].cuda()) for item in data]
    if args.input_protocol=='noised-reference':
        from PIL import Image
        im=Image.open(args.reference_image).convert('RGB')
        if im.size!=(a['width'],a['height']):raise ValueError('Reference image must match the native asset dimensions exactly')
        clean=torch.from_numpy(np.array(im)).cuda().float().permute(2,0,1)[None]/127.5-1
        noise=torch.randn(clean.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(a['seed']))
        data=[dict(target=t,x=((1-t/1000)*clean+(t/1000)*noise).cpu(),t=torch.tensor([float(t)])) for t in args.times]
        save_tensor(path,data);im.save(root/'original.png')
        return [(item['target'],item['x'].cuda(),item['t'].cuda()) for item in data]
    from .runtime import sample as pixel_sample
    from pid._src.linear_pid.runtime import sample as pid_sample
    targets=args.times;best={};calls=[0]
    def capture(module,inputs,kwargs):
        index=calls[0];calls[0]+=1
        if index%2==0:return # Sequential CFG uses unconditional, then conditional.
        x,t=inputs[:2];value=float(t[0])
        for target in targets:
            distance=abs(value-target)
            if target not in best or distance<best[target][0]:
                best[target]=(distance,x.detach().clone(),t.detach().clone())
    hook=net.register_forward_pre_hook(capture,with_kwargs=True)
    try:
        fn=pid_sample if args.model=='pid' else pixel_sample
        with teacher_context():
            image=fn(net,a['caption_embs'],a['caption_mask'],a['null_embs'],a['null_mask'],a.get('latent'),
                     height=a['height'],width=a['width'],seed=a['seed'],
                     steps=25 if args.model=='pid' else 50,cfg=5 if args.model=='pid' else 2.75,
                     shift=6 if args.model=='pid' else 4)
        save_image(root/'original.png',image)
    finally:hook.remove()
    if len(best)!=len(targets):raise RuntimeError('Did not capture all requested conditional times')
    data=[dict(target=t,x=best[t][1].cpu(),t=best[t][2].cpu()) for t in targets]
    save_tensor(path,data)
    return [(item['target'],item['x'].cuda(),item['t'].cuda()) for item in data]


def layouts():
    # Nested and distributed throughout depth; layer zero's input is untouched.
    order=[0,4,9,13,2,6,8,11,1,3,5,7,10,12]
    return {str(n):sorted(order[:n]) for n in (1,4,8,14)}


def run_mask(net,a,args,root):
    wrappers=[]
    for block in net.patch_blocks:
        block.attn=QuadrantMaskAttention(block.attn);wrappers.append(block.attn)
    # A previous boundary worker may already have the identical frozen-teacher trajectory.
    if not (root/'teacher-trajectory.pt').exists():
        for source in sorted((args.output/'boundary'/args.model).glob('*/'+args.case)):
            if not (source/'teacher-trajectory.pt').exists():continue
            cfg=json.loads((source/'config.json').read_text());old=cfg['arguments']
            if (old.get('input_protocol','trajectory')=='trajectory' and old['times']==args.times
                and Path(old['assets']).resolve()==args.assets.resolve()
                and Path(old['weights']).resolve()==args.weights.resolve()
                and cfg['seed']==a['seed'] and (cfg['height'],cfg['width'])==(a['height'],a['width'])):
                os.link(source/'teacher-trajectory.pt',root/'teacher-trajectory.pt')
                os.link(source/'original.png',root/'original.png')
                break
    data=snapshots(net,a,args,root,nullcontext)
    for target,x,t in data:
        dest=root/f'target_{target}.json'
        if dest.exists():continue
        rows=[]
        for index,w in enumerate(wrappers):
            def callback(ref,changed,grid,i=index):
                rows.append(dict(layer=i,image=error_metrics(changed[0],ref[0]),
                                 text=error_metrics(changed[1],ref[1]),
                                 error_spatial=spatial_summary(token_map(changed[0],ref[0],grid))))
            w.callback=callback
        reference=call(net,a,x,t,kind=args.model)
        for w in wrappers:w.callback=None
        null=call(net,a,x,t,conditional=False,kind=args.model)
        cfg=5 if args.model=='pid' else 2.75
        guided=null+cfg*(reference-null)
        individual=[]
        for index,w in enumerate(wrappers):
            w.masked=True
            value=call(net,a,x,t,kind=args.model)
            individual.append(dict(layer=index,prediction=error_metrics(value,reference)))
            w.masked=False
        cumulative=[]
        for count,indices in layouts().items():
            for i,w in enumerate(wrappers):w.masked=i in indices
            value=call(net,a,x,t,kind=args.model)
            value_null=call(net,a,x,t,conditional=False,kind=args.model)
            cumulative.append(dict(count=int(count),layers=indices,prediction=error_metrics(value,reference),
                                   guided_prediction=error_metrics(value_null+cfg*(value-value_null),guided)))
        for w in wrappers:w.masked=False
        write_json(dest,dict(target=target,actual_t=float(t[0]),attention=rows,single_layer=individual,cumulative=cumulative))
        print(f'{args.model} {args.case} mask target={target} t={float(t[0]):.2f} saved',flush=True)
    if args.generate:
        from .runtime import sample as pixel_sample
        from pid._src.linear_pid.runtime import sample as pid_sample
        fn=pid_sample if args.model=='pid' else pixel_sample
        for count in args.generate:
            path=root/f'mask_{count}.png'
            if path.exists():continue
            for i,w in enumerate(wrappers):w.masked=i in layouts()[str(count)]
            image=fn(net,a['caption_embs'],a['caption_mask'],a['null_embs'],a['null_mask'],a.get('latent'),
                     height=a['height'],width=a['width'],seed=a['seed'],steps=25 if args.model=='pid' else 50,
                     cfg=5 if args.model=='pid' else 2.75,shift=6 if args.model=='pid' else 4)
            save_image(path,image)
            print(f'{args.model} {args.case} mask {count}-layer image saved',flush=True)
        for w in wrappers:w.masked=False


def run_control(net,a,args,root):
    wrappers=[]
    for block in net.patch_blocks:
        block.attn=QuadrantMaskAttention(block.attn);wrappers.append(block.attn)
        block.attn.restrict_quadrants=False
    source=args.output/'mask'/args.model/args.case
    for name in ('teacher-trajectory.pt','original.png'):
        if not (root/name).exists():os.link(source/name,root/name)
    data=snapshots(net,a,args,root,nullcontext)
    rows=[]
    for target,x,t in data:
        for w in wrappers:w.masked=False
        reference=call(net,a,x,t,kind=args.model)
        for w in wrappers:w.masked=True
        value=call(net,a,x,t,kind=args.model)
        rows.append(dict(target=target,actual_t=float(t[0]),prediction=error_metrics(value,reference)))
    write_json(root/'numerical-control.json',rows)
    from .runtime import sample as pixel_sample
    from pid._src.linear_pid.runtime import sample as pid_sample
    fn=pid_sample if args.model=='pid' else pixel_sample
    image=fn(net,a['caption_embs'],a['caption_mask'],a['null_embs'],a['null_mask'],a.get('latent'),
             height=a['height'],width=a['width'],seed=a['seed'],steps=25 if args.model=='pid' else 50,
             cfg=5 if args.model=='pid' else 2.75,shift=6 if args.model=='pid' else 4)
    save_image(root/'full-bank-split.png',image)


def shifted_remote(module,x,qkv,grid,pos_img,split):
    from pid._src.vdn_pid.spatial_memory import joint_memory
    base=module.linear_attention.base
    q,k,v=base.features(qkv,grid,pos_img=pos_img if module.remote_rope else None)
    beta=torch.sigmoid(base.beta_proj(x));b,n,c=x.shape
    routes=F.one_hot(quadrant_labels(grid,x.device,split),4).float()[None].expand(b,-1,-1)
    out,_=joint_memory(q,k,v,beta,routes,sparse=True)
    out=out.to(q.dtype)
    out=base.norm(out)*base.output_gate(x.reshape(-1,c)).reshape_as(out)
    return base.to_out_linear(out.reshape(b,n,c))


def run_boundary(net,a,args,root):
    from pid._src.vdn_pid.attention import execution_mode
    modules=[b.attn for b in net.patch_blocks]
    def teacher():return execution_mode(net,teacher=True,inference=True)
    # Reuse an original teacher trajectory from priority 1 when available.
    source=args.output/'mask'/args.model/args.case
    if args.input_protocol=='trajectory' and (source/'teacher-trajectory.pt').exists() and not (root/'teacher-trajectory.pt').exists():
        os.link(source/'teacher-trajectory.pt',root/'teacher-trajectory.pt')
        if (source/'original.png').exists():os.link(source/'original.png',root/'original.png')
    data=snapshots(net,a,args,root,teacher)
    for target,x,t in data:
        dest=root/f'target_{target}.json'
        if dest.exists():continue
        rows=[];maps={}
        for index,m in enumerate(modules):
            def callback(module,inputs,kw,ref,i=index):
                xx,yy,pi,pt=inputs;grid=kw['grid_size'];h,w=grid
                qkv,tqkv=module.raw_qkv(xx,yy)
                local,local_text=module.local(xx,yy,qkv,tqkv,pi,pt,grid,kw['attn_mask'])
                center=((h+1)//2,(w+1)//2)
                splits=[center,(max(1,3*h//8),max(1,3*w//8)),(min(h-1,5*h//8),min(w-1,5*w//8))]
                variants=[]
                for v,split in enumerate(splits):
                    remote=shifted_remote(module,xx,qkv,grid,pi,split)
                    if v==0:
                        actual=module.linear_attention(xx,yy,qkv,tqkv,grid,kw['text_valid_mask'],True,
                                                       **({'pos_img':pi} if module.remote_rope else {}))
                        torch.testing.assert_close(remote,actual,rtol=0,atol=0)
                        original_remote=remote
                    combined=local+remote
                    e=token_map(combined,ref[0],grid)
                    variants.append(dict(split=split,image=error_metrics(combined,ref[0]),
                        remote_rms=remote.float().square().mean().sqrt().item(),
                        error_spatial=spatial_summary(e,split),old_center_spatial=spatial_summary(e,center),
                        remote_discontinuity=discontinuity(remote,grid),
                        remote_delta=error_metrics(remote,original_remote)))
                    maps[f'l{i:02d}_split{v}_error']=e.cpu().numpy()
                    maps[f'l{i:02d}_split{v}_remote_energy']=token_map(remote,grid=grid).cpu().numpy()
                    # Movement of the remote feature itself under a changed partition.
                    maps[f'l{i:02d}_split{v}_remote_change']=token_map(remote,original_remote,grid).cpu().numpy()
                maps[f'l{i:02d}_local_error']=token_map(local,ref[0],grid).cpu().numpy()
                rows.append(dict(layer=i,grid=grid,local_image=error_metrics(local,ref[0]),
                                 text=error_metrics(local_text,ref[1]),local_discontinuity=discontinuity(local,grid),
                                 teacher_discontinuity=discontinuity(ref[0],grid),variants=variants))
            m.alignment_callback=callback
        with teacher():reference=call(net,a,x,t,kind=args.model)
        for m in modules:m.alignment_callback=None
        with execution_mode(net,teacher=False,inference=True):value=call(net,a,x,t,kind=args.model)
        diff=(value-reference).float()
        low=F.avg_pool2d(diff,64)
        hh,ww=diff.shape[-2:];sy,sx=(hh+1)//2,(ww+1)//2
        quadrants=[diff[:,:,ys,xs].mean((0,2,3)).tolist() for ys,xs in (
            (slice(0,sy),slice(0,sx)),(slice(0,sy),slice(sx,ww)),
            (slice(sy,hh),slice(0,sx)),(slice(sy,hh),slice(sx,ww)))]
        pixel_stats=dict(rgb_mean_error=diff.mean((0,2,3)).tolist(),quadrant_rgb_mean_error=quadrants,
                         low_frequency_mse=low.square().mean().item())
        np.savez_compressed(root/f'target_{target}_maps.npz',**maps)
        write_json(dest,dict(target=target,actual_t=float(t[0]),layers=rows,
                             prediction=error_metrics(value,reference),pixel_prediction=pixel_stats))
        print(f'{args.model} {args.label} {args.case} boundary target={target} saved',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['mask','boundary','control'],required=True)
    p.add_argument('--model',choices=['pid','pixeldit'],required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--text-cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--checkpoint',default='');p.add_argument('--label',default='')
    p.add_argument('--input-protocol',choices=['trajectory','noised-reference'],default='trajectory')
    p.add_argument('--reference-image',type=Path)
    p.add_argument('--times',type=int,nargs='+',default=[900,500,200])
    p.add_argument('--generate',type=int,nargs='*',default=[4,14]);p.add_argument('--threads',type=int,default=1)
    args=p.parse_args();torch.set_num_threads(args.threads);torch.manual_seed(42)
    root=args.output/args.mode/args.model
    if args.mode=='boundary':root=root/args.label
    root=root/args.case;root.mkdir(parents=True,exist_ok=True)
    task_lock=(root/'.lock').open('a')
    try:fcntl.flock(task_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print(f'Another worker owns {root}',flush=True);return
    request={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()
             if k not in ('output','threads')}
    for key in ('checkpoint','reference_image'):
        if request.get(key):
            source=Path(request[key]);source=source/'state.pt' if source.is_dir() else source
            stat=source.stat();request[key+'_stat']=[stat.st_size,stat.st_mtime_ns]
    request_path=root/'request.json'
    if request_path.exists() and json.loads(request_path.read_text())!=request:
        raise ValueError(f'Diagnostic request changed; use a new output/label: {root}')
    write_json(request_path,request)
    if (root/'complete.json').exists():print(f'Already complete: {root}');return
    start=time.time()
    try:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            if args.mode in ('mask','control'):net=load_base(args.model,args.weights);metadata={}
            else:net,metadata=load_student(args.model,args.weights,args.checkpoint)
            a=load_asset(args)
            write_json(root/'config.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                caption=a['caption'],seed=a['seed'],height=a['height'],width=a['width'],metadata=metadata,
                torch_version=torch.__version__,gpu=torch.cuda.get_device_name(),
                inference_only=True,trajectory=args.input_protocol,text_queries_global=True))
            if args.mode=='mask':run_mask(net,a,args,root)
            elif args.mode=='control':run_control(net,a,args,root)
            else:run_boundary(net,a,args,root)
            torch.cuda.synchronize()
            write_json(root/'complete.json',dict(seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                                                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30))
            if (root/'failure.json').exists():os.replace(root/'failure.json',root/'failure.recovered.json')
    except Exception as error:
        write_json(root/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            free_total_bytes=torch.cuda.mem_get_info(),oom=isinstance(error,torch.OutOfMemoryError)))
        raise


if __name__=='__main__':main()
