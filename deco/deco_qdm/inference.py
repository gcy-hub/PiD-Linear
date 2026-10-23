"""Native DeCo AdamLM sampling and resumable teacher/student comparisons."""
import argparse
import contextlib
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from .prompts import PROMPTS


def schedule(steps=25, shift=3., order=2):
    if steps<1 or shift<=0 or order not in (1,2):
        raise ValueError('steps>0, shift>0 and AdamLM order 1 or 2 are required')
    # Matches official AdamLMSampler: last_step=1/num_steps, FP32 coefficients.
    ts=torch.cat((torch.linspace(0.,1.-1./steps,steps),torch.tensor([1.])))
    ts=ts/(ts+(1-ts)*shift)
    coeffs=[]
    for i in range(steps):
        if order==1 or i==0:
            coeffs.append([torch.tensor(1.)])
        else:
            t1,t2,a,b=ts[i-1],ts[i],ts[i],ts[i+1]
            i1=.5/(t1-t2)*((b-t2)**2-(a-t2)**2)
            i2=.5/(t2-t1)*((b-t1)**2-(a-t1)**2)
            coeffs.append([i1/(i1+i2),i2/(i1+i2)])
    return ts,coeffs


@torch.no_grad()
def sample(model, noise, condition, uncondition, steps=25,cfg=4.,shift=3.,order=2):
    ts,coeffs=schedule(steps,shift,order)
    x=noise.clone()
    history=[]
    y=torch.cat((uncondition,condition),0)
    # Keep the official accumulated FP32 time and CFG-off boundary at exactly t=0.
    t=torch.zeros(noise.shape[0],device=x.device,dtype=x.dtype)
    with torch.autocast(device_type=x.device.type,dtype=torch.bfloat16,enabled=x.is_cuda):
        for i,weights in enumerate(coeffs):
            pred=model(torch.cat((x,x),0),t.repeat(2),y)
            u,c=pred.float().chunk(2)
            guidance=cfg if 0<float(t[0])<1 else 1.
            history.append(u+guidance*(c-u))
            history=history[-order:]
            v=sum(w.to(x.device)*p for w,p in zip(weights,history[-len(weights):]))
            dt=(ts[i+1]-ts[i]).to(x.device)
            x=x+v*dt
            t=t+dt
    return x


def _atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def _save_image(x,path):
    pixels=((x.detach().float().clamp(-1,1)+1)*127.5).to(torch.uint8)
    image=Image.fromarray(pixels.permute(1,2,0).cpu().numpy())
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.stem+f'.{os.getpid()}.tmp.png')
    image.save(tmp);tmp.replace(path)


def _pair(left,right,destination,crop=False):
    with Image.open(left) as a,Image.open(right) as b:
        a,b=a.convert('RGB'),b.convert('RGB')
        if crop:
            w,h=a.size;roi=(w//4,h//4,3*w//4,3*h//4);a,b=a.crop(roi),b.crop(roi)
        assert a.size==b.size
        w,h=a.size;gap=16
        out=Image.new('RGB',(w*2+gap,h+32),'white');draw=ImageDraw.Draw(out)
        draw.text((8,8),'Original DeCo',fill='black');draw.text((w+gap+8,8),'DeCo-QDM',fill='black')
        out.paste(a,(0,32));out.paste(b,(w+gap,32));out.save(destination)


def _model_settings(model):
    modules=list(model.hybrid_modules())
    return dict(region_size=modules[0][1].region_size,halo=modules[0][1].halo)


def _image_run(model,noise,condition,empty,destination,settings,teacher=False):
    marker=Path(destination).with_suffix('.done.json')
    if marker.exists() and Path(destination).exists():
        old=json.loads(marker.read_text())
        if old.get('settings')==settings:return old
        raise ValueError(f'Existing image has different settings: {destination}')
    with model.execution(teacher=teacher):
        if noise.is_cuda:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        output=sample(model,noise,condition,empty,steps=settings['steps'],cfg=settings['cfg'],shift=settings['shift'],order=settings['order'])
        if noise.is_cuda:torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        peak=torch.cuda.max_memory_allocated()/2**30 if noise.is_cuda else 0.
    _save_image(output[0],destination)
    record=dict(settings=settings,seconds=elapsed,peak_memory_gib=peak)
    _atomic_json(marker,record);return record


def validate_checkpoint(*,model,stage,step,config,output_dir):
    """All ranks call; each generates a disjoint subset without loading another model."""
    from torch import distributed as dist
    rank=dist.get_rank() if dist.is_initialized() else 0
    world=dist.get_world_size() if dist.is_initialized() else 1
    device=next(model.parameters()).device
    cache=Path(config['text_cache_root'])
    from .data import load_text, canonical_digest
    manifest=json.loads((cache/'manifest.json').read_text())
    if manifest.get('status')!='ready' or manifest.get('validation_prompts')!=PROMPTS:
        raise ValueError('Validation text cache is incomplete or uses different fixed prompts')
    text_fingerprint=canonical_digest(manifest)
    empty=load_text(cache/'empty.pt')
    if isinstance(empty,dict):empty=empty.get('embedding',empty.get('text'))
    empty=empty.to(device)[None] if empty.ndim==2 else empty.to(device)
    empty=empty[:, :model.txt_max_length, :model.txt_embed_dim]
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    settings=dict(steps=25,cfg=4.,shift=3.,order=2,height=config.get('resolution',512),width=config.get('resolution',512),**_model_settings(model))
    reference=Path(config['output_root'])/'reference'/f"{settings['height']}px"
    reference.mkdir(parents=True,exist_ok=True)
    records=[]
    complete=True
    was_training=model.training;model.eval()
    try:
        for prompt in PROMPTS[rank::world]:
            deadline=config.get('max_seconds',0)
            started=config.get('started_at',time.monotonic())
            if config.get('_stop_requested',lambda:False)() or (Path(config['output_root'])/stage/'STOP').exists() or (deadline and time.monotonic()-started>=deadline):
                complete=False
                break
            text=load_text(cache/'validation'/(prompt['id']+'.pt'))
            if isinstance(text,dict):text=text.get('embedding',text.get('text'))
            if text.ndim==2:text=text[None]
            text=text[:, :model.txt_max_length, :model.txt_embed_dim].to(device)
            noise=torch.randn((1,3,settings['height'],settings['width']),generator=torch.Generator().manual_seed(prompt['seed'])).to(device)
            common=dict(settings,prompt=prompt['prompt'],seed=prompt['seed'],weights_root=str(config['weights_root']),base_fingerprint=getattr(model,'base_fingerprint',config.get('base_fingerprint')),text_fingerprint=text_fingerprint)
            original=reference/(prompt['id']+'.png');student=output_dir/(prompt['id']+'.png')
            teacher_settings={k:v for k,v in common.items() if k not in ('region_size','halo')}
            a=_image_run(model,noise,text,empty,original,teacher_settings,teacher=True)
            b=_image_run(model,noise,text,empty,student,dict(common,stage=stage,step=step))
            _pair(original,student,output_dir/(prompt['id']+'_comparison.png'))
            _pair(original,student,output_dir/(prompt['id']+'_crop.png'),crop=True)
            records.append(dict(id=prompt['id'],teacher=a,student=b))
        _atomic_json(output_dir/f'rank_{rank:03d}.json',dict(stage=stage,step=step,images=records))
    finally:model.train(was_training)
    return {'validation/complete':complete,'validation/images':len(records),'records':records}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-root',default=os.path.expanduser('~/huggingface_ckpts/DeCo'))
    parser.add_argument('--checkpoint',help='Complete student checkpoint directory; omit for original model')
    parser.add_argument('--prompt',required=True);parser.add_argument('--negative-prompt',default='')
    parser.add_argument('--height',type=int,default=512);parser.add_argument('--width',type=int,default=512)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--steps',type=int,default=25)
    parser.add_argument('--cfg',type=float,default=4.);parser.add_argument('--shift',type=float,default=3.)
    parser.add_argument('--order',type=int,default=2);parser.add_argument('--region-size',type=int,default=None)
    parser.add_argument('--halo',type=int,default=None);parser.add_argument('--output',default='outputs/deco-qdm/inference/image.png')
    parser.add_argument('--compare',action='store_true');parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args();torch.set_num_threads(args.threads)
    if min(args.height,args.width)<16 or args.height%16 or args.width%16:parser.error('height and width must be positive multiples of 16')
    from .model import build_model,load_delta_state_dict
    from .cache_text import TextEncoder
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Encode first in a separate model lifetime; no residency/offload during DiT sampling.
    encoder=TextEncoder(Path(args.weights_root)/'Qwen3-1.7B',device=device)
    texts=encoder.encode([args.prompt,args.negative_prompt]);del encoder
    if device.type=='cuda':torch.cuda.empty_cache()
    state=None;stage='a1'
    if args.checkpoint:
        checkpoint=Path(args.checkpoint)
        from .checkpoint import load_state
        state=load_state(checkpoint)
        stage=state['stage']
        architecture=state['metadata']['architecture']
        if architecture.get('tiny'):raise ValueError('Tiny test checkpoints are not native DeCo inference weights')
        for key in ['region_size','halo']:
            explicit=getattr(args,key)
            if explicit is not None and explicit!=architecture[key]:
                raise ValueError(f'Checkpoint {key}={architecture[key]} differs from the requested {explicit}')
            setattr(args,key,architecture[key])
    args.region_size=8 if args.region_size is None else args.region_size
    args.halo=2 if args.halo is None else args.halo
    from .checkpoint import fingerprint_file
    base_fingerprint=fingerprint_file(Path(args.weights_root)/'t2i_DeCo.ckpt')
    if state and state['metadata']['base_fingerprint']!=base_fingerprint:
        raise ValueError('Inference base weights differ from the checkpoint teacher')
    model=build_model(args.weights_root,stage=stage,region_size=args.region_size,halo=args.halo,device=device)
    model.base_fingerprint=base_fingerprint
    if state:load_delta_state_dict(model,state['delta'])
    model.eval();condition,empty=texts[0:1].to(device),texts[1:2].to(device)
    noise=torch.randn((1,3,args.height,args.width),generator=torch.Generator().manual_seed(args.seed)).to(device)
    settings={k:getattr(args,k) for k in ['prompt','negative_prompt','height','width','seed','steps','cfg','shift','order','region_size','halo']}
    settings['checkpoint']=str(Path(args.checkpoint).resolve()) if args.checkpoint else None
    settings['base_fingerprint']=base_fingerprint
    settings['text_fingerprint']=hashlib.sha256(texts.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    if args.checkpoint:
        settings['checkpoint_fingerprint']=fingerprint_file(Path(args.checkpoint)/'state.pt')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    result=_image_run(model,noise,condition,empty,out,settings,teacher=not bool(args.checkpoint))
    if args.compare:
        original=out.with_name(out.stem+'_original.png')
        _image_run(model,noise,condition,empty,original,dict(settings,checkpoint=None),teacher=True)
        _pair(original,out,out.with_name(out.stem+'_comparison.png'))
        _pair(original,out,out.with_name(out.stem+'_crop.png'),crop=True)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
