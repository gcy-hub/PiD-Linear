"""Independent student-only FLUX3-QDM inference with separate GPU asset phases."""
from __future__ import annotations
import argparse
import gc
import hashlib
import os
from pathlib import Path
import time
from .compute import DIT_COMPUTE, dit_autocast

from .evaluate import atomic_json, checkpoint_config, file_identity, fingerprint, read_json


def cfg_combine(positive,negative,scale):
    return positive if scale == 1 else negative + scale*(positive-negative)


def initial_noise(height,width,seed):
    import torch
    return torch.randn((height//16*width//16,128),
        generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.bfloat16)


def needed_phases(conditions,latent_complete,image):
    return ([] if conditions else ['conditions'])+([] if latent_complete else ['denoise'])+([] if image else ['decode'])


def input_identity(args):
    return dict(**{key:getattr(args,key) for key in ('prompt','negative_prompt','seed','height','width','steps','cfg','alpha','vae_tiling')},
                dit_compute=DIT_COMPUTE)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--weights-root')
    p.add_argument('--prompt',required=True)
    p.add_argument('--negative-prompt',default='')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--height',type=int,default=2048)
    p.add_argument('--width',type=int,default=2048)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--cfg',type=float,default=3.5)
    p.add_argument('--alpha',type=float,default=6.93)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--gpu-id',type=int,default=0)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--max-seconds',type=float,default=1200)
    p.add_argument('--vae-tiling',choices=('on','off'),default='off')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    if args.height<=0 or args.width<=0 or args.height%16 or args.width%16:
        p.error('height and width must be positive multiples of 16')
    if args.steps<1 or args.threads<1 or args.gpu_id<0 or args.max_seconds<=0:
        p.error('steps, threads and max-seconds must be positive; GPU ID must be nonnegative')
    args.stage=None
    args.checkpoint=args.checkpoint.resolve();args.output_root=args.output_root.resolve()
    return args


def generate(args):
    # Set visibility before importing torch; this process owns one selected GPU.
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_id)
    os.environ['OMP_NUM_THREADS']=str(args.threads)
    os.environ['MKL_NUM_THREADS']=str(args.threads)
    import torch
    torch.set_num_threads(args.threads)
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.flux3_image import Flux3ImagePipeline,Flux3Unit_PromptEmbedder,Flux3Unit_PackedBuilder,Flux3Unit_ImageIDs
    from .model import build_model,load_delta
    from .resources import resource_paths,resource_manifest
    config=checkpoint_config(args)
    resources=resource_paths(args.weights_root)
    resource_info=resource_manifest(args.weights_root)
    from .inference_identity import validate_checkpoint_identity
    checkpoint_identity=validate_checkpoint_identity(args.checkpoint,resources['dit'],config)
    identity=dict(parameters=input_identity(args),checkpoint_identity=checkpoint_identity,checkpoint=file_identity(args.checkpoint/'delta.pt'),
        base=file_identity(resources['dit']),resources=resource_info['fingerprints'],
        dtype='bfloat16',torch_version=str(torch.__version__),stage=args.stage,
        architecture={k:config.get(k,default) for k,default in [('attention_mode','hybrid'),('halo',0),('region_size',8),('lora_rank',64),('lora_alpha',64),('qdm_layers',None)]})
    if config.get('memory_text_write',False):identity['architecture']['memory_text_write']=True
    manifest=args.output_root/'run.json'
    old=read_json(manifest)
    if old and old!=identity:
        raise RuntimeError('Sampling inputs/checkpoint changed; select a new output root')
    if old and not args.resume:
        raise RuntimeError('Output exists; use --resume or select a new output root')
    atomic_json(manifest,identity)
    condition_path=args.output_root/'conditions.pt'
    state_path=args.output_root/'latent.pt'
    image_path=args.output_root/'sample.png'
    started=time.monotonic()
    def expired():
        return time.monotonic()-started>=args.max_seconds
    def save(path,value):
        temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
        torch.save(value,temp);os.replace(temp,path)
    def load(path):
        return torch.load(path,map_location='cpu',weights_only=False)
    def sync():
        torch.cuda.synchronize()
    def phase_pipe(kind):
        path=resources[kind]
        pipe=Flux3ImagePipeline.from_pretrained(device='cuda',torch_dtype=torch.bfloat16,
            model_configs=[ModelConfig(path=path,computation_device='cuda',preparing_device='cuda',
                computation_dtype=torch.bfloat16,preparing_dtype=torch.bfloat16)],
            processor_config=ModelConfig(path=resources['processor']) if kind=='text_encoder' else None)
        model=getattr(pipe,kind)
        if model is None or pipe.dit is not None or pipe.vram_management_enabled:
            raise RuntimeError(f'Expected only resident {kind}, with no CPU offload')
        model.eval().requires_grad_(False)
        return pipe
    state=load(state_path) if state_path.exists() else {}
    phases=needed_phases(condition_path.exists(),state.get('complete',False),image_path.exists() and (args.output_root/'sample.json').exists())
    with torch.inference_mode():
        if 'conditions' in phases:
            pipe=phase_pipe('text_encoder')
            embeddings={}
            for label,prompt in [('positive',args.prompt),('negative',args.negative_prompt)]:
                embeddings[label]={k:v.detach().cpu() for k,v in Flux3Unit_PromptEmbedder().process(pipe,prompt).items()}
            save(condition_path,embeddings)
            del pipe,embeddings
            gc.collect();torch.cuda.empty_cache()
            atomic_json(args.output_root/'progress.json',dict(phase='conditions',complete=True))
        if expired():
            return 75
        if 'denoise' in phases:
            pipe=Flux3ImagePipeline(device='cuda',torch_dtype=torch.bfloat16)
            pipe.dit=build_model(resources['dit'],args.stage,'cuda',**identity['architecture'])
            load_delta(pipe.dit,load(args.checkpoint/'delta.pt'),strict=True)
            # Fresh QDM models default to student mode. No teacher forward is invoked.
            pipe.dit.eval().requires_grad_(False)
            pipe.scheduler.set_timesteps(args.steps,denoising_strength=1.,alpha=args.alpha)
            noise=initial_noise(args.height,args.width,args.seed)
            noise_sha=hashlib.sha256(noise.view(torch.uint8).numpy().tobytes()).hexdigest()
            latents=state.get('latents',noise).to('cuda')
            conditions=load(condition_path)
            image_ids=Flux3Unit_ImageIDs.latent_ids(args.height//16,args.width//16,'cuda')
            packed=[]
            for label in ('positive','negative'):
                values={k:v.to('cuda') for k,v in conditions[label].items()}
                pack=Flux3Unit_PackedBuilder().process(pipe,**values,reference_tokens=None,
                    reference_ids=None,latents=latents,image_ids=image_ids)['packed']
                pack.image_grids=((args.height//16,args.width//16),)
                packed.append(pack)
            seconds=state.get('model_seconds',0.)
            for step in range(state.get('next_step',0),args.steps):
                if expired():
                    return 75
                sync();tick=time.monotonic()
                timestep=pipe.scheduler.timesteps[step]
                predictions=[]
                for pack in packed[:1 if args.cfg==1 else 2]:
                    pack.data['image']=latents
                    pack.timesteps['image'].fill_((timestep/1000).item())
                    with dit_autocast():
                        predictions.append(pipe.dit(pack,use_gradient_checkpointing=False))
                pred=cfg_combine(predictions[0],predictions[1] if len(predictions)>1 else None,args.cfg)
                latents=pipe.scheduler.step(pred,timestep,latents)
                if not bool(torch.isfinite(latents).all()):
                    raise FloatingPointError(f'nonfinite student inference latents at step {step}')
                sync();seconds+=time.monotonic()-tick
                state=dict(latents=latents.cpu(),next_step=step+1,complete=step+1==args.steps,
                    model_seconds=seconds,noise_sha256=noise_sha,input_fingerprint=fingerprint(identity),
                    cfg_schedule=[args.cfg]*args.steps)
                save(state_path,state)
                atomic_json(args.output_root/'progress.json',dict(phase='denoise',next_step=step+1,steps=args.steps))
            del pipe,packed,pack,latents,predictions,pred,values,image_ids
            gc.collect();torch.cuda.empty_cache()
        if expired():
            return 75
        if 'decode' in phases:
            state=load(state_path)
            pipe=phase_pipe('vae')
            pipe.vae.use_tiling=args.vae_tiling=='on'
            decode={name:getattr(pipe.vae,name,None) for name in ('use_tiling','tile_sample_min_size','tile_latent_min_size','tile_overlap_factor')}
            latent=state['latents'].reshape(args.height//16,args.width//16,-1).permute(2,0,1)[None].to('cuda')
            sync();tick=time.monotonic()
            decoded=pipe.vae.decode(latent)
            if not bool(torch.isfinite(decoded).all()):
                raise FloatingPointError('nonfinite native VAE output')
            sync();seconds=time.monotonic()-tick
            image=pipe.vae_output_to_image(decoded)
            temp=image_path.with_name(f'sample.{os.getpid()}.tmp.png')
            image.save(temp);os.replace(temp,image_path)
            atomic_json(args.output_root/'sample.json',dict(**identity,
                model_seconds=state['model_seconds'],vae_seconds=seconds,decode=decode,
                noise_sha256=state['noise_sha256'],input_fingerprint=state['input_fingerprint'],
                cfg_schedule=state['cfg_schedule'],image=str(image_path),native_size=[args.width,args.height]))
            del pipe,latent,decoded
            gc.collect();torch.cuda.empty_cache()
    atomic_json(args.output_root/'complete.json',dict(image=str(image_path),input_fingerprint=fingerprint(identity)))
    print(str(image_path),flush=True)
    return 0


def main(argv=None):
    args=parse_args(argv)
    args.output_root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (args.output_root/'sample.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return generate(args)


if __name__=='__main__':
    raise SystemExit(main())
