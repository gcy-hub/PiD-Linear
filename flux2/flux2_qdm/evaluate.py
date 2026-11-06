"""Ten student-only native FLUX.2 2K samples, resuming denoising and image tasks."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from .checkpoint import atomic_json,read_checkpoint
from .data import cache_path,atomic_tensor,fingerprint
from .prompts import THEMES,PROMPT_SEED


def worker(args):
    from diffusers import AutoencoderKLFlux2,Flux2KleinPipeline,FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
    from .model import build_model,load_delta
    from .compute import dit_autocast
    start=time.monotonic();stopping=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopping.__setitem__(0,True))
    torch.set_num_threads(args.threads);torch.cuda.set_device(0)
    state=read_checkpoint(args.checkpoint);cfg=state['config']
    root=Path(args.output_root);folder=root/'student'/'2048';folder.mkdir(parents=True,exist_ok=True)
    tasks=[(i,name,prompt) for i,(name,prompt) in enumerate(THEMES) if i%args.worker_count==args.worker_index and not (folder/f'{name}.done.json').exists()]
    if not tasks:return 0
    model=build_model(args.weights_root,args.stage,'cuda',sliding_kernel_size=cfg['sliding_kernel_size'],
                      lora_rank=cfg['lora_rank'],lora_alpha=cfg['lora_alpha'])
    load_delta(model,torch.load(Path(args.checkpoint)/'delta.pt',weights_only=True));model.eval();model.requires_grad_(False)
    vae=AutoencoderKLFlux2.from_pretrained(args.weights_root,subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
    vae.eval();vae.requires_grad_(False)
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(args.weights_root,subfolder='scheduler',local_files_only=True)
    pipe=Flux2KleinPipeline(vae=vae,transformer=model.base,scheduler=scheduler,tokenizer=None,text_encoder=None,is_distilled=False)
    null=torch.load(cache_path(cfg['cache_root'],'text',''),weights_only=True)['prompt_tokens'][None].to('cuda')
    stopped=lambda:stopping[0] or time.monotonic()-start>=args.max_seconds or (Path(args.stage_root)/'STOP').exists()
    for index,name,prompt in tasks:
        if stopped():return 75
        identity=fingerprint(dict(checkpoint=str(Path(args.checkpoint).resolve()),prompt=prompt,seed=PROMPT_SEED+index,
                                  steps=args.steps,cfg=args.cfg,resolution=2048,kernel=cfg['sliding_kernel_size']))
        condition=torch.load(cache_path(cfg['cache_root'],'text',prompt),weights_only=True)
        text=condition['prompt_tokens'][None].to('cuda');txt=condition['prompt_ids'].to('cuda')
        saved_path=folder/f'{name}.pt'
        saved=torch.load(saved_path,weights_only=True) if saved_path.exists() else {}
        if saved and saved['identity']!=identity:raise ValueError('Evaluation settings changed; use a new output directory')
        z,ids=pipe.prepare_latents(1,32,2048,2048,torch.bfloat16,'cuda',torch.Generator('cpu').manual_seed(PROMPT_SEED+index))
        next_step=int(saved.get('next_step',0))
        if saved:z=saved['latents'].to('cuda')
        scheduler.set_timesteps(args.steps,device='cuda',sigmas=np.linspace(1.,1/args.steps,args.steps),
                                 mu=compute_empirical_mu(z.shape[1],args.steps))
        scheduler.set_begin_index(next_step)
        sample_start=time.monotonic()
        with torch.inference_mode():
            for step in range(next_step,args.steps):
                t=scheduler.timesteps[step]
                with dit_autocast():
                    pos=model.native_forward(grid=(128,128),hidden_states=z,encoder_hidden_states=text,
                        timestep=t.reshape(1)/1000,img_ids=ids,txt_ids=txt,return_dict=False)[0]
                    neg=model.native_forward(grid=(128,128),hidden_states=z,encoder_hidden_states=null,
                        timestep=t.reshape(1)/1000,img_ids=ids,txt_ids=txt,return_dict=False)[0]
                pred=neg+args.cfg*(pos-neg)
                z=scheduler.step(pred,t,z,return_dict=False)[0].to(torch.bfloat16)
                if (step+1)%5==0 or stopped() or step+1==args.steps:
                    atomic_tensor(saved_path,dict(identity=identity,latents=z,next_step=step+1),identity)
                atomic_json(root/f'progress_{args.worker_index}.json',dict(phase='denoise',task=name,next_step=step+1))
                if stopped():return 75
            latent=pipe._unpack_latents_with_ids(z,ids)
            mean=vae.bn.running_mean.view(1,-1,1,1).to(latent)
            std=torch.sqrt(vae.bn.running_var.view(1,-1,1,1)+vae.config.batch_norm_eps).to(latent)
            latent=pipe._unpatchify_latents(latent*std+mean)
            image=vae.decode(latent.to(torch.bfloat16),return_dict=False)[0]
            image=pipe.image_processor.postprocess(image,output_type='pil')[0]
            path=folder/f'{name}.png';temporary=path.with_suffix('.partial.png');image.save(temporary);os.replace(temporary,path)
        atomic_json(folder/f'{name}.done.json',dict(identity=identity,seed=PROMPT_SEED+index,prompt=prompt,
            seconds=time.monotonic()-sample_start,checkpoint=str(args.checkpoint),image=str(path),steps=args.steps,cfg=args.cfg))
        print(f'Completed {name}: {path}',flush=True)
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--weights-root',required=True)
    p.add_argument('--output-root',required=True);p.add_argument('--stage-root',required=True)
    p.add_argument('--stage',required=True,choices=('a1','a2','b'));p.add_argument('--gpu-ids',default=os.getenv('GPU_IDS','0'))
    p.add_argument('--workers',type=int,default=4);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--steps',type=int,default=50);p.add_argument('--cfg',type=float,default=4.)
    p.add_argument('--max-seconds',type=float,default=1200);p.add_argument('--resolutions',default='2048')
    p.add_argument('--student-only',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--worker-index',type=int);p.add_argument('--worker-count',type=int)
    args=p.parse_args(argv)
    if args.resolutions!='2048':p.error('This experiment validates 2048x2048')
    if args.worker_index is not None:return worker(args)
    root=Path(args.output_root);root.mkdir(parents=True,exist_ok=True)
    children=[];ids=args.gpu_ids.split(',')
    common=list(sys.argv[1:] if argv is None else argv)
    try:
        for rank,gpu in enumerate(ids):
            with (root/f'worker_{rank}.log').open('a') as log:
                children.append(subprocess.Popen([sys.executable,'-m','flux2_qdm.evaluate',*common,
                    '--worker-index',str(rank),'--worker-count',str(len(ids))],
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu},stdout=log,stderr=subprocess.STDOUT))
        codes=[child.wait() for child in children]
        if any(code not in (0,75) for code in codes):raise RuntimeError(f'Evaluation failed {codes}; see {root}/worker_*.log')
        if not all((root/'student/2048'/f'{name}.done.json').exists() for name,_ in THEMES):return 75
        marker=dict(checkpoint=str(Path(args.checkpoint).resolve()),stage=args.stage,samples=10,evaluation=str(root))
        stage=Path(args.stage_root);atomic_json(root/'complete.json',marker)
        step=int(json.loads((Path(args.checkpoint)/'complete.json').read_text())['step'])
        atomic_json(stage/'evaluations'/f'step_{step:09d}.done.json',marker)
        config=json.loads((stage/'config.json').read_text())
        if step>=config['settings']['steps']:atomic_json(stage/'finished.json',marker)
        (stage/'validation_pending.json').unlink(missing_ok=True)
        if config.get('swanlab_mode')!='disabled':
            import swanlab
            tracking=json.loads((stage/'swanlab.json').read_text())
            run=swanlab.init(project='FLUX2-QDM',name=args.stage,id=tracking['run_id'],resume=True,
                mode='online' if config['swanlab_mode']=='cloud' else config['swanlab_mode'],log_dir=str(stage/'swanlog'))
            run.log({f'validation/{name}':swanlab.Image(str(root/'student/2048'/f'{name}.png')) for name,_ in THEMES},step=read_checkpoint(args.checkpoint)['progress']['total_step'])
            run.finish()
        return 0
    finally:
        for child in children:
            if child.poll() is None:child.terminate()


if __name__=='__main__':raise SystemExit(main())
