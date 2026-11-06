"""Resumable multi-GPU FLUX.2-native text/VAE encoding, without CPU offload."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from .data import initialize_index,cache_path,valid,atomic_tensor,prepare_image,image_ids,prompt_ids,cache_identity
from .checkpoint import atomic_json
from .prompts import THEMES


def worker(args):
    from diffusers import Flux2KleinPipeline,AutoencoderKLFlux2
    from transformers import Qwen3ForCausalLM,Qwen2TokenizerFast
    torch.set_num_threads(args.threads);torch.cuda.set_device(0)
    start=time.monotonic();stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    interrupted=lambda:stop[0] or time.monotonic()-start>=args.max_seconds or (args.stop_file and Path(args.stop_file).exists())
    manifest=json.loads((Path(args.cache_root)/'manifest.json').read_text());identity=manifest['fingerprint'];text_identity=cache_identity(manifest,'text')
    rows=manifest['rows'];texts=list(dict.fromkeys(['']+[r['caption'] for r in rows]+[prompt for _,prompt in THEMES]))
    texts=texts[args.worker_index::args.worker_count];assigned=rows[args.worker_index::args.worker_count]
    pending=[s for s in texts if not valid(cache_path(args.cache_root,'text',s),text_identity)]
    if pending and not interrupted():
        encoder=Qwen3ForCausalLM.from_pretrained(args.weights_root,subfolder='text_encoder',torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
        tokenizer=Qwen2TokenizerFast.from_pretrained(args.weights_root,subfolder='tokenizer',local_files_only=True)
        with tqdm(total=len(texts),initial=len(texts)-len(pending),desc=f'text GPU {args.worker_index}') as bar:
            for offset in range(0,len(pending),args.text_batch_size):
                if interrupted():break
                batch=pending[offset:offset+args.text_batch_size]
                with torch.inference_mode():
                    embeds=Flux2KleinPipeline._get_qwen3_prompt_embeds(encoder,tokenizer,batch,device='cuda',dtype=torch.bfloat16)
                for caption,feature in zip(batch,embeds):
                    atomic_tensor(cache_path(args.cache_root,'text',caption),dict(prompt_tokens=feature,prompt_ids=prompt_ids(len(feature))),text_identity)
                bar.update(len(batch))
        del encoder,tokenizer;gc.collect();torch.cuda.empty_cache()
    pending=[r for r in assigned if not valid(cache_path(args.cache_root,'image',r['id']),identity)]
    if pending and not interrupted():
        vae=AutoencoderKLFlux2.from_pretrained(args.weights_root,subfolder='vae',torch_dtype=torch.bfloat16,local_files_only=True).to('cuda')
        vae.eval();vae.requires_grad_(False)
        with ThreadPoolExecutor(args.workers) as pool,tqdm(total=len(assigned),initial=len(assigned)-len(pending),desc=f'latent GPU {args.worker_index}') as bar:
            # Bounded prefetch keeps source images out of long-lived worker memory.
            for start_at in range(0,len(pending),args.workers):
                if interrupted():break
                batch=pending[start_at:start_at+args.workers]
                images=list(pool.map(lambda r:prepare_image(r['image_path'],tuple(r['size'])),batch))
                for row,image in zip(batch,images):
                    if interrupted():break
                    pixels=torch.from_numpy(np.array(image,copy=True)).permute(2,0,1)[None].float().div(127.5).sub(1).to(device='cuda',dtype=torch.bfloat16)
                    with torch.inference_mode():
                        z=vae.encode(pixels).latent_dist.mode()
                        z=Flux2KleinPipeline._patchify_latents(z)
                        mean=vae.bn.running_mean.view(1,-1,1,1).to(z)
                        std=torch.sqrt(vae.bn.running_var.view(1,-1,1,1)+vae.config.batch_norm_eps)
                        z=((z-mean)/std).to(torch.bfloat16)
                    h,w=z.shape[-2:]
                    atomic_tensor(cache_path(args.cache_root,'image',row['id']),dict(
                        image_latents=z.flatten(2).transpose(1,2)[0],image_ids=image_ids(h,w),grid=(h,w)),identity)
                    bar.update(1)
        del vae;gc.collect();torch.cuda.empty_cache()
    return 75 if interrupted() else 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights-root',required=True);p.add_argument('--dataset-root',required=True)
    p.add_argument('--resolution',type=int,choices=(1024,2048),default=1024)
    p.add_argument('--reuse-text-cache',default=os.getenv('FLUX2_TEXT_CACHE_SOURCE'))
    p.add_argument('--stop-file',default='')
    p.add_argument('--cache-root',required=True);p.add_argument('--gpu-ids',default=os.getenv('GPU_IDS','0'))
    p.add_argument('--workers',type=int,default=4);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--text-batch-size',type=int,default=4);p.add_argument('--max-seconds',type=float,default=1200)
    p.add_argument('--worker-index',type=int);p.add_argument('--worker-count',type=int)
    args=p.parse_args(argv)
    if args.worker_index is not None:return worker(args)
    manifest=initialize_index(args.dataset_root,args.cache_root,args.weights_root,args.resolution,args.reuse_text_cache)
    print(f'FLUX.2 dataset: {len(manifest["rows"])} completed image/caption pairs; native {args.resolution}px cache',flush=True)
    children=[];root=Path(args.cache_root);(root/'logs').mkdir(exist_ok=True)
    ids=args.gpu_ids.split(',')
    try:
        for i,gpu in enumerate(ids):
            command=[sys.executable,'-m','flux2_qdm.prepare','--weights-root',args.weights_root,
                '--dataset-root',args.dataset_root,'--cache-root',args.cache_root,'--workers',str(args.workers),
                '--threads',str(args.threads),'--text-batch-size',str(args.text_batch_size),
                '--worker-index',str(i),'--worker-count',str(len(ids)),'--max-seconds',str(args.max_seconds),
                '--stop-file',args.stop_file]
            with (root/'logs'/f'worker_{i}.log').open('a') as log:
                children.append(subprocess.Popen(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu},stdout=log,stderr=subprocess.STDOUT))
        codes=[child.wait() for child in children]
        if any(code not in (0,75) for code in codes):raise RuntimeError(f'Cache preparation failed: {codes}; see {root}/logs')
        complete=all(valid(cache_path(root,'text',caption),cache_identity(manifest,'text')) for caption in ['']+[r['caption'] for r in manifest['rows']]+[prompt for _,prompt in THEMES]) and all(valid(cache_path(root,'image',r['id']),manifest['fingerprint']) for r in manifest['rows'])
        if complete:atomic_json(root/'complete.json',dict(fingerprint=manifest['fingerprint'],samples=len(manifest['rows'])))
        return 0 if complete else 75
    finally:
        for child in children:
            if child.poll() is None:child.terminate()


if __name__=='__main__':raise SystemExit(main())
