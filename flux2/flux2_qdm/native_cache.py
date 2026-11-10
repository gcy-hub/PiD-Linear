"""Read CLEAR's immutable DiffSynth caches directly; no image/feature copies."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from .data import atomic_tensor,cache_path,valid,fingerprint,file_identity,image_ids,prompt_ids
from .checkpoint import atomic_json
from .prompts import THEMES


def source_digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def load_sample(manifest,row):
    from safetensors import safe_open
    results={}
    native=manifest['native_cache']
    for kind,suffix,key in [('text','_prompt_embed.safetensors','prompt_embeds'),('latent','_latent_code.safetensors','latents')]:
        path=Path(native['root'])/kind/(row['id']+suffix)
        with safe_open(str(path),framework='pt') as handle:
            meta=handle.metadata() or {}
            source=dict(prompt=row['caption']) if kind=='text' else dict(bytes=row['source']['size'],mtime_ns=row['source']['mtime_ns'])
            if meta.get('fingerprint')!=native[kind+'_fingerprint'] or meta.get('model_type')!='flux2' or meta.get('source')!=source_digest(source):
                raise ValueError(f'Reference native cache identity/source mismatch: {path}')
            value=handle.get_tensor(key)
        if not torch.isfinite(value).all():raise ValueError(f'Nonfinite native cache: {path}')
        results[kind]=value
    h,w=row['size'][1]//16,row['size'][0]//16
    if results['latent'].shape!=(h*w,128):raise ValueError('Native latent grid mismatch')
    return dict(id=row['id'],image_latents=results['latent'],image_ids=image_ids(h,w),grid=(h,w),
                prompt_tokens=results['text'],prompt_ids=prompt_ids(len(results['text'])))


def initialize(reference,output):
    cfg=json.loads((reference/'run.json').read_text())['config'];root=Path(cfg['cache']['directory'])
    manifests={kind:json.loads((root/(kind+'-manifest.json')).read_text()) for kind in ('text','latent')}
    rows=[]
    with Path(cfg['data']['metadata']).open() as f:
        for record in csv.DictReader(f):
            image=Path(record['image']).resolve();n=cfg['data']['resolution']
            rows.append(dict(id=image.stem,caption=record['prompt'],image_path=str(image),source=file_identity(image),size=[n,n],bucket=f'{n}x{n}'))
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate native sample IDs')
    for kind in manifests:
        done=json.loads((root/(kind+'-COMPLETE.json')).read_text())
        if done['count']!=len(rows) or done['fingerprint']!=manifests[kind]['fingerprint']:raise ValueError('Reference native cache is incomplete')
    manifest=dict(schema=1,format='clear-diffsynth',rows=rows,dataset_root=cfg['data']['root'],
        image=dict(resolution=cfg['data']['resolution']),weights_root=cfg['model']['pretrained_model_name_or_path'],
        diffsynth_root=cfg['model']['diffsynth_path'],native_cache=dict(root=str(root),
        text_fingerprint=manifests['text']['fingerprint'],latent_fingerprint=manifests['latent']['fingerprint']))
    manifest['fingerprint']=fingerprint(manifest)
    output.mkdir(parents=True,exist_ok=True);target=output/'manifest.json'
    if target.exists() and json.loads(target.read_text())!=manifest:raise ValueError('Native reference cache changed; use another output')
    atomic_json(target,manifest);return manifest


def encode_aux(root,rank,world,threads,suite='default'):
    from tqdm import tqdm
    from .diffsynth_model import load_pipeline
    torch.set_num_threads(threads);torch.cuda.set_device(0)
    manifest=json.loads((root/'manifest.json').read_text());identity=manifest['fingerprint']
    from .prompts import CLEAR_THEMES
    prompts=['']+[p for _,p in (CLEAR_THEMES if suite=='clear' else THEMES)];todo=[p for i,p in enumerate(prompts) if i%world==rank and not valid(cache_path(root,'text',p),identity)]
    if not todo:return
    pipe=load_pipeline(manifest['weights_root'],'cuda',('text_encoder',),manifest['diffsynth_root'])
    from diffsynth.pipelines.flux2_image import Flux2Unit_Qwen3PromptEmbedder
    with torch.inference_mode():
        for prompt in tqdm(todo,desc=f'DiffSynth auxiliary text GPU{rank}'):
            tensor=Flux2Unit_Qwen3PromptEmbedder().process(pipe,[prompt])['prompt_embeds'][0]
            atomic_tensor(cache_path(root,'text',prompt),dict(prompt_tokens=tensor,prompt_ids=prompt_ids(len(tensor))),identity)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-run',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--gpu-ids',default='0');p.add_argument('--threads',type=int,default=1)
    p.add_argument('--validation-suite',choices=('default','clear'),default='default')
    p.add_argument('--worker-index',type=int);p.add_argument('--worker-count',type=int,default=1)
    args=p.parse_args();root=args.output_root.resolve()
    if args.worker_index is not None:
        encode_aux(root,args.worker_index,args.worker_count,args.threads,args.validation_suite);return
    manifest=initialize(args.reference_run.resolve(),root)
    print(f"Reference cache: {len(manifest['rows'])} native image/text pairs; no features copied",flush=True)
    ids=args.gpu_ids.split(',');children=[]
    for i,gpu in enumerate(ids):
        children.append(subprocess.Popen([sys.executable,'-m','flux2_qdm.native_cache','--reference-run',str(args.reference_run),
            '--output-root',str(root),'--worker-index',str(i),'--worker-count',str(len(ids)),'--threads',str(args.threads),'--validation-suite',args.validation_suite],
            # All workers read the same immutable training feature references.
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}))
    if any(c.wait()!=0 for c in children):raise RuntimeError('Native auxiliary text preparation failed')
    from .prompts import CLEAR_THEMES
    for prompt in ['']+[v for _,v in (CLEAR_THEMES if args.validation_suite=='clear' else THEMES)]:
        if not valid(cache_path(root,'text',prompt),manifest['fingerprint']):raise ValueError('Auxiliary text missing')
    atomic_json(root/'complete.json',dict(fingerprint=manifest['fingerprint'],samples=len(manifest['rows'])))
    print('Native cache ready',flush=True)


if __name__=='__main__':main()
