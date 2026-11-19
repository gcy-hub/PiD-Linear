"""One allocated GPU per rank. Run each stage in a fresh process."""
import argparse
import gc
import json
import os
from pathlib import Path
import socket
import sys
import time

import torch
from omegaconf import OmegaConf
from PIL import Image

from prompts import HERE, MODELS, manifest, pairs

REPO = HERE.parent
SETTINGS = dict(height=4096, width=4096, steps=25, cfg=5., shift=6., sigma=0.)


def save_pt(obj, path):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


@torch.inference_mode()
def conditions(out, jobs, cfg, device):
    from data.encoders import build_encoder
    from diffusers import ZImagePipeline
    from pid._src.common.text_encoder import NEGATIVE_PROMPT
    from pid._src.inference.pipeline_registry import PIPELINE_REGISTRY, extract_latent

    todo = [(rows,path) for rows,path in jobs if not path.exists()]
    if not todo:
        return
    encoder = build_encoder(cfg, device=device, image=False, text=True)
    neg = {k:v.cpu() for k,v in encoder.encode_text([NEGATIVE_PROMPT]).items()}
    for rows,path in todo:
        pos = encoder.encode_text([r['prompt'] for r in rows])
        save_pt(dict(rows=rows, positive={k:v.cpu() for k,v in pos.items()}, negative=neg), path.with_suffix('.text'))
    del encoder, pos
    gc.collect()
    torch.cuda.empty_cache()
    spec = PIPELINE_REGISTRY['zimage-turbo']
    pipe = ZImagePipeline.from_pretrained(Path(cfg.paths.weights)/'Z-Image-Turbo', torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.enable_model_cpu_offload(gpu_id=device.index)
    for rows,path in todo:
        latents = []
        for r in rows:
            raw = pipe(prompt=r['prompt'], height=1024, width=1024, generator=torch.Generator(device=device).manual_seed(r['seed']),
                       num_inference_steps=spec.default_num_inference_steps, guidance_scale=spec.default_guidance_scale,
                       output_type='latent', **spec.extra_generate_kwargs)
            latents.append(extract_latent(pipe,raw,spec,1024,1024).cpu())
        obj = torch.load(path.with_suffix('.text'), weights_only=True)
        latent = torch.cat(latents)
        assert latent.shape == (len(rows),16,128,128) and latent.isfinite().all()
        save_pt(dict(**obj,latent=latent),path)
        path.with_suffix('.text').unlink()


@torch.inference_mode()
def decode(out, jobs, cfg, device, model):
    from pid_decoder import load_decoder
    from pid._src.common.runtime import sample
    from pid._src.vdn_pid.attention import execution_mode
    from contextlib import nullcontext

    export = Path(cfg.model.resources.export)
    ck = cfg.model.resources.weights if model == 'pid' else export
    cfg.runtime.gradient_checkpointing = False
    net, fused = load_decoder(cfg, device, {'pid':'original','pid-dit':'export'}[model])
    assert fused == {'pid':0, 'pid-dit':56}[model]
    prop = torch.cuda.get_device_properties(device)
    meta = dict(model=model, checkpoint=str(ck), gpu=prop.name, gpu_uuid=str(prop.uuid), host=socket.gethostname(),
                torch=str(torch.__version__), cuda=torch.version.cuda, compile=False, fused=fused, **SETTINGS)
    folder = out/model
    folder.mkdir(exist_ok=True)
    with (nullcontext() if model=='pid' else execution_mode(net,inference=True)):
        for rows,path in jobs:
            meta['batch_size'] = len(rows)
            record = folder/path.with_suffix('.json').name
            pngs = [folder/f'{r["id"]:04}.png' for r in rows]
            if record.exists():
                old = json.loads(record.read_text())
                if old['rows'] != rows or any(old[k]!=meta[k] for k in meta) or not all(p.exists() for p in pngs):
                    raise ValueError(f'Incompatible resume data: {record}')
                continue
            obj = torch.load(path, map_location=device, weights_only=True)
            assert obj['rows'] == rows
            args = [obj[side][key] for side in ('positive','negative') for key in ('embeddings','mask')]
            assert args[1].shape == (len(rows),300) and args[3].shape==(1,300)
            latent = obj['latent'].to(torch.bfloat16)
            assert latent.shape == (len(rows),16,128,128)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            result = sample(net,*args,latent,seed=rows[0]['seed'],**SETTINGS)
            torch.cuda.synchronize(device)
            duration = time.perf_counter()-start
            assert result.shape==(len(rows),3,4096,4096) and result.isfinite().all()
            for x,png in zip(result,pngs):
                rgb = ((x.float().clamp(-1,1)+1)*127.5).round().byte().permute(1,2,0).cpu().numpy()
                Image.fromarray(rgb).save(png,compress_level=1)
            del x, result
            tmp = record.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(**meta,rows=rows,batch_seed=rows[0]['seed'],decode_seconds=duration))+'\n')
            tmp.replace(record)
            print(model,path.stem,duration,flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('conditions',*MODELS))
    p.add_argument('output',type=Path)
    p.add_argument('--single', action='store_true', help='Decode only DPG sample 0 at BS=1')
    args = p.parse_args()
    rank = 0 if args.single else int(os.environ['RANK'])
    if not args.single and int(os.environ['WORLD_SIZE']) != 8:
        raise ValueError('Expected 8 GPU workers')
    device = torch.device('cuda',0 if args.single else int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    sys.path[:0] = [str(REPO),str(REPO/'inference')]
    os.environ['PID_REPO_ROOT'] = str(REPO)
    rows = manifest()
    cache = args.output/'conditions'
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [([rows[i],rows[j]],cache/f'{i:04}.pt') for i,j in pairs(rank)]
    if args.single:
        jobs = [([rows[0]], cache/'single-0000.pt')]
    weights = Path(os.environ['WEIGHTS_ROOT']).expanduser().resolve()
    export = Path(os.environ['PID_EXPORT']).expanduser().resolve()
    cfg = OmegaConf.load(export/'config.yaml')
    cfg.paths.weights = str(weights)
    cfg.model.resources.export = str(export)
    cfg.model.resources.text_encoder = str(weights/'gemma-2-2b-it')
    cfg.model.resources.weights = str(weights/'PiD/checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_flux_undistilled/model_ema_bf16.pth')
    if args.stage=='conditions':
        conditions(args.output,jobs,cfg,device)
    else:
        decode(args.output,jobs,cfg,device,args.stage)


if __name__=='__main__':
    main()
