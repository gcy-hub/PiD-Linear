"""Deterministic global draws and native FLUX.2 training inputs."""
from __future__ import annotations

import hashlib
import math
import numpy as np
import torch


class GlobalSampler:
    def __init__(self, buckets, batch_size=1, world_size=1, accumulation=1, seed=42):
        self.buckets = {str(k): list(v) for k,v in buckets.items() if len(v)}
        self.batch_size, self.world_size, self.accumulation, self.seed = batch_size, world_size, accumulation, seed
        self.effective = batch_size * world_size * accumulation
        if not self.buckets or min(map(len,self.buckets.values())) < self.effective:
            raise ValueError('Every nonempty resolution bucket must contain at least one global batch')

    def global_batch(self, cursor):
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,int(cursor)]))
        groups=list(self.buckets.values())
        counts=np.array([len(v) for v in groups],dtype=np.float64)
        group=groups[rng.choice(len(groups),p=counts/counts.sum())]
        return rng.choice(group,size=self.effective,replace=False).tolist()

    def rank_batch(self,cursor,rank,micro=0):
        start=(micro*self.world_size+rank)*self.batch_size
        return self.global_batch(cursor)[start:start+self.batch_size]


def lr_factor(step,total,warmup):
    if step < warmup:
        return (step+1)/max(1,warmup)
    progress=min(1.,max(0.,(step-warmup)/max(1,total-warmup)))
    return .05+.95*.5*(1+math.cos(math.pi*progress))


def sample_seed(seed,stage,cursor,sample_id):
    return int.from_bytes(hashlib.sha256(f'{seed}:{stage}:{cursor}:{sample_id}'.encode()).digest()[:8],'little')%(2**63-1)


from functools import lru_cache

@lru_cache(maxsize=16)
def sigma_table(root,length):
    import os
    if os.environ.get('FLUX2_BACKEND')=='diffsynth':
        from .diffsynth_model import import_framework
        import_framework()
        from diffsynth.diffusion import FlowMatchScheduler
        scheduler=FlowMatchScheduler('FLUX.2');scheduler.set_timesteps(1000)
        return scheduler.sigmas
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(root,subfolder='scheduler',local_files_only=True)
    scheduler.set_timesteps(1000,device='cpu',mu=compute_empirical_mu(length,1000))
    return scheduler.sigmas


def training_inputs(samples,seed,stage,cursor,device,caption_dropout=0.,empty_prompt=None):
    from types import SimpleNamespace
    import os
    root=os.environ['FLUX2_WEIGHTS_ROOT']
    items=[];targets=[];mask=[]
    for sample in samples:
        generator=torch.Generator(device=device).manual_seed(sample_seed(seed,stage,cursor,sample['id']))
        index=int(torch.randint(0,1000,(),generator=generator,device=device).item())
        sigma=sigma_table(root,len(sample['image_latents']))[index].to(device)
        x0=sample['image_latents'].to(device=device,dtype=torch.bfloat16)
        noise=torch.randn(x0.shape,generator=generator,device=device,dtype=torch.bfloat16)
        dropped=sample_seed(seed,stage,cursor,'caption:'+sample['id'])/float(2**63-1)<caption_dropout
        text=empty_prompt if dropped else sample
        items.append(dict(hidden_states=((1-sigma)*x0+sigma*noise)[None],
            encoder_hidden_states=text['prompt_tokens'][None].to(device=device,dtype=torch.bfloat16),
            timestep=sigma.reshape(1).to(device),img_ids=sample['image_ids'].to(device),
            txt_ids=text['prompt_ids'].to(device),grid=tuple(sample['grid'])))
        targets.append(noise.float()-x0.float());mask.append(dropped)
    return SimpleNamespace(items=items,caption_dropout_mask=tuple(mask)),torch.cat(targets)
