"""Deterministic global draws and native FLUX.3 packed training inputs."""
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


def pack_samples(samples, noisy_latents, timesteps, device):
    from .vendor.native import native_module
    Packed=native_module().Packed
    modes=('image','image_cond','text')
    data={m:[] for m in modes};ids={m:[] for m in modes};ts={m:[] for m in modes}
    bounds={m:[] for m in modes}; counts={m:0 for m in modes}
    for s,z,t in zip(samples,noisy_latents,timesteps):
        for m,features,positions in (
            ('image',z,s['image_ids']),('text',s['prompt_tokens'],s['prompt_ids']),
            ('image_cond',torch.empty(0,128),torch.empty(0,4))):
            n=len(features);bounds[m].append((counts[m],counts[m]+n));counts[m]+=n
            data[m].append(features.to(device=device,dtype=torch.bfloat16))
            ids[m].append(positions.to(device=device,dtype=torch.float32))
            ts[m].append(torch.full((n,),float(t) if m=='image' else 0.,device=device,dtype=torch.bfloat16))
    packed=Packed(data={m:torch.cat(data[m]) for m in modes},ids={m:torch.cat(ids[m]) for m in modes},
                  timesteps={m:torch.cat(ts[m]) for m in modes},mode_indices=bounds)
    packed.image_grids=tuple(tuple(s['grid']) for s in samples)
    return packed


def condition_samples(samples,empty_prompt,probability,seed,stage,cursor):
    """Replace whole text sequences with native null conditioning, independently of noise RNG."""
    if not 0 <= probability <= 1:
        raise ValueError('Caption dropout must lie in [0, 1]')
    if probability and empty_prompt is None:
        raise ValueError('Caption dropout requires the cached empty prompt')
    mask=[sample_seed(seed,stage,cursor,'caption:'+s['id'])/float(2**63-1)<probability for s in samples]
    conditioned=[dict(s,prompt_tokens=empty_prompt['prompt_tokens'],prompt_ids=empty_prompt['prompt_ids'])
                 if dropped else s for s,dropped in zip(samples,mask)]
    return conditioned,mask


def training_inputs(samples,seed,stage,cursor,device,caption_dropout=0.,empty_prompt=None):
    """Native FLUX.3 1000-point alpha6.93 schedule, epsilon-x0 velocity."""
    zs=[];timesteps=[];targets=[]
    for sample in samples:
        generator=torch.Generator(device=device).manual_seed(sample_seed(seed,stage,cursor,sample['id']))
        # Same scheduler table as set_timesteps(training=True,num_inference_steps=1000).
        index=int(torch.randint(0,1000,(),generator=generator,device=device).item())
        u=1-index/1000
        sigma=6.93*u/(1+(6.93-1)*u)
        x0=sample['image_latents'].to(device=device,dtype=torch.bfloat16)
        noise=torch.randn(x0.shape,generator=generator,device=device,dtype=torch.bfloat16)
        zs.append((1-sigma)*x0+sigma*noise)
        targets.append(noise.float()-x0.float());timesteps.append(sigma)
    conditioned,mask=condition_samples(samples,empty_prompt,caption_dropout,seed,stage,cursor)
    packed=pack_samples(conditioned,zs,timesteps,device)
    packed.caption_dropout_mask=tuple(mask)
    return packed,torch.cat(targets)
