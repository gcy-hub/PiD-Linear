# SPDX-License-Identifier: Apache-2.0
# Vendored CLEAR-QDM a4262ad attention_processor.py shared math.
# Source SHA256: 9ac2f581072dcb3407310d0c7b3131449c9aab18197a3ac80fd1de969c2164aa
"""CLEAR's shared attention math; native model projections live in adapters.

Preserves the original public FLUX1 processors and spatial_weight key/layout.
Layout is explicit per forward: no shared masks or global feature append lists.
"""
from functools import lru_cache
import math
import torch
from torch import nn
from torch.nn import functional as F

from attention_framework.contracts import AttentionLayout

@lru_cache(maxsize=24)
def _coordinates(layout,down_factor,device):
    sample=[];image=[];ys=[];xs=[];extra=[]
    pools=[]
    for s,(h,w,t) in enumerate(layout.samples):
        if down_factor>1 and (h%down_factor or w%down_factor): raise ValueError('Grid must be divisible by down_factor')
        iy=torch.arange(h,device=device).repeat_interleave(w)
        ix=torch.arange(w,device=device).repeat(h)
        it=torch.ones(h*w,device=device,dtype=torch.bool)
        tt=torch.zeros(t,device=device,dtype=torch.bool)
        iy0=torch.zeros(t,device=device,dtype=torch.long)
        if layout.text_first:
            image.append(torch.cat((tt,it)));ys.append(torch.cat((iy0,iy)));xs.append(torch.cat((iy0,ix)))
        else:
            image.append(torch.cat((it,tt)));ys.append(torch.cat((iy,iy0)));xs.append(torch.cat((ix,iy0)))
        sample.append(torch.full((h*w+t,),s,device=device,dtype=torch.long))
        if down_factor>1: pools.append(torch.full(((h//down_factor)*(w//down_factor),),s,device=device,dtype=torch.long))
    sid=torch.cat(sample);img=torch.cat(image);y=torch.cat(ys);x=torch.cat(xs)
    base=sid.numel()
    if pools:
        ps=torch.cat(pools);sid=torch.cat((sid,ps));img=torch.cat((img,torch.ones_like(ps,dtype=torch.bool)))
        y=torch.cat((y,torch.zeros_like(ps)));x=torch.cat((x,torch.zeros_like(ps)))
    extra=torch.arange(sid.numel(),device=device)>=base
    return sid,img,y,x,extra


def _predicate(layout,radius,down_factor,device,window_shape='circle',kernel_size=11):
    sid,img,y,x,extra=_coordinates(layout,down_factor,str(device))
    qlen,klen=layout.length,len(sid)
    def mask(b,h,q,k):
        qi=q.clamp(0,qlen-1);ki=k.clamp(0,klen-1)
        same=sid[qi]==sid[ki]
        base=(~extra[ki])
        if window_shape=='circle':
            local=(y[qi]-y[ki])**2+(x[qi]-x[ki])**2<radius**2
        else:
            local=((y[qi]-y[ki]).abs()<=kernel_size//2)&((x[qi]-x[ki]).abs()<=kernel_size//2)
        allowed=torch.where(img[qi],(~img[ki])|local|extra[ki],base)
        return (q<qlen)&(k<klen)&same&allowed
    return mask,klen


def clear_mask(layout,radius,device,down_factor=1,window_shape='circle',kernel_size=11):
    fn,klen=_predicate(layout,radius,down_factor,device,window_shape,kernel_size)
    return fn(None,None,torch.arange(layout.length,device=device)[:,None],torch.arange(klen,device=device)[None,:])

@lru_cache(maxsize=24)
def _block_mask(layout,radius,down_factor,device,window_shape='circle',kernel_size=11):
    from torch.nn.attention.flex_attention import create_block_mask
    fn,klen=_predicate(layout,radius,down_factor,device,window_shape,kernel_size)
    return create_block_mask(fn,B=None,H=None,Q_LEN=layout.length,KV_LEN=klen,device=device,_compile=True)

@lru_cache(maxsize=1)
def _kernel():
    from torch.nn.attention.flex_attention import flex_attention
    return torch.compile(flex_attention,dynamic=False)


class ClearAttention(nn.Module):
    def __init__(self,window_size=16,down_factor=1,train_grid=64,window_shape='circle',kernel_size=11):
        super().__init__()
        self.window_size=window_size;self.down_factor=down_factor;self.train_grid=train_grid
        if window_shape not in ('circle','square') or kernel_size<1 or kernel_size%2!=1:
            raise ValueError('Attention needs circle/square and a positive odd kernel')
        self.window_shape,self.kernel_size=window_shape,kernel_size
        if down_factor>1:
            self.spatial_weight=nn.Parameter(torch.ones(1,1,1,down_factor,1,down_factor,1)/(down_factor**2))
        else: self.register_parameter('spatial_weight',None)

    def _pool(self,x,layout):
        chunks=[];offset=0;d=self.down_factor
        for h,w,t in layout.samples:
            start=offset+(t if layout.text_first else 0)
            part=x[:,:,start:start+h*w].unflatten(2,(h//d,d,w//d,d))
            chunks.append((part*self.spatial_weight.to(x.dtype)).sum((3,5)).flatten(2,3))
            offset+=h*w+t
        return torch.cat(chunks,2)

    def forward(self,q,k,v,layout,proportional_attention=False):
        q,k=q.to(v.dtype),k.to(v.dtype)
        if q.shape[-2]!=layout.length or k.shape[-2]!=layout.length: raise ValueError('Attention/layout length mismatch')
        if proportional_attention:
            ratios=[]
            for h,w,t in layout.samples:
                numerator=(h*w+t)*(10 if self.down_factor>1 else 1)
                ratio=math.sqrt(math.log(numerator,self.train_grid**2+t))
                ratios.extend([ratio]*(h*w+t))
            q=q*torch.tensor(ratios,device=q.device,dtype=q.dtype)[None,None,:,None]
        if self.window_size is None:
            # Native full attention, with explicit Packed sample separation.
            result=[];offset=0
            for h,w,t in layout.samples:
                end=offset+h*w+t
                result.append(F.scaled_dot_product_attention(q[:,:,offset:end],k[:,:,offset:end],v[:,:,offset:end]))
                offset=end
            return torch.cat(result,2)
        if self.down_factor>1:
            k=torch.cat((k,self._pool(k,layout)),2);v=torch.cat((v,self._pool(v,layout)),2)
        if q.device.type=='cuda':
            # Ampere's 99 KiB shared-memory limit cannot fit PyTorch 2.10's
            # default 128x128 head-dim128 kernel with distinct Q/K/V tensors.
            # Keep the exact sparse computation and use smaller native tiles.
            options={}
            if torch.cuda.get_device_capability(q.device)[0]<9:
                options=dict(BLOCK_M=64,BLOCK_N=64,num_stages=2,
                    bwd_BLOCK_M1=32,bwd_BLOCK_N1=64,bwd_BLOCK_M2=64,bwd_BLOCK_N2=32)
            return _kernel()(q,k,v,block_mask=_block_mask(layout,self.window_size,self.down_factor,str(q.device),self.window_shape,self.kernel_size),kernel_options=options)
        return F.scaled_dot_product_attention(q,k,v,attn_mask=clear_mask(layout,self.window_size,q.device,self.down_factor,self.window_shape,self.kernel_size))


def sample_slices(layout):
    offset = 0
    for h, w, t in layout.samples:
        image_start = offset + (t if layout.text_first else 0)
        text_start = offset if layout.text_first else offset + h * w
        yield (h, w), slice(offset, offset + h*w+t), slice(image_start, image_start+h*w), slice(text_start, text_start+t)
        offset += h*w+t


def halo_attention(q, k, v, layout, region_size, halo):
    """Positioned QKV [B,L,H,D]; gather valid original-grid keys per region."""
    from .qdm import pack_regions, halo_indices, unpack_regions
    pieces = []
    for grid, whole, image, text in sample_slices(layout):
        qp, _ = pack_regions(q[:, image], grid, region_size)
        indices, valid = halo_indices(grid, region_size, halo, q.device)
        windows = []
        # Bound temporary K/V gathers independently of image resolution.
        for start in range(0, qp.shape[1], 16):
            stop = min(start+16, qp.shape[1])
            count = stop-start
            kk, vv = (z[:, image][:, indices[start:stop]].transpose(2, 3) for z in (k, v))
            tk, tv = (z[:, text].transpose(1, 2)[:, None].expand(-1, count, -1, -1, -1) for z in (k, v))
            mask = torch.cat((valid[start:stop], torch.ones(count, tk.shape[-2], device=q.device, dtype=torch.bool)), -1)
            out = F.scaled_dot_product_attention(qp[:, start:stop].transpose(2, 3), torch.cat((kk, tk), -2), torch.cat((vv, tv), -2), attn_mask=mask[None, :, None, None, :])
            windows.append(out.transpose(2, 3))
        image_out = unpack_regions(torch.cat(windows, 1), grid, region_size)
        text_out = F.scaled_dot_product_attention(q[:, text].transpose(1, 2), k[:, whole].transpose(1, 2), v[:, whole].transpose(1, 2)).transpose(1, 2)
        pieces.extend((text_out, image_out) if layout.text_first else (image_out, text_out))
    return torch.cat(pieces, 1)
