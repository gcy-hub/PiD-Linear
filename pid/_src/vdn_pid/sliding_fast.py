"""Inference-only centered 2-D Sliding + text, fused with exact edge masking.

A query tile shares a rectangular union of image keys. An explicit spatial
predicate selects each query's own neighborhood. Text and image keys share
one online Softmax; no per-query K/V expansion or border repair is needed.
Training keeps the differentiable NATTEN path in sliding.py.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sliding(Q,K,V,TK,TV,VALID,O,
             QS:tl.constexpr,KS:tl.constexpr,VS:tl.constexpr,
             TKS:tl.constexpr,TVS:tl.constexpr,MS:tl.constexpr,
             H:tl.constexpr,W:tl.constexpr,HEADS:tl.constexpr,D:tl.constexpr,T:tl.constexpr,
             L:tl.constexpr,R:tl.constexpr,M:tl.constexpr,BLOCK_K:tl.constexpr,BLOCK_D:tl.constexpr):
    group=tl.program_id(0);head=tl.program_id(1);batch=tl.program_id(2)
    groups_x=tl.cdiv(W,M)
    yy=group//groups_x;start_x=group%groups_x*M
    x=start_x+tl.arange(0,M);token=yy*W+x
    dd=tl.arange(0,BLOCK_D)
    qq=tl.load(Q+batch*QS[0]+token[:,None]*QS[1]+head*QS[2]+dd[None,:]*QS[3],
               (x[:,None]<W)&(dd[None,:]<D),0)
    span=M+L+R
    image_keys=(L+R+1)*span
    m=tl.full((M,),-float('inf'),tl.float32)
    denom=tl.zeros((M,),tl.float32)
    accum=tl.zeros((M,BLOCK_D),tl.float32)
    scale=1.4426950408889634*tl.rsqrt(float(D))
    for first in range(tl.cdiv(T+image_keys,BLOCK_K)):
        position=first*BLOCK_K+tl.arange(0,BLOCK_K)
        is_text=position<T
        offset=position-T
        ky=yy-L+offset//span
        kx=start_x-L+offset%span
        ik=ky*W+kx
        image_valid=(offset>=0)&(offset<image_keys)&(ky>=0)&(ky<H)&(kx>=0)&(kx<W)
        text_valid=tl.load(VALID+batch*MS[0]+position*MS[1],is_text,0)
        kk=tl.load(K+batch*KS[0]+ik[:,None]*KS[1]+head*KS[2]+dd[None,:]*KS[3],
                   image_valid[:,None]&(dd[None,:]<D),0)
        vv=tl.load(V+batch*VS[0]+ik[:,None]*VS[1]+head*VS[2]+dd[None,:]*VS[3],
                   image_valid[:,None]&(dd[None,:]<D),0)
        kt=tl.load(TK+batch*TKS[0]+position[:,None]*TKS[1]+head*TKS[2]+dd[None,:]*TKS[3],
                   is_text[:,None]&(dd[None,:]<D),0)
        vt=tl.load(TV+batch*TVS[0]+position[:,None]*TVS[1]+head*TVS[2]+dd[None,:]*TVS[3],
                   is_text[:,None]&(dd[None,:]<D),0)
        kk=tl.where(is_text[:,None],kt,kk)
        vv=tl.where(is_text[:,None],vt,vv)
        visible=(is_text[None,:]&text_valid[None,:])|(
            image_valid[None,:]&(kx[None,:]>=x[:,None]-L)&(kx[None,:]<=x[:,None]+R))
        scores=tl.dot(qq,tl.trans(kk)).to(tl.float32)*scale
        scores=tl.where(visible,scores,-float('inf'))
        updated=tl.maximum(m,tl.max(scores,1))
        safe=tl.where(updated==-float('inf'),0.,updated)
        alpha=tl.exp2(m-safe)
        probability=tl.exp2(scores-safe[:,None])
        denom=denom*alpha+tl.sum(probability,1)
        accum=accum*alpha[:,None]+tl.dot(probability.to(vv.dtype),vv)
        m=updated
    result=accum/tl.maximum(denom[:,None],1.e-20)
    tl.store(O+((batch*H*W+token[:,None])*HEADS+head)*D+dd[None,:],result,
             (x[:,None]<W)&(dd[None,:]<D))


def sliding_forward(q,k,v,tk,tv,grid,kernel,text_valid=None,*,query_tile=32):
    """No-autograd fused forward; batch and text validity are isolated explicitly."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q,k,v,tk,tv)):
        raise RuntimeError('Fused Sliding forward is inference-only; use NATTEN for training')
    b,n,heads,d=q.shape;h,w=map(int,grid)
    if h*w!=n or kernel<1 or query_tile not in (16,32):
        raise ValueError('Invalid Sliding grid/kernel/tile')
    if q.dtype not in (torch.float16,torch.bfloat16) or not q.is_cuda:
        raise ValueError('Fused Sliding requires CUDA FP16/BF16')
    t=tk.shape[1]
    if text_valid is None:text_valid=torch.ones(b,t,device=q.device,dtype=torch.bool)
    out=torch.empty(q.shape,device=q.device,dtype=q.dtype)
    _sliding[(h*triton.cdiv(w,query_tile),heads,b)](
        q,k,v,tk,tv,text_valid,out,q.stride(),k.stride(),v.stride(),tk.stride(),tv.stride(),text_valid.stride(),
        h,w,heads,d,t,kernel//2,(kernel-1)//2,query_tile,64,triton.next_power_of_2(d),num_warps=4,
        num_stages=1 if d == 128 else 3)
    return out
