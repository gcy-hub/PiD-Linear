"""Inference RMSNorm + RoPE fusion, retaining each original BF16 rounding stage.

Reused from this project's PiD-QDM-Halo2 fast_points.py; kept local so the
Sliding branch is self-contained. Training continues the native autograd path.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope(Q,K,WQ,WK,COS,SIN,OQ,OK,
               Q0:tl.constexpr,Q1:tl.constexpr,Q2:tl.constexpr,Q3:tl.constexpr,
               K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,K3:tl.constexpr,
               C0:tl.constexpr,C1:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,
               N:tl.constexpr,H:tl.constexpr,D:tl.constexpr,TOTAL:tl.constexpr,
               EQ:tl.constexpr,EK:tl.constexpr,ROTARY:tl.constexpr,ROWS:tl.constexpr,BD:tl.constexpr):
    row=tl.program_id(0)*ROWS+tl.arange(0,ROWS);dim=tl.arange(0,BD)
    batch=row//(N*H);token=row//H%N;head=row%H
    valid=(row[:,None]<TOTAL)&(dim[None,:]<D)
    q=tl.load(Q+batch[:,None]*Q0+token[:,None]*Q1+head[:,None]*Q2+dim[None,:]*Q3,valid,0).to(tl.float32)
    k=tl.load(K+batch[:,None]*K0+token[:,None]*K1+head[:,None]*K2+dim[None,:]*K3,valid,0).to(tl.float32)
    wq=tl.load(WQ+dim,dim<D,0).to(tl.float32);wk=tl.load(WK+dim,dim<D,0).to(tl.float32)
    rq=tl.rsqrt(tl.sum(q*q,1)/D+EQ);rk=tl.rsqrt(tl.sum(k*k,1)/D+EK)
    # Native RMSNorm casts normalized values down BEFORE multiplying the weight.
    q=(q*rq[:,None]).to(Q.dtype.element_ty).to(tl.float32)*wq[None,:]
    k=(k*rk[:,None]).to(K.dtype.element_ty).to(tl.float32)*wk[None,:]
    q=q.to(OQ.dtype.element_ty).to(tl.float32);k=k.to(OK.dtype.element_ty).to(tl.float32)
    if ROTARY:
        pair=tl.broadcast_to((dim^1)[None,:],(ROWS,BD))
        qr=tl.gather(q,pair,1);kr=tl.gather(k,pair,1)
        cos=tl.load(COS+token[:,None]*C0+(dim[None,:]//2)*C1,valid,0)
        sin=tl.load(SIN+token[:,None]*S0+(dim[None,:]//2)*S1,valid,0)
        q=tl.where(dim[None,:]%2==0,q*cos-qr*sin,q*cos+qr*sin)
        k=tl.where(dim[None,:]%2==0,k*cos-kr*sin,k*cos+kr*sin)
    tl.store(OQ+row[:,None]*D+dim[None,:],q,valid);tl.store(OK+row[:,None]*D+dim[None,:],k,valid)


def norm_rope(q,k,qnorm,knorm,pos):
    b,n,h,d=q.shape
    if pos is not None:
        cos,sin=pos if isinstance(pos,tuple) else (pos.real,pos.imag)
    else:cos=sin=qnorm.weight
    oq=torch.empty(q.shape,device=q.device,dtype=torch.promote_types(q.dtype,qnorm.weight.dtype))
    ok=torch.empty(k.shape,device=k.device,dtype=torch.promote_types(k.dtype,knorm.weight.dtype))
    cstride=cos.stride() if pos is not None else (0,0);sstride=sin.stride() if pos is not None else (0,0)
    _norm_rope[(triton.cdiv(b*n*h,4),)](q,k,qnorm.weight,knorm.weight,cos,sin,oq,ok,
        *q.stride(),*k.stride(),*cstride,*sstride,n,h,d,b*n*h,qnorm.variance_epsilon,knorm.variance_epsilon,
        pos is not None,4,triton.next_power_of_2(d),enable_fp_fusion=False)
    return oq,ok
