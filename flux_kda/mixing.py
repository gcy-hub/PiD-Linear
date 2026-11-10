"""Calibrate image-query text logits while keeping both attention branches intact."""
from functools import lru_cache
import json
from pathlib import Path
import types
import torch
from torch.nn import functional as F
from flux3_qdm.sliding import _block_mask, _predicate


@lru_cache(maxsize=1)
def compiled_attention():
    from torch.nn.attention.flex_attention import BlockMask, flex_attention
    def attend(q,k,v,bias,image,mask):
        def adjust(score,b,h,query,key):
            return score + torch.where((query < image) & (key >= image),bias[h],0.)
        current=BlockMask(seq_lengths=(q.shape[-2],k.shape[-2]),
                          kv_num_blocks=mask.kv_num_blocks,kv_indices=mask.kv_indices,
                          full_kv_num_blocks=mask.full_kv_num_blocks,full_kv_indices=mask.full_kv_indices,
                          q_num_blocks=mask.q_num_blocks,q_indices=mask.q_indices,
                          full_q_num_blocks=mask.full_q_num_blocks,full_q_indices=mask.full_q_indices,
                          BLOCK_SIZE=mask.BLOCK_SIZE,mask_mod=mask.mask_mod)
        return flex_attention(q,k,v,score_mod=adjust,block_mask=current,
                              kernel_options=dict(BACKEND='TRITON',BLOCK_M=64,BLOCK_N=64,num_stages=2,
                                                  bwd_BLOCK_M1=32,bwd_BLOCK_N1=64,bwd_BLOCK_M2=64,bwd_BLOCK_N2=32))
    return torch.compile(attend,dynamic=True,fullgraph=True)


def biased_attention(q,k,v,grid,text,bias,kernel=11):
    height,width=map(int,grid);image=height*width
    if q.is_cuda:
        return compiled_attention()(q,k,v,bias,image,_block_mask(height,width,text,kernel,str(q.device)))
    ids=torch.arange(image+text,device=q.device)
    allowed=_predicate(height,width,text,kernel)(None,None,ids[:,None],ids[None,:])
    mask=torch.where(allowed,0.,float('-inf'))[None,None].expand(1,q.shape[1],-1,-1).clone()
    mask[:,:,:image,image:]+=bias[None,:,None,None]
    return F.scaled_dot_product_attention(q,k,v,attn_mask=mask)


def apply_bias(model,values):
    for block in model.qdm_blocks:
        if str(block.layer_index) not in values:continue
        bias=torch.tensor(values[str(block.layer_index)],device=next(block.parameters()).device,dtype=torch.float32)
        if bias.shape != (block.heads,) or not torch.isfinite(bias).all():
            raise ValueError('Expected one finite text-logit bias per attention head')
        block.register_buffer('text_logit_bias',bias,persistent=False)
        def local_attention(self,x,q,k,v,samples):
            pieces=[]
            for sample in samples:
                output=biased_attention(*(z[:,:,sample.start:sample.end] for z in (q,k,v)),
                                         sample.grid,sample.text_end-sample.text_start,self.text_logit_bias,self.sliding_kernel_size)
                pieces.append(output[0].transpose(0,1))
            return torch.cat(pieces)
        block.local_attention=types.MethodType(local_attention,block)


def read_bias(path):
    return json.loads(Path(path).read_text())['text_logit_bias']
