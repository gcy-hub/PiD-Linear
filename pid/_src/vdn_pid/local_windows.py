"""Chunked spatial Q/KV layouts; native coordinates and joint text Softmax stay intact."""
import torch
from torch.nn import functional as F
from pid._src.networks.pixeldit_official import apply_rotary_emb


def region_indices(grid,size,halo,offset,device):
    h,w=map(int,grid)
    if min(h,w,size)<1 or halo<0 or not 0<=offset<size:
        raise ValueError('Invalid spatial attention grid/halo/offset')
    gh,gw=(h+offset+size-1)//size,(w+offset+size-1)//size
    row,col=torch.meshgrid(torch.arange(gh,device=device)*size-offset,
                           torch.arange(gw,device=device)*size-offset,indexing='ij')
    def indices(border):
        delta=torch.arange(-border,size+border,device=device)
        dy,dx=torch.meshgrid(delta,delta,indexing='ij')
        yy=row.flatten()[:,None]+dy.flatten();xx=col.flatten()[:,None]+dx.flatten()
        valid=(yy>=0)&(yy<h)&(xx>=0)&(xx<w)
        return yy.clamp(0,h-1)*w+xx.clamp(0,w-1),valid
    qi,qvalid=indices(0);ki,kvalid=indices(halo)
    return qi,qvalid,ki,kvalid


def local_attention(module,x,y,qkv,tqkv,pos_img,pos_txt,grid,attn_mask=None):
    orig=module.orig;b,n,c=x.shape;heads,d=orig.num_heads,orig.head_dim
    q,k,v=qkv;tq,tk,tv=tqkv
    # Rotate the ORIGINAL full-grid coordinates before gathering either layout.
    q,k=apply_rotary_emb(orig.q_norm_x(q),orig.k_norm_x(k),freqs_cis=pos_img)
    tq,tk=orig.q_norm_y(tq),orig.k_norm_y(tk)
    if pos_txt is not None:tq,tk=apply_rotary_emb(tq,tk,freqs_cis=pos_txt)
    qi,qvalid,ki,kvalid=region_indices(grid,module.region_size,module.local_halo,module.window_offset,x.device)
    regions,s=qi.shape;kv_size=ki.shape[1];length=y.shape[1];pieces=[]
    text_valid=torch.ones(b,length,device=x.device,dtype=torch.bool)
    if attn_mask is not None:text_valid=attn_mask.reshape(b,-1)[:,:length].bool()
    needs_mask=attn_mask is not None or not bool(kvalid.all())
    for start in range(0,regions,module.region_chunk):
        stop=min(regions,start+module.region_chunk);count=stop-start
        qq=q[:,qi[start:stop]].permute(0,1,3,2,4).reshape(b*count,heads,s,d)
        kk=k[:,ki[start:stop]].permute(0,1,3,2,4)
        vv=v[:,ki[start:stop]].permute(0,1,3,2,4)
        textk=tk.transpose(1,2)[:,None].expand(b,count,heads,length,d)
        textv=tv.transpose(1,2)[:,None].expand(b,count,heads,length,d)
        kk=torch.cat((textk,kk),3).reshape(b*count,heads,length+kv_size,d)
        vv=torch.cat((textv,vv),3).reshape_as(kk)
        mask=None
        if needs_mask:
            mask=torch.cat((text_valid[:,None].expand(b,count,length),kvalid[None,start:stop].expand(b,count,kv_size)),-1).reshape(b*count,1,1,length+kv_size)
        out=F.scaled_dot_product_attention(qq,kk,vv,attn_mask=mask,dropout_p=0.)
        pieces.append(out.reshape(b,count,heads,s,d).permute(0,1,3,2,4))
    # Each valid query appears once. Only K/V neighborhoods overlap for Halo.
    order=qi[qvalid];packed=torch.cat(pieces,1).reshape(b,regions*s,heads,d)
    image=packed[:,qvalid.flatten()][:,torch.argsort(order)]
    text=F.scaled_dot_product_attention(tq.transpose(1,2),torch.cat((tk,k),1).transpose(1,2),
        torch.cat((tv,v),1).transpose(1,2),attn_mask=attn_mask,dropout_p=0.).transpose(1,2)
    if not module.bypass_softmax_gate:
        image=image*module.softmax_gate_x(x.reshape(-1,c)).reshape(b,n,heads,1)
        text=text*module.softmax_gate_y(y.reshape(-1,c)).reshape(b,length,heads,1)
    return orig.proj_drop_x(orig.proj_x(image.reshape(b,n,c))),orig.proj_drop_y(orig.proj_y(text.reshape(b,length,c)))
