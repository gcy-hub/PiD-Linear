"""Inference-only spatial partition interventions. Formal training modules are untouched."""
import torch
from torch import nn
from torch.nn import functional as F
from pid._src.networks.pixeldit_official import apply_rotary_emb


def quadrant_labels(grid, device, split=None):
    h,w=map(int,grid)
    sy,sx=split or ((h+1)//2,(w+1)//2)
    if not (0<sy<h and 0<sx<w):
        raise ValueError(f'Invalid interior split {split} for {grid}')
    yy,xx=torch.meshgrid(torch.arange(h,device=device),torch.arange(w,device=device),indexing='ij')
    return ((yy>=sy).long()*2+(xx>=sx).long()).flatten()


def select_mask(mask, queries, keys):
    if mask is None:return None
    # SDPA masks broadcast across batch/heads/query. Preserve boolean OR additive semantics.
    if mask.shape[-2]!=1:mask=mask.index_select(-2,queries)
    if mask.shape[-1]!=1:mask=mask.index_select(-1,keys)
    return mask


def restricted_sdpa(q,k,v,text_len,labels,attn_mask=None):
    """Exact quadrant-restricted image queries; text queries retain all original keys.

    q/k/v: [B,H,text+image,D]. No dense image NxN mask is constructed.
    Labels may also all be zero to verify exact full-attention equivalence.
    """
    device=q.device
    text_ix=torch.arange(text_len,device=device)
    full_ix=torch.arange(k.shape[-2],device=device)
    text=F.scaled_dot_product_attention(q[:,:,:text_len],k,v,
        attn_mask=select_mask(attn_mask,text_ix,full_ix),dropout_p=0.)
    image=torch.empty_like(q[:,:,text_len:])
    for label in labels.unique():
        ix=torch.where(labels==label)[0]
        qi=ix+text_len
        ki=torch.cat((text_ix,qi))
        value=F.scaled_dot_product_attention(q.index_select(2,qi),k.index_select(2,ki),
            v.index_select(2,ki),attn_mask=select_mask(attn_mask,qi,ki),dropout_p=0.)
        image[:,:,ix]=value
    return image,text


class QuadrantMaskAttention(nn.Module):
    is_vdn_attention=True
    def __init__(self,original):
        super().__init__()
        self.orig=original
        self.masked=False
        self.restrict_quadrants=True
        self.callback=None

    def restricted(self,x,y,pos_img,pos_txt,grid,attn_mask=None):
        o=self.orig;b,n,c=x.shape;nt=y.shape[1];h=o.num_heads;d=o.head_dim
        q,k,v=o.qkv_x(x).reshape(b,n,3,h,d).unbind(2)
        tq,tk,tv=o.qkv_y(y).reshape(b,nt,3,h,d).unbind(2)
        q,k=apply_rotary_emb(o.q_norm_x(q),o.k_norm_x(k),freqs_cis=pos_img)
        tq,tk=o.q_norm_y(tq),o.k_norm_y(tk)
        if pos_txt is not None:tq,tk=apply_rotary_emb(tq,tk,freqs_cis=pos_txt)
        image,text=restricted_sdpa(*[torch.cat((t,z),1).transpose(1,2) for t,z in ((tq,q),(tk,k),(tv,v))],
                                  nt,quadrant_labels(grid,x.device) if self.restrict_quadrants else torch.zeros(n,device=x.device,dtype=torch.long),attn_mask)
        image=image.transpose(1,2).reshape(b,n,c);text=text.transpose(1,2).reshape(b,nt,c)
        return o.proj_drop_x(o.proj_x(image)),o.proj_drop_y(o.proj_y(text))

    def forward(self,x,y,pos_img,pos_txt=None,*,grid_size,text_valid_mask=None,attn_mask=None):
        if self.masked:return self.restricted(x,y,pos_img,pos_txt,grid_size,attn_mask)
        output=self.orig(x,y,pos_img,pos_txt,attn_mask)
        if self.callback is not None:
            changed=self.restricted(x,y,pos_img,pos_txt,grid_size,attn_mask)
            self.callback(output,changed,grid_size)
        return output


def error_metrics(value,reference):
    error=(value.float()-reference.float()).square().sum().item()
    energy=reference.float().square().sum().item()
    return dict(relative_l2=(error/max(energy,1e-30))**.5,
                mse=error/reference.numel(),reference_rms=(energy/reference.numel())**.5,
                output_rms=value.float().square().mean().sqrt().item())


def token_map(value,reference=None,grid=None):
    value=value.float()
    if reference is not None:value=value-reference.float()
    return value.square().mean((0,2)).reshape(grid)


def spatial_summary(energy,split=None,band=2):
    """Energy map: report absolute means as well as ratios; ratios alone can mislead."""
    energy=energy.float();h,w=energy.shape;sy,sx=split or ((h+1)//2,(w+1)//2)
    yy,xx=torch.meshgrid(torch.arange(h,device=energy.device),torch.arange(w,device=energy.device),indexing='ij')
    boundary=(yy-sy+.5).abs()<band
    boundary=boundary|((xx-sx+.5).abs()<band)
    interior=~boundary
    bmean=energy[boundary].mean().item();imean=energy[interior].mean().item() if interior.any() else 0.
    quadrants=[]
    for ys,xs in ((slice(0,sy),slice(0,sx)),(slice(0,sy),slice(sx,w)),(slice(sy,h),slice(0,sx)),(slice(sy,h),slice(sx,w))):
        quadrants.append(energy[ys,xs].mean().item())
    return dict(mean=energy.mean().item(),center_boundary_mean=bmean,interior_mean=imean,
                boundary_to_interior=bmean/max(imean,1e-30),quadrant_mean=quadrants,
                row_profile=energy.mean(1).tolist(),column_profile=energy.mean(0).tolist())


def discontinuity(value,grid):
    b,n,c=value.shape;z=value.float().reshape(b,*grid,c)
    dx=(z[:,:,1:]-z[:,:,:-1]).square().mean((0,3))
    dy=(z[:,1:]-z[:,:-1]).square().mean((0,3))
    sy,sx=(grid[0]+1)//2,(grid[1]+1)//2
    # Exclude center from the reference; also preserve full profiles for matched controls.
    xp=dx.mean(0);yp=dy.mean(1)
    xcontrol=torch.cat((xp[:max(0,sx-2)],xp[min(len(xp),sx+1):])).mean()
    ycontrol=torch.cat((yp[:max(0,sy-2)],yp[min(len(yp),sy+1):])).mean()
    return dict(vertical_center= xp[sx-1].item(),horizontal_center=yp[sy-1].item(),
                vertical_control=xcontrol.item(),horizontal_control=ycontrol.item(),
                vertical_ratio=(xp[sx-1]/xcontrol.clamp_min(1e-30)).item(),
                horizontal_ratio=(yp[sy-1]/ycontrol.clamp_min(1e-30)).item(),
                vertical_profile=xp.tolist(),horizontal_profile=yp.tolist())
