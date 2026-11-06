"""FLUX.2 Klein: 5 double-stream + 20 single-stream native Attention adapters.

Single-stream FFN shares fused projections: LoRA changes only Q/K/V rows and
Attention output columns, never the parallel FFN. A1 aligns the projected
Attention contribution, excluding the frozen FFN from its denominator.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from diffusers import Flux2Transformer2DModel
from diffusers.models.embeddings import apply_rotary_emb
from .memory import QuadrantMemory
from .vendor.gates import OutputGate
from .sliding import sliding_joint_attention


class SliceLoRA(nn.Module):
    def __init__(self, base, rank=64, alpha=64, rows=None, columns=None):
        super().__init__()
        self.base=base; self.rank=rank; self.alpha=alpha; self.adapters_enabled=True
        self.rows=rows or base.out_features; self.columns=columns or base.in_features
        self.lora_A=nn.Parameter(torch.empty(rank,self.columns,device=base.weight.device,dtype=torch.float32))
        self.lora_B=nn.Parameter(torch.zeros(self.rows,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A,a=5**.5)
        base.requires_grad_(False)

    @property
    def weight(self): return self.base.weight
    @property
    def bias(self): return self.base.bias
    @property
    def in_features(self): return self.base.in_features
    @property
    def out_features(self): return self.base.out_features

    def delta(self,x):
        with torch.autocast(x.device.type,enabled=False):
            return F.linear(F.linear(x[...,:self.columns].float(),self.lora_A),self.lora_B)*(self.alpha/self.rank)

    def forward(self,x):
        y=self.base(x)
        if self.adapters_enabled:
            d=self.delta(x).to(y.dtype)
            y=y+F.pad(d,(0,y.shape[-1]-self.rows))
        return y


def linear(layer,x,teacher=False):
    return layer.base(x) if teacher and isinstance(layer,SliceLoRA) else layer(x)


def attention_projection(layer,x,hidden,teacher=False):
    """Apply only the leading Attention columns of a fused Attention+FFN O."""
    base=layer.base if isinstance(layer,SliceLoRA) else layer
    y=F.linear(x,base.weight[:,:hidden])
    if isinstance(layer,SliceLoRA) and not teacher and layer.adapters_enabled:
        y=y+layer.delta(x).to(y.dtype)
    return y


@contextmanager
def disable_lora(module):
    adapters=[m for m in module.modules() if isinstance(m,SliceLoRA)]
    previous=[m.adapters_enabled for m in adapters]
    for m in adapters:m.adapters_enabled=False
    try:yield
    finally:
        for m,enabled in zip(adapters,previous):m.adapters_enabled=enabled


@dataclass(frozen=True)
class Layout:
    start:int; end:int; image_start:int; image_end:int; text_start:int; text_end:int; grid:tuple


@dataclass(frozen=True)
class AttentionInput:
    image:torch.Tensor
    text:torch.Tensor|None
    rotary:tuple
    grid:tuple
    text_length:int
    samples:tuple


class HybridAttention(nn.Module):
    def __init__(self,orig,kernel_size=11,single=False,layer_index=0):
        super().__init__()
        self.orig=orig;self.single=single;self.layer_index=layer_index
        self.hidden=orig.inner_dim;self.heads=orig.heads;self.kernel_size=kernel_size
        self.remote=QuadrantMemory(self.hidden,self.heads)
        self.softmax_gate=OutputGate(self.hidden,self.heads,init_value=.99)
        self.remote_enabled=True;self.bypass_softmax_gate=False

    def info(self,image,text,rotary,grid,text_length):
        if grid is None or grid[0]*grid[1] != image.shape[1]-(text_length if self.single else 0):
            raise ValueError('Explicit image grid does not match FLUX.2 sequence')
        n=grid[0]*grid[1];b=image.shape[0]
        layouts=tuple(Layout(i*(n+text_length),(i+1)*(n+text_length),i*(n+text_length),
                             i*(n+text_length)+n,i*(n+text_length)+n,(i+1)*(n+text_length),grid) for i in range(b))
        return AttentionInput(image,text,rotary,tuple(grid),text_length,layouts)

    def features(self,info,teacher):
        a=self.orig;h=self.hidden;heads=self.heads;t=info.text_length
        if self.single:
            fused=linear(a.to_qkv_mlp_proj,info.image,teacher)
            q,k,v=fused[...,:3*h].chunk(3,-1)
            mlp=a.mlp_act_fn(fused[...,3*h:])
        else:
            q,k,v=[linear(p,info.image,teacher) for p in (a.to_q,a.to_k,a.to_v)]
            tq,tk,tv=[linear(p,info.text,teacher).unflatten(-1,(heads,-1)) for p in (a.add_q_proj,a.add_k_proj,a.add_v_proj)]
            tq,tk=a.norm_added_q(tq),a.norm_added_k(tk);mlp=None
        q,k,v=[z.unflatten(-1,(heads,-1)) for z in (q,k,v)]
        q,k=a.norm_q(q),a.norm_k(k)
        if not self.single:q,k,v=[torch.cat(pair,1) for pair in ((tq,q),(tk,k),(tv,v))]
        # Native sequence is text-first. The shared sliding kernel is image-first.
        qrope=apply_rotary_emb(q,info.rotary,sequence_dim=1).to(v.dtype)
        krope=apply_rotary_emb(k,info.rotary,sequence_dim=1).to(v.dtype)
        if teacher:
            out=F.scaled_dot_product_attention(qrope.transpose(1,2),krope.transpose(1,2),v.transpose(1,2)).transpose(1,2)
        else:
            reorder=lambda z:torch.cat((z[:,t:],z[:,:t]),1).transpose(1,2)
            vals=[]
            qr,kr,vr=reorder(qrope),reorder(krope),reorder(v)
            for batch in range(q.shape[0]):
                z=sliding_joint_attention(qr[batch:batch+1],kr[batch:batch+1],
                    vr[batch:batch+1],info.grid,t,self.kernel_size).transpose(1,2)
                vals.append(torch.cat((z[:,-t:] if t else z[:,:0],z[:,:-t] if t else z),1))
            out=torch.cat(vals,0)
        return out,(q[:,t:],k[:,t:],v[:,t:]),mlp

    def projected(self,info,teacher=False,with_mlp=False):
        out,qkv,mlp=self.features(info,teacher);a=self.orig;t=info.text_length;h=self.hidden
        image_x=info.image[:,t:] if self.single else info.image
        if not teacher and not self.bypass_softmax_gate:
            gate=self.softmax_gate(image_x.reshape(-1,h)).reshape(*image_x.shape[:2],self.heads,1)
            out=torch.cat((out[:,:t],out[:,t:]*gate),1)
        flat=out.flatten(2)
        if self.single:
            projected=linear(a.to_out,torch.cat((flat,mlp),-1),teacher) if with_mlp else attention_projection(a.to_out,flat,h,teacher)
            image,text=projected[:,t:],projected[:,:t]
        else:
            image=linear(a.to_out[0],flat[:,t:],teacher);image=a.to_out[1](image)
            text=linear(a.to_add_out,flat[:,:t],teacher)
        if not teacher and self.remote_enabled:
            memory=torch.stack([self.remote(image_x[b],tuple(z[b] for z in qkv),info.grid) for b in range(image_x.shape[0])])
            image=image+memory
        return image,text

    def student_attention(self,info):
        image,text=self.projected(info)
        return torch.cat((image,text),1).flatten(0,1)

    def forward(self,hidden_states,encoder_hidden_states=None,attention_mask=None,image_rotary_emb=None,
                grid=None,text_length=None,teacher_mode=False,alignment_callback=None,**kwargs):
        if attention_mask is not None:raise ValueError('FLUX.2 native fixed text padding is used; external masks are unsupported')
        t=int(text_length) if self.single else encoder_hidden_states.shape[1]
        info=self.info(hidden_states,encoder_hidden_states,image_rotary_emb,grid,t)
        if teacher_mode:
            # Delegate the returned teacher path exactly to native code (including
            # its fused FFN rounding), never to a decomposed surrogate.
            with disable_lora(self.orig):
                result=self.orig(hidden_states,image_rotary_emb=image_rotary_emb) if self.single else self.orig(
                    hidden_states,encoder_hidden_states=encoder_hidden_states,image_rotary_emb=image_rotary_emb)
                if alignment_callback is not None:
                    image,text=self.projected(info,teacher=True)
                    alignment_callback(self,info,torch.cat((image,text),1).flatten(0,1).detach())
            return result
        image,text=self.projected(info,with_mlp=self.single)
        return torch.cat((text,image),1) if self.single else (image,text)


class Flux2QDM(nn.Module):
    def __init__(self,base,kernel_size=11):
        super().__init__();self.base=base;self.teacher_mode=False
        for index,block in enumerate([*base.transformer_blocks,*base.single_transformer_blocks]):
            old=block.attn
            wrapped=HybridAttention(old,kernel_size,index>=len(base.transformer_blocks),index)
            wrapped.remote.to(device=next(old.parameters()).device)
            wrapped.softmax_gate.to(device=next(old.parameters()).device)
            block.attn=wrapped
        self.base.requires_grad_(False)

    @property
    def qdm_blocks(self):return tuple(m for m in self.modules() if isinstance(m,HybridAttention))

    def native_forward(self,grid,alignment_callback=None,use_gradient_checkpointing=False,**kwargs):
        self.base.gradient_checkpointing=use_gradient_checkpointing
        options=dict(kwargs.pop('joint_attention_kwargs',None) or {},grid=tuple(grid),
                     text_length=kwargs['encoder_hidden_states'].shape[1],teacher_mode=self.teacher_mode,
                     alignment_callback=alignment_callback)
        return self.base(**kwargs,joint_attention_kwargs=options)

    def forward(self,packed,use_gradient_checkpointing=False,alignment_callback=None):
        first=packed.items[0]
        for sample in packed.items:
            if tuple(sample['grid'])!=tuple(first['grid']) or sample['encoder_hidden_states'].shape[1:]!=first['encoder_hidden_states'].shape[1:]:
                raise ValueError('A batch must share its image bucket and native text length')
            if not torch.equal(sample['img_ids'],first['img_ids']) or not torch.equal(sample['txt_ids'],first['txt_ids']):
                raise ValueError('Batched native position IDs must agree')
        # The full native batch reaches each A1 callback once. Relative MSE then
        # normalizes by the entire batch energy instead of summing bs losses.
        batch={key:torch.cat([s[key] for s in packed.items],0)
               for key in ('hidden_states','encoder_hidden_states','timestep')}
        batch.update(grid=first['grid'],img_ids=first['img_ids'],txt_ids=first['txt_ids'])
        result=self.native_forward(**batch,use_gradient_checkpointing=use_gradient_checkpointing,
            alignment_callback=alignment_callback,return_dict=False)[0]
        return result.flatten(0,1)


@contextmanager
def execution_mode(model,teacher=False):
    old=model.teacher_mode;model.teacher_mode=teacher
    try:yield model
    finally:model.teacher_mode=old


def iter_qdm_blocks(model):return iter(model.qdm_blocks)


def normalize_qdm_layers(layers=None,depth=25):
    expected=tuple(range(depth))
    if layers is None:return expected
    layers=tuple(map(int,layers.split(','))) if isinstance(layers,str) else tuple(layers)
    if layers!=expected:raise ValueError('FLUX2 initial experiment replaces all 5 double + 20 single Attention layers')
    return expected


def configure_stage(model,stage,rank=64,alpha=64,**unused):
    stage=stage.lower()
    if stage=='b':
        for block in model.qdm_blocks:
            a=block.orig;h=block.hidden
            names=['to_qkv_mlp_proj','to_out'] if block.single else ['to_q','to_k','to_v','add_q_proj','add_k_proj','add_v_proj','to_add_out']
            for name in names:
                layer=getattr(a,name)
                if not isinstance(layer,SliceLoRA):
                    setattr(a,name,SliceLoRA(layer,rank,alpha,rows=3*h if block.single and name=='to_qkv_mlp_proj' else None,
                                            columns=h if block.single and name=='to_out' else None))
            if not block.single and not isinstance(a.to_out[0],SliceLoRA):a.to_out[0]=SliceLoRA(a.to_out[0],rank,alpha)
    model.requires_grad_(False)
    for block in model.qdm_blocks:
        block.remote.requires_grad_(True);block.softmax_gate.requires_grad_(stage=='b')
    for module in model.modules():
        if isinstance(module,SliceLoRA):
            module.lora_A.requires_grad_(stage=='b');module.lora_B.requires_grad_(stage=='b')
    model.stage=stage
    return model


def delta_state_dict(model):
    ids={id(p) for b in model.qdm_blocks for m in (b.remote,b.softmax_gate) for p in m.parameters()}
    ids.update(id(p) for m in model.modules() if isinstance(m,SliceLoRA) for p in (m.lora_A,m.lora_B))
    return {n:p.detach() for n,p in model.named_parameters() if id(p) in ids}


def load_delta(model,state,strict=True):
    expected=delta_state_dict(model);missing=sorted(expected.keys()-state.keys());extra=sorted(state.keys()-expected.keys())
    if strict and (missing or extra):raise ValueError(f'Delta keys mismatch {missing=} {extra=}')
    with torch.no_grad():
        for name in expected.keys()&state.keys():
            if expected[name].shape!=state[name].shape:raise ValueError(f'Delta shape mismatch {name}')
            expected[name].copy_(state[name])
    return torch.nn.modules.module._IncompatibleKeys(missing,extra)


def optimizer_groups(model,stage):
    buckets={k:[] for k in ('big','small','lora')}
    small={id(p) for b in model.qdm_blocks for m in (b.remote.short_conv_k,b.remote.short_conv_v,b.remote.norm,b.remote.output_gate.up) for p in m.parameters()}
    delta=delta_state_dict(model)
    for name,p in model.named_parameters():
        if name in delta and p.requires_grad:buckets['lora' if 'lora_' in name else 'small' if id(p) in small else 'big'].append(p)
    mult={'big':.4 if stage=='b' else 1.,'small':2. if stage=='b' else 5.,'lora':1.}
    return [dict(params=p,group_name=k,lr=1e-4*mult[k],eps=2e-9 if k=='small' else 1e-8,weight_decay=0.) for k,p in buckets.items() if p]


def build_model(weights_path,stage,device,region_size=8,lora_rank=64,lora_alpha=64,sliding_kernel_size=11,**unused):
    base,info=Flux2Transformer2DModel.from_pretrained(str(weights_path),subfolder='transformer',
        torch_dtype=torch.bfloat16,local_files_only=True,output_loading_info=True)
    if any(info.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError(f'Native FLUX.2 checkpoint did not load strictly: {info}')
    base=base.to(device)
    if len(base.transformer_blocks)!=5 or len(base.single_transformer_blocks)!=20:raise ValueError('Expected Klein Base 4B layout (5+20)')
    model=Flux2QDM(base,sliding_kernel_size)
    # Diffusers checkpoint closures receive an immutable teacher_mode kwarg.
    model.base.enable_gradient_checkpointing()
    return configure_stage(model,stage,lora_rank,lora_alpha)
