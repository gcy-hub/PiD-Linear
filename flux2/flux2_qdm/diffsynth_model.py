"""DiffSynth FLUX.2 + exact CLEAR square Sliding11; two training strategies.

No QDM, branch gate or separate output projection. Single-stream LoRA is
restricted to QKV rows and attention output columns, preserving frozen FFN.
CLEAR alternatively updates those full matrices using FP32 master weights.
The frozen native projections are also the full-attention teacher weights.
"""
import os
import sys
from contextlib import nullcontext
from pathlib import Path
import torch
from torch import nn
from .vendor.clear_attention import ClearAttention,AttentionLayout
from .model import SliceLoRA,disable_lora


class AttentionProjection(nn.Module):
    """Directly train a complete Attention matrix/slice, retaining its teacher.

    The original frozen projection is also the source of the unchanged FFN
    slice. Teacher and student never share a mutable trainable weight.
    """
    _is_projection_update=True

    def __init__(self,base,rows=None,columns=None):
        super().__init__();self.base=base;self.adapters_enabled=True
        self.rows=rows or base.out_features;self.columns=columns or base.in_features
        if rows is not None and columns is not None:raise ValueError('Only one projection axis may be partitioned')
        base.requires_grad_(False)
        # ZeRO2 in the reference updates FP32 master weights. Keep the same
        # precision here; only the graph's materialized weights are BF16.
        self.train_weight=nn.Parameter(base.weight[:self.rows,:self.columns].detach().float().clone())

    @property
    def weight(self):
        if not self.adapters_enabled:return self.base.weight
        weight=self.train_weight.to(self.base.weight.dtype)
        if self.rows<self.base.out_features:return torch.cat((weight,self.base.weight[self.rows:]),0)
        if self.columns<self.base.in_features:return torch.cat((weight,self.base.weight[:,self.columns:]),1)
        return weight

    @property
    def bias(self):return self.base.bias

    def forward(self,x):return torch.nn.functional.linear(x,self.weight,self.bias)


def import_framework(root=None):
    root=Path(root or os.getenv('FLUX2_DIFFSYNTH_ROOT',Path(__file__).resolve().parents[2]/'flux3/DiffSynth-Studio')).resolve()
    if not (root/'diffsynth/models/flux2_dit.py').is_file():raise FileNotFoundError(f'DiffSynth FLUX.2 source not found: {root}')
    sys.path.insert(0,str(root));os.environ['DIFFSYNTH_ATTENTION_IMPLEMENTATION']='torch'
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    os.environ.setdefault('DIFFSYNTH_SKIP_DOWNLOAD','True')
    return root


def load_pipeline(weights,device,components=('transformer',),framework=None):
    import_framework(framework)
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline,ModelConfig
    configs=[]
    for component in components:
        files=sorted((Path(weights)/component).glob('*.safetensors'))
        if not files:raise FileNotFoundError(f'{weights}/{component}')
        configs.append(ModelConfig(path=[str(p) for p in files],skip_download=True))
    pipe=Flux2ImagePipeline.from_pretrained(torch_dtype=torch.bfloat16,device=device,
        model_configs=configs,tokenizer_config=ModelConfig(path=str(Path(weights)/'tokenizer'),skip_download=True) if 'text_encoder' in components else None)
    return pipe.eval()


class LocalAttention(nn.Module):
    def __init__(self,orig,single,index,kernel_size,rank,alpha,training_strategy='lora-kd'):
        super().__init__();self.orig=orig;self.single=single;self.layer_index=index
        h=orig.inner_dim
        # Match CLEAR's trainable image projection scope; text projections stay frozen.
        projection=(lambda layer,**kw:AttentionProjection(layer,**kw)) if training_strategy=='clear' else (lambda layer,**kw:SliceLoRA(layer,rank,alpha,**kw))
        if single:
            orig.to_qkv_mlp_proj=projection(orig.to_qkv_mlp_proj,rows=3*h)
            orig.to_out=projection(orig.to_out,columns=h)
        else:
            for name in ('to_q','to_k','to_v'):setattr(orig,name,projection(getattr(orig,name)))
            orig.to_out[0]=projection(orig.to_out[0])
        self.engine=ClearAttention(window_size=16,down_factor=1,window_shape='square',kernel_size=kernel_size)

    def forward(self,hidden_states,encoder_hidden_states=None,image_rotary_emb=None,grid=None,
                text_length=None,teacher_mode=False,attention_mask=None,kv_cache=None,return_attention_feature=False,**unused):
        if attention_mask is not None or kv_cache is not None:raise ValueError('CLEAR local text-to-image only; extra masks/edit cache unsupported')
        a=self.orig
        if teacher_mode and not return_attention_feature:
            with disable_lora(a):
                return a(hidden_states,image_rotary_emb=image_rotary_emb) if self.single else a(hidden_states,encoder_hidden_states=encoder_hidden_states,image_rotary_emb=image_rotary_emb)
        from diffsynth.models.flux2_dit import apply_rotary_emb
        t=int(text_length) if self.single else encoder_hidden_states.shape[1]
        if grid is None or grid[0]*grid[1]!=hidden_states.shape[1]-(t if self.single else 0):raise ValueError('Explicit image grid mismatch')
        if self.single:
            with disable_lora(a) if teacher_mode else nullcontext():
                fused=a.to_qkv_mlp_proj(hidden_states)
            qkv,mlp=fused.split((3*a.inner_dim,a.mlp_hidden_dim*a.mlp_mult_factor),-1)
            q,k,v=[z.unflatten(-1,(a.heads,-1)) for z in qkv.chunk(3,-1)]
        else:
            q,k,v=[p(hidden_states).unflatten(-1,(a.heads,-1)) for p in (a.to_q,a.to_k,a.to_v)]
            tq,tk,tv=[p(encoder_hidden_states).unflatten(-1,(a.heads,-1)) for p in (a.add_q_proj,a.add_k_proj,a.add_v_proj)]
            tq,tk=a.norm_added_q(tq),a.norm_added_k(tk)
        q,k=a.norm_q(q),a.norm_k(k)
        if not self.single:q,k,v=[torch.cat(pair,1) for pair in ((tq,q),(tk,k),(tv,v))]
        if image_rotary_emb is not None:
            q=apply_rotary_emb(q,image_rotary_emb,sequence_dim=1);k=apply_rotary_emb(k,image_rotary_emb,sequence_dim=1)
        layout=AttentionLayout(((int(grid[0]),int(grid[1]),t),),text_first=True)
        if teacher_mode:
            out=torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))
        else:out=self.engine(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),layout,False)
        out=out.transpose(1,2).flatten(2,3)
        if not teacher_mode and hasattr(self, 'kda'):
            image_x = hidden_states[:,t:] if self.single else hidden_states
            out = torch.cat((out[:,:t], out[:,t:] + self.kda(image_x, grid)), 1)
        if self.single:
            with disable_lora(a) if teacher_mode else nullcontext():
                result=a.to_out(torch.cat((out,a.mlp_act_fn(mlp)),-1))
            return (result,out) if return_attention_feature else result
        text,image=out[:,:t],out[:,t:]
        return a.to_out[1](a.to_out[0](image)),a.to_add_out(text)


class DiffSynthLocal(nn.Module):
    _is_native_local=True
    def __init__(self,base,kernel_size=11,lora_rank=64,lora_alpha=64,training_strategy='lora-kd'):
        super().__init__();self.base=base;self.teacher_mode=False;self.stage='b';self.training_strategy=training_strategy
        if training_strategy not in ('lora-kd','clear'):raise ValueError('Unknown native training strategy')
        base.requires_grad_(False)
        offset=len(base.transformer_blocks)
        for i,b in enumerate([*base.transformer_blocks,*base.single_transformer_blocks]):
            b.attn=LocalAttention(b.attn,i>=offset,i,kernel_size,lora_rank,lora_alpha,training_strategy)

    @property
    def qdm_blocks(self):
        # Legacy trainer interface; these blocks contain no memory or gate.
        return tuple(m for m in self.modules() if isinstance(m,LocalAttention))

    def native_forward(self,grid,use_gradient_checkpointing=False,alignment_callback=None,collect_features=False,feature_indices=None,**kwargs):
        if alignment_callback is not None:raise ValueError('This reference-aligned experiment directly uses stage B')
        kwargs.pop('return_dict',None)
        options=dict(kwargs.pop('joint_attention_kwargs',None) or {},grid=tuple(grid),
                     text_length=kwargs['encoder_hidden_states'].shape[1],teacher_mode=self.teacher_mode)
        if collect_features:return self.feature_forward(kwargs,options,use_gradient_checkpointing,feature_indices)
        return (self.base(**kwargs,joint_attention_kwargs=options,use_gradient_checkpointing=use_gradient_checkpointing),)

    def feature_forward(self,args,options,checkpointing,feature_indices):
        """CLEAR's selected pre-O Attention features, returned through recompute."""
        from torch.utils.checkpoint import checkpoint
        base=self.base;x=args['hidden_states'];text=args['encoder_hidden_states'];t=text.shape[1]
        temb=base.time_guidance_embed(args['timestep'].to(x.dtype)*1000,None)
        im=base.double_stream_modulation_img(temb);tm=base.double_stream_modulation_txt(temb)
        sm=base.single_stream_modulation(temb)[0]
        x=base.x_embedder(x);text=base.context_embedder(text)
        ids=lambda z:z[0] if z.ndim==3 else z
        ir=base.pos_embed(ids(args['img_ids']));tr=base.pos_embed(ids(args['txt_ids']))
        rotary=tuple(torch.cat((tx,ix),0) for tx,ix in zip(tr,ir))
        def call(fn,**kw):return checkpoint(fn,use_reentrant=False,**kw) if checkpointing and torch.is_grad_enabled() else fn(**kw)
        for block in base.transformer_blocks:
            text,x=call(block,hidden_states=x,encoder_hidden_states=text,temb_mod_params_img=im,
                        temb_mod_params_txt=tm,image_rotary_emb=rotary,joint_attention_kwargs=options)
        x=torch.cat((text,x),1);features=[];offset=len(base.transformer_blocks)
        wanted=set(feature_indices if feature_indices is not None else [i for i in range(len(base.single_transformer_blocks)) if (offset+i)%4==0])
        for i,block in enumerate(base.single_transformer_blocks):
            if i in wanted:
                x,feature=call(single_with_feature,block=block,hidden_states=x,modulation=sm,rotary=rotary,options=options)
                features.append(feature)
            else:x=call(block,hidden_states=x,encoder_hidden_states=None,temb_mod_params=sm,image_rotary_emb=rotary,joint_attention_kwargs=options)
        return base.proj_out(base.norm_out(x[:,t:],temb)),features

    def forward(self,packed,use_gradient_checkpointing=False,alignment_callback=None,collect_features=False):
        first=packed.items[0]
        for sample in packed.items:
            if tuple(sample['grid'])!=tuple(first['grid']) or not torch.equal(sample['img_ids'],first['img_ids']) or not torch.equal(sample['txt_ids'],first['txt_ids']):
                raise ValueError('Native batched samples must share image grid and position IDs')
        batch={k:torch.cat([s[k] for s in packed.items],0) for k in ('hidden_states','encoder_hidden_states','timestep')}
        batch.update(grid=first['grid'],img_ids=first['img_ids'],txt_ids=first['txt_ids'])
        result=self.native_forward(**batch,use_gradient_checkpointing=use_gradient_checkpointing,alignment_callback=alignment_callback,collect_features=collect_features)
        return (result[0].flatten(0,1),result[1]) if collect_features else result[0].flatten(0,1)


def single_with_feature(block,hidden_states,modulation,rotary,options):
    shift,scale,gate=modulation
    norm=(1+scale)*block.norm(hidden_states)+shift
    out,feature=block.attn(norm,image_rotary_emb=rotary,return_attention_feature=True,**options)
    hidden_states=hidden_states+gate*out
    if hidden_states.dtype==torch.float16:hidden_states=hidden_states.clip(-65504,65504)
    return hidden_states,feature


def delta_state_dict(model):return {n:p.detach() for n,p in model.named_parameters() if ('train_weight' in n if model.training_strategy=='clear' else 'lora_' in n)}


def configure_stage(model,stage,*unused,**kwargs):
    if stage!='b':raise ValueError('DiffSynth pure Sliding experiment supports direct stage B only')
    model.requires_grad_(False)
    for n,p in model.named_parameters():
        if ('train_weight' in n if model.training_strategy=='clear' else 'lora_' in n):p.requires_grad_(True)
    return model


def build_model(weights,stage,device,kernel_size=11,rank=64,alpha=64,framework=None,training_strategy='lora-kd'):
    if stage!='b':raise ValueError('Native local-only experiment is direct B')
    pipe=load_pipeline(weights,device,framework=framework);base=pipe.dit
    if len(base.transformer_blocks)!=5 or len(base.single_transformer_blocks)!=20:raise ValueError('Expected Klein Base4B 5+20 layers')
    # The upstream FLUX.2 converter preserves native parameter names. Verify
    # every source key/shape rather than accepting a partially loaded model.
    from safetensors import safe_open
    expected=base.state_dict();seen=set()
    for file in sorted((Path(weights)/'transformer').glob('*.safetensors')):
        with safe_open(str(file),framework='pt') as handle:
            for key in handle.keys():
                if key not in expected or tuple(handle.get_slice(key).get_shape())!=tuple(expected[key].shape):raise ValueError(f'Native weight key/shape mismatch: {key}')
                seen.add(key)
    if seen!=set(expected):raise ValueError(f'Missing native source weights: {set(expected)-seen}')
    return configure_stage(DiffSynthLocal(base,kernel_size,rank,alpha,training_strategy),stage)
