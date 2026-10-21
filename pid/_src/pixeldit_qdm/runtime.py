"""Original PixelDiT-T2I: pixels + Gemma only; no VAE or latent conditioning."""
from pathlib import Path
import torch
from pid._src.networks.pixeldit_official import PixDiT_T2I
from pid._src.linear_pid.text_cache import TextCacheReader
from .conditioning import PixelDiTTextEncoder, load_null_cache

PIXELDIT_KWARGS = dict(
    in_channels=3, num_groups=24, hidden_size=1536,
    pixel_hidden_size=16, pixel_attn_hidden_size=1152, pixel_num_groups=16,
    patch_depth=14, pixel_depth=2, patch_size=16,
    txt_embed_dim=2304, txt_max_length=300, use_text_rope=True,
    rope_mode='original', rope_ref_h=1024, rope_ref_w=1024,
    repa_encoder_index=-1,
)


def checkpoint_path(root):
    root=Path(root)
    for relative in ['PixelDiT_finetune_2kto4k/model_ema_bf16.pth','model_ema_bf16.pth','model.safetensors']:
        if (root/relative).is_file():return root/relative
    candidates=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.pth','.pt','.safetensors') and '.cache' not in p.parts]
    if len(candidates)!=1:
        raise FileNotFoundError(f'Expected one complete PixelDiT checkpoint in {root}; found {len(candidates)}. Set --teacher-checkpoint explicitly.')
    return candidates[0]


def build_net(kwargs=None):
    net=PixDiT_T2I(**(PIXELDIT_KWARGS if kwargs is None else kwargs))
    net.cache_repa_features=False
    net.pixel_embedder.compact_image_positions=True
    return net


def load_original(net,path,*,assign=False):
    path=Path(path)
    if path.suffix=='.safetensors':
        from safetensors.torch import load_file
        state=load_file(str(path),device='cpu')
    else:state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if 'state_dict' in state and isinstance(state['state_dict'],dict):state=state['state_dict']
    if 'net' in state and isinstance(state['net'],dict):state=state['net']
    for prefix in ('module.','net.','_orig_mod.'):
        if state and all(k.startswith(prefix) for k in state):state={k[len(prefix):]:v for k,v in state.items()}
    if any(k.startswith('core.') for k in state):
        extra=[k for k in state if not k.startswith('core.')]
        if any(not k.startswith('_repa_projector.') for k in extra):
            raise ValueError(f'Unexpected non-core checkpoint tensors: {extra}')
        # Official training wrapper also exports its unused REPA projection.
        # The complete inference backbone itself is always loaded strictly.
        net.checkpoint_auxiliary_keys=tuple(extra)
        state={k[5:]:v for k,v in state.items() if k.startswith('core.')}
    if not state or not all(isinstance(v,torch.Tensor) for v in state.values()):
        raise ValueError('Expected native PixelDiT tensor state dictionary')
    net.load_state_dict(state,strict=True,assign=assign)


def load_net(path):
    # Every native parameter is supplied by the strictly checked checkpoint.
    # Avoid allocating and randomly filling 1.3B throwaway CPU parameters.
    with torch.device('meta'):
        net=build_net()
    load_original(net,path,assign=True)
    return net


class TextConditioning:
    image_conditioned=False
    def __init__(self,weights_root,device='cuda',text_cache_root=None):
        self.device=torch.device(device)
        self.cache=TextCacheReader(text_cache_root) if text_cache_root else None
        self.encoder=None if self.cache else PixelDiTTextEncoder(weights_root,self.device)
        if self.cache:
            special=load_null_cache(text_cache_root)
            self.empty_embs=special['empty_embs'].to(self.device)
            self.empty_mask=special['empty_mask'].to(self.device)
            self.null_embs=special['null_embs'].to(self.device)
            self.null_mask=special['null_mask'].to(self.device)
        else:
            from pid._src.linear_pid.text_encoder import NEGATIVE_PROMPT
            embeddings,masks=self.encoder.encode_raw(['',NEGATIVE_PROMPT])
            self.empty_embs,self.empty_mask=embeddings[:1],masks[:1]
            self.null_embs,self.null_mask=embeddings[1:],masks[1:]
    @torch.no_grad()
    def encode_text(self,captions,*,cached_embs=None,cached_mask=None):
        if self.encoder:embs,mask=self.encoder.encode_text(captions)
        else:
            if cached_embs is None or cached_mask is None:raise ValueError('Cached embeddings and masks required')
            embs,mask=cached_embs.to(self.device,non_blocking=True),cached_mask.to(self.device,non_blocking=True)
        drop=torch.tensor([c=='' for c in captions],device=self.device)
        return torch.where(drop[:,None,None],self.empty_embs,embs),torch.where(drop[:,None],self.empty_mask,mask)


def pixel_noise(batch,height,width,device,seed):
    if isinstance(seed,(list,tuple)):
        if len(seed)!=batch:
            raise ValueError('One noise seed is required per image')
        return torch.cat([torch.randn(1,3,height,width,device=device,
                          generator=torch.Generator(device=device).manual_seed(int(value))) for value in seed])
    return torch.randn(batch,3,height,width,device=device,
                       generator=torch.Generator(device=device).manual_seed(seed))


@torch.no_grad()
def sample(net,caption_embs,caption_mask,null_embs,null_mask,latent=None,*,height,width,seed=42,steps=50,cfg=2.75,shift=4.,**unused):
    from pid._src.modules.dpmsolver import DPMS
    if latent is not None:raise ValueError('PixelDiT-T2I inference must not receive an image latent')
    device=next(net.parameters()).device;b=caption_embs.shape[0]
    noise=pixel_noise(b,height,width,device,seed)
    null_embs,null_mask=null_embs.expand(b,-1,-1),null_mask.expand(b,-1)
    dtype=next(net.parameters()).dtype
    def predict(x,timestep,y,mask):
        return net(x.to(dtype),timestep.to(dtype),y.to(dtype),text_valid_mask=mask)
    def forward(x,timestep,y,**kwargs):
        if cfg==1:return predict(x,timestep,y,caption_mask)
        half=x.shape[0]//2
        return torch.cat([predict(x[:half],timestep[:half],y[:half],null_mask),
                          predict(x[half:],timestep[half:],y[half:],caption_mask)])
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        solver=DPMS(forward,caption_embs,null_embs,cfg_scale=cfg,model_type='flow',schedule='FLOW')
        return solver.sample(noise,steps=steps,order=min(steps,2),skip_type='time_uniform_flow',method='multistep',flow_shift=shift)
