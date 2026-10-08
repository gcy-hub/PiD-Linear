"""Local-only PiD construction, shared frozen conditioning, and undistilled sampling."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch

from pid._src.linear_pid.attention import JointKDA, convert_attention
from pid._src.linear_pid.pit_attention import PatchKDA, convert_pit_attention
from pid._src.linear_pid.text_cache import TextCacheReader
from pid._src.linear_pid.text_encoder import NEGATIVE_PROMPT, PROMPT_PREFIX, GemmaTextEncoder
from pid._src.networks.pid_net import PidNet

NETWORK_KWARGS = dict(
    in_channels=3,
    num_groups=24,
    hidden_size=1536,
    pixel_hidden_size=16,
    pixel_attn_hidden_size=1152,
    pixel_num_groups=16,
    patch_depth=14,
    pixel_depth=2,
    patch_size=16,
    txt_embed_dim=2304,
    txt_max_length=300,
    use_text_rope=True,
    rope_mode="ntk_aware",
    rope_ref_h=2048,
    rope_ref_w=2048,
    repa_encoder_index=-1,
    lq_in_channels=0,
    lq_latent_channels=16,
    lq_hidden_dim=1024,
    lq_num_res_blocks=4,
    lq_conv_padding_mode="replicate",
    lq_aux_rgb_head=False,
    lq_gate_type="sigma_aware_per_token",
    lq_interval=2,
    zero_init_lq=True,
    train_lq_proj_only=False,
    sr_scale=4,
    pit_lq_inject=True,
)


def build_net(layers=(), local_mixing=True, backend="fla", kwargs=None, *, pit_kda=False, pit_kda_heads=64):
    net = PidNet(**(NETWORK_KWARGS if kwargs is None else kwargs))
    net.cache_repa_features = False
    net.pixel_embedder.compact_image_positions = True
    if layers:
        convert_attention(net, layers, local_mixing=local_mixing, backend=backend)
    else:
        net.kda_layers = []
    if pit_kda:
        convert_pit_attention(net, heads=pit_kda_heads, backend=backend)
    return net


def load_original(net, path):
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not all(key.startswith("net.") for key in state):
        raise ValueError("Expected the complete undistilled PiD net.* checkpoint")
    net.load_state_dict({key[4:]: value for key, value in state.items()}, strict=True)


def freeze_student(net):
    net.requires_grad_(True)
    net.lq_proj.requires_grad_(False)
    if net.pit_lq_gate is not None:
        net.pit_lq_gate.requires_grad_(False)
    net.patch_blocks[-1].freeze_unused_text_output_branch()
    if isinstance(net.patch_blocks[-1].attn, JointKDA):
        # Text Q/K/V and forget/write gates still condition the image recurrence.
        net.patch_blocks[-1].attn.gates_y.output.requires_grad_(False)
        net.patch_blocks[-1].attn.gates_y.norm_weight.requires_grad_(False)
        if net.patch_blocks[-1].attn.local_mixing:
            net.patch_blocks[-1].attn.text_conv[0].requires_grad_(False)
    else:
        net.patch_blocks[-1].attn.q_norm_y.requires_grad_(False)


def optimizer_groups(net, config):
    new_ids = {id(p) for module in net.modules() if isinstance(module, (JointKDA, PatchKDA)) for p in module.parameters()}
    buckets = {}
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        is_new = id(parameter) in new_ids
        decay = parameter.ndim >= 2 and not any(s in name for s in ["norm", "A_log", "dt_bias", "pos_embedding"])
        buckets.setdefault((is_new, decay), []).append(parameter)
    groups = [
        {
            "params": parameters,
            "lr": config.lr_new if new else config.lr_backbone,
            "weight_decay": config.weight_decay if decay else 0.0,
            "group_name": "kda" if new else "backbone",
        }
        for (new, decay), parameters in buckets.items()
    ]
    if not groups:
        raise ValueError("No trainable student parameters")
    return groups


class Conditioning:
    def __init__(self, weights_root, device="cuda", text_cache_root=None):
        from pid._src.tokenizers.flux_vae import FluxVAE

        root = Path(weights_root)
        self.device = torch.device(device)
        self.cache = TextCacheReader(text_cache_root) if text_cache_root else None
        self.encoder = None if self.cache else GemmaTextEncoder(weights_root, self.device)
        self.text = self.encoder.text if self.encoder else None
        self.tokenizer = self.encoder.tokenizer if self.encoder else None
        self.vae = FluxVAE(
            vae_pth=str(root / "PiD/checkpoints/ae.safetensors"),
            dtype=torch.bfloat16,
            device=str(self.device),
            is_amp=False,
        )
        self.prefix = PROMPT_PREFIX
        if self.cache:
            special = self.cache.special
            self.empty_embs, self.empty_mask = (
                special["empty_embs"].to(self.device),
                special["empty_mask"].to(self.device),
            )
            self.null_embs, self.null_mask = special["null_embs"].to(self.device), special["null_mask"].to(self.device)
        else:
            self.null_embs, self.null_mask = self.encode_text([NEGATIVE_PROMPT])

    @torch.no_grad()
    def encode_text(self, captions, *, cached_embs=None, cached_mask=None):
        if not self.cache:
            return self.encoder.encode_text(captions)
        if cached_embs is None or cached_mask is None:
            raise ValueError("Cached training requires embeddings and masks from the data loader")
        embs = cached_embs.to(self.device, non_blocking=True)
        mask = cached_mask.to(self.device, non_blocking=True)
        # Training dropout is the empty caption, distinct from CFG's negative prompt.
        drop = torch.tensor([caption == "" for caption in captions], device=self.device)
        embs = torch.where(drop[:, None, None], self.empty_embs, embs)
        mask = torch.where(drop[:, None], self.empty_mask, mask)
        return embs, mask

    @torch.no_grad()
    def encode_image(self, image):
        # Same frozen bicubic 4x downsampling as PiD's simple_downsample_image.
        from pid._src.degradation import simple_downsample_image

        lq = simple_downsample_image(image, 4.0)
        return self.vae.encode(lq), lq


@contextmanager
def evaluation_weights(net):
    """Evaluate the live student in place, without copies or device transfers."""
    was_training = net.training
    net.eval()
    try:
        yield
    finally:
        net.train(was_training)


@torch.no_grad()
def sample(
    net,
    caption_embs,
    caption_mask,
    null_embs,
    null_mask,
    latent,
    *,
    height,
    width,
    seed=42,
    steps=25,
    cfg=5.0,
    shift=6.0,
    sigma=0.0,
):
    from pid._src.modules.dpmsolver import DPMS

    device = latent.device
    b = latent.shape[0]
    noise = torch.randn(b, 3, height, width, device=device, generator=torch.Generator(device=device).manual_seed(seed))
    sigma = torch.full((b,), sigma, device=device, dtype=torch.float32)
    null_embs, null_mask = null_embs.expand(b, -1, -1), null_mask.expand(b, -1)

    def forward(x, timestep, y, **kwargs):
        # CFG is sequential so peak memory stays at batch one.
        def predict(xx, tt, yy, mask):
            return net(xx.float(), tt.float(), yy, lq_latent=latent, degrade_sigma=sigma, text_valid_mask=mask)

        if cfg == 1:
            return predict(x, timestep, y, caption_mask)
        half = x.shape[0] // 2
        return torch.cat(
            [
                predict(x[:half], timestep[:half], y[:half], null_mask),
                predict(x[half:], timestep[half:], y[half:], caption_mask),
            ]
        )

    # Training galleries retain FP32 optimizer states and DDP gradients. Avoid
    # caching a second BF16 copy of the whole FP32 student during sampling.
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        solver = DPMS(forward, caption_embs, null_embs, cfg_scale=cfg, model_type="flow", schedule="FLOW")
        return solver.sample(
            noise, steps=steps, order=min(steps, 2), skip_type="time_uniform_flow", method="multistep", flow_shift=shift
        ).clamp(-1, 1)
