"""Local-only, independently loaded native image and text encoders."""
from pathlib import Path
import os
import sys
import torch


def _native_runtime():
    root = Path(__file__).resolve().parents[1] / 'models/vendor/DiffSynth-Studio'
    if not (root / 'diffsynth').is_dir():
        raise FileNotFoundError(f'Native DiffSynth runtime missing: {root}')
    sys.path.insert(0, str(root))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('DIFFSYNTH_SKIP_DOWNLOAD', 'True')


def _resource(config, name):
    value = config.model.resources[name]
    if value is None:
        raise ValueError(f'model.resources.{name} must name a local resource')
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)


def image_ids(h, w, device):
    return torch.cartesian_prod(torch.arange(1, device=device), torch.arange(h, device=device),
                               torch.arange(w, device=device), torch.arange(1, device=device)).float()


class NativeEncoder:
    def __init__(self, config, device, image, text):
        self.family, self.device = config.model.family, torch.device(device)
        self.dtype = getattr(torch, config.runtime.dtype)
        self.image_enabled, self.text_enabled = image, text
        if self.family == 'pid':
            if image:
                from pid._src.tokenizers.flux_vae import FluxVAE
                self.vae = FluxVAE(vae_pth=_resource(config, 'vae'), dtype=self.dtype, device=str(device), is_amp=False)
            if text:
                from pid._src.common.text_encoder import GemmaTextEncoder
                # The native helper expects the parent of gemma-2-2b-it.
                root = Path(_resource(config, 'text_encoder'))
                self.text = GemmaTextEncoder(root.parent, device)
        elif self.family == 'flux2':
            if image:
                from diffusers import AutoencoderKLFlux2
                self.vae = AutoencoderKLFlux2.from_pretrained(_resource(config, 'vae'), torch_dtype=self.dtype, local_files_only=True).to(device).eval().requires_grad_(False)
            if text:
                from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
                self.text = Qwen3ForCausalLM.from_pretrained(_resource(config, 'text_encoder'), torch_dtype=self.dtype, local_files_only=True).to(device).eval().requires_grad_(False)
                self.tokenizer = Qwen2TokenizerFast.from_pretrained(_resource(config, 'tokenizer'), local_files_only=True)
        elif self.family == 'flux3':
            if image or text:
                _native_runtime()
                from diffsynth.core import ModelConfig
                from diffsynth.pipelines.flux3_image import Flux3ImagePipeline
                for enabled, name in ((image, 'vae'), (text, 'text_encoder')):
                    if not enabled:
                        continue
                    path = Path(_resource(config, name))
                    files = sorted(str(p) for p in path.glob('*.safetensors')) if path.is_dir() else str(path)
                    if not files:
                        raise FileNotFoundError(f'No native weights: {path}')
                    spec = ModelConfig(path=files, computation_device=device, computation_dtype=self.dtype,
                                       preparing_device=device, preparing_dtype=self.dtype, skip_download=True)
                    pipe = Flux3ImagePipeline.from_pretrained(torch_dtype=self.dtype, device=device, model_configs=[spec],
                        processor_config=ModelConfig(path=_resource(config, 'tokenizer'), skip_download=True) if name == 'text_encoder' else None)
                    model = getattr(pipe, name)
                    if model is None:
                        raise RuntimeError(f'Native DiffSynth did not load {name}')
                    model.eval().requires_grad_(False)
                    if name == 'vae':
                        self.vae = model
                    else:
                        self.text = pipe
        else:
            raise ValueError(f'Unknown model family: {self.family}')

    @torch.no_grad()
    def encode_image(self, pixels, grids):
        if not self.image_enabled:
            raise RuntimeError('Image encoder was not requested')
        pixels = pixels.to(device=self.device, dtype=self.dtype)
        if self.family == 'pid':
            from pid._src.degradation import simple_downsample_image
            return {'pixels': pixels, 'condition': self.vae.encode(simple_downsample_image(pixels, 4.0))}
        if self.family == 'flux2':
            from diffusers import Flux2KleinPipeline
            z = Flux2KleinPipeline._patchify_latents(self.vae.encode(pixels).latent_dist.mode())
            mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(z)
            std = (self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).sqrt().to(z)
            z = (z - mean) / std
        else:
            z = self.vae.encode(pixels)
        h, w = z.shape[-2:]
        if any(grid != (h, w) for grid in grids):
            raise ValueError(f'Native VAE grid {(h, w)} differs from configured grids {grids}')
        return {'latents': z.flatten(2).transpose(1, 2),
                'image_ids': image_ids(h, w, self.device).unsqueeze(0).expand(len(pixels), -1, -1)}

    @torch.no_grad()
    def encode_text(self, captions):
        if not self.text_enabled:
            raise RuntimeError('Text encoder was not requested')
        if self.family == 'pid':
            embeddings, mask = self.text.encode_text(captions)
            return {'embeddings': embeddings, 'mask': mask}
        if self.family == 'flux2':
            from diffusers import Flux2KleinPipeline
            embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(self.text, self.tokenizer, captions, device=self.device, dtype=self.dtype)
            ids = torch.zeros(len(captions), embeds.shape[1], 4, device=self.device)
            ids[:, :, -1] = torch.arange(embeds.shape[1], device=self.device)
            return {'embeddings': embeds, 'text_ids': ids}
        from diffsynth.pipelines.flux3_image import Flux3Unit_PromptEmbedder
        # Native FLUX.3 has variable padded lengths; batching is handled by the loader.
        if len(captions) != 1:
            raise ValueError('Encode FLUX.3 captions individually to preserve native lengths')
        result = Flux3Unit_PromptEmbedder().process(self.text, captions[0])
        return {'embeddings': result['prompt_tokens'][None], 'text_ids': result['prompt_ids'][None]}


def build_encoder(config, *, device, image, text):
    return NativeEncoder(config, device, image, text)
