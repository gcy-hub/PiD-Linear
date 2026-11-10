"""Native model lifecycle shared by the three external adapters."""
from contextlib import nullcontext
import importlib
import inspect
import json
from pathlib import Path
import torch
from torch import nn
from attention_framework.contracts import LayerSpec, ModelOutput, TrainingBatch
from .attention import AttentionSite, original_projections
from .vendor import native_module


def load_native(config, device):
    family = config.model.family
    if config.model.architecture != 'native':
        raise ValueError('Only native model architectures are supported')
    resource = config.model.resources.weights
    if resource is None:
        raise ValueError('model.resources.weights must name a local native checkpoint')
    path = Path(resource)
    if not path.exists():
        raise FileNotFoundError(path)
    if family == 'pid':
        from pid._src.common.runtime import build_net, load_original
        base = build_net()
        load_original(base, path)
    else:
        native = native_module(family)
        cls = native.Flux2DiT if family == 'flux2' else native.Flux3DiT
        kwargs = {}
        config_path = path / 'config.json' if path.is_dir() else path.parent / 'config.json'
        if config_path.is_file():
            raw = json.loads(config_path.read_text())
            kwargs = {k: v for k, v in raw.items() if k in inspect.signature(cls).parameters}
        with torch.device('meta'):
            base = cls(**kwargs)
        files = sorted(path.glob('*.safetensors')) if path.is_dir() else [path]
        if not files:
            raise FileNotFoundError(f'No native safetensors files in {path}')
        from safetensors.torch import load_file
        state = {}
        for file in files:
            shard = load_file(str(file))
            if state.keys() & shard.keys():
                raise ValueError('Duplicate native checkpoint keys')
            state.update(shard)
        base.load_state_dict(state, strict=True, assign=True)
    return base.to(device=device, dtype=getattr(torch, config.runtime.dtype))


class NativeModel(nn.Module):
    """The DDP-visible module also owns the adapter lifecycle."""
    def __init__(self, base, config, algorithm_factory):
        super().__init__()
        self.base, self.config = base, config
        self.family = config.model.family
        if self.family == 'pid':
            base.activation_checkpointing = config.runtime.gradient_checkpointing
        adapter = importlib.import_module('models.' + self.family)
        groups = adapter.layer_groups(base)
        counts = {name: len(blocks) for name, blocks in groups.items()}
        if counts != dict(config.model.layer_counts):
            raise ValueError(f'Native layer counts {counts} differ from configured {dict(config.model.layer_counts)}')
        self.layers = tuple(LayerSpec(**dict(s)) for s in config.layer_specs)
        adapter.install(base, self.layers, config, algorithm_factory)
        self.sites = tuple(m for m in base.modules() if isinstance(m, AttentionSite))

    @property
    def module(self):
        return self

    def role_parameters(self):
        roles = []
        paths = {id(m): n for n, m in self.named_modules()}
        for site in self.sites:
            for role in site.role_parameters():
                role.name = paths[id(site)] + '.' + role.name
                roles.append(role)
        return roles

    def forward(self, batch, *, teacher=False, capture=()):
        return self._forward(batch, teacher=teacher, capture=capture)

    def _forward(self, batch, *, teacher, capture=(), callback=None):
        if set(capture) - {layer.name for layer in self.layers}:
            raise ValueError('Feature capture must name selected layers')
        inputs = dict(batch.inputs)
        grids = inputs.pop('_grids')
        mask = inputs.pop('_text_mask', None)
        if self.family == 'flux2' and mask is not None and not mask.bool().all():
            outputs = []
            for i in range(len(batch.sample_ids)):
                row = {k: (v[i:i+1] if v.ndim != 2 or k not in ('img_ids', 'txt_ids') else v)
                       if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                valid = mask[i].bool()
                row['encoder_hidden_states'] = row['encoder_hidden_states'][:, valid]
                row['txt_ids'] = row['txt_ids'][:, valid] if row['txt_ids'].ndim == 3 else row['txt_ids'][valid]
                row['_grids'] = (grids[i],)
                sample = TrainingBatch(row, batch.target[i:i+1], batch.timesteps[i:i+1], (batch.sample_ids[i],))
                outputs.append(self._forward(sample, teacher=teacher, capture=capture, callback=callback))
            return ModelOutput(torch.cat([o.prediction for o in outputs]),
                               {name: torch.cat([o.features[name] for o in outputs]) for name in capture})

        if len(set(grids)) != 1 and self.family != 'flux3':
            raise ValueError('Native batched execution requires bucket-compatible image grids')
        previous = [(site.teacher, site.capture, site.callback) for site in self.sites]
        for site in self.sites:
            site.teacher, site.capture, site.callback = teacher, site.layer.name in capture, callback
            site.feature = None
            site.grid = grids[0]
            if self.family == 'flux2':
                site.text_length = inputs['encoder_hidden_states'].shape[1]
            elif self.family == 'flux3':
                packed = inputs['packed']
                site.samples = tuple((h, w, b-a+d-c) for (h, w), (a, b), (c, d) in
                                     zip(grids, packed.mode_indices['text'], packed.mode_indices['image_cond']))
        checkpoint = self.config.runtime.gradient_checkpointing and not teacher
        try:
            with original_projections(self.base) if teacher else nullcontext():
                with torch.no_grad() if teacher else nullcontext():
                    if self.family == 'pid':
                        prediction = self.base(**inputs)
                    else:
                        prediction = self.base(**inputs, use_gradient_checkpointing=checkpoint,
                                               use_gradient_checkpointing_offload=self.config.runtime.activation_cpu_offload and checkpoint)
            features = {s.layer.name: s.feature for s in self.sites if s.capture}
            return ModelOutput(prediction, features)
        finally:
            for site, state in zip(self.sites, previous):
                site.teacher, site.capture, site.callback = state
                site.student_inputs = None

    def layerwise(self, batch, callback):
        self._forward(batch, teacher=True, callback=callback)

    def prepare_batch(self, batch, *, generator):
        if batch.family != self.family:
            raise ValueError('Encoded batch belongs to a different model family')
        device = next(self.base.parameters()).device
        image = {k: v.to(device) for k, v in batch.image.items()}
        text = {k: v.to(device) for k, v in batch.text.items()}
        clean = image['pixels'] if self.family == 'pid' else image['latents']
        n = len(batch.sample_ids)
        noise = torch.randn(clean.shape, device=device, dtype=clean.dtype, generator=generator)
        t = torch.sigmoid(torch.randn(n, device=device, generator=generator))
        if self.family == 'pid':
            # Native PiD resolution-dependent flow shift and latent degradation.
            shift = 6.0 * ((clean.shape[-2]*clean.shape[-1])**0.5 / 2048)**0.5
            t = shift*t / (1+(shift-1)*t)
        s = t.reshape(n, *([1]*(clean.ndim-1)))
        noisy = ((1-s)*clean.float() + s*noise.float()).to(clean.dtype)
        target = noise.float()-clean.float()
        embeddings = text['embeddings']
        drop = torch.rand(n, device=device, generator=generator) < self.config.data.caption_dropout
        embeddings = embeddings * (~drop).reshape(n, *([1]*(embeddings.ndim-1)))
        if self.family == 'pid':
            condition = image['condition']
            sigma = torch.rand(n, device=device, generator=generator)*0.8
            cs = sigma.reshape(n, *([1]*(condition.ndim-1))).to(condition.dtype)
            condition = (1-cs)*condition + cs*torch.randn(condition.shape, device=device, dtype=condition.dtype, generator=generator)
            keep = torch.rand(n, device=device, generator=generator) >= self.config.data.image_dropout
            condition = condition * keep.reshape(n, *([1]*(condition.ndim-1)))
            inputs = dict(x=noisy.float(), t=t*1000, y=embeddings, lq_latent=condition,
                          degrade_sigma=sigma, text_valid_mask=text.get('mask'))
        elif self.family == 'flux2':
            inputs = dict(hidden_states=noisy, encoder_hidden_states=embeddings, timestep=t,
                          img_ids=image['image_ids'], txt_ids=text['text_ids'], _text_mask=text.get('mask'))
            # Native guidance-distilled FLUX2 expects its guidance embedding.
            if self.base.time_guidance_embed.guidance_embedder is not None:
                inputs['guidance'] = torch.full_like(t, 4.0)
        else:
            native = native_module('flux3')
            mask = text.get('mask', torch.ones(embeddings.shape[:2], device=device, dtype=torch.bool)).bool()
            text_chunks = [e[m] for e, m in zip(embeddings, mask)]
            id_chunks = [ids[m] for ids, m in zip(text['text_ids'], mask)]
            data = {'image': noisy.flatten(0, 1), 'text': torch.cat(text_chunks),
                    'image_cond': noisy.new_empty(0, noisy.shape[-1])}
            data_ids = image['image_ids']
            if data_ids.ndim == 2:
                data_ids = data_ids[None].expand(n, -1, -1)
            ids = {'image': data_ids.reshape(-1, 4), 'text': torch.cat(id_chunks),
                   'image_cond': data_ids.new_empty(0, 4)}
            lengths = {'image': [noisy.shape[1]]*n, 'text': [len(c) for c in text_chunks], 'image_cond': [0]*n}
            bounds, times = {}, {}
            for mode, sizes in lengths.items():
                offset = 0
                bounds[mode] = []
                for size in sizes:
                    bounds[mode].append((offset, offset+size))
                    offset += size
                times[mode] = torch.cat([t[i].expand(size) for i, size in enumerate(sizes)])
            inputs = {'packed': native.Packed(data, ids, times, bounds)}
            target = target.flatten(0, 1)
        inputs['_grids'] = batch.grids
        return TrainingBatch(inputs, target, t, batch.sample_ids)
