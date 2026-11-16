"""Native validation sampling; checkpoint restoration belongs to the caller."""
from pathlib import Path
import torch
from PIL import Image
from attention_framework.contracts import EncodedBatch, TrainingBatch
from .vendor import flow_scheduler, native_module


@torch.no_grad()
def _sample_batch(adapter, batch, *, vae, steps, seed):
    if steps < 1:
        raise ValueError('Sampling requires at least one step')
    if batch.family != adapter.family or len(batch.sample_ids) != 1:
        raise ValueError('Native validation sampling expects one matching-family sample')
    device = next(adapter.parameters()).device
    image = {k: v.to(device) for k, v in batch.image.items()}
    text = {k: v.to(device) for k, v in batch.text.items()}
    if adapter.family == 'pid':
        from pid._src.common.runtime import sample as pid_sample
        pixels, condition = image['pixels'], image['condition']
        def predict(x, t, y, **kwargs):
            inputs = dict(x=x, t=t, y=y, _grids=batch.grids, **kwargs)
            return adapter(TrainingBatch(inputs, x, t, batch.sample_ids)).prediction
        mask = text.get('mask', torch.ones(text['embeddings'].shape[:2], device=device, dtype=torch.bool))
        return pid_sample(predict, text['embeddings'], mask, text['embeddings'], mask,
                          condition, height=pixels.shape[-2], width=pixels.shape[-1],
                          seed=seed, steps=steps, cfg=1.0, sigma=0.0)
    h, w = batch.grids[0]
    clean = image['latents']
    # DiffSynth's native sampling default draws noise on CPU before moving it.
    latent = torch.randn(clean.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32).to(clean)
    scheduler = flow_scheduler(adapter.family)
    scheduler.set_timesteps(steps, **({'dynamic_shift_len': h*w} if adapter.family == 'flux2' else {}))
    if adapter.family == 'flux3':
        native = native_module('flux3')
        embeddings = text['embeddings'][0]
        ids = text['text_ids'][0]
        if 'mask' in text:
            valid = text['mask'][0].bool()
            embeddings, ids = embeddings[valid], ids[valid]
        data = {'image': latent[0], 'text': embeddings, 'image_cond': latent.new_empty(0, latent.shape[-1])}
        position_ids = {'image': image['image_ids'].reshape(-1, 4), 'text': ids,
                        'image_cond': ids.new_empty(0, 4)}
        packed = native.Packed(data, position_ids,
                               {m: latent.new_zeros(len(z)) for m, z in data.items()},
                               {m: [(0, len(z))] for m, z in data.items()})
    for index, timestep in enumerate(scheduler.timesteps):
        t = timestep.reshape(1).to(device=device, dtype=torch.float32) / 1000
        if adapter.family == 'flux2':
            inputs = dict(hidden_states=latent, encoder_hidden_states=text['embeddings'], timestep=t,
                          img_ids=image['image_ids'], txt_ids=text['text_ids'], _text_mask=text.get('mask'))
            if adapter.base.time_guidance_embed.guidance_embedder is not None:
                inputs['guidance'] = torch.full_like(t, 4.0)
        else:
            packed.data['image'] = latent[0]
            packed.timesteps['image'].fill_(t.item())
            inputs = {'packed': packed}
            if adapter.base.flow_map:
                next_sigma = scheduler.sigmas[index+1] if index+1 < len(scheduler.sigmas) else 0.0
                inputs['timesteps_ref'] = t.new_full((1,), float(next_sigma))
        inputs['_grids'] = batch.grids
        prediction = adapter(TrainingBatch(inputs, latent, t, batch.sample_ids)).prediction
        prediction = prediction.reshape_as(latent)
        latent = scheduler.step(prediction, timestep, latent)
    latent = latent.transpose(1, 2).reshape(1, -1, h, w)
    if adapter.family == 'flux2':
        from diffusers import Flux2KleinPipeline
        # Invert data.NativeEncoder's patchified VAE batch normalization.
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latent)
        std = (vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).sqrt().to(latent)
        latent = Flux2KleinPipeline._unpatchify_latents(latent*std + mean)
        return vae.decode(latent, return_dict=False)[0].clamp(-1, 1)
    return vae.decode(latent).clamp(-1, 1)


@torch.no_grad()
def sample(adapter, config, *, output_dir: Path, encoder=None, dataset=None,
           rank=0, world_size=1) -> list[Path]:
    """Write fixed validation samples, partitioned across independent ranks."""
    from data.dataset import ImageDataset
    from data.preprocess import load_pixels
    from data.encoders import build_encoder
    import json
    dataset = dataset or ImageDataset(config.validation.manifest or config.data.manifest or config.paths.dataset,
                                      config.validation.get('buckets') or config.data.buckets, config.data.min_size)
    count = config.validation.count
    if len(dataset) < count:
        raise ValueError(f'Validation requires {count} samples, found {len(dataset)}')
    device = next(adapter.parameters()).device
    encoder = encoder or build_encoder(config, device=device, image=True, text=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    was_training = adapter.training
    adapter.eval()
    paths = []
    try:
        for index in range(rank, count, world_size):
            path = output_dir / f'{index:02d}.png'
            metadata = path.with_suffix('.json')
            if path.exists() and metadata.exists():
                paths.append(path)
                continue
            row, size = dataset.samples[index], dataset.sizes[index]
            grid = (size[1]//16, size[0]//16)
            pixels = load_pixels(row.image_path, size)[None]
            batch = EncodedBatch((row.sample_id,), adapter.family,
                                 encoder.encode_image(pixels, (grid,)), encoder.encode_text([row.caption]), (grid,))
            seed = config.validation.seed + index
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []), \
                    torch.autocast(device.type, dtype=getattr(torch, config.runtime.dtype),
                                   enabled=config.runtime.dtype != 'float32'):
                generated = _sample_batch(adapter, batch, vae=encoder.vae, steps=config.validation.steps, seed=seed)
            rgb = ((generated[0].float().clamp(-1, 1)+1)*127.5).round().byte().permute(1, 2, 0).cpu().numpy()
            temporary = path.with_suffix('.partial')
            Image.fromarray(rgb).save(temporary, format='PNG')
            temporary.replace(path)
            temporary = metadata.with_suffix('.partial')
            temporary.write_text(json.dumps(dict(sample_id=row.sample_id, image_path=row.image_path,
                                                caption=row.caption, size=size, seed=seed,
                                                steps=config.validation.steps, cfg=1.0)))
            temporary.replace(metadata)
            paths.append(path)
    finally:
        adapter.train(was_training)
    return paths
