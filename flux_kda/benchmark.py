"""Generate one original and one untrained KDA image with synchronized timing."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
from flux2_qdm.diffsynth_model import import_framework, load_pipeline
from .model import build_model, load_delta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=('flux2', 'flux3'), required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--resources', default=str(Path.home() / 'huggingface_ckpts'), help='FLUX3 text encoder/VAE root')
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--variant', choices=('both', 'original', 'kda'), default='both')
    p.add_argument('--prompt', default='A red double-decker bus on a rainy London street, reflections on wet asphalt')
    p.add_argument('--size', type=int, default=1024)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--cfg', type=float, default=3.5)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-condition-cache', action='store_true', help='Encode validation prompts without a disk cache')
    args = p.parse_args()
    if args.size % 16 or min(args.size, args.steps) < 1:
        p.error('Size must be a positive multiple of 16 and steps positive')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    import_framework()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    grid = (args.size // 16, args.size // 16)
    def cleanup():
        gc.collect()
        torch.cuda.empty_cache()
    def sync():
        torch.cuda.synchronize()
    def flux3_pipe(component):
        from diffsynth.core import ModelConfig
        from diffsynth.pipelines.flux3_image import Flux3ImagePipeline
        from flux3_qdm.resources import resource_paths
        paths = resource_paths(args.resources)
        return Flux3ImagePipeline.from_pretrained(device='cuda', torch_dtype=torch.bfloat16,
            model_configs=[ModelConfig(path=paths[component])],
            processor_config=ModelConfig(path=paths['processor']) if component == 'text_encoder' else None)
    conditions_file = root / 'conditions.pt'
    with torch.inference_mode():
        if conditions_file.exists() and not args.no_condition_cache:
            saved = torch.load(conditions_file, weights_only=True)
            if saved['prompt'] != args.prompt or saved['family'] != args.family:
                raise ValueError('Cached condition belongs to a different prompt/model')
            conditions = saved['conditions']
        else:
            if args.family == 'flux2':
                from diffsynth.pipelines.flux2_image import Flux2Unit_Qwen3PromptEmbedder
                pipe = load_pipeline(args.weights, 'cuda', ('text_encoder',))
                encode = Flux2Unit_Qwen3PromptEmbedder()
            else:
                from diffsynth.pipelines.flux3_image import Flux3Unit_PromptEmbedder
                pipe = flux3_pipe('text_encoder')
                encode = Flux3Unit_PromptEmbedder()
            conditions = [{k: v.cpu() for k, v in encode.process(pipe, prompt).items()} for prompt in (args.prompt, '')]
            if not args.no_condition_cache:
                torch.save(dict(prompt=args.prompt, family=args.family, conditions=conditions), conditions_file)
            del pipe, encode
            cleanup()
        conditions = [{k: v.cuda() for k, v in c.items()} for c in conditions]
        for variant in (('original', 'kda') if args.variant == 'both' else (args.variant,)):
            torch.manual_seed(args.seed)
            modified = variant == 'kda'
            checkpoint_step = None
            calibrated_mixing = False
            if modified:
                state = torch.load(args.checkpoint, map_location='cpu', weights_only=False) if args.checkpoint else None
                kernel = state.get('identity', {}).get('kernel', 11) if state is not None else 11
                model = build_model(args.family, args.weights, 'cuda', kernel=kernel)
                if state is not None:
                    load_delta(model, state['delta'])
                    checkpoint_step = state['step']
                    calibrated_mixing = bool(state.get('identity', {}).get('text_logit_bias'))
                    if state.get('identity', {}).get('text_logit_bias'):
                        from .mixing import apply_bias
                        apply_bias(model, state['identity']['text_logit_bias'])
                    del state
            elif args.family == 'flux2':
                pipe = load_pipeline(args.weights, 'cuda')
                model = pipe.dit
                del pipe
            else:
                from flux3_qdm.model import native
                from safetensors.torch import load_file
                with torch.device('meta'):
                    model = native.Flux3DiT()
                path = Path(args.weights)
                model.load_state_dict(load_file(str(path / 'model.safetensors' if path.is_dir() else path)), assign=True, strict=True)
                model.cuda()
            model.eval().requires_grad_(False)
            if modified:
                from .attention import ImageKDA
                for module in model.modules():
                    if isinstance(module, ImageKDA):
                        module.inference_weights()
                for name, param in model.named_parameters():
                    if name.endswith('train_weight'):
                        param.data = param.data.to(torch.bfloat16)
            noise = torch.randn((1,128,*grid), generator=torch.Generator('cpu').manual_seed(args.seed), dtype=torch.bfloat16)
            from diffsynth.diffusion import FlowMatchScheduler
            if args.family == 'flux2':
                from flux2_qdm.data import image_ids
                scheduler = FlowMatchScheduler('FLUX.2')
                scheduler.set_timesteps(args.steps, dynamic_shift_len=grid[0]*grid[1])
                z = noise.flatten(2).transpose(1,2).cuda()
                ids = image_ids(*grid).cuda()
                def predict(z, timestep, c):
                    kw = dict(hidden_states=z, encoder_hidden_states=c['prompt_embeds'], timestep=timestep.reshape(1).to(z)/1000,
                              img_ids=ids, txt_ids=c['text_ids'])
                    return model.native_forward(grid=grid, **kw)[0] if modified else model(**kw)
            else:
                from diffsynth.pipelines.flux3_image import Flux3ImagePipeline, Flux3Unit_PackedBuilder, Flux3Unit_ImageIDs
                pipe = Flux3ImagePipeline(device='cuda', torch_dtype=torch.bfloat16)
                scheduler = pipe.scheduler
                scheduler.set_timesteps(args.steps, alpha=6.93)
                z = noise.flatten(2).transpose(1,2)[0].cuda()
                ids = Flux3Unit_ImageIDs.latent_ids(*grid, 'cuda')
                packs = [Flux3Unit_PackedBuilder().process(pipe, **c, reference_tokens=None, reference_ids=None, latents=z, image_ids=ids)['packed'] for c in conditions]
                for pack in packs:
                    pack.image_grids = (grid,)
                del pipe
                def predict(z, timestep, c):
                    pack = packs[c]
                    pack.data['image'] = z
                    pack.timesteps['image'].fill_((timestep/1000).item())
                    return model(pack)
            inputs = conditions if args.family == 'flux2' else [0,1]
            sync(); cold = time.perf_counter()
            predict(z, scheduler.timesteps[0], inputs[0])
            sync(); warmup_seconds = time.perf_counter() - cold
            torch.cuda.reset_peak_memory_stats()
            sync(); tick = time.perf_counter()
            for step_index, timestep in enumerate(scheduler.timesteps):
                pos = predict(z, timestep, inputs[0])
                neg = predict(z, timestep, inputs[1]) if args.cfg != 1 else pos
                z = scheduler.step(neg + args.cfg*(pos-neg), timestep, z).to(torch.bfloat16)
                if (step_index + 1) % 10 == 0:
                    print(f'{args.family} {args.size}px {variant}: {step_index + 1}/{args.steps}', flush=True)
            sync(); denoise_seconds = time.perf_counter()-tick
            peak = torch.cuda.max_memory_allocated()
            if not torch.isfinite(z).all():
                raise FloatingPointError(f'Nonfinite {variant} latents')
            latent = z.reshape(-1,128).T.reshape(1,128,*grid).cpu()
            torch.save(dict(latent=latent, denoise_seconds=denoise_seconds, warmup_seconds=warmup_seconds,
                            peak_allocated_gib=peak/2**30), root / f'{variant}_latent.pt')
            del model, z, pos, neg
            cleanup()
            pipe = load_pipeline(args.weights, 'cuda', ('vae',)) if args.family == 'flux2' else flux3_pipe('vae')
            sync(); tick = time.perf_counter()
            decoded = pipe.vae.decode(latent.cuda())
            sync(); decode_seconds = time.perf_counter()-tick
            if not torch.isfinite(decoded).all():
                raise FloatingPointError(f'Nonfinite {variant} decoded image')
            image = pipe.vae_output_to_image(decoded)
            image.save(root / f'{variant}.png')
            record = dict(family=args.family, variant=variant, architecture='sliding-stock-fla-kda-v2' if modified else 'native', trained=bool(args.checkpoint) if modified else True,
                          checkpoint_step=checkpoint_step, calibrated_mixing=calibrated_mixing,
                          prompt=args.prompt, seed=args.seed, steps=args.steps, cfg=args.cfg, size=list(image.size),
                          denoise_seconds=denoise_seconds, decode_seconds=decode_seconds,
                          generation_seconds=denoise_seconds+decode_seconds, warmup_seconds=warmup_seconds,
                          peak_allocated_gib=peak/2**30, gpu=torch.cuda.get_device_name(), torch=str(torch.__version__))
            (root / f'{variant}.json').write_text(json.dumps(record, indent=2)+'\n')
            print(json.dumps(record), flush=True)
            del pipe, decoded
            cleanup()


if __name__ == '__main__':
    main()
