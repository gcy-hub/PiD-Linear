"""Independent, phased, resumable student or ORIGINAL/QDM evaluation.

No text encoder, DiT and VAE coexist on the GPU. Conditions, initial noise,
denoising progress and native PNGs are persisted. Exit 75 means budget expired.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .compute import DIT_COMPUTE, dit_autocast

PROMPT_SEED = 271828
THEMES = (
    ('distant_faces', 'A documentary photograph of a busy outdoor cafe, full scene from across the street, many small distant human faces with distinct natural expressions, fine hair and clothing details, daylight.'),
    ('portrait', 'An intimate photographic close-up portrait of an elderly woman, natural skin pores, fine wrinkles, individual eyelashes, silver hair, soft window light, neutral expression.'),
    ('small_text', 'A realistic shopfront on a quiet street with a small clean sign reading "OPEN 24 HOURS", tiny menu labels and price tags visible through the window, natural daylight.'),
    ('bicycle', 'A detailed photograph of a vintage bicycle leaning against a plain wall, thin straight wheel spokes, coherent rims, chain links, gears and brake cables, side view.'),
    ('uniform_wall', 'A minimalist interior photograph with a large uniform matte pale gray wall, soft continuous illumination, a simple wooden chair in the lower right, smooth flat surfaces.'),
    ('fur', 'A macro wildlife photograph of a red fox resting on snow, sharply resolved individual fur strands, whiskers and eyes, soft winter sunlight.'),
    ('architecture', 'A straight-on architectural photograph of an ornate historic library facade, repeating windows, fine stone carvings, crisp brickwork and railings, physically consistent perspective.'),
    ('patterns', 'A studio photograph of woven tartan fabric and a fine black and white geometric patterned ceramic bowl, precise repeating lines and regular weave, soft even lighting.'),
    ('multiobject', 'A realistic tabletop photograph of five separate objects: a glass mug, a brass key, a red apple, a blue notebook and a small green toy car, distinct silhouettes, soft daylight.'),
    ('foliage', 'A detailed landscape photograph of a woodland stream, small leaves and twigs, delicate fern fronds, rippling clear water and pebbles, natural morning light.'),
)


def read_json(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def baseline_identity(settings):
    return fingerprint(dict(settings=settings,prompts=[list(item) for item in THEMES],
                            seeds=[PROMPT_SEED+index for index in range(len(THEMES))]))


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--weights-root')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--baseline-root', type=Path)
    p.add_argument('--stage-root', type=Path)
    p.add_argument('--stage', choices=('a1','a2','b'))
    p.add_argument('--gpu-ids', default=os.getenv('GPU_IDS','0'))
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--resolutions', default='1024,2048,4096')
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--cfg', type=float, default=3.5)
    p.add_argument('--alpha', type=float, default=6.93)
    p.add_argument('--max-seconds', type=float, default=1200)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--student-only', action='store_true', help='Generate only student images, without teacher sampling or comparisons')
    p.add_argument('--swanlab-mode',choices=('offline','disabled'),default='offline')
    p.add_argument('--vae-tiling', choices=('on','off'), default='off')
    p.add_argument('--worker-index', type=int, help=argparse.SUPPRESS)
    p.add_argument('--worker-count', type=int, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    args.resolution_values = [int(s) for s in args.resolutions.split(',')]
    if any(r <= 0 or r % 16 for r in args.resolution_values):
        p.error('Resolutions must be positive multiples of 16')
    if min(args.workers,args.threads,args.steps) < 1 or args.max_seconds <= 0:
        p.error('Workers, threads, steps and max-seconds must be positive')
    if not args.gpu_ids.strip() or any(not g.strip().isdigit() for g in args.gpu_ids.split(',')):
        p.error('--gpu-ids must contain comma-separated nonnegative GPU IDs')
    if len(set(args.gpu_ids.split(','))) != len(args.gpu_ids.split(',')):
        p.error('Duplicate GPU IDs are not allowed')
    return args


def checkpoint_config(args):
    checkpoint = args.checkpoint.resolve()
    from .repeat import checkpoint_complete
    if not checkpoint_complete(checkpoint):
        raise RuntimeError(f'Incomplete checkpoint: {checkpoint}')
    marker = read_json(checkpoint / 'complete.json')
    config = {}
    for parent in (checkpoint, *checkpoint.parents):
        if (parent / 'config.json').is_file():
            config = read_json(parent / 'config.json')
            break
    stage = args.stage or marker.get('stage') or config.get('stage')
    if stage not in ('a1','a2','b') or (marker.get('stage') and marker['stage'] != stage):
        raise RuntimeError('Checkpoint stage does not match evaluation stage')
    args.stage = stage
    args.weights_root = args.weights_root or config.get('weights_root')
    if not args.weights_root:
        raise ValueError('--weights-root is required when checkpoint config lacks it')
    return config


def task_records(args):
    return [dict(name=name, prompt=prompt, seed=PROMPT_SEED + index, resolution=res)
            for res in args.resolution_values for index,(name,prompt) in enumerate(THEMES)]


def task_id(task):
    return f"{task['resolution']}/{task['name']}_{task['seed']}"


def sampling_modes(args):
    return (False,) if getattr(args, 'student_only', False) else (True, False)


def done_path(args, task):
    directory = 'student' if getattr(args, 'student_only', False) else 'pairs'
    return args.output_root / directory / (task_id(task) + '.done.json')


def pair_images(original, student, pair, crop):
    from PIL import Image
    with Image.open(original) as a, Image.open(student) as b:
        if a.size != b.size:
            raise RuntimeError('Paired native image dimensions differ')
        width,height = a.size
        collage = Image.new('RGB',(width*2,height))
        collage.paste(a,(0,0)); collage.paste(b,(width,0))
        collage.save(pair)
        side = min(512,width,height)
        box = ((width-side)//2,(height-side)//2,(width+side)//2,(height+side)//2)
        details = Image.new('RGB',(side*2,side))
        details.paste(a.crop(box),(0,0)); details.paste(b.crop(box),(side,0))
        details.save(crop)
    return dict(crop_box=box, pair_size=[width*2,height], crop_size=[side*2,side])


def publish_task(args, task, images, timing, baseline_key, checkpoint_key):
    destination=done_path(args,task)
    destination.parent.mkdir(parents=True,exist_ok=True)
    marker=dict(task=task,student=str(images['student']),baseline_key=baseline_key,
                checkpoint_key=checkpoint_key,timing=timing)
    if getattr(args,'student_only',False):
        marker['student_only']=True
    else:
        pair=destination.with_name(destination.name.replace('.done.json','.png'))
        crop=destination.with_name(destination.name.replace('.done.json','.crop.png'))
        sizes=pair_images(images['original'],images['student'],pair,crop)
        if any(timing['original'][key] != timing['student'][key] for key in ('noise_sha256','input_fingerprint','cfg_schedule')):
            raise RuntimeError('ORIGINAL and QDM sampling inputs differ')
        marker.update(original=str(images['original']),**sizes)
    atomic_json(destination,marker)


def validate_compute_manifest(output_root):
    prior=read_json(Path(output_root)/'run.json')
    if prior and prior.get('settings',{}).get('dit_compute')!=DIT_COMPUTE:
        raise RuntimeError('Evaluation compute policy changed; select a new output root')


def pending_worker_tasks(args):
    validate_compute_manifest(args.output_root)
    # Keep task identities and rank ownership stable; only reorder execution.
    assigned = task_records(args)[args.worker_index::args.worker_count]
    pending = [task for task in assigned if not done_path(args, task).exists()]
    priority = {2048: 0, 1024: 1}
    return sorted(pending, key=lambda task: priority.get(task['resolution'], 2))


def worker(args):
    """Finish and publish one pair before a longer resolution can block it."""
    started = time.monotonic()
    for task in pending_worker_tasks(args):
        remaining = args.max_seconds - (time.monotonic() - started)
        if remaining <= 0 or (args.stage_root and (args.stage_root / 'STOP').exists()):
            return 75
        batch_args = argparse.Namespace(**{**vars(args), 'max_seconds': remaining})
        code = _worker_batch(batch_args, [task])
        if code:
            return code
        print(json.dumps(dict(event='student-complete' if getattr(args,'student_only',False) else 'pair-complete', task=task_id(task),
                              output=str(done_path(args, task).with_suffix('').with_suffix('.png')))),
              flush=True)
    return 0


def _worker_batch(args, tasks):
    # This environment is set before importing torch or DiffSynth.
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['MKL_NUM_THREADS'] = str(args.threads)
    import torch
    torch.set_num_threads(args.threads)
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.flux3_image import Flux3ImagePipeline, Flux3Unit_PromptEmbedder, Flux3Unit_ImageIDs, Flux3Unit_PackedBuilder
    from .resources import resource_paths, resource_manifest
    from .model import build_model, execution_mode, load_delta
    config = checkpoint_config(args)
    resources = resource_paths(args.weights_root)
    manifest = resource_manifest(args.weights_root)
    from .inference_identity import validate_checkpoint_identity
    checkpoint_identity = validate_checkpoint_identity(args.checkpoint,resources['dit'],config)
    settings = dict(steps=args.steps,cfg=args.cfg,alpha=args.alpha,negative_prompt='',
                    dtype='bfloat16',dit_compute=DIT_COMPUTE,vae_tiling=args.vae_tiling,decode_precision='bfloat16',torch_version=str(torch.__version__),
                    resource_fingerprints=manifest['fingerprints'],dit=file_identity(resources['dit']),
                    base_fingerprint=checkpoint_identity['base_fingerprint'],native_source_sha256=checkpoint_identity['native_source_sha256'])
    baseline_key = baseline_identity(settings)
    checkpoint_key = fingerprint(file_identity(args.checkpoint / 'delta.pt'))
    baseline_root = args.baseline_root / baseline_key
    started = time.monotonic()
    def progress(**value):
        atomic_json(args.output_root/f'progress_{args.worker_index}.json',value)
    def expired():
        return time.monotonic()-started >= args.max_seconds or (args.stage_root and (args.stage_root/'STOP').exists())
    def sync():
        torch.cuda.synchronize()
    def save_tensor(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
        torch.save(value,temp); os.replace(temp,path)
    def cpu(value):
        return {k:v.detach().cpu() for k,v in value.items()}
    def release(pipe):
        del pipe
        gc.collect(); torch.cuda.empty_cache()
    def pipe_for(kind):
        path = resources['text_encoder'] if kind == 'text_encoder' else resources[kind]
        pipe = Flux3ImagePipeline.from_pretrained(device='cuda',torch_dtype=torch.bfloat16,
            model_configs=[ModelConfig(path=path,computation_device='cuda',computation_dtype=torch.bfloat16,
                                       preparing_device='cuda',preparing_dtype=torch.bfloat16)],
            processor_config=ModelConfig(path=resources['processor']) if kind == 'text_encoder' else None)
        selected = getattr(pipe,kind)
        if selected is None or pipe.dit is not None or pipe.vram_management_enabled:
            raise RuntimeError(f'Evaluation requires only resident {kind}, with no CPU offload')
        selected.eval().requires_grad_(False)
        return pipe
    def condition_path(task):
        return baseline_root/'conditions'/(task['name']+'.pt')
    def sample_path(task,teacher):
        return (baseline_root if teacher else args.output_root/'student')/(task_id(task)+'.pt')
    def image_path(task,teacher):
        return sample_path(task,teacher).with_suffix('.png')
    # Reject cross-checkpoint resume rather than silently reusing student artifacts.
    run_manifest = args.output_root/'run.json'
    identity = dict(settings=settings,baseline_key=baseline_key,checkpoint_key=checkpoint_key,
                    prompts=[list(theme) for theme in THEMES],seeds=[t['seed'] for t in task_records(args)[:10]],resolutions=args.resolution_values)
    if getattr(args,'student_only',False):
        identity['student_only']=True
    prior = read_json(run_manifest)
    if prior and prior != identity:
        raise RuntimeError('Evaluation configuration/checkpoint changed; select a new output root')
    atomic_json(run_manifest,identity)
    tasks = [t for t in tasks if not done_path(args,t).exists()]
    if not tasks:
        return 0
    # Phase 1: native Qwen3-VL conditional and unconditional embeddings.
    pending = {t['name']:t for t in tasks if not condition_path(t).exists()}
    if pending:
        pipe = pipe_for('text_encoder')
        with torch.inference_mode():
            negative = cpu(Flux3Unit_PromptEmbedder().process(pipe,''))
            for task in pending.values():
                if expired():
                    del pipe; gc.collect(); torch.cuda.empty_cache(); return 75
                pos = cpu(Flux3Unit_PromptEmbedder().process(pipe,task['prompt']))
                save_tensor(condition_path(task),dict(positive=pos,negative=negative,prompt=task['prompt']))
                progress(phase='conditions',task=task_id(task))
        del pipe; gc.collect(); torch.cuda.empty_cache()
    if expired():
        return 75
    # Phase 2: one base DiT with mode context held over the complete trajectory.
    needs_dit = any(not sample_path(t,teacher).exists() or not torch.load(sample_path(t,teacher),map_location='cpu',weights_only=False).get('complete')
                    for t in tasks for teacher in sampling_modes(args))
    if needs_dit:
        pipe = Flux3ImagePipeline(device='cuda',torch_dtype=torch.bfloat16)
        pipe.dit = build_model(resources['dit'],args.stage,'cuda',
                               region_size=config.get('region_size',8),
                               lora_rank=config.get('lora_rank',64),lora_alpha=config.get('lora_alpha',64),
                               qdm_layers=config.get('qdm_layers'),halo=config.get('halo',0),
                               attention_mode=config.get('attention_mode','hybrid'),
                               memory_text_write=config.get('memory_text_write',False))
        load_delta(pipe.dit,torch.load(args.checkpoint/'delta.pt',map_location='cpu',weights_only=False),strict=True)
        pipe.dit.eval()
        with torch.inference_mode():
            for teacher in sampling_modes(args):
                for task in tasks:
                    path = sample_path(task,teacher)
                    saved = torch.load(path,map_location='cpu',weights_only=False) if path.exists() else {}
                    if saved.get('complete'):
                        continue
                    if expired():
                        del pipe; gc.collect(); torch.cuda.empty_cache(); return 75
                    res = task['resolution']
                    pipe.scheduler.set_timesteps(args.steps,denoising_strength=1.0,alpha=args.alpha)
                    cond = torch.load(condition_path(task),map_location='cpu',weights_only=False)
                    # Both trajectories start from exactly the same seeded CPU tensor.
                    noise = torch.randn((res//16*res//16,128),generator=torch.Generator(device='cpu').manual_seed(task['seed']),dtype=torch.bfloat16)
                    latents = saved.get('latents',noise).to('cuda')
                    image_ids = Flux3Unit_ImageIDs.latent_ids(res//16,res//16,'cuda')
                    packed = []
                    for name in ('positive','negative'):
                        c = {k:v.to('cuda') for k,v in cond[name].items()}
                        pack = Flux3Unit_PackedBuilder().process(pipe,**c,reference_tokens=None,reference_ids=None,latents=latents,image_ids=image_ids)['packed']
                        pack.image_grids = ((res//16,res//16),)
                        packed.append(pack)
                    elapsed = saved.get('model_seconds',0.)
                    with execution_mode(pipe.dit,teacher=teacher):
                        for step in range(saved.get('next_step',0),args.steps):
                            sync(); tick = time.monotonic()
                            timestep = pipe.scheduler.timesteps[step]
                            predictions = []
                            for pack in packed[:1 if args.cfg == 1 else 2]:
                                pack.data['image'] = latents
                                pack.timesteps['image'].fill_((timestep/1000).item())
                                with dit_autocast():
                                    predictions.append(pipe.dit(pack,use_gradient_checkpointing=False))
                            pred = predictions[0] if args.cfg == 1 else predictions[1]+args.cfg*(predictions[0]-predictions[1])
                            latents = pipe.scheduler.step(pred,timestep,latents)
                            if not bool(torch.isfinite(latents).all()):
                                raise FloatingPointError(f'nonfinite evaluation latents: {task_id(task)}, step {step}')
                            sync(); elapsed += time.monotonic()-tick
                            save_tensor(path,dict(latents=latents.cpu(),next_step=step+1,complete=step+1==args.steps,
                                model_seconds=elapsed,prompt=task['prompt'],seed=task['seed'],settings=settings,
                                noise_sha256=hashlib.sha256(noise.view(torch.uint8).numpy().tobytes()).hexdigest(),
                                input_fingerprint=fingerprint(dict(task=task,baseline_key=baseline_key)),
                                cfg_schedule=[args.cfg]*args.steps,teacher=teacher))
                            progress(phase='denoise',task=task_id(task),teacher=teacher,next_step=step+1)
                            if expired():
                                del pipe; gc.collect(); torch.cuda.empty_cache(); return 75
        del pipe; gc.collect(); torch.cuda.empty_cache()
    if expired():
        return 75
    # Phase 3: VAE only; fixed tiling and precision for ORIGINAL and QDM.
    pipe = pipe_for('vae')
    pipe.vae.use_tiling = args.vae_tiling == 'on'
    decode_config = {name:getattr(pipe.vae,name,None) for name in
                     ('use_tiling','tile_sample_min_size','tile_latent_min_size','tile_overlap_factor')}
    with torch.inference_mode():
        for task in tasks:
            for teacher in sampling_modes(args):
                image = image_path(task,teacher)
                if image.exists():
                    continue
                if expired():
                    del pipe; gc.collect(); torch.cuda.empty_cache(); return 75
                sample = torch.load(sample_path(task,teacher),map_location='cpu',weights_only=False)
                res=task['resolution']
                latent=sample['latents'].reshape(res//16,res//16,-1).permute(2,0,1)[None].to('cuda')
                sync(); tick=time.monotonic()
                decoded=pipe.vae.decode(latent)
                if not bool(torch.isfinite(decoded).all()):
                    raise FloatingPointError(f'nonfinite VAE output: {task_id(task)}')
                sync(); seconds=time.monotonic()-tick
                output=pipe.vae_output_to_image(decoded)
                temp=image.with_name(image.stem+f'.{os.getpid()}.tmp.png'); output.save(temp); os.replace(temp,image)
                atomic_json(image.with_suffix('.decode.json'),dict(vae_seconds=seconds,decode=decode_config,precision='bfloat16'))
                del decoded,latent
            timing = {}
            images = {}
            for teacher in sampling_modes(args):
                label='original' if teacher else 'student'
                saved = torch.load(sample_path(task,teacher),map_location='cpu',weights_only=False)
                timing[label] = {key:saved[key] for key in ('model_seconds','noise_sha256','input_fingerprint','cfg_schedule')}
                timing[label].update(read_json(image_path(task,teacher).with_suffix('.decode.json')))
                images[label]=image_path(task,teacher)
            publish_task(args,task,images,timing,baseline_key,checkpoint_key)
            progress(phase='student-complete' if getattr(args,'student_only',False) else 'paired',task=task_id(task))
    del pipe; gc.collect(); torch.cuda.empty_cache()
    return 0


def log_gallery(args):
    """Append images to the checkpoint's existing run; preserve its saved config."""
    if args.swanlab_mode == 'disabled' or (args.output_root/'swanlab.done.json').exists():
        return
    import torch
    state=torch.load(args.checkpoint/'state.pt',map_location='cpu',weights_only=False,mmap=True)
    config=state.get('config',{})
    run_id=state.get('metadata',{}).get('swanlab_run_id')
    if not run_id or config.get('swanlab_mode') == 'disabled':
        return
    import swanlab
    step=int(read_json(args.checkpoint/'complete.json')['step'])
    swanlab.init(project='FLUX3-QDM',id=run_id,resume='allow',mode='offline',
                 log_dir=str((args.stage_root or args.output_root)/'swanlog'),config=config)
    try:
        for task in task_records(args):
            done=done_path(args,task)
            label=f"validation/{task['resolution']}/{task['name']}"
            if getattr(args,'student_only',False):
                swanlab.log({label:swanlab.Image(read_json(done)['student'])},step=step)
            else:
                pair=done.with_name(done.name.replace('.done.json','.png'))
                crop=done.with_name(done.name.replace('.done.json','.crop.png'))
                swanlab.log({label:swanlab.Image(str(pair)),label+'_crop':swanlab.Image(str(crop))},step=step)
    finally:
        swanlab.finish()
    atomic_json(args.output_root/'swanlab.done.json',dict(run_id=run_id,step=step,mode='offline'))


def finalize_stage(args,marker):
    if args.stage_root:
        checkpoint_marker=read_json(args.checkpoint/'complete.json')
        step=int(checkpoint_marker['step'])
        atomic_json(args.stage_root/'evaluations'/f'step_{step:09d}.done.json',marker)
        def same_checkpoint(value):
            if not value.get('checkpoint'):
                return False
            path=Path(value['checkpoint'])
            if not path.is_absolute():
                path=args.stage_root/path
            return path.resolve()==args.checkpoint
        pending=read_json(args.stage_root/'validation_pending.json')
        final=read_json(args.stage_root/'training_finished.json')
        config=read_json(args.stage_root/'config.json')
        target=int(config.get('settings',{}).get('steps') or config.get('steps') or config.get('max_steps') or {'a1':500,'a2':1000,'b':10000}[checkpoint_marker['stage']])
        if same_checkpoint(final) and step>=target:
            atomic_json(args.stage_root/'finished.json',marker)
        if same_checkpoint(pending):
            (args.stage_root/'validation_pending.json').unlink(missing_ok=True)


def main(argv=None):
    args=parse_args(argv)
    args.checkpoint=args.checkpoint.resolve(); args.output_root=args.output_root.resolve()
    if args.stage_root:
        args.stage_root=args.stage_root.resolve()
    args.baseline_root=(args.baseline_root or ((args.stage_root.parent/'original-evaluation') if args.stage_root else args.output_root/'original')).resolve()
    checkpoint_config(args)
    validate_compute_manifest(args.output_root)
    if args.worker_index is not None:
        return worker(args)
    if (args.output_root/'run.json').exists() and not args.resume:
        raise RuntimeError('Evaluation output already exists; use --resume or a new output root')
    args.output_root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (args.output_root/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu_ids=[g.strip() for g in args.gpu_ids.split(',')]
        count=min(len(gpu_ids),args.workers)
        children=[]
        base=list(argv if argv is not None else sys.argv[1:])
        for i,gpu in enumerate(gpu_ids[:count]):
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':str(args.threads)}
            command=[sys.executable,'-m','flux3_qdm.evaluate',*base,'--worker-index',str(i),'--worker-count',str(count)]
            with (args.output_root/f'worker_{i}.log').open('a') as log:
                children.append(subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT))
        try:
            codes=[p.wait() for p in children]
        except BaseException:
            for process in children:
                if process.poll() is None:
                    process.terminate()
            for process in children:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            raise
        if any(code not in (0,75) for code in codes):
            raise RuntimeError(f'Evaluation worker failed {codes}; see {args.output_root}/worker_*.log. Decode settings were not changed.')
        complete=all(done_path(args,t).is_file() for t in task_records(args))
        if complete:
            marker=dict(checkpoint=str(args.checkpoint),stage=args.stage,evaluation=str(args.output_root),samples=len(task_records(args)))
            atomic_json(args.output_root/'complete.json',marker)
            log_gallery(args)
            finalize_stage(args,marker)
            print(json.dumps(dict(status='complete',**marker)),flush=True)
            return 0
        print(json.dumps(dict(status='pending',output_root=str(args.output_root))),flush=True)
        return 75


if __name__=='__main__':
    raise SystemExit(main())
