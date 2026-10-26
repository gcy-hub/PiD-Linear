"""Resumable prompt-only DeCo sampling with a renewable multi-GPU controller."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from PIL import Image

from .repeat import (REPO, OWN_STOP, event, process_identity, read_json, run_cycle,
                     write_json)


def freeze_manifest(root, manifest):
    path = Path(root) / 'manifest.json'
    if path.exists() and read_json(path) != manifest:
        raise ValueError('Output directory has different prompts, weights or sampling settings')
    write_json(path, manifest)


def image_path(root, item):
    return Path(root) / 'generation/validation' / item['category'] / (item['id'] + '.png')


def settings_for(manifest, item):
    return {**manifest['settings'], **item}


def complete(root, manifest, item):
    path = image_path(root, item)
    try:
        record = read_json(path.with_suffix('.done.json'))
        if record.get('settings') != settings_for(manifest, item):
            return False
        with Image.open(path) as image:
            if image.size != (manifest['settings']['width'], manifest['settings']['height']):
                return False
            image.verify()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def pending_items(root, manifest, rank=0, world=1):
    if not 0 <= rank < world:
        raise ValueError('Invalid inference rank/world size')
    return [x for x in manifest['items'][rank::world] if not complete(root, manifest, x)]


def generate_pending(root, manifest, rank, world, render, should_stop):
    for item in pending_items(root, manifest, rank, world):
        if should_stop():
            break
        # The existing single-image sampler trusts its done marker. Remove a
        # marker for an invalid PNG so an interrupted/corrupt image is regenerated.
        image_path(root, item).with_suffix('.done.json').unlink(missing_ok=True)
        render(item)


def continuation(result, done, total, stalled):
    if result.get('reason') == 'manual-stop':
        return 'stopped'
    if result.get('fatal') or (result['returncode'] != 0 and not (result.get('forced') or result.get('killed'))):
        return 'failed'
    if done == total:
        return 'completed'
    if stalled >= 3:
        return 'failed'
    return 'restarting'


def sample_worker(root):
    import torch
    from tqdm import tqdm
    from .cache_text import QwenTextEncoder, atomic_tensor
    from .checkpoint import load_state
    from .data import load_text
    from .inference import _image_run
    from .model import build_model, load_delta_state_dict

    manifest = read_json(root / 'manifest.json')
    request = read_json(root / 'repeat/request.json')
    rank, world = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.set_num_threads(request['threads'])
    if not torch.cuda.is_available():
        raise RuntimeError('This native inference workload requires CUDA')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    started = time.monotonic()
    interrupted = {'value': False}
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, lambda *_: interrupted.update(value=True))

    def stopping():
        return (interrupted['value'] or (root / 'repeat/STOP').exists()
                or (root / 'generation/STOP').exists()
                or time.monotonic() - started >= request['cycle_seconds'])

    pending = pending_items(root, manifest, rank, world)
    if not pending or stopping():
        return
    cache = root / 'text_cache'
    todo = [(item['prompt'], cache / (item['id'] + '.pt')) for item in pending]
    # Empty text is encoded with the same native Qwen3 process as positive text.
    # Each rank owns its own empty embedding to avoid concurrent cache writes.
    empty_path = cache / f'empty_rank_{rank}.pt'
    todo.append(('', empty_path))
    missing = []
    for prompt, path in todo:
        try:
            load_text(path)
        except (OSError, ValueError, RuntimeError):
            missing.append((prompt, path))
    if missing:
        encoder = QwenTextEncoder(Path(manifest['settings']['weights_root']) / 'Qwen3-1.7B', device)
        batch_size = request['text_batch_size']
        for start in tqdm(range(0, len(missing), batch_size), desc=f'GPU {rank}: encode prompts', position=rank):
            if stopping():
                return
            batch = missing[start:start + batch_size]
            encoded = encoder.encode([prompt for prompt, _ in batch])
            for (_, path), tensor in zip(batch, encoded):
                atomic_tensor(path, tensor)
        del encoded, encoder
        torch.cuda.empty_cache()
    if stopping():
        return
    state = load_state(manifest['settings']['checkpoint'])
    architecture = state['metadata']['architecture']
    model = build_model(manifest['settings']['weights_root'], stage=state['stage'],
                        region_size=architecture['region_size'], halo=architecture['halo'], device=device,
                        full_attention_layers=architecture.get('full_attention_layers',[]))
    load_delta_state_dict(model, state['delta'])
    del state
    model.eval()
    empty = load_text(empty_path)[None].to(device)
    settings = manifest['settings']
    assigned = manifest['items'][rank::world]
    bar = tqdm(total=len(assigned), initial=len(assigned) - len(pending),
               desc=f'GPU {rank}: images', position=rank)

    def render(item):
        text = load_text(cache / (item['id'] + '.pt'))[None].to(device)
        noise = torch.randn((1, 3, settings['height'], settings['width']),
                            generator=torch.Generator().manual_seed(item['seed'])).to(device)
        destination = image_path(root, item)
        record = _image_run(model, noise, text, empty, destination,
                            settings_for(manifest, item), teacher=False)
        destination.with_suffix('.txt').write_text(item['prompt'] + '\n', encoding='utf-8')
        print(json.dumps(dict(event='image-complete', rank=rank, id=item['id'],
                              seconds=record['seconds'], path=str(destination))), flush=True)
        bar.update(1)
    try:
        generate_pending(root, manifest, rank, world, render, stopping)
    finally:
        bar.close()


def spawn_controller(root, fd):
    with (root / 'repeat/controller.log').open('a') as log:
        return subprocess.Popen([sys.executable, '-m', 'deco_qdm.batch_inference',
                                 '--controller', '--output-root', str(root), '--lease-fd', str(fd)],
                                cwd=REPO, pass_fds=(fd,), start_new_session=True,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)


def controller(root, fd):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    interrupted = {'value': False}
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: interrupted.update(value=True))
    with os.fdopen(fd, 'a') as lease:
        request = read_json(root / 'repeat/request.json')
        manifest = read_json(root / 'manifest.json')
        if (root / 'repeat/STOP').exists():
            write_json(root / 'repeat/state.json', dict(status='stopped'))
            return
        total = len(manifest['items'])
        before = total - len(pending_items(root, manifest))
        if before == total:
            write_json(root / 'repeat/state.json', dict(status='completed', done=total, total=total))
            return
        cycle = request.get('cycle', 0) + 1
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(request['gpu_ids'])
        os.environ['OMP_NUM_THREADS'] = str(request['threads'])
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   f"--nproc_per_node={len(request['gpu_ids'])}", '-m', 'deco_qdm.batch_inference',
                   '--worker', '--output-root', str(root)]
        result = run_cycle(root, 'generation', request, cycle, command=command,
                           interrupted=lambda: interrupted['value'])
        done = total - len(pending_items(root, manifest))
        stalled = 0 if done > before else request.get('stalled_cycles', 0) + 1
        status = continuation(result, done, total, stalled)
        write_json(root / 'repeat/state.json', dict(status=status, done=done, total=total,
                    cycle=cycle, controller_pid=os.getpid(), **result))
        event(root, status, done=done, total=total)
        if status == 'completed':
            write_json(root / 'finished.json', dict(done=done, total=total,
                       checkpoint=manifest['settings']['checkpoint']))
        if status != 'restarting':
            return
        request.update(cycle=cycle, stalled_cycles=stalled)
        write_json(root / 'repeat/request.json', request)
        time.sleep(2)
        if (root / 'repeat/STOP').exists() or interrupted['value']:
            write_json(root / 'repeat/state.json', dict(status='stopped', done=done, total=total))
            return
        successor = spawn_controller(root, lease.fileno())
        event(root, 'controller-handoff', successor_pid=successor.pid, cycle=cycle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--weights-root', default=os.path.expanduser('~/huggingface_ckpts/DeCo'))
    parser.add_argument('--prompts', help='JSON list: id, category, prompt, seed')
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--gpu-ids', default=os.getenv('GPU_IDS', '0,1,2,3'))
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--cfg', type=float, default=4.)
    parser.add_argument('--shift', type=float, default=3.)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--text-batch-size', type=int, default=8)
    parser.add_argument('--cycle-seconds', type=float, default=1200)
    parser.add_argument('--grace-seconds', type=float, default=180)
    parser.add_argument('--stop', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--controller', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--lease-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    if args.worker:
        sample_worker(root)
        return
    if args.controller:
        controller(root, args.lease_fd)
        return
    if args.stop:
        (root / 'repeat').mkdir(parents=True, exist_ok=True)
        (root / 'repeat/STOP').touch()
        print('Stop requested; completed images are retained and no successor is started')
        return
    if args.status:
        state = read_json(root / 'repeat/state.json')
        manifest = read_json(root / 'manifest.json')
        if manifest:
            state.update(done=len(manifest['items']) - len(pending_items(root, manifest)),
                         total=len(manifest['items']))
        print(json.dumps(state, indent=2))
        return
    if not args.checkpoint or not args.prompts:
        parser.error('--checkpoint and --prompts are required to start/resume')
    if (min(args.height, args.width) < 16 or args.height % 16 or args.width % 16
            or min(args.steps, args.threads, args.text_batch_size) < 1 or args.shift <= 0):
        parser.error('Invalid resolution, sampling steps, threads or text batch size')
    if not 0 < args.cycle_seconds or args.grace_seconds <= 0 or args.cycle_seconds + args.grace_seconds > 1680:
        parser.error('Cycle + grace must fit within 28 minutes')
    gpu_ids = [s.strip() for s in args.gpu_ids.split(',') if s.strip()]
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        parser.error('GPU IDs must be nonempty and unique')
    items = json.loads(Path(args.prompts).read_text())
    import re
    if not items or len({i['id'] for i in items}) != len(items):
        parser.error('Prompt IDs must be nonempty and unique')
    for item in items:
        if (not all(re.fullmatch(r'[A-Za-z0-9_-]+', item[k]) for k in ['id', 'category'])
                or not item['prompt'].strip() or not isinstance(item['seed'], int)):
            parser.error('Invalid prompt id/category, text or seed')
    from .checkpoint import load_state, fingerprint_file
    checkpoint = Path(args.checkpoint).resolve()
    state = load_state(checkpoint)
    base = fingerprint_file(Path(args.weights_root) / 't2i_DeCo.ckpt')
    if state['metadata']['base_fingerprint'] != base:
        raise ValueError('Base checkpoint differs from training')
    manifest = dict(settings=dict(checkpoint=str(checkpoint),
                    checkpoint_sha256=fingerprint_file(checkpoint / 'state.pt'),
                    stage=state['stage'], stage_step=state['progress']['stage_step'],
                    weights_root=str(Path(args.weights_root).resolve()), base_fingerprint=base,
                    height=args.height, width=args.width, steps=args.steps,
                    cfg=args.cfg, shift=args.shift, order=2), items=items)
    retained=state['metadata']['architecture'].get('full_attention_layers',[])
    if retained:manifest['settings']['full_attention_layers']=retained
    del state
    (root / 'repeat').mkdir(parents=True, exist_ok=True)
    with (root / 'repeat/controller.lock').open('a') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A generation controller already owns this output directory') from None
        old = read_json(root / 'repeat/state.json')
        identity = process_identity(old.get('training_pid', 0))
        if identity and identity[0] == old.get('training_start_ticks'):
            raise RuntimeError('A previous generation worker is still alive')
        freeze_manifest(root, manifest)
        (checkpoint / 'KEEP').touch()
        (root / 'repeat/STOP').unlink(missing_ok=True)
        stop = root / 'generation/STOP'
        if stop.exists() and stop.read_text().startswith(OWN_STOP):
            stop.unlink()
        write_json(root / 'repeat/request.json', dict(gpu_ids=gpu_ids, threads=args.threads,
                   text_batch_size=args.text_batch_size, cycle_seconds=args.cycle_seconds,
                   grace_seconds=args.grace_seconds, cycle=0, stalled_cycles=0))
        process = spawn_controller(root, lease.fileno())
    print(f'Started prompt generation controller PID {process.pid}; output: {root}', flush=True)
    print(f'Images: {root / "generation/validation"}', flush=True)
    print(f'Log: {root / "repeat/latest-training.log"}', flush=True)


if __name__ == '__main__':
    main()
