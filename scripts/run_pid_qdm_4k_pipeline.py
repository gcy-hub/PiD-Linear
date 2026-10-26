"""Resume downloads/preparation, then train full-data 4K PiD QDM A1 -> A2 -> B."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
CASE_SPEC = REPO / 'docs/pid-qdm-4k-cases.json'
SCHEDULE = {'a1': (1000, 500), 'a2': (2000, 1000), 'b': (20000, 2000)}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True, help='Full dataset destination, separate from the 50K export')
    p.add_argument('--subset-root', type=Path, help='Extracted pid-vdn-50k export to reuse without copying image bytes')
    p.add_argument('--weights-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, default=Path('outputs/pid-qdm-halo2-4k-8gpu'))
    p.add_argument('--gpu-ids', default=os.getenv('GPU_IDS', '0,1,2,3,4,5,6,7'))
    p.add_argument('--batch-size', type=int, choices=(4, 8), default=8)
    p.add_argument('--fallback-batch-size', type=int, choices=(4,), default=4)
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--download-workers', type=int, default=16)
    p.add_argument('--min-download-coverage', type=float, default=0.95,
                   help='Do not train on a mostly incomplete download; default minimum is 95%%')
    p.add_argument('--index-workers', type=int, default=8)
    p.add_argument('--text-batch-size', type=int, default=8)
    p.add_argument('--skip-model-download', action='store_true', help='Weights are already complete locally')
    p.add_argument('--dry-run', action='store_true', help='Print stages/commands without downloading or training')
    a = p.parse_args(argv)
    for name in ('dataset_root', 'weights_root', 'output_root', 'subset_root'):
        if getattr(a, name) is not None:
            setattr(a, name, getattr(a, name).expanduser().resolve())
    ids = a.gpu_ids.split(',')
    if len(ids) != len(set(ids)) or not ids or any(not x.strip() for x in ids):
        p.error('gpu-ids must contain unique GPU identifiers')
    if a.grad_accum < 1 or a.threads < 1 or a.workers < 0 or min(a.download_workers, a.index_workers, a.text_batch_size) < 1:
        p.error('Invalid batch/workers/threads')
    if not 0 < a.min_download_coverage <= 1:
        p.error('min-download-coverage must be in (0,1]')
    if a.dataset_root == a.subset_root:
        p.error('dataset-root must be a separate full-dataset directory')
    return a


def reuse_export(dataset_root, subset_root):
    """Reconnect stable, zero-based export IDs to original numbered JSON records."""
    dataset_root, subset_root = Path(dataset_root), Path(subset_root)
    groups = {}
    with (subset_root / 'manifest.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            stem, index = row['id'].split('/')
            groups.setdefault(stem, []).append((int(index), row))
    reused = missing = 0
    for stem, rows in tqdm(sorted(groups.items()), desc='Reuse exported image shards'):
        path = dataset_root / 'data_jsons' / (stem + '.json')
        records = json.loads(path.read_text())
        changed = False
        for index, row in rows:
            record = records[index]
            if row['source_url'] != record.get('image_url'):
                raise ValueError(f'Export/metadata URL mismatch for {row["id"]}')
            source = (subset_root / row['image_path']).resolve()
            # Do not allow an export entry to escape its declared image package.
            if not source.is_relative_to(subset_root.resolve()):
                raise ValueError(f'Export path escapes package: {row["id"]}')
            if not source.is_file() or source.stat().st_size == 0:
                missing += 1
                continue
            fid = f'{index + 1:0{len(str(len(records)))}d}'
            destination = dataset_root / 'datas' / stem / (fid + source.suffix.lower())
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                # Symlinks keep the existing 50K image bytes in one place. Keep
                # the exported package on disk for the lifetime of this dataset.
                destination.symlink_to(os.path.relpath(source, destination.parent))
            relative = os.path.relpath(destination, path.parent)
            if record.get('image_path') != relative:
                record['image_path'] = relative
                changed = True
            reused += 1
        if changed:
            atomic_json(path, records)
    return dict(reused=reused, missing_export_files=missing)


def environment(a):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=a.gpu_ids, GPU_IDS=a.gpu_ids, CPU_THREADS=str(a.threads),
               OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads),
               OPENBLAS_NUM_THREADS=str(a.threads), TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONPATH=str(REPO) + os.pathsep + env.get('PYTHONPATH', ''))
    env.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
    # Preserve the caller's proxy settings; downloads never modify the shell.
    return env


def run_logged(command, log, env):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f'Running: {" ".join(map(str, command))}\nLog: {log}', flush=True)
    with log.open('w') as stream:
        process = subprocess.Popen(list(map(str, command)), cwd=REPO, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
            raise


def distributed_command(a, module, *arguments):
    return [sys.executable, '-m', 'torch.distributed.run', '--standalone',
            f'--nproc_per_node={len(a.gpu_ids.split(","))}', '-m', module, *map(str, arguments)]


def training_command(a, stage, batch):
    steps, interval = SCHEDULE[stage]
    names = ','.join(f'{c["kind"]}_{c["ordinal"]:03d}' for c in json.loads(CASE_SPEC.read_text())['cases'])
    return distributed_command(a, 'scripts.train_pid_vdn',
        '--stage', stage, '--backbone', 'pid', '--resume', 'auto', '--allow-batch-change',
        '--weights-root', a.weights_root, '--index-root', a.dataset_root/'linear_pid_index',
        '--text-cache-root', a.dataset_root/'linear_pid_text_cache', '--training-data', 'full',
        '--output-root', a.output_root, '--resolution', '4096', '--allow-upscale',
        '--memory-mode', 'fixed4', '--local-mode', 'halo', '--local-halo', '2', '--region-size', '8',
        '--no-remote-rope', '--batch-size', batch, '--grad-accum', a.grad_accum,
        '--effective-batch', len(a.gpu_ids.split(','))*batch*a.grad_accum,
        '--attention-batch-chunk', '1', '--workers', a.workers, '--threads', a.threads,
        '--max-steps', steps, '--save-steps', interval, '--validation-steps', interval,
        '--gallery-root', a.output_root/'assets', '--validation-resolution', '4096',
        '--validation-cases', names, '--swanlab-mode', 'offline', '--log-steps', '1')


def is_cuda_oom(log):
    return bool(re.search(r'CUDA out of memory|CUDA error: out of memory', log, re.I))


def stage_complete(root, stage, steps):
    marker = Path(root) / stage / 'finished.json'
    if not marker.exists():
        return False
    from pid._src.vdn_pid.checkpoint import load_state
    state = load_state(json.loads(marker.read_text())['checkpoint'])
    return (state['stage'] == stage and state['progress']['stage_step'] >= steps
            and not state['progress'].get('validation_pending', False))


def train_stages(a):
    state_path = a.output_root / 'pipeline/training.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'batch_size': a.batch_size, 'fallbacks': []}
    for stage, (steps, _) in SCHEDULE.items():
        if stage_complete(a.output_root, stage, steps):
            print(f'{stage}: complete; preserving existing weights.', flush=True)
            continue
        while True:
            batch = state['batch_size']
            log = a.output_root / f'pipeline/logs/{stage}-bs{batch}-{time.time_ns()}.log'
            code = run_logged(training_command(a, stage, batch), log, environment(a))
            if code == 0:
                if not stage_complete(a.output_root, stage, steps):
                    raise RuntimeError(f'{stage} stopped before completion; rerun the same command to resume. See {log}')
                break
            tail = log.read_text(errors='replace') if log.exists() else ''
            if is_cuda_oom(tail) and batch > a.fallback_batch_size:
                state['batch_size'] = a.fallback_batch_size
                state['fallbacks'].append(dict(stage=stage, previous_batch=batch, batch=a.fallback_batch_size, log=str(log)))
                atomic_json(state_path, state)
                print(f'{stage}: CUDA OOM, resume at bs{a.fallback_batch_size}, accum remains {a.grad_accum}.', flush=True)
                continue
            raise RuntimeError(f'{stage} failed (exit {code}); no further batch reduction. Log {log}\n{tail[-2000:]}')
        atomic_json(state_path, state)


def prepare(a):
    root = a.output_root / 'pipeline'
    marker = root / 'preparation.json'
    done = json.loads(marker.read_text()) if marker.exists() else {}
    env = environment(a)
    online = {**env, 'HF_HUB_OFFLINE': '0', 'TRANSFORMERS_OFFLINE': '0'}

    def phase(name, command, outputs, phase_env=env):
        if done.get(name) and all(Path(p).exists() for p in outputs):
            print(f'Preparation {name}: complete.', flush=True)
            return
        code = run_logged(command, root/f'logs/{name}-{time.time_ns()}.log', phase_env)
        if code:
            raise RuntimeError(f'Preparation {name} failed with exit {code}; rerun to resume')
        if not all(Path(p).exists() for p in outputs):
            raise RuntimeError(f'Preparation {name} did not publish complete outputs')
        done[name] = True
        atomic_json(marker, done)

    if not a.skip_model_download:
        phase('models', [sys.executable, 'download/download_models.py', '--models', 'all',
                        '--weights-root', a.weights_root, '--workers', '4'],
              [a.weights_root/'PiD/checkpoints/ae.safetensors', a.weights_root/'gemma-2-2b-it',
               a.weights_root/'Z-Image-Turbo'], online)
    phase('metadata', [sys.executable, 'download/download_metadata.py', '--output-root', a.dataset_root],
          [a.dataset_root/'data_jsons/0001.json'], online)
    # Check the published metadata inventory before expensive GPU preparation.
    count = len(list((a.dataset_root/'data_jsons').glob('[0-9][0-9][0-9][0-9].json')))
    if count != 1008:
        raise ValueError(f'Expected all 1008 source JSONs, found {count}; training is not started')
    if a.subset_root and not done.get('reuse_export'):
        done['reuse_export'] = reuse_export(a.dataset_root, a.subset_root)
        atomic_json(marker, done)
    phase('images', [sys.executable, 'download/download_images.py', 'all',
                    '--json-dir', a.dataset_root/'data_jsons', '--datas-dir', a.dataset_root/'datas',
                    '--workers', a.download_workers, '--flush-every', '200', '--retries', '3'],
          [a.dataset_root/'datas'], online)
    # Persist full-inventory download coverage; dead source URLs are not silently
    # presented as one million usable images. Index rejection logs remain authoritative.
    if not done.get('download_coverage'):
        total = available = 0
        for path in tqdm(sorted((a.dataset_root/'data_jsons').glob('*.json')), desc='Download coverage'):
            for row in json.loads(path.read_text()):
                total += 1
                image = path.parent / row.get('image_path', '__missing__')
                available += image.is_file() and image.stat().st_size > 0
        done['download_coverage'] = dict(metadata_records=total, available_files=available, missing_files=total-available)
        atomic_json(marker, done)
        print(f'Download coverage: {done["download_coverage"]}; index validates usable samples.', flush=True)
    coverage = done['download_coverage']
    if coverage['available_files'] / max(1, coverage['metadata_records']) < a.min_download_coverage:
        done.pop('images', None)
        done.pop('download_coverage', None)
        atomic_json(marker, done)
        raise RuntimeError(f'Download incomplete: {coverage}. Rerun to retry missing source URLs; training has not started.')
    reserved = ','.join(c['sample_id'] for c in json.loads(CASE_SPEC.read_text())['cases'] if c['kind']=='real')
    phase('index', [sys.executable, 'scripts/prepare_linear_pid_data.py', '--dataset-root', a.dataset_root,
                    '--output-root', a.dataset_root/'linear_pid_index', '--workers', a.index_workers,
                    '--seed', '42', '--validation-size', '1024', '--validation-ids', reserved,
                    '--image-verification', 'header', '--resume'], [a.dataset_root/'linear_pid_index/index.json'])
    phase('text', distributed_command(a, 'scripts.prepare_linear_pid_text_cache',
              '--dataset-root', a.dataset_root, '--output-root', a.dataset_root/'linear_pid_text_cache',
              '--weights-root', a.weights_root, '--batch-size', a.text_batch_size, '--workers', a.workers,
              '--threads', a.threads, '--max-seconds', '0', '--resume'), [a.dataset_root/'linear_pid_text_cache/cache.json'])
    phase('assets', distributed_command(a, 'scripts.prepare_pid_qdm_portable_assets',
              '--weights-root', a.weights_root, '--index-root', a.dataset_root/'linear_pid_index',
              '--output-root', a.output_root/'assets', '--case-spec', CASE_SPEC, '--threads', a.threads),
          [a.output_root/'assets/assets.json'])


def main(argv=None):
    a = parse_args(argv)
    os.chdir(REPO)
    if a.dry_run:
        print(json.dumps({'dataset_root': str(a.dataset_root), 'subset_root': str(a.subset_root),
                         'stages': {s: training_command(a,s,a.batch_size) for s in SCHEDULE},
                         'oom_batch': a.fallback_batch_size}, default=str, indent=2))
        return
    a.output_root.mkdir(parents=True, exist_ok=True)
    with (a.output_root/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = dict(dataset_root=str(a.dataset_root), subset_root=str(a.subset_root) if a.subset_root else None,
                    weights_root=str(a.weights_root), schedule=SCHEDULE, grad_accum=a.grad_accum,
                    case_spec_sha256=hashlib.sha256(CASE_SPEC.read_bytes()).hexdigest())
        # Normalize tuples through JSON so resume comparisons are stable.
        spec = json.loads(json.dumps(spec))
        path = a.output_root/'pipeline/settings.json'
        if path.exists() and json.loads(path.read_text()) != spec:
            raise ValueError('Pipeline inputs changed; use the original paths/settings or a new output-root')
        atomic_json(path, spec)
        prepare(a)
        train_stages(a)
        print('A1, A2 and B complete.', flush=True)


if __name__ == '__main__':
    main()
