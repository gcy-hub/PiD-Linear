"""Four-GPU acceptance controller; missing mandatory evidence fails the run."""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Bindings are argv lists, not shell commands. Integration supplies resource-specific tests.
SCENARIOS = {
    'config_data': (1, 'cpu', 'Overrides/rejection, counts, buckets/EXIF/padding, rank+accumulation, paths, mocked download'),
    'convergence': (7, 'cpu', 'Imports, package/CLI discovery, pid/history unchanged, portable paths/defaults'),
    'handoff': (8, 'cpu', 'Due/interrupted/completed handoff, durable zzz model records and publication gate'),
    'pid': (2, 'parallel', 'Four algorithms on native PiD layers; real encoders and four cache modes (Verification 4/6)'),
    'flux2': (2, 'parallel', 'Four algorithms on native FLUX2 layers; real encoders and four cache modes (Verification 4/6)'),
    'flux3': (2, 'parallel', 'Four algorithms on native FLUX3 layers; real encoders and four cache modes (Verification 4/6)'),
    'common': (3, 'parallel', 'CUDA/reference/gradient, 21 strategies, taps/teacher isolation, token visibility/projections'),
    'distributed': (5, 'distributed', 'A1 average; A2/B/CLEAR DDP; accumulation/FP32; shard failure, RNG/cursor, validation isolation'),
    'pid_cycle': (6, 'serial', 'Real PiD train/save/resume/generate, stage interior/boundary and pending validation'),
    'flux2_cycle': (6, 'serial', 'Real FLUX2 train/save/resume/generate, stage interior/boundary and pending validation'),
    'flux3_cycle': (6, 'serial', 'Real FLUX3 train/save/resume/generate, stage interior/boundary and pending validation'),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_bindings(bindings):
    if not isinstance(bindings, dict) or set(bindings) - set(SCENARIOS):
        raise ValueError('Bindings must map known scenario IDs to argv lists')
    for name, item in bindings.items():
        if not isinstance(item, dict) or set(item) != {'argv', 'config', 'resources'}:
            raise ValueError(f'{name}: expected argv, config and resources')
        if not isinstance(item['argv'], list) or not item['argv'] or not all(isinstance(v, str) for v in item['argv']):
            raise ValueError(f'{name}: argv must be a nonempty string list')
        if not item['config'] or not item['resources']:
            raise ValueError(f'{name}: record actual configuration and resources')


def run_case(name, binding, report_dir, *, execute, gpu):
    item, phase, purpose = SCENARIOS[name]
    result = dict(verification=item, phase=phase, purpose=purpose, gpu=gpu, started=now(),
                  status='blocked', reason='Integration test binding missing')
    if binding is None:
        return result
    result.update(binding)
    if not execute:
        result['reason'] = 'Not executed: planning only'
        return result
    argv = binding['argv']
    if phase == 'distributed':
        # Bindings provide a Python test script or -m module, without a second launcher.
        argv = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4', *argv]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    env.update(ACCEPTANCE_SCENARIO=name, ACCEPTANCE_REPORT_DIR=str(report_dir / name))
    (report_dir / name).mkdir(exist_ok=True)
    result['command'] = argv
    try:
        with (report_dir / f'{name}.log').open('w') as log:
            completed = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        result.update(returncode=completed.returncode,
                      status='passed' if completed.returncode == 0 else 'blocked' if completed.returncode == 77 else 'failed',
                      reason='Required local resources or execution evidence unavailable; see scenario log' if completed.returncode == 77 else '')
    except OSError as exc:
        result['reason'] = str(exc)
    result['finished'] = now()
    return result


def summarize(results):
    # Every Verification item appears, even when downstream interfaces are unavailable.
    required = {
        '1': ('config_data',), '2': ('pid', 'flux2', 'flux3', 'common'),
        '3': ('common',), '4': ('pid', 'flux2', 'flux3'),
        '5': ('distributed', 'pid_cycle', 'flux2_cycle', 'flux3_cycle'),
        '6': ('pid', 'flux2', 'flux3', 'pid_cycle', 'flux2_cycle', 'flux3_cycle'),
        '7': ('convergence',), '8': ('handoff',),
    }
    groups = {key: [results.get(name, {}).get('status', 'blocked') for name in names]
              for key, names in required.items()}
    return {key: ('failed' if 'failed' in values else
                  'passed' if all(v == 'passed' for v in values) else 'blocked')
            for key, values in groups.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, help='JSON scenario -> {argv, config, resources}')
    parser.add_argument('--execute', action='store_true', help='Run CPU and GPU scenarios; requires the durable machine hold')
    args = parser.parse_args(argv)
    args.reports = args.reports.resolve()
    args.reports.mkdir(parents=True, exist_ok=True)
    bindings = json.loads(args.bindings.read_text()) if args.bindings else {}
    validate_bindings(bindings)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    report = dict(commit=commit, dirty=dirty, started=now(), physical_gpus=[0, 1, 2, 3],
                  bindings=bindings, results={}, h800_8gpu_4k_batch4='unverified')
    results = report['results']

    def save():
        report['verification'] = summarize(results)
        report['publish_allowed'] = len(results) == len(SCENARIOS) and not dirty and all(
            r['status'] == 'passed' for r in results.values())
        (args.reports / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')

    gpu_uuids = {}

    def run(name, gpu):
        visible = ','.join(gpu_uuids.get(index, index) for index in gpu.split(','))
        result = run_case(name, bindings.get(name), args.reports, execute=args.execute, gpu=visible)
        result['physical_gpus'] = gpu
        return result

    # Do not touch GPUs unless CPU and resource preflight have passed.
    for name, (_, phase, _) in SCENARIOS.items():
        if phase == 'cpu':
            results[name] = run(name, '')
            save()
    ready = args.execute and not dirty and all(r['status'] == 'passed' for r in results.values())
    if ready:
        try:
            inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'], text=True)
            processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
            devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
            gpu_uuids.update(line.strip().split(', ') for line in devices.splitlines()
                             if line.split(',')[0].strip() in ('0', '1', '2', '3'))
            selected = list(gpu_uuids.values())
            report['gpu_uuids'] = gpu_uuids
            report['gpu_inventory'] = inventory
            report['gpu_processes'] = processes
            if len(selected) != 4 or any(uuid in processes for uuid in selected):
                raise RuntimeError('Physical GPUs 0/1/2/3 unavailable or occupied; reschedule through the existing workflow')
        except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
            report['resource_blocker'] = str(exc)
            ready = False
    if ready:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {name: pool.submit(run, name, str(i)) for i, name in enumerate(('pid', 'flux2', 'flux3', 'common'))}
            for name, future in futures.items():
                results[name] = future.result()
                save()
        # All single-card children have exited before four-rank launch.
        for name in ('distributed', 'pid_cycle', 'flux2_cycle', 'flux3_cycle'):
            if all(r['status'] == 'passed' for r in results.values()):
                results[name] = run(name, '0,1,2,3')
            else:
                results[name] = dict(verification=SCENARIOS[name][0], status='blocked', reason='Prior mandatory scenario failed')
            save()
    for name, (item, phase, purpose) in SCENARIOS.items():
        results.setdefault(name, dict(verification=item, phase=phase, purpose=purpose, status='blocked',
                                     reason=report.get('resource_blocker', 'Awaiting execution or mandatory CPU checks')))
    report['finished'] = now()
    save()
    print(json.dumps(dict(commit=commit, verification=report['verification'], publish_allowed=report['publish_allowed'])))
    return 0 if report['publish_allowed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
