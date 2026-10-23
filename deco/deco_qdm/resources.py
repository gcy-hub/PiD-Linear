"""Resumable verified downloads for the two official DeCo T2I resources only."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm

from .data import atomic_json, file_sha256

DECO_REVISION = 'e8eec86604cee226c28fbf365d38f3e36bd77ee6'
DECO_SHA256 = '433f55684b19c446d9fad4591f840fcdf9770a668c383ea910da86362651492f'
DECO_SIZE = 4558758567
DECO_URL = f'https://hf-mirror.com/zehongma/DeCo/resolve/{DECO_REVISION}/t2i_DeCo.ckpt'
QWEN_REPO = 'Qwen/Qwen3-1.7B'


def no_proxy_env():
    """Return a child-only environment; never change the caller/Codex proxy."""
    return {key: value for key, value in os.environ.items()
            if key.lower() not in ('http_proxy', 'https_proxy', 'all_proxy')}


@lru_cache(maxsize=1)
def curl_supports_retry_all_errors():
    result = subprocess.run(['curl', '--help', 'all'], capture_output=True, text=True,
                            env=no_proxy_env(), check=True)
    return '--retry-all-errors' in result.stdout


def curl_download(url, partial):
    """Support old cluster curl, retry network failures and handle missing ranges."""
    base = ['curl', '--fail', '--location', '--connect-timeout', '30', '--progress-bar']
    # --retry-all-errors first appeared after the cluster's curl 7.68. Python
    # handles all errors below, while newer curl can also retry within a transfer.
    if curl_supports_retry_all_errors():
        base += ['--retry', '2', '--retry-all-errors', '--retry-delay', '3']
    resume = True
    for attempt in range(9):
        command = base + (['--continue-at', '-'] if resume else []) + ['--output', str(partial), url]
        result = subprocess.run(command, env=no_proxy_env(), check=False)
        if result.returncode == 0:
            return
        if result.returncode == 2:
            raise subprocess.CalledProcessError(result.returncode, command)
        if result.returncode == 33 and resume:
            print(f'Server rejected byte-range resume; restarting {partial}', flush=True)
            resume = False
            continue
        if attempt == 8:
            raise subprocess.CalledProcessError(result.returncode, command)
        # A failed full restart can resume the bytes it managed to download.
        resume = True
        time.sleep(min(3 * (attempt + 1), 30))


def modelscope_files(revision='master'):
    query = urllib.parse.urlencode({'Revision': revision, 'Recursive': 'true'})
    url = f'https://modelscope.cn/api/v1/models/{QWEN_REPO}/repo/files?{query}'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=60) as response:
        payload = json.load(response)
    if not payload.get('Success', payload.get('Code') == 200):
        raise RuntimeError(f'ModelScope file listing failed: {payload}')
    files = payload.get('Data', {}).get('Files', [])
    selected = []
    for item in files:
        path = item.get('Path') or item.get('Name')
        if not path or item.get('Type') in ('tree', 'directory'):
            continue
        # Download the complete inference model/tokenizer, without unrelated docs/media.
        if Path(path).suffix not in ('.json', '.safetensors', '.bin', '.model', '.txt', '.tiktoken'):
            continue
        if Path(path).is_absolute() or '..' in Path(path).parts:
            raise ValueError(f'Unsafe ModelScope path: {path}')
        sha256 = item.get('Sha256') or item.get('SHA256') or item.get('sha256')
        if sha256 and not re.fullmatch('[0-9a-fA-F]{64}', sha256):
            raise ValueError(f'Invalid remote SHA256 for {path}')
        selected.append({'path': path, 'size': item.get('Size'), 'sha256': sha256,
                         'url': f'https://modelscope.cn/models/{QWEN_REPO}/resolve/'
                                f'{urllib.parse.quote(revision, safe="")}/{urllib.parse.quote(path)}'})
    if not selected or not any(item['path'].endswith(('.safetensors', '.bin')) for item in selected):
        raise RuntimeError('ModelScope listing has no Qwen weights')
    return selected


def aria2_download(url, partial, connections):
    binary = shutil.which('aria2c')
    if binary is None:
        print('aria2c unavailable; using resumable curl', flush=True)
        curl_download(url, partial)
        return
    subprocess.run([binary, '--continue=true', '--auto-file-renaming=false',
                    '--allow-overwrite=true', '--file-allocation=none', '--max-tries=9',
                    '--retry-wait=3', '--connect-timeout=30', '--timeout=60',
                    '--summary-interval=10', '--console-log-level=warn',
                    f'--split={connections}', f'--max-connection-per-server={connections}',
                    '--min-split-size=16M', '--dir', str(partial.parent),
                    '--out', partial.name, url], env=no_proxy_env(), check=True)


def download_file(url, destination, expected_size=None, expected_sha256=None, connections=1):
    destination = Path(destination).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if expected_sha256 and not re.fullmatch('[0-9a-fA-F]{64}', expected_sha256):
        raise ValueError('SHA256 must have exactly 64 hex digits')

    def verify(path):
        if expected_size is not None and path.stat().st_size != int(expected_size):
            raise ValueError(f'Size mismatch for {path}: {path.stat().st_size} != {expected_size}')
        actual = file_sha256(path)
        if expected_sha256 and actual.lower() != expected_sha256.lower():
            raise ValueError(f'SHA256 mismatch for {path}: {actual}')
        return actual

    if destination.exists():
        return {'path': str(destination), 'size': destination.stat().st_size,
                'sha256': verify(destination), 'status': 'ready'}
    partial = destination.with_name(destination.name + '.incomplete')
    aria_control = partial.with_name(partial.name + '.aria2')
    # An interrupted transfer may already have all bytes. Verify before asking for
    # a range past EOF (some mirrors correctly return HTTP 416 in that case).
    if partial.exists() and not aria_control.exists() and expected_size is not None and partial.stat().st_size == int(expected_size):
        actual = verify(partial)
    else:
        if connections > 1 or aria_control.exists():
            if aria_control.exists() and shutil.which('aria2c') is None:
                raise RuntimeError(f'{partial} has a segmented aria2 resume map; install aria2c to resume safely')
            aria2_download(url, partial, connections)
        else:
            curl_download(url, partial)
        actual = verify(partial)
    os.replace(partial, destination)
    return {'path': str(destination), 'size': destination.stat().st_size,
            'sha256': actual, 'status': 'ready'}


def download_resources(weights_root, threads=4, only='all', qwen_revision='master', connections=1):
    if not 1 <= threads <= 4:
        raise ValueError('--threads must be between 1 and 4')
    if not 1 <= connections <= 4:
        raise ValueError('--connections must be between 1 and 4')
    root = Path(weights_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'resources.manifest.json'
    manifest = {'version': 1, 'status': 'downloading', 'deco_revision': DECO_REVISION,
                'qwen_repository': QWEN_REPO, 'qwen_revision': qwen_revision, 'files': {}}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        manifest['files'].update(old.get('files', {}))
    atomic_json(manifest_path, manifest)
    jobs = []
    if only in ('all', 'deco'):
        jobs.append(('t2i_DeCo.ckpt', DECO_URL, root / 't2i_DeCo.ckpt', DECO_SIZE, DECO_SHA256))
    if only in ('all', 'qwen'):
        files = modelscope_files(qwen_revision)
        manifest['qwen_remote_files'] = files
        jobs.extend((f'Qwen3-1.7B/{row["path"]}', row['url'], root / 'Qwen3-1.7B' / row['path'],
                     row['size'], row['sha256']) for row in files)
    atomic_json(manifest_path, manifest)
    lock = threading.Lock()

    def task(job):
        name, url, path, size, sha256 = job
        try:
            result = download_file(url, path, size, sha256,
                                   connections=connections if name == 't2i_DeCo.ckpt' else 1)
            result['source_url'] = url
        except Exception as error:
            with lock:
                manifest['files'][name] = {'status': 'failed', 'error': str(error), 'source_url': url}
                manifest['status'] = 'failed'
                atomic_json(manifest_path, manifest)
            raise
        with lock:
            manifest['files'][name] = result
            atomic_json(manifest_path, manifest)
        return result

    try:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(task, job) for job in jobs]
            for future in tqdm(as_completed(futures), total=len(futures), desc='Verified resource files'):
                future.result()
    except Exception:
        manifest['status'] = 'failed'
        atomic_json(manifest_path, manifest)
        raise
    manifest['status'] = 'ready'
    manifest['completed_selection'] = only
    atomic_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-root', type=Path, default=Path.home() / 'huggingface_ckpts/DeCo')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--only', choices=('all', 'deco', 'qwen'), default='all')
    parser.add_argument('--qwen-revision', default='master')
    parser.add_argument('--connections', type=int, default=1,
                        help='Connections per DeCo checkpoint transfer (1=curl; 2-4=optional aria2c)')
    args = parser.parse_args()
    print(json.dumps(download_resources(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
