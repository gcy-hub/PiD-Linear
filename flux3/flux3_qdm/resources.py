"""Pinned, local-only FLUX.3 resources and explicit official companion download."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

DEFAULT_WEIGHTS_ROOT = os.environ.get('FLUX3_WEIGHTS_ROOT', str(Path.home() / 'huggingface_ckpts'))
COMPANION_REPO = 'black-forest-labs/flux-3-image-klein'
HF_ENDPOINT = 'https://huggingface.co'
MANIFEST_NAME = 'flux3_qdm_resources.json'


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def resource_paths(weights_root=DEFAULT_WEIGHTS_ROOT):
    root = Path(weights_root).expanduser().resolve()
    companion = root / 'flux-3-image-klein'
    text = sorted(str(p) for p in (companion / 'text_encoder').glob('*.safetensors'))
    return dict(dit=str(root / 'flux-3-image-klein-base' / 'model.safetensors'),
                text_encoder=text, processor=str(companion / 'text_encoder'),
                vae=str(companion / 'vae' / 'diffusion_pytorch_model.safetensors'))


def resource_manifest(weights_root=DEFAULT_WEIGHTS_ROOT, verify_hashes=False):
    """Reject absent or changed assets; normal reads use SHA-verified file stat records."""
    root = Path(weights_root).expanduser().resolve()
    path = root / 'flux-3-image-klein' / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f'Missing pinned companion resource manifest: {path}; run download_flux3_qdm.sh')
    manifest = json.loads(path.read_text())
    if manifest.get('repository') != COMPANION_REPO or manifest.get('endpoint') != HF_ENDPOINT or not manifest.get('revision'):
        raise RuntimeError('Resource manifest does not describe a pinned official FLUX.3 companion snapshot')
    required = {'vae/diffusion_pytorch_model.safetensors', 'text_encoder/config.json',
                'text_encoder/tokenizer_config.json', 'text_encoder/tokenizer.json',
                'text_encoder/preprocessor_config.json', 'text_encoder/chat_template.json',
                'text_encoder/model.safetensors.index.json'}
    if not required <= set(manifest.get('files', {})):
        raise RuntimeError('Pinned resource manifest is incomplete; rerun resource download')
    for name, item in manifest['files'].items():
        asset = path.parent / name
        if not asset.is_file():
            raise FileNotFoundError(f'Missing resource {asset}; rerun download_flux3_qdm.sh')
        stat = asset.stat()
        if stat.st_size != item['size'] or stat.st_mtime_ns != item['mtime_ns']:
            raise RuntimeError(f'Changed resource {asset}; rerun resource download/verification')
        if verify_hashes and sha256_file(asset) != item['sha256']:
            raise RuntimeError(f'SHA256 mismatch for resource {asset}')
    paths = resource_paths(root)
    if not paths['text_encoder'] or not Path(paths['vae']).is_file() or not Path(paths['dit']).is_file():
        raise FileNotFoundError('FLUX.3 base DiT or companion TE/VAE assets are missing')
    # Cache identity depends on exact verified model/tokenizer bytes, not download time.
    manifest['fingerprints'] = {
        kind: hashlib.sha256(json.dumps({name: entry['sha256'] for name, entry in manifest['files'].items()
                                       if name.startswith(prefix)}, sort_keys=True).encode()).hexdigest()
        for kind, prefix in [('text', 'text_encoder/'), ('image', 'vae/')]
    }
    return manifest


def _modelscope_exact_download(repo, filename, expected_sha, size, target, range_workers=8):
    """Official publisher replica, exact HF identity, scoped direct HTTP, durable range chunks."""
    import requests
    import shutil
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    from urllib.parse import quote
    session = requests.Session()
    session.trust_env = False  # ModelScope requests only; never modify process/global proxy state.
    response = session.get(f'https://modelscope.cn/api/v1/models/{repo}/repo/files',
                           params={'Revision': 'master', 'Recursive': 'true'}, timeout=30)
    response.raise_for_status()
    listing = response.json()['Data']['Files']
    remote = next((f for f in listing if f['Path'] == filename), None)
    if remote is None or remote['Sha256'] != expected_sha or remote['Size'] != size:
        raise RuntimeError(f'Official ModelScope replica is not byte-identical: {repo}/{filename}')
    revision = remote['Revision']
    target = Path(target)
    origin = dict(endpoint='https://modelscope.cn', repository=repo, revision=revision,
                  filename=filename, sha256=expected_sha)
    if target.is_file() and target.stat().st_size == size and sha256_file(target) == expected_sha:
        return origin
    parts = target.parent / '.download_chunks' / expected_sha
    parts.mkdir(parents=True, exist_ok=True)
    chunk_size = 16 << 20
    total = (size + chunk_size - 1) // chunk_size
    def valid(index):
        part = parts / f'{index:06d}.part'
        expected_size = min(chunk_size, size - index * chunk_size)
        try:
            meta = json.loads(part.with_suffix('.json').read_text())
            return part.stat().st_size == expected_size and sha256_file(part) == meta['sha256']
        except (OSError, ValueError, KeyError):
            return False
    pending = [i for i in range(total) if not valid(i)]
    url = f'https://modelscope.cn/models/{repo}/resolve/{revision}/{quote(filename, safe="/")}'
    def fetch(index):
        import time
        start = index * chunk_size
        end = min(size, start + chunk_size) - 1
        part = parts / f'{index:06d}.part'
        for attempt in range(5):
            temp = None
            try:
                local = requests.Session()
                local.trust_env = False
                fd, temp = tempfile.mkstemp(prefix=f'.{index:06d}.', suffix='.partial', dir=parts)
                with os.fdopen(fd, 'wb') as out:
                    with local.get(url, headers={'Range': f'bytes={start}-{end}'}, stream=True,
                                   timeout=(20, 90)) as result:
                        result.raise_for_status()
                        if result.status_code != 206 or result.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                            raise RuntimeError('Server did not honor exact byte range')
                        digest = hashlib.sha256()
                        count = 0
                        for block in result.iter_content(1 << 20):
                            count += len(block)
                            digest.update(block)
                            out.write(block)
                        if count != end - start + 1:
                            raise RuntimeError('Truncated byte range')
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temp, part)
                _atomic_json(part.with_suffix('.json'), dict(size=count, sha256=digest.hexdigest()))
                return count
            except Exception:
                if temp and os.path.exists(temp):
                    os.unlink(temp)
                if attempt == 4:
                    raise RuntimeError(f'Official ModelScope range transfer failed for {filename} chunk {index}') from None
                time.sleep(min(2 ** attempt, 8))
    initial = size - sum(min(chunk_size, size - i * chunk_size) for i in pending)
    with tqdm(total=size, initial=initial, unit='B', unit_scale=True, desc=filename, mininterval=2) as progress:
        with ThreadPoolExecutor(max_workers=range_workers) as pool:
            for future in as_completed([pool.submit(fetch, i) for i in pending]):
                progress.update(future.result())
    fd, temporary = tempfile.mkstemp(prefix=f'.{target.name}.', suffix='.partial', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            for index in range(total):
                with (parts / f'{index:06d}.part').open('rb', buffering=1 << 20) as source:
                    shutil.copyfileobj(source, out, length=8 << 20)
            out.flush()
            os.fsync(out.fileno())
        if sha256_file(temporary) != expected_sha:
            raise RuntimeError(f'Full official SHA256 mismatch after range assembly: {filename}')
        os.replace(temporary, target)
        # Keep no second complete checkpoint copy after successful verification.
        shutil.rmtree(parts)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f'SHA256 verified: {target}', flush=True)
    return origin


def download(weights_root=DEFAULT_WEIGHTS_ROOT, revision=None, workers=4, exact_upstream_fallback=False, exact_modelscope_fallback=False, range_workers=8):
    """Download only official TE/VAE; never fetch or replace the local base DiT."""
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download
    root = Path(weights_root).expanduser().resolve()
    paths = resource_paths(root)
    if not Path(paths['dit']).is_file():
        raise FileNotFoundError(f'Expected existing local base DiT: {paths["dit"]}')
    companion = root / 'flux-3-image-klein'
    old_manifest = companion / MANIFEST_NAME
    if revision is None and old_manifest.is_file():
        revision = json.loads(old_manifest.read_text())['revision']
    api = HfApi(endpoint=HF_ENDPOINT)
    info = api.model_info(COMPANION_REPO, revision=revision or 'main', files_metadata=True)
    selected = [s for s in info.siblings if s.rfilename.startswith(('text_encoder/', 'vae/'))]
    if not selected:
        raise RuntimeError('Official repository returned no companion assets')
    provenance = {}
    omitted = []
    if exact_upstream_fallback:
        # Runtime uses native model constructors, so VAE config and documentation are not read.
        # Never substitute a model/config: retain only files whose official Git/LFS identity matches.
        omitted = [f.rfilename for f in selected if f.rfilename in
                   ['text_encoder/LICENSE', 'text_encoder/README.md', 'vae/config.json']]
        selected = [f for f in selected if f.rfilename not in omitted]
        selected.sort(key=lambda f: (not f.rfilename.startswith('vae/'), f.rfilename))
        sources = {}
        for repo in ['Qwen/Qwen3-VL-4B-Instruct', 'black-forest-labs/FLUX.2-dev']:
            alt = api.model_info(repo, files_metadata=True)
            sources[repo] = (alt.sha, {f.rfilename: f for f in alt.siblings})
        def lfs_sha(remote):
            lfs = getattr(remote, 'lfs', None)
            return lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
        def fetch_exact(remote):
            name = remote.rfilename
            repo = 'Qwen/Qwen3-VL-4B-Instruct' if name.startswith('text_encoder/') else 'black-forest-labs/FLUX.2-dev'
            source_name = name.removeprefix('text_encoder/') if name.startswith('text_encoder/') else name
            alt_revision, alternatives = sources[repo]
            alt = alternatives.get(source_name)
            expected, actual = lfs_sha(remote), lfs_sha(alt) if alt else None
            same = alt is not None and remote.size == alt.size and (
                (expected and actual == expected) or
                (not expected and not actual and remote.blob_id == alt.blob_id))
            if not same:
                raise RuntimeError(f'No byte-identical official upstream resource for {name}; original FLUX.3 access required')
            asset = companion / name
            asset.parent.mkdir(parents=True, exist_ok=True)
            print(f'Exact official upstream: {name} <- {repo}@{alt_revision}', flush=True)
            if exact_modelscope_fallback and expected:
                origin = _modelscope_exact_download(repo, source_name, expected, remote.size, asset, range_workers)
                origin['official_huggingface'] = dict(repository=repo, revision=alt_revision)
                origin['verified_against'] = dict(repository=COMPANION_REPO, revision=info.sha,
                                                  lfs_sha256=expected, git_blob=remote.blob_id)
                return name, origin
            downloaded = hf_hub_download(repo_id=repo, filename=source_name, revision=alt_revision,
                                         endpoint=HF_ENDPOINT)
            # Do not mutate Hub cache or cross-link credentials; atomically copy only this verified file.
            import shutil
            fd, temp = tempfile.mkstemp(prefix=f'.{asset.name}.', suffix='.partial', dir=asset.parent)
            os.close(fd)
            try:
                shutil.copyfile(downloaded, temp)
                os.replace(temp, asset)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
            return name, dict(repository=repo, revision=alt_revision, filename=source_name,
                              verified_against=dict(repository=COMPANION_REPO, revision=info.sha,
                                                    lfs_sha256=expected, git_blob=remote.blob_id))
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for name, origin in pool.map(fetch_exact, selected):
                provenance[name] = origin
    else:
        snapshot_download(repo_id=COMPANION_REPO, revision=info.sha, endpoint=HF_ENDPOINT,
                          allow_patterns=['text_encoder/*', 'vae/*'], local_dir=str(companion),
                          max_workers=workers)
    files = {}
    for remote in selected:
        asset = companion / remote.rfilename
        if not asset.is_file():
            raise FileNotFoundError(f'Incomplete official snapshot: {remote.rfilename}')
        digest = sha256_file(asset)
        lfs = getattr(remote, 'lfs', None)
        expected = getattr(lfs, 'sha256', None) if lfs else None
        if isinstance(lfs, dict):
            expected = lfs.get('sha256')
        if expected and digest != expected:
            raise RuntimeError(f'Official SHA256 mismatch: {remote.rfilename}')
        # Git blob SHA verifies small files that do not use LFS.
        if not expected and getattr(remote, 'blob_id', None):
            data = asset.read_bytes()
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if blob != remote.blob_id:
                raise RuntimeError(f'Official Git blob mismatch: {remote.rfilename}')
        stat = asset.stat()
        files[remote.rfilename] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest,
                                           provenance=provenance.get(remote.rfilename, dict(repository=COMPANION_REPO, revision=info.sha)))
    manifest = dict(schema=1, repository=COMPANION_REPO, revision=info.sha,
                    endpoint=HF_ENDPOINT, files=files, base_dit=paths['dit'],
                    omitted_nonruntime_files=omitted, exact_official_upstream=exact_upstream_fallback,
                    exact_official_modelscope=exact_modelscope_fallback,
                    text_encoder_attribution='Qwen3-VL-4B-Instruct, Qwen Team, Apache-2.0')
    _atomic_json(old_manifest, manifest)
    resource_manifest(root)
    _atomic_json(companion / 'flux3_qdm_download_status.json',
                 dict(state='complete', pid=os.getpid(), manifest=str(old_manifest),
                      revision=info.sha, files=len(files), sha256_verified=True))
    print(json.dumps(dict(repository=COMPANION_REPO, revision=info.sha, files=len(files),
                          manifest=str(old_manifest)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-root', default=DEFAULT_WEIGHTS_ROOT)
    parser.add_argument('--revision', help='Optional official commit/tag; first download resolves and pins main')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--exact-upstream-fallback', action='store_true',
                        help='Fetch byte-identical official Qwen/FLUX.2-dev runtime files after comparing FLUX.3 Git/LFS identities')
    parser.add_argument('--verify-hashes', action='store_true')
    parser.add_argument('--exact-modelscope-fallback', action='store_true',
                        help='Use exact-SHA official publisher ModelScope weights, scoped direct HTTP with resumable chunks')
    parser.add_argument('--range-workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.range_workers < 1:
        parser.error('--workers must be positive')
    if args.check:
        manifest = resource_manifest(args.weights_root, args.verify_hashes)
        print(json.dumps(dict(revision=manifest['revision'], paths=resource_paths(args.weights_root)), indent=2))
    else:
        status = Path(args.weights_root).expanduser() / 'flux-3-image-klein' / 'flux3_qdm_download_status.json'
        _atomic_json(status, dict(state='downloading', pid=os.getpid(),
                                 transport='official-modelscope-direct' if args.exact_modelscope_fallback else 'official-huggingface',
                                 workers=args.workers, range_workers=args.range_workers))
        try:
            download(args.weights_root, args.revision, args.workers,
                     args.exact_upstream_fallback or args.exact_modelscope_fallback,
                     args.exact_modelscope_fallback, args.range_workers)
        except Exception as exc:
            _atomic_json(status, dict(state='failed', pid=os.getpid(), error_type=type(exc).__name__,
                                     resumable=True))
            raise


if __name__ == '__main__':
    main()
