#!/usr/bin/env python3
"""Download the official 2K transformer plus T5/SDXL-VAE without proxy."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

TRANSFORMER = 'PixArt-alpha/PixArt-Sigma-XL-2-2K-MS'
REVISION = 'cd274f4c8b01de0dcc87b65887dd357d473bf516'
COMPONENTS = 'PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers'
MS_COMPONENTS = 'AI-ModelScope/pixart_sigma_sdxlvae_T5_diffusers'


def no_proxy_opener():
    # Explicitly bypass environment proxies; never modify the caller's shell.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    opener.addheaders = [('User-Agent', 'curl/8.5.0')]
    return opener


def get_json(opener, url):
    with opener.open(url, timeout=20) as response:
        return json.load(response)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def download_file(spec, root, retries=5):
    opener = no_proxy_opener()
    path = root / spec['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = spec.get('sha256')
    if path.exists() and path.stat().st_size == spec['size'] and (not expected or sha256(path) == expected):
        print(f'Already complete: {spec["path"]}', flush=True)
        return
    partial = path.with_name(path.name + '.incomplete')
    for attempt in range(retries):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == spec['size']:
                if expected and sha256(partial) != expected:
                    partial.unlink()
                    raise ValueError('SHA256 mismatch; removed corrupt partial')
                os.replace(partial, path)
                return
            request = urllib.request.Request(spec['url'], headers={'Range': f'bytes={offset}-'} if offset else {})
            with opener.open(request, timeout=60) as response:
                if offset and response.status != 206:
                    offset = 0
                if offset:
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                        raise ValueError('Server returned an incorrect resume offset')
                last_print = 0
                with partial.open('ab' if offset else 'wb') as stream:
                    for chunk in iter(lambda: response.read(4 * 1024**2), b''):
                        stream.write(chunk)
                        offset += len(chunk)
                        if time.monotonic() - last_print > 10:
                            print(f'{spec["path"]}: {offset/2**30:.2f}/{spec["size"]/2**30:.2f} GiB', flush=True)
                            last_print = time.monotonic()
                    stream.flush()
                    os.fsync(stream.fileno())
            if partial.stat().st_size != spec['size']:
                raise ValueError('Incomplete response size')
            if expected and sha256(partial) != expected:
                partial.unlink()
                raise ValueError('SHA256 mismatch; removed corrupt partial')
            os.replace(partial, path)
            print(f'Complete: {spec["path"]}', flush=True)
            return
        except Exception as exc:
            # URLs can contain expiring signatures; do not print them.
            print(f'{spec["path"]}: attempt {attempt+1}/{retries}: {type(exc).__name__}', flush=True)
            if attempt + 1 == retries:
                raise RuntimeError(f'Download failed: {spec["path"]}. Repeat command to resume.') from None
            time.sleep(min(2**attempt, 15))


def component_specs(opener, source, endpoint='https://hf-mirror.com'):
    if source == 'modelscope':
        data = get_json(opener, f'https://modelscope.cn/api/v1/models/{MS_COMPONENTS}/repo/files?Revision=master&Recursive=true')
        if not data.get('Success'):
            raise RuntimeError('Cannot list ModelScope component mirror')
        return [dict(path=f['Path'], size=f['Size'], sha256=f.get('Sha256'),
                     url=f'https://modelscope.cn/api/v1/models/{MS_COMPONENTS}/repo?Revision=master&FilePath={urllib.parse.quote(f["Path"])}')
                for f in data['Data']['Files'] if f['Type'] == 'blob'
                and f['Path'].split('/')[0] in {'text_encoder', 'vae', 'tokenizer', 'scheduler'}]
    data = get_json(opener, f'{endpoint}/api/models/{COMPONENTS}/tree/main?recursive=true&expand=true&limit=100')
    return [dict(path=f['path'], size=f['size'], sha256=f.get('lfs', {}).get('oid'),
                 url=f'{endpoint}/{COMPONENTS}/resolve/main/{f["path"]}')
            for f in data if f['type'] == 'file'
            and f['path'].split('/')[0] in {'text_encoder', 'vae', 'tokenizer', 'scheduler'}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights-root', type=Path, default=Path('weights'))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--endpoint', default='https://hf-mirror.com', help='Hugging Face download endpoint')
    parser.add_argument('--component-source', choices=['huggingface', 'modelscope'], default='huggingface')
    parser.add_argument('--components-only', action='store_true')
    parser.add_argument('--transformer-only', action='store_true')
    args = parser.parse_args()
    if args.components_only and args.transformer_only:
        parser.error('Select only one component subset')
    root = args.weights_root / 'PixArt-Sigma-XL-2-2K-MS'
    root.mkdir(parents=True, exist_ok=True)
    opener = no_proxy_opener()
    specs = []
    if not args.transformer_only:
        specs.extend(component_specs(opener, args.component_source, args.endpoint.rstrip('/')))
    if not args.components_only:
        data = get_json(opener, f'{args.endpoint.rstrip("/")}/api/models/{TRANSFORMER}/tree/{REVISION}/transformer?expand=true')
        specs.extend(dict(path=f['path'], size=f['size'], sha256=f.get('lfs', {}).get('oid'),
                          url=f'{args.endpoint.rstrip("/")}/{TRANSFORMER}/resolve/{REVISION}/{f["path"]}')
                     for f in data if f['type'] == 'file')
    print(f'Download {sum(s["size"] for s in specs)/2**30:.2f} GiB to {root}; proxy disabled.', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda spec: download_file(spec, root), specs))
    index = {'_class_name': 'PixArtSigmaPipeline', '_diffusers_version': '0.37.1',
             'scheduler': ['diffusers', 'DPMSolverMultistepScheduler'],
             'text_encoder': ['transformers', 'T5EncoderModel'], 'tokenizer': ['transformers', 'T5Tokenizer'],
             'transformer': ['diffusers', 'Transformer2DModel'], 'vae': ['diffusers', 'AutoencoderKL']}
    (root / 'model_index.json').write_text(json.dumps(index, indent=2) + '\n')
    record = root / ('components_download.json' if args.components_only else 'transformer_download.json' if args.transformer_only else 'download_complete.json')
    record.write_text(json.dumps({'transformer_repo': TRANSFORMER, 'transformer_revision': REVISION,
        'component_repo': MS_COMPONENTS if args.component_source == 'modelscope' else COMPONENTS,
        'endpoint': args.endpoint, 'proxy': False,
        'files': [{k:v for k,v in s.items() if k != 'url'} for s in specs]}, indent=2) + '\n')
    print('Selected files downloaded and verified. Full generation also requires transformer + components.', flush=True)


if __name__ == '__main__':
    main()
