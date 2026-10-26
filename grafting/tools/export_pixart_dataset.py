#!/usr/bin/env python3
"""Snapshot completed pairs without stopping generation, then write a portable tar."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time

from generate_pixart_dataset import atomic_json, completed, digest, model_identity, write_manifests


RESUME = '''# 在其他服务器继续生成

此包只包含打包时已完成的图文对、完整 12K prompt、生成配置及模型 SHA256 参考。
不包含模型权重、运行锁、日志或停止标记。

拉取 PiD-Linear 的 main 分支并进入仓库后：

```bash
conda create -n linear-pid python=3.12 -y
conda activate linear-pid
pip install torch==2.10.0
pip install -r grafting/tools/requirements-pixart.txt

# 下载与原数据相同的模型（约 21.8 GB）；已有完整模型可跳过。
python grafting/tools/download_pixart_models.py \\
  --weights-root "$HOME/huggingface_ckpts" --endpoint https://hf-mirror.com --workers 2

# 解压后将 PIXART_DATA_ROOT 指向本目录。
export PIXART_DATA_ROOT="$HOME/dataset/PixArt-Sigma-Grafting-12K"
export PIXART_WEIGHTS="$HOME/huggingface_ckpts/PixArt-Sigma-XL-2-2K-MS"
GPU_IDS=0,1,2,3,4,5,6,7 CPU_THREADS=1 \\
bash grafting/tools/generate_pixart_8gpu.sh
```

默认每卡 batch 2、2048×2048、20 步、CFG 4.5，seed 和编号保持不变。
重复相同命令即可续做。不要重新抽取 prompt 或修改 generation_config.json。
迁移后第一次会核对权重 SHA256，并检查已保存图片，耗时比后续启动长。
模型文件内容不同时会报错，不会跳过校验。不同硬件和 batch 可能产生浮点差异。
打包包含的确切样本数见 export_manifest.json。模型权重需另外下载或传输。
'''


def model_proof(model_root, recipe):
    identity = recipe['model_identity']
    if model_identity(model_root) != identity:
        raise ValueError('Export model directory differs from the original dataset model identity.')
    manifest = json.loads((model_root / 'download_complete.json').read_text())
    downloaded = {r['path']: r for r in manifest['files']}
    result = {}
    for name, (size, _) in identity['weights'].items():
        row = downloaded[name]
        if row['size'] != size or len(row.get('sha256') or '') != 64:
            raise ValueError(f'Missing verified model SHA256: {name}')
        result[name] = {'size': size, 'sha256': row['sha256']}
    return {'model_identity_hash': digest(identity), 'weights': result}


class HashingWriter:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()

    def write(self, data):
        size = self.stream.write(data)
        self.hash.update(data[:size])
        return size


def export_dataset(root, archive, proof, workers=4):
    from tqdm import tqdm
    root = root.resolve();archive = archive.resolve()
    if archive.exists():
        raise FileExistsError(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    recipe = json.loads((root / 'generation_config.json').read_text())
    records = json.loads((root / 'records.json').read_text())
    recipe_hash = digest(recipe)
    # Freeze the candidate list once. Later generation is intentionally excluded.
    candidates = {int(p.stem) for p in (root / 'metadata').glob('*.json') if p.stem.isdigit()}
    with tempfile.TemporaryDirectory(prefix='.pixart-snapshot-', dir=archive.parent) as temporary:
        snapshot = Path(temporary) / root.name
        (snapshot / 'images').mkdir(parents=True)
        (snapshot / 'metadata').mkdir()

        def capture(record):
            if not completed(root, record, recipe_hash, recipe['width'], recipe['height']):
                return False
            source = root / record['image_path'];destination = snapshot / record['image_path']
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
            # Generator publishes by rename; hard links keep the selected bytes
            # stable even if a source path is later regenerated.
            shutil.copy2(root / 'metadata' / f'{record["index"]:08d}.json',
                         snapshot / 'metadata' / f'{record["index"]:08d}.json')
            return True

        selected = [r for r in records if r['index'] in candidates]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            count = sum(tqdm(pool.map(capture, selected), total=len(selected), desc='Snapshot complete pairs'))
        for name in ['generation_config.json', 'records.json', 'prompts_12k.jsonl',
                     'prompt_pool_30k.jsonl', 'prompt_provenance.json', 'README.md']:
            if (root / name).is_file():
                shutil.copy2(root / name, snapshot / name)
        if (root / 'source').is_dir():
            shutil.copytree(root / 'source', snapshot / 'source', ignore=shutil.ignore_patterns('*.incomplete'))
        atomic_json(snapshot / 'model_files_sha256.json', proof)
        manifest_count = write_manifests(snapshot, records, recipe_hash, recipe['width'],
                                         recipe['height'], recipe['stage1_size'])
        if manifest_count != count:
            raise RuntimeError('Snapshot count changed during validation')
        summary = dict(complete=count, total=len(records), remaining=len(records)-count,
                       created_at=time.time(), recipe_hash=recipe_hash,
                       dataset_directory=root.name, model_weights_included=False)
        atomic_json(snapshot / 'export_manifest.json', summary)
        (snapshot / 'RESUME.md').write_text(RESUME)
        files = sorted(p for p in snapshot.rglob('*') if p.is_file())
        partial = archive.with_name(archive.name + '.incomplete')
        with partial.open('wb') as stream:
            writer = HashingWriter(stream)
            with tarfile.open(fileobj=writer, mode='w|', format=tarfile.PAX_FORMAT,
                              bufsize=1024**2, dereference=True) as tar:
                for path in tqdm(files, desc='Write portable tar'):
                    tar.add(path, arcname=str(path.relative_to(snapshot.parent)), recursive=False)
            stream.flush();os.fsync(stream.fileno())
        os.replace(partial, archive)
        archive.with_suffix(archive.suffix + '.sha256').write_text(f'{writer.hash.hexdigest()}  {archive.name}\n')
        summary.update(archive=str(archive), archive_bytes=archive.stat().st_size,
                       archive_sha256=writer.hash.hexdigest())
        atomic_json(archive.with_suffix(archive.suffix + '.json'), summary)
        return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if args.workers < 1:
        p.error('workers must be positive')
    recipe = json.loads((args.dataset_root / 'generation_config.json').read_text())
    proof = model_proof(args.model_dir, recipe)
    print(json.dumps(export_dataset(args.dataset_root, args.archive, proof, args.workers), indent=2), flush=True)


if __name__ == '__main__':
    main()
