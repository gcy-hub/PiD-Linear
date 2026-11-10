# Framework downloads and caches

Training and cache preparation load local resources only. Download model resources explicitly:

```bash
python -m download.framework --model-config configs/models/flux2.yaml --components vae text_encoder tokenizer
```

The model YAML supplies local destinations in `model.resources` and optional download declarations in `model.downloads`, keyed by the same component names:

```yaml
model:
  resources:
    vae: ./weights/example/vae
  downloads:
    vae:
      repo_id: publisher/model
      revision: pinned-revision
      allow_patterns: ['*.json', '*.safetensors']
```

Each declaration downloads into that component's destination. Select patterns matching the repository layout; encoders take component directories (FLUX2), native checkpoint files or shard directories (FLUX3), and a native checkpoint file (PiD VAE). PiD text encoding uses the native `gemma-2-2b-it` directory. Existing local resources without download declarations are reused. Hugging Face's downloader resumes interrupted transfers. Missing destinations, declarations and unknown components fail explicitly.

Dataset downloads keep the existing resumable entry points:

```bash
python -m download.download_metadata --output-root ./dataset/raw
python -m download.download_images --help
```

`data.build_dataloader` accepts JSONL and CSV manifests, downloaded `data_jsons/*.json`, or indexed subsets containing `index.json`, `offsets.npy`, `manifest.jsonl` and `train_*.npy`. Counts reflect selected completed records, not preset subset sizes. Image paths resolve from each manifest's directory. Indexed manifests use the same path rule. Captions may be empty. Images too small for the configured buckets are counted in `dataset.rejected`.

Prepare caches with the shared model/algorithm/training/data config arguments:

```bash
python -m data.prepare --phase image --devices cuda:0 cuda:1 --cache-batch-size 2 \
  --model-config configs/models/flux2.yaml \
  cache.root=./cache data.manifest=./dataset/manifest.jsonl
python -m data.prepare --phase text --devices cuda:0 \
  --model-config configs/models/flux2.yaml \
  cache.root=./cache data.manifest=./dataset/manifest.jsonl
```

Supply local model resources through the same config overrides used for training. Matching entries are skipped. Device lists partition work; devices are processed sequentially, releasing each encoder before loading the next. Image batches share a bucket. FLUX3 text uses individual native variable-length encodings; batch collation pads embeddings and supplies a validity mask. Failed samples are recorded in `cache.root/failures.json`; preparation exits unsuccessfully if any fail.

Set `cache.image` and `cache.text` independently at training time. A cached component does not load its encoder. PiD caches contain clean low-resolution conditions; full RGB targets are loaded from source images. Cached values contain no training noise, degradation, dropout or timesteps. Caption dropout configurations additionally prepare an empty-caption entry and yield `empty_embeddings`, `empty_mask` and native `empty_text_ids` as applicable.

Cache metadata compares native family, resource filenames and file metadata, dtype, preprocessing version, sample identity, image file metadata and bucket or caption. Moving dataset/weight/cache roots preserves reuse when names and file metadata are preserved. Resource identities are collected once at the read boundary; training batches do not scan model files. New framework caches are independent of historical experiment caches.

Run the bounded CPU checks with `CUDA_VISIBLE_DEVICES='' python -m data.check_cpu`. Real native encoders and GPU execution require the scheduled acceptance checks.
