# Downloads and training preparation

Run all commands from the repository root. Paths below are relative and can be
changed through script arguments or environment variables. Downloads use the CPU;
text caching and validation-condition preparation require CUDA GPUs.

## Download dependencies and authentication

Use the `linear-pid` environment. For downloads alone, only these packages are needed:

```bash
conda activate linear-pid
pip install 'huggingface-hub>=0.36,<1' requests tqdm
HF_ENDPOINT=https://huggingface.co hf auth login
```

Before downloading Gemma, accept the terms on
[google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) using the same
Hugging Face account. A logged-in account without model access will receive a
403 error. Authentication can also use the `HF_TOKEN` environment variable.
For downloads, `HF_HUB_OFFLINE` must be unset or `0`. Model and metadata
downloaders explicitly use the official `https://huggingface.co` endpoint,
including when a different endpoint is configured in the caller's environment.

## Download everything

The following script contains the individual download commands, with no launcher
or complex argument handling:

```bash
conda activate linear-pid
WEIGHTS_ROOT=./weights \
DATASET_ROOT=./raw_data/MultiAspect-4K-1M \
MODEL_WORKERS=4 IMAGE_WORKERS=8 \
bash download/download_all.sh
```

Repeat the same command after an interruption. Hugging Face retains download
progress; the image downloader skips downloaded images and resumes temporary
files when the source server supports HTTP Range. Existing metadata JSONs are
preserved because the image downloader adds `id` and relative `image_path` fields.
Image download failures are recorded in `datas/failed_downloads.tsv`; rerun the
image command to retry them. Some source URLs may no longer be available.

The download scripts pin official repository revisions. Model repositories and
file patterns are listed in `download_models.py`; the metadata archive source is
listed in `download_metadata.py`.

| Component | Source | Used for |
| --- | --- | --- |
| PiD v1.5 FLUX undistilled | [nvidia/PiD](https://huggingface.co/nvidia/PiD) | Student initialization and original PiD baseline; optional teacher supervision |
| FLUX VAE (`ae.safetensors`) | [nvidia/PiD](https://huggingface.co/nvidia/PiD) | Frozen image encoder |
| Gemma 2 2B IT | [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) | Caption encoding and offline text cache |
| Z-Image-Turbo | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Prompt-only generation and generated validation conditions |
| MultiAspect-4K-1M metadata | [Owen777/UltraFlux-v1](https://huggingface.co/Owen777/UltraFlux-v1/tree/main) | Captions and source image URLs |

The PiD download selects only the undistilled FLUX checkpoint, rather than all
PiD variants. KDA is initialized in code and requires no additional model weights.
Trained Linear-PiD checkpoints are produced by training and are not included in
these upstream downloads.

## Individual download commands

```bash
conda activate linear-pid

python download/download_models.py --models pid --weights-root ./weights --workers 4
python download/download_models.py --models vae --weights-root ./weights --workers 4
python download/download_models.py --models gemma --weights-root ./weights --workers 4
python download/download_models.py --models zimage --weights-root ./weights --workers 4

python download/download_metadata.py --output-root ./raw_data/MultiAspect-4K-1M
python download/download_images.py all \
    --json-dir ./raw_data/MultiAspect-4K-1M/data_jsons \
    --datas-dir ./raw_data/MultiAspect-4K-1M/datas \
    --workers 8 --max-per-json 0 --timeout 60 --retries 3 --flush-every 200
```

`--models all` downloads all four components in one Python invocation. The
provided `download_images.py` remains unchanged. Keep its default JSON updates
enabled so the training index can locate the downloaded images.

## Prepare the training index, text cache, and galleries

After downloads, use the complete training environment with the pinned FLA KDA
dependency and working CUDA. This script calls the existing preparation tools;
it does not start training:

```bash
conda activate linear-pid
WEIGHTS_ROOT=./weights \
DATASET_ROOT=./raw_data/MultiAspect-4K-1M \
GALLERY_ROOT=./outputs/linear-pid/assets \
GPU_IDS=0,1,2,3 INDEX_WORKERS=8 TEXT_WORKERS=2 CPU_THREADS=1 TEXT_BATCH_SIZE=8 \
bash download/prepare_training.sh
```

GPU IDs and worker counts are configurable. By default, the index checks image
headers and dimensions; full decoding occurs in the training loader. To decode
every image during indexing, change `--image-verification header` to `full` in
`prepare_training.sh`. Text caching runs until completion (`--max-seconds 0`),
saves progress periodically, and resumes when the same script is run again.
The script does not install a timed restart service.

Do not change the metadata or dataset location after building the index/cache;
the preparation tools fingerprint their inputs, and the image index stores
resolved image paths. Download missing images before building the final index.

## Resulting layout

```text
weights/
  PiD/checkpoints/
    ae.safetensors
    PiD_v1pt5_res2kto4k_sr4x_official_flux_undistilled/model_ema_bf16.pth
  gemma-2-2b-it/
  Z-Image-Turbo/
raw_data/MultiAspect-4K-1M/
  .downloads/MultiAspect-4K-1M.tar.gz
  data_jsons/0001.json ...
  datas/0001/...
  linear_pid_index/
  linear_pid_text_cache/
outputs/linear-pid/assets/
```

Pass these locations explicitly to training using `--weights-root`, `--index-root`,
`--text-cache-root`, and `--gallery-root`. These scripts do not change existing
training runs or their saved configurations. Downloaded data, weights, caches,
logs, and outputs under the default directories are excluded by the repository
`.gitignore`.
