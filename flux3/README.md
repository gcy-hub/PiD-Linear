# FLUX.3-QDM

FLUX.3 Image Klein Base training with QDM replacing the 28 single-stream attention blocks. Native independent-stream blocks remain unchanged. The implementation uses the model's 3072 hidden channels, 24 attention heads and 128 head dimension, 8×8 local image regions and four FP32 joint memory states. Model weights, datasets, caches and generated outputs live outside tracked source. The supplied `DiffSynth-Studio` source is unmodified; this code has no runtime dependency on another PiD checkout or FLA.

The native text encoder is **Qwen3-VL-4B-Instruct**. Its hidden layers 4, 8, 12, 16, 20, 24, 28 and 32 concatenate to 20,480 channels, with native chat formatting and padding to multiples of 80 (maximum 4096). The FLUX.3 VAE uses its native BN-normalized 128-channel latents with spatial stride 16.

## Environment

Run from the repository root. The commands target Python 3.12 and the verified CUDA 12.8 package versions; they require no administrator privileges.

```bash
conda create -n flux3-qdm python=3.12 -y
conda activate flux3-qdm
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r flux3/requirements.txt
```

The scripts add `flux3` and its supplied `DiffSynth-Studio` directory to `PYTHONPATH`. They do not require installing the older PiD environment. SwanLab records loss, learning rates, alignment metrics, throughput, configuration and validation images; the default is offline logging. Add `--swanlab-mode cloud` to a training command after configuring your own SwanLab account.

Set portable data and output locations once:

```bash
conda activate flux3-qdm
export FLUX3_WEIGHTS_ROOT="$HOME/huggingface_ckpts"
export FLUX3_DATASET_ROOT="$HOME/dataset/MultiAspect-4K-1M"
export FLUX3_SUBSET_ROOT="$FLUX3_DATASET_ROOT/pid_vdn_subsets"
export FLUX3_CACHE_ROOT="$FLUX3_DATASET_ROOT/flux3_qdm_cache"
export FLUX3_QDM_RUN="$PWD/outputs/flux3-qdm"
export GPU_IDS="1,2"  # Replace with the GPUs allocated to this workload.
```

GPU IDs are configurable. Preparation uses the current process's logical CUDA IDs; the training launcher applies `GPU_IDS` as `CUDA_VISIBLE_DEVICES` and assigns one training process to each selected GPU. Keep that selection fixed when resuming so the effective batch and sampler state remain consistent. Scripts do not request Slurm allocations.

## Weights

The base DiT is expected at `$FLUX3_WEIGHTS_ROOT/flux-3-image-klein-base/model.safetensors`. Download that exact base checkpoint through the official Hugging Face CLI if it is absent. Complete any repository access authorization in your own Hugging Face account first.

```bash
conda activate flux3-qdm
HF_ENDPOINT=https://huggingface.co hf download black-forest-labs/flux-3-image-klein-base model.safetensors \
  --local-dir "$FLUX3_WEIGHTS_ROOT/flux-3-image-klein-base"
bash flux3/scripts/download_flux3_qdm.sh --weights-root "$FLUX3_WEIGHTS_ROOT" --workers 4
```

The companion downloader fetches the official FLUX.3 `text_encoder/` and `vae/` resources, pins a revision and verifies their hashes. It never replaces the local base DiT. If the companion repository is gated, the explicitly selected fallback verifies that official upstream Qwen text files and the official FLUX.2-dev VAE have **identical Git/LFS identities** to the intended FLUX.3 resources:

```bash
conda activate flux3-qdm
bash flux3/scripts/download_flux3_qdm.sh \
  --weights-root "$FLUX3_WEIGHTS_ROOT" --workers 4 --exact-upstream-fallback
```

An optional faster transport uses only the official publishers' ModelScope replicas after comparing their SHA256 values to the same expected FLUX.3 files. It supports verified durable range chunks and resume. Direct requests bypass the proxy only inside those ModelScope HTTP sessions; the global proxy configuration is preserved.

```bash
conda activate flux3-qdm
bash flux3/scripts/download_flux3_qdm.sh \
  --weights-root "$FLUX3_WEIGHTS_ROOT" --workers 4 \
  --exact-modelscope-fallback --range-workers 8
bash flux3/scripts/download_flux3_qdm.sh --weights-root "$FLUX3_WEIGHTS_ROOT" --check
```

The companion directory contains `flux3_qdm_resources.json` with exact file identities and actual provenance, plus `flux3_qdm_download_status.json`. The exact upstream fallback omits unused documentation and the unused VAE JSON configuration; native model constructors and the verified runtime files define inference. Qwen3-VL-4B-Instruct is attributed to the Qwen Team under Apache-2.0.

## Data and conditioning cache

The existing nested `20000` and `50000` subset directories contain `index.json`, memory-mapped `offsets.npy`, `manifest.jsonl` and `train_<bucket>.npy` row selections. Preparation reads original image paths and captions from those records. The PiD 2K bucket dimensions are copied locally, with EXIF transpose, RGB conversion, Lanczos cover resize and center crop.

Text and image preparation use separate resident encoder processes. No DiT is loaded during preparation. The cache stores one BF16 embedding per caption hash, one BF16 image latent per image ID, FP32 native IDs and the empty-prompt embedding. Atomic tensor/metadata writes and model, preprocessing and source fingerprints support resume. The 20K subset needs approximately **219 GiB** of conditioning tensor payload for the inspected captions; the 50K stage reuses matching first-stage assets.

Run both phases, or run either phase independently:

```bash
conda activate flux3-qdm
bash flux3/scripts/prepare_flux3_qdm.sh \
  --subset-root "$FLUX3_SUBSET_ROOT" --cache-root "$FLUX3_CACHE_ROOT" \
  --weights-root "$FLUX3_WEIGHTS_ROOT" --subset-size 20000 --phase all \
  --gpu-ids "$GPU_IDS" --workers 4 --threads 1 --resume --max-seconds 1200

bash flux3/scripts/prepare_flux3_qdm.sh \
  --subset-size 20000 --phase text --gpu-ids "$GPU_IDS" \
  --workers 4 --threads 1 --resume --max-seconds 1200
bash flux3/scripts/prepare_flux3_qdm.sh \
  --subset-size 20000 --phase image --gpu-ids "$GPU_IDS" \
  --workers 4 --threads 1 --resume --max-seconds 1200
bash flux3/scripts/prepare_flux3_qdm.sh --subset-size 20000 --check-only
```

Rerun a preparation command after its time budget expires. To prepare B's remaining records, use the same cache root and `--subset-size 50000 --phase all --resume`; the 50K selection includes the 20K records. For a bounded correctness run, use `--limit 32` with a separate smoke cache root and pass `--sample-limit 32` to training tests. Production runs use the full selected subset.

## Preparation followed by A1

After the CPU and real 2K correctness checks pass, this detached controller prepares the full 20K cache and starts **A1 only**. It retains an owned lease while renewing its manager PID using a new `Popen`, so each preparation manager stays within its configured 20-minute lifetime including grace and cleanup. Preparation is resumable at every atomic cache file. Once the cache passes `CachedDataset` validation, it delegates A1 to the training repeat controller with batch size 1, accumulation 1 and automatic checkpoint resume.

```bash
conda activate flux3-qdm
bash flux3/scripts/prepare_and_train_flux3_qdm.sh \
  --output-root "$FLUX3_QDM_RUN" --weights-root "$FLUX3_WEIGHTS_ROOT" \
  --subset-root "$FLUX3_SUBSET_ROOT" --cache-root "$FLUX3_CACHE_ROOT" \
  --gpu-ids "$GPU_IDS" --workers 4 --threads 1 --cycle-seconds 1200
```

The command returns after starting the detached controller. It refuses a duplicate controller for the same run root. Only its own child processes are supervised. Three consecutive preparation cycles without new cache files fail explicitly. Source/model/preprocessing mismatches also fail instead of restarting indefinitely.

```bash
conda activate flux3-qdm
bash flux3/scripts/prepare_and_train_flux3_qdm.sh --output-root "$FLUX3_QDM_RUN" --status
tail -f "$FLUX3_QDM_RUN/bootstrap/controller.log"
tail -f "$FLUX3_QDM_RUN/repeat/latest-training.log"

bash flux3/scripts/prepare_and_train_flux3_qdm.sh --output-root "$FLUX3_QDM_RUN" --stop
```

`--stop` writes both bootstrap and training stop markers. A training cycle saves at an update/image boundary; preparation retains all completed cache files. Running the original launch command again resumes after the previous owned child exits. A1 completion includes its validation and does not trigger A2.

## 8卡四组并行，再合并全部28层

需要“4个双卡队列完成14组独立实验，再合并各组B权重，8卡联合训练A1 500步、A2 1000步、B 10000步”时，使用[并行及联合训练说明](docs/parallel-pairs.md)。每卡bs2、累积1；联合阶段每500步保存并生成10张学生2K图。数据使用现有50K及其20K子集。

## 14 组独立双层实验

该入口按顺序测试 `[0,1]`、`[2,3]`，直到 `[26,27]`，编号为从 0 开始的 Joint Attention 层索引。每组只将选定的两层替换为 **Local Softmax + Halo2 + 四象限 QDM**，其余 26 层保持原始 Full Attention。每组独立从原始 base 初始化，不继承其他组的学生权重。

| 阶段 | 更新次数 | 每卡 batch | 梯度累积 | 验证 |
|---|---:|---:|---:|---|
| A1 | 50 | 2 | 1 | 不生成 |
| A2 | 50 | 2 | 1 | 不生成 |
| B | 50 | 2 | 1 | 结束后生成 10 张学生 2K 图 |

三个阶段均使用同一份 20K 数据和原生 FLUX.3 缓存，warmup 均为 5 步。A2 加载本组 A1，B 加载本组 A2。局部分支保留 RoPE；QDM 为图像写入、NoPE，不启用文本写入。有效 batch 为 `GPU 数 × 2`。

### 在另一台服务器准备资源

拉取 `main` 即包含固定版本的 DiffSynth 运行源码，不需要初始化子模块。先按上文安装 `flux3-qdm` 环境、准备原始 base 权重和配套文本编码器、VAE，然后设置这三项路径：

```bash
conda activate flux3-qdm
export FLUX3_WEIGHTS_ROOT="$HOME/huggingface_ckpts"
export FLUX3_SUBSET_ROOT="$HOME/dataset/MultiAspect-4K-1M/pid_vdn_subsets"
export FLUX3_CACHE_ROOT="$HOME/dataset/MultiAspect-4K-1M/flux3_qdm_cache"
export GPU_IDS=0,1,2,3
```

`FLUX3_SUBSET_ROOT` 下应有 `20000/` 索引，也可以直接指向含 `index.json` 的 20K 索引目录。若从现有 50K 导出包部署，保留其内的 20K 子集和样本 ID，核对索引中的图片路径；无需下载完整 1M。只改数据根目录或路径时，JSONL 偏移索引也必须与修改后的文件一致。

FLUX.3 必须使用原生 Qwen3-VL 文本和 VAE latent 缓存，不能复用 Gemma 特征。缓存校验包含源图片路径和时间戳；迁移服务器后先检查，检查不通过时按新路径重新运行准备命令，禁止绕过指纹检查：

```bash
conda activate flux3-qdm
bash flux3/scripts/prepare_flux3_qdm.sh \
  --weights-root "$FLUX3_WEIGHTS_ROOT" \
  --subset-root "$FLUX3_SUBSET_ROOT" --cache-root "$FLUX3_CACHE_ROOT" \
  --subset-size 20000 --check-only

# 缓存不存在或校验不通过时执行；0 表示本次准备不设置运行时限。
# 中断后重复同一命令，跳过匹配的完整缓存。
bash flux3/scripts/prepare_flux3_qdm.sh \
  --weights-root "$FLUX3_WEIGHTS_ROOT" \
  --subset-root "$FLUX3_SUBSET_ROOT" --cache-root "$FLUX3_CACHE_ROOT" \
  --subset-size 20000 --phase all --gpu-ids "$GPU_IDS" \
  --workers 4 --threads 1 --resume --max-seconds 0
```

### 启动、查看与恢复

从仓库根目录执行。下列命令后台运行完整的 14 组实验，每约 20 分钟在更新或图片边界保存并续跑；无运行时限的服务器也可以使用此入口。

```bash
conda activate flux3-qdm
GPU_IDS="$GPU_IDS" CPU_THREADS=1 \
bash flux3/scripts/train_flux3_pairwise_hybrid.sh \
  --weights-root "$FLUX3_WEIGHTS_ROOT" \
  --subset-root "$FLUX3_SUBSET_ROOT" --cache-root "$FLUX3_CACHE_ROOT" \
  --batch-size 2 --grad-accum 1 --steps 50 --warmup-steps 5 \
  --workers 4 --threads 1 --output-root outputs/flux3-pairwise-hybrid-50
```

重复同一启动命令可恢复中断实验；已完成阶段和已生成图片会跳过。运行中不重复启动。保存的计划要求恢复时路径、GPU 集合和 batch 等配置保持一致。

```bash
conda activate flux3-qdm
bash flux3/scripts/train_flux3_pairwise_hybrid.sh \
  --output-root outputs/flux3-pairwise-hybrid-50 --status
tail -f outputs/flux3-pairwise-hybrid-50/repeat/latest-training.log
cat outputs/flux3-pairwise-hybrid-50/summary.csv

# 手动停止整个 14 组实验，等待当前更新或图片保存完成。
bash flux3/scripts/train_flux3_pairwise_hybrid.sh \
  --output-root outputs/flux3-pairwise-hybrid-50 --stop
```

各组目录为 `outputs/flux3-pairwise-hybrid-50/pair_00_01/` 等，内部包含 `a1/`、`a2/`、`b/`。最终验证图片位于 `b/galleries/step_000000050/student/2048/`。`summary.csv` 在阶段/续训周期结束时更新，当前逐步进度以日志为准。该入口不提交 Slurm，不自动停止其他任务。

## 常规三阶段训练

### Partial joint-layer replacement

`--qdm-layers` selects zero-based indices among the 28 joint blocks. All other
joint blocks retain their native Full Attention; early independent blocks remain
unchanged. Omitting the option keeps the original all-28-layer behavior. The
layout is saved in checkpoints and checked on resume, stage transitions and
inference; evaluation rebuilds exactly that layout.

```bash
conda activate flux3-qdm
GPU_IDS=0,1 CPU_THREADS=1 bash flux3/scripts/train_flux3_qdm.sh \
  --stage a1 --qdm-layers 0,9,18,27 --steps 500 \
  --batch-size 4 --grad-accum 1 --save-steps 50 \
  --workers 4 --threads 1 --resume auto \
  --output-root outputs/flux3-qdm-layer-sweep-20261020/layers-4
```

For independent two-GPU Slurm experiments, submit one job for each layer count.
The script distributes the selected indices uniformly from 0 to 27, uses the
original base weights, trains A1 for 500 updates, and then runs validation.
Each job requests one node, two GPUs and up to six days. These are four
independent jobs; use your cluster's node-placement options if exactly two
physical nodes are required. Outputs and SwanLab runs remain separate.

```bash
conda activate flux3-qdm
mkdir -p outputs/flux3-qdm-layer-sweep-20261020/slurm
for count in 4 8 14 20; do
  QDM_LAYER_COUNT="$count" sbatch --job-name="flux3-qdm-a1-l$count" \
    flux3/scripts/train_flux3_qdm_layers.slurm
done
```

Repeat the same submission for a failed/interrupted experiment to resume its
latest complete checkpoint in the same output directory.

Each selected GPU processes one sample per update, with accumulation 1. Effective batch therefore equals the number of selected GPUs. The default schedule is:

| Stage | Subset | Updates | Big matrix LR | Small tensor LR | LoRA LR | Save / validate |
|---|---:|---:|---:|---:|---:|---|
| A1 | 20K | 500 | 1e-4 | 5e-4 | Frozen | Save every 50; validate at end |
| A2 | 20K | 1000 | 1e-4 | 5e-4 | Frozen | Save every 100; validate at end |
| B | 50K | 10000 | 4e-5 | 2e-4 | 1e-4 | Save and validate every 2000 |

AdamW uses zero weight decay; warmup is 50/50/200 updates followed by cosine scheduling to a 5% floor. A1 optimizes each layer's **image-only relative squared error**, `sum((image_student-image_teacher)^2) / sum(image_teacher^2)`, after output projection. Text relative error, the previous joint relative error and image energy fraction are diagnostics. A2 distills the prediction. B adds rank-64, alpha-64 LoRA with the approved KD objective.

All stages replace 10% of captions with the cached native empty-prompt sequence (`--caption-dropout 0.1`). Selection depends on sample ID and the saved sampler cursor, so resume and rank assignment preserve it. Teacher and student share the resulting conditions, noisy latents and timesteps. The observed dropout fraction and cumulative empty-prompt exposure are logged. DiT calls in training, validation and independent sampling share BF16 autocast with its cache disabled; FP32 trainable weights and FP32 memory solves remain unchanged. Evaluation cache identities include this compute policy and reject reuse of older incompatible output directories.

Checkpoints include trainable deltas, optimizer, scheduler, random states, the global sampler cursor and the training policy. A complete checkpoint marker is required for automatic resume. Changing the loss, caption dropout or compute policy requires a new experiment directory; older A1 checkpoints are not silently resumed under the new objective.

Run one stage in the foreground:

```bash
conda activate flux3-qdm
GPU_IDS="$GPU_IDS" bash flux3/scripts/train_flux3_qdm.sh \
  --stage a1 --weights-root "$FLUX3_WEIGHTS_ROOT" \
  --subset-root "$FLUX3_SUBSET_ROOT" --cache-root "$FLUX3_CACHE_ROOT" \
  --output-root "$FLUX3_QDM_RUN" --batch-size 1 --grad-accum 1 \
  --workers 4 --threads 1 --resume auto --max-seconds 1200
```

Use `--stage a2` after A1 completes, and `--stage b` after A2 plus the full 50K cache are ready. For renewable background training with an already complete cache:

```bash
conda activate flux3-qdm
bash flux3/scripts/train_flux3_qdm_repeat.sh --stage a1 \
  --gpu-ids "$GPU_IDS" --cycle-seconds 1200 --output-root "$FLUX3_QDM_RUN" \
  --weights-root "$FLUX3_WEIGHTS_ROOT" --subset-root "$FLUX3_SUBSET_ROOT" \
  --cache-root "$FLUX3_CACHE_ROOT" --batch-size 1 --grad-accum 1 --workers 4 --threads 1
```

For an explicitly requested complete A1 → A2 → B run, first prepare both subset cache manifests, then replace `--stage a1` with `--stage all` in the repeat command. Each stage waits for its required validation before the next begins. The preparation-to-A1 bootstrap always stops at A1.

For two nodes with four allocated GPUs each, this Slurm script runs **A1 only**, with batch 4 per GPU, accumulation 1, effective batch 32, 500 updates and a checkpoint every 50 updates. It starts from the original base on a new output directory; submitting the same directory again restores a compatible checkpoint. After training exits, it validates the final checkpoint on four allocated GPUs, then stops. Resource paths use the environment variables above; account and partition can be overridden with `sbatch` options.

```bash
conda activate flux3-qdm
export FLUX3_QDM_RUN="$PWD/outputs/flux3-qdm-a1-image-kd-cfg"
mkdir -p "$FLUX3_QDM_RUN/slurm"
sbatch --output="$FLUX3_QDM_RUN/slurm/%x-%j.out" \
  flux3/scripts/train_flux3_qdm_8gpu.slurm
```

For all 28 QDM layers, run **A2 only** from a saved A1 checkpoint on two nodes/eight GPUs. This job uses batch 2 per GPU, accumulation 1 (effective batch 16), 4,000 A2 updates and saves/validates every 500 updates. It loads the A1 delta and creates a new A2 optimizer/scheduler; rerunning the job resumes a compatible A2 checkpoint. Use the same run root to retain A1 and A2 in one experiment directory. `FLUX3_A1_CHECKPOINT` may select a specific complete A1 checkpoint; otherwise the script reads the run root's `a1/checkpoints/latest.json`.

```bash
conda activate flux3-qdm
export FLUX3_QDM_RUN="$PWD/outputs/flux3-qdm-all28-a1a2-bs4-acc4"
# Optional: export FLUX3_A1_CHECKPOINT=/path/to/complete-a1-checkpoint
mkdir -p "$FLUX3_QDM_RUN/slurm"
sbatch --output="$FLUX3_QDM_RUN/slurm/%x-%j.out" \
  flux3/scripts/train_flux3_qdm_all28_a2_8gpu.slurm
```

Monitor or stop a training-only repeat controller:

```bash
conda activate flux3-qdm
bash flux3/scripts/train_flux3_qdm_repeat.sh --output-root "$FLUX3_QDM_RUN" --status
bash flux3/scripts/train_flux3_qdm_repeat.sh --output-root "$FLUX3_QDM_RUN" --stop
```

## Evaluation and sampling

Training validation generates **only the student model's 10 fixed 2048×2048 images**, using the same prompts/seeds at each checkpoint, **50 steps, CFG 3.5 and alpha 6.93**. No teacher trajectory, 1K/4K images, paired images or crops are generated during training validation. Images are saved under `galleries/step_<step>/student/2048/` and recorded in SwanLab. Generation and VAE decoding remain resumable; validation completes before training resumes with all selected ranks.

Standalone evaluation retains the optional ORIGINAL/STUDENT comparison at 1K/2K/4K. Add `--student-only --resolutions 2048` for the same lightweight gallery used during training. Independent TE, DiT and VAE phases do not coexist on the GPU. Completed images are published one at a time; paired evaluation saves comparisons under `pairs/<resolution>/`.

```bash
conda activate flux3-qdm
CHECKPOINT="$(PYTHONPATH="$PWD/flux3" python -c 'import os; from pathlib import Path; from flux3_qdm.repeat import latest_complete; print(latest_complete(Path(os.environ["FLUX3_QDM_RUN"])/"a1"))')"
bash flux3/scripts/evaluate_flux3_qdm.sh --checkpoint "$CHECKPOINT" --stage a1 \
  --stage-root "$FLUX3_QDM_RUN/a1" --output-root "$FLUX3_QDM_RUN/a1/galleries/manual" \
  --weights-root "$FLUX3_WEIGHTS_ROOT" --gpu-ids "$GPU_IDS" \
  --workers 4 --threads 1 --steps 50 --cfg 3.5 --alpha 6.93 \
  --resolutions 1024,2048,4096 --max-seconds 1200 --resume

bash flux3/scripts/sample_flux3_qdm.sh --checkpoint "$CHECKPOINT" \
  --prompt 'A red fox in snow, fine fur detail' --seed 42 --height 2048 --width 2048 \
  --steps 50 --cfg 3.5 --alpha 6.93 --output-root outputs/flux3-qdm/sample-fox \
  --gpu-id "${GPU_IDS%%,*}" --threads 1 --max-seconds 1200 --resume
```

The student-only sampling command saves `sample.png` and `sample.json`; rerun it to resume a time-limited phase. Use a complete checkpoint directory from the selected stage.

## Checks

The verification programs below are local development files; `flux3/tests/` is excluded from the public source bundle. They are not required to train or sample.

CPU checks do not allocate GPUs:

```bash
conda activate flux3-qdm
export PYTHONPATH="$PWD/flux3:$PWD/flux3/DiffSynth-Studio${PYTHONPATH:+:$PYTHONPATH}"
python flux3/tests/check_data.py
python flux3/tests/check_runtime.py
python flux3/tests/check_model.py
python flux3/tests/check_repeat.py
python flux3/tests/check_launcher.py
python flux3/tests/check_evaluation_schedule.py
python flux3/tests/check_bootstrap.py
```

The production launch also requires real 2K forward/backward and interruption/resume checks with batch 1 and accumulation 1 for A1, A2 and B on the explicitly selected free GPUs. The full model shape/checkpoint check uses the first selected GPU:

```bash
conda activate flux3-qdm
CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}" python flux3/tests/check_full_gpu.py \
  --weights "$FLUX3_WEIGHTS_ROOT/flux-3-image-klein-base/model.safetensors" \
  --output-root "$FLUX3_QDM_RUN/full-gpu-check" --resolution 2048 --threads 1 --stages a1 a2 b
```

Use staged real-cache training runs with `--sample-limit 32`, batch 1, accumulation 1 and `--resume auto` to verify checkpoint continuation before the full workload. Model checkpoints and the large conditioning cache are deliberately excluded from Git.

### Eight-layer A2 on two nodes

Continue the eight-layer A1-500 checkpoint in the same experiment root. This job
uses two nodes with four GPUs each, per-GPU batch 4 and accumulation 2 (effective
batch 64). A2 runs 1,000 updates, saves every 100, then validates. The selected
zero-based layers remain `0,4,8,12,15,19,23,27`. The completed A1 marker supplies
the parent; same-stage restarts use the latest complete A2 checkpoint.

```bash
conda activate flux3-qdm
mkdir -p outputs/flux3-qdm-layer-sweep-20261020/slurm
sbatch flux3/scripts/train_flux3_qdm_layers_a2_8gpu.slurm
```

Outputs: `outputs/flux3-qdm-layer-sweep-20261020/layers-8/a2`. Do not submit a
second copy while the same experiment is running.

### All 28 layers: serial A1 and A2 on two nodes

This independent run starts from the official undistilled base and replaces all
28 joint layers. It trains A1 for 500 updates, then initializes A2 from that
completed A1 checkpoint and trains for 1,000 updates. Both stages use per-GPU
batch 4 and accumulation 4 on eight GPUs (effective batch 128), saving and
validating every 500 stage updates. A2's midpoint validation returns to the
same two-node, eight-rank trainer with full optimizer/scheduler/RNG/data resume.
The training allocation is held during four-GPU validation. B is not started.

```bash
conda activate flux3-qdm
mkdir -p outputs/flux3-qdm-all28-a1a2-bs4-acc4/slurm
sbatch flux3/scripts/train_flux3_qdm_all28_a1a2_8gpu.slurm
```

Outputs: `outputs/flux3-qdm-all28-a1a2-bs4-acc4/{a1,a2}`. To recover from an
interrupted job, repeat the submission only after the previous job has exited.

Slurm allocation/step-creation socket timeouts are retried at most three times,
ten seconds apart, only before any remote training/evaluation process starts.
The job log marks each remote launch with `FLUX3_QDM_PAYLOAD_STARTED`.
Training errors, OOM and expired allocations without a transport failure exit
immediately. Override `FLUX3_SLURM_START_ATTEMPTS` and
`FLUX3_SLURM_RETRY_SECONDS` in the submission environment if needed.

## Four low-error layers with Halo2

This experiment replaces only zero-based joint layers **4, 24, 26, 27**. The other
24 joint layers and all independent processing blocks retain original Full
Attention. Each replaced layer uses 8×8 image queries, Halo2 image keys/values
(12×12 before edge masking), all text keys/values, and fixed-four-quadrant QDM.
Native RoPE is applied before spatial gathering in the Softmax branch; the QDM
branch uses no RoPE. Original base weights initialize the experiment.

A1 trains 500 steps on the existing 20K cache; A2 defaults to 1,000 on the same
20K; B defaults to 10,000 on the 50K cache. All use per-GPU batch 2 and accumulation
1 (effective batch 8 on four GPUs). Checkpoints and **student-only 10-prompt 2K
validation** run every 100 steps and at stage completion. A1/A2 train added QDM
parameters; B additionally trains the Softmax gate and zero-initialized Q/K/V/O
LoRA. Full Attention layers remain frozen throughout.

The background controller saves and starts a fresh process about every 20 minutes,
resuming optimizer, data position and tracking in the same directory. The first
command runs **only A1** and stops after its final validation:

```bash
conda activate flux3-qdm
GPU_IDS=0,1,2,3 CPU_THREADS=1 bash flux3/scripts/train_flux3_lowmse4_halo.sh

# Status / intentional stop (stopping prevents automatic restart).
bash flux3/scripts/train_flux3_lowmse4_halo.sh --status
bash flux3/scripts/train_flux3_lowmse4_halo.sh --stop
```

After reviewing A1, start the next stages individually; each loads the preceding
stage checkpoint from the same output root. To run all three serially instead,
use `--stage all` (already completed stages are skipped):

```bash
conda activate flux3-qdm
GPU_IDS=0,1,2,3 bash flux3/scripts/train_flux3_lowmse4_halo.sh --stage a2
GPU_IDS=0,1,2,3 bash flux3/scripts/train_flux3_lowmse4_halo.sh --stage b
# Or: bash flux3/scripts/train_flux3_lowmse4_halo.sh --stage all
```

Default output: `outputs/flux3-qdm-lowmse4-halo2-node01/{a1,a2,b}`.
Controller state and current log are in `repeat/state.json` and
`repeat/latest-training.log`. Set `FLUX3_LOW4_RUN` or pass `--output-root` to choose
another directory. Existing checkpoints without `halo` reconstruct as Halo0;
resuming with a different Halo setting is rejected.

### Nine lowest-error layers

To replace the nine layers with the lowest batch128 A1 image-relative-MSE (mean
of steps451–500), use zero-based `[2,4,7,8,9,24,25,26,27]`. All other19 joint
layers stay Full Attention. The architecture and training settings match the
four-layer Halo2 experiment above, with a separate output root and original
FLUX.3 base initialization. This starts A1 only (500steps, per-GPU batch2,
accumulation1, student-only validation every100steps):

```bash
conda activate flux3-qdm
GPU_IDS=0,1,2,3 CPU_THREADS=1 bash flux3/scripts/train_flux3_lowmse9_halo.sh
bash flux3/scripts/train_flux3_lowmse9_halo.sh --status
bash flux3/scripts/train_flux3_lowmse9_halo.sh --stop
```

Outputs: `outputs/flux3-qdm-lowmse9-halo2-node01`. Set `FLUX3_LOW9_RUN` to override.
The next stages remain available using `--stage a2` and `--stage b` after reviewing
A1; they are not started by the command above.

### Local Softmax + Halo2 only: direct B-stage ablation

Use the same nine zero-based layers `[2,4,7,8,9,24,25,26,27]`, with the other19
joint layers remaining original Full Attention. This variant **does not construct
QDM or any remote-memory module**. Image queries attend to their8×8 region with
Halo2 K/V and all text; text queries keep global attention. Original RoPE and
QK normalization remain in the Softmax path.

Start directly from original FLUX.3 base weights, with zero-initialized rank64,
alpha64 Q/K/V/O LoRA and trainable Softmax gates on these nine layers. Original
weights stay frozen, and teacher forward bypasses all LoRA while using original
Full Attention. The loss is B-stage final-velocity prediction MSE; FM is logged
only. No A1/A2 checkpoint is required or loaded.

The diagnostic run keeps the previous budget: **500 B steps**, per-GPU batch2,
accumulation1, four GPUs (effective batch8), the existing **20K 2K cache**, and
student-only10-prompt validation plus saving every100steps. B-stage optimizer,
LR multipliers and200-step warmup are retained. The controller saves and renews
processes about every20minutes and stops after B finishes.

```bash
conda activate flux3-qdm
GPU_IDS=0,1,2,3 CPU_THREADS=1 bash flux3/scripts/train_flux3_lowmse9_local_b.sh
bash flux3/scripts/train_flux3_lowmse9_local_b.sh --status
bash flux3/scripts/train_flux3_lowmse9_local_b.sh --stop
```

Output: `outputs/flux3-local-lowmse9-halo2-direct-b-node01/b`. Set
`FLUX3_LOW9_LOCAL_RUN` to override. Repeating the same command resumes optimizer,
scheduler, RNG and data position from a completed checkpoint. The general
training entry supports `--from-base` for an explicit single stage and
`--training-subset-size 20000|50000`; without these flags, existing staged
initialization and data defaults are unchanged. A1/A2 local-only runs with frozen
projections/gates have no trainable parameters and are rejected.

### Nine-layer QDM-only image ablation

`--attention-mode memory-only` removes image Local Softmax entirely from joint
layers `2,4,7,8,9,24,25,26,27`. Their image outputs come only from the existing
fixed-four QDM: 5×5 K/V DWConv, SiLU, Q/K L2 normalization, FP32 joint solve,
output normalization/gate and independent projection, without remote RoPE.
Text queries keep sample-isolated global image/text Softmax with native RoPE and
the text output gate. The other 19 joint layers and early independent blocks
remain Full Attention. No image Halo or local-window operation runs in these
nine layers; the serialized Halo value is retained only for configuration
compatibility.

This B-only diagnostic starts from original base weights, with freshly
initialized QDM and zero-increment rank64/alpha64 Q/K/V/O LoRA. It does not load
the local-only run. It uses the existing 20K 2K cache, per-GPU batch2,
accumulation1, 500 steps, and saving/student-only validation every100steps.
On three GPUs the effective batch is6. QDM, LoRA and text output gates are
trainable; original weights stay frozen. Only teacher-prediction MSE is
optimized. SwanLab defaults to offline mode.

```bash
conda activate flux3-qdm
GPU_IDS=1,2,3 CPU_THREADS=1 bash flux3/scripts/train_flux3_lowmse9_memory_b.sh
bash flux3/scripts/train_flux3_lowmse9_memory_b.sh --status
bash flux3/scripts/train_flux3_lowmse9_memory_b.sh --stop
```

Output: `outputs/flux3-memory-lowmse9-direct-b-node01/b`. Override with
`FLUX3_LOW9_MEMORY_RUN`. The repeat controller saves and renews its processes
about every20minutes, resumes completed checkpoints in this same output, and
stops after B. Manual stop disables further restarts.

### All-28-layer Local Softmax + Halo2 ablation

This run replaces all 28 joint blocks with image Local Softmax + Halo2 and
retains global text-query attention. No QDM parameters are allocated. Early
independent mode blocks remain native. It starts B from original base weights
with zero-increment rank64/alpha64 Q/K/V/O LoRA and trainable output gates;
it does not inherit the nine-layer runs.

```bash
conda activate flux3-qdm
GPU_IDS=0,1,2,3 CPU_THREADS=1 bash flux3/scripts/train_flux3_all_local_b.sh
bash flux3/scripts/train_flux3_all_local_b.sh --status
bash flux3/scripts/train_flux3_all_local_b.sh --stop
```

Defaults: 500 B steps, per-GPU batch2, accumulation1 (effective batch8 on four
GPUs), the existing 20K 2K cache, saving and student-only validation every100steps,
and automatic20-minute process renewal. Output is
`outputs/flux3-local-all28-halo2-direct-b-node01/b`; override with
`FLUX3_ALL_LOCAL_RUN`.

### Nine-layer QDM with shared text writes

`--memory-text-write` adds the sample's valid text K/V to every spatial memory
bank. For bank `c`, statistics are `A_c = A_image,c + A_text` and
`B_c = B_image,c + B_text`, followed by the existing FP32 solve
`S_c = B_c (I + A_c)^-1`. Image queries still read their own bank. Text statistics
are computed once and reused; each bank receives all text tokens once, without
dividing text weights by four. This is a joint image/text solve, not a persistent
text state carried between diffusion steps.

Text uses the shared original K/V projections, SiLU, K L2 normalization and the
existing beta projection. Text gets no spatial convolution or RoPE in memory;
image feature processing and all memory gates/projections are unchanged. Text
query outputs retain the global Softmax path with native RoPE. Packed sample
bounds exclude padding and isolate samples. No parameters are added relative
to memory-only. The option defaults off, and checkpoint resume, stage loading
and inference reject incompatible text-write settings.

```bash
conda activate flux3-qdm
GPU_IDS=1,2,3 CPU_THREADS=1 bash flux3/scripts/train_flux3_lowmse9_text_memory_b.sh --allow-batch-change
bash flux3/scripts/train_flux3_lowmse9_text_memory_b.sh --status
bash flux3/scripts/train_flux3_lowmse9_text_memory_b.sh --stop
```

This controlled run replaces joint layers `2,4,7,8,9,24,25,26,27` with QDM-only
image attention plus text writes; no image Local Softmax runs. The other19 joint
layers stay Full Attention. It starts B from original weights (fresh QDM,
zero-increment rank64/alpha64 LoRA), uses the same 20K 2K cache, per-GPU batch2,
accumulation1, 500 steps, and saving/student-only validation every100steps. On
three GPUs the effective batch is6. Automatic20-minute checkpoint continuation
uses `outputs/flux3-memory-text-lowmse9-direct-b-node01/b`, overridden by
`FLUX3_LOW9_TEXT_MEMORY_RUN`.

## 服务器重启后自动恢复

已有`plan.json`的14组串行实验和8卡四队列/联合训练都支持此监控。需要服务器提供用户`crontab`及运行中的cron服务，不需要sudo或用户systemd。先启动一次对应训练入口，再安装：

```bash
conda activate flux3-qdm
bash flux3/scripts/watch_flux3_resume.sh --install \
  --output-root outputs/flux3-pairs-parallel-then-joint --min-free-mib 32768
```

串行14组实验则将输出目录改为`outputs/flux3-pairwise-hybrid-50`。安装会保留其他cron任务，新增开机检查和每分钟检查；读取已有计划中的GPU、数据路径及训练参数，不重新猜测配置。只在安装时的主机恢复该目录。

驱动或显存尚未就绪时继续等待；已有控制器、训练或验证进程时不重复启动。手动`--stop`、失败状态或全部完成不会被自动拉起。恢复从同一目录的完整checkpoint及已保存验证进度继续。监控本身不加载模型，不占GPU显存。

```bash
conda activate flux3-qdm
bash flux3/scripts/watch_flux3_resume.sh --status \
  --output-root outputs/flux3-pairs-parallel-then-joint

# 查看监控日志。
tail -f outputs/flux3-pairs-parallel-then-joint/repeat/reboot-watch/watch.log

# 只移除重启恢复监控，不停止正在运行的训练。
bash flux3/scripts/watch_flux3_resume.sh --remove \
  --output-root outputs/flux3-pairs-parallel-then-joint
```

连续三次启动异常后监控暂停尝试并记录原因；排查后重新安装监控可重新启用。实际重启恢复取决于开机后cron、代码/数据所在文件系统及GPU驱动恢复可用。

## 从14组已训练层合并初始化，做三阶段短程联合训练

可用`flux3_qdm.merge_pairs`按层合并14组已完成验证的B checkpoint，再用`--a1-init-from`只初始化A1。A2/B自动加载本输出目录内上一阶段的结果。`--retain-lora`保留已有适配器，在A1/A2冻结，在B解冻。

```bash
conda activate flux3-qdm
export FLUX3_MERGED_RUN="$PWD/outputs/flux3-merged28-100steps"
PYTHONPATH="$PWD/flux3:$PWD/flux3/DiffSynth-Studio" \
python -m flux3_qdm.merge_pairs \
  --group-roots "$PWD/outputs/flux3-pairwise-hybrid-50" \
  --output-root "$FLUX3_MERGED_RUN/merged"

FLUX3_MERGED_CHECKPOINT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$FLUX3_MERGED_RUN/merged/ready.json")"
GPU_IDS=1,2,3 CPU_THREADS=1 bash flux3/scripts/train_flux3_qdm_repeat.sh \
  --stage all --output-root "$FLUX3_MERGED_RUN" \
  --weights-root "$FLUX3_WEIGHTS_ROOT" \
  --subset-root "$FLUX3_SUBSET_ROOT" --cache-root "$FLUX3_CACHE_ROOT" \
  --a1-init-from "$FLUX3_MERGED_CHECKPOINT" --retain-lora \
  --attention-mode hybrid --region-size 8 --halo 2 \
  --training-subset-size 20000 --steps 100 --warmup-steps 10 \
  --batch-size 2 --grad-accum 1 --save-steps 100 --validation-steps 100 \
  --workers 4 --threads 1 --resume auto --cycle-seconds 1200
```

以上是独立的100/100/100步联合诊断配置，三个阶段均使用现有20K缓存。每阶段结束生成10张学生2K图，20分钟周期退出时额外保存恢复状态。也可使用`watch_flux3_resume.sh --install --output-root "$FLUX3_MERGED_RUN"`安装开机恢复监控。
