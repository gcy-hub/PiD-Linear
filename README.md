# Linear-PiD

本分支用于 **Spatial Error Diagnosis**：在不改变模型的情况下，诊断空间误差集中度与 latent 条件可预测性。运行命令和结果判读见 [空间误差诊断](docs/research/spatial-error-diagnosis.md)。

基于 [PiD](https://huggingface.co/nvidia/PiD) 的 KDA 改造与恢复训练。下面以 **10 层 KDA、4 卡、每卡 batch 4** 为例，介绍从下载到推理的完整流程。所有命令均在仓库根目录执行，路径和 GPU 编号可自行修改。

模型从未蒸馏的 PiD v1.5 FLUX 初始化，替换层为 `[0,1,2,4,5,6,8,9,10,12]`；其余 4 层 MMDiT Attention 和 2 层 PiT 保留。训练后的 KDA 权重由下面的训练阶段生成。

## 1. 资源下载

先安装独立的 Python 3.12／PyTorch 2.10 环境及固定版本的 FLA，然后下载模型和 MultiAspect-4K-1M 数据集。需要已经安装 Conda，并有支持 CUDA 的 GPU。下载 Gemma 前，请使用同一 Hugging Face 账号在 [Gemma 页面](https://huggingface.co/google/gemma-2-2b-it)接受使用条款。

下载脚本依次下载 PiD 未蒸馏权重、FLUX VAE、Gemma、Z-Image-Turbo、数据集元数据和原始图片，使用官方 Hugging Face 地址。结果保存在 `weights/` 和 `raw_data/MultiAspect-4K-1M/`。中断后重复下载命令即可续做。

| 下载内容 | 硬盘空间 | 训练中的作用 | 推理中的作用 |
| --- | --- | --- | --- |
| [PiD v1.5 FLUX 未蒸馏权重](https://huggingface.co/nvidia/PiD) | 2.61 GiB | 初始化学生；可选教师监督，当前 FM-only 配置不加载独立教师 | 原始 PiD 对比基线；训练后的 KDA 使用自己的 checkpoint |
| [FLUX VAE：`ae.safetensors`](https://huggingface.co/nvidia/PiD) | 0.31 GiB | 将下采样图像编码为条件 latent，参数冻结 | 图像条件推理时编码输入图像 |
| [Gemma 2 2B IT](https://huggingface.co/google/gemma-2-2b-it) | 4.89 GiB | 编码 caption；缓存完成后训练无需加载 Gemma | 编码用户输入的 prompt |
| [Z-Image-Turbo 完整管线](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | 30.59 GiB | 提前生成固定验证条件，不参与训练梯度计算 | 将 prompt 生成 latent，再交给 PiD／KDA 解码 |
| [MultiAspect-4K-1M 元数据](https://huggingface.co/Owen777/UltraFlux-v1/tree/main) | 压缩包 0.48 GiB；解压并补充字段后约 1.7 GiB | 提供图片 URL、caption，构建索引和文本缓存 | 纯 prompt 推理不需要 |
| MultiAspect-4K-1M 原始图片 | 约 3.3 TiB | 提供训练图像及真实图像验证条件 | 纯 prompt 推理不需要；图像条件推理可使用 |

四个模型合计约 **38.40 GiB**。图片及元数据解压后的空间参考约 1M 样本的实测，会随实际下载数量变化。

```bash
# 首次使用时安装环境；安装结束包含一次小规模 KDA CUDA 检查
bash scripts/setup_linear_pid_env.sh
conda activate linear-pid
HF_ENDPOINT=https://huggingface.co hf auth login

WEIGHTS_ROOT=./weights \
DATASET_ROOT=./raw_data/MultiAspect-4K-1M \
MODEL_WORKERS=4 IMAGE_WORKERS=8 \
bash download/download_all.sh
```

各组件的单独下载命令见 [下载说明](download/README.md)。

## 2. 训练准备

构建有效样本索引、固定训练／验证划分，使用 Gemma 缓存全部 caption，再准备真实图像及 Z-Image-Turbo 生成的固定验证条件。索引检查图片头和尺寸，训练时执行完整图片解码。

结果为数据集目录下的 `linear_pid_index/`、`linear_pid_text_cache/`，以及 `outputs/linear-pid/assets/`。文本缓存使训练无需加载 Gemma；准备过程中断后，重复下面的命令继续。

| 准备操作 | 生成结果 | 硬盘空间 | 作用 |
| --- | --- | --- | --- |
| 检查样本并构建索引 | 数据集目录下的 `linear_pid_index/` | 约 1.8 GiB | 保存有效样本、分辨率桶、训练／验证划分及读取偏移 |
| 缓存全部 caption | 数据集目录下的 `linear_pid_text_cache/` | 约 1.3 TiB | 训练直接读取 Gemma 编码结果，省去文本编码前向计算 |
| 准备固定验证条件 | `outputs/linear-pid/assets/` | 约 0.42 GiB | 保存真实图像及生成条件的 latent、文本编码、固定 seed 和条件预览图，用于 2K／4K 验证 |

这些文件由准备脚本在本地生成，无需另外下载。空间参考约 1M 样本和默认验证数量的实测；加上原始图片及模型，合计约 **4.6 TiB**，不含后续训练 checkpoint 和生成图。

```bash
conda activate linear-pid

WEIGHTS_ROOT=./weights \
DATASET_ROOT=./raw_data/MultiAspect-4K-1M \
GALLERY_ROOT=./outputs/linear-pid/assets \
GPU_IDS=0,1,2,3 INDEX_WORKERS=8 TEXT_WORKERS=2 CPU_THREADS=1 TEXT_BATCH_SIZE=8 \
bash download/prepare_training.sh
```

下载和准备完成后，保持数据集及模型的位置固定。该脚本完成准备后退出，不会启动训练。

也可以跳过文本缓存，在训练时加载 Gemma 实时编码 caption。数据索引仍然必需：已有索引可直接复用；没有索引时，只执行下面的命令，然后使用下一节的在线 Gemma 训练命令。

```bash
conda activate linear-pid

python scripts/prepare_linear_pid_data.py \
    --dataset-root ./raw_data/MultiAspect-4K-1M \
    --output-root ./raw_data/MultiAspect-4K-1M/linear_pid_index \
    --workers 8 --validation-size 1024 --seed 42 \
    --image-verification header --resume
```

如果同时跳过固定验证条件的准备，需要关闭训练中的生成验证。在线 Gemma 且关闭生成验证时，训练只需要 PiD 初始化权重、FLUX VAE、Gemma、原始图片和数据索引；Z-Image-Turbo 用于生成验证条件或后续纯 prompt 推理。

## 3. 正式训练

首次运行加载原始 PiD，随机初始化 10 层 KDA，并微调整个可训练像素主干。使用 FM loss、冻结 VAE，默认读取磁盘文本缓存；不创建教师网络或 EMA。下面配置的有效 batch 为 `4 卡 × 4 × 1 = 16`，KDA／继承主干的默认学习率分别为 `1e-4`／`1e-5`。

checkpoint、验证图片和 SwanLab 离线日志统一保存在 `outputs/linear-pid-kda10/kda_0-1-2-4-5-6-8-9-10-12/`。相同命令重复运行会恢复最近完整 checkpoint，包括优化器和训练进度，不会重新初始化 KDA。

```bash
conda activate linear-pid

GPU_IDS=0,1,2,3 CPU_THREADS=1 \
FLA_DISABLE_TENSOR_CACHE=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
bash scripts/train_linear_pid.sh \
    --preset 4gpu --layers 10 --lambda-out 0 \
    --weights-root ./weights \
    --index-root ./raw_data/MultiAspect-4K-1M/linear_pid_index \
    --text-cache-root ./raw_data/MultiAspect-4K-1M/linear_pid_text_cache \
    --gallery-root ./outputs/linear-pid/assets \
    --output-root ./outputs/linear-pid-kda10 \
    --batch-size 4 --grad-accum 1 --workers 4 --threads 1 \
    --pit-chunk-size 2048 \
    --save-steps 5000 --save-epochs 1 \
    --validation-steps 5000 --validation-epochs 1 \
    --swanlab-mode offline --resume auto \
    --max-steps 0 --max-seconds 0
```

默认每 **5,000 个 optimizer steps 或 1 epoch** 保存并运行完整验证。任一条件满足即触发，同时满足只执行一次；epoch 按累计成功训练样本数除以有效训练集大小计算。验证图在运行目录的 `galleries/`，SwanLab 数据在 `swanlog/`。需要实时上传时，先执行 `swanlab login`，再将 `--swanlab-mode offline` 改为 `cloud`。

| 周期设置 | 保存参数 | 验证参数 |
| --- | --- | --- |
| 仅按 step | `--save-steps 5000 --save-epochs 0` | `--validation-steps 5000 --validation-epochs 0` |
| 仅按 epoch | `--save-steps 0 --save-epochs 1` | `--validation-steps 0 --validation-epochs 1` |
| step 和 epoch 同时生效（默认） | `--save-steps 5000 --save-epochs 1` | `--validation-steps 5000 --validation-epochs 1` |

周期参数设为 `0` 表示关闭该条件。保存和验证周期可以分别设置；每次验证前也会保存对应 checkpoint。中断的验证会在续训时补完，已生成的样本会跳过。

该命令持续训练至手动停止，不安装自动重启服务。中断后保持输出目录和 batch 配置不变，重新运行即可续训。要启动独立的新实验，请使用新的 `--output-root`；要改为首轮 4 层实验，可将 `--layers 10` 改为 `--layers 4`。

**在线 Gemma 训练（无需文本缓存）**

使用 `--online-text` 时，每个 GPU 上的训练进程加载冻结的 Gemma，实时编码当前 batch 的 caption。Gemma 始终驻留 GPU，只做无梯度前向；这样省去文本缓存的生成和磁盘空间，但增加显存占用和每步文本编码耗时。

下面命令复用数据索引，关闭生成验证，无需提前准备文本缓存和固定验证条件。模型仍每 5,000 steps 或 1 epoch 保存一次，并记录 SwanLab 日志；输出使用独立目录，与上面的缓存训练示例分开。

```bash
conda activate linear-pid

GPU_IDS=0,1,2,3 CPU_THREADS=1 \
FLA_DISABLE_TENSOR_CACHE=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
bash scripts/train_linear_pid.sh \
    --preset 4gpu --layers 10 --lambda-out 0 \
    --weights-root ./weights \
    --index-root ./raw_data/MultiAspect-4K-1M/linear_pid_index \
    --online-text \
    --output-root ./outputs/linear-pid-kda10-online \
    --batch-size 4 --grad-accum 1 --workers 4 --threads 1 \
    --pit-chunk-size 2048 \
    --save-steps 5000 --save-epochs 1 \
    --validation-steps 0 --validation-epochs 0 \
    --swanlab-mode offline --resume auto \
    --max-steps 0 --max-seconds 0
```

在线 Gemma 也支持生成验证：准备好固定验证条件后，添加 `--gallery-root ./outputs/linear-pid/assets`，并将验证周期恢复为 `--validation-steps 5000 --validation-epochs 1`。续训时保持相同的在线／缓存模式和输出目录；使用在线示例训练的模型进行推理时，将下一节的 `RUN_DIR` 改为 `./outputs/linear-pid-kda10-online/kda_0-1-2-4-5-6-8-9-10-12`。

## 4. 推理

输入 prompt，先由 Z-Image-Turbo 生成 latent，再由原始 PiD 和训练后的 KDA 分别生成像素图像。两者共享相同 latent、文本条件和像素噪声 seed，在同一 GPU 上依次运行。

下面命令自动选择运行目录中的最近完整 checkpoint，分别生成两个模型的 2K 图像，并保存 `original/`、`trained/`、`summary.json` 和 `summary.csv`。耗时统计中的 `PiD sampling` 只包含 PiD／KDA 解码，不包含前面的 Z-Image-Turbo 生成、文本编码或模型加载。

```bash
conda activate linear-pid

RUN_DIR=./outputs/linear-pid-kda10/kda_0-1-2-4-5-6-8-9-10-12
RESOLUTION=2048

GPU_IDS=0 CPU_THREADS=1 \
bash scripts/compare_linear_pid_inference.sh \
    --checkpoint "$RUN_DIR" --baseline original \
    --weights-root ./weights \
    --prompt "A corgi sitting in a sunlit garden, detailed fur, natural colors." \
    --height "$RESOLUTION" --width "$RESOLUTION" --seed 42 \
    --steps 25 --cfg 5 --shift 6 \
    --repeats 1 --sample-warmup 0 \
    --network-repeats 3 --network-warmup 1 \
    --threads 1 --workers 2 \
    --output-dir "./outputs/inference/${RESOLUTION}"
```

生成 4K 时，将 `RESOLUTION=2048` 改为 `RESOLUTION=4096`，重新执行该命令。更换 prompt、seed 或 checkpoint 时，也请更换输出目录，以免与已保存的结果冲突。只做推理不需要下载训练图片或构建训练文本缓存，但需要训练后的 checkpoint 和对应模型权重。

本项目沿用原 PiD 的 [Apache 2.0 许可证](LICENSE)；各模型和数据集按其各自的许可使用。
