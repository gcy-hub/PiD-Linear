# PiD-VDN

将原始 PiD v1.5 FLUX 未蒸馏模型的 **14 层 MMDiT** 改为 **二维区域 Softmax + 双向 VDA**，两层 PiT 保持原样。局部 Softmax 的图像、文本 Q/K 和 PiT **保留 RoPE**，VDA 分支 **不使用 RoPE**。

默认区域为 `8×8 token`。VDA 使用区域联合更新 `vdn_solve`、正反扫描、5×5 图像 K/V depthwise convolution 和有效文本初始记忆；移植来源及许可见 [VDN attribution](pid/_src/vdn_pid/vendor/NOTICE.md)。第一版只使用数据并行。

## 1. 资源下载

下载原始 PiD、FLUX VAE、Gemma、Z-Image-Turbo 和 MultiAspect-4K-1M。PiD 提供共享的冻结主干，VAE 编码训练图像条件，Gemma 编码 caption，Z-Image-Turbo 提前生成验证和纯 prompt 推理条件。接受 Gemma 的 Hugging Face 使用条款后执行：

```bash
bash scripts/setup_linear_pid_env.sh
conda activate linear-pid
HF_ENDPOINT=https://huggingface.co hf auth login

WEIGHTS_ROOT=./weights DATASET_ROOT=./raw_data/MultiAspect-4K-1M \
MODEL_WORKERS=4 IMAGE_WORKERS=8 bash download/download_all.sh
```

已有本地模型和数据可直接复用。单独下载命令见 [下载说明](download/README.md)。

| 内容 | 参考硬盘空间 | 用途 |
| --- | ---: | --- |
| PiD v1.5 FLUX 未蒸馏权重 | 2.61 GiB | 初始化学生和提供固定教师计算路径 |
| FLUX VAE | 0.31 GiB | 编码下采样图像条件 |
| Gemma 2 2B IT | 4.89 GiB | 文本缓存、在线 caption/prompt 编码 |
| Z-Image-Turbo | 30.59 GiB | 准备生成验证条件和纯 prompt 推理 latent |
| 数据元数据 | 解压后约 1.7 GiB | 图片路径、caption 和索引 |
| 原始图片 | 约 3.3 TiB | 训练图像与真实图像验证条件 |

空间是约 1M 样本的参考值；无需重新下载已有资源。

## 2. 训练准备

先构建有效数据索引、文本缓存和固定验证条件。图片不复制，训练使用原分辨率图片经 Lanczos 缩放和中心裁剪后的 **2K 多长宽比桶**。

```bash
conda activate linear-pid

WEIGHTS_ROOT=./weights DATASET_ROOT=./raw_data/MultiAspect-4K-1M \
GALLERY_ROOT=./outputs/linear-pid/assets \
GPU_IDS=0,1,2,3 INDEX_WORKERS=4 TEXT_WORKERS=2 CPU_THREADS=1 TEXT_BATCH_SIZE=8 \
bash download/prepare_training.sh
```

| 准备结果 | 参考空间 | 作用 |
| --- | ---: | --- |
| `linear_pid_index/` | 约 1.8 GiB | 有效样本、训练/验证划分、分桶、mmap 偏移 |
| `linear_pid_text_cache/` | 约 1.3 TiB | 训练直接读取 Gemma 编码，避免常驻 Gemma |
| `outputs/linear-pid/assets/` | 约 0.42 GiB | 固定 2K/4K latent、caption、mask 和 seed |
| `pid_vdn_subsets/` | 少量索引文件 | 嵌套 20K/50K 子集；不复制图片或文本缓存 |

设置训练和推理共用的路径，再生成子集。已有索引和缓存时，从这里开始即可。子集按长宽比分层抽取，20K 包含于 50K；排除验证集合，保存样本 ID，并记录人物、文字、纹理 caption 关键词分布。重复命令可续做。

```bash
conda activate linear-pid

export PID_WEIGHTS_ROOT=./weights
export PID_INDEX_ROOT=./raw_data/MultiAspect-4K-1M/linear_pid_index
export PID_TEXT_CACHE_ROOT=./raw_data/MultiAspect-4K-1M/linear_pid_text_cache
export PID_VDN_SUBSET_ROOT=./raw_data/MultiAspect-4K-1M/pid_vdn_subsets
export PID_GALLERY_ROOT=./outputs/linear-pid/assets

python -m scripts.prepare_pid_vdn_data --workers 4 --seed 42 --resume
```

可以不生成文本缓存：将 `PID_TEXT_CACHE_ROOT` 设为空字符串，训练会加载 Gemma 实时编码。数据索引和训练子集仍然必需。没有固定验证条件时，显式添加 `--validation-steps 0` 可关闭生成验证。

## 3. 性能测试与正式训练

### 训练前性能测试

先比较 1K/2K/4K 下原始 Full Attention、局部 Softmax、VDA、完整 Hybrid 的速度和显存，分别测试 `4×4、8×8、16×16` 区域。默认推理重复20次，前向+反向重复5次。每项保存完成标记，重复命令可跳过已完成结果。

```bash
conda activate linear-pid

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
python -m scripts.benchmark_pid_vdn \
  --regions 4 8 16 --resolutions 1024 2048 4096 \
  --output-root outputs/pid-vdn/benchmark

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
python -m scripts.benchmark_pid_vdn_network \
  --region-size 8 --resolutions 2048 4096 \
  --output-root outputs/pid-vdn/network-benchmark
```

第二条命令先加载原始 PiD、再加载未训练的 Hybrid，选择固定画廊中的 2048²／4096² 正方形条件，顺序生成同一条件下的图像，记录网络单步、MMDiT、PiT、25步解码时间和峰值显存。未训练生成结果只用于性能测试，不能据此评价训练质量。区域大小不会根据测速自动修改。

### 三阶段训练

教师和学生共享一套冻结主干。教师执行原始 Full Attention，学生执行 Hybrid；B阶段教师禁用LoRA。**不常驻第二套教师、不使用CPU offload、不启用EMA。** A2/B启用激活checkpointing。三个阶段均使用蒸馏目标，FM loss只记录。

| 阶段 | 更新次数 | 数据 | 有效batch | 可训练部分 |
| --- | ---: | ---: | ---: | --- |
| A1 | 500 | 20K | 8 | 新VDA参数；逐层Attention输出对齐 |
| A2 | 1,000 | 同一批20K | 16 | 新VDA参数；整网预测对齐 |
| B | 10,000 | 50K | 32 | VDA、Softmax门、Q/K/V/O LoRA（rank64、alpha64） |

A1一次教师前向同时对齐全部14层，每层重算学生Attention并立即反传，主干继续使用教师输出。A1总计500次更新，不乘以层数。FFN、AdaLN、PiT和latent注入参数全程冻结。Softmax门A1/A2固定0.99，B解冻。

4卡每卡bs为1/2/4，均累积2次；8卡每卡bs为1/2/4，均累积1次。单卡或2卡自动增加累积，保持阶段有效batch。可显式设置 `--batch-size`、`--grad-accum`、`--effective-batch`，三者必须一致，不自动降低batch。

```bash
conda activate linear-pid

GPU_IDS=0,1,2,3 CPU_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
bash scripts/train_pid_vdn.sh \
  --preset 4gpu --stage all --region-size 8 --resume auto \
  --workers 4 --threads 1 --output-root outputs/pid-vdn
```

也可分阶段运行；A2/B自动寻找输出目录下的上一阶段checkpoint，或用 `--init-from CHECKPOINT` 显式指定。

```bash
conda activate linear-pid

GPU_IDS=0,1,2,3 bash scripts/train_pid_vdn.sh --stage a1 --resume auto
GPU_IDS=0,1,2,3 bash scripts/train_pid_vdn.sh --stage a2 --resume auto
GPU_IDS=0,1,2,3 bash scripts/train_pid_vdn.sh --stage b  --resume auto
```

双节点在每个节点各运行一次以下命令，两个节点分别设置 `NODE_RANK=0` 和 `NODE_RANK=1`，`MASTER_ADDR`使用节点0的可达地址；不会自动提交Slurm。

```bash
conda activate linear-pid

NNODES=2 NODE_RANK=0 MASTER_ADDR=node0 MASTER_PORT=29583 \
GPU_IDS=0,1,2,3 CPU_THREADS=1 \
bash scripts/train_pid_vdn.sh --preset 8gpu --stage all --resume auto \
  --workers 4 --output-root outputs/pid-vdn
```

AdamW基础LR为1e-4，warmup后cosine至5e-6；A1/A2 warmup50，B warmup200。分支大/小参数倍率分别为1/5和0.4/2，B LoRA倍率1。weight decay为0，梯度裁剪A1每层0.1，A2/B全局1.0。冻结主干BF16，可训练参数及优化器FP32，计算BF16 autocast，VDA求解和扫描状态FP32。

输出统一在 `outputs/pid-vdn/a1/`、`a2/`、`b/`。A1每50步、A2每100步保存，阶段结束验证；**B每2000阶段步保存并验证**，不依赖真实时间。保留最近3个恢复状态、验证节点和阶段最终状态。重复同一命令严格恢复优化器、scheduler、随机状态及数据进度；阶段切换只重建优化器和scheduler。4/8卡切换保持有效batch及样本进度，不承诺计算逐位一致。

checkpoint保存新增分支和LoRA权重及完整训练状态，因此恢复必须保留同一原始PiD权重。发布完整标记前的目录不会被恢复。没有定时重启；进程被kill后手动重复命令续训。

SwanLab默认离线，可用 `--swanlab-mode cloud` 开启云端记录。监控重点为 `train/kd_loss`、A1逐层 `alignment/` 的图像/文本误差、FM指标、样本覆盖、LR、梯度范数、耗时和显存。生成验证使用固定条件和seed，25步、CFG5、shift6。

## 4. 推理与评估

对选定checkpoint生成固定画廊。多卡按样本分工，每卡各加载一套学生，不创建独立教师。输出包含原尺寸图像、局部裁剪、caption、采样设置及计时；重复同一命令跳过已完成图像。

```bash
conda activate linear-pid

GPU_IDS=0,1,2,3 CPU_THREADS=1 \
bash scripts/sample_pid_vdn.sh \
  --checkpoint outputs/pid-vdn/b/checkpoints/SELECTED_CHECKPOINT \
  --suite qualitative --include-4k --output-dir outputs/pid-vdn/inference
```

使用训练后的模型进行完整速度对比：

```bash
conda activate linear-pid

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
python -m scripts.benchmark_pid_vdn_network \
  --checkpoint outputs/pid-vdn/b/checkpoints/SELECTED_CHECKPOINT \
  --resolutions 2048 4096 --output-root outputs/pid-vdn/trained-benchmark
```

导出完整学生权重，随后将导出文件作为 `--checkpoint`，无需原始 PiD 基础权重即可运行固定条件推理。普通训练 checkpoint 为增量格式，需要同一原始基础权重。

```bash
conda activate linear-pid

GPU_IDS=0 bash scripts/sample_pid_vdn.sh \
  --checkpoint outputs/pid-vdn/b/checkpoints/SELECTED_CHECKPOINT \
  --export outputs/pid-vdn/student.pt --export-only

GPU_IDS=0,1 bash scripts/sample_pid_vdn.sh \
  --checkpoint outputs/pid-vdn/student.pt --suite qualitative --include-4k \
  --output-dir outputs/pid-vdn/student-inference
```

首次检查可添加 `--limit-per-resolution 1`，分别生成一张 2K 和 4K 图像。

固定画廊读取提前保存的文本编码和latent，不会重新加载Gemma或Z-Image-Turbo。纯prompt使用场景需先通过对应文本编码器和生成管线准备条件；本分支新增接口首先服务于固定条件的质量与速度评估。

对已训练 checkpoint 做空间扫描诊断：

```bash
conda activate linear-pid

GPU_IDS=0 bash scripts/sample_pid_vdn.sh \
  --checkpoint outputs/pid-vdn/student.pt --suite qualitative --limit-per-resolution 1 \
  --include-4k --diagnostics --diagnostic-layers 0 7 13 \
  --output-dir outputs/pid-vdn/spatial-diagnostics

GPU_IDS=0 bash scripts/sample_pid_vdn.sh \
  --checkpoint outputs/pid-vdn/student.pt --suite qualitative --limit-per-resolution 1 \
  --include-4k --scan-order column --output-dir outputs/pid-vdn/column-gallery
```

`*.spatial.json` 记录相同特征按行／按列扫描的输出差异、水平／垂直邻接区域的纯遗忘门衰减，以及固定 K 和门后扰动 V 的输出响应。探针使用观测 QKV 的 FP32 副本，避免微小扰动被 BF16 舍入。采集早、中、晚的 CFG 前向，不改变原始生成轨迹；开启诊断时计时包含诊断开销。第二条命令只在推理时切换扫描顺序，用于查看方向差异和区域边界；训练默认仍按行正反扫描。
