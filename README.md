# PiD：固定空间四象限记忆

分支：`codex/spatially-disambiguated-delta-memory`。

将 PiD 的 **14 层 MMDiT Attention** 改为 **局部 Softmax + 固定四象限记忆**，两层 PiT 保持原样。局部 Softmax 是互不重叠的 `8×8 token` 窗口，直接访问文本，保留图像与文本 Q/K 的 RoPE。记忆分支不使用 RoPE：图像 K/V 经 `5×5 DWConv`，四个象限分别联合求解记忆，每个 Q 只读所属象限；取消双向 VDA 的扫描和遗忘衰减，包含自身写入。

这是需要重新恢复训练的研究结构，测速不代表已保持质量。详见 [研究说明](docs/research/spatial-delta-memory.md)。原双向 VDA 入口仍支持 `--memory-mode vda`；本页串行脚本固定使用 `fixed4`。

## 1. 资源下载

在新服务器拉取分支并安装环境。下文均在仓库根目录执行，路径可自行调整。

```bash
git clone --branch codex/spatially-disambiguated-delta-memory \
  https://github.com/gcy-hub/PiD-Linear.git
cd PiD-Linear
bash scripts/setup_linear_pid_env.sh
conda activate linear-pid

# Gemma 需要先在 Hugging Face 接受模型条款。
HF_ENDPOINT=https://huggingface.co hf auth login
python download/download_models.py --models pid vae gemma \
  --weights-root ./weights --workers 4
```

| 资源 | 参考空间 | 用途 |
| --- | ---: | --- |
| PiD v1.5 FLUX 未蒸馏权重 | 2.61 GiB | 原始初始化、教师路径、增量 checkpoint 重建 |
| FLUX VAE | 0.31 GiB | 编码训练图像的低分辨率条件 |
| Gemma 2 2B IT | 4.89 GiB | 准备文本缓存，或训练时在线编码 |
| `pid-vdn-50k.tar` | 约 166 GiB | 50K 图片与 caption，内含 A1/A2 的 20K 名单 |
| 固定验证条件包 | 约 334 MiB | 已编码的验证文本、latent、mask、seed 与来源索引信息 |
| 可选 50K 文本缓存 | 约 64.4 GiB | 避免训练时常驻 Gemma |

图片包解压后还需约一份同等大小的空间。包内没有训练索引、文本缓存、模型或验证条件。已有模型可直接设置 `PID_WEIGHTS_ROOT`。复用验证条件时不需要 Z-Image-Turbo；重新从 prompt 生成 latent 才需要它，下载方法见 [下载说明](download/README.md)。

## 2. 训练准备

### 原服务器：导出固定验证条件

只需导出一次，再把这个小包传到新服务器。验证条件已经编码，不需要搬原始验证图片，也不需要重新运行 Z-Image-Turbo。

```bash
conda activate linear-pid
python -m pid._src.vdn_pid.portable_data export-validation \
  --index-root "$PID_INDEX_ROOT" --gallery-root "$PID_GALLERY_ROOT" \
  --output ./pid-vdn-validation.tar
```

这里的两个环境变量指向原服务器已有索引和验证条件目录。导出会检查来源指纹与条件文件哈希。

### 新服务器：解压、建立训练索引

填写两个包的位置。导入不复制图片、不重新抽样，保留原 20K/50K 成员关系，生成 loader 可读的 manifest、mmap 偏移与桶索引。

```bash
conda activate linear-pid
export DATA_ARCHIVE=./pid-vdn-50k.tar
export VALIDATION_ARCHIVE=./pid-vdn-validation.tar
mkdir -p dataset runtime/pid-vdn-validation
tar -xf "$DATA_ARCHIVE" -C dataset
tar -xf "$VALIDATION_ARCHIVE" -C runtime/pid-vdn-validation

export PID_DATA_ROOT="$PWD/dataset/pid-vdn-50k"
export PID_TRAINING_ROOT="$PID_DATA_ROOT/training"
python -m pid._src.vdn_pid.portable_data import \
  --package-root "$PID_DATA_ROOT" --output-root "$PID_TRAINING_ROOT" \
  --source-index runtime/pid-vdn-validation/source_index.json \
  --workers 4 --resume

export PID_WEIGHTS_ROOT="$PWD/weights"
export PID_INDEX_ROOT="$PID_TRAINING_ROOT/index"
export PID_VDN_SUBSET_ROOT="$PID_TRAINING_ROOT/subsets"
export PID_GALLERY_ROOT="$PWD/runtime/pid-vdn-validation/gallery"
export PID_TEXT_CACHE_ROOT=""
```

导入检查图片路径、ID、caption、桶与成员关系；完整解码由训练 loader 执行。`--resume` 跳过已完成且输入一致的结果。来源索引必须能够复现包内子集指纹，不能随意配对其他数据。

| 结果 | 作用 |
| --- | --- |
| `training/index/` | 50K manifest、偏移、分桶；图片路径相对于包位置解析 |
| `training/subsets/20000/` | A1/A2 共用的原 20K 成员 |
| `training/subsets/50000/` | B 使用的全部 50K 成员 |
| `training/prompts/data_jsons/` | 保留原 ID 的 caption 元数据，供缓存工具读取 |
| `runtime/pid-vdn-validation/gallery/` | 固定 2K/4K 条件，不依赖旧服务器的图片路径 |

### 可选：提前缓存 50K 文本

不执行也能训练：保持 `PID_TEXT_CACHE_ROOT=""`，训练时在线加载 Gemma。缓存可减少训练显存与文本编码开销；示例使用单节点 8 卡，可调整 GPU 列表和进程数。

```bash
conda activate linear-pid
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 -m scripts.prepare_linear_pid_text_cache \
  --dataset-root "$PID_TRAINING_ROOT/prompts" \
  --output-root "$PID_TRAINING_ROOT/text_cache" \
  --weights-root "$PID_WEIGHTS_ROOT" --batch-size 8 \
  --workers 2 --threads 1 --max-seconds 0 --resume
export PID_TEXT_CACHE_ROOT="$PID_TRAINING_ROOT/text_cache"
```

中断后重复命令续做；也可以搬完整缓存后直接设置目录。每次新终端启动训练前，需重新设置上述路径环境变量。

## 3. 正式训练：串行 A1 → A2 → B

三阶段共享冻结的原始 PiD 主干执行教师计算，不常驻第二套完整教师，不使用 CPU offload 或 EMA。A1 逐层对齐 Attention 输出，A2 对齐整网预测，B 增加 Q/K/V/O LoRA 并解冻 Softmax Gate。四象限未使用的衰减门冻结。

| 阶段 | 每卡 batch | GPU 总数 | 累积 | 有效 batch | 阶段总 step | 保存、验证周期 | 数据 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A1 | 2 | 8 | 1 | 16 | 1,000 | 500 | 20K |
| A2 | 4 | 8 | 1 | 32 | 4,000 | 2,000 | 同一批 20K |
| B | 8 | 8 | 2 | 128 | 20,000 | 2,000 | 50K |

合计 25,000 次更新。A1 从原始 PiD 初始化新增模块；A2 在 A1 训练与验证完成后加载其最新完整 checkpoint，B 同样加载 A2。阶段切换保留参数，重建优化器和 scheduler。

先查看三个阶段的完整配置，不占用 GPU：

```bash
conda activate linear-pid
GPU_IDS=0,1,2,3,4,5,6,7 CPU_THREADS=1 \
bash scripts/train_pid_vdn_fixed4.sh --dry-run
```

单节点 8 卡正式训练，使用新的实验目录：

```bash
conda activate linear-pid
GPU_IDS=0,1,2,3,4,5,6,7 CPU_THREADS=1 \
bash scripts/train_pid_vdn_fixed4.sh \
  --workers 2 --threads 1 --output-root outputs/pid-vdn-fixed4-8gpu
```

后台运行可用下面的命令替代上一条：

```bash
conda activate linear-pid
mkdir -p outputs/pid-vdn-fixed4-8gpu
nohup env GPU_IDS=0,1,2,3,4,5,6,7 CPU_THREADS=1 \
  bash scripts/train_pid_vdn_fixed4.sh \
  --workers 2 --threads 1 --output-root outputs/pid-vdn-fixed4-8gpu \
  > outputs/pid-vdn-fixed4-8gpu/pipeline.log 2>&1 &
```

失败或暂停不会进入后续阶段。重复同一命令，会跳过已完成阶段，恢复当前阶段的权重、优化器、scheduler、随机状态及数据位置。已完成阶段的架构、batch、累积、目标 step 或保存、验证周期不同会被拒绝，不能把旧 500-step A1 当成新 1,000-step A1。

保存和验证按 step 执行；捕获停止信号后在更新边界保存。目标 batch 不会自动缩小，尤其 B 的每卡 batch 8 需要在目标 GPU 验证显存。SwanLab 默认离线，添加 `--swanlab-mode cloud` 可在线记录。验证为 25 步、CFG 5、shift 6。

### 双节点，各 4 卡

两节点需共享相同路径的权重、数据和输出，并设置相同资源环境变量。将 `node0` 替换为节点 0 的可达地址；两边各执行一次，不要额外套 torchrun。

```bash
conda activate linear-pid
# 节点 0
NNODES=2 NODE_RANK=0 MASTER_ADDR=node0 MASTER_PORT=29583 \
GPU_IDS=0,1,2,3 CPU_THREADS=1 \
bash scripts/train_pid_vdn_fixed4.sh --workers 2 \
  --output-root outputs/pid-vdn-fixed4-8gpu

# 节点 1：在另一台节点执行
NNODES=2 NODE_RANK=1 MASTER_ADDR=node0 MASTER_PORT=29583 \
GPU_IDS=0,1,2,3 CPU_THREADS=1 \
bash scripts/train_pid_vdn_fixed4.sh --workers 2 \
  --output-root outputs/pid-vdn-fixed4-8gpu
```

导入的 50K 索引重新编号了行号，上述步骤用于按新配置开始完整实验。若搬本流程已有断点，同时搬输出目录和已导入索引；原百万行索引创建的旧断点不能直接配重编号索引。

## 4. 推理与评估

checkpoint 记录 `fixed4` 结构，推理自动重建。固定条件推理需要原始 PiD 权重和验证条件，不加载教师模型、Gemma 或 Z-Image-Turbo。

```bash
conda activate linear-pid
GPU_IDS=0,1,2,3 CPU_THREADS=1 bash scripts/sample_pid_vdn.sh \
  --checkpoint outputs/pid-vdn-fixed4-8gpu/b/checkpoints/SELECTED_CHECKPOINT \
  --weights-root "$PID_WEIGHTS_ROOT" --index-root "$PID_INDEX_ROOT" \
  --gallery-root "$PID_GALLERY_ROOT" --suite qualitative --include-4k \
  --output-dir outputs/pid-vdn-fixed4-8gpu/inference
```

输出原尺寸图像、局部裁剪、caption、seed、参数和耗时。多卡按图像分工，重复命令跳过已完成图像。增量 checkpoint 必须配合同一原始 PiD 基础权重。数据、模型、缓存、输出与本地测试均不上传 Git。
