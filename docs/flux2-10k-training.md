# FLUX2 10K：目标服务器 agent 执行指令

在目标服务器的仓库根目录执行。先询问用户已下载的模型和数据路径，再使用 8 张 GPU 补齐 10,000 对图文并启动 Sliding Window Softmax + QDM 三阶段训练。A1、A2、B 的每卡 batch size 统一为 16，梯度累积为 1，全局 batch size 为 128。

## 确认已有资源

向用户询问两项路径：

- **完整 FLUX.2-klein-base-4B 模型目录**，包含 `transformer/`、`text_encoder/`、`tokenizer/`、`vae/`、`scheduler/` 和 `model_index.json`。
- **FLUX2-10K 数据目录**，包含 `images/`、`prompts.json`、`run.json`。如果用户提供的是已下载的 tar 文件，先在目标服务器解包，再使用解包后的目录。

获得答复后，将下方 `MODEL_DIR`、`DATA_ROOT` 替换为用户提供的目标服务器路径。模型和数据直接复用，不重新下载；所有相对路径以仓库根目录为基准。

```bash
export MODEL_DIR='<用户提供的完整模型目录>'
export DATA_ROOT='<用户提供的已解包数据目录>'
export EXPERIMENT_ROOT=../experiment/flux2-sliding11-wavelet-10k
export GPU_IDS=0,1,2,3,4,5,6,7
export GENERATION_WORKERS=8
export CPU_THREADS=4
```

使用目标服务器的 8 张 GPU，设备编号按实际分配填写。`GENERATION_WORKERS` 控制补生成进程数，训练使用全部指定 GPU。

## 环境与资源

使用 Python 3.12、PyTorch 2.10/CUDA 12.8。若尚未创建环境，先执行：

```bash
conda create -n flux3-qdm python=3.12 -y
```

激活环境、安装依赖，并查看目标服务器的 GPU：

```bash
conda activate flux3-qdm
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install 'swanlab>=0.10,<1' 'prodigyopt>=1.1,<2'
nvidia-smi
```

## 启动完整训练

```bash
conda activate flux3-qdm
mkdir -p "$EXPERIMENT_ROOT"
nohup bash scripts/train_flux2_10k.sh \
  --model-dir "$MODEL_DIR" \
  --data-root "$DATA_ROOT" \
  --output "$EXPERIMENT_ROOT" \
  --gpu-ids "$GPU_IDS" \
  --workers "$GENERATION_WORKERS" \
  --cpu-threads "$CPU_THREADS" \
  --batch-size 16 --accumulation 1 \
  > "$EXPERIMENT_ROOT/launcher.log" 2>&1 &
```

入口依次执行：补齐原始图文对 → A1 → A2 → B。补生成沿用原始 prompt、seed、1024×1024、50 steps、CFG 4.0、BF16、CPU 随机数和原生 full attention。已有完整样本会跳过，缺失或损坏的图像/JSON 会重新生成；实际补齐 10,000 对后直接训练。冻结的 VAE 和文本编码器在训练中逐批在线编码，配置为 `cache: {image: false, text: false, root: null}`。

训练参数来自 `configs/experiments/flux2_sliding11_wavelet_10k.yaml`，目标服务器的资源路径写入 `$EXPERIMENT_ROOT/config.yaml`。学生的 5 个 double block、20 个 single block 全部替换为 Sliding Window Softmax + QDM，kernel 为 11；教师保持原始 full attention 并冻结权重。

| 阶段 | 每卡 batch | 优化步数 | 保存/验证间隔（阶段步数） | 可训练部分 | 损失 |
|---|---:|---:|---:|---|---|
| A1 | 16 | 500 | 500 | QDM 分支 | 逐层 relative attention MSE |
| A2 | 16 | 1,000 | 1,000 | QDM 分支 | 预测蒸馏 MSE + 0.5 × wavelet |
| B | 16 | 50,000 | 5,000 | QDM、local gate、LoRA | 预测蒸馏 MSE + 0.5 × wavelet |

每次验证生成 10 张 1024×1024 图像，50 个采样步。Wavelet 在原生 latent 空间的 Haar 频带上对齐教师预测。训练使用 AdamW、FP32 优化器更新、BF16 前向、梯度检查点和优化器分片。

## 恢复与输出

中断后在仓库根目录重新设置上述变量，再执行同一启动命令。流程跳过已完成的图文对，并恢复最近完整 checkpoint 中的模型、优化器和随机数状态；续训保持 GPU 数、模型、数据和训练配置一致。

训练需要平滑停止时执行：

```bash
touch "$EXPERIMENT_ROOT/run/STOP"
```

当前优化步完成后保存并退出。确认进程退出后，恢复前移除停止文件，再执行原启动命令：

```bash
rm -f "$EXPERIMENT_ROOT/run/STOP"
```

| 输出 | 相对实验目录的路径 |
|---|---|
| 入口日志 | `launcher.log` |
| 训练日志 | `run/worker.log` |
| 训练配置 | `config.yaml` |
| 最新 checkpoint 指针 | `run/latest.json` |
| 完整训练状态 | `run/step-*/` |
| 验证图片 | `run/validation/step-*/` |
| SwanLab 离线记录 | `run/swanlog/` |
| 训练完成标记 | `run/DONE.json` |

Checkpoint 保留最近一份完整训练状态，验证图片按步数保留。SwanLab 记录训练损失、各损失项、学习率、阶段、峰值显存和验证图片。
