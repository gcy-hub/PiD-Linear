# FLUX2 / FLUX3：Sliding + 标准 FLA KDA

每个图像 attention 层保留 Sliding，并加入完整的 FLA `KimiDeltaAttention` 分支。滑动窗口大小由配置项 `kernel` 或训练参数 `--kernel` 控制，默认值为 11，支持其他正奇数。FLUX2 覆盖 5 个双流层和 20 个单流层；FLUX3 覆盖 5 个早期图像层和 28 个联合层。两路输出在 DiT 的 O 投影前相加，文本保留全局 attention。该混合接入是本仓库的模型适配，不是 FLA 提供的 FLUX 模型。

KDA 使用安装版本的标准层及其默认设置：独立 Q/K/V 投影、Q/K/V 的核大小 4 的一维因果短卷积、低秩衰减与输出门控投影、融合 RMSNorm/sigmoid 门控、输出投影。使用标准初始化，每次 forward 不传缓存、不保留状态。

跨层方向循环为行正向、行反向、列正向、列反向。进入完整标准层前排列图像 token，退出后恢复坐标；卷积和递推均沿该层选定的序列方向执行。

## 环境与检查

在仓库根目录执行，使用安装了 FLA、Torch 2.10.0+cu128、Prodigy 1.1.2 的现有环境：

```bash
export PYTHONPATH="$PWD:$PWD/flux2:$PWD/flux3"
PYTHON=/home/ganchangyi/software/miniconda3/envs/flux3-qdm/bin/python
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m flux_kda.check --cuda
```

检查覆盖矩形网格四方向、与标准 FLA 层输出一致性、各参数梯度、FLUX2 两类 attention、FLUX3 checkpoint 重算与 teacher 隔离，以及小型 block 的 Prodigy 更新和参数恢复。

## CLEAR 微调

训练 DiT 图像 attention 的 Q/K/V/O 及 KDA 分支；其他参数冻结。FLUX2 fused projection 仅训练 attention 切片。FLUX3 联合层投影仍由图像与文本共享。保留原始权重作为 teacher，计算使用 BF16。FLUX2 使用 FP32 master；FLUX3 的计算参数与梯度使用 BF16，Prodigy 使用 FP32 主权重及状态，更新后同步回 BF16 参数。主权重随优化器保存，断点恢复保留未达到 BF16 表示精度的小增量。默认不启用优化器分片、activation checkpointing 或 CPU offload。

```text
loss = FM MSE + 0.5 × prediction KD MSE + 0.5 × attention feature KD MSE
```

特征取 O 投影前的合并输出。FLUX2 选择全局编号为 4 的倍数的单流层，FLUX3 选择被替换层编号为 4 的倍数的层。使用各模型原有噪声时间表，caption dropout=0。Prodigy：lr=1、betas=(0.9,0.999)、weight_decay=1e-4、常数学习率。

## 在线训练

统一入口为 `python -m flux_kda.run --config CONFIG.json`。修改同一配置中的路径、GPU 数、每卡 batch、累积次数和训练步数即可复用流程。`flux_kda/configs/h800-8gpu-4k.json` 配置为 8 卡、每卡 batch 4、累积 1，即全局 batch 32；训练 50000 步，每 2000 步保存并验证。

先把配置中的 `weights`、`resources`、`subset_root` 改为服务器实际路径。`weights` 指向 DiT 的 `model.safetensors`；`resources` 下需要 `flux-3-image-klein/text_encoder/` 的完整模型与 tokenizer/processor 文件，以及 `flux-3-image-klein/vae/diffusion_pytorch_model.safetensors`。

```bash
export PYTHONPATH="$PWD:$PWD/flux2:$PWD/flux3"
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python -m flux_kda.run --config flux_kda/configs/h800-8gpu-4k.json
```

`subset_root` 指向导出数据目录或其 `manifest.jsonl`。每行包含 `image_path`、`caption`，可选 `id`；相对图像路径相对于 manifest 所在目录解析。样本数由实际条目决定，不依赖目录名或固定子集大小。图像按 EXIF 方向读取尺寸，使用与 PiD augmentor 共享的官方分桶：最大 4096、最小 2048、256 间隔、五种比例和 64 对齐；选取原图能容纳的最大桶，不放大不足尺寸的图像。启动日志报告接受和排除的数量。小桶不足一个全局 batch 时有放回采样，其余桶无放回。

VAE 和文本编码器冻结并在线编码。编码逐样本进行，所得 latent/文本 token 仍按配置的每卡 batch 合并送入 DiT；不读写图像或文本磁盘缓存。全局 batch = `batch_size × gpus × grad_accum`。

每个训练段结束后全部训练进程退出，再启动独立验证进程；验证结束后恢复训练。验证时没有存活的 NCCL 进程组。验证默认 4096×4096、20 步、CFG 3.5、seed 0，图片在 `validation/step-N/`。若生成中断，重跑同一命令会先补齐这次验证，再进入下一段。

默认只保留 `latest.pt` 一份训练断点，包含可训练权重、FP32 主权重、优化器、scheduler 和步数。写完临时文件再原子替换，图片保留各次验证结果。已有 `latest.pt` 时自动恢复；训练总步数是累计目标。可在配置中指定 `resume` 加载其他路径。换数据集需 `allow_dataset_change: true`；改变 batch 或非分片训练的 GPU 数需 `allow_batch_change: true`。

可选显存设置均默认关闭：`shard_optimizer`、`gradient_checkpointing`、`activation_cpu_offload`、`optimizer_cpu_offload`。开启分片后，`latest.pt` 和同一步的 `optimizer-N-rank-R.pt` 共同构成最新断点，旧步的分片在新断点发布后删除；恢复要求 GPU 数和分片设置一致。配置中的 `distributed_timeout_minutes` 控制训练、加载和保存时的集合操作超时，不用于等待验证。

已测试官方分桶、数据计数、动态全局 batch、配置覆盖、验证中断恢复和断点覆盖流程。H800 8 卡、每卡 4 张 4K 的完整训练尚未实测；关闭所有显存优化时，模型、优化器、在线编码器和激活的总显存可能超过单卡容量，需要在目标机器上检查峰值。

## 已有缓存训练

FLUX2 和旧缓存训练显式选择 `--data-mode cached`，旧 node01 启动脚本保留原有分片和 checkpointing 配置。

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m flux_kda.train \
  --data-mode cached --family flux2 --weights "$HOME/huggingface_ckpts/FLUX.2-klein-base-4B" \
  --subset-root "$HOME/dataset/CLEAR-FLUX2-klein-base-4B-10K" \
  --cache-root "$HOME/dataset/CLEAR-FLUX2-klein-base-4B-10K/flux2_diffsynth_sliding11_cache" \
  --output outputs/flux2-stock-kda-clear --steps 1000

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m flux_kda.train \
  --data-mode cached --family flux3 --weights "$HOME/huggingface_ckpts/flux-3-image-klein-base/model.safetensors" \
  --subset-root /path/to/subset --cache-root /path/to/cache --subset-size 20000 \
  --output outputs/flux3-stock-kda-clear --steps 1000
```

FLUX3 缓存模式支持 `flux3_qdm.data.CachedDataset` 和 CLEAR CSV 格式。使用 `--shard-optimizer --gradient-checkpointing` 显式启用原有显存配置，多卡通过 `python -m torch.distributed.run --nproc_per_node=N -m flux_kda.train ...` 启动。通过 `--resume` 恢复。架构标识为 `sliding-stock-fla-kda-v2`。

node01 的三卡训练入口为 `bash flux_kda/train_flux2_node01.sh`：GPU 1/2/3、每卡 batch 2、累积 1、总计 1000 步。使用现有 6689 张 1024×1024 缓存，先在第 1、2 步保存并恢复以检查完整训练断点，随后每 100 步保存 checkpoint，在 300/600/900 步及结束时暂停训练，用固定 seed 0、20 步、CFG 3.5 生成一张 1024×1024 验证图，然后恢复训练。验证结果位于训练输出的 `validation/step-N/`；训练阶段退出后才加载验证模型，避免额外占用训练显存。

`flux_kda/reboot_watch.py` 通过 node01 用户 crontab 每分钟及开机时执行短检查。训练或验证仍在运行时不会重复启动；进程被清理后，等待 GPU 1/2/3 空闲并调用启动脚本，从最新完整 checkpoint 恢复。若 checkpoint 已到验证步数而验证尚未完成，会先补齐该次验证。`watch-status.json` 记录状态，`watch.log` 记录检查异常。创建输出目录下的 `STOP` 文件可禁止自动重启，已有进程不受影响；明确的程序错误会暂停自动重启，完成目标步数的验证后不再启动（FLUX2 为 1000 步，FLUX3 为 10000 步）。

node01 的 FLUX3 入口为 `bash flux_kda/train_flux3_node01.sh`：GPU 1/2/3、每卡 batch 1、累积 1，使用 2000 张图片的 1024×1024 缓存，目标 10000 步。每 25 步保存 checkpoint，每 200 步暂停训练并生成一张验证图，随后断点恢复；验证参数为 1024×1024、20 步、CFG 3.5、seed 0。使用无偏置的 Sliding + KDA，BF16 计算和分片 FP32 主权重、优化器状态。

## 生成与计时

默认 20 步、CFG 3.5、seed 0。使用 `--size 2048` 或 `--size 4096` 指定方形输出。`--variant original|kda|both` 选择版本。修改版默认未训练，可用 `--checkpoint` 加载本架构训练产物。推理前将矩阵权重一次性转为 BF16，KDA 的 A_log/dt_bias 保留 FP32。

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m flux_kda.benchmark \
  --family flux2 --weights "$HOME/huggingface_ckpts/FLUX.2-klein-base-4B" \
  --size 2048 --steps 20 --cfg 3.5 --output outputs/stock-kda-speed/flux2-2k

CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m flux_kda.benchmark \
  --family flux3 --weights "$HOME/huggingface_ckpts/flux-3-image-klein-base" \
  --size 2048 --steps 20 --cfg 3.5 --output outputs/stock-kda-speed/flux3-2k

CUDA_VISIBLE_DEVICES=2 "$PYTHON" -m flux_kda.benchmark \
  --family flux3 --weights "$HOME/huggingface_ckpts/flux-3-image-klein-base" \
  --size 4096 --steps 20 --cfg 3.5 --variant original --output outputs/stock-kda-speed/flux3-4k-original

CUDA_VISIBLE_DEVICES=3 "$PYTHON" -m flux_kda.benchmark \
  --family flux3 --weights "$HOME/huggingface_ckpts/flux-3-image-klein-base" \
  --size 4096 --steps 20 --cfg 3.5 --variant kda --output outputs/stock-kda-speed/flux3-4k-kda
```

每个版本先执行一次 denoiser forward 预热。CUDA 同步的采样、VAE 解码分别计时，二者相加为生成时间；不包括文本编码、权重加载、首次编译和 PNG 写盘。保存原尺寸图片、latent 和 JSON 指标。

## 限制

修改版新增标准 KDA 参数尚未微调，生成图片不能代表训练后质量。完整模型训练显存、收敛及多卡恢复尚未验证。`benchmark_results.md` 和 `performance_analysis.md` 记录的是旧架构测量，不能作为本架构的速度数据。
