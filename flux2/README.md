# FLUX.2 Klein Base 4B · Sliding11 + QDM

把原生5层双流和20层单流 Attention 改为二维 Sliding Softmax + 四象限 QDM。局部分支保留图文 RoPE，QDM不使用 RoPE，不进行文本写入。文本 query 仍访问全部图文 token。

单流层的 Attention 与 FFN 共享融合线性层；只改变 Attention 路径，A1的损失不包含 FFN 输出，B阶段的LoRA只作用于Q/K/V行和Attention输出列。

## 准备与后台训练

使用已有 `flux3-qdm` 环境（PyTorch2.10、Diffusers0.37.1），所有资源从本地读取。模型目录需要包含原生 `transformer/`、`text_encoder/`、`tokenizer/`、`vae/` 和 `scheduler/`。

数据目录使用 `prompts.json` 与 `images/<id>.png`。只纳入已完成的图像；先按1K长宽比桶Lanczos缩放、中心裁剪，再用FLUX.2原生VAE缓存latent。Qwen3采用原生聊天模板、512长度和9/18/27层特征，连同空提示词及验证prompt一起缓存。缓存保存在数据目录的 `flux2_qdm_cache_1k/`，不复制源图。

```bash
conda activate flux3-qdm
cd /path/to/PiD-Linear
export FLUX2_WEIGHTS_ROOT=/path/to/FLUX.2-klein-base-4B
export FLUX2_DATASET_ROOT=/path/to/CLEAR-FLUX2-klein-base-4B-10K
export FLUX2_CACHE_ROOT="$FLUX2_DATASET_ROOT/flux2_qdm_cache_1k"

GPU_IDS=1,2,3 CPU_THREADS=1 bash flux2/scripts/train_flux2_sliding_qdm.sh \
  --stage all --sliding-kernel-size 11 \
  --resolution 1024 --batch-size 4 --grad-accum 1 \
  --a1-steps 200 --a2-steps 200 --b-steps 1000 \
  --validation-steps 200 --save-steps 25 \
  --output-root ../experiment/flux2-sliding11-qdm-1k-node01
```

命令在后台先完成缓存，然后串行训练三阶段，每轮约20分钟保存、更新进程并续跑。再次运行同一命令恢复同一目录；控制器有锁，拒绝重复启动。连续数值错误、OOM或无进展会停止重试。手动停止后不会自动重启。

| 阶段 | 更新数 | 三卡有效batch | 训练参数 | 目标 |
|---|---:|---:|---|---|
| A1 | 200 | 12 | 新增QDM | 相同教师层输入上的图像Attention相对MSE |
| A2 | 200 | 12 | 新增QDM | 最终latent预测MSE |
| B | 1000 | 12 | QDM、局部门、rank64 LoRA | 最终latent预测MSE |

使用一套冻结BF16主干，新增参数和优化器FP32，计算BF16；QDM统计和求解FP32。不使用EMA或CPU offload。AdamW、基础LR1e-4、warmup50/50/200，cosine至5e-6。师生FM仅记录，caption dropout0.1。噪声时间采用原生FlowMatchEuler调度，velocity为noise−clean。

每200步生成10张学生1K图像（50步、CFG4、原生分辨率相关shift）。验证逐图、每5个去噪步保存进度，中断可续。SwanLab默认离线。

如已有同一数据、同一FLUX.2编码器的文本缓存，可设置 `FLUX2_TEXT_CACHE_SOURCE=/path/to/previous-cache` 共享文本特征；图像latent始终按新分辨率重新编码。

## 日志、状态和停止

```bash
conda activate flux3-qdm
cd /path/to/PiD-Linear
# 实际阶段、周期与进程
GPU_IDS=1,2,3 bash flux2/scripts/train_flux2_sliding_qdm.sh \
  --output-root ../experiment/flux2-sliding11-qdm-1k-node01 --status
# 完成当前更新/图像边界后停止自动续跑
bash flux2/scripts/train_flux2_sliding_qdm.sh \
  --output-root ../experiment/flux2-sliding11-qdm-1k-node01 --stop
# 训练/准备主日志；缓存细分日志在 $FLUX2_CACHE_ROOT/logs/
tail -f ../experiment/flux2-sliding11-qdm-1k-node01/repeat/latest-training.log
```

## 独立验证

```bash
conda activate flux3-qdm
cd /path/to/PiD-Linear
GPU_IDS=1,2,3 bash flux2/scripts/evaluate_flux2_qdm.sh \
  --checkpoint /path/to/run/b/checkpoints/step_000001000_ID \
  --stage b --stage-root /path/to/run/b \
  --weights-root "$FLUX2_WEIGHTS_ROOT" \
  --output-root /path/to/run/b/galleries/step_000001000 \
  --gpu-ids 1,2,3 --student-only --resolutions 1024 --resume
```
