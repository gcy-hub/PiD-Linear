# QDM 象限诊断（无训练）

本工具只做两项诊断，不修改训练模型，不启动优化器。

1. **原始教师的跨象限 mask。** 在 PiD 与 PixelDiT-T2I 中，图像 query 仅访问本象限图像及全部文本；文本 query 仍访问全部图文 token。RoPE、原始 Q/K/V/O 和 PiT 不变。通过分象限收集 K/V 执行精确 SDPA，不构建图像 N×N mask。
2. **已有 QDM checkpoint 的边界分析。** 固定原始教师的层输入，计算 local、remote、融合输出。仅将 remote 的水平和垂直分界从 1/2 移到 3/8、5/8；不改变图像位置、RoPE 或 Halo。默认划分与原实现逐元素校验一致。

两项实验使用同一组固定 prompt 和 seed。在原始教师的原生采样轨迹上捕获最接近 t=900、500、200 的条件分支输入，并记录实际时间。逐层 Attention 误差、单层替换后的最终预测误差与累积 1/4/8/14 层后的条件及 CFG 预测误差分别保存。累积层列表相互包含，详见输出 JSON。

PiD 使用原生 2K、25 步、CFG 5、shift 6；PixelDiT 使用原生 1K、50 步、CFG 2.75、shift 4。PixelDiT 的负条件使用它原生的不带 CHI 前缀的文本缓存。四个 prompt 分别涉及小孩与狗、自行车、多人室内和绿色鹦鹉。

## 运行

先准备原仓库说明中的模型和固定验证条件，指定两个**完整 checkpoint 目录**。PiD 选择 Halo2 QDM A2，PixelDiT 选择 Halo2 QDM A1；工具会严格检查 backbone 与 fixed4 类型。支持远程分支有或无 RoPE 的 checkpoint。

```bash
conda activate linear-pid
export PID_WEIGHTS_ROOT="/path/to/weights"
export PID_GALLERY_ROOT="/path/to/fixed-gallery-assets"
export PID_TEXT_CACHE_ROOT="/path/to/linear_pid_text_cache"
export PID_QDM_CHECKPOINT="/path/to/pid-qdm/a2/checkpoints/step_DIRECTORY"
export PIXELDIT_QDM_CHECKPOINT="/path/to/pixeldit-qdm/a1/checkpoints/step_DIRECTORY"
export PID_PARTITION_OUTPUT="outputs/research/qdm-partition-diagnosis"

GPU_IDS=0,1,2,3 CPU_THREADS=1 bash scripts/diagnose_qdm_partitions.sh
```

四个主队列分别执行 PiD mask、PixelDiT mask、PiD 边界、PixelDiT 边界；另有两个数值控制队列，在小孩与狗样本上保持全部图像 K/V 可见，仅将图像、文本 query 分开调用 SDPA，排除 BF16 内核形状差异。`GPU_IDS` 可减少，队列会依次执行；不依赖 DDP。单次前向 batch=1。OOM 或异常会保存 `failure.json` 并使对应队列退出，不改变分辨率或模型。

中断后重复命令会复用教师轨迹、跳过已保存的噪声时间和已完成任务。每张生成图单独保存。没有自动重启，也不提交 Slurm。每个任务持有独占锁，同一任务已有 worker 时会跳过，避免并发覆盖。改变 checkpoint、输入协议或重要参数需新建输出标签。

只运行某些队列或 prompt：

```bash
# 队列编号：0=PiD mask，1=PixelDiT mask，2=PiD 边界，3=PixelDiT 边界
# 4=PiD 全可见数值控制，5=PixelDiT 全可见数值控制
GPU_IDS=0,1 PID_DIAGNOSTIC_SLOTS=2,3 \
PID_DIAGNOSTIC_CASES="generated_009_2048 generated_029_2048" \
bash scripts/diagnose_qdm_partitions.sh
```

汇总表、完整数值、原尺寸拼图和空间热图：

```bash
python -m pid._src.pixeldit_qdm.report_partition_diagnostics \
  --output "$PID_PARTITION_OUTPUT" --plots
```

单独诊断其他已准备好的条件，例如真实 VAE 编码得到的 4K latent（不可直接把 2K latent 改标成 4K）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m pid._src.pixeldit_qdm.run_partition_diagnostics \
  --mode boundary --model pid --label halo-a2-4k \
  --checkpoint "$PID_QDM_CHECKPOINT" \
  --weights "$PID_WEIGHTS_ROOT" --assets "/path/to/4k-conditions" \
  --text-cache "$PID_TEXT_CACHE_ROOT" \
  --output "$PID_PARTITION_OUTPUT" --case generated_009_4096 --threads 1
```

## 解释范围

- 相对 L2 是 `||changed-reference|| / ||reference||`，不是相对 MSE。额外保留 MSE、输出及参考 RMS。
- 边界误差能量同时报告绝对值、内部值和比例；中心可能恰好有物体轮廓，必须结合教师控制、平移分界及多个 prompt。
- remote 邻接特征跳变可以定位记忆分区的影响，但不等于最终图片一定存在同等强度的接缝。
- remote 特征通道不是 RGB。只有整网像素预测的均值误差才按 RGB 报告，它仍不等同于最终成图偏色。
- PiD 与 PixelDiT 的任务、权重、原生分辨率及采样不同。两者对 mask 的敏感度差异可以提供机制线索，不能单独证明 latent condition 是原因。
- 运行中移动分区属于分布干预，只用于定位；不能凭临时改变分区后的画质否定或肯定新架构。
- A1 与 A2 checkpoint 训练阶段不同，不是预算匹配的跨模型对照。

输出保存在仓库忽略的 `outputs` 中；正式训练代码和 checkpoint 均不会改变。

已有同输入 2K/4K 原图与学生成图时，可以在输出目录提供 `image-pairs.json` 数组（每项为 `case`、`resolution`、`original`、`student`），统计原尺寸低频 RGB 差异与四象限均值：

```bash
conda activate linear-pid
python -m pid._src.pixeldit_qdm.partition_colour \
  --output "$PID_PARTITION_OUTPUT" --workers 2
```

这只描述匹配成图的颜色及内容差异，不将图像内容变化误认为纯色彩偏差。
