# PiD-QDM Halo2：官方混合分桶与 CP=2

训练采用 PiD 官方 `2048_3840` 的五种长宽比和原生尺寸选择规则。每张图进入其原始尺寸能够容纳的最大桶，Lanczos 缩小后中心裁剪，不放大图片。最大正方形为 3584×3584；横图可到 4096×2816 或 4096×2176，竖图对应转置。latent 条件仍由训练图下采样 4 倍后经 FLUX VAE 编码。

`--context-parallel-size 2` 用两张 GPU 共同处理同一批图像：图像 query 分片，K/V 汇集；QDM 的四象限统计量跨两卡求和，保持原有四象限求解语义。文本流复制。全网反向包含跨卡 K/V 梯度通信。当前支持 PiD、fixed4、Halo 和远程分支 NoPE，PiT 保持原结构。

有效 batch 的计算是：

```text
有效 batch = (GPU 总数 / CP 大小) × batch-size × grad-accum
```

例如两卡 CP=2、batch2、累积4，有效 batch 是 8；八卡相同配置为 32。CP 副本不重复计入样本数。

## 运行

沿用已有图片索引和 Gemma 文本缓存。原生分桶的偏移数组会自动建立在索引目录的 `native_buckets_3840/`，不复制图片。每个 microbatch 形状一致，梯度累积的不同 microbatch 可以使用不同尺寸。极少数样本不足一个全局 microbatch 的桶使用有放回采样，桶名与数量会写入日志和 checkpoint 元数据。

设置已有资源的位置：

```bash
conda activate linear-pid
export PID_WEIGHTS_ROOT=/path/to/weights
export PID_INDEX_ROOT=/path/to/MultiAspect-4K-1M/linear_pid_index
export PID_TEXT_CACHE_ROOT=/path/to/MultiAspect-4K-1M/linear_pid_text_cache
export PID_GALLERY_ROOT=/path/to/prepared-gallery-assets
```

在仓库根目录运行三个阶段，默认步数为 500 / 1000 / 10000：

```bash
GPU_IDS=2,3 CPU_THREADS=1 CUDA_LAUNCH_BLOCKING=0 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
bash scripts/train_pid_vdn.sh \
  --stage all --resume auto \
  --memory-mode fixed4 --region-size 8 --local-mode halo --local-halo 2 \
  --context-parallel-size 2 \
  --training-data full --resolution 3840 \
  --resolution-policy native --native-max-resolution 3840 \
  --linalg-backend cusolver \
  --batch-size 2 --grad-accum 4 --effective-batch 8 \
  --attention-batch-chunk 1 --workers 2 --threads 1 \
  --validation-resolution 4096 \
  --validation-cases generated_010,generated_029,generated_003,generated_009,generated_019,generated_022,real_011,real_028 \
  --output-root outputs/pid-qdm-halo2-native-cp2
```

中断后重复同一命令恢复。也可以单独选择 `--stage a1|a2|b`，以 `--max-steps` 设置阶段总步数。阶段切换加载上一阶段权重，重建优化器；同阶段恢复完整训练状态。原生分桶与之前强制 4096 的训练配置不同，应使用独立输出目录。

验证期间临时关闭 CP，各 GPU 分别生成不同样本，再恢复 CP。验证仍只生成上面指定的八张图。

## 耗时和显存口径

SwanLab 和 `metrics.jsonl` 分别记录教师前向、学生纯前向、学生反向及 loss、完整更新耗时，以及张量分配峰值与 allocator 预留峰值。教师仅前向；学生还需要反向和激活重算。比较推理速度时，应比较两者纯前向，不能拿学生前向加反向与教师前向比较。

`--linalg-backend cusolver` 显式选择 QDM 矩阵求解后端，不改变记忆公式。CP 实现保留全网格 K/V 卷积，确保跨分片边缘的邻域正确；该预处理目前仍在两卡各执行一次。
