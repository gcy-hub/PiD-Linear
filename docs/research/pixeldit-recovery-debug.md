# PixelDiT-QDM：逐层误差和 Full Attention 恢复诊断

不修改训练权重、不重训。固定原始验证图像及其 caption，按原生 1K 多长宽比 resize/center crop 后加噪。16 个样本 × 5 档噪声，时间尺度 0–1000（950 高噪声、50 低噪声）。输出包含 Attention、完整 Block 的图像/文本相对 L2、余弦相似度和 RMS，以及最终预测误差；文本统计包含原生序列的 padding 位置，与原始模型的 Attention 行为一致。

Teacher-forced 在教师相同层输入上重放学生 Attention/Block，主路径继续传递教师输出；Student rollout 从相同 noisy pixels 开始，按学生自己的层输出传播。恢复层直接调用对应原始 Attention 的完整 QKV、QK norm、RoPE 和输出投影，保留其他学生模块。

```bash
conda activate linear-pid
# 在本分支工作树根目录运行。
export DEBUG_ROOT="$HOME/code/PiD-Linear/outputs/research/pixeldit-qdm-recovery-debug-20261023"
export DEBUG_CHECKPOINT="$HOME/code/PiD-Linear/outputs/pixeldit-t2i-qdm-halo2/a2/checkpoints/step_000002297_batch_000002297_gen_1792731626398928445"
GPU_IDS=0,1,2,3 CPU_THREADS=1 bash scripts/diagnose_pixeldit_recovery.sh \
  --checkpoint "$DEBUG_CHECKPOINT" --output "$DEBUG_ROOT" --phase errors
python -m pid._src.pixeldit_qdm.recovery_diagnostics \
  --checkpoint "$DEBUG_CHECKPOINT" --output "$DEBUG_ROOT" --phase report
GPU_IDS=0,1,2,3 CPU_THREADS=1 bash scripts/diagnose_pixeldit_recovery.sh \
  --checkpoint "$DEBUG_CHECKPOINT" --output "$DEBUG_ROOT" --phase recovery
GPU_IDS=0,1,2,3 CPU_THREADS=1 bash scripts/diagnose_pixeldit_recovery.sh \
  --checkpoint "$DEBUG_CHECKPOINT" --output "$DEBUG_ROOT" --phase generate
python -m pid._src.pixeldit_qdm.recovery_diagnostics \
  --checkpoint "$DEBUG_CHECKPOINT" --output "$DEBUG_ROOT" --phase report
```

前三种固定恢复方案是前四层 `[0,1,2,3]`、中间四层 `[5,6,7,8]`、后四层 `[10,11,12,13]`；另测分散四层、隔层七层，以及按 teacher-forced 图像 Attention 误差选出的四层。索引从 0 开始。

生成使用四个固定 caption、原始相同 seed、50 步、CFG 2.75、shift 4；原生约 1K。所有列从同一噪声出发。每个样本、时间点和生成图独立保存，重复命令跳过已完成项；不配置定时重启。`report.md`、`summary.json`、逐层曲线、原尺寸比较图保存在指定输出目录。

相对 L2 不等于相对 MSE。该实验中的真实图像加噪输入与实际采样轨迹有分布差异；恢复层也是推理时结构干预，需要结合成图观察，不能仅据误差大小推断因果。

首批结果中，中间四层和高误差四层的交集为第 5 层。补充只恢复第 5 层、恢复 `[6,7,8]`、恢复 `[2,3,4]` 的生成干预：

```bash
conda activate linear-pid
GPU_IDS=0,1,2,3 CPU_THREADS=1 bash scripts/diagnose_pixeldit_recovery.sh \
  --checkpoint "$DEBUG_CHECKPOINT" --output "$DEBUG_ROOT" --phase followup
python -m pid._src.pixeldit_qdm.recovery_diagnostics \
  --checkpoint "$DEBUG_CHECKPOINT" --output "$DEBUG_ROOT" --phase report
```

主实验共 32 张生成图；补充 12 张。补充结果单独拼图，保留原先的层布局与指标。
