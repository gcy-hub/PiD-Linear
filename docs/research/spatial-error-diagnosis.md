# Spatial Error Diagnosis：先验证空间修正的动机

此分支只增加诊断工具，不改变训练模型。共同基线是官方未蒸馏 PiD 和已微调的 MMDiT-KDA 学生，两者都保留原有 PiT 结构。

要检验两件事：KDA 学生与官方参考的差异是否集中在少数区域；这些区域能否由已有 latent 条件预测。即使两件事成立，也还没有证明局部注意力能修复画质，需要后续修正模块和生成消融。

## 运行

准备学生 checkpoint 和固定条件 `.pt` 文件。条件格式与现有推理对比脚本相同，包含 latent、Gemma embedding、mask、尺寸、caption 和 seed。可以复用原始/学生推理对比中已经生成的输入，避免重新加载文本编码器。

```bash
conda activate linear-pid
pip install "matplotlib<3.11" "contourpy<1.4" "numpy==1.26.4"
cd PiD-Linear
git switch Spatial-Error-Diagnosis

GPU_IDS=2 CPU_THREADS=1 bash scripts/diagnose_spatial_error.sh \
  --checkpoint ./outputs/training/checkpoints/step_000003314_batch_000003314 \
  --weights-root ./weights \
  --asset-dir ./outputs/inference_comparison/assets \
  --reference-dir ./outputs/inference_comparison/original \
  --student-images-dir ./outputs/inference_comparison/trained \
  --output-dir ./outputs/research/spatial-error-diagnosis \
  --layers 0,4,8,12,7 --timesteps 100,500,900 \
  --tile-size 8 --feature-dim 32 --train-prompts 3 \
  --workers 2 --threads 1 --swanlab-mode offline
```

`GPU_IDS` 可以指定一张或多张卡，多个进程按完整输入样本分配任务；每张卡都完整加载教师和学生。每个样本/时间步完成后原子发布结果，重复运行同一命令会跳过完成项。输入、checkpoint 或测量设置变更时需使用新的输出目录。无需启动正式训练，也不会修改 checkpoint。

`--reference-dir` 和 `--student-images-dir` 读取既有推理脚本生成的 PNG/JSON，核对条件指纹、caption、seed 和模型身份。参考图用于构造 `x_t=(1-t/1000)*reference+(t/1000)*noise`；它是教师生成的代理参考，既非数据集真值，也不是记录下来的实际采样轨迹。不传参考目录时使用固定高斯像素，结果会明确标为离轨迹诊断；低噪声阶段不宜据此判断生成质量。

默认输入文件为 `prompt_*_2048.pt` 和 `prompt_*_4096.pt`，默认分析第 0、4、8、12 层 KDA，并用第 7 层保留的 Full Attention 作为微调变化控制。可用 `--asset-glob`、`--layers`、`--timesteps` 修改。时间步使用网络的 0–1000 尺度，latent sigma 固定为 0，记录条件分支，不执行 CFG 双分支采样。

## 记录什么

| 对照 | 执行方式 | 用途 |
|---|---|---|
| 同教师输入 | 教师的 Attention 输入同时送入两个模块 | 去掉上游隐藏输入不同的影响 |
| 同学生输入 | 学生的 Attention 输入同时送入两个模块 | 检查学生实际运行区域的模块差异 |
| 各自输入 | 两个完整网络各自的 Attention 输出 | 观察累积变化；不解释为纯 Attention 机制误差 |
| 网络预测 | 同一 noisy pixels、时间步、文本与 latent 的 velocity 差异 | 检查模块差异是否对应下游预测差异 |
| 既有生成图 | 原始和学生的相同条件输出 | 检查空间误差与最终像素差异的关联 |

保存误差能量 `sum((S-T)^2)` 和相对误差 `sum((S-T)^2)/(sum(T^2)+epsilon)`，使用绝对误差统计集中度，避免参考能量较小造成相对误差虚高。默认把 8×8 个 patch 合并成 tile，比较误差最高 20% 的 tile 占总误差多少，并给出 Gini、与 velocity/最终像素差异的 Spearman 相关。横竖构图保留二维网格，边缘不完整 tile 不按零值稀释。

latent 可预测性使用三个轻量 ridge 探针：位置坐标、原始 latent 加坐标、冻结 LQ 适配器特征加坐标。适配器通道先做固定随机投影到 32 维；这只影响诊断特征，不改模型。训练/留出以整个 prompt 为单位，同一 prompt 的不同尺寸和时间步不会分到两侧；标准化只拟合训练集合。每个层、分辨率、时间步独立拟合，目标是同学生输入的绝对误差经过 `log1p`。

留出结果同时比较随机选择、位置选择、边缘选择、latent/适配器选择和使用真实误差的 oracle 选择。oracle 只能展示选择误差区域的上限，不是可部署方法，也不代表可修复的误差量。

## 输出与判读

- `cases/*/maps.npz`：二维误差图、tile 条件特征、预测/生成差异。
- `cases/*/heatmaps.png`：同输入和各自输入对照；每张图独立色标，不用颜色跨层比较大小。
- `spatial_metrics.csv`：误差集中度、参考边缘与下游误差相关。
- `router_evaluation.csv/json`：只使用留出 prompt 的条件预测结果及明确的训练/留出 ID。
- `experiment.json`、`summary.json`：来源、参数、完成数量和限制。
- `swanlog/`：SwanLab 离线记录，可切换 cloud 或 disabled。

对均匀误差，选择 20% 的 tile 预期覆盖约 20% 总误差。如果覆盖率显著提高且能跨 prompt 重现，支持“空间误差非均匀”的假设。latent 探针需超过随机、位置与简单边缘控制，才能支持条件感知选择的进一步研究。

Attention 输出差异混合了算子、已学习权重、特征坐标系和 padding 行为变化。同输入实验去掉了上游输入变化，但不能单独证明某种机制导致画质下降。默认保持官方 Full Attention 的 padding 行为和学生有效文本打包行为。五个 prompt 只能提供初步动机证据；像素差异也不等于感知质量下降，未引入 LPIPS/FID 或自动成功阈值。计时包含 hooks 和额外对照前向，不能用于推理加速结论。

## 第一轮结果

使用 step 3314 的十层 KDA checkpoint、5 个固定 prompt、2K/4K 和 t=100/500/900，共完成 30 个条件。下表只汇总第 0、4、8、12 层 KDA 的同学生输入对照，每个条件/层等权平均；第 7 层 Full Attention 单独保留在原始结果中。

| 分辨率 | 误差最高的 20% tile 覆盖率 | Gini | 与最终生成像素差异的 Spearman |
|---|---:|---:|---:|
| 2K | 23.46% | 0.058 | -0.164 |
| 4K | 23.53% | 0.061 | -0.142 |

误差集中程度较弱，且这些模块差异与最终生成像素差异没有稳定正相关。当前标签不能直接解释为需要修复的画质缺陷。

按完整 prompt 分成 3 个训练、2 个留出组后，选择 20% tile 的留出结果如下。取整使均匀预期略高于 20%；这里平均 4 层×3 时间步×2 个留出 prompt。

| 分辨率 | 随机选择 | 位置探针 | 原始 latent 探针 | LQ 适配器探针 | 已知误差 oracle |
|---|---:|---:|---:|---:|---:|
| 2K | 20.22% | 20.70% | 21.83% | 22.84% | 23.17% |
| 4K | 20.05% | 20.36% | 21.26% | 22.49% | 22.94% |

适配器特征能预测当前差异，排序 Spearman 为 0.768/0.709，但 oracle 的覆盖上限本身仅约 23%。第一轮对“latent 可预测模块差异”提供了初步支持，对“少量局部修正能覆盖主要画质损失”支持不足。进一步研究应先确认更贴近生成质量的误差标签或做局部干预实验，再决定是否实现路由修正。结果不包含新修正模块、完整生成消融或推理加速验证。
