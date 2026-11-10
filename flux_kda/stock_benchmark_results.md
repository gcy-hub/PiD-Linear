# 标准 FLA KDA：2K / 4K 生成效率

本实验的滑动窗口大小为 11×11 tokens。

FLUX2、FLUX3 均使用本地原始权重；修改版为 Sliding + 标准 FLA `KimiDeltaAttention`，按层轮换行正向、行反向、列正向、列反向。新增分支未训练，使用标准初始化。架构标识：`sliding-stock-fla-kda-v2`。

配置：20 步、CFG 3.5、seed 0，提示词 `A red double-decker bus on a rainy London street, reflections on wet asphalt`。四张 NVIDIA A40：GPU0 顺序执行 FLUX2 2K 两版，GPU1 顺序执行 FLUX3 2K 两版，GPU2 执行 FLUX3 4K 原版，GPU3 执行 FLUX3 4K 修改版。Torch 2.10.0+cu128，BF16 矩阵计算；修改版推理前一次性转换矩阵权重，KDA A_log/dt_bias 保留 FP32。

## 完整结果

| 模型 / 分辨率 | 版本 | 去噪秒 | VAE 秒 | 合计秒 | 去噪峰值 allocated GiB | 预热秒 |
|---|---|---:|---:|---:|---:|---:|
| FLUX2 2048² | 原版 | 89.55 | 1.95 | 91.50 | 9.33 | 2.50 |
| FLUX2 2048² | 标准 KDA 混合 | 93.79 | 1.38 | 95.17 | 13.84 | 73.38 |
| FLUX3 2048² | 原版 | 114.23 | 1.49 | 115.72 | 13.21 | 3.10 |
| FLUX3 2048² | 标准 KDA 混合 | 114.88 | 1.35 | 116.23 | 18.52 | 61.02 |
| FLUX3 4096² | 原版 | 987.24 | 179.31 | 1166.55 | 21.79 | 24.21 |
| FLUX3 4096² | 标准 KDA 混合 | 501.50 | 184.98 | 686.48 | 28.82 | 145.88 |

2K 下没有观察到生成加速：FLUX2 总耗时增加 4.0%，FLUX3 增加 0.4%，后者在此次单次测量中接近持平。

FLUX3 4K 去噪为原版的 50.8%，即约 1.97 倍速度；包含 VAE 的总耗时从 19 分 26.6 秒降至 11 分 26.5 秒，减少 41.2%，约 1.70 倍速度。4K 两版 VAE 解码均耗时约 3 分钟，其成本不随 attention 替换下降。

## 计时口径

每个版本先执行一次 denoiser forward。去噪和 VAE 解码分别进行 CUDA 同步计时，合计不包含预热、文本编码、权重加载与 PNG 写盘。预热可能包含首次内核编译。显存列为去噪阶段 PyTorch 最大已分配显存，不包含 VAE 阶段峰值。

每个条件生成一张图，没有重复测量统计。4K 两个版本分别在两张同型号卡上执行。旧实现使用不同步数和 CFG，不能将不同轮次总耗时直接作为纯架构改动的收益。

## 图片与验证

![左原版，右未训练的标准 KDA 混合版](../outputs/stock-kda-speed/comparison.png)

六张原图已确认尺寸：FLUX2/FLUX3 2K 为 2048×2048，FLUX3 4K 为 4096×4096。原始模型生成连贯公交街景；修改版 FLUX2 出现密集碎片，FLUX3 两个分辨率均出现明显公交车重复与空间结构错乱。新增 KDA 分支尚未训练，当前结果用于效率测量，不代表微调后图像质量。

四方向与标准 FLA 层输出一致性、各参数有限梯度、BF16 推理权重一致性、FLUX2 两类 attention 接入、FLUX3 checkpoint 重算与 teacher 隔离、小型 block 的 Prodigy 更新及 delta 恢复均通过。检查日志：`outputs/kda-check/stock-fla-final.log`。

图片、latent、原始 JSON 和运行日志位于 `outputs/stock-kda-speed/flux2-2k/`、`flux3-2k/`、`flux3-4k-original/`、`flux3-4k-kda/`。运行命令见 [README](README.md)。本轮任务已全部结束，无训练任务。
