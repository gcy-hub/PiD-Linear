# Sliding + KDA 性能诊断

本实验的滑动窗口大小为 11×11 tokens。

剩余 FLUX3 4K 整图生成及等待任务已停止。诊断只做预热后的短时单层 CUDA event 计时与一次 profiler 采样，无训练。模型实现未修改。

## 2K：新增分支抵消局部 attention 收益

A40，BF16，24 heads × 128 head dim，图像 16384 tokens，batch 1。FLUX2 文本为 512 tokens；FLUX3 文本为 80 tokens，均与生成缓存一致。每项预热一次后重复 3–5 次。

| 单层组件 | 毫秒 |
|---|---:|
| 原生 attention，512 文本 tokens | 33.09 |
| FLUX2 Sliding，512 文本 tokens | 11.79 |
| 原生 attention，80 文本 tokens | 30.38 |
| FLUX3 Sliding，80 文本 tokens | 8.18 |
| KDA 完整分支，行正向、FP32 master | 25.62 |
| KDA 核心 chunk_kda | 8.00 |
| 新增 gate/out 两个稠密投影 | 5.14 |
| KDA 完整分支，行正向、BF16 矩阵权重 | 25.57 |
| KDA 完整分支，行反向、BF16 矩阵权重 | 27.62 |
| KDA 完整分支，列正向、BF16 矩阵权重 | 27.79 |
| KDA 完整分支，列反向、BF16 矩阵权重 | 29.98 |

FLUX3 的局部 attention 节省约 22.2 ms，而新增 KDA 分支超过此值。FLUX2 也相同。与整图 2K 测量一致：FLUX2 229.87→250.43 秒，FLUX3 291.40→307.47 秒。

KDA 完整分支包括原始二维 K/V depthwise 卷积、SiLU、衰减与 beta 投影、通道衰减构造、扫描重排、递推内核、FP32 RMSNorm、门控和输出投影。两个新增主投影均为 3072×3072；只看递推内核会严重低估代价。

列反向分支的一次 profiler 采样中，CUDA self time 总计约 29.85 ms：ChunkKDAFunction 7.70 ms，矩阵乘 5.51 ms，copy 5.40 ms，depthwise 卷积 2.27 ms，flip 2.19 ms，另有多个逐元素算子。Profiler 的 ATen 与底层 kernel 行存在层次关系，不能全部再次相加。

预先将 KDA 矩阵权重转为 BF16 仅将行正向分支从 25.62 降至 25.57 ms，说明其 master 权重转换不是该分支的主要瓶颈；这不等于完整模型所有权重转换都已测量。

## Sliding 的块稀疏粒度

FlexAttention 已实际减少工作量，没有证据表明局部 attention 退回全局密集计算。但二维滑窗映射到一维 token 块后，稀疏块覆盖范围大于实际窗口。

2K、80 文本 tokens 的 BlockMask 块稀疏率约 90.12%，即保留约 9.88% 的块；按 11×11 图像窗口加全文本可见性，理想 token 对可见比例上界约 1.70%。这表示块覆盖冗余，不能直接当作实际 kernel FLOPs 或同倍性能损失。

## 4K 单层短测

65536 图像 tokens、80 文本 tokens：原生 attention 497.82 ms，Sliding 55.90 ms，完整行正向 KDA 分支 135.07 ms。后两项合计约 190.97 ms，单层 attention 部分约为原生耗时的 38%。因此 4K 已显示计算量优势；FFN、投影等仍需计入整图时间，不能据此宣称整图提速 2.6 倍。

## 优化优先级

1. 保持数学和结构不变，减少 KDA 布局转换、扫描拷贝与 FP32 中间张量；评估融合衰减构造、RMSNorm 与门控。以完整分支计时作为验收依据。
2. 改进二维局部 attention 的分块/布局，减少一维块对二维窗口的覆盖冗余；必须保持图像与文本联合 softmax 语义。
3. 若 2K 仍无收益，再评估减少 KDA 层频率或分支宽度。这改变模型结构，应与单纯算子优化区分。

原始日志：`../experiment/kda-speed/operator-profile.log`、`../experiment/kda-speed/operator-flux3-shapes.log`。短测脚本：`/tmp/profile_crosslayer_kda.py`、`/tmp/profile_crosslayer_flux3_shapes.py`。单层输入为同尺寸随机张量；结果用于定位成本，不是新的整图生成测量。
