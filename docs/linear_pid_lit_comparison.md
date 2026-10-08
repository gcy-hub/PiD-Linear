# LiT 迁移流程、训练配置与 Linear-PiD 预算对照

首次核查日期：2026-10-03；两卡速度与无教师消融更新：2026-10-04。首次核查只读取代码、查阅官方资料并运行 CPU 小模块权重加载检查；第 6 节记录后续带教师短测，第 7 节记录最新的无教师默认配置。

## 1. 核查的仓库和范围

本地 `/home/ganchangyi/code/LiT` 的 origin 是 [官方 LiT 仓库](https://github.com/techmonsterwang/LiT)。本地 HEAD 和核查时远端 main 均为 `a00449410f75a1656b02bbcd25fc7320bef47df2`，工作区没有改动。

LiT 包含两个迁移入口：`class2image` 将预训练 DiT 转换为 LiT，`text2image` 将预训练 PixArt-Sigma 转换为 LiT。后者与当前图文条件恢复训练更接近。对照原始 [PixArt-Sigma](https://github.com/PixArt-alpha/PixArt-sigma) 后，LiT 保留的 `AttentionKVCompress` 和 `MultiHeadCrossAttention` 两个类与上游的 AST 一致；迁移主要发生在新增学生网络、预训练权重加载和教师输出损失中。

本文比较的是继承、替换、监督和训练预算。模型容量和分辨率不同，使相同步数或样本消费量不能保证相同质量。

## 2. 迁移方式是否一致

继承与替换路线一致：使用充分训练的原模型，继承学生主干，随机初始化新 Attention，同时训练新模块与继承主干。LiT 的主要实验使用冻结教师的最终输出监督；Linear-PiD 在 2026-10-04 根据无教师监督消融改为首轮默认只用 FM loss，保留可选教师。没有要求先做特征对齐、双分支插值或减步蒸馏。

| 环节 | 官方 LiT 实现 | 当前 Linear-PiD 实现 |
|---|---|---|
| 首次迁移 | 新建学生，加载能匹配的非替换权重 | 先严格完整加载原 PiD，再替换选定 Attention |
| 替换范围 | 一次转换全部图像 Self Attention；T2I 保留文本 Cross Attention 参数 | 首轮只转换 `[1,4,8,12]`，保留其他 Attention 和两层 PiT |
| Q/K/V/O 初始化 | C2I 全部随机；T2I 实际仅 Q/K/V 随机，O 被继承，详见下文 | 替换模块的图像／文本 Q/K/V/O 全部重新随机初始化 |
| 新投影与局部卷积 | Linear 使用 Xavier uniform；DWC 为 Conv2d 默认随机初始化 | Q/K/V/O 使用 trunc_normal(std=0.02)；局部残差卷积分支零初始化 |
| 学生训练范围 | 优化器接收整个学生 | 新 KDA 和继承的有效像素主干均训练；条件适配器及无输出用途的分支冻结 |
| 教师运行 | 冻结；每个训练输入一次无梯度前向 | 默认不创建；正的 `--lambda-out` 才启用，共享条件编码器 |
| 训练目标 | 原 DDPM 噪声预测与 VB loss，加噪声输出、方差头输出蒸馏 | 默认原 FM velocity loss；可选教师最终 velocity 输出 MSE |
| 条件 dropout | 公开代码学生 train、教师 eval；学生内部 caption dropout 会改变条件 | 在网络前统一处理 caption／latent dropout，师生条件一致 |
| 学习率策略 | T2I 公开配置对全学生使用同一基准 LR | 新 KDA `1e-4`，继承主干 `1e-5` |
| 后续阶段 | 1024 阶段继承已转换好的 512 LiT | 支持继承已恢复学生，只初始化新增替换层；尚未开展扩层实验 |
| EMA | C2I 有 GPU EMA；T2I 公开训练入口没有创建／更新 EMA | 当前按用户要求关闭 EMA，无模型 CPU offload |
| 条件预计算 | T2I 示例读取缓存的 T5 特征 | 默认读取已完成的本地 Gemma 文本缓存 |

对应代码：LiT [C2I 初始化](https://github.com/techmonsterwang/LiT/blob/a00449410f75a1656b02bbcd25fc7320bef47df2/class2image/train_distillation_mean_var_linear35_initialize.py#L148)、[T2I 训练入口](https://github.com/techmonsterwang/LiT/blob/a00449410f75a1656b02bbcd25fc7320bef47df2/text2image/train_scripts/train_distillation_mean_var.py#L418)、[蒸馏损失](https://github.com/techmonsterwang/LiT/blob/a00449410f75a1656b02bbcd25fc7320bef47df2/text2image/diffusion/model/gaussian_diffusion.py#L744)。Linear-PiD 对应 `pid/_src/linear_pid/{runtime,attention,training}.py`。

### 需要区分的 T2I 初始化细节

LiT 论文描述整个新 Attention 不继承原权重。C2I 代码通过过滤所有含 `attn` 的键落实这一点。T2I 代码却直接 `load_state_dict(..., strict=False)`：

- 原模块使用 `attn.qkv`，学生改为 `attn.q` 和 `attn.kv`，这些键不匹配，保持新初始化。
- 原模块和学生都使用 `attn.proj`，形状相同，因此输出投影 O 的 weight 和 bias 会被加载。
- `cross_attn` 的投影键和形状也不变，因此继承其权重。学生的 head 分组变化仍会改变计算，不能把“权重继承”理解为“函数完全不变”。

证据为 [LinearAttention 定义](https://github.com/techmonsterwang/LiT/blob/a00449410f75a1656b02bbcd25fc7320bef47df2/text2image/diffusion/model/nets/PixArtLinearMS.py#L50) 和 [checkpoint 加载](https://github.com/techmonsterwang/LiT/blob/a00449410f75a1656b02bbcd25fc7320bef47df2/text2image/diffusion/utils/checkpoint.py#L62)。

CPU 检查提取官方类，使用其指定的 timm 0.6.12 Attention 基类，构建 dim=24 的原／新模块并加载带标记的原权重：O 与 Cross Attention 被继承，Q 保持原先随机值；缺失键为 q、kv、dwc，意外键为 qkv。这里只验证键匹配与加载行为，没有用实际发布权重复测质量。

所以当前 Linear-PiD 全随机 Q/K/V/O 符合论文建议和 C2I 初始化策略，但不是 T2I 公开代码的逐项复刻。此次没有据此改变已确定的初始化实验。

## 3. 找到的详细 T2I 配置

公开的 LiT 专用配置只有 [512 配置](https://github.com/techmonsterwang/LiT/blob/a00449410f75a1656b02bbcd25fc7320bef47df2/text2image/configs/lit_config/lit_xl2_img512_internalms_distillation.py)，继承 `configs/PixArt_xl2_internal.py`。没有找到独立、完整的 LiT 1024 训练配置或论文实际运行日志。

| 配置项 | 公开 LiT 512 示例 | 当前 Linear-PiD 默认值 |
|---|---|---|
| 数据 | PixArt toy dataset；多长宽比；缓存 T5 特征 | 全量有效 MultiAspect-4K-1M；多长宽比；缓存 Gemma 特征 |
| 每卡 batch／累积 | `2`／`1`；batch 行注释称默认 `48` | `1`／4 卡累积 `2`，8 卡累积 `1` |
| 有效 batch | 取决于卡数；README 是单卡 debug 示例 | `8` |
| 停止设置 | `num_epochs=10` | 不预设收敛步数；可设 max_steps，默认 job 时间上限 |
| 优化器 | CAMEWrapper，betas=(0.9,0.999,0.9999)，eps=(1e-30,1e-16) | AdamW，betas=(0.9,0.999)，eps=1e-8 |
| 基准 LR | 全学生 `2e-5` | 新模块 `1e-4`；继承主干 `1e-5` |
| LR 自动缩放 | 基类启用 `sqrt(effective_batch/256)` | 不自动按 batch 缩放 |
| weight decay | `0` | 矩阵参数 `1e-3`；归一化／门等排除项 `0` |
| warmup／后续调度 | 配置 `1000`；随后恒定 | `500` optimizer steps；随后恒定 |
| 梯度裁剪 | `0.01` | `1.0` |
| 混合精度 | FP16 autocast，Attention 计算 FP32；没有将学生整体 half 的代码 | BF16 autocast，学生／优化器 FP32 |
| 激活 checkpointing | 开启 | 开启 |
| 文本长度／dropout | `300`／`0.1` | `300`／`0.1` |
| 蒸馏系数 | noise `1.0`，variance `0.05` | 默认最终 velocity 输出 `0.0`，可用 `--lambda-out 1` 开启 |
| workers | 每进程 `10` | 每 rank `4`，可配置 |
| 验证／保存 | 每 `500` steps 验证；每 `2500` steps 保存，另按 epoch 保存 | 固定快速／完整画廊；每 `500` optimizer steps 或 `600` 秒保存 |

公开入口先按 effective batch 修改 LR，再构建优化器。按该代码计算，单卡 batch 2 的 LR 为约 `1.77e-6`；8 卡各 48 时为约 `2.45e-5`。论文写的是 `2e-5`，缺少实际日志，不能断言论文使用了示例中的自动缩放。warmup 列是配置值；Accelerate 的分布式 scheduler 推进还取决于其运行设置。对应 [自动缩放代码](https://github.com/techmonsterwang/LiT/blob/a00449410f75a1656b02bbcd25fc7320bef47df2/text2image/diffusion/utils/optimizer.py#L18)。

README 命令显式传 `--debug`，将每卡 batch 设为 2。示例原 caption 比例为 0.5，与论文叙述也不同。**不能把“toy 数据集训练 10 epochs”当成论文优质结果的训练预算。**

PiD 的 FM 网络没有 LiT 所监督的 DDPM 方差输出头，因此当前没有对应的 variance loss。不能为了照抄 LiT，直接添加不存在的方差目标。

## 4. 论文的步数与对应质量

下面均为学生转换／恢复训练步数，不包含教师预训练。来源：[LiT 论文 Table 5–8、§4.1–4.2](https://arxiv.org/html/2501.12976)。

| 实验 | 学生训练 steps | global batch | 已报告结果／阶段说明 |
|---|---:|---:|---|
| ImageNet 256，S/B/L/XL | 100,000 | 256 | 优于对应训练 400k 的 DiT；不代表追平充分训练的教师 |
| ImageNet 256，S/B/L/XL | 400,000 | 256 | 相比 100k 继续改善 |
| ImageNet 256，XL 最终结果 | 1,400,000 | 256 | CFG 1.5：FID 2.32，教师 2.27 |
| ImageNet 512，XL | 700,000 | 128 | CFG 1.5：FID 3.69，教师 3.04 |
| 文生图 512 | 约 106,000 | 384＝8×48 | GenEval 0.47 |
| 文生图 1024 | 从 512 LiT 再训练约 155,000 | 96＝2×48 | GenEval 0.48；表中 PixArt-Sigma 为 0.52 |

1024 的 155k 是后续阶段长度，不能用 checkpoint 文件名把它解释成从 106k 只追加 49k。论文明确写从 512 学生继续训练约 155k。

ImageNet 官方训练集为 [1,281,167 张](https://www.image-net.org/download.php)。用 `steps × global_batch / N` 粗略换算：100k×256 约 20 遍，400k×256 约 80 遍，1.4m×256 约 280 遍，700k×128 约 70 遍。实际整轮 batch 数受 drop_last 等影响。

文生图使用的内部数据集规模没有披露，因此无法得到可信的 epoch 数。论文教师初始化消融中的 200k／400k／800k 是教师 checkpoint 的训练年龄，不能当成学生恢复的步数。

C2I 脚本的 `--epochs 1400` 是入口默认值，不是“论文训练 1400k steps”的另一种写法。按标准 ImageNet 和 batch 256，它意味着约 700 万更新；脚本注释也说明训练迭代数与论文不一致。

## 5. 对当前训练长度的建议

LiT 512 文生图消费约 `106000×384 = 40,704,000` 个样本。当前有效 batch 8 消费相同数量需要约 `5,088,000` optimizer steps。1024 后续阶段另消费约 `155000×96 = 14,880,000` 个样本，batch 8 下对应 1,860,000 steps。这里只比较数据消费量，不能预测 PiD 达到同等质量所需的更新数。

当前 `GlobalBatchSampler` 在一次全局 batch 内不重复抽取，但不同更新之间有放回抽样，没有“完整遍历所有样本再开始下一 epoch”的保证。因此只能报告等效数据遍数，并同时记录实际 steps 和 samples_seen。

完整图片解码／分桶索引尚未完成，最终 N 仍未知。下表暂按 **N≈100 万有效图像**估计；文本缓存中的 1,007,222 条 caption 不能作为最终有效图片数。

| 当前 optimizer steps | batch 8 的样本消费量 | 等效数据遍数 | 用途建议 |
|---:|---:|---:|---|
| 1,000 | 8,000 | 0.008 | 看数值稳定与生成是否开始恢复 |
| 5,000 | 40,000 | 0.04 | 首次较完整地比较固定画廊 |
| 10,000 | 80,000 | 0.08 | 检查布局、颜色、条件一致性改善是否持续 |
| 20,000 | 160,000 | 0.16 | 首轮判断是否值得继续投入的观察节点 |
| 50,000 | 400,000 | 0.4 | 有持续改善时继续评估 |
| 100,000 | 800,000 | 0.8 | 较长恢复阶段，仍不是 LiT 相同训练量 |
| 约 125,000 | 1,000,000 | 1.0 | 一遍样本量；不保证所有样本均已见过 |

建议先为 4 层配置规划 **10k–20k 的观察窗口**，保留 step 0／100／500 和每 1k 的快速画廊、每 5k 的完整画廊。20k 是复查节点，不是成功标准或自动停止依据。图像持续改善时，再按 50k／100k 节点继续；如果出现明显停滞，先核查条件一致性、梯度和初始化行为。

这个建议是资源规划判断，论文没有提供 PiD 的恢复曲线，也没有证据保证 10k／20k 能恢复好。部分替换可能减轻恢复难度，但不能据此推导所需步数。

6 天 job 扣除 30 分钟退出余量约有 516,600 秒。完成 steps 应在训练验证后按 `516600 / 实测每次 optimizer update 秒数` 估计，并扣除画廊和保存开销。每更新 20／40／60 秒时，纯训练上限分别约 25.8k／12.9k／8.6k；这些是算例。后续带教师实测见第 6 节，无教师实测见第 8 节。

首次核查只新增本对照文档，没有改变训练超参数。最新默认 loss 变更见第 7 节；未启动正式实验或扩展替换层数。

## 6. 两卡实测速度后的预算更新（2026-10-04）

本节速度均来自**带教师监督（`--lambda-out 1`）**配置：在空闲的 0、1 号 A40 连续完成 6 次更新，无 OOM。每卡 batch 1、累积 4 次，有效 batch 8；覆盖 `1792×2688` 与 `2688×1792`。第 2–6 步平均 `20.5077` 秒／optimizer update，约 `0.3901` 张／秒。一次 microbatch 迭代平均约 `5.13` 秒，两卡同时各处理一张图。

第二步的具体拆分为：教师前向约 `3.95` 秒，学生前向与反向约 `16.03` 秒，总更新约 `20.24` 秒。该拆分累计了四次梯度累积，并按 rank 平均统计模型时间；总时间取较慢 rank。因此只能作近似占比判断：约 80% 在学生计算，约 20% 在教师，数据、VAE 和更新等剩余时间约 0.26 秒。

这一耗时对完整教师、约 12.85 亿可训练学生参数、真实 2K、激活重计算的配置有合理解释，但没有算子 profiler 或同硬件原训练基准，不能据此认定已经接近最优。学生已用 BF16 autocast 计算；FP32 参数不代表所有矩阵计算都在 FP32。

### 同步数与同样本量是不同预算

下表把所有样本按当前 PiD 2K 工作负载计算，假设两卡持续可用，不包含加载、保存、画廊、重启和调度中断。它不等于 LiT 原实验耗时，也不代表匹配 FLOPs 或质量。

| LiT 文生图阶段 | 只匹配 optimizer 更新数 | 匹配图像样本消费量，当前 batch 8 |
|---|---|---|
| 512：106k steps，batch 384 | 约 25.2 天；样本量只有 LiT 的 1/48 | 5,088,000 updates，约 1,208 天／3.31 年 |
| 后续 1024：155k steps，batch 96 | 约 36.8 天；样本量只有该阶段的 1/12 | 1,860,000 updates，约 441 天／1.21 年 |
| 两阶段累计 | 261k updates，约 62.0 天 | 6,948,000 updates，约 1,649 天／4.52 年 |

固定有效 batch 8，若只增加 GPU 且完全线性扩展，512 阶段相同样本量的理想耗时为：两卡 1,208 天，四卡 604 天，八卡 302 天。对应 step 约 20.5／10.3／5.1 秒；四卡与八卡尚未实测，实际通信和不同桶计算会影响这些估计。

只把梯度累积加大，使 global batch 达到 384，会减少更新数，但一个更新包含的 microbatch 也相应增多，不能把总样本训练时间直接减少 48 倍。

当前两卡 10k／20k／50k／100k updates 的纯训练时间分别约 2.37／4.75／11.87／23.74 天。六天扣除半小时退出余量，可做约 25.2k updates；正式运行还要扣除保存、画廊及重新加载的开销。这些是预算估计，仍不预设恢复所需步数。

### 值得优先尝试的提速方向

1. **先做算子与模块 profile。** 区分保留 Full Attention、KDA、PiT、FFN、重计算和通信，确认 SDPA 实际选择的 CUDA 后端；目前只能确定学生占大部分时间。调用 SDPA 已能自动选择融合后端，不能把“安装 FlashAttention”当成尚不存在的全部提速收益。[PyTorch SDPA 说明](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html)
2. **在可用时使用四／八卡。** 现有配置分别累积两次／一次，维持有效 batch 8。这是最直接的并行方式，但速度需实测。
3. **利用显存余量减少重计算。** 当前整个 MMDiT／PiT block 都 checkpoint；可尝试分块选择 checkpoint、恢复部分 autocast 权重缓存。需要覆盖不同桶后复测显存，不能直接全部关闭。checkpoint 本身通过反向重跑前向节省显存。[PyTorch checkpoint 说明](https://docs.pytorch.org/docs/2.10/checkpoint.html)
4. **尝试 torch.compile 与点运算融合。** 优先编译形状稳定的教师或 FFN／PiT／归一化部分，减少零碎算子；KDA 文本打包和多桶需处理动态形状。收益没有实测，不能承诺倍率。[PyTorch compile 说明](https://docs.pytorch.org/docs/2.10/generated/torch.compile.html)
5. **引入 GPU 内的 ZeRO-1／2。** 不开启 CPU／NVMe offload，先分片 optimizer／gradient 以释放显存，再测试更大每卡 microbatch 或减少 checkpoint。ZeRO 的主要直接收益是内存，额外通信可能抵消速度收益；当前训练入口尚未接入 DeepSpeed。[DeepSpeed ZeRO 说明](https://www.deepspeed.ai/tutorials/zero/)
6. **另行评估低分辨率恢复再升分辨率。** 例如先 1K 恢复后 2K 微调，最后检查 4K。这会改变既定训练策略，必须用固定样本确认质量；不能直接把其速度当成当前 2K 方案的速度。此次没有改变分辨率或训练范围。

建议先测 profile、精细 checkpoint／编译优化，再按真实可用卡数测吞吐。LiT 的样本量用于判断资源差距；当前仅替换四层，没有证据要求一定消费相同数量的图像才会恢复。

## 7. 无教师监督消融与首轮配置更新（2026-10-04）

[LiT 论文 Table 4／附录 Table 14](https://arxiv.org/html/2501.12976) 包含将两项蒸馏权重都设为零的实验。设置为 ImageNet 256、两头 LiT-S/2；学生继承训练 800k updates 的 DiT-S/2，随机初始化新 Attention，再训练 400k updates，batch 256。表中的 800k 是初始化 checkpoint 的年龄，不是学生本轮长度。以下为无 CFG 结果。

| 监督设置 | noise／variance 权重 | FID↓ | IS↑ |
|---|---|---:|---:|
| 无教师监督 | `0 / 0` | 53.83 | 27.16 |
| 小教师 DiT-S/2，噪声监督 | `0.1 / 0` | 55.11 | 26.28 |
| 强教师 DiT-XL/2，噪声监督 | `0.5 / 0` | 51.13 | 28.89 |
| 强教师 DiT-XL/2，噪声＋方差监督 | `0.5 / 0.05` | 50.79 | 29.17 |

相对无教师监督，强教师仅噪声监督的 FID 下降 2.70（约 5.0%），加方差监督下降 3.04（约 5.6%）；小教师这一设置反而退化。这支持将无教师作为可训练的恢复起点，但不能保证最终高质量。未找到对应 512／1024 文生图的无教师监督消融，也不能把这个收益比例迁移到 PiD 2K。

首轮默认已改为 `--lambda-out 0`：保留原 PiD 初始化和原 FM 目标，真正跳过教师构造、权重加载与前向。学生、VAE 常驻 GPU，文本使用本地缓存，不新增 CPU offload 或 EMA。可选的 `--lambda-out 1` 保留；同阶段恢复严格核对 loss 权重，切换目标需新阶段／新输出目录。

无教师模式的数值梯度、教师创建开关、恢复权重检查和学生单独画廊路径已纳入 CPU 回归测试。最初按第 6 节总耗时减去约 4 秒教师前向，估算两卡、有效 batch 8 为 16–17 秒／update。后续实际结果见第 8 节；第 6 节预算仍明确对应带教师配置。

## 8. 无教师配置的实测速度与预算（2026-10-04）

原定 1、3 卡测试在启动时发现 1 卡被另一任务占用约 39.4 GB，于训练更新前停止。随后先在 3 卡完成单卡验证，再用空闲的 0、3 卡完成双卡测试。两次均为完整模型、真实 2K 数据、文本缓存、无教师、无 EMA、无 CPU offload，覆盖 `1792×2688`／`2688×1792`。学生保持原 PiD 初始化，未创建或加载独立教师。

单卡每卡 batch 1、累积 8 次、有效 batch 8，完成 6 次更新，第 2–6 步均值 32.6866 秒。双卡每卡 batch 1、累积 4 次、有效 batch 8，完成 10 次更新，第 2–10 步均值 **16.5967 秒**、中位数 16.5760 秒、范围 16.36–16.95 秒，吞吐约 0.482 张／秒；学生前反向均值约 16.344 秒／update。双卡 PyTorch 峰值分配 29.42 GiB，nvidia-smi 0／3 卡峰值 32,641／32,210 MiB。无 OOM、无跳过 batch，loss／梯度有限，教师耗时全为零，完整 checkpoint 保存成功。所有测试输出均已删除。

与第 6 节带教师的两卡历史结果相比，更新耗时约减少 19%，吞吐约为 1.24 倍；卡号不同，因此不视作严格同卡速度消融。下表只计持续可用双卡的训练循环，不含加载、保存、画廊、重启或资源中断，也不预测收敛。

| 目标 | optimizer 更新数，当前 batch 8 | 两卡纯训练估计 |
|---|---:|---:|
| 首个观察窗口 | 10,000 | 1.92 天 |
| 首轮复查节点 | 20,000 | 3.84 天 |
| 较长恢复阶段 | 100,000 | 19.21 天 |
| 只匹配 LiT 512 更新数 | 106,000 | 20.36 天，样本量为其 1/48 |
| 匹配 LiT 512 的 40.704M 图像样本消费量 | 5,088,000 | 977 天，约 2.68 年 |
| 匹配 LiT 512＋1024 两阶段样本消费量 | 6,948,000 | 1,335 天，约 3.66 年 |

六天扣除半小时退出余量，纯训练上限约 31.1k updates，实际还需扣除上述开销。当前只替换四层，LiT 样本预算用于资源比较，不能据此认定必须训练相同样本量。四／八卡无教师吞吐仍需实测。

## 9. 单卡 batch 8 测试与资源限制（2026-10-04）

按用户指定在 0 卡尝试 `batch_size=8, grad_accum=2`。使用原数据集中的 64 张完整解码真实图片，两种横竖桶各 32 张；完整四层 KDA 学生、2K 输入、FP32 参数／optimizer、BF16 计算、激活 checkpointing、离线文本缓存，无教师／EMA／CPU offload。单卡有效 batch 16，计划检查三次 optimizer 更新。

启动前 0 卡空闲，但模型加载期间另一个任务占用了约 18.2 GB。首次学生前向在 FLA `chunk_kda_fwd_intra` 的 `Akk` 缓冲分配时 OOM，未完成任何 optimizer 更新。错误快照：

| 显存项 | OOM 时记录 |
|---|---:|
| GPU 可用总容量 | 44.34 GiB |
| 其他任务 | 17.84 GiB |
| 我们进程（包含非 PyTorch 内存） | 26.28 GiB |
| PyTorch 已分配 | 22.24 GiB |
| PyTorch 预留但未分配 | 3.73 GiB |
| 全卡剩余 | 205.50 MiB |
| 本次失败申请 | 444 MiB |

这是受共享资源影响的失败测试，不能据此判断独占 A40 时 batch 8 必定 OOM，也没有获得该配置稳定速度。后续 0 卡被其他任务占用约 39 GB，因此没有重试或等待资源；我们的进程已退出，测试日志／临时索引均已删除。

原 batch 1 的约 29 GiB 包含固定的参数／梯度／optimizer 和随 batch 增长的激活／临时缓冲，不能由剩余显存直接推导 batch 8 可用。空卡重测必须覆盖首次更新，以及 optimizer 状态建立后的后续前反向。八卡、每卡 batch 8、累积 2 次的有效 batch 是 128，此时每次更新处理的图片数为此前 batch 8 配置的 16 倍，不能沿用此前八卡约 4.15 秒／step 的线性扩展估算。

同日 12:26 按用户指示在物理 3 卡重试相同配置。开始时空闲，使用 64 张真实图像的临时索引、两种横竖 2K 桶，图片在实际 loader 中完整解码。模型加载期间另一个任务进入，使用 12.97 GiB；首次学生前向在 KDA `v_new` 申请 444 MiB 时 OOM。我们进程占 31.01 GiB，PyTorch 已分配 27.03 GiB、预留未分配 3.67 GiB，全卡仅余 345.50 MiB。未完成 optimizer 更新，因此仍不是独占 A40 的有效 batch 8 测试。由于 `CUDA_VISIBLE_DEVICES=3`，错误中的 GPU 0 是进程内编号，对应物理卡 3。测试进程和监控已退出，所有输出已删除。

同日 14:20 的物理 3 卡测试没有其他计算进程干扰。完整模型 batch 8／累积 2 在首次 PiT AdaLN 前向 OOM，申请 2.30 GiB 时进程占用 42.52 GiB、剩余 1.81 GiB；PyTorch 已分配 40.15 GiB、预留未分配 2.05 GiB。随后使用可扩展分配器重试，PyTorch 已分配达 41.90 GiB、预留未分配仅 165.39 MiB，仍需额外 2.30 GiB，而全卡只剩 1.95 GiB，再次 OOM。batch 8 在当前实现和精度下无法完成首次前向，不只是此前其他任务挤占或大块未使用缓存造成的问题；没有获得稳定耗时或 optimizer 更新。

随后的 0、3 卡 batch 4／累积 2 测试因 0 卡其他任务占用 18.38 GiB 而在首次前向 OOM，不能据此判断空卡 batch 4 的完整训练。测试输出均已清理，用户选择自行运行后续单卡测试。

## 10. batch 4 的 PiT 显存修复与实测（2026-10-04）

用户单卡 batch 4 在首次反向 PiT 整块重计算的 MLP GELU 申请 2.30 GiB 时 OOM，进程占用 44.11 GiB、剩余 230.81 MiB。新增 PiT 局部计算分块与分段 checkpoint 后，不改变全局 Attention、权重布局、FP32 参数／optimizer、BF16 计算或训练范围，CLI 在 batch ≥4 自动启用 chunk 2048；较小 batch 保留原实现，亦可显式指定 `--pit-chunk-size`。

完整模型先在空闲 0 卡完成 6 次更新，再从 checkpoint 在空闲 2、3 卡恢复至第 9 次。单卡 batch 4／累积 2／有效 batch 8，step 2–6 平均 38.7236 秒，约 0.207 张／秒；双卡每卡 batch 4／累积 1／有效 batch 8，step 8–9 平均 19.4685 秒，约 0.411 张／秒。两者峰值分配显存均约 42.20 GiB，无跳过 batch、loss 和梯度有限，完整 checkpoint 与 optimizer／scheduler／数据进度恢复检查通过。覆盖两种横竖 2K 桶，未测八卡、所有桶、长程或质量。

batch 4 吞吐低于此前 batch 1 的历史结果，且新计时包含同步后的实际 AdamW 更新，不能宣称更大 microbatch 已实现提速，也不能把第 8 节八卡 batch 1 的理想耗时套到 batch 4。全部本次测试日志／checkpoint 已清理，可复用的小型数据索引保留在数据集目录。复测命令见 [batch 4 验证](linear_pid.md#当前交付与验证状态2026-10-04)。
