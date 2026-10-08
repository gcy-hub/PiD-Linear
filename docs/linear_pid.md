# Linear-PiD：4 层 KDA 恢复训练

第一阶段从完整 PiD v1.5 FLUX undistilled 加载所有权重，然后将 MMDiT 的 `[1,4,8,12]` 直接替换为随机初始化的 KDA。保留两层 PiT；像素主干参与恢复训练，FLUX latent 适配器及其注入门冻结。默认 `--lambda-out 0`，只用 FM loss，不创建训练教师或执行教师前向。学生使用离线 Gemma 文本条件和冻结 FLUX VAE。显式设置正的 `--lambda-out` 才加载原始 PiD 教师，共享条件编码器。训练入口复用 `scripts.train` 的分派，不进入 DMD 训练流程。

训练输出默认位于 `/home/ganchangyi/code/PiD-Linear/outputs/linear-pid`，日志、画廊和训练 checkpoint 均在仓库的 `outputs/` 下，已由 `.gitignore` 排除。原始模型权重、数据索引、文本缓存和环境仍使用原有本地目录。以下命令是完整工作负载；代码不会自动提交 Slurm，也不会根据 loss 自动增加替换层数。

2026-10-05 按用户要求，原四层 node01 训练在 step 2833 保存退出，自动续跑已停用，八卡 job `116623` 已取消。此前 `experiments/linear-pid*` 输出整体移动到当前仓库的 `outputs/`，旧路径仅保留兼容符号链接。随后四层到十层的渐进扩层对照也已在 total step 3041 / stage step 208 保存退出，并停用其 cron。当前主实验从原始 PiD 直接随机初始化全部十层 KDA，所有训练计数从 0 开始，新 optimizer、warmup 500。先使用物理卡 `1,2,3`、每卡 batch 4、累积 1；step 500 后因物理 1 卡其他进程占用显存而 OOM，改用两卡 `2,3`、每卡 batch 3 / 累积 2，完成两次更新并保存 step 502。0 卡空闲后改为 `0,2,3`，从完整 step 502 继续，每卡 batch 4、累积 1。随后其他进程再次占用显存而 OOM，最新完整断点为 step 591；随后使用 `0,1` 至 step 652。该配置恢复时因其他进程占用显存而 OOM，检查四卡余量后曾使用 `2,3` 从 step 652 恢复，每卡 batch 3、累积 2，训练至 step 777。随后四卡全部空闲，曾使用 `0,1,2,3` 从完整 step 777 断点继续，每卡 batch 3、累积 1、有效 batch 12。在 step 798 保存退出后，按用户要求改为四卡每卡 batch 4、累积 1、有效 batch **16**；通过完整状态续跑保留模型、optimizer、scheduler 和累计步数，新的独立输出为 `outputs/linear-pid-kda10-direct/node01-bs4/`。bs4 那次运行没有发布新的完整断点；按用户最新要求，现已停用 bs4 目录的 cron，回到原 `node01` 目录从完整 step 798 继续，四卡每卡 batch 3、累积 1、有效 batch 12。当前目录 cron 每分钟检查，每轮训练进程上限 28 分钟。配置与完整运行命令见 [十层直接初始化](linear_pid_kda10_direct.md)；旧 [渐进扩层阶段](linear_pid_kda10.md) 保留作对照。

2026-10-06 四卡 bs3 训练从 step 1221 恢复后已超过原报错位置，随后在 step 1374 后发生 CUDA 非法内存访问，rank 0 异常退出；其他 rank 被关闭，watcher 将本次错误标记为 `training_error`。最新完整断点为 step 1349。按用户要求，现以四卡每卡 bs4、累积 1、有效 batch 16，从该完整断点独立续跑到 `outputs/linear-pid-kda10-direct/node01-bs4-from1349/`，保留优化器及 scheduler；首轮开启 CUDA 同步诊断，具体触发算子尚待定位。

## 当前交付与验证状态（2026-10-04）

代码及 `linear-pid` 独立环境已准备好，包含固定 commit 的 FLA。CPU 功能测试、双进程 Gloo 分布式测试、断点恢复测试及静态检查已通过。官方 PiD 权重的全部 461 个键和形状已核对。

2026-10-04 已完成 1,008 个元数据分片的正式全量索引：998,694 条训练样本、1,024 条固定验证样本。用户同意先检查图片头、尺寸与 caption，把完整解码检查放在训练读取图片时执行；候选中已有 143,993 条经过完整解码检查。准备使用 16 个单线程 worker，全部完成后已退出。剔除原因包括缺少字段 7,314 条、Pillow 超大图片限制 175 条、空英文 caption 8 条、尺寸不足 15 条；具体明细见索引的 `shards/*.rejected.jsonl`。

全部 112 份固定验证条件已发布（64 个真实 2K、32 个生成 2K、两类各 8 个 4K）。已在 node01 启动 0、3 卡的正式训练：每卡 batch 4、累积 1、有效 batch 8、PiT chunk 2048、FM-only、无 EMA／模型卸载，完整数据索引及默认固定画廊均启用。训练根目录为 `/home/ganchangyi/code/PiD-Linear/outputs/linear-pid`，每次运行在 28 分钟到时后保存退出，手动重复命令恢复；实际更新与结果以该目录日志及 checkpoint 为准。

正式启动完成 step 0 的 16 张验证图后，发现推理模式创建的 RoPE 缓存不能被训练反向使用。训练内的画廊与网络测速现已使用 `no_grad`，保留普通无梯度缓存；新增“画廊后 checkpointed 训练”回归测试，核对完整小型学生的输出与参数梯度。

随后正式运行完成 29 个更新，在第 30 个更新反向申请 1.15 GiB 时 OOM，于 19:02 退出，并非 node01 到时清理。最近常见桶约 15–20 秒／step，已记录峰值 43.22 GiB；这次尚未达到首次保存时间，29 次更新没有 checkpoint，不能恢复权重。此次记录保留在运行目录的 `attempts/20261004_185219_oom`，原错误日志为 `launch_logs/formal_0_3_fixed_20261004.log`。

检查发现原像素位置编码按每种图像尺寸缓存完整 `[H*W,16]` FP32 网格；七种 2K 桶共需 1.791 GiB 常驻显存。Linear-PiD 已改为缓存等价的水平／垂直一维 sin/cos 编码，再在 GPU 上展开当前网格；七种桶的持久缓存共 0.902 MiB，不涉及模型 CPU 卸载或精度变更，原 PiD 入口仍保留原实现。FP32／BF16 的多长宽比位置值、完整嵌入输出及输入／参数梯度逐位一致，完整小型学生与画廊后的反向回归也通过。首个 optimizer 更新后额外保存一次完整 checkpoint，之后仍按 500 更新／600 秒以及退出条件保存。19:13 已在 0、3 卡重启完整数据训练，使用原始初始化并复用 step 0 画廊；截至 19:16 已完成 7 次更新，无 OOM／跳过 batch，最近更新约 15.8–19.9 秒，峰值 41.65 GiB。首个 checkpoint 已完整发布（15,895,491,844 字节），核对包含 458 个 optimizer state、scheduler step 1、两个 rank RNG 及数据 cursor 1。最新更新以当前 `metrics.jsonl` 和 `launch_logs/formal_0_3_axes_20261004.log` 为准。

全量英文文本缓存已完成：1,008 个分片、1,007,222 条非空英文 caption，输出在数据集的 `linear_pid_text_cache` 子目录；此数量不是最终有效训练图片数。每两分钟的 cron 检查和断点自动续跑已实际验证；抽查 8 条真实 caption，缓存与原在线 Gemma 输出及 mask 逐位一致。文本缓存读取、空 caption dropout、分片恢复和定时任务保护经过回归测试。

0、3 卡正式训练于 19:41 正常保存退出，完整断点为 step 86，峰值 41.69 GiB。21:34 尝试四卡恢复时，另一用户任务进入物理 0 卡并占用 14.60 GiB，导致首次前向 OOM，未完成新更新；原 step 86 断点不受影响。随后按用户指示改用物理 1、2、3 卡，使用 `--batch-size 3 --effective-batch 8 --grad-accum 1`：三 rank 分别消费 3、3、2 张，同一个全局 batch 的采样顺序及有效 batch 保持 8。loss 和记录的平均 loss 根据本地样本数加权，抵消 DDP 的等 rank 平均；三进程 Gloo 检查覆盖累积 1／2，loss、全部梯度及 AdamW 更新与完整 8 张参考一致。完整模型已从 step 86 的 458 个 optimizer state／scheduler 恢复，并于 21:51 达到 step 100、开始快速画廊；常见桶约 14 秒／step，峰值 36.18 GiB，无跳过 batch。

训练自动续跑现已单独安装到 node01 的用户 cron，每分钟检查一次，固定 GPU `1,2,3`，每个完整训练进程最多 28 分钟。检查识别已有手动训练及 rank zero 的运行锁，避免重复启动；进程结束或被清理后，在所选卡满足显存条件时从最近完整断点重新启动。状态、配置及日志均在运行目录，详见第 4 节。

已使用 node01 的 0、2 号 A40 验证 KDA CUDA 输出／梯度、变长序列与 padding 隔离，并运行完整教师／学生、真实数据和 SwanLab 离线记录。两卡各 batch 1、累积 4 次，有效 batch 8；完成 3 次 optimizer 更新，包含从第 1 步 checkpoint 恢复后继续至第 3 步。恢复后的 batch 覆盖 `1792×2688` 和 `2688×1792`。

此前测试发现 AdamW 状态建立后第二个更新会 OOM。保留 DDP 梯度缓冲、关闭学生 BF16 权重缓存并启用 Gemma CPU 卸载后，曾完成上述短测，峰值分配显存为 36.93 GiB／rank。

按你的最新要求，当前实现已移除 EMA 的创建、更新和保存，也移除训练及画廊中的模型 CPU 卸载。默认学生、VAE 常驻 GPU；开启输出监督时教师也常驻 GPU。训练默认从磁盘缓存读取 Gemma embedding，不创建 Gemma 模型；`--online-text` 可显式使用 GPU 常驻 Gemma。生成、导出及阶段扩展默认使用当前学生权重 `raw`。新的无 EMA checkpoint／恢复、旧 checkpoint 兼容及画廊不复制模型到 CPU 的路径已经过 CPU 回归测试。

2026-10-04 在 node01 的空闲 0、1 号 A40 上重测文本缓存、无 EMA、无模型卸载、**带教师监督（`--lambda-out 1`）**的配置。使用首个 JSON 建立的 999 张真实有效图片索引，完整教师与四层 KDA 学生、每卡 batch 1、累积 4 次、有效 batch 8，连续完成 6 次 optimizer 更新，覆盖 `1792×2688` 和 `2688×1792`，无 OOM、无跳过 batch，loss 和梯度范数均有限。最终完整 checkpoint 保存成功，约 15.90 GB，保存耗时约 29 秒。

| 带教师配置的两卡实测项 | 结果 |
|---|---|
| 第一次更新（含首次运行开销） | 28.02 秒 |
| 第 2–6 次更新平均／中位数 | 20.51／20.51 秒 |
| 第 2–6 次更新范围 | 20.24–20.78 秒 |
| 稳态吞吐（有效 batch 8） | 约 0.390 张／秒 |
| PyTorch 最大分配显存（两 rank 取最大值） | 32.68 GiB |
| nvidia-smi 每秒采样的 0／1 卡占用峰值 | 35,310／35,264 MiB，即 34.48／34.44 GiB |

耗时为训练循环记录的 optimizer update 耗时，包含四次累积的教师前向、学生前反向、数据等待和更新调用；不含模型加载、独立画廊及 checkpoint 保存。每个 microbatch 的 GPU 计算通过 CUDA events 同步测量。此短测没有覆盖其他全部长宽比桶、4K 训练、长时间运行或四卡吞吐；此前 36.93 GiB 是旧卸载配置的结果，应以此次结果判断带教师配置的两卡短测。

随后根据 LiT 的无教师监督消融，将首轮默认改为纯 FM 恢复训练，详见 [消融与取舍](linear_pid_lit_comparison.md#7-无教师监督消融与首轮配置更新2026-10-04)。`lambda_out=0` 会跳过教师构造、加载和前向，保持原 PiD 初始化、训练范围与 FM 目标。此路径经过 CPU 回归测试，以及以下完整模型 GPU 短测。

无教师测试原定使用 1、3 卡，但准备索引期间，另一任务占用了 1 卡约 39.4 GB。原双卡启动在训练更新前停止，随后先用 3 卡完成单卡验证，再用空闲的 0、3 卡完成双卡验证。两次均使用真实图片、完整四层 KDA 学生、冻结 VAE、本地文本缓存、FP32 学生／optimizer、BF16 计算，无 EMA／模型卸载。学生初始化仍读取原 PiD checkpoint，没有加载独立教师网络。

| 无教师实测项 | 3 卡单卡 | 0、3 卡双卡 |
|---|---:|---:|
| 每卡 batch／累积／有效 batch | 1／8／8 | 1／4／8 |
| 连续 optimizer 更新 | 6 | 10 |
| 第 2 步起平均／中位耗时 | 32.69／32.64 秒 | 16.60／16.58 秒 |
| 稳定耗时范围 | 32.48–33.04 秒 | 16.36–16.95 秒 |
| 吞吐 | 0.245 张／秒 | 0.482 张／秒 |
| PyTorch 峰值分配（多卡取最大） | 29.43 GiB | 29.42 GiB |
| nvidia-smi 占用峰值 | 31,902 MiB | 0 卡 32,641 MiB；3 卡 32,210 MiB |

两次覆盖 `1792×2688`／`2688×1792`，均无 OOM、无跳过 batch，FM loss 和梯度有限，教师耗时与输出监督项为零，元数据为 `teacher_enabled=false`、`training_objective=fm`。完整 checkpoint 均保存成功，约 15.90 GB。与此前带教师两卡的 20.51 秒相比，本次两卡更新耗时约减少 19%，峰值分配显存约减少 3.27 GiB；实际卡号不同，不作为严格同卡的速度消融。测试不覆盖全部分辨率桶、长程恢复或生成质量。测试 checkpoint、日志、临时索引均已清理。训练预算见 [无教师实测预算](linear_pid_lit_comparison.md#8-无教师配置的实测速度与预算2026-10-04)。

随后在 0 卡测试 `batch_size=8, grad_accum=2`，启动前卡空闲；测试期间其他任务占用 17.84 GiB，首次学生前向的 FLA `chunk_kda_fwd_intra` 分配 444 MiB 缓冲时 OOM。错误快照显示我们进程使用 26.28 GiB（PyTorch 已分配 22.24 GiB、预留但未分配 3.73 GiB），全卡仅余 205.50 MiB。未完成 optimizer 更新，不能判断独占 A40 时 batch 8 是否可训练，也不能给出该配置稳定速度。0 卡随后被其他任务占用约 39 GB，未继续重测。测试日志和临时索引已删除，详见 [batch 8 测试限制](linear_pid_lit_comparison.md#9-单卡-batch-8-测试与资源限制2026-10-04)。

12:26 按用户指示切到空闲的物理 3 卡重测相同 batch 8／累积 2 配置。模型加载期间另一个任务进入，占用 12.97 GiB。首次前向在 KDA 分配 444 MiB 的 `v_new` 缓冲时 OOM：我们进程占 31.01 GiB（PyTorch 已分配 27.03 GiB、预留未分配 3.67 GiB），全卡剩余 345.50 MiB。未完成 optimizer 更新；本次同样受共享资源影响，未获得独占 A40 的有效 batch 8 结论。错误中的 GPU 0 为进程内编号，实际物理卡为 3。测试进程退出，输出已删除。

14:20 再次在物理 3 卡单独测试 batch 8／累积 2，确认当时没有其他计算进程。首次前向在 PiT AdaLN 申请 2.30 GiB 时 OOM，进程占 42.52 GiB（PyTorch 已分配 40.15 GiB、预留未分配 2.05 GiB），只剩 1.81 GiB。使用 `PYTORCH_ALLOC_CONF=expandable_segments:True` 复测也在同处 OOM：已分配 41.90 GiB、预留未分配 165.39 MiB、进程总占用 42.38 GiB，剩余 1.95 GiB，仍无法分配 2.30 GiB。这两次确认当前完整模型配置在独占 A40 上 batch 8 也无法完成首次前向，尚未建立 optimizer 状态。

随后按用户指示在 0、3 卡测试每卡 batch 4／累积 2，使用可扩展显存分配。0 卡在加载期间被其他任务占用 18.38 GiB，首次前向申请 3.45 GiB 时 OOM；因此 batch 4 的双卡完整训练仍未验证。该次测试进程与输出已清理。

已另行准备可复用的小型测试索引 `/home/ganchangyi/dataset/MultiAspect-4K-1M/linear_pid_test_index`，包含首个 JSON 的 999 张完整解码有效图片，验证集大小为零。它仅用于运行检查，不替代正式全量索引。

同日 15:00 的用户单卡 batch 4／累积 2 测试，在第一次反向的 PiT 整块重计算中于 MLP GELU 申请 2.30 GiB 时 OOM；进程占用 44.11 GiB，PyTorch 已分配 43.28 GiB、预留未分配 512.69 MiB，全卡仅剩 230.81 MiB。尚未完成 optimizer 更新。异常堆栈从 SwanLab 的 `.swanlab` 文件提取；仅看 `rank_0.log` 不包含这次未捕获异常。

为控制此峰值，Linear-PiD 新增 `--pit-chunk-size`；CLI 在每卡 batch ≥4 时自动使用 2048 个 patch／块，小 batch 默认沿用原实现：PiT 的逐像素归一化、AdaLN、投影、残差及 MLP 按块 checkpoint；跨 patch Attention 仍处理整个图像序列。AdaLN 权重按每个像素内的六路门顺序选取，不重新初始化或改变 state dict。默认教师、普通推理路径保持原实现；显式设置 `--pit-chunk-size 0` 可使用原训练实现。已通过独立 PiT 和完整小型学生的输出、所有参数与输入梯度检查。训练耗时现在同步 GPU 后记录，包含实际 AdamW 更新。

修复后在空闲物理 0 卡连续完成 6 次完整更新，再从 step 6 checkpoint 在空闲的 2、3 卡恢复至 step 9。两次均为每卡 batch 4、完整 FP32 学生／optimizer、BF16 计算、无教师／EMA／CPU offload、文本缓存、PiT chunk 2048，覆盖 `1792×2688`／`2688×1792`。无 OOM、无跳过 batch，loss／梯度有限。

| batch 4 修复后实测 | 单卡 | 双卡恢复 |
|---|---:|---:|
| 每卡 batch／累积／有效 batch | 4／2／8 | 4／1／8 |
| 本次更新 | 1–6 | 7–9 |
| 首个更新 | 42.27 秒 | 24.42 秒 |
| 后续更新均值 | 38.72 秒（2–6） | 19.47 秒（8–9） |
| 后续更新范围 | 38.49–39.24 秒 | 19.05–19.89 秒 |
| 吞吐 | 0.207 张／秒 | 0.411 张／秒 |
| PyTorch 峰值分配 | 42.20 GiB | 42.20 GiB（两 rank 取最大） |

checkpoint 6 与 9 均完整发布，大小约 15.90 GB；最终验证恢复了 458 个 optimizer state、scheduler step 9、数据 cursor 9、累计 72 张样本、两个 rank RNG，已有 KDA 权重保留。0、2 卡的首次恢复尝试因加载期间另一个任务进入 0 卡、占用 18.38 GiB 而 OOM，未完成新更新；随后 2、3 卡恢复成功。DDP 首次反向出现 depthwise conv 单维 stride 提示，但不影响此次梯度与更新检查。

此结果证明所测两种 2K 桶的 batch 4 可以运行及恢复，显存余量仍小；没有覆盖其他全部桶、八卡长程训练或生成质量。batch 4 实测吞吐低于此前 batch 1 的历史结果；它不是已经验证的提速方案。本次训练日志／checkpoint 均已清理，保留小型数据索引用于手动复测：

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
GPU_IDS=3 CPU_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
bash scripts/train_linear_pid.sh \
  --preset 4gpu --layers 4 --lambda-out 0 \
  --batch-size 4 --grad-accum 2 --pit-chunk-size 2048 --workers 4 --threads 1 \
  --index-root /home/ganchangyi/dataset/MultiAspect-4K-1M/linear_pid_test_index \
  --output-root /home/ganchangyi/code/PiD-Linear/outputs/linear-pid-bs4-single-test/run \
  --resume auto --max-steps 3 --max-seconds 1200 \
  --log-steps 1 --quick-every 0 --full-every 0
```

本次只验证训练功能，不运行正式实验或画廊。2K／4K 多步生成、生成质量、完整验证资产与独占硬件效率仍待后续检查。测试 checkpoint、日志及临时索引均作为临时输出清理，不提供可继续训练的测试权重。

## 1. 独立环境

```bash
cd /fs1/private/user/ganchangyi/code/PiD-Linear
LINEAR_PID_DOWNLOAD_DIRECT=1 bash scripts/setup_linear_pid_env.sh
conda activate linear-pid
python scripts/verify_linear_pid_env.py
python -m pytest tests/linear_pid -q
```

安装脚本创建 Python 3.12 环境，安装仓库固定的 PyTorch 2.10 / torchvision 0.25 与推理依赖、SwanLab 0.10.1、SciPy 1.15.2，再安装 [FLA 固定 commit](https://github.com/fla-org/flash-linear-attention/tree/9f38d24980c46d46bd38614e743cdacd21906578)。不修改 `pixel` 环境。`LINEAR_PID_DOWNLOAD_DIRECT=1` 仅让 PyTorch／CUDA 的 pip 下载子进程绕过代理，GitHub 仍使用原代理，可省略以使用原代理；不会修改全局或 Codex 代理。验证程序检查 Git 安装来源，并在 A40 上编译 KDA 内核、比较短序列的输出和梯度。选卡可用 `CUDA_VISIBLE_DEVICES=2 python scripts/verify_linear_pid_env.py`。

模型均通过 `local_files_only=True` 或本地文件读取。权重根目录默认 `/home/ganchangyi/huggingface_ckpts`，可在所有入口通过 `--weights-root` 修改。

| 组件 | 权重根目录下的路径 |
|---|---|
| 原始 PiD 学生初始化／可选教师及评估参考 | `PiD/checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_flux_undistilled/model_ema_bf16.pth` |
| FLUX VAE | `PiD/checkpoints/ae.safetensors` |
| Gemma | `gemma-2-2b-it` |
| 固定生成条件 | `Z-Image-Turbo` |

本地官方 PiD 文件不含 RGB 辅助头，因此恢复网络不创建该头，严格检查全部 `net.*` 键和形状。第一阶段同时关闭 REPA 特征缓存及 RGB 对齐损失。

## 2. 全量数据准备

### 2.1 离线 Gemma 文本缓存

缓存放在数据集同一目录下：`/home/ganchangyi/dataset/MultiAspect-4K-1M/linear_pid_text_cache`。文本准备直接读取全部 `data_jsons/*.json` 的非空英文 caption，不等待图片解码检查；最终图片索引按相同样本 ID 读取对应条件，少量被图片检查剔除的样本缓存不参与训练。保持完整 `300×2304` BF16 embedding，包括 padding 位置，另存有效 mask、空 caption 和 CFG 负面提示词。

node01 使用 4 张卡，并安装每两分钟检查一次的定时续跑程序：

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
GPU_IDS=0,1,2,3 python scripts/watch_linear_pid_text_cache.py \
  --install --batch-size 8 --workers 2 --threads 1 \
  --max-seconds 1680 --min-free-mib 12000 \
  --output-root /home/ganchangyi/dataset/MultiAspect-4K-1M/linear_pid_text_cache
```

每次编码最多 28 分钟，每 64 条或 30 秒持久化进度；中断后只重算未提交的尾部。每次检查都是短进程，由系统 cron 启动，即使先前的 Python／torchrun 被清理也会再检查。已有活跃进程或编码锁时不重复启动；启动前要求每张指定 GPU 至少空闲 12,000 MiB，显存不足时本轮不启动。完成后自动移除这条 cron，保留其他已有定时任务。

查看进度和最近启动的日志：

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
python scripts/watch_linear_pid_text_cache.py --status
cat /home/ganchangyi/dataset/MultiAspect-4K-1M/linear_pid_text_cache/run.json
```

需要暂停时创建 `STOP`，各 rank 在 batch 边界保存退出，cron 也停止重启。移除标记后下次定时检查会继续：

```bash
touch /home/ganchangyi/dataset/MultiAspect-4K-1M/linear_pid_text_cache/STOP
# 准备继续时：
rm -f /home/ganchangyi/dataset/MultiAspect-4K-1M/linear_pid_text_cache/STOP
```

也可直接运行完整缓存命令，关闭定时检查后由你手动恢复：

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
python scripts/watch_linear_pid_text_cache.py --uninstall
GPU_IDS=0,1,2,3 CPU_THREADS=1 bash scripts/prepare_linear_pid_text_cache.sh \
  --dataset-root /home/ganchangyi/dataset/MultiAspect-4K-1M \
  --output-root /home/ganchangyi/dataset/MultiAspect-4K-1M/linear_pid_text_cache \
  --batch-size 8 --workers 2 --threads 1 --max-seconds 1680 --resume
```

已有编码进程需要先退出，再运行手动入口。`--max-seconds 0` 可取消单次时间上限。GPU IDs、CPU 线程、预取线程和 batch 均可配置；变更卡数后按未完成的元数据分片重新分工。根目录 `cache.json` 只在全部分片完成后发布，训练拒绝不完整缓存。模型／tokenizer／编码规则和源元数据均保存指纹，每条记录还核对 caption 哈希。约百万条缓存占 1.38 TB；文件为分片 mmap 数组，不创建百万个小文件。

### 2.2 图片检查与训练索引

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/prepare_linear_pid_data.py \
  --dataset-root /home/ganchangyi/dataset/MultiAspect-4K-1M \
  --output-root /home/ganchangyi/dataset/MultiAspect-4K-1M/linear_pid_index \
  --workers 16 --validation-size 1024 --seed 42 --resume \
  --image-verification header
```

直接解析 `data_jsons/*.json`，相对路径以每个 JSON 的目录为基准。样本 ID 使用文件编号与零起始记录编号。图片头、英文 caption 字段和 2K 桶适用性检查的结果保存到 `shards/*.jsonl`；所有剔除原因写入 `shards/*.rejected.jsonl`。`header` 模式不保证所有图片均已完整解码，运行时发现损坏会让所有 rank 协调跳过该 batch；`index.json` 记录验证模式和已经完整解码的样本数量。CLI 默认仍为 `full`，预先完整解码时可显式设置 `--image-verification full`。

准备进度按 JSON 发布，重复运行会跳过已完成分片。完整验证分片可以被 header 模式复用，header 分片不能冒充完整验证分片。单个输出目录有进程锁，防止重复准备同时写入。正式训练开始后应保持该索引不变；若另做全量完整检查，使用新的输出目录，避免改变断点恢复依赖的数据指纹。

验证集按长宽比和 seed 42 固定选择 URL 组，同 URL 的样本整体划分。重复 URL 组可能令验证样本数略高于 1,024，最终数量以 `index.json` 为准。训练直接读取原图片；各 worker 通过 `manifest.jsonl` 和 mmap 偏移索引按需获取 caption。归一化在 GPU 上进行，worker 只传 uint8 图像。

batch 使用原样本数量分布选择长宽比桶，再无放回抽取一个全局有效 batch，划分到各 rank 和累积步。同一全局 batch 内不重复；不同更新之间允许重复。记录的是样本经过次数，不是遍历整套数据的 epoch。数据进度只在 optimizer 边界前进，预取不会推进 checkpoint 的 cursor。4 卡与 8 卡使用相同有效 batch 时，同一 cursor 对应相同的全局样本集合。

运行时遇到新的解码错误，所有 rank 协调跳过该全局 batch，写入 `rejected_rank_*.jsonl`，不进行 optimizer 更新。

## 3. 固定验证条件

第一次训练前准备；中断后重复运行同一命令即可。

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
GPU_IDS=0,1,2,3 CPU_THREADS=1 \
bash scripts/prepare_linear_pid_assets.sh \
  --index-root /home/ganchangyi/dataset/MultiAspect-4K-1M/linear_pid_index \
  --output-root /home/ganchangyi/code/PiD-Linear/outputs/linear-pid/assets
```

建立 64 个真实图像条件、32 个 Z-Image-Turbo 条件及两类各 8 个 4K 条件，共 112 个资产。真实样本来自固定验证集，按长宽比轮换；4K 样本是其中具有足够原生尺寸的子集。生成提示词固定，覆盖人物、自然、建筑、纹理、颜色、多物体和空间关系；生成 latent 分别来自 512 与 1024 输出，使用 Turbo 的 9 步和 CFG 0。

每个资产保存 latent、条件／空条件 embedding 和有效 mask、像素噪声 seed、caption 及 LQ 预览。Z-Image-Turbo 只在准备阶段加载。每个生成 latent 和条件文件独立发布，中断后无需重新生成已完成条件。最终 `assets.json` 保存文件指纹及数据索引指纹。

数据索引尚未完成时，可先并行准备生成条件，使用同一个输出目录。该阶段不发布完整画廊，索引完成后仍须运行上面的完整准备命令；已完成文件会复用，种子、提示词或权重来源改变时会拒绝混用：

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
GPU_IDS=0,3 CPU_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
bash scripts/prepare_linear_pid_assets.sh --generated-conditions-only
```

## 4. 完整 4 卡训练及恢复

node01 手动运行使用 0、3 卡、每卡 batch 4、累积 1（有效 batch 8），启用 PiT 分块、不设 step 上限。直接命令每次运行 28 分钟后在更新边界保存退出；需要自动续跑时安装下方 watcher。数据索引和固定验证条件准备完成后运行：

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
GPU_IDS=0,3 CPU_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
bash scripts/train_linear_pid.sh \
  --preset 4gpu --layers 4 --lambda-out 0 \
  --batch-size 4 --grad-accum 1 --pit-chunk-size 2048 \
  --workers 4 --threads 1 --resume auto --max-seconds 1680 --log-steps 1
```

四层阶段的历史三卡配置及自动续跑安装命令（该阶段已于 2026-10-05 停止；当前十层命令见上述十层文档）：

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
python scripts/watch_linear_pid_training.py --install \
  --gpu-ids 1,2,3 --batch-size 3 --grad-accum 1 --effective-batch 8 \
  --workers 4 --threads 1 --max-seconds 1680 --min-free-mib 38000
```

`--batch-size` 是每 rank 的上限；显式 `--effective-batch 8` 把每个全局更新均匀分配给各 rank，余数分配给前面的 rank，因此三卡实际为 `[3,3,2]`。有效 batch 必须能整除累积次数，每个 micro-batch 的每个 rank 至少获得一张，且不能超过上限。未指定 `--effective-batch` 时保持原有均匀分配行为。恢复仍检查实际有效 batch，不会重建 optimizer 或 warmup。

watcher 每分钟由系统 cron 启动短进程，避免常驻 Python supervisor 被清理后无法续跑。配置保存完整训练参数并沿用本地模型、索引、缓存及画廊路径；CPU 线程数及 GPU IDs 固定写入子进程环境。启动前每张指定卡至少需要 `--min-free-mib` 空闲显存；不足时本次检查退出，下一分钟重新检查。记录 PID 的进程身份、防止 PID 复用；手动启动的相同实验、仍存活的 rank zero 或启动器持有的运行锁都会阻止重复启动。普通训练代码错误会停止自动重试并记录 `training_error`，修复后重新安装可继续；外部进程清理、正常限时退出从完整断点恢复。不同 GPU 数量不保证逐位一致。

查看状态或停用自动续跑（停用不会杀掉现有训练）：

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
python scripts/watch_linear_pid_training.py \
  --run-dir /home/ganchangyi/code/PiD-Linear/outputs/linear-pid/kda_1-4-8-12 --status
python scripts/watch_linear_pid_training.py \
  --run-dir /home/ganchangyi/code/PiD-Linear/outputs/linear-pid/kda_1-4-8-12 --uninstall
```

要求当前训练在更新边界保存退出并阻止续跑时，创建运行目录的 `STOP` 文件；恢复前删除该文件，再安装 watcher。自动训练日志在运行目录的 `watch_logs`，用户 crontab 中只修改带本实验唯一 marker 的条目，保留其他任务。

在 node01 上手动运行：

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
GPU_IDS=0,1,2,3 CPU_THREADS=1 \
bash scripts/train_linear_pid.sh --preset 4gpu --layers 4 --lambda-out 0 --resume auto
```

默认每卡 batch 1、梯度累积 2，有效 batch 8。`--workers 4 --threads 1` 可覆盖 worker 和进程内 CPU 线程数。默认没有 step 上限；每次启动最多运行 516,600 秒，即六天减去 30 分钟。停止不会判定模型已恢复。

两卡完整运行时，将累积次数改为 4，仍保持有效 batch 8：

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
GPU_IDS=0,1 CPU_THREADS=1 bash scripts/train_linear_pid.sh \
  --preset 4gpu --layers 4 --grad-accum 4 --lambda-out 0 --resume auto
```

| 配置 | 默认值 |
|---|---|
| KDA 参数 LR / 继承主干 LR | `1e-4` / `1e-5` |
| AdamW / weight decay | betas `(0.9,0.999)`、eps `1e-8` / `1e-3` |
| warmup / 后续 LR | 500 optimizer steps / 恒定 |
| loss | 默认 FP32 FM loss；`--lambda-out 0` 关闭教师监督 |
| 梯度裁剪 | global norm 1 |
| 精度 | FP32 学生及 optimizer，BF16 autocast；VAE 权重及文本缓存 BF16；启用时教师权重 BF16 |
| EMA | 关闭，不创建副本、不更新、不保存 |
| 模型放置 | 学生、VAE 常驻 GPU；默认不创建 Gemma／教师；启用的模型也常驻 GPU |
| 显存措施 | MMDiT 与 PiT 非重入 activation checkpointing；复用 DDP 梯度缓冲；关闭学生 BF16 权重缓存 |
| 并行 | DDP；第一版不支持 context parallel |
| 保存 | 首个 optimizer 更新后保存一次；之后每 500 更新或 600 秒，先到者触发 |
| SwanLab | 默认离线；`--swanlab-mode cloud` 或 `disabled` 可覆盖 |

原 PiD 对 FM trainer 包装了负号，新的 raw network FM target 仍为 `noise - x0`。动态 shift、时间尺度和条件归一化与原模型保持一致。启用教师监督时，教师与学生使用同一份 noisy pixels、FP32 timestep、caption／latent dropout、latent noising 和编码结果。

需要重新开启教师监督时，使用独立输出目录：

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
GPU_IDS=0,1,2,3 CPU_THREADS=1 bash scripts/train_linear_pid.sh \
  --preset 4gpu --layers 4 --lambda-out 1 --resume auto \
  --output-root /home/ganchangyi/code/PiD-Linear/outputs/linear-pid-with-teacher
```

同阶段恢复严格检查 `lambda_out`，不会把原带教师 checkpoint 静默改为无教师训练；恢复原实验需显式保留 `--lambda-out 1`。若有意切换目标，应在新输出目录用 `--init-from <checkpoint> --init-weights raw --resume none` 开启新阶段，此时 optimizer 和 warmup 会重建。

训练计算及画廊生成均保持模型权重在 GPU，启用教师时条件 embedding 由师生共享。默认 `--text-cache-root /home/ganchangyi/dataset/MultiAspect-4K-1M/linear_pid_text_cache`，各 DataLoader worker 使用有界 mmap 读取 BF16 条件，训练端保持原 caption dropout，并使用独立的空 caption 缓存。离线文本条件使每卡无需保存约 4.87 GiB 的 Gemma 权重；`--online-text` 切回 GPU 编码。关闭 EMA 可省去每步 GPU→CPU 权重复制和 CPU 平均计算，以及约 5.23 GiB 的 CPU 副本；此前 EMA 本来就不占 GPU 显存，因此取消 EMA 不会释放同等大小的显存。checkpoint 保存仍将状态复制到 CPU 后序列化，保留 GPU 上的训练模型。

输出目录：`/home/ganchangyi/code/PiD-Linear/outputs/linear-pid/kda_1-4-8-12`。快速画廊在 stage step 0、100、500 及之后每 1,000 更新生成，两类各 8 张；完整 2K 画廊每 5,000 更新生成 96 张。生成使用当前学生、25 步、CFG 5、shift 6、batch 1，保存整图和三个统一局部裁剪，目录为 `galleries/step_<total>_raw`。无教师训练不会为画廊加载原模型；已有原 checkpoint 参考缓存时生成师生并排预览，否则只保存学生。参考输出可通过第 7 节的独立评估命令生成，按资产和采样设置缓存。

SwanLab 与 `metrics.jsonl` 记录 FM／输出 MSE、LR、梯度范数、样本数、耗时、吞吐、显存、教师／学生耗时以及 KDA 输出尺度与门统计。无教师时输出 MSE 和教师耗时记为 0，表示关闭，而非测得教师误差为零。不能用 loss 下降代替检查图像。

同阶段继续时重跑完整训练命令。严格恢复学生、optimizer、scheduler、每个 rank 的随机状态和全局 batch cursor；不会重新初始化 KDA。旧 checkpoint 中的 EMA 及 CPU 卸载配置被忽略，不会构造 EMA 副本。不同卡数恢复要求有效 batch 不变，随机状态按新 rank 数重新播种，不保证逐位一致。

## 5. 双节点 8 卡

只在你明确要求时提交。Slurm 默认使用独立的 `/home/ganchangyi/code/PiD-Linear/outputs/linear-pid-slurm/job-<jobid>`，不会与 node01 的训练共用输出目录。提交前创建日志父目录；以下命令在拿到资源时复制 node01 最新完整断点，并恢复学生、optimizer、scheduler 和数据进度：

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
mkdir -p /home/ganchangyi/code/PiD-Linear/outputs/linear-pid-slurm
LINEAR_PID_SOURCE_RUN_DIR=/home/ganchangyi/code/PiD-Linear/outputs/linear-pid/kda_1-4-8-12 \
LINEAR_PID_BATCH_SIZE=4 LINEAR_PID_GRAD_ACCUM=1 LINEAR_PID_EFFECTIVE_BATCH=32 \
sbatch scripts/train_linear_pid_8gpu.slurm
```

默认 account `students`、partition `gpujl`，每节点 4 GPU、1 个 torchrun 进程和 24 CPU，六天上限。按用户要求，Slurm 配置使用每卡 batch **4**、梯度累积 **1**、有效 batch **32**，FM-only、无 EMA／模型卸载、PiT chunk 2048；遵守 Slurm 分配的 GPU，不继承 node01 的 `GPU_IDS`。每 rank 4 个数据 worker、一个计算线程，均可通过环境变量调整。`LINEAR_PID_EFFECTIVE_BATCH` 未设置时按 GPU 总数、单卡 batch 与累积次数自动计算，不会以旧的有效 batch 8 把实际单卡 batch 限制成 1。

每实际 18,000 秒（5 小时）在 optimizer 更新边界保存完整 checkpoint，并运行 96 张完整 2K 画廊；画廊开始前一定已有同一步断点。关闭固定 step 保存及原来的快速／完整 step 画廊间隔，退出或停止信号仍触发安全保存。`LINEAR_PID_INTERVAL_SECONDS` 可覆盖时间间隔。此前估算的约 5.23 秒／step、五小时约 3,500 steps 对应每卡 batch 1／有效 batch 8，不能沿用到 batch 4。历史两卡、每卡 batch 4 的完整更新约 19.47 秒；若八卡更新耗时接近，五小时约 924 次更新，但八卡跨节点尚未实测，因此以实际时间触发。

独立续跑采用 `--resume-from`，完整保留同阶段状态并建立新 SwanLab run ID；它与重新初始化 optimizer 的 `--init-from` 不同。Slurm 脚本对独立源断点显式传入 `--allow-batch-size-change`，允许把 node01 的有效 batch 8 改为 32，同时保留学生、optimizer、scheduler、累计步数和样本计数，并记录 batch 转换。之后每次更新消费 32 张，同一 batch 的样本分组会改变；默认恢复要求有效 batch 不变；显式传入 `--allow-batch-size-change` 可在同一目录保留完整训练状态并记录有效 batch 变化。源断点复制到新根目录的 `initial_state/checkpoints`，完整复制后才发布，源训练不受影响。job 重新启动时优先恢复自己的最新 checkpoint；已有初始副本不再随 node01 的新权重改变。不设置 `LINEAR_PID_SOURCE_RUN_DIR` 时从原 PiD 开始。`LINEAR_PID_OUTPUT_ROOT` 可指定独立输出根目录，必须与源训练分开。

覆盖集群参数的示例：

```bash
sbatch --account=students --partition=gpujl \
  scripts/train_linear_pid_8gpu.slurm --workers 4 --threads 1
```

已分配的双节点资源中也可直接在两个节点分别运行，不再次申请资源：

```bash
# 两个节点都先 conda activate linear-pid，并切到同一仓库目录。
# MASTER_ADDR 换成该 allocation 的第一个节点名；两个节点使用同一端口。
GPU_IDS=0,1,2,3 NNODES=2 NODE_RANK=0 MASTER_ADDR=<first-node> MASTER_PORT=29571 \
bash scripts/train_linear_pid.sh --preset 8gpu --layers 4 --resume auto

GPU_IDS=0,1,2,3 NNODES=2 NODE_RANK=1 MASTER_ADDR=<first-node> MASTER_PORT=29571 \
bash scripts/train_linear_pid.sh --preset 8gpu --layers 4 --resume auto
```

## 6. 停止、checkpoint 与阶段扩展

要求正常保存退出时，在另一个终端执行：

```bash
touch /home/ganchangyi/code/PiD-Linear/outputs/linear-pid/kda_1-4-8-12/STOP
```

准备继续前删除该停止标记，再重跑训练命令：

```bash
rm -f /home/ganchangyi/code/PiD-Linear/outputs/linear-pid/kda_1-4-8-12/STOP
conda activate linear-pid
GPU_IDS=0,1,2,3 bash scripts/train_linear_pid.sh --preset 4gpu --layers 4 --resume auto
```

各 rank 也处理可捕获的 SIGTERM／SIGINT／SIGUSR1。Slurm 提前 30 分钟的信号通过共享停止文件通知 worker，等待当前 optimizer 更新后保存退出。SIGKILL 或 node01 强制清理只能从最近完整 checkpoint 恢复。

checkpoint 使用 `step_<total>_batch_<cursor>` 目录。先写 `.incomplete` 临时目录，fsync 完整状态后发布 `complete.json` 并原子改名。恢复忽略未发布目录及尺寸不符的目录；rank 0 的写入失败广播给所有 rank。默认保留最近三个完整 checkpoint。保留里程碑：

```bash
touch /absolute/path/to/checkpoint/KEEP
```

状态包含布局、局部卷积开关、原 PiD 初始化／参考文件 SHA256（兼容字段名 `teacher_sha256`）、FLA commit／运行版本、数据及文本缓存指纹、父 checkpoint、stage／total steps、采样器 cursor 和 SwanLab run ID，并记录 `ema_enabled=false`、`model_offload=false`、`teacher_enabled` 和 `training_objective`。无教师模式仍保留原始文件指纹，用于核对初始化来源。同阶段恢复拒绝布局、LR、warmup、loss 权重、裁剪、数据或已有文本缓存指纹不匹配。新实验应使用新的 `--output-root`。

检查 4 层质量后，才手动启动更高替换数量。以下是 6 层示例，不会自动执行：

2026-10-05 用户指定的 `K K K F K K K F K K K F K F` 十层配置和完整三卡／四卡／八卡命令见 [十层扩展阶段](linear_pid_kda10.md)。独立启动器首次固定父断点，只初始化新增层；重复运行自动恢复自己的阶段。原四层 node01 训练已保存退出，四层 Slurm job 已按用户要求取消。

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
GPU_IDS=0,1,2,3 bash scripts/train_linear_pid.sh \
  --preset 4gpu --layers 6 --resume none \
  --init-from /absolute/path/to/selected/4-layer/checkpoint --init-weights raw
```

先按父布局加载完整学生，保留已有 KDA 和主干，只随机初始化新增层，重新创建 optimizer 和 warmup，累计训练历史继续保留。后续继续该阶段时去掉 `--init-from`，使用 `--layers 6 --resume auto`。

| preset | KDA 层（零起始） |
|---|---|
| 4 | `[1,4,8,12]` |
| 6 | `[1,2,4,6,8,12]` |
| 8 | `[0,1,2,4,6,8,10,12]` |
| 10 | `[0,1,2,4,5,6,8,9,10,12]` |

也可使用 `--layers 1,4,8,12` 或 JSON 列表 `--layers '[1]'` 指定单层；`--kda-layers` 是同义入口。检查重复、越界以及父布局的包含关系。

## 7. 2K／4K 评估与独立学生推理

完整固定画廊及 4K 检查：

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
GPU_IDS=0,1,2,3 bash scripts/evaluate_linear_pid.sh \
  --checkpoint /absolute/path/to/checkpoint --suite qualitative
```

默认分配各样本到多张卡，输出完整 112 张、参考结果及裁剪；重复命令会跳过已完成样本。`--suite quick` 只做 16 张 2K 快速画廊。评估在独立进程进行，不加载 optimizer，也不加载 Gemma、VAE 或上游模型。`--no-reference` 可仅加载学生。

同步 GPU 后测单步网络及完整多步解码，预热后记录单步重复测量；结果在 `measurements_rank_*.json`。比较同一 GPU、BF16、分辨率、batch、25 步、CFG 5、shift 6 的结果。首次编译与图片保存时间不应计入解码测量。

导出当前学生权重（无需 GPU）：

```bash
conda activate linear-pid
python -m pid._src.linear_pid.evaluation \
  --checkpoint /absolute/path/to/checkpoint \
  --export /home/ganchangyi/code/PiD-Linear/outputs/linear-pid/student_raw.pt
```

导出仅包含学生、重建配置和必要元数据。评估、导出和独立采样默认 `--weights raw`；只有读取含 EMA 的旧 checkpoint 时，才可显式使用 `--weights ema`。新 checkpoint 指定 EMA 会报错，不会静默改用另一份权重。使用已有 normalized FLUX latent 独立采样，或使用 `--image /absolute/path/image.jpg --resolution 4096` 从本地图片构造条件：

```bash
conda activate linear-pid
CUDA_VISIBLE_DEVICES=0 python scripts/sample_linear_pid.py \
  --checkpoint /home/ganchangyi/code/PiD-Linear/outputs/linear-pid/student_raw.pt \
  --latent /absolute/path/to/normalized_flux_latent.pt \
  --caption 'A red fox standing in snowy woodland at sunrise.' \
  --output /home/ganchangyi/code/PiD-Linear/outputs/linear-pid/samples/fox.png \
  --steps 25 --cfg 5 --shift 6 --seed 42
```

独立采样不构造教师。KDA 每次网络调用都从零状态开始；文本在前、图像 raster 在后，有效文本打包并用序列边界隔离样本。图像 3×3 depthwise 残差卷积与文本 kernel 3 卷积互相独立且初始为零。条件及空条件各传自己的有效文本 mask，原 Full Attention／教师仍保持原默认推理行为。

## 8. 完整模型启动验证

已有 allocation 中进行以下短程验证。模型、训练范围、2K 分辨率和有效 batch 均与正式实验一致；验证输出使用独立目录，关闭画廊以测量实际训练开销。

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
GPU_IDS=0,1,2,3 bash scripts/train_linear_pid.sh \
  --preset 4gpu --layers 4 --resume auto --max-steps 20 \
  --output-root /home/ganchangyi/code/PiD-Linear/outputs/linear-pid/startup-verification \
  --quick-every 0 --full-every 0 --save-steps 10 --log-steps 1
```

重复该命令并将 `--max-steps 20` 改为 `40`，检查学生／optimizer／scheduler 和进度继续恢复。随后对该 checkpoint 运行完整评估命令。用 `metrics.jsonl` 的稳态耗时与显存测量估算六天可完成的更新数，并计入正式运行中的保存和画廊开销。

两卡测试时明确覆盖梯度累积，保持有效 batch 8。以下仍使用完整模型与真实 2K 数据，临时关闭画廊；`--index-root` 默认指向第 2 节准备的全量索引：

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
GPU_IDS=0,1 CPU_THREADS=1 bash scripts/train_linear_pid.sh \
  --preset 4gpu --layers 4 --grad-accum 4 --workers 2 \
  --max-steps 2 --quick-every 0 --full-every 0 \
  --save-steps 1 --log-steps 1 --resume auto \
  --output-root /home/ganchangyi/code/PiD-Linear/outputs/linear-pid/two-gpu-verification
```

将 `--max-steps 2` 改为 `3` 重跑以验证恢复。退出并完成检查后，删除此专用测试目录：

```bash
rm -r -- /home/ganchangyi/code/PiD-Linear/outputs/linear-pid/two-gpu-verification
```

若出现 OOM，先检查实现和显存策略，不自动冻结主干、减少层数或改变有效 batch。A40 上是否能容纳完整恢复训练、生成质量与实际加速幅度，都由这些运行结果确认。

2026-10-06 续训诊断补充：CUDA 同步检查定位到 FLA 的 KDA intra backward 算子。关闭索引张量缓存后已越过此前报错批次，此设置已持久保存到四卡 bs4 的自动续跑配置；底层根因尚未确认。运行命令见 `docs/linear_pid_kda10_direct.md`。


2026-10-06 输出目录修正：同一个十层直接初始化实验已统一回 `outputs/linear-pid-kda10-direct/node01/kda_0-1-2-4-5-6-8-9-10-12`。此前因 batch 变化创建的续训目录已合并，细节与当前完整命令见 `docs/linear_pid_kda10_direct.md`。更换卡数/batch 不再强制使用独立 `--resume-from` 或新输出目录；使用 `--resume auto --allow-batch-size-change` 即可，恢复时保留学生、optimizer、scheduler、进度和 SwanLab ID。
