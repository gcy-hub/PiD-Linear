# 模型与 Attention 实验框架重构计划

## Context

当前代码按历史实验分散在 `flux_kda/`、`flux2/`、`flux3/` 和 `scripts/`：模型替换、训练策略、缓存与启动参数相互依赖。目标是按模型、数据、attention 算法、训练策略、配置、启动脚本和下载职责组织代码，配置变化不再产生新实现或新启动脚本。

约束：尽量不修改 `pid/`；复用现有实现；保留在线编码和可选图像/文本缓存；模型与训练配置可分别复用；参考 VDN、CLEAR 对应源码；README 说明完整流程。规划期间只修改本计划。

交付包含代码实现、durable 子 agent 并行协作，以及 **2026-11-11 03:30（Asia/Shanghai，UTC+8）** 启动的本机 GPU 0/1/2/3 正确性验收。使用 agents-cron 持久化定时任务；测试通过后提交并推送 `origin/main`。批准计划后即开展实现并登记任务，不等到 03:30 才开始编写代码。

## HERO 实施约束

采用 [HERO-Anti-OverDefense / RULES.md](https://github.com/wanshuiyin/HERO-Anti-OverDefense/blob/main/RULES.md) 的范围规则，贯穿实现、子 agent 任务和集成审阅。操作者是本机合作者，工作范围为已确认的三种模型、四种 attention、两种策略及其有效组合。

- 从现有实现抽取能完成当前任务的最小模块；使用直接函数调用、简单数据结构和明确分派。目录表示职责，不要求每项都建立基类、注册器或包装层。旧 checkpoint 兼容和未来插件扩展不在实现范围。
- 配置、外部数据和 checkpoint 的必要校验放在读取边界；内部沿用已成立的形状与参数约定，不逐层重复校验。真实数据、文档示例或支持的配置可达的问题都要报告并修复。
- 直接比较已有配置与元数据，复用 Git commit、durable request/workflow ID。仅当摘要能省掉实质上更贵的操作、且结果改变下一步时才使用，不为验收额外生成哈希或证据文件。
- 每次检查须能指出尚未解决的具体失败及对应行动；已有测试或实际运行覆盖同一路径时复用结果。修复后重跑受影响项，集成审阅针对实质缺陷，不另建评分表或反复审计流程。
- 用户要求的四卡验收、定时执行、缓存/OOM 开关，以及项目已发生的重启、保存失败和 BF16 更新丢失，属于本次功能需求，继续实现。普通可逆选择由负责 agent 决定并推进；限制集中记入 Limitations。

## Approach

- 基础模型与 attention 替换适配分离，配置指定模型家族、替换层与算法参数。
- attention 算法与训练策略分离：交叉层扫描 KDA、Sliding QDM、QDM Halo、CLEAR attention；VDN 三阶段和 CLEAR 蒸馏分别作为训练策略。
- 数据加载、分桶、预处理、在线编码和缓存统一放入数据模块，同一训练样本接口覆盖在线/缓存输入。
- 单一训练启动入口读取模型配置与训练配置；下载、缓存、采样为独立任务入口，不按 GPU 数和 batch 拆脚本。
- 现有 `pid/` 通过外部适配器调用；不得将所有历史实现机械搬入新目录后继续维持多套训练器。

已确认：PiD、FLUX2、FLUX3 都须支持四种 attention（共 12 种模型/算法组合）；CLEAR 同时提供独立 attention 方案和蒸馏策略。交叉扫描 KDA、Sliding QDM、QDM Halo 可选两种训练策略；CLEAR attention 仅使用 CLEAR 策略，共 21 个有效模型/算法/策略组合。新框架从新训练开始，不导入或续跑旧 checkpoint；历史实验文件保留。

已确认的拆分依据：`flux_kda/model.py` 用名字后缀决定训练参数并直接耦合 CLEAR；`flux_kda/train.py` 的在线模式仅支持 FLUX3。模型构建、算法选择和训练参数选择必须分离，补齐 PiD/FLUX2 在线编码。

## 源码核对

- VDN 官方版本：[OpenVDN/vdn-minimax-h3](https://github.com/OpenVDN/vdn-minimax-h3/tree/e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b)，revision `e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b`。`src/training/train_stage_a1.py:50–95`：teacher 主干继续执行，student 从 detached 层输入独立计算并立即 backward，目标为 `sum((student-teacher)^2)/sum(teacher^2)`；A1/A2 默认冻结 softmax gate。`train_stage_a2.py:43–64`：端到端 teacher prediction MSE。`train_stage_b.py:52–102`：新增 LoRA、训练分支与门控，发布配置使用 teacher objective。`fsdp_stage.py:95–112`：teacher 关闭 LoRA、共享冻结基础模型。结构化 AdamW 分组与 warmup/cosine 跟随策略配置，不能写死到算法中。
- CLEAR 官方版本：[Huage001/CLEAR](https://github.com/Huage001/CLEAR/tree/ddca13160b7c2fe5d340c8a421db72b7bdaa6ba9)，revision `ddca13160b7c2fe5d340c8a421db72b7bdaa6ba9`。`attention_processor.py:252–260` 为圆形邻域 `dy²+dx² < window_size²`，文本行/列全局可见；`117–153,221–226` 为带可训练池化权重的下采样 K/V 变体。`distill.py:690–739,1069–1074`：训练图像侧 QKVO 与池化权重，FM + 0.5 prediction KD + 0.5 feature KD，官方 FLUX1 在选中的 single blocks 收集 pre-O 特征。移植到其他模型须显式配置对应层，不能依靠模块遍历序号碰巧对齐。
- 本地 `../CLEAR` 是 `gcy-hub/CLEAR-MultiFlux` 适配仓库，可复用三模型编码经验，但不标为作者原版。已有 `flux2/flux2_qdm/vendor/clear_attention.py` 支持显式多样本布局、圆形/方形窗口、可选池化，适合收敛为统一算子。保留来源和许可证。
- 用户已选择：CLEAR attention 不使用 VDN 微调；不实现旧 checkpoint 的权重导入或优化器迁移。

### 模型边界与数据语义

- `pid/` 保持实现不动，外部 `models/pid.py` 复用原生 PidNet、基础权重加载；替换 patch MMDiT 的 attention，原生 pixel blocks 保留。FLUX2 适配 double/single blocks，FLUX3 适配 early-image/joint blocks；各家族以稳定层名选择 all 或指定索引，未选层和文本/参考条件模块保留原生行为。
- PiD 的训练目标是完整 RGB 像素，VAE 只编码下采样条件。缓存保存干净条件，随机噪声、退化、dropout 仍在训练时生成。FLUX2 使用其原生 VAE 打包和归一化；FLUX3 使用原生 latent tokens。统一的是数据生命周期，不统一为错误的同一种潜空间。
- KDA 现有公式为 `O(local + KDA)`；QDM 为 `O(local) + memory`，memory 自带输出投影。公共 attention 结果提供 `pre_o` 和可选 `post_o_residual`，模型适配器恰好执行一次原生 O，保留 fused FFN、残差和调制。原始 QKV 与带位置编码 QK 明确区分，QDM/KDA 的输入位置不因抽取而改变。
- CLEAR feature KD：KDA/CLEAR 默认使用 pre-O 特征；QDM 使用包含 memory 的 post-O 特征，teacher 对齐相同位置，避免蒸馏损失漏掉远程分支。采样层名和 feature tap 写入解析后的配置及日志。


## 目标目录与依赖

以下目录列出职责与预期归属；小功能可留在同一模块，不为凑齐文件名拆文件。

```text
models/                 # 三种基础模型加载、层定位、替换适配、原生前向与采样
  pid.py / flux2.py / flux3.py
  attention.py          # 布局/结果约定、投影切片、teacher 与训练投影隔离
  vendor/DiffSynth-Studio/  # 迁入现有运行时，保留来源与许可证
algorithms/             # 不加载模型、不读取数据、不创建优化器
  cross_scan_kda.py / sliding_qdm.py / qdm_halo.py / clear.py
  local.py / qdm.py      # 共用邻域算子、QDM 求解与区域布局
data/                   # 数据处理与编码生命周期
  dataset.py / buckets.py / sampler.py / preprocess.py
  encoders.py / cache.py / prepare.py
training/
  run.py / engine.py / distributed.py / checkpoint.py / optim.py
  strategies/vdn.py / strategies/clear.py
  validate.py / config.py
configs/
  models/pid.yaml / flux2.yaml / flux3.yaml
  algorithms/cross_scan_kda.yaml / sliding_qdm.yaml / qdm_halo.yaml / clear.yaml
  training/vdn.yaml / clear.yaml
  data/default.yaml
scripts/                # 薄启动入口：训练、采样、缓存、下载、Slurm
download/               # 复用下载与断点重试实现
pid/                    # 原生实现与现有共享工具，保持不动
```

依赖单向：训练入口组合模型、算法、数据；算法只接收张量和布局；数据不导入训练器；模型加载不依赖训练策略。模型编码器的调用由 `data/encoders.py` 管理，原生模型类由模型模块提供。不建立插件系统或重复的通用 Trainer 层。

### 模型与算法接口

- 基础模型配置包含架构、权重/TE/VAE 资源、原生噪声与采样约定。构建入口按三种家族直接分派；适配器集中处理命名层、QKV/O 切片、token 顺序、RoPE、条件输入和前向特征，公共训练循环复用这些操作。
- 修改模型由基础模型 + 算法配置 + 替换层列表构成，同一类适配器同时服务四种算法，不再为每种算法派生一套 FLUX block。未选层执行原始 attention；原生 teacher 绕开全部替换分支、LoRA 与直接投影更新。
- attention 输入包含归一化 hidden states、原始 QKV、原生位置处理后的 QK、每样本图像网格/文本范围与稳定层序号。布局显式处理 image-first、text-first、变长和多样本；不把固定文本长度、全局 grid 变量或同 batch 样本互相可见带入公共算子。
- `cross_scan_kda` 保留现有 Sliding + 标准 FLA KDA：每层只扫描一次，按原生图像 attention 层序号轮换行正/行反/列正/列反，再逆置换回二维顺序，调用间不保留状态。选择部分层时方向不重新编号。KDA 仅作用图像 token。
- `sliding_qdm` 与 `qdm_halo` 共用 QDM 记忆核心，区别是逐像素滑窗与不重叠 query 区域加 halo 的局部分支。保留现有四区域路由、padding mask、文本可见性、原始 QKV 输入及分支投影位置；窗口、区域和 halo 大小属于配置。
- `clear` 保留官方圆形窗口与可选下采样 K/V；文本查询与文本 K/V 的可见性、同样本约束显式实现。方形 Sliding 和 CLEAR 圆形窗口复用布局与执行底座，但不混同窗口参数。
- 训练策略通过显式参数角色选择 `branch`、`local_gate`、`image_qkvo`、`lora`、`pooling` 等集合；模型构建不设置策略特有的 `requires_grad`，不依赖字符串后缀识别全部参数。FLUX2 single 层的 QKV 行/O 列切片不得更新融合 FFN 权重。

### 两种训练策略

| attention | PiD | FLUX2 | FLUX3 | 可用策略 |
|---|---|---|---|---|
| cross_scan_kda | 支持 | 支持 | 支持 | VDN / CLEAR |
| sliding_qdm | 支持 | 支持 | 支持 | VDN / CLEAR |
| qdm_halo | 支持 | 支持 | 支持 | VDN / CLEAR |
| clear | 支持 | 支持 | 支持 | CLEAR |

- VDN A1：冻结主干与独立 local gate，从 teacher 的 detached 输入逐层对齐，立即反传；统一比较实际被替换的图像-query attention 输出（含 O 和远程分支、不含残差/FFN），按该范围计算相对 MSE。这是面向 joint image/text 模型的明确适配。A1 不套用全模型 DDP 的重复 backward：复用现有逐层反传、累积后显式梯度平均与逐层裁剪流程。
- VDN A2：组装 student，全模型前向对齐冻结 teacher 的预测，训练新增分支，主干与独立 local gate 仍冻结。B：训练分支、已有 gate 与 QKVO LoRA；teacher 绕开 LoRA。KDA 自身的内部 gate 属于 branch，不额外发明 softmax gate。
- VDN 使用 AdamW 与角色参数组；官方 A1=200/A2=500/B=2000、warmup=50/50/200、rank=alpha=64 作为默认参考写在一份策略配置中，阶段步数、数据源和目标可覆盖。A2/B 使用 DDP 与梯度累积。阶段切换重建规定的优化器集合，新框架断点记录阶段和阶段内步数。
- CLEAR：复用 FM、teacher prediction MSE、attention feature MSE，默认权重 1/0.5/0.5；训练选中层的图像 QKVO 与算法分支/池化参数，其余主干、TE、VAE 冻结。对 QDM 的 post-O feature KD 和对其他模型的特征层映射标明为移植约定。发布默认采用 Prodigy、常数调度和 FP32 更新累积，避免 BF16 小更新舍入；精度处理归训练模块。
- 公共 engine 负责设备、数据、计步、保存与日志，策略负责参数组、阶段和本次更新方式。A1 的独立反传与端到端训练使用各自明确步骤，不强行套成一个只返回标量 loss 的接口。

### 数据、缓存和 OOM 选项

- dataset 读取实际 manifest/metadata 数量；复用现有 JSONL、CSV 与索引读取逻辑，输出样本 ID、相对图像路径、caption、尺寸。分桶与采样独立，默认复用 `pid/_src/datasets/buckets.py` 的官方 2K–4K 网格，可配置小分辨率桶；保留 EXIF、等比缩放/裁剪与拒绝过小样本的统计。
- `data/encoders.py` 为三种家族保留原生预处理/latent packing/文本 token 和 IDs。只统一返回训练所需信息的约定，允许 PiD 的像素目标与低分辨率条件并存。在线编码使用 `no_grad`，产物可参加后续有梯度计算。
- `cache.image`、`cache.text` 独立开关，支持全在线、仅图像、仅文本和全缓存。两者都缓存时训练进程不构建 TE/VAE；只缓存一种时只加载另一种编码器。PiD 缓存条件图像并继续加载 RGB 目标；训练随机性不缓存。
- 缓存准备复用同一预处理和编码函数，支持设备列表、batch、跳过已完成项、原子写入和失败样本记录。读取边界比较现有模型/编码器标识、tokenizer、尺寸桶、预处理设置与样本身份，以决定复用或重算；一致内容可更换根目录，训练内循环直接读取，不反复扫描或校验大文件。
- 梯度 checkpointing、优化器分片、激活/优化器 CPU offload 继续可配且默认关闭；算法内部必需的 FP32 状态和优化器可靠更新不作为可省略的 OOM 开关。保持固定 world size 的分片恢复边界，未测试配置不能声称不会 OOM。

### 配置与启动

复用已安装 OmegaConf 的 load/merge/dotlist，实现一条确定的覆盖链：共享默认 → 模型/算法/训练/数据 YAML → 命令行覆盖。使用已有依赖，不新增配置框架。每个模型只有一份基础配置，每个策略只有一份配置，层列表、窗口、步数、batch 和路径通过覆盖修改。

```bash
python -m training.run \
  --model-config configs/models/flux3.yaml \
  --algorithm-config configs/algorithms/cross_scan_kda.yaml \
  --training-config configs/training/clear.yaml \
  --data-config configs/data/default.yaml \
  paths.weights=./weights paths.dataset=./dataset/pid-vdn-80k \
  paths.output=../experiment/flux3-kda \
  runtime.gpus=8 training.batch_size=4 training.accumulation=1 \
  training.steps=50000 checkpoint.every=2000 validation.every=2000
```

模型配置中的 `replacement.layers` 按模型原生组名定位，例如 FLUX3 `{early_image: all, joint: [0, 4, 8]}`；未选组不替换。修改算法参数示例为 `algorithm.local.window_size=15`，算法与文件名称不包含 11/4K/8gpu 等参数。解析后校验未知字段、层索引、策略组合、窗口合法性和缓存配置，提供 `--print-config`；全局 batch 由 per-GPU batch × world size × accumulation 派生，不另设相互冲突的值。相对路径以启动工作目录为基准，在切换子进程 cwd 前解析；数据集内部图像路径以 manifest 所在目录为基准，解析结果写入实验配置。

下载入口读取同一模型资源配置，复用 HF 下载和已有数据集元数据/图像断点下载；训练不隐式联网或下载。`scripts/` 保留通用训练、缓存、采样、下载、Slurm 启动器，业务逻辑移入上述模块；真实多阶段/合成数据编排复用新入口，参数变体脚本删除。

### 保存、验证与目录收敛

- 新框架保存基础模型身份、替换层/算法配置、可训练权重、优化器/调度器、RNG、数据 cursor、阶段与总步数；默认只保留最近一次完整恢复集合。分片先各自保存，完整发布后再淘汰上一集合；不回到大对象 NCCL 汇聚。
- 验证沿用 `flux_kda/run.py` 的进程边界：到保存/验证/阶段边界后保存，所有训练 rank 退出，再独立采样，完成后恢复训练。验证期间无训练 NCCL group 等待；中断后先补该 checkpoint 未完成的验证。保存与验证频率独立配置，需要验证的边界也发布可恢复 checkpoint。
- 现有 `flux_kda/`、`flux2/flux2_qdm/`、`flux3/flux3_qdm/` 中所需实现按职责迁入新目录，移除被替代训练器、重复算法/缓存逻辑和参数脚本；DiffSynth 保留一份 vendored 源码，修正导入与打包，不让 FLUX2 从 FLUX3 实验目录导入。
- `pid/` 原生训练和工具继续保留。必须维持的外部调用入口（如原生 DMD 或 `pid/` 内调用的启动器）只保留薄入口，不维持第二套新框架；迁移范围以调用关系确定。历史实验、权重、日志和其他 Git 分支不做清理。

## Durable 并行实施

使用 **pi-durable-subagents 的 `subagents` / CLI** 执行，工作流、子任务 ID、交付 commit 与依赖均持久化。所有 agent 显式指定 zzz provider，不继承当前会话默认模型，不使用含其他 provider 的 fallback pool。本机实际注册的模型名与执行设置如下：

| 工作 | Durable 模型 |
|---|---|
| attention 算法及训练流程的实现、相应缺陷修复 | `zzz/gpt-6-astra:high` |
| 模型加载/适配、数据、配置、工具、测试编写、审阅、文档、集成与提交 | `zzz/gpt-6.1-sol:high` |

这里 `:high` 指推理级别，`gpt-6-astra-high` 不是本机的 model ID。agents-cron 入口也固定 `--agent pi --model zzz/gpt-6.1-sol --thinking high`。本会话只负责调度和向用户报告，实际修改按上述分工交给指定模型。

| 子任务 | 专属写入范围 | 模型 | 依赖与交付 |
|---|---|---|---|
| C：共享约定与集成 | 最小共享数据结构、公共配置组合、包发现、集成提交 | sol high | 先写清 attention 输入/结果、参数角色、样本与 feature taps；收取各工作树 commit 并进行一次集成审阅 |
| A：算法 | `algorithms/`、对应算法配置和局部检查 | astra high | 基于 C，抽取四种算法；保留投影位置与数值参考 |
| M：模型 | `models/`、`configs/models/`、vendor 导入 | sol high | 基于 C，与 A 并行；实现三家族加载、层定位和原生适配，不修改算法实现 |
| D：数据 | `data/`、`configs/data/`、`download/` | sol high | 基于 C，与 A/M/T 并行；实现原生编码、缓存和实际数据计数 |
| T：训练 | `training/`、`configs/training/` | astra high | 基于 C，与 A/M/D 并行；阶段执行、损失、参数组、优化器、分布式、恢复与验证生命周期 |
| V：验证与交付工具 | `tests/`、薄 `scripts/`、README/docs、定时任务描述 | sol high | 根据 C 先写验收场景；集成 A/M/D/T 后接真实接口，不改写算法或训练逻辑 |

执行顺序：`C 共享约定 → 并行 A/M/D/T（V 可同步准备测试）→ C 集成与审阅、V 验证 → 定时 GPU 验收 → C 提交推送`。durable workflow 脚本协调依赖；C 每阶段交付后释放写锁。接口变化直接通知受影响任务，普通实现选择由所属 writer 决定。集成审阅发现的实质问题交回所属 writer 修复；算法和训练实现仍由 astra 完成。

- 每个并行 writer 使用 durable 的 `isolation: "worktree"`，从同一个契约 commit 开始。主代码目录仍为 `/home/ganchangyi/code/PiD-Linear`；隔离工作树是本次实施的临时资源，集成且验收完成后回收，不再形成长期并行代码目录。主分支同一时刻只有集成任务写入。
- 每个任务携带起点 commit、允许写入范围、共享约定、完成条件和上述 HERO 实施约束；保留 `pid/` 与历史实验目录。模型设置显式固定，交付 commit 和实际检查结果。子任务发现真实问题时报告并就地提出范围内的修法。
- 工作流定义及任务简报放入项目的执行资料，运行记录放在 `../experiment/attention-framework-refactor/`；使用 durable request ID 幂等提交，重启后读取现有工作流和 Git 状态再继续，不重新扇出同一批任务。
- 新调度的长期工作通过 durable CLI 持久化 spec 启动，清除继承的 `DSA_SESSION`，不绑定短命的 cron Pi 会话，也不修改全局 `onQuit` 设置。需要承接已有交互工作流时，只恢复本次记录的 workflow ID。失败、模型不匹配或工具不可用时保留现场并报告，不切到官方 Codex 或另起 CLI agent 绕过失败。

## 03:30 定时四卡验收与推送

### 登记与重启恢复

1. 批准后登记 agents-cron 一次性任务：`--at 2026-11-11T03:30:00+08:00`，工作目录绑定本仓库，prompt 引用本计划和验收工作流，固定 zzz/sol high。回报 task ID 和 next run time；当前规划阶段尚未创建任务。
2. 安装带标记的每分钟 `agents-cron run-due --all` 系统 crontab 项，配置 Node/Pi 运行路径，保留其他项目任务及历史暂停项。机器路径只进入本机调度配置。
3. durable 保存工作流与结果，cron Pi TaskSession 只交接持久化 spec；清除 `DSA_SESSION` 后启动，使短会话退出不暂停验收。针对每日重启和 once 默认 5 分钟错过窗口，仅保留一个项目衔接入口，由每分钟驱动和 `@reboot` 共用：到点后尚未交接且 cron 不在执行时补交接；已有工作流则按保存的 ID 接续；完成或明确失败即结束。锁、幂等请求和运行状态直接使用 agents-cron/durable，额外只保存定位本任务所需的 ID 与计划时间，不另建任务数据库、重试队列或通用恢复框架。
4. 到点代码未就绪时继续完成集成，GPU 被占用时等待资源；由同一工作流继续验收，记录实际开始时间。用一个有界的衔接测试覆盖到点启动、进程中断后接续与完成后不重复交接；测试本项目新增逻辑，不重复验证调度器内部实现，也不主动重启机器。

### 四卡安排与失败处理

- 03:30 起使用 **物理 GPU 0、1、2、3**。集成代码先通过 CPU 配置/接口检查，固定测试起点 commit；测试期间暂停该分支上的无关写入。资源由测试控制器统一分配，durable command 使用 `pi-durable-subagents hold machine -- ...` 包住本轮 GPU 验收，防止本项目的单卡并行与四卡 DDP 同时抢卡；运行前检查外部占用，lease 不视为独占整台共享机器的保证。
- **第一轮并行单卡**：GPU0 跑 PiD 真实层/编码器，GPU1 跑 FLUX2，GPU2 跑 FLUX3，GPU3 跑公共 attention 的 CUDA/reference、梯度与 teacher 隔离检查。三家族各覆盖四种 attention；21 个有效策略组合用小模型覆盖更新与阶段语义。大组件逐项加载、释放，测试输出分别保存。
- **第二轮四卡分布式**：待单卡进程全部退出，GPU0–3 共同运行 `torchrun --nproc_per_node=4`。依次检查 A1 手动梯度平均、A2/B/CLEAR DDP、梯度累积、FP32 master 更新、分片保存/恢复与验证进程隔离；真实模型按家族顺序运行短程训练→保存→恢复→生成闭环。采用小分辨率、每卡 batch 1 和明确记录的 OOM 选项，以验证代码正确性。
- 真实在线编码与四种缓存组合按 Verification 执行；完整模型遇到显存不足时只能使用已有的可选内存机制或缩小测试输入，并记录实际配置，不能悄悄跳过该项或把 mock 测试当成真实模型通过。缺少权重、显存或依赖的必需项记为 blocked，全部必需项通过后才允许推送。H800 八卡 4K/batch4 的性能与显存验证仍是单列的未验证目标，不由本次四卡验收代替。
- GPU 测试使用 durable **command call**。失败交给所属 writer 修复，再跑受影响项；没有新改动或新证据时，不重复运行已经通过的检查，也不循环重试相同失败。测试与代码结果由 C 汇总后提交。

### 提交与报告

- 验收完成后由 `zzz/gpt-6.1-sol:high` 集成任务核对已测试版本、必要检查和工作区内容，提交代码及精简的验收说明，随后普通 `git push origin main`。用户已授权这一步，无需在凌晨再次等待交互确认；中间子任务可本地提交，远端只接收通过验收的集成结果。
- 推送前 fetch 并检查远端变化。如有更新，集成后补跑受影响测试，再正常推送；不 force push，不提交运行日志、大权重、缓存、密钥或本机调度状态。若重启发生在 push 返回前，先比较本地与远端 commit，避免重复提交。
- 原始日志、测试结果、GPU/精度/分辨率、失败项和最终 commit 放在 `../experiment/attention-framework-refactor/`；仓库中保留可复用测试及简洁验收文档。定时任务的运行报告保存在 `.agents/cron/`，不等同于最终通过报告。
- 当前没有通知渠道；默认通过本地报告和远端 commit 交付，不发送邮件或 webhook。如需外部完成通知，可另选 agents-cron 的通知配置，不影响本次执行。

## Files to modify

| 目标文件/目录 | 主要来源或改动 |
|---|---|
| `models/{pid,flux2,flux3,attention}.py` | `flux_kda/model.py`、`flux_kda/flux3.py`、FLUX2/3 `model.py`、FLUX2 `diffsynth_model.py`；PiD 调用原生实现 |
| `models/vendor/DiffSynth-Studio/` | 迁移 `flux3/DiffSynth-Studio/`，保留 `LICENSE`、`UPSTREAM.md` 与包资源 |
| `algorithms/{cross_scan_kda,sliding_qdm,qdm_halo,clear,local,qdm}.py` | `flux_kda/attention.py`、FLUX2 `memory.py`、FLUX3 `attention.py`/`sliding.py`、CLEAR vendor；PiD helper 复用 |
| `data/{dataset,buckets,sampler,preprocess,encoders,cache,prepare}.py` | `flux_kda/online.py`、FLUX2/3 `data.py`/`runtime.py`/`prepare.py`、PiD common 数据/编码器 |
| `training/` | `flux_kda/{train,run,checkpoint,optimizer}.py`、VDN/FLUX QDM 训练流程、CLEAR strategy |
| `configs/{models,algorithms,training,data}/` | 将现有 family/hardware JSON 和 CLI 参数列表合成可复用配置；停止保留参数变体 |
| `scripts/`、`download/` | 更新所有新入口调用方、现有下载器、通用 Slurm；删除被替代包装脚本 |
| `README.md`、`docs/`、`scripts/README.md`、`download/README.md` | 改为新结构与完整使用流程，清理过时路径和训练指令 |
| `pyproject.toml`、`.gitignore`、`tests/` | 新包发现与资源、生成物路径、保留有意义测试并迁移导入 |
| `scripts/` 中的验收入口、durable workflow 与任务简报 | 四卡测试、结果汇总、依赖与模型固定；复用现有 durable/agents-cron，不另造任务调度器 |
| 本机 agents-cron registry 与带标记的 crontab 项（非仓库源码） | 批准后登记 03:30 任务、短恢复入口；不改变其他项目任务 |

## Reuse

| 现有实现 | 保留和抽取的内容 |
|---|---|
| `flux_kda/attention.py`：`scan`、`ImageKDA` | 矩形网格四方向置换、标准 FLA 层、内部 FP32 参数；不重写 KDA 递推 |
| `flux2/flux2_qdm/memory.py`：`joint_memory`、`fixed_routes`、`pack_regions`、`halo_indices`、`unpack_regions` | QDM 的 FP32 求解、区域路由、halo 布局；收敛 FLUX3 重复实现 |
| `flux2/flux2_qdm/vendor/clear_attention.py`：`AttentionLayout`、`ClearAttention` | 显式布局、圆形/方形邻域、可选池化、FlexAttention/CPU 参考路径 |
| `flux2/flux2_qdm/diffsynth_model.py`：`AttentionProjection` | 冻结 teacher 投影、直接更新与 fused QKV/O 切片；移出模型家族专属命名空间 |
| `flux3/flux3_qdm/lora.py`：`LoRALinear`、`base_linear` | LoRA 与 teacher 原始投影隔离；复用 PiD 的分段 QKV 处理约定 |
| `pid/_src/common/runtime.py`：`build_net`、`load_original`、`sample` | 直接调用 PiD 构造、加载和原生采样；编码器组合在外部拆为按需加载 |
| `pid/_src/datasets/buckets.py` | 直接导入官方分桶定义，不复制一份桶表 |
| `flux_kda/online.py`、FLUX2/3 `data.py` 与 `runtime.py` | manifest、图像预处理、确定性 sampler；模型不同的训练输入构造迁入相应适配器 |
| `flux2/flux2_qdm/clear_strategy.py`：`objective`、`make_optimizer` | CLEAR 损失和 Prodigy 设置，参数选择从算法构建分离 |
| `flux3/flux3_qdm/training.py` A1 callback/梯度同步；`pid/_src/vdn_pid/training.py` | 复用已经存在的逐层反传模式、阶段训练语义；不调用三个历史训练器完成“统一” |
| `flux_kda/optimizer.py`：`MasterProdigy`；`checkpoint.py`；`run.py` | 保留可靠 FP32 更新、rank-local optimizer 保存、训练/验证进程隔离 |
| `download/download_models.py`、`download_metadata.py`、`download_images.py` | 显式下载、元数据整理和可恢复图像下载；资源列表来自统一模型配置 |

## Steps

- [x] **0. 登记执行任务与 durable 分工。** 批准后建立持久化任务简报、显式 zzz 模型设置和隔离工作树；登记 2026-11-11 03:30 的 agents-cron 任务、系统驱动与项目恢复入口，回报实际任务 ID。
- [x] **1. 建立新配置与模块骨架。** 固定参数命名、支持矩阵和数据/attention 约定，接入 OmegaConf 与包发现；先提供可打印的解析配置。记录旧实现的最小数值参考，不更改 `pid/` 或历史实验目录。
- [x] **2. 抽取 attention 核心。** 迁移标准 KDA、共同 Sliding/CLEAR 布局与 QDM 核心，消除 FLUX2/3 重复；用已有输入覆盖输出、梯度、halo 边缘和 token 可见性。
- [x] **3. 完成三模型适配。** 迁移共享投影/LoRA helper 与唯一 DiffSynth，完成 12 种组合、指定层替换、teacher 和 feature taps。基础模型无替换时对齐原生输出，保证 O 投影和 fused FFN 位置不变。
- [x] **4. 合并数据与编码。** 迁移 loader/分桶/sampler，完成三种家族在线编码、两种缓存的独立准备与读取；保持 PiD 像素目标和各模型原生 preprocessing。下载器输出可直接接入数据准备，实际样本数由数据决定。
- [x] **5. 合并训练执行。** 公共 engine + 两种策略，实现 A1/A2/B 的差别、CLEAR feature taps/损失与参数组、确定性训练、精度和可选 OOM 机制。清楚记录每阶段训练参数数量及角色。
- [x] **6. 接入保存、验证和任务入口。** 新断点完整续跑；分片保存与 latest 发布；训练 rank 退出后验证；统一训练、采样、缓存、下载和 Slurm 启动器，保留必要的原生入口。
- [x] **7. 删除替代路径并更新文档。** 按调用图移除重复训练器/算法/参数脚本，迁移仍有价值的诊断工具，不保留跨家族实验目录依赖。完成 README、配置注释、下载说明和所有示例。
- [ ] **8. 定时四卡验收并交付。** CPU 检查在实现期间完成；2026-11-11 03:30 起按四卡安排运行 GPU/恢复检查，修复失败并重测。全部必需项通过后由 sol high 提交并推送 `origin/main`，保存报告与远端 commit。H800 八卡配置单列硬件限制，不用本次四卡测试代替。

README 必须包含：目录职责与依赖图；12 个模型/算法及 21 个策略组合；基础/替换模型的关系；一次基础模型配置复用到不同算法的示例；如何选择层、修改窗口/batch/步数；VDN 与 CLEAR 的源码来源、参数角色和移植差异；下载→数据/桶准备→可选缓存→训练→验证/恢复的可复制命令；缓存何时能省掉 TE/VAE；OOM 开关、FP32 更新与全局 batch；新算法/新模型各需实现的最小接口及测试。默认输出为 `../experiment/`，路径全部可配置。

## Verification

检查只针对下面列出的实际失败路径，优先复用已有测试与一次真实闭环的结果。12 个模型/算法和 21 个有效策略组合分别做小规模参数化检查；缓存模式、精度与 OOM 选项按各自行为测试，不把所有开关与全部组合相乘。目录移动等机械修改通过导入/入口检查完成，不为每次移动新增测试。检查失败时修复对应实现或报告具体资源阻塞；成功后推进交付。

1. **配置与数据（CPU）**：参数覆盖和未知参数错误；数据规模自动识别；官方桶、矩形图像、EXIF、边缘 padding、小桶采样和多 rank/accumulation 分配；解析配置与运行路径一致。下载命令用假响应验证资源选择/跳过，不为测试重新下载模型或整个数据集。
2. **算法与适配（CPU reference + CUDA）**：全部 12 种组合的替换层、形状、前向/反向；未替换层与原生全局 attention 一致；文本可见性和样本隔离；部分层选择不改变 KDA 方向；QDM/KDA 投影位置正确。现有组合与抽取前数值参考比对，新增组合与密集/逐样本参考比对。
3. **训练策略（小模型覆盖 21 个有效组合）**：VDN A1 teacher trunk 不被 student 覆盖、逐层梯度独立、跨 rank 平均与单进程全局 batch 一致；A2/B 参数组与阶段切换；CLEAR 的实际被选 feature tap 梯度确实到达远程分支；teacher 的冻结权重与输出在更新 student 后保持不变；错误的 CLEAR+VDN 组合明确拒绝。
4. **真实编码与缓存（逐模型 GPU）**：同图/同文在线与缓存结果在相应 dtype 容差内一致；四种缓存开关只加载必要组件；空 caption、不同文本长度和不同 bucket；PiD 仍含完整 RGB 目标，改变随机种子会改变噪声/退化而不改变缓存内容。除构造计数测试外，实际检查不加载对应编码器。
5. **新断点恢复（单卡与分布式）**：短训练连续运行与中断恢复的下一步、参数/优化器、RNG、sampler cursor 一致；VDN 在阶段内和边界恢复；latest 发布前异常保留上一次完整集合；恢复时缺少 rank shard 明确报错；验证中断后先补图再继续，验证期间无训练 NCCL group。测试 FP32 master 的细小更新跨保存恢复保留。
6. **真实模型冒烟**：三种原生模型逐一加载并在实际层形状上检查四种算法；每家族至少一次小分辨率训练→保存→恢复→验证闭环，使用记录在配置中的显存选项。H800 8 GPU、batch 4、4K 目标单独验证峰值显存和多步训练；无该硬件时记录未验证，不能承诺不 OOM。
7. **收敛检查**：扫描删除目录的 imports/命令引用；新包可安装发现，示例 CLI/help 可用；`pid/` diff 应为空，历史实验文件不变；代码中无个人机器路径，README 与实际默认配置一致。
8. **定时执行与发布**：复用登记时的有界衔接测试，覆盖漏交接和重复启动；从 durable 实际记录确认子任务使用指定 zzz 模型。报告测试结果与最终远端 commit，必需测试未通过时不推送。

## Limitations

原始 CLEAR 发布代码面向 FLUX1、原始 VDN 面向视频模型；三种图像模型的 feature taps、训练输入和可训练角色属于本项目显式适配，验证数值与更新正确性不等于证明最终画质。全组合功能验收与长程收敛/质量评估分别报告。旧 checkpoint 和缓存继续留给历史版本使用，新框架不提供格式兼容层。
