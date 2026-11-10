# Attention 实验框架

PiD、FLUX2、FLUX3 共用算法、数据生命周期和训练入口。基础模型配置指定架构、资源和原生采样；替换模型由同一基础配置、attention 配置和层选择组成。原生 `pid/` 与外部历史实验文件保留，旧 checkpoint 不迁移。旧 FLUX 训练器、算法副本、缓存工具和参数脚本已由共享入口替代；读取旧实验请使用对应历史 Git revision。保留的原生入口见 [原生训练](docs/historical-training.md)。

| 目录 | 职责 |
|---|---|
| `models/` | 原生模型加载、稳定层名、QKV/O、teacher、采样 |
| `algorithms/` | 张量 attention 算子，不加载数据/模型或创建优化器 |
| `data/` | manifest、官方桶、原生编码、图像/文本独立缓存 |
| `training/` | 策略、精度、分布式、checkpoint 和验证进程边界 |
| `attention_framework/`、`configs/` | 直接接口、OmegaConf 配置与校验 |
| `scripts/`、`download/` | 薄入口、验收、显式下载 |
| `pid/` | 保留的原生实现 |

依赖方向：训练组合模型、算法和数据；模型调用算法；数据管理编码器；算法不依赖训练策略。唯一 DiffSynth 运行时由模型适配器管理。

| attention | PiD | FLUX2 | FLUX3 | 策略 |
|---|---|---|---|---|
| `cross_scan_kda` | 支持 | 支持 | 支持 | VDN / CLEAR |
| `sliding_qdm` | 支持 | 支持 | 支持 | VDN / CLEAR |
| `qdm_halo` | 支持 | 支持 | 支持 | VDN / CLEAR |
| `clear` | 支持 | 支持 | 支持 | CLEAR |

共 12 个模型/算法组合、21 个有效策略组合；CLEAR attention + VDN 明确拒绝。这是支持范围；实际验收状态见报告，不能从配置可解析推断 GPU 已通过。

从仓库根目录运行，Python 3.12 环境中安装 `pip install -e .`。训练所需 FLA、Prodigy 和模型编码依赖由所选实现要求；`scripts/setup_pid_env.sh` 保留现有 PiD 环境准备。训练不隐式下载。先确认 Hugging Face 访问权限，按 [下载说明](download/README.md) 准备本地权重及数据：

```bash
python download/download_models.py --models pid vae gemma --weights-root ./weights --workers 4
python -m attention_framework.config --print-config \
  --model-config configs/models/flux3.yaml \
  --algorithm-config configs/algorithms/cross_scan_kda.yaml \
  --training-config configs/training/clear.yaml \
  --data-config configs/data/default.yaml
```

同一基础配置复用到不同算法，只替换 `--algorithm-config`：

```bash
python -m training.run \
  --model-config configs/models/flux3.yaml \
  --algorithm-config configs/algorithms/sliding_qdm.yaml \
  --training-config configs/training/clear.yaml \
  --data-config configs/data/default.yaml \
  paths.weights=./weights paths.dataset=./dataset/pid-vdn-80k \
  data.manifest=./dataset/pid-vdn-80k/manifest.jsonl \
  paths.output=../experiment/flux3-sliding \
  'replacement.layers.early_image=[]' 'replacement.layers.joint=[0,4,8]' \
  algorithm.local.window_size=15 training.batch_size=1 training.steps=20 \
  runtime.gpus=4 training.accumulation=2 checkpoint.every=10 validation.every=10
```

权重/TE/tokenizer/VAE 资源在 `model.resources` 中设置，代表配置中的 null 必须填写为实际资源。换用 `configs/algorithms/cross_scan_kda.yaml` 即复用该基础模型进行 KDA 训练。PiD 层名为 `patch`，FLUX2 为 `double`/`single`，FLUX3 为 `early_image`/`joint`；组可选 `all` 或索引列表，未选层保留原生行为。KDA 扫描按原生全层序号轮换，部分替换不重新编号。

覆盖顺序为共享默认 → 模型 → 算法 → 策略 → 数据 → 命令行 dotlist。未知字段、非法窗口/层与策略组合在读取边界报错。相对运行路径按启动目录解析，manifest 内图像路径按 manifest 目录解析。默认输出 `../experiment/attention-framework`，所有资源/数据/输出路径可配。每卡 batch × world size × accumulation 得到全局 batch；上述例子为 8。

VDN 参考 [官方源码 e262cb5](https://github.com/OpenVDN/vdn-minimax-h3/tree/e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b)：A1 teacher detached 输入逐层反传，相对 MSE 比较图像 attention 的 post-O 输出；A2 端到端 teacher prediction MSE；B 加 gate 与 QKVO LoRA。A1/A2 训练 branch/必要 pooling；B 加 local_gate/lora。默认阶段步数 200/500/2000、warmup 50/50/200、LoRA rank=alpha=64 可覆盖。AdamW 角色分组与阶段切换属于训练策略。

CLEAR 参考 [官方源码 ddca131](https://github.com/Huage001/CLEAR/tree/ddca13160b7c2fe5d340c8a421db72b7bdaa6ba9)：圆形邻域、文本全局可见、可训练 K/V 池化；损失 FM + 0.5 prediction KD + 0.5 feature KD。训练 branch/local_gate/image_qkvo/pooling，冻结其余主干、TE/VAE，禁用 LoRA。默认 Prodigy、常数调度与 FP32 更新，避免 BF16 小更新舍入。官方 CLEAR 面向 FLUX1，VDN 面向视频；本项目显式适配图像模型和稳定层名。KDA/CLEAR feature KD 取 pre-O，QDM 取含 memory 的 post-O；teacher 取同位置并绕开 student 投影/LoRA。`../CLEAR` 是适配仓库，不是作者原版。

数据准备与缓存语义、恢复、显存选项、扩展接口见 [使用流程](docs/attention-framework.md)。四卡验收入口及每项证据要求见 [验收](docs/attention-framework-acceptance.md)。入口索引见 [scripts](scripts/README.md)。

## Limitations

prepare/cache/sample CLI 见使用流程，资源绑定生成命令见验收文档。真实资源及四卡数值验收尚待执行，支持矩阵不代表运行通过。H800 八卡 4K、每卡 batch 4 的峰值显存与多步训练单列未验证。功能验收不替代长程收敛和画质评估。
