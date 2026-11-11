# 执行入口

从仓库根目录运行。Python 逻辑按职责放在 `training/`、`data/`、`download/`；本目录保存 shell 与 Slurm 启动入口，参数原样传递。

| 用途 | 入口 |
|---|---|
| 共享 attention 训练 | `python -m training.run` |
| 共享缓存准备 / 采样 | `python -m data.prepare` / `python -m training.validate` |
| 原生 PiD / PixelDiT 训练与续训 | `bash scripts/train_adapt.sh --config configs/training/pid-qdm.json --mode train`（或 `repeat`） |
| 原生 DMD | `torchrun --standalone --nproc_per_node=4 -m training.train --config pid/_src/configs/pid_training/config.py` |
| 原生索引 / 子集准备 | `python -m data.prepare_pid_data --help` / `python -m data.prepare_pid_vdn_data --help` |
| 原生文本缓存 / 固定条件 | `bash scripts/prepare_pid_text_cache.sh` / `bash scripts/prepare_pid_assets.sh` |
| 原生独立采样 | `bash scripts/sample_pid_vdn.sh` |
| 原生完整数据准备与 A1→A2→B | `bash scripts/train_pid_qdm_4k_8gpu_pipeline.sh --help` |
| 原生阶段编排 / Slurm | `bash scripts/train_qdm_stages.sh` / `sbatch scripts/train.slurm` |
| PixelDiT 合成图像 A1 | `bash scripts/train_pixeldit_synthetic_a1.sh` |
| DCP 权重合并 | `python -m training.convert_distcp_to_pt --help` |
| 下载 / WebDataset 分片 | `download/` / `python -m data.sharding_wds --help` |

原生资源、阶段及恢复用法见 [原生训练](../docs/historical-training.md)。共享框架见 [生命周期](../docs/attention-framework.md)，FLUX3 八卡目标命令见 [训练说明](../docs/flux3-crosslayer-kda-80k.md)。
