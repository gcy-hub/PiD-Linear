# 训练与工具入口

从仓库根目录运行。训练参数写入配置或命令行，不为卡数、batch、窗口大小或训练步数创建新脚本。命令行参数覆盖配置中的同名参数。

| 工作 | 入口 |
|---|---|
| FLUX Sliding + KDA，训练及周期验证 | `python -m flux_kda.run --config <JSON>` |
| PiD/PixelDiT 适配训练 | `bash scripts/train_adapt.sh --config <JSON>` |
| PiD/PixelDiT 周期恢复 | 同一入口加 `--mode repeat` |
| Slurm 适配训练 | `sbatch scripts/train.slurm --config <JSON>` |
| 原生 PiD DMD 训练 | `torchrun ... -m scripts.train --config <Python配置>` |
| 数据下载、缓存到训练的完整流程 | `python -m scripts.run_pid_qdm_4k_pipeline` |
| 固定空间分区的阶段恢复流程 | `python -m pid._src.vdn_pid.pipeline` |
| PixelDiT 合成数据与训练 | `bash scripts/train_pixeldit_synthetic_a1.sh start ...` |

`configs/training/` 中的四个 JSON 分别配置 PiD VDN、PiD QDM、PiD Sliding 和 PixelDiT QDM 的算法参数。它们是原生训练器的参数列表。阶段、步数、batch、验证间隔和路径可以直接追加：

```bash
GPU_IDS=0,1,2,3 bash scripts/train_adapt.sh \
  --config configs/training/pid-sliding.json --mode repeat \
  --stage b --batch-size 2 --grad-accum 4 \
  --sliding-kernel-size 7 --max-steps 10000 \
  --output-root ../experiment/pid-sliding
```

同时指定 batch 和累积次数时，入口自动计算全局 batch。使用 `--print-command` 可以只查看最终命令而不启动训练。默认使用当前 Python；可通过 `PYTHON` 选择解释器。

周期恢复模式用于单机；多机由 Slurm 启动训练模式。Slurm 的机器资源通过提交参数配置，使用提交环境里的 Python，不在脚本中激活固定 Conda 环境：

```bash
RUN_OUTPUT=../experiment/pid-qdm sbatch --nodes=2 --gpus-per-node=4 \
  scripts/train.slurm --config configs/training/pid-qdm.json \
  --stage all --batch-size 2 --grad-accum 4
```

`train_qdm_stages.sh` 保留逐阶段完成与验证检查，`repeat_pid_vdn.py` 保留周期恢复逻辑，合成数据流程保留生成和训练切换。它们调用共用训练入口。

`prepare_*`、`sample_*`、`benchmark_*` 和 `diagnose_*` 分别用于数据准备、采样、测速与诊断，不是训练配置变体。

## FLUX 目录

- `flux_kda/`：FLUX2/FLUX3 共用的 Sliding + 标准 FLA KDA 适配、CLEAR 训练、验证与配置；KDA 按层轮换四个扫描方向。
- `flux2/`：FLUX2 加载、数据及原有 QDM 实现，KDA 入口复用其中的基础工具。
- `flux3/`：FLUX3 加载、打包数据及原有 QDM 实现，包含两代 FLUX 都会使用的 DiffSynth 运行时。

默认实验输出为仓库旁的 `../experiment/`，仓库内不再保留 `outputs` 链接。显式传入的输出路径优先。
