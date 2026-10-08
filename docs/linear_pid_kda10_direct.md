# 十层 KDA 直接初始化微调

配置为 `pid/_src/configs/linear_pid/kda10_direct.json`，`initialization=original_pid`。首次完整加载原始未蒸馏 PiD v1.5 FLUX，再随机替换全部十层 Attention：

```text
索引： 0 1 2 3 4 5 6 7 8 9 10 11 12 13
结构： K K K F K K K F K K  K  F  K  F
```

KDA `[0,1,2,4,5,6,8,9,10,12]` 的 Q/K/V/O 随机初始化，局部卷积零初始化。其他主干、Full Attention `[3,7,11,13]` 和两层 PiT 保留原始 PiD 权重。此实验最初从原模型直接转换十层，没有继承四层学生。之后所有续训恢复已有学生，不再次随机初始化。

## 同一实验使用固定目录

当前唯一输出根目录为 `/home/ganchangyi/code/PiD-Linear/outputs/linear-pid-kda10-direct/node01`，实际运行目录为其下的 `kda_0-1-2-4-5-6-8-9-10-12`。checkpoint、日志、SwanLab 离线数据和验证图均保存在该目录。换卡数或 batch 仍使用这个目录；只有用户要求独立实验或改变模型架构时才创建其他实验目录。

2026-10-06 原有 `node01-bs4-from1349` 和 `node01-3gpu-bs4-from1596` 已合并回这个目录，保留 checkpoint 文件原文、所有 SwanLab 分段、验证图和日志。当前续训保存到 step 1613 后完成合并，再从 1613 恢复。各分段原配置和日志副本在 `history/`，没有完整 checkpoint 的旧 `node01-bs4` 尝试在 `history/abandoned-bs4-retry/`。旧绝对路径与统一路径的映射记录在 `directory_migration.json`，历史 checkpoint 中的路径字段保持原文。源目录合并时没有复制大型 checkpoint。

有效 batch 的历史变化为 step 1349 时 12 → 16、step 1596 时 16 → 12。`--allow-batch-size-change` 现在允许在原目录续训并记录变化，保留学生、optimizer、scheduler、累计步数、样本数和已有 SwanLab ID；不需要 `--resume-from`，不重新 warmup。更换卡数会重新设定各 rank RNG，不保证逐位一致。

当前 GPU `1,2,3`，每卡 batch 4、梯度累积 1、有效 batch 12，每 rank 4 workers、线程 1。当前保持纯 FM loss、`lambda_out=0`，不创建教师网络，不启用 EMA 或模型 CPU 卸载。主干 LR `1e-5`，KDA LR `1e-4`，AdamW weight decay `1e-3`，global norm clip `1.0`。学生参数和 optimizer 为 FP32，计算 BF16；MMDiT/PiT activation checkpointing、PiT local chunk 2048。冻结 FLUX VAE，Gemma 使用磁盘文本缓存。

## 自动续训

当前已经安装 cron，每分钟检查一次。单轮上限 1680 秒（28 分钟），到时在 optimizer step 边界保存退出，下一次检查自动恢复。载入、验证和保存也需要时间，实际间隔不精确等于 28 分钟。普通代码异常暂停重试；所选卡空闲显存不足 45300 MiB 时等待。不会自动降低 batch。

重新安装或改变 GPU 配置时，仍使用同一输出根目录：

```bash
conda activate linear-pid
cd /home/ganchangyi/code/PiD-Linear
python scripts/watch_linear_pid_training.py --install \
  --layers 10 \
  --output-root /home/ganchangyi/code/PiD-Linear/outputs/linear-pid-kda10-direct/node01 \
  --gpu-ids 1,2,3 --batch-size 4 --grad-accum 1 --effective-batch 12 \
  --workers 4 --threads 1 --max-seconds 1680 --min-free-mib 45300 \
  --disable-fla-tensor-cache --allow-batch-size-change
```

`--install` 不会停掉已经运行的进程，新设置在下一轮启动时生效。不要让手动命令与自动续训竞争。停用自动续训后创建运行目录下的 `STOP`，当前训练会在更新边界保存退出。

完整训练的手动启动命令如下；仅在已经停用自动续训、没有同目录训练进程时运行。手动命令退出后不会自行再次启动：

```bash
conda activate linear-pid
cd /home/ganchangyi/code/PiD-Linear
GPU_IDS=1,2,3 CPU_THREADS=1 FLA_DISABLE_TENSOR_CACHE=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
bash scripts/train_linear_pid.sh \
  --preset 4gpu --layers 10 --lambda-out 0 \
  --batch-size 4 --grad-accum 1 --effective-batch 12 --workers 4 --threads 1 \
  --text-cache-root /home/ganchangyi/dataset/MultiAspect-4K-1M/linear_pid_text_cache \
  --pit-chunk-size 2048 --max-seconds 1680 \
  --output-root /home/ganchangyi/code/PiD-Linear/outputs/linear-pid-kda10-direct/node01 \
  --resume auto --allow-batch-size-change
```

`4gpu` 是配置预设名，实际 GPU 数由 `GPU_IDS` 决定。换卡数后相应调整有效 batch；每卡 batch 4、累积 1 时有效 batch 为卡数 × 4。

快速验证使用 16 张固定样本，step 0、100、500 及之后每 1000 步生成；完整 2K 验证为 96 张样本，每 5000 步生成。输出为 `galleries/step_<total_step>_raw/`。每 500 步或 600 秒保存，退出时也保存；保留最近三个完整 checkpoint 和带 `KEEP` 标记的历史里程碑。

## 故障记录与验证

此前 bs3 在 step 1374 后出现 CUDA 非法内存访问，最近完整断点为 1349。同步诊断定位到 FLA 的 `chunk_kda_bwd_kernel_intra`，关闭索引张量缓存 `FLA_DISABLE_TENSOR_CACHE=1` 后越过此前报错批次。独立算子的 memcheck 未检出越界，完整训练的 memcheck 因额外开销 OOM。此设置是已验证的规避措施，底层根因尚未确认。正常续训没有设置 `CUDA_LAUNCH_BLOCKING=1`。

2026-10-06 四卡续训在 step 1596 恢复后的第一次前向中，GPU 0 的其他进程占用约 16.87 GiB，导致 OOM，随后切换到三卡。三卡 bs4 完成 step 1600，耗时 17.89 s/step；合并前在更新边界完整保存 step 1613。

同目录 batch 变更、恢复身份和自动续训的相关 12 项测试通过。更换 GPU 或有效 batch 不再强制分出新的输出目录或 SwanLab ID。历史独立分段的离线 ID 保留，后续继续使用当前 ID `1746e8272c594e43be76f94f5f089106`；已有云端实验如需汇总可逐段 `swanlab sync <run-dir> --id <云端实验ID>`。

旧渐进十层实验仍在 `outputs/linear-pid-kda10/node01/`，它继承四层训练历史，是独立实验，未与本实验合并。
