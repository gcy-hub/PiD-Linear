# 十层 KDA 扩展阶段

本页记录四层到十层的渐进扩层对照。该训练现已停止；当前主实验从原始 PiD 直接随机替换全部十层 KDA，配置和运行命令见 [十层直接初始化](linear_pid_kda10_direct.md)。

配置：[kda10.json](../pid/_src/configs/linear_pid/kda10.json)。14 个 MMDiT 按零起始索引排列：

```text
索引： 0 1 2 3 4 5 6 7 8 9 10 11 12 13
结构： K K K F K K K F K K  K  F  K  F
```

KDA 为 `[0,1,2,4,5,6,8,9,10,12]`，Full Attention 为 `[3,7,11,13]`，两层 PiT 保留 Full Attention。

默认父模型来自 node01 四层训练 `/home/ganchangyi/code/PiD-Linear/outputs/linear-pid/kda_1-4-8-12` 的最新完整 checkpoint。第一次运行只读原训练目录，将完整父断点复制到新输出根目录的 `initial_state/checkpoints`。已有 KDA `[1,4,8,12]`、其他继承权重保持；只随机初始化新增 KDA `[0,2,5,6,9,10]`。新阶段重新建立 optimizer，warmup 500 次更新；累计步数、样本数和采样 cursor 继承父阶段，stage step 从 0 开始。有效 batch 改变后后续采样分组也会改变。

启动器读取 JSON 并转换成现有训练 CLI，首次使用 `--init-from <固定父断点> --init-weights raw --resume none`；有自己的完整 checkpoint 后自动使用 `--resume auto`，恢复十层模型、optimizer、scheduler、随机状态及数据进度。同一条命令可以重复运行。已有父副本固定下来后，不会随原四层训练的新 checkpoint 改变。

| 参数 | node01 四卡 | 双节点八卡 |
|---|---|---|
| 单卡 batch / 梯度累积 | 4 / 1 | 4 / 1 |
| 有效 batch | 16 | 32 |
| 全部 KDA 参数 LR / 继承主干 LR | `1e-4` / `1e-5` | 相同 |
| AdamW / weight decay / clip | AdamW / `1e-3` / `1.0` | 相同 |
| warmup | 500 更新，之后恒定 LR | 相同 |
| 训练目标 | 纯 FM，`lambda_out=0` | 相同 |
| 精度 | 学生及 optimizer FP32，计算 BF16 | 相同 |
| 显存策略 | MMDiT / PiT checkpointing，PiT chunk 2048 | 相同 |
| 条件 | 冻结 FLUX VAE，离线 Gemma 缓存 | 相同 |
| EMA / 模型 CPU 卸载 | 均关闭 | 相同 |
| 数据 worker / 计算线程 | 每 rank 4 / 1 | 相同 |
| 保存 | 首步、每 500 更新或 600 秒、退出时 | 首步、每 18,000 秒、退出时 |
| 验证 | 快速：0、100、500、每 1,000；完整每 5,000 stage steps | 每实际 18,000 秒完整 96 张 2K |
| 单次运行上限 | 28 分钟 | 六天减 30 分钟 |
| SwanLab | 离线、独立运行 ID | 相同 |

`effective_batch=0` 表示按实际 GPU 总数 × 单卡 batch × 累积次数计算，避免固定有效 batch 把单卡实际样本数限制成 1。十层完整 GPU 显存、吞吐及生成质量以实际运行日志为准。

2026-10-05 渐进扩层实验曾使用物理 GPU `1,2,3`，每卡 batch 4、累积 1，有效 batch **12**，现已停止并停用自动续跑。原四层训练已在 step 2833 保存退出，旧自动续跑和八卡 job `116623` 已取消。输出与固定验证资产已迁入 `/home/ganchangyi/code/PiD-Linear/outputs/`，旧路径只保留历史引用兼容链接。十层父副本为 step 2833；该扩层阶段累计 total step 从 2833 继续。下面的命令供恢复渐进扩层对照时使用，当前主实验请使用上方链接的直接初始化配置。

当前三卡完整自动续跑安装／恢复命令：

```bash
conda activate linear-pid
cd /home/ganchangyi/code/PiD-Linear
python scripts/watch_linear_pid_training.py --install \
  --layers 10 --stage-config pid/_src/configs/linear_pid/kda10.json \
  --output-root /home/ganchangyi/code/PiD-Linear/outputs/linear-pid-kda10/node01 \
  --gpu-ids 1,2,3 --batch-size 4 --grad-accum 1 \
  --workers 4 --threads 1 --max-seconds 1680 --min-free-mib 44000
```

cron 每分钟检查。每轮训练最多 1,680 秒（28 分钟），随后在 optimizer 更新边界保存退出；进程结束后下一次 cron 检查会恢复自己的完整断点。若首次运行还没有自己的 checkpoint，仍通过阶段启动器从固定四层父副本初始化，不会退回全十层随机初始化。已有 launcher／训练锁时不重复启动；启动前所选卡需分别至少空闲 44,000 MiB。进程清理后恢复最近完整断点，普通代码异常停止自动重试，修复后重新安装继续。模型载入和保存需要额外时间，28 分钟是训练进程的时间上限，不是精确到秒的重启周期。

状态和日志位于 `.../node01/kda_0-1-2-4-5-6-8-9-10-12/training_watch.status.json`、`watch_logs/`；暂停时在该运行目录创建 `STOP`，训练保存退出，cron 不再续跑。

node01 完整四卡运行命令，GPU IDs 可改为实际选定的卡：

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
GPU_IDS=0,1,2,3 CPU_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
bash scripts/train_linear_pid_kda10.sh \
  --preset 4gpu --batch-size 4 --grad-accum 1 --workers 4 --threads 1
```

输出为 `/home/ganchangyi/code/PiD-Linear/outputs/linear-pid-kda10/node01/kda_0-1-2-4-5-6-8-9-10-12`。时间到或进程清理后，重复完整命令恢复。直接启动命令不安装 cron，当前自动续跑使用上述 watcher；需要手动运行时先暂停对应 watcher，避免与它竞争。需要另一个独立实验时通过 `--output-root /absolute/new/root` 覆盖。

手动提交双节点八卡、最长六天：

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
mkdir -p /home/ganchangyi/code/PiD-Linear/outputs/linear-pid-kda10/slurm
CPU_THREADS=1 sbatch scripts/train_linear_pid_kda10_8gpu.slurm \
  --batch-size 4 --grad-accum 1 --workers 4 --threads 1
```

默认 account `students`、partition `gpujl`，每节点 4 GPU / 24 CPU，一个 torchrun 启动进程；新 job 在分配资源后固定最新完整父断点。输出为 `/home/ganchangyi/code/PiD-Linear/outputs/linear-pid-kda10/slurm/job-<新jobid>/kda_0-1-2-4-5-6-8-9-10-12`，Slurm 日志在其父级 `slurm/slurm-<新jobid>.out`。每个新 job 使用独立目录，不修改或覆盖已提交四层 job 的输出。Slurm 提前 30 分钟通知所有 rank 在更新边界保存退出。

配置预览（不复制父断点、不启动训练）：

```bash
conda activate linear-pid
cd /fs1/private/user/ganchangyi/code/PiD-Linear
bash scripts/train_linear_pid_kda10.sh --preset 4gpu --dryrun
bash scripts/train_linear_pid_kda10.sh --preset 8gpu --dryrun
```

如果要从你选中的另一份四层结果开始，先选择新的输出根目录，再传 `--source-run-dir /path/to/selected/run`。来源必须是包含 `checkpoints/step_*/complete.json` 的运行目录，启动器会选择其中最新完整断点。
