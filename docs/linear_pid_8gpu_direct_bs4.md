# 两节点八卡十层直接初始化实验

2026-10-06 提交 job `116643`。使用 `students` account、`gpujl` partition，两节点各四张 GPU、每卡 batch 4、梯度累积 1，有效 batch 32。每节点一个 torchrun launcher，每 rank 四个数据 worker，计算线程 1。最长六天，提前 30 分钟保存退出。

独立输出根目录：`/home/ganchangyi/code/PiD-Linear/outputs/linear-pid-kda10-direct/2node-8gpu-bs4`。实际训练目录为其下的 `kda_0-1-2-4-5-6-8-9-10-12`，Slurm stdout 为根目录中的 `slurm-<jobid>.out`。node01 使用另一个输出根目录，当前训练不受影响。

配置：`pid/_src/configs/linear_pid/kda10_direct_8gpu_bs4.json`。首次加载完整原始 PiD v1.5 权重，随机初始化全部十层 KDA，optimizer、scheduler、训练计数从零开始，不读取四层或 node01 学生 checkpoint。对应布局为 `K K K F K K K F K K K F K F`，两层 PiT 保留。已验证首次启动计划为 `initialize_original`，`resume=none`、`init_from` 和 `resume_from` 均为空。之后重复提交同一配置只恢复这个独立目录的完整 checkpoint。

保持 FM-only、无教师网络、无 EMA、无模型 CPU 卸载；Gemma 从共享磁盘缓存读取，FLUX VAE 冻结。KDA LR `1e-4`，主干 LR `1e-5`，warmup 500 步。保留 `FLA_DISABLE_TENSOR_CACHE=1`，不设置同步 CUDA 调试。SwanLab 仍为 offline 模式，云端上传需要另行同步。

最近 node01 500 步的平均更新耗时为 17.106 秒/步。`28800 / 17.106 ≈ 1684`，选取整齐的 1700 步，预计常规间隔约 8.08 小时。八卡与三卡的每卡 batch 相同，单步工作量近似，不能按卡数比直接缩短单步时间；双节点通信和共享磁盘开销可能改变实际间隔，八卡实际吞吐尚未测量。96 张完整画廊自身耗时额外计入两次验证的墙钟间隔。

常规保存与完整 2K 画廊验证每 1700 optimizer steps 同时触发，即 1700、3400、5100……。关闭快速画廊和墙钟触发：`save_seconds=0`、`quick_every=0`、`validation_seconds=0`。训练代码已支持零值禁用墙钟 checkpoint，不影响 node01 的 `save_seconds=600`。step 1 仍保存初始恢复点，主动退出和可捕获停止信号也会保存，不必等到 1700 步。保留最近三个完整 checkpoint 及 `KEEP` 里程碑。

提交命令（本次已执行，不要重复提交同一输出目录的并行训练）：

```bash
conda activate linear-pid
cd /home/ganchangyi/code/PiD-Linear
mkdir -p /home/ganchangyi/code/PiD-Linear/outputs/linear-pid-kda10-direct/2node-8gpu-bs4
sbatch --export=ALL,LINEAR_PID_STAGE_OUTPUT_ROOT=/home/ganchangyi/code/PiD-Linear/outputs/linear-pid-kda10-direct/2node-8gpu-bs4,REPO_ROOT=/home/ganchangyi/code/PiD-Linear,LINEAR_PID_ENV=linear-pid,CPU_THREADS=1 \
  scripts/train_linear_pid_kda10_direct_8gpu_bs4.slurm
```

查看排队状态与日志：

```bash
squeue -j 116643
tail -f /home/ganchangyi/code/PiD-Linear/outputs/linear-pid-kda10-direct/2node-8gpu-bs4/slurm-116643.out
```

Slurm 脚本语法检查、实际离线路径 preflight 和首次初始化计划检查通过；初始化转换、恢复、周期保存、自动续训相关 19 项测试通过。提交后首次查询为 `PENDING (Priority)`，尚未执行八卡训练。
