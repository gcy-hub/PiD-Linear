# PiD-QDM Halo2：全量数据、4K 微调

模型保持14层 Hybrid Attention（8×8 Local Softmax、Halo2、四象限记忆），PiT不变。
使用原始未蒸馏 PiD 权重重新初始化，A1 → A2 → B 依次传递新增参数。

三个阶段的 `--training-data full` 均直接读取全量索引，不使用20K/50K子集。
保留原有训练/验证划分、文本缓存、分桶采样和图像加载失败时的多卡协调跳过。
`--resolution 4096` 使用原生4K长宽比桶，正方形4096×4096，其余桶长边4096。
`--allow-upscale` 允许较小源图通过 Lanczos 缩放进入目标桶，避免将其全部跳过。

默认训练配置（四卡）：

| 阶段 | 更新步数 | 每卡 batch | 梯度累积 | 有效 batch |
|---|---:|---:|---:|---:|
| A1 | 500 | 2 | 4 | 32 |
| A2 | 1,000 | 2 | 4 | 32 |
| B | 10,000 | 2 | 4 | 32 |

八卡Slurm任务使用相同单卡batch和累积设置，有效batch为64。
4K最大正方形桶的bs4整网反传会超过A40显存，默认采用经过测试的bs2。
`--attention-batch-chunk 1` 对学生Attention按样本分块重算，PiT训练期的局部计算也分块执行；保持batch、Attention公式和损失不变，不进行CPU offload。

这些是更新步数配置。采样池覆盖全量有效训练集；当前分桶采样器在不同更新间有放回抽样，不保证每阶段完整遍历一轮。

验证每500阶段步及阶段结束时运行，仅生成六个 generated 条件（010、029、003、009、019、022）和两个 real 条件（011、028）的4K图像。
采样为25步、CFG5、shift6。生成条件使用原始固定 caption、seed 和文本编码；generated 条件重新用 Z-Image-Turbo 生成1024像素条件对应的 latent，real 条件来自原有留出图像。

```bash
conda activate linear-pid
export PID_WEIGHTS_ROOT="$HOME/huggingface_ckpts"
export PID_INDEX_ROOT="$HOME/dataset/MultiAspect-4K-1M/linear_pid_index"
export PID_TEXT_CACHE_ROOT="$HOME/dataset/MultiAspect-4K-1M/linear_pid_text_cache"
export PID_QDM_RUN="outputs/pid-qdm-halo2-4k-full"

# 仅需准备一次；中断后重复执行会保留已完成条件。
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
python -m scripts.prepare_pid_qdm_selected_assets \
  --source-assets outputs/linear-pid/assets \
  --output-root "$PID_QDM_RUN/assets" \
  --weights-root "$PID_WEIGHTS_ROOT" --index-root "$PID_INDEX_ROOT" \
  --text-cache-root "$PID_TEXT_CACHE_ROOT" \
  --cases generated_010,generated_029,generated_003,generated_009,generated_019,generated_022,real_011,real_028

# 后台串行训练。定期保存退出，更新控制器和训练进程后续训。
GPU_IDS=0,1,2,3 CPU_THREADS=1 bash scripts/train_pid_qdm_4k_full.sh

# 状态和手动停止；停止后不会自动重启。
python -m scripts.repeat_pid_vdn --output-root "$PID_QDM_RUN" --status
python -m scripts.repeat_pid_vdn --output-root "$PID_QDM_RUN" --stop
```

同阶段恢复会检查数据来源、分辨率、缩放策略和 Attention 布局，不能将原2K断点当作本次4K断点直接恢复。
SwanLab默认离线，数据位于各阶段的 `swanlog`；训练日志位于 `repeat/latest-training.log`。

循环控制器在600秒时请求保存，最多等待1,100秒完成当前更新或图像，随后更新进程；遇到OOM、数值错误或连续无进展会停止。

两节点任务使用 `scripts/train_pid_qdm_4k_full_8gpu.slurm`，需设置上述模型、索引和文本缓存环境变量，并将 `PID_GALLERY_ROOT` 指向已准备的8张4K条件。
将 `PID_QDM_RUN` 设置为独立于node01的目录，然后执行：

```bash
conda activate linear-pid
mkdir -p outputs/slurm
sbatch scripts/train_pid_qdm_4k_full_8gpu.slurm
```
