# FLUX.3：四组双卡独立实验，再合并28层联合训练

该流程在同一台服务器使用8张GPU。先并行运行四个双卡队列，完成14组独立的双层替换实验；全部训练和验证完成后，按层合并最终B权重，再使用全部8卡联合训练28层。

## 任务分配和训练配置

以下以GPU `0,1,2,3,4,5,6,7`为例。GPU编号可通过`GPU_IDS`修改，必须提供8个不同编号。

| 队列 | GPU | 依次替换的层 |
|---|---|---|
| group_0 | 0、1 | [0,1] → [8,9] → [16,17] → [24,25] |
| group_1 | 2、3 | [2,3] → [10,11] → [18,19] → [26,27] |
| group_2 | 4、5 | [4,5] → [12,13] → [20,21] |
| group_3 | 6、7 | [6,7] → [14,15] → [22,23] |

每组实验独立从原始FLUX.3 Klein Base初始化，只替换指定两层为Local Softmax（8×8、Halo2）+图像写入QDM。其余26层保持原始Full Attention。QDM不使用RoPE，局部分支保留RoPE。

| 阶段 | 更新次数 | 每卡batch | 累积 | 有效batch | 数据池 | 保存与验证 |
|---|---:|---:|---:|---:|---|---|
| 独立实验A1 | 50 | 2 | 1 | 4 | 20K | 结束保存 |
| 独立实验A2 | 50 | 2 | 1 | 4 | 20K | 结束保存 |
| 独立实验B | 50 | 2 | 1 | 4 | 20K | 结束保存并生成10张学生2K图 |
| 合并后A1 | 500 | 2 | 1 | 16 | 20K | 每500步保存并验证 |
| 合并后A2 | 1,000 | 2 | 1 | 16 | 20K | 每500步保存并验证 |
| 合并后B | 10,000 | 2 | 1 | 16 | 50K | 每500步保存并验证 |

20K是现有50K的固定子集，不下载完整1M。独立实验warmup为5步；合并后A1/A2/B warmup分别为50/50/200步。

## 环境、模型和数据

先按[FLUX.3主说明](../README.md#environment)安装`flux3-qdm`环境，并按[权重说明](../README.md#weights)下载未蒸馏Base、配套Qwen3-VL文本编码器和VAE。权重下载支持恢复，配套资源使用仓库下载工具生成并核验清单。下载进程的代理设置只作用于下载进程。

在仓库根目录设置实际路径：

```bash
conda activate flux3-qdm
export FLUX3_WEIGHTS_ROOT="$HOME/huggingface_ckpts"
export FLUX3_SUBSET_ROOT="$HOME/dataset/MultiAspect-4K-1M/pid_vdn_subsets"
export FLUX3_CACHE_ROOT="$HOME/dataset/MultiAspect-4K-1M/flux3_qdm_cache"
export GPU_IDS=0,1,2,3,4,5,6,7
```

子集根目录需要包含`20000/`和`50000/`两个索引。使用已有50K导出数据中的索引；若只有50K索引，先在其样本内构建固定20K子集，保留样本ID、caption和分桶。核对图片路径与JSONL偏移，无需补全1M数据。

准备两份索引所需的FLUX.3原生缓存，不能复用Gemma特征。两个子集共享已有图片latent和相同caption的文本缓存，不重复复制图片：

```bash
conda activate flux3-qdm
for size in 20000 50000; do
  bash flux3/scripts/prepare_flux3_qdm.sh \
    --weights-root "$FLUX3_WEIGHTS_ROOT" \
    --subset-root "$FLUX3_SUBSET_ROOT" --cache-root "$FLUX3_CACHE_ROOT" \
    --subset-size "$size" --phase all --gpu-ids "$GPU_IDS" \
    --workers 4 --threads 1 --resume --max-seconds 0
  bash flux3/scripts/prepare_flux3_qdm.sh \
    --subset-root "$FLUX3_SUBSET_ROOT" --cache-root "$FLUX3_CACHE_ROOT" \
    --subset-size "$size" --check-only
done
```

迁移服务器后缓存会校验源图片路径、时间戳、模型和预处理指纹。检查失败时重新运行准备命令，不绕过校验。上述准备命令中断后可重复执行。

## 启动完整流程

若正在运行旧的14组串行入口，先用旧入口的`--stop`正常停止并等待退出，避免占用同一批GPU。此命令只启动新流程，不操作其他进程、不提交Slurm。

```bash
conda activate flux3-qdm
GPU_IDS="$GPU_IDS" CPU_THREADS=1 \
bash flux3/scripts/train_flux3_pairs_parallel_then_joint.sh \
  --weights-root "$FLUX3_WEIGHTS_ROOT" \
  --subset-root "$FLUX3_SUBSET_ROOT" --cache-root "$FLUX3_CACHE_ROOT" \
  --batch-size 2 --grad-accum 1 --workers 4 --threads 1 \
  --output-root outputs/flux3-pairs-parallel-then-joint
```

入口检查20K和50K缓存后返回，后台自动执行全部流程。四个队列并行，队列内部串行；只有14组B验证全部完成且GPU进程退出后，才会合并并开始8卡联合训练。

合并按层复制各组的QDM、Softmax Gate及Q/K/V/O LoRA，不平均权重，也不改动原始Base。检查基础权重、训练策略、数据指纹、层覆盖、键、形状和数值，记录14个来源及SHA256。合并文件是初始化权重，不是某一组优化器的恢复状态。

合并后A1/A2保留已训练LoRA和Softmax Gate并将它们冻结，只继续训练QDM；B再解冻LoRA和Gate。教师始终使用原始Full Attention和原始投影，不启用LoRA。联合阶段从新的优化器、scheduler、数据位置和阶段计数开始。

后台每约20分钟在更新或图片边界保存退出并续跑，控制器也更新进程；500步的正式保存、验证节点单独保留。手动停止后不会自行重启。恢复时重复同一命令，保持输出路径及配置一致。

## 状态、停止和输出

```bash
conda activate flux3-qdm
bash flux3/scripts/train_flux3_pairs_parallel_then_joint.sh \
  --output-root outputs/flux3-pairs-parallel-then-joint --status

# 一个队列的当前训练或验证日志。
tail -f outputs/flux3-pairs-parallel-then-joint/pairs/group_0/repeat/latest-training.log

# 合并后的联合训练日志（进入联合阶段后出现）。
tail -f outputs/flux3-pairs-parallel-then-joint/joint/repeat/latest-training.log

# 停止全部队列及联合训练，在安全边界保存。
bash flux3/scripts/train_flux3_pairs_parallel_then_joint.sh \
  --output-root outputs/flux3-pairs-parallel-then-joint --stop
```

目录结构：

```text
outputs/flux3-pairs-parallel-then-joint/
  repeat/                       # 总控制器状态及日志
  pairs/group_0/pair_00_01/      # 每组实验内有a1、a2、b
  pairs/group_0/pair_08_09/
  pairs/group_1/...
  pairs/group_2/...
  pairs/group_3/...
  merged/                       # 28层合并初始化权重及来源清单
  joint/a1/                     # 联合A1
  joint/a2/                     # 联合A2
  joint/b/                      # 联合B
```

验证仍为50步、CFG3.5、alpha6.93，生成10张学生2K图片。图片在各阶段`galleries/step_<step>/student/2048/`中逐张保存、恢复，SwanLab默认离线记录。

## 开机后自动续训

启动过一次上述流程后，在同一台服务器安装用户cron监控：

```bash
conda activate flux3-qdm
bash flux3/scripts/watch_flux3_resume.sh --install \
  --output-root outputs/flux3-pairs-parallel-then-joint --min-free-mib 32768
```

监控在开机时及每分钟检查同一输出目录，资源就绪后恢复四个队列或联合训练，保留已有完整checkpoint和验证图片。不会重复启动已有进程，手动停止后不会拉起。状态、移除命令见[重启恢复说明](../README.md#服务器重启后自动恢复)。
