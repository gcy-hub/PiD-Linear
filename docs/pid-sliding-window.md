# PiD Sliding Window + QDM

本分支将14层MMDiT中的固定8×8 Query Window + Halo2，改为逐token居中的二维Sliding Window。默认11×11（radius=5、stride=1），也可设置13×13。QDM四象限求解、DWConv、原始Q/K/V/O、图文RoPE、文本query的全局Attention及两层PiT沿用原实现。

图像query同时读取自己的视觉邻域与文本K/V，使用一个联合Softmax。图像边缘严格截断，不循环连接、不平移RoPE坐标。NATTEN负责高效的内部查询；其默认边缘邻域与严格截断不同，因此边缘查询用分块SDPA校正。不会对整张图像展开全部K/V邻域，也不创建完整N×N mask。原有padding行为保留：传入Attention mask时按该mask处理；不额外改变原PiD默认的Full/Text Attention行为。

## 环境

继续使用`linear-pid`，不升级PyTorch。当前验证的版本是PyTorch2.10.0+cu128、NATTEN0.21.6+torch2100cu128、NVIDIA A40。

```bash
conda activate linear-pid
python -m pip install --no-deps 'natten==0.21.6+torch2100cu128' -f https://whl.natten.org
```

其他PyTorch/CUDA版本应选择对应wheel：[安装说明](https://natten.org/install/)。算子使用NATTEN的额外文本上下文接口：[算子说明](https://natten.org/operations/)。没有可用CUDA算子时会明确报错；不会静默换成另一种邻域算法。`--sliding-backend reference`仅供小规模数值核对。

## A1训练

设置资源路径，复用全量有效数据索引、图片和Gemma缓存。验证assets目录应包含下面列出的10个4K条件，以及与训练索引对应的`assets.json`；可通过符号链接复用已有资产，无需复制图片或重新编码。

```bash
conda activate linear-pid
cd "$PID_SLIDING_REPO"
export PID_WEIGHTS_ROOT="$MODEL_ROOT"
export PID_INDEX_ROOT="$DATASET_ROOT/linear_pid_index"
export PID_TEXT_CACHE_ROOT="$DATASET_ROOT/linear_pid_text_cache"
export PID_GALLERY_ROOT="$VALIDATION_ASSETS"

GPU_IDS=1,2,3 CPU_THREADS=1 \
bash scripts/train_adapt.sh --config configs/training/pid-sliding.json --mode repeat \
  --output-root ../experiment/pid-qdm-sliding11-4k
```

默认仅运行A1，从原始PiD v1.5 FLUX未蒸馏权重初始化新增QDM，不继承B阶段权重。训练使用现有官方4K多长宽比分桶（native 3840档位族，可包含4096长边），保持原缩放、裁剪、动态shift和条件构造。

| 项目 | 设置 |
|---|---|
| 替换层 | 14层MMDiT，Sliding11×11 + 原QDM |
| 并行 | DDP、CP=1 |
| 每卡batch / 累积 | 1 / 1 |
| 有效batch | GPU数；1、2、3三张卡时为3 |
| 训练步数 | 500 |
| 保存 | 每25步恢复断点；验证节点与最终断点保留 |
| 验证 | 250、500步，各10张4K学生结果 |
| 验证案例 | generated 010、029、003、009、019、022、000；real 011、028、000 |
| 采样 | 25步、CFG5、shift6，固定条件与seed |
| 目标 | 沿用A1逐层Attention相对MSE，主干继续传递教师输出 |
| 可训练参数 | 新增QDM参数；原始主干与Local Gate冻结 |
| 监控 | SwanLab离线 |

脚本每轮约20分钟在安全边界保存并自动续训，控制器也更新进程。验证逐张保存，中断后跳过已完成图像。500步验证全部完成后停止，不自动进入A2/B。手动重复命令也会从同一输出目录恢复。

```bash
# 状态与日志
cat ../experiment/pid-qdm-sliding11-4k/repeat/state.json
tail -f ../experiment/pid-qdm-sliding11-4k/repeat/latest-training.log

# 保存并停止，禁止自动重启
python scripts/repeat_pid_vdn.py --output-root ../experiment/pid-qdm-sliding11-4k --stop
```

## 等现有验证结束再切换

先将以上完整训练命令写入一个shell文件，然后启动切换控制器。它只在旧训练指定验证的所有图片和验证后断点完整保存后，才请求旧训练安全退出，再运行新命令。切换控制器每20分钟更新进程。它不会停止其他GPU任务、提交Slurm或从旧B权重初始化A1。

```bash
conda activate linear-pid
python scripts/start_pid_sliding_after_validation.py \
  --source-root "$OLD_RUN" --source-stage b --step 250 \
  --target-root "$SLIDING_RUN" --command-file "$SLIDING_LAUNCH_SCRIPT"

python scripts/start_pid_sliding_after_validation.py \
  --target-root "$SLIDING_RUN" --status

# 取消尚未发生的切换；不停止已经启动的训练
python scripts/start_pid_sliding_after_validation.py \
  --target-root "$SLIDING_RUN" --stop
```

切换后的训练日志在新输出目录的`repeat/latest-training.log`，切换记录在`handoff/state.json`。旧训练提前失败/手动停止，或无法确认干净退出时，控制器会记录原因并停止，不强行占用GPU。

## 检查与独立推理

局部数值检查覆盖联合归一化、文本padding、非方形/小网格、居中截断边缘、输出及梯度、原始Full路径一致性。实际4K A1的启动与断点恢复已单独短测；测试断点不作为正式初始化。

```bash
conda activate linear-pid
GPU_IDS=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
python -m pytest tests/vdn/test_sliding.py tests/vdn/test_sliding_handoff.py -q

# 沿用导出配置中的Sliding结构，独立加载学生进行固定画廊推理
GPU_IDS=0 bash scripts/sample_pid_vdn.sh \
  --checkpoint "$SLIDING_CHECKPOINT" --gallery-root "$VALIDATION_ASSETS" \
  --weights-root "$MODEL_ROOT" --index-root "$DATASET_ROOT/linear_pid_index" \
  --output-dir ../experiment/pid-qdm-sliding11-evaluation --suite qualitative
```

本地测试目录遵守仓库原有忽略规则，不包含于公开代码提交；当前环境留有这些测试。恢复时会核对Sliding kernel、stride及边缘语义，拒绝混用Halo或其他kernel的断点。
