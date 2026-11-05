# FLUX.3 Sliding11 + QDM

从原始 FLUX.3 Klein Base 初始化，替换全部 28 层联合 Attention；前面的独立图像、文本和参考图层保持原样。

图像 query 使用以自身为中心、边缘截断的 11×11 图像邻域，并与有效文本 K/V 进行一次联合 Softmax。文本 query 保留全部图文 Attention。局部路径继承原始 Q/K/V/O、QK Norm 和 RoPE；CUDA 使用 FlexAttention 前向和反向，不展开邻域 K/V。

并行的 fixed4 QDM 继承现有实现：原始未旋转 QKV、K/V 全图 5×5 DWConv、SiLU、Q/K L2 normalization，四象限各自 FP32 求解 `S=B(I+A)^-1`。图像 query 读取自身象限，经过 Norm、Gate 和独立投影后与局部分支相加。QDM 不使用 RoPE，不写入文本，不扫描、不跨 timestep 保留状态。

| 阶段 | 更新步数 | 训练参数 | 验证 |
|---|---:|---|---|
| A1 | 200 | 新增 QDM，原始投影冻结 | 每 200 步 |
| A2 | 400 | 同 A1，整网预测蒸馏 | 每 200 步 |
| B | 10000 | QDM、局部门、Q/K/V/O LoRA | 每 200 步 |

每次验证仅生成学生的 10 张固定 prompt、2048×2048 图像，50 步、CFG 3.5、alpha 6.93。训练使用现有原生 FLUX.3 文本/latent 缓存和 PiD 2K 多长宽比分桶；默认三个阶段均用已有完整缓存的 20K 子集。每卡 bs2、累积 1，有效 batch 为 GPU 数×2。SwanLab 默认离线。

资源准备沿用 [主说明](../README.md)。从仓库根目录运行：

```bash
conda activate flux3-qdm
export FLUX3_WEIGHTS_ROOT="$HOME/huggingface_ckpts"
export FLUX3_SUBSET_ROOT="$HOME/dataset/MultiAspect-4K-1M/pid_vdn_subsets"
export FLUX3_CACHE_ROOT="$HOME/dataset/MultiAspect-4K-1M/flux3_qdm_cache"
export FLUX3_QDM_RUN="$PWD/outputs/flux3-sliding11-qdm-2k-node01"
GPU_IDS=1,2,3 CPU_THREADS=1 bash flux3/scripts/train_flux3_sliding_qdm.sh
```

入口后台串行 A1→A2→B，阶段切换加载本次上一阶段权重。同阶段恢复参数、优化器、scheduler、随机状态和采样进度。每 25 步保留恢复状态，每 200 步保留验证节点。每轮约 20 分钟在更新或图片边界保存退出，随后重新启动训练及控制器。系统整机重启后，重复启动命令即可恢复。连续 OOM、非有限数值或三轮无进展会停止，不改变 batch 或算法。

```bash
conda activate flux3-qdm
bash flux3/scripts/train_flux3_sliding_qdm.sh --status
tail -f "$FLUX3_QDM_RUN/repeat/latest-training.log"
bash flux3/scripts/train_flux3_sliding_qdm.sh --stop
```

验证位于 `a1|a2|b/galleries/step_XXXXXXXXX/student/2048/`。同一实验始终使用同一输出目录。checkpoint 保存局部几何及 kernel，恢复、阶段衔接和推理都拒绝不一致的窗口配置。
