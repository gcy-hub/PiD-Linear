# FLUX.3 Sliding11 + QDM

从原始 FLUX.3 Klein Base 初始化，替换前 5 层图像独立 Attention 和全部 28 层联合 Attention，共 33 层，均使用 Sliding11 + fixed4 QDM。文本独立处理层和参考图独立处理层保持原样。

图像 query 使用以自身为中心、边缘截断的 11×11 图像邻域，并与有效文本 K/V 进行一次联合 Softmax。文本 query 保留全部图文 Attention。局部路径继承原始 Q/K/V/O、QK Norm 和 RoPE；CUDA 使用 FlexAttention 前向和反向，不展开邻域 K/V。

前 5 层只有图像输入，因此局部分支仅关注滑动图像邻域，不添加文本输入；其 QDM 使用相同的四象限规则。后 28 层使用上述图文联合计算。两部分都纳入 A1 对齐；A2/B 的梯度和激活 checkpointing 覆盖前 5 层，B 也为它们的 Q/K/V/O 添加 LoRA。

并行的 fixed4 QDM 继承现有实现：原始未旋转 QKV、K/V 全图 5×5 DWConv、SiLU、Q/K L2 normalization，四象限各自 FP32 求解 `S=B(I+A)^-1`。图像 query 读取自身象限，经过 Norm、Gate 和独立投影后与局部分支相加。QDM 不使用 RoPE，不写入文本，不扫描、不跨 timestep 保留状态。

| 阶段 | 更新步数 | 训练参数 | 验证 |
|---|---:|---|---|
| A1 | 500 | 新增 QDM，原始投影冻结 | 每 500 步 |
| A2 | 1000 | 同 A1，整网预测蒸馏 | 每 500 步 |
| B | 10000 | QDM、局部门、Q/K/V/O LoRA | 每 500 步 |

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

入口后台串行 A1→A2→B，阶段切换加载本次上一阶段权重。同阶段恢复参数、优化器、scheduler、随机状态和采样进度。每 500 步保存并验证；轮换退出时额外保存恢复状态。每轮约 20 分钟在更新或图片边界保存退出，随后重新启动训练及控制器。系统整机重启后，重复启动命令即可恢复。连续 OOM、非有限数值或三轮无进展会停止，不改变 batch 或算法。

```bash
conda activate flux3-qdm
bash flux3/scripts/train_flux3_sliding_qdm.sh --status
tail -f "$FLUX3_QDM_RUN/repeat/latest-training.log"
bash flux3/scripts/train_flux3_sliding_qdm.sh --stop
```

验证位于 `a1|a2|b/galleries/step_XXXXXXXXX/student/2048/`。同一实验始终使用同一输出目录。checkpoint 保存局部几何及 kernel，恢复、阶段衔接和推理都拒绝不一致的窗口配置。

专用入口默认包含 `--replace-image-blocks`。底层通用入口仍默认只替换联合层，兼容旧实验；其 `--no-replace-image-blocks` 可显式恢复 28 层范围。33 层与 28 层 checkpoint 的范围标记不同，不能互相作为同阶段恢复或阶段衔接来源。新实验从原始 Base 开始 A1。

## 在另一台服务器训练

代码分支为 `FLUX3-sliding-QDM`。本地交付的 `FLUX3-sliding-QDM.bundle` 包含该分支及 Git 历史，不包含权重、数据、缓存和训练输出。复制到目标服务器后，可以在新目录克隆；已有仓库也可直接 fetch 该 bundle 并创建分支。

```bash
git clone -b FLUX3-sliding-QDM /path/to/FLUX3-sliding-QDM.bundle PiD-Linear-sliding
cd PiD-Linear-sliding
```

按 [环境及权重说明](../README.md) 安装 `flux3-qdm`。资源根目录需要 `flux-3-image-klein-base/model.safetensors`，以及 `flux-3-image-klein/` 下的原生 Qwen3-VL 编码器、processor、VAE 和校验 manifest。可复制这两个完整目录，或使用主说明中的下载、校验入口。

若服务器已有 `pid-vdn-50k` 图文导出包，无需完整 1M，也无需 PiD 的 latent 验证包。先将导出包导入本机索引；导入只检查原图并建立索引，不复制图片。三个阶段均选择其中相同的 20K。

```bash
conda activate flux3-qdm
export FLUX3_WEIGHTS_ROOT="$HOME/huggingface_ckpts"
export FLUX3_EXPORT_ROOT="/path/to/pid-vdn-50k"
python -m pid._src.vdn_pid.portable_data import \
  --package-root "$FLUX3_EXPORT_ROOT" \
  --output-root "$FLUX3_EXPORT_ROOT/training" --workers 4 --resume
export FLUX3_SUBSET_ROOT="$FLUX3_EXPORT_ROOT/training/subsets"
export FLUX3_CACHE_ROOT="$FLUX3_EXPORT_ROOT/flux3_qdm_cache"
```

随后准备 FLUX.3 原生缓存，不能使用 Gemma/PiD 缓存代替。缓存含模型、源图片路径和文件时间戳校验；从其他服务器复制来的缓存必须先核验，不通过时按本机路径重跑准备，禁止绕过指纹检查。

```bash
conda activate flux3-qdm
bash flux3/scripts/prepare_flux3_qdm.sh \
  --weights-root "$FLUX3_WEIGHTS_ROOT" \
  --subset-root "$FLUX3_SUBSET_ROOT" --cache-root "$FLUX3_CACHE_ROOT" \
  --subset-size 20000 --phase all --gpu-ids 0,1,2,3,4,5,6,7 \
  --workers 4 --threads 1 --resume --max-seconds 0
bash flux3/scripts/prepare_flux3_qdm.sh \
  --weights-root "$FLUX3_WEIGHTS_ROOT" \
  --subset-root "$FLUX3_SUBSET_ROOT" --cache-root "$FLUX3_CACHE_ROOT" \
  --subset-size 20000 --check-only
```

准备完成后再启动训练。示例为单机八卡、每卡 bs2、累积1，有效 batch16；更改卡数时有效 batch 随之改变。训练和验证步数沿用本页配置，正式训练从原始权重开始，无需本机 node01 的 checkpoint。

```bash
conda activate flux3-qdm
GPU_IDS=0,1,2,3,4,5,6,7 CPU_THREADS=1 \
bash flux3/scripts/train_flux3_sliding_qdm.sh \
  --batch-size 2 --grad-accum 1 \
  --output-root "$PWD/outputs/flux3-sliding11-qdm-2k-server"
```

双节点训练不能仅靠 GPU_IDS 指定远程卡；上面的命令是单机八卡入口。更改为双节点时需另行配置分布式启动和共享路径。
