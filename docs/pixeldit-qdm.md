# PixelDiT-T2I + QDM Halo2

这个实验直接使用官方 1024px PixelDiT-T2I 权重，将全部 14 层 MMDiT Attention 替换为 QDM Halo2，两层 PiT 保持冻结。模型只接收 noisy pixels、时间步和 Gemma 文本，不使用 PiD 的低分辨率图像 latent 条件，不加载 FLUX VAE 或 Z-Image。默认训练和阶段验证使用原生 1K 多长宽比设置。

局部分支以 `8×8 token` 为 query 区域，halo=2 时读取周围 `12×12` 图像 K/V 和全部文本，保留原有 RoPE。远程分支使用四象限 QDM，默认开启 `--remote-rope`：图像 K/V 完成全网格 DWConv、SiLU，Q/K 完成 L2 normalization 后，使用原始主干的全图二维坐标旋转 Q/K，再分配到四个记忆。V 不旋转，象限内不重置坐标。远程分支仍不写入文本；文本继续通过局部分支交互，PiT 的 RoPE 保持原样。

A1/A2 冻结原始投影，仅训练新增记忆分支；B 继续训练记忆分支、Softmax Gate 和 rank=64、alpha=64 的 Q/K/V/O LoRA。FFN、AdaLN、PiT 和原始权重始终冻结，EMA 关闭，无 CPU offload。恢复远程 RoPE 是定位空间重复、错位问题的对照实验，是否改善质量需要观察验证结果。

## 1. 准备资源

安装仓库的 `linear-pid` 环境。下载完整 PixelDiT-T2I checkpoint，放入 `weights/PixelDiT-T2I/`；Gemma 放入 `weights/gemma-2-2b-it/`。加载器严格检查完整主干的键和形状；官方 checkpoint 中额外的 REPA projector 不参与推理和蒸馏。

可以直接复用本项目的图片索引、20K/50K 子集及 Gemma caption 缓存：两种模型的 Gemma、正提示词模板、300 个 token 与 2304 维输出相同。A1/A2 使用同一 20K，B 使用包含这 20K 的 50K。图像按 1K 多长宽比桶进行 Lanczos 缩放和中心裁剪，不预编码图像 latent。PixelDiT 原生的 CFG 空文本/负提示词不添加 CHI 模板，因此单独补充这两条小缓存，不重编码全部 caption。

在仓库根目录设置路径；也可以将这些参数直接传给训练脚本：

```bash
conda activate linear-pid
export PID_WEIGHTS_ROOT="$PWD/weights"
export PIXELDIT_BASE="$PID_WEIGHTS_ROOT/PixelDiT-T2I/pixeldit_t2i_v1.pth"
export PID_INDEX_ROOT="$PWD/dataset/MultiAspect-4K-1M/linear_pid_index"
export PID_VDN_SUBSET_ROOT="$PWD/dataset/MultiAspect-4K-1M/pid_vdn_subsets"
export PID_TEXT_CACHE_ROOT="$PWD/dataset/MultiAspect-4K-1M/linear_pid_text_cache"
export PID_GALLERY_ROOT="$PWD/runtime/pid-vdn-validation/gallery"
export PIXELDIT_RUN="$PWD/outputs/pixeldit-t2i-qdm-halo2-rope"

# 一次性补充两条原生 CFG 条件，结果写入现有文本缓存目录。
CUDA_VISIBLE_DEVICES=0 python -m pid._src.pixeldit_qdm.conditioning \
  --weights-root "$PID_WEIGHTS_ROOT" --text-cache-root "$PID_TEXT_CACHE_ROOT" --threads 1
```

已有 PiD 数据准备结果可以直接填入以上路径。未建立索引或子集时，先执行仓库原有的数据准备命令。没有文本缓存时跳过上述补充命令，设置 `PID_TEXT_CACHE_ROOT=""`，训练会在线加载冻结 Gemma。验证条件包只复用 caption、正文本条件、seed 和长宽比，替换为原生 CFG 条件并丢弃 latent；验证也从像素噪声开始，默认输出 1K。

## 2. 三阶段训练

默认采用以下预算，`--a1-steps`、`--a2-steps`、`--b-steps` 可分别覆盖。每次更新使用同一套原始权重，教师与学生顺序执行。

| 阶段 | 更新数 | 每卡 batch | 累积 | 4 卡有效 batch | 目标 | 保存 | 验证 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| A1 | 500 | 4 | 1 | 16 | 相同教师层输入上的 Attention 相对 MSE，逐层立即反传 | 50 steps | 阶段结束 |
| A2 | 1,000 | 8 | 1 | 32 | 最终像素 velocity 预测 MSE | 100 steps | 阶段结束 |
| B | 10,000 | 8 | 1 | 32 | 最终预测 MSE，加入零增量 LoRA | 2,000 steps | 每 2,000 steps 与阶段结束 |

8 卡时有效 batch 分别为 32、64、64。FM loss 仅记录。优化器、学习率、精度及梯度裁剪沿用 QDM 三阶段配置。训练动态 shift 以 1024 为参考、基础值 4；原始主干和局部 Softmax 使用官方 original RoPE。OOM 会报错退出，不自动减小 batch。

前台串行运行 A1→A2→B；阶段切换加载本实验上一阶段最近的完整 checkpoint，并重新创建优化器：

```bash
conda activate linear-pid
GPU_IDS=0,1,2,3 CPU_THREADS=1 bash scripts/train_pixeldit_qdm.sh \
  --stage all --resume auto --teacher-checkpoint "$PIXELDIT_BASE" \
  --output-root "$PIXELDIT_RUN" --workers 4 --threads 1
```

只运行某阶段时将 `--stage all` 改为 `--stage a1`、`--stage a2` 或 `--stage b`。同一输出目录内自动查找上一阶段；也可用 `--init-from` 指定父 checkpoint。`--a1-batch-size`、`--a2-batch-size`、`--b-batch-size` 分别设置 batch；`--grad-accum` 控制累积。`--batch-size` 会覆盖三个阶段的 batch。

同阶段重复原命令即可恢复参数、优化器、scheduler、随机状态、数据进度及 SwanLab run。保存周期按 optimizer step 计算；停止信号在更新边界触发保存，不完整 checkpoint 不会用于恢复。

远程 RoPE 设置写入配置和 checkpoint，恢复及阶段切换均检查一致性。旧 checkpoint 缺少此字段时按 NoPE 处理，独立推理自动遵循 checkpoint。重新训练默认从原始 PixelDiT 开始，使用上述独立输出目录；不会接续旧 NoPE 权重。复现旧 NoPE 实验时，训练命令加 `--no-remote-rope`，并明确指定旧实验目录。

## 3. 后台自动续训

以下命令启动后返回终端。控制器约每 20 分钟让训练保存退出，随后同时更新控制器和训练进程，继续同一输出目录；依次完成三个阶段及其验证。

```bash
conda activate linear-pid
GPU_IDS=0,1,2,3 CPU_THREADS=1 bash scripts/train_pixeldit_qdm_repeat.sh \
  --stage all --resume auto --teacher-checkpoint "$PIXELDIT_BASE" \
  --output-root "$PIXELDIT_RUN" --cycle-seconds 1200 \
  --workers 4 --threads 1 --swanlab-mode offline

bash scripts/train_pixeldit_qdm_repeat.sh --output-root "$PIXELDIT_RUN" --status
tail -f "$PIXELDIT_RUN/repeat/latest-training.log"
bash scripts/train_pixeldit_qdm_repeat.sh --output-root "$PIXELDIT_RUN" --stop
```

手动停止后不会自动拉起。重复启动命令可继续。OOM、非有限数值或连续三轮无进展会停止重试，原因保留在 `repeat/controller.log`。SwanLab 默认离线，各阶段日志位于 `a1/swanlog`、`a2/swanlog`、`b/swanlog`，训练指标同时写入各阶段 `metrics.jsonl`。

## 4. 独立推理

输入新 prompt，仅加载 Gemma 编码一次，然后释放编码器，再加载 PixelDiT；不会创建第二套教师或进行模型 CPU offload。默认使用原始 PixelDiT 的 50 步、CFG 2.75、shift 4、1024×1024。省略 `--checkpoint` 即使用原始模型；`--compare-original` 保存同 prompt、seed 的原始/学生图及左右拼图。

```bash
conda activate linear-pid
export PIXELDIT_CHECKPOINT="$PIXELDIT_RUN/a2/checkpoints/COMPLETE_CHECKPOINT_DIRECTORY"
CUDA_VISIBLE_DEVICES=0 python -m pid._src.pixeldit_qdm.inference \
  --checkpoint "$PIXELDIT_CHECKPOINT" --weights-root "$PID_WEIGHTS_ROOT" \
  --teacher-checkpoint "$PIXELDIT_BASE" \
  --prompt "A child playing with a dog in a sunlit garden, detailed natural photography" \
  --height 1024 --width 1024 --seed 42 --compare-original \
  --output-dir "$PIXELDIT_RUN/prompt-1k"
```

多 prompt 可用 `--prompts-file prompts.txt`，配合 `torchrun --standalone --nproc_per_node=4 -m pid._src.pixeldit_qdm.inference ...` 按样本分配给各 GPU。输出包含 caption、seed、采样参数、耗时和峰值显存；重复命令会跳过完整且设置一致的结果。2K/4K 推理可分别设置 `--height 2048 --width 2048` 或 `--height 4096 --width 4096`，属于分辨率外推，需要与相同设置的原始模型比较。

复现固定验证画廊：

```bash
conda activate linear-pid
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m pid._src.pixeldit_qdm.evaluation \
  --checkpoint "$PIXELDIT_CHECKPOINT" --weights-root "$PID_WEIGHTS_ROOT" \
  --teacher-checkpoint "$PIXELDIT_BASE" --index-root "$PID_INDEX_ROOT" \
  --text-cache-root "$PID_TEXT_CACHE_ROOT" \
  --gallery-root "$PID_GALLERY_ROOT" --suite qualitative --include-4k \
  --output-dir "$PIXELDIT_RUN/inference"
```

这是用于判断无图像 latent 条件时 QDM 能否恢复生成质量的实验；代码通过性和训练 loss 不能代替阶段验证图的质量判断。

当前代码的实际训练、推理及恢复检查见 [验证记录](pixeldit-qdm-verification.md)。

## 5. 用原始 PixelDiT 合成图像进行 A1 对照

此入口先用原始、未修改的 PixelDiT-T2I 生成正方形 1024×1024 图像，再从原始权重开始 QDM Halo2 的 A1。生成直接读取原有 Gemma caption 缓存和原生 CFG 负提示词缓存，采用 50 步、CFG 2.75、shift 4，不加载 Gemma、VAE 或已有学生权重。

默认 A1 为 500 步、4卡、每卡 bs4、累积1，需要 8,000 次样本曝光。以 seed=42 对现有 20K 子集执行原采样器，共抽到 6,602 个不同样本 ID，因此每条记录生成一张图即可。入口保存原500步的样本顺序及重复关系；训练使用相同 caption 顺序和已有文本缓存，不重新随机抽一套数据。源验证样本不参与生成。

在前面的资源变量设置完成后运行：

```bash
conda activate linear-pid
export PIXELDIT_SYNTH_DATA="$PWD/dataset/MultiAspect-4K-1M/pixeldit_t2i_synthetic_a1"
export PIXELDIT_SYNTH_RUN="$PWD/outputs/pixeldit-t2i-qdm-halo2-rope-synthetic-a1"

GPU_IDS=0,1,2,3 CPU_THREADS=1 bash scripts/train_pixeldit_synthetic_a1.sh start \
  --data-root "$PIXELDIT_SYNTH_DATA" --output-root "$PIXELDIT_SYNTH_RUN" \
  --teacher-checkpoint "$PIXELDIT_BASE" --generation-batch-size 4 \
  --cycle-seconds 1200 --workers 4 --threads 1 --swanlab-mode offline

bash scripts/train_pixeldit_synthetic_a1.sh status --output-root "$PIXELDIT_SYNTH_RUN"
tail -f "$PIXELDIT_SYNTH_RUN/repeat/latest-training.log"
bash scripts/train_pixeldit_synthetic_a1.sh stop --output-root "$PIXELDIT_SYNTH_RUN"
```

首次命令启动后台控制器后返回终端。四卡按样本批次并行生成，每张图有独立固定噪声 seed；图片和完成标记原子保存，重启跳过已完成结果。`--generation-batch-size` 控制生成 batch，与 A1 的 batch 独立。图片全部校验完成后才发布训练索引，自动进入 A1；A1 完成及验证结束后停止，不进入 A2/B。控制器约每20分钟更新自身和子进程，生成与训练均可恢复；手动停止后不再重启。

可以先 `stop`，再用同一命令和同一目录将 `--generation-batch-size` 改为20后 `start`；已完成图片及 A1 抽样计划保持不变。当前实际生成 batch 记录在输出目录的 `generation-runtime.json`，数据目录 `generation.json` 保留首次准备时的配置。生成进程会参考最近一个 batch 的耗时，尽量在本轮时间不足时提前保存退出。被强制中断的未完成图片会在下一轮重新生成。

`--a1-steps`、`--a1-batch-size`、`--grad-accum` 和实际 GPU 数决定数据需求。修改预算会改变数据指纹，应使用新的数据目录。生成或训练 OOM 会保留原因并停止，不自动降低 batch 或分辨率。已有的真实图片、索引、caption 缓存和旧训练结果保持原样。

数据目录内 `images/` 保存合成图像及逐图完成标记，`index/` 保存 manifest、原始样本映射和固定 batch plan。合成数据保留源样本 ID 与 caption，因此直接复用原有文本缓存。训练记录保存在独立的 `PIXELDIT_SYNTH_RUN/a1`，仍使用 SwanLab。若只想准备 caption 计划而不生成或训练，将 `start` 改成 `prepare`。

该实验检验图像来源分布对迁移的影响；caption 仍来自原数据集，且本轮固定正方形，与此前多长宽比训练也有区别，因此结果不能单独证明全部问题都来自图像分布。
