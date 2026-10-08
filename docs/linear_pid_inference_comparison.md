# 相同 prompt 下的 PiD 推理速度对比

入口为 `scripts/compare_linear_pid_inference.sh`，实现为 `pid/_src/linear_pid/compare_inference.py`。不读取训练 optimizer 到 GPU，不改动训练进程或训练输出。输入为完整学生 checkpoint、训练目录（自动选择最近完整 checkpoint）或学生导出文件。

默认基线 `--baseline original` 是完整未改造的 PiD v1.5；`--baseline untrained-kda` 则完整加载原始 PiD 后，用训练配置中的 seed 随机初始化相同 KDA 布局，代表微调前的学生。两侧均使用 BF16、batch 1、相同 PiT chunk、相同 prompt embedding、有效文本 mask、latent、像素噪声 seed、采样步数、CFG 和 shift。参数由 checkpoint 重建，不加载四层学生。

两个模型依次加载到同一 GPU、测量后删除释放，编码器也在准备输入后释放；采样计时期间没有模型 CPU 卸载。多 GPU 时不同输入分配到不同 GPU，但同一输入的基线和学生始终由同一 GPU 测量。硬件繁忙程度也会影响结果，请在对应 GPU 可用时手动运行。脚本不会停止已有训练。

## 使用已有固定 prompt 和 latent

下面是一个 prompt 的完整对比：单次网络前向预热 3 次、测量 10 次；完整 PiD 采样预热 1 次、测量 3 次。重复测量使用同一 seed。

```bash
conda activate linear-pid
cd /home/ganchangyi/code/PiD-Linear
GPU_IDS=0 CPU_THREADS=1 bash scripts/compare_linear_pid_inference.sh \
  --checkpoint /home/ganchangyi/code/PiD-Linear/outputs/linear-pid-kda10-direct/node01/kda_0-1-2-4-5-6-8-9-10-12 \
  --baseline original --case generated_000_2048 \
  --steps 25 --cfg 5 --shift 6 \
  --repeats 3 --sample-warmup 1 --network-repeats 10 --network-warmup 3 \
  --threads 1 --workers 2 \
  --output-dir /home/ganchangyi/code/PiD-Linear/outputs/inference_comparison/original-vs-trained-2k
```

这个输入的 prompt 是 `A red fox standing in snowy woodland at sunrise, fine fur, soft light.`，文本条件和 latent 直接复用现有画廊文件，不需要重新加载 Gemma 或 Z-Image。`--asset /absolute/path/to/prepared.pt` 可以指定其他已经准备好的条件包。多个 `--case` 时加 `--limit 0`，防止默认 limit 1 限制选择。

想比较随机 KDA 学生与训练后学生，改成 `--baseline untrained-kda`，并使用另一个 `--output-dir`。这反映训练前后速度，不是 Full Attention 与 KDA 的架构速度对比。

多卡、多 prompt 完整对比示例：

```bash
conda activate linear-pid
cd /home/ganchangyi/code/PiD-Linear
GPU_IDS=0,1,2,3 CPU_THREADS=1 bash scripts/compare_linear_pid_inference.sh \
  --checkpoint /absolute/path/to/complete/checkpoint \
  --baseline original --resolution 2048 --kind generated --limit 8 \
  --steps 25 --cfg 5 --shift 6 --repeats 3 --sample-warmup 1 \
  --threads 1 --workers 2 \
  --output-dir /home/ganchangyi/code/PiD-Linear/outputs/inference_comparison/eight-prompts-2k
```

4K 对比使用 `--resolution 4096`，选择已有 4K 画廊；不要直接把 2K 条件包中的尺寸改成 4K。

## 自定义 prompt

```bash
conda activate linear-pid
cd /home/ganchangyi/code/PiD-Linear
GPU_IDS=0 CPU_THREADS=1 bash scripts/compare_linear_pid_inference.sh \
  --checkpoint /absolute/path/to/complete/checkpoint \
  --baseline original \
  --prompt 'A mountain lake reflecting snow-covered peaks, pine trees, crisp fine details.' \
  --height 2048 --width 2048 --seed 42 \
  --steps 25 --cfg 5 --shift 6 --repeats 3 \
  --output-dir /home/ganchangyi/code/PiD-Linear/outputs/inference_comparison/custom-lake
```

PiD 需要 latent 条件。只有 prompt 时，先用本地 Z-Image-Turbo 生成一份固定 latent，再用 Gemma 编码 prompt，缓存到 `conditions/custom.pt`。两侧复用该文件。也可以提供 `--image /absolute/path/to/image.jpg`，经 Lanczos 中心裁剪、4 倍下采样及冻结 FLUX VAE 编码；或提供 `--latent /absolute/path/to/latent.pt`。输出尺寸必须是 32 的倍数，latent 必须为 `[1,16,height/32,width/32]`。自定义 4K 使用 `--height 4096 --width 4096`。

## 统计结果与恢复

- `original/` 或 `untrained-kda/`：基线生成图和逐输入测量 JSON。
- `trained/`：训练后生成图和逐输入测量 JSON。
- `summary.csv`、`summary.json`：相同输入的中位采样耗时、单次前向耗时、显存峰值及加速比。
- `conditions/`：自定义 prompt 的一次性条件缓存及准备耗时。

`sampling_speedup = 基线中位采样耗时 / 学生中位采样耗时`，大于 1 表示学生更快。原始逐次耗时、平均值、中位数、标准差、最小和最大值在每个输入的 JSON 中。显存分别记录 PyTorch allocated 和 reserved 峰值。

单次网络前向是条件分支，固定 t=500，不含 CFG；完整采样计时包含指定 CFG 的两个分支和全部采样步数、噪声初始化与 solver。它不包含模型加载、文本编码、latent 生成/VAE 编码、输入传输、预热、GPU→CPU 图片转换和 PNG 写盘，不能解释为整个文本生成图像流水线的端到端时间。两侧输入准备只执行一次，避免改变条件造成比较偏差。

同一命令、同一 checkpoint、同一 GPU、同一输出目录重复运行即可续做。生成图和测量 JSON 都完整存在后才跳过该输入；checkpoint、条件文件、参数或硬件发生变化会拒绝复用旧结果。部分完成也保留在各模型子目录，最终全部完成后再输出汇总。不要同时运行两个命令写入相同输出目录。

无 GPU 的输入检查可使用直接 Python 入口：

```bash
conda activate linear-pid
cd /home/ganchangyi/code/PiD-Linear
python -m pid._src.linear_pid.compare_inference \
  --checkpoint /absolute/path/to/complete/checkpoint \
  --case generated_000_2048 --dryrun
```

已完成相同输入/seed 复用、采样预热排除、计时统计、缓存失效、硬件配对及采样回归的 CPU 测试；GPU 实际测速需要在可用卡上运行上述命令。

## MMDiT / PiT 分段计时

添加 `--profile-sections`，使用 CUDA events 记录测量采样中的每一次网络调用。预热不计入，也不在各层之间 synchronize。hook 和 event 记录仍有少量开销，因此分段计时的总耗时与关闭计时的独立测速可能不同。共享 GPU 上其他任务会影响测量，结果应注明卡是否独占。

```bash
conda activate linear-pid
cd /home/ganchangyi/code/PiD-Linear
GPU_IDS=2 CPU_THREADS=1 bash scripts/compare_linear_pid_inference.sh \
  --checkpoint /home/ganchangyi/code/PiD-Linear/outputs/linear-pid-kda10-direct/node01/kda_0-1-2-4-5-6-8-9-10-12/checkpoints/step_000003314_batch_000003314 \
  --baseline original --case generated_000_2048 \
  --steps 25 --cfg 5 --shift 6 --repeats 3 --profile-sections \
  --threads 1 --workers 2 \
  --output-dir /home/ganchangyi/code/PiD-Linear/outputs/inference_comparison/sections-2k-step3314
```

每侧测量 JSON 的 `section_profile` 包含调用数、各 section 的逐次耗时、总耗时、平均/中位每次前向耗时。`summary.csv/json` 同时增加两侧 MMDiT、PiT、Attention 和其他工作的每次前向平均耗时及加速比。

- `mmdit`：14 层主干循环，含各层间的 LQ 注入门、Attention、FFN、AdaLN 与残差。
- `pit`：两层完整 PiT block 之和，含调制、压缩、Attention、展开、像素 MLP 和残差。
- `other`：全网络调用减去上述两部分，包括输入/条件准备、像素 embedding、PiT 条件注入、输出头及 fold。
- `mmdit_attention` / `pit_attention`：对应 Attention 子模块，已经包含在父 section 中，不能再次相加。
- `kda_attention` / `full_attention`：MMDiT Attention 的两种类别之和；还保留逐层统计。

分段耗时的单位是毫秒，汇总为一次完整网络前向（一个 CFG 分支）的平均值；每张图的采样统计包含所有时间步及两个 CFG 分支。`network_calls` 可以核对实际调用次数。

开启分段计时后，汇总中的 `baseline_forward_ms` / `trained_forward_ms` 使用这些采样网络调用的平均值，与分段耗时保持相同口径；`forward_timing_scope` 标明口径。每侧 JSON 中的 `network_forward` 仍保留采样预热前的独立 t=500 条件前向测量，不能直接与分段均值相加或混用。GPU 负载或运行阶段发生变化时，两套测量可能不同。
