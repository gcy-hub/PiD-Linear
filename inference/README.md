# 两张全局 8K 图

脚本内置教堂、人物两段 prompt，每张图先运行 Gemma 文本编码和 Z-Image-Turbo 2048×2048 latent 生成，再由 PiD **单次全局解码为 8192×8192**，不切块、不拼接。需要足够显存；44GB 单卡不适合全局 8K。

训练后 export（默认模式，`--repo` 指向导出目录）：

```bash
python inference/generate_pid_8k_two_prompts.py \
  --repo /home/ganchangyi/code/experiment/pid-qdm-halo2-4k-to-sliding-and-wavelet/b-7629 \
  --weights-root /home/ganchangyi/huggingface_ckpts --gpus 0,1
```

原始 PiD（直接加载原始权重，不安装训练 adapter，无需 export）：

```bash
python inference/generate_pid_8k_two_prompts.py --model original \
  --weights-root /home/ganchangyi/huggingface_ckpts --gpus 0,1
```

可从其他工作目录使用脚本绝对路径运行。权重目录需包含 `gemma-2-2b-it`、`Z-Image-Turbo` 和 `PiD/checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_flux_undistilled/model_ema_bf16.pth`，全部从本地加载。

`--gpus 0,1` 两卡并行，`--gpus 0,0` 同卡逐张顺序执行。默认 seed 为 1045、1046，PiD steps=25、cfg=5、shift=6。可通过同名参数修改。

两种模式默认分别保存到当前目录的 `outputs/pid-export-8k-two-prompts` 和 `outputs/pid-original-8k-two-prompts`，避免覆盖历史 step7629 输出。可用 `--output` 指定目录；重复使用同一输出目录会覆盖其中的结果。每张图在 `image-01` / `image-02` 下保存 PNG、条件张量、日志和 `result.json`。

每张图的 `result.json` 记录：

- `generation_seconds`：从文本编码器加载前，到文本编码、Z-Image 模型加载及 latent 生成、PiD 加载及采样、PNG 保存完成。
- `sampling_seconds`：仅 PiD 采样，调用前后都执行 CUDA 同步。
- `decode_seconds`：从 PiD 加载前到 PNG 保存完成，可用于和历史解码阶段耗时比较。

`run.json` 的 `elapsed_seconds` 是两张图整个调度过程的墙钟时间，包括子进程启动开销。

CPU mock 回归检查：`python -m pytest -q inference/check_pid_8k_two_prompts.py`。
