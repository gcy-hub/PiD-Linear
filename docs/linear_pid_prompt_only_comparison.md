# 仅输入 prompt 的 PiD／KDA 对比

正确的文生图流程是 `prompt → Z-Image-Turbo 生成 latent → 原始 PiD／十层 KDA 解码`。这次不会读取数据集源图，不使用真实图片经 VAE 编码的 latent，也不会读取训练 Gemma 特征缓存。Gemma 条件从 prompt 重新编码。

使用此前选出的 5 条未参与 step 3314 学生微调的训练集 caption，保留对应的长宽比；每条 caption 分别生成 2K、4K 条件。两个解码器共享固定的生成 latent、文本特征和像素噪声 seed，避免基础模型的随机差异影响比较。结果各 10 张，共 20 张；预览只有 PiD 和 PiD-KDA 两列。

输出目录：`/home/ganchangyi/code/PiD-Linear/outputs/inference_comparison/unseen-prompts-step3314`。之前 `unseen-train-step3314` 目录是源图条件重建，不属于这次 prompt-only 结果。

完整运行／续跑命令：

```bash
conda activate linear-pid
cd /home/ganchangyi/code/PiD-Linear
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 先生成基础 latent、重新编码文本；GPU IDs 与进程数可调整。
CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 \
  scripts/prepare_linear_pid_prompt_inputs.py \
  --checkpoint /home/ganchangyi/code/PiD-Linear/outputs/linear-pid-kda10-direct/node01/kda_0-1-2-4-5-6-8-9-10-12/checkpoints/step_000003314_batch_000003314 \
  --selection /home/ganchangyi/code/PiD-Linear/outputs/inference_comparison/unseen-train-step3314/assets/selection.json \
  --output-root /home/ganchangyi/code/PiD-Linear/outputs/inference_comparison/unseen-prompts-step3314/assets \
  --threads 1

LINEAR_PROMPT_ASSET_FLAGS=()
for LINEAR_PROMPT_RESOLUTION in 2048 4096; do
  for LINEAR_PROMPT_ORDINAL in 000 001 002 003 004; do
    LINEAR_PROMPT_ASSET_FLAGS+=(--asset "/home/ganchangyi/code/PiD-Linear/outputs/inference_comparison/unseen-prompts-step3314/assets/prompt_${LINEAR_PROMPT_ORDINAL}_${LINEAR_PROMPT_RESOLUTION}.pt")
  done
done

GPU_IDS=0,1,2,3 CPU_THREADS=1 bash scripts/compare_linear_pid_inference.sh \
  --checkpoint /home/ganchangyi/code/PiD-Linear/outputs/linear-pid-kda10-direct/node01/kda_0-1-2-4-5-6-8-9-10-12/checkpoints/step_000003314_batch_000003314 \
  --baseline original "${LINEAR_PROMPT_ASSET_FLAGS[@]}" \
  --case-gpu-map /home/ganchangyi/code/PiD-Linear/outputs/inference_comparison/unseen-prompts-step3314/case_gpu_map.json \
  --steps 25 --cfg 5 --shift 6 --repeats 1 --sample-warmup 0 \
  --network-repeats 1 --network-warmup 1 --threads 1 --workers 2 \
  --output-dir /home/ganchangyi/code/PiD-Linear/outputs/inference_comparison/unseen-prompts-step3314
```

准备和生成均支持重新运行后跳过完整结果。准备阶段每个生成 latent 保存一次，并记录独立阶段耗时；生成阶段每张输出完成后发布 PNG、JSON。本次 GPU 1 的共享负载变高后，未完成的两组 4K 配对转移到 GPU 0／2，2K 蝴蝶配对在 GPU 3 重跑；上面的 `--case-gpu-map` 保留最终分配，使已完成结果能正确复用。全新实验可以省略此参数，按输入顺序平均分配。

计时分别记录 Z-Image-Turbo 的 prompt 编码与 latent 生成、Gemma 条件编码，以及 PiD 采样。两阶段时间之和可用于估算 prompt 到像素输出的计算时间，但不包含模型加载和阶段间 I/O。共享 GPU、本次单次采样及未做完整采样预热的设置不适合严格速度结论。
