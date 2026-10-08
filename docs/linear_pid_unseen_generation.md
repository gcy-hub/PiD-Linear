# 未参与此次微调的训练集图片：原始 PiD 与十层 KDA 对比

本次固定使用 step 3314 的十层直接转换学生，以及初始化它的官方未蒸馏 PiD。两侧共享 caption embedding、FLUX VAE latent、像素噪声 seed、25 步、CFG 5、shift 6、BF16 和 batch 1。5 张源图分别生成 2K、4K 输出，每种分辨率每个模型各 5 张，共 20 张生成图。

这是以真实图像 latent 为条件的 PiD 生成／重建比较，不是仅给 prompt 的纯文生图。未见样本只指该学生截至 checkpoint 的微调历史，不能证明发布权重的预训练没有使用这些图片。

`prepare_linear_pid_unseen.py` 回放确定性的 `GlobalBatchSampler`，验证 batch 转换历史与 `samples_seen` 一致，并同时排除被使用的 manifest 行、图片路径和同源 URL。它仅支持 `cursor == stage_step == total_step` 的直接初始化阶段；对无法完整回放的继承阶段明确报错。源图须足够大以进行原生 4K 裁剪，选中时完整解码检查。选择记录与准备好的条件包支持重复运行后跳过，不会重新抽样。

本次输出：`/home/ganchangyi/code/PiD-Linear/outputs/inference_comparison/unseen-train-step3314`。`assets/selection.json` 保存来源及采样回放审计；`original/`、`trained/` 各有 10 张完整输出。命名中的 `2048`／`4096` 表示 PiD 的分辨率桶，非方形图像的实际尺寸记录在 JSON 中。

```bash
conda activate linear-pid
cd /home/ganchangyi/code/PiD-Linear

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/prepare_linear_pid_unseen.py \
  --checkpoint /home/ganchangyi/code/PiD-Linear/outputs/linear-pid-kda10-direct/node01/kda_0-1-2-4-5-6-8-9-10-12/checkpoints/step_000003314_batch_000003314 \
  --output-root /home/ganchangyi/code/PiD-Linear/outputs/inference_comparison/unseen-train-step3314/assets \
  --count 5 --workers 2 --threads 1

LINEAR_UNSEEN_ASSET_FLAGS=()
for LINEAR_UNSEEN_RESOLUTION in 2048 4096; do
  for LINEAR_UNSEEN_ORDINAL in 000 001 002 003 004; do
    LINEAR_UNSEEN_ASSET_FLAGS+=(--asset "/home/ganchangyi/code/PiD-Linear/outputs/inference_comparison/unseen-train-step3314/assets/unseen_${LINEAR_UNSEEN_ORDINAL}_${LINEAR_UNSEEN_RESOLUTION}.pt")
  done
done

GPU_IDS=0,1,2,3 CPU_THREADS=1 bash scripts/compare_linear_pid_inference.sh \
  --checkpoint /home/ganchangyi/code/PiD-Linear/outputs/linear-pid-kda10-direct/node01/kda_0-1-2-4-5-6-8-9-10-12/checkpoints/step_000003314_batch_000003314 \
  --baseline original "${LINEAR_UNSEEN_ASSET_FLAGS[@]}" \
  --steps 25 --cfg 5 --shift 6 --repeats 1 --sample-warmup 0 \
  --network-repeats 1 --network-warmup 1 --threads 1 --workers 2 \
  --output-dir /home/ganchangyi/code/PiD-Linear/outputs/inference_comparison/unseen-train-step3314
```

GPU IDs 可以调整；进程数须与准备阶段的 `CUDA_VISIBLE_DEVICES` 对应。生成脚本会分配不同图片到不同 GPU，同一条件的两个模型在同一卡上顺序运行，每张输出完成后保存 PNG、JSON，重复运行跳过完整结果。已有结果还会校验硬件，因此中断后优先用相同的 GPU 列表续跑。

本次为定性生成而将完整采样预热设为 0，只运行一次正式采样。日志中的时间可能包含首次采样路径的编译／缓存开销，且 GPU 上还有其他任务，不作为严格速度结论；原有速度基准默认仍保留完整预热。
