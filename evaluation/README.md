# 4K PiD evaluation

The frozen [DPG manifest](data/dpg1000.jsonl) selects 1,000 of the 1,065 unique prompts in TencentQQGYLab/ELLA (`3c228f1dc6c4d3cad0a47493816151a419f14db3`, `dpg_bench/dpg_bench.csv`). Proposition rows are not additional prompts. To reproduce: `python evaluation/prompts.py ~/dataset/DPG-Bench/dpg_bench.csv`. IDs, source IDs, captions and upstream seeds are stored in order. This is a reproducible subset, not a claim to know the authors' unpublished sample IDs. The [1,000 ImageNet classes](../pid/_src/inference/prompts/imagenet_classes.txt) are a separate conditioning set for the paper's DINOv2 experiments, not captions for these VAE decoders.

## Generate

Use one reserved 8-GPU node or a single Slurm allocation spanning eight GPUs, with the repository, weights and output on shared storage. Use the same cards for all phases. The output is 2,000 RGB PNGs at 4096 x 4096. Ten measured pairs indicate about 57 GB for PNGs; the 500 shared condition batches add about 3.1 GB. Reserve 75 GB for outputs (model weights are separate). From an activated inference environment inside the allocation:

```bash
export WEIGHTS_ROOT=/shared/weights  # contains PiD/checkpoints, gemma-2-2b-it, Z-Image-Turbo
export PID_EXPORT=/shared/weights/b-7629  # ChangyiGan/PiD-halo2-sliding11-7629, including config.yaml
bash evaluation/run.sh /shared/results/pid-eval-dpg1000-4k
```

Eight independent workers cache Z-Image-Turbo 1024-square latents and full Gemma positive/negative text conditions. Decode stages are serial: official PiD, then b-7629 with native PiT. Each GPU processes 62 or 63 pairs at batch size two, with sequential CFG branches. Both models use the same conditions, pair order and decoder generator seeded by the first row's seed (the second image consumes the second noise draw). The upstream LDM uses each individual manifest seed. Both decoders use global 4096-square sampling, 25 steps, CFG 5, shift 6 and sigma 0. Each completed pair has one timing record. Resume on the same hardware and configuration.

## Efficiency

Record synchronized decoder latency from on-device inputs to the returned pixel tensor. Exclude encoding, LDM, transfers, model loading, PNG writing and quality scoring. Include each worker's first decode. `efficiency.json` gives mean two-image batch latency, amortized seconds per image (`sum(batch_seconds) / 1000`), speedup versus official PiD and hardware/software identity. These are not batch-one user latency or cluster throughput. Both models run eagerly with optimized inference kernels and no whole-model `torch.compile`.

## No-reference quality

Score each full-resolution PNG using the repository's PiD metric implementation, including its metric-specific preprocessing. No common downsampling or reference image. Use a separate environment with PyTorch/torchvision, `torchmetrics`, `pyiqa` and compatible Transformers/VLM dependencies. Weights and caches must be staged before offline scoring. The metric keys below are CLI arguments; each writes one JSON with 1,000 ordered scores and a mean per model. Run one process per metric to avoid retaining multiple large scorers in GPU memory.

| Metric key | Resource | Better |
|---|---|---|
| `musiq_paq2piq` | pyiqa `musiq-paq2piq` weights, not default MUSIQ | Higher |
| `niqe` | pyiqa natural-scene statistics; no learned neural checkpoint | Lower |
| `deqa_score` | `zhiyuanyou/DeQA-Score-Mix3` with vendored scorer | Higher |
| `maniqa` | pyiqa `maniqa`, including its backbone | Higher |
| `qalign_native` | pyiqa `qalign`, native full-image quality score (not aesthetic or multi-crop) | Higher |
| `unipercept_iaa`, `unipercept_iqa` | `Thunderbolt215215/UniPercept`, one checkpoint for aesthetics and quality | Higher |
| `visualquality_r1` | `TianheWu/VisualQuality-R1-7B`, Transformers greedy generation | Higher |

```bash
for metric in musiq_paq2piq niqe deqa_score maniqa qalign_native unipercept_iaa unipercept_iqa visualquality_r1; do
    HF_HUB_OFFLINE=1 python evaluation/report.py /shared/results/pid-eval-dpg1000-4k "$metric" || break
done
```

Source: local PiD manuscript, `sections/experiments.tex`, “Evaluation Protocol”, and `tables/main-table-vae.tex`.
