# FLUX3 image/text mixing ablation

本实验的滑动窗口大小为 11×11 tokens。

Starting checkpoint: step 600, BF16 computation with FP32 master parameters and optimizer state. The two arms resume identical model, optimizer and scheduler states and use steps 601–700 of the same deterministic sampler. Each update consumes three samples: GPUs 1/2/3, batch 1, accumulation 1. Both arms retain the CLEAR loss, Sliding and stock FLA KDA.

The calibrated arm adds a fixed per-layer, per-head bias to text-key logits for image queries in the 28 joint layers. Text queries and the five image-only layers are unchanged. Within each conditional image or text distribution, relative scores are unchanged; the bias changes their joint softmax mixture. KDA itself is unchanged. This ablation does not implement MM-EDiT's kernel attention or token-count mixture.

The calibration bias is the mean of `logsumexp(local image scores) - logsumexp(all image scores)` on four cached samples at noise levels 0.2, 0.5 and 0.8. A query-specific value would restore the full-attention text mass for that same Q/K; a fixed per-head mean is an approximation. Two further samples are used for the independent check. Measurements use 64 spatial query positions, all heads, and all 33 replaced layers. Text-mass summaries exclude image-only layers and average sampled queries, heads, layers and noise levels.

| Inputs | Full-attention text mass | Sliding text mass |
|---|---:|---:|
| Teacher trajectory, calibration | 36.79% | 55.36% |
| Teacher trajectory, independent samples | 35.10% | 53.28% |
| Student trajectory, calibration | 24.08% | 48.19% |
| Student trajectory, independent samples | 22.32% | 45.42% |

Each row compares full and local attention with identical Q/K. Teacher and student rows have different hidden states and projections, so their difference is not a controlled measure of mixing alone. On the independent samples, the mean KDA/local-output RMS ratio is 7.08%. This ratio does not measure whether KDA supplies the missing spatial information.

On the two independent samples, calibration reduces sampled text mass from **45.42% to 21.95%**, compared with **22.32%** for full attention using the same Q/K. Mean relative attention-output MSE falls from **0.31719 to 0.24056**, a **24.2%** reduction. These are attention-level comparisons on unchanged student hidden states, not end-to-end generation results. The baseline local-only error is 0.31200; adding the current KDA branch raises it to 0.31719. Mean cosine similarity between KDA and the missing global-minus-local output is 0.00042.

End-to-end evaluation uses the same three prompts as the step-600 samples: London bus, cafe, lighthouse; seed 0, 1024×1024, 20 denoising steps, CFG 3.5. Each arm generates three images after 100 updates.

Outputs are under `../experiment/flux3-mixing-ab/`: `diagnostics/`, `heldout-check/`, `start/`, `baseline/`, `calibrated/`, and the final `comparison.jpg`. `mixing_experiment.py --check` is scheduled by node01 cron. It waits for at least 40960 MiB free on each selected GPU, avoids duplicate workers with an inherited lock, resumes published checkpoints after process cleanup, and stops on explicit errors. The original training launcher is paused while the experiment owns its checkpoint and GPU allocation, then released after completion.

## Paired 100-step results

Both arms completed steps 601–700 and generated all three matched images. Mean training losses over the same 100 batches:

| Metric | Baseline | Fixed text-logit bias |
|---|---:|---:|
| CLEAR total | 0.69800 | 0.75125 |
| Prediction distillation | 0.08528 | 0.11677 |
| Attention distillation | 0.16442 | 0.17785 |

These averages include adaptation immediately after introducing the bias. They measure training behavior, not held-out image quality.

In the six-image comparison, both arms produce a recognizable cafe and a single lighthouse. The calibrated bus is strongly elongated and deformed; the baseline bus is small and clipped at the edge of the scene. The calibrated cafe contains an additional small cup. The lighthouse compositions differ, with no clear overall quality advantage from the bias. These three prompts at one seed do not establish a general quality ranking. The attention-level error reduction did not translate into a consistent visible improvement in this comparison.

The unmodified main training continues from the baseline step-700 model and optimizer, with validation at steps 800, 900 and 1000. The calibrated checkpoint remains a separate experiment.

Validation: zero-bias identity, unchanged text-query outputs, CUDA forward/backward agreement with a dense CPU reference, and scheduler waiting/launch/completion checks.
