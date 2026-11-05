# FLUX.3 Sliding-QDM Implementation Plan

**Goal:** Train 28 joint-layer Sliding11+fixed4 QDM on native 2K buckets, serial A1 200/A2 400/B 10000 updates, validation every 200 with ten student images, GPUs 1/2/3.

**Architecture:** Preserve original Base, early mode blocks and existing three-stage objectives. Add per-query clipped square FlexAttention with jointly normalized text. Carry window identity through training, checkpoint, phase transition and both inference entry points. Preserve source worktree changes in an independent FLUX3-sliding-QDM branch.

- [x] Inspect native model, cache, training and renewable controller. Confirm complete 20K cache and seven 2K buckets; 50K cache incomplete. Ask data-scope preference while implementing independent code.
- [x] Write failing output/gradient, clipped-edge, checkpoint-backward, schedule and identity tests.
- [x] Add `flux3_qdm/sliding.py`; integrate geometry/kernel into QDMBlock/model and inference reconstruction. Preserve legacy Halo behavior by default.
- [x] Add explicit per-stage step options; serial/resumable launcher and portable documentation.
- [x] Run CPU/CUDA numerical and native-block checks. Verify student gradients, teacher equivalence and sample isolation.
- [x] Run real 2K three-stage short training and checkpoint continuation; remove temporary weights after checks.
- [ ] Launch requested background experiment and verify completed optimizer updates, actual batch, bucket sizes, peak memory and live controller.

No Slurm submission, remote push, CPU offload, encoder loading during training or inheritance of prior experimental deltas. A1/A2 train new QDM parameters; B adds rank64 alpha64 LoRA and local gate. New root under ignored outputs, existing experiments preserved.

## Verification

11 local CPU/CUDA checks passed, including clipped edges, packed isolation, native teacher equivalence, full-coverage reconstruction, zero LoRA, non-reentrant checkpoint backward, variable text lengths and all seven aspect layouts. Dynamic/fullgraph FlexAttention avoids a reproduced static-compile fallback to dense eager attention. A1 restored step1 and completed step2; A2/B each completed one real 2K bs2 update and loaded their preceding stage. Peaks: A1 23.87 GiB, A2 30.71 GiB, B 31.77 GiB. Disposable test checkpoints removed; evidence remains in ignored research outputs.

Formal 1/2/3 launch paused before any update: an unrelated process took about25GB on GPU3 during startup. GPU selection question is pending; do not alter other workloads or silently change effective batch.
