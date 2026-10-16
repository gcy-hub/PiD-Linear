# Spatially Disambiguated Delta Memory: first-stage probe

This branch is a research probe. It does not change the running VDN training job.
It uses a completed A2 checkpoint, not an in-progress B checkpoint. No new model
is trained, and no claim of quality-preserving acceleration follows from its images.

## Questions and controls

1. On 100 distinct held-out real images, are high-similarity VDA keys associated
   with different values, and do those locations have higher Attention error?
2. How much time would global multi-bank joint solves and routing actually cost
   inside the complete 25-step 2K/4K PiD decoder?

The diagnostic uses true images in the original 2K aspect buckets and the frozen
FLUX VAE on their 4x downsampled conditions. Captions come from the Gemma cache.
It evaluates `x_t=(1-t/1000)*x0+(t/1000)*noise` at t=100,500,900 and MMDiT layers
0,7,13. For each observed student hidden input, the original frozen Attention is
evaluated on that **same** input. This removes upstream hidden-input differences;
it is not the teacher's own sampling trajectory or a causal image intervention.

Keys/values come from the actual VDA path: raw shared projections, 5x5 K/V
convolution, SiLU and Q/K L2 normalization, **without RoPE**. The local branch and
PiT retain their existing RoPE. The conflict proxy is
`relu(cos(k_i,k_j)-0.8)*||v_i-v_j||^2/(||v_i||^2+||v_j||^2+eps)`.
Each image/layer uses 512 anchors and 64 candidates per anchor, equally divided
between global random candidates and nearby spatial offsets; self pairs are
excluded. No full N-by-N diagnostic matrix is formed. Near/far and same/cross
8x8-region statistics are separate. Effective key dimension uses FP32 covariance
on up to 4096 fixed-seed sampled keys and is an estimate of representation
redundancy, not proof of lost information.

Raw and write-gate-weighted conflict/error correlations are reported. Partial
rank correlations control teacher output energy, value energy and source-image
texture. Bootstrap intervals resample **images**, not individual tokens. Caption
keywords are weak image-level categories; Haar face boxes are candidate ROIs and
miss small/profile faces. Text ROIs need explicit manual annotations. A positive
correlation alone does not establish that memory collisions cause face/text
generation failures. A null result should not be hidden by selecting a different
threshold or layer after inspecting the results.

## Untrained speed prototypes

All versions preserve 14 MMDiT layers, the local Softmax branch, trained shared
QKV/output projections, K/V convolutions, VDA output normalization/gates and the
two original PiT layers. The variants are:

- `original`: official full Attention PiD.
- `hybrid`: the trained A2 regional bidirectional VDA.
- `fixed1`: optional single global-bank control.
- `fixed4`: four spatial quadrant banks, tied one-hot write/read routes and sparse
  gather. Non-square grids and incomplete partitions are supported.
- `dense4`: four banks, independent content+2D-position read/write routers.
  Routers are random and untrained. Every bank processes every token, so this
  measures the real dense routing cost, not an imaginary sparse implementation.

For each bank, FP32 statistics and Cholesky solve implement
`S_b=B_b*(I+A_b)^-1`, and read `sum_b pi_i,b*S_b*q_i`.
**These prototypes start image memory at zero, omit raster recurrence and decay,
include their own writes, and do not write text into their banks.** Text remains
available through unchanged joint local Softmax. Thus speed differences cannot
be attributed solely to adding banks; they also include removing regional solves
and scans. `fixed4` restricts image-memory exchange to quadrants; it is a control,
not the final conflict-aware routing method. No conflict-routing training loss,
single-layer calibration or full quality comparison is implemented in phase one.

The full benchmark warms the exact shapes and times the complete layer-8
Attention (including QKV/O), the whole network, MMDiT/PiT separately and actual
25-step CFG5/shift6 image generation. All methods run sequentially on the same
GPU, with the same assets, seeds and BF16 compute. Five full-network repeats
and twenty single-layer repeats are measured after warmup. Startup is separate.

## Run

```bash
conda activate linear-pid
export CHECKPOINT=./outputs/pid-vdn/a2/checkpoints/<completed-checkpoint>
export PID_WEIGHTS_ROOT=./weights
export PID_INDEX_ROOT=./dataset/MultiAspect-4K-1M/linear_pid_index
export PID_TEXT_CACHE_ROOT=./dataset/MultiAspect-4K-1M/linear_pid_text_cache
export PID_GALLERY_ROOT=./outputs/linear-pid/assets
export MEMORY_RESULTS=./outputs/research/spatial-delta-memory

# GPU IDs, workers and CPU threads are configurable. Multiple GPUs split images.
GPU_IDS=0,1 CPU_THREADS=1 bash scripts/diagnose_spatial_delta_memory.sh \
  --checkpoint "$CHECKPOINT" --weights-root "$PID_WEIGHTS_ROOT" \
  --index-root "$PID_INDEX_ROOT" --text-cache-root "$PID_TEXT_CACHE_ROOT" \
  --images 100 --workers 2 --threads 1 --max-seconds 1200 \
  --output-root "$MEMORY_RESULTS/diagnosis"

# Use one idle GPU for sequential, comparable model timings.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 python -m scripts.benchmark_spatial_delta_memory \
  --checkpoint "$CHECKPOINT" --weights-root "$PID_WEIGHTS_ROOT" \
  --gallery-root "$PID_GALLERY_ROOT" --warmup 2 --repeats 5 \
  --output-root "$MEMORY_RESULTS/speed"

python -m scripts.summarize_spatial_delta_memory --output-root "$MEMORY_RESULTS"
```

Repeat an interrupted command with identical arguments: finished layer/time
records and finished model/resolution measurements are skipped. Configuration
changes require a new output directory. The diagnostic deadline pauses between
images; the benchmark can be resumed between model/resolution measurements.
SwanLab is offline by default. Generated artifacts, local tests and model weights
are ignored by Git. Read `report.md`, `summary.json`, source ROI metadata, sampled
maps and per-condition records together; incomplete runs are labelled by counts.

Correctness checks compare joint solves and gradients against independent normal
equations, distinguish redundant keys from conflicting values, test the exact
two-identical-keys counterexample, non-square sparse/dense equivalence, padding,
batch isolation and state reset on repeated calls.

## First run: 2026-10-16

All 100 held-out images and 900 layer/time records completed. Eight full-network
measurements and actual images completed on the same A40 (one fixed square
condition at each resolution, matched across methods, 25 steps/CFG5/shift6).
These are PiD decoding times with cached text/latent inputs, not the frontend
Gemma/Z-Image latency. One generated sample per cell is a preliminary timing,
not a population-level generation benchmark.

| Model | 2K seconds | 4K seconds |
|---|---:|---:|
| Original PiD | 43.12 | 334.95 |
| Trained A2 VDN | 69.68 | 314.61 |
| Untrained fixed spatial 4-bank | 49.95 | 236.80 |
| Untrained content+position dense 4-bank | 70.47 | 314.77 |

The sparse fixed prototype has a speed opportunity at 4K (1.33x vs VDN, 1.41x
vs original PiD), but is slower than original PiD at 2K. Dense four-bank routing
does not improve latency in this implementation. The bank images do not establish
preserved generation quality; the state mechanism changed without adaptation.

The strongest association is layer7/t500: mean within-image Spearman0.380
(95% image bootstrap CI0.347–0.411), and partial rank correlation0.238 after
teacher/value energy and source texture controls. Layer13/t100 is negative
(-0.138; partial-0.171). The associations are not uniformly positive across
layers and times. Uncentered effective dimension is1.26–2.01, but centered
dimension is7.81–10.38 and common key means account for75–89% of energy. It
would be incorrect to interpret the raw dimension alone as collapsed memory.

Four manually checked face images and three text images give only small-sample
ROI observations. Those annotations are source-image boxes and are insufficient
to establish that collisions cause generated face/text failures. Current evidence
supports testing a targeted single-layer mechanism next, rather than claiming
that spatial conflicts are already proven to be the main quality bottleneck.
The ongoing B training has not been changed. All artifacts stay under the
configured output root; this branch is local and has not been pushed.
