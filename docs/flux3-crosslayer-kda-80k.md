# FLUX3 cross-layer KDA：80K 多长宽比训练

目标服务器：8 张 H800，Python 3.12。使用 FLUX3 klein-base 原始基础权重，全部 5 个 early_image 层与 28 个 joint 层替换为局部窗口 11 + 标准 FLA KDA。每层只扫描一个方向，按完整原生层序号 `ordinal % 4` 轮换；不在单层做四方向融合。

训练使用官方 PiD 2K–4K 多长宽比桶（`data.buckets=official`，max-resolution 表为 4096），不强制 4096 正方形。每卡 batch 4、累积 1、全局 batch 32，50,000 步；每 2,000 步保存与验证，训练 seed 42，验证 20 步、seed 0。CLEAR 损失为 FM + 0.5 prediction KD + 0.5 feature KD；Prodigy lr=1、weight_decay=0.0001、constant。BF16 计算，FP32 master 与优化器历史状态，保持 `training.fp32_updates=true`。

## 环境与本地资源

```bash
git clone https://github.com/gcy-hub/PiD-Linear.git
cd PiD-Linear
conda create -n flux3-kda python=3.12 pip -y
conda activate flux3-kda
python -m pip install -e .
python -m pip install prodigyopt \
  'flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git@9f38d24980c46d46bd38614e743cdacd21906578'
git rev-parse HEAD
```

记录实际运行 commit、环境与 GPU 型号。PyTorch/CUDA、FLA 的内核依赖应使用服务器已有的兼容环境；不要覆盖其他任务环境。查找服务器已下载的模型和数据，设置以下本地绝对路径。模型资源必须来自同一 FLUX3 klein-base 模型包：weights 为基础模型权重文件，text_encoder 和 tokenizer 为本地目录，vae 为 safetensors 文件。PiD 下载脚本不包含这些 FLUX3 资源。

```bash
export FLUX3_WEIGHTS=/absolute/path/to/flux3-klein-base.safetensors
export FLUX3_TEXT_ENCODER=/absolute/path/to/text_encoder
export FLUX3_TOKENIZER=/absolute/path/to/tokenizer
export FLUX3_VAE=/absolute/path/to/vae.safetensors
export SOURCE_MANIFEST=/absolute/path/to/multiaspect-80k/manifest.jsonl
export DATA_ROOT=/absolute/path/to/prepared-flux3-80k
export RUN_OUTPUT=../experiment/flux3-crosslayer-kda-80k
```

使用新的输出目录。这里不恢复本机旧 4,000 步快照，不导入历史 FLUX 缓存或 checkpoint。

## 导出实际可用 manifest 与桶统计

源 manifest 每行含 `id`、`image_path`、`caption`，相对图像路径按源 manifest 目录解释。80K 是数据集目标名称，训练长度取实际导出记录数。先下载完成，再运行下列 CPU 导出；缺失/损坏图像应修复，不能把读取错误当成成功过滤。过滤统计包含缺图像字段、尺寸不足及官方桶拒绝；导出路径转为绝对路径。

```bash
python - <<'PY'
import json, os
from collections import Counter
from pathlib import Path
from data.dataset import ImageDataset

source = Path(os.environ['SOURCE_MANIFEST'])
output = Path(os.environ['DATA_ROOT'])
output.mkdir(parents=True, exist_ok=True)
dataset = ImageDataset(source, buckets='official', min_size=256)
manifest = output / 'manifest.jsonl'
with manifest.open('w') as stream:
    for sample in dataset.samples:
        stream.write(json.dumps(dict(id=sample.sample_id, image_path=sample.image_path,
                                     caption=sample.caption), ensure_ascii=False) + '\n')
report = dict(source=str(source.resolve()), accepted=len(dataset),
              rejected=dict(dataset.rejected),
              buckets={f'{w}x{h}': n for (w, h), n in Counter(dataset.sizes).items()})
(output / 'filter-report.json').write_text(json.dumps(report, indent=2))
assert sum(report['buckets'].values()) == report['accepted']
print(json.dumps(report, indent=2))
PY
export MANIFEST="$DATA_ROOT/manifest.jsonl"
```

官方桶由图像尺寸和长宽比选择，分辨率等级 2048–4096；非正方形的具体宽高以 `filter-report.json` 为准。报告实际数量、各桶数量及拒绝原因，不把过滤后的数量写成固定 80,000。

## 配置检查与训练

Bash 数组复用同一份参数，先只打印解析后的配置，再启动训练：

```bash
args=(
  --model-config configs/models/flux3.yaml
  --algorithm-config configs/algorithms/cross_scan_kda.yaml
  --training-config configs/training/clear.yaml
  --data-config configs/data/default.yaml
  "model.resources.weights=$FLUX3_WEIGHTS"
  "model.resources.text_encoder=$FLUX3_TEXT_ENCODER"
  "model.resources.tokenizer=$FLUX3_TOKENIZER"
  "model.resources.vae=$FLUX3_VAE"
  "paths.dataset=$DATA_ROOT" "data.manifest=$MANIFEST"
  "paths.output=$RUN_OUTPUT"
  replacement.layers.early_image=all replacement.layers.joint=all
  algorithm.local.window_size=11 data.buckets=official
  training.batch_size=4 training.accumulation=1 runtime.gpus=8
  training.steps=50000 training.seed=42
  training.optimizer=prodigy training.lr=1.0 training.weight_decay=0.0001
  training.scheduler=constant training.fp32_updates=true
  training.loss.fm=1.0 training.loss.prediction=0.5 training.loss.feature=0.5
  runtime.dtype=bfloat16 cache.image=false cache.text=false
  checkpoint.every=2000 validation.every=2000 validation.steps=20 validation.seed=0
)
python -m training.run --print-config "${args[@]}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m training.run "${args[@]}"
```

检查打印结果中 global_batch_size=32、33 个选定层、资源及输出路径。`training.run` 内部分段启动 torchrun workers；全部训练 rank 退出后才独立验证，再继续训练。当前 FLUX3 原生验证使用 CFG=1。保存 `resolved.yaml`、训练日志、完整 checkpoint 和验证图片。

默认在线编码图像和文本。可选当前框架缓存：

```bash
export CACHE_ROOT=/absolute/path/to/current-framework-cache
python -m data.prepare "${args[@]}" --phase all --devices cuda:0 --cache-batch-size 1 \
  "cache.root=$CACHE_ROOT"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m training.run "${args[@]}" \
  cache.image=true cache.text=true "cache.root=$CACHE_ROOT"
```

准备和训练必须使用同一导出 manifest、资源及桶设置。缓存是否存在不改变默认在线模式。显存选项默认全部关闭，按实际 OOM 情况显式选择 `runtime.gradient_checkpointing=true`、`runtime.optimizer_sharding=true`、`runtime.activation_cpu_offload=true`、`runtime.optimizer_cpu_offload=true`；记录实际开关与峰值，不降低精度、层数、分辨率或 batch 来冒充目标运行。

同一输出目录会自动选择最新完整 checkpoint；显式恢复只能使用当前框架的完整目录，固定分片 world size：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m training.run "${args[@]}" \
  checkpoint.resume=/absolute/path/to/current-run/step-000002000
```

## 可复制给目标服务器 Pi agent 的任务

> 在 https://github.com/gcy-hub/PiD-Linear.git 当前 main 上执行本页训练，先记录实际 commit。查找本机已有 FLUX3 klein-base 基础权重、TE、tokenizer、VAE 和 80K MultiAspect 数据，填写本地资源路径；导出官方 2K–4K 多长宽比 manifest，报告实际样本数量、过滤原因和桶统计。按上述 args 先 print-config，再执行真实八张 H800 启动和 50,000 步训练，每卡 batch4、累积1、global32，33 层局部窗口11加标准单方向 FLA KDA，原生层序号轮换方向；CLEAR FM+0.5pred+0.5feature，Prodigy lr1、wd0.0001、constant，BF16计算和FP32更新。默认在线编码，可按本文启用当前框架缓存。每2000步保存和验证，训练seed42，验证20步seed0。使用仓库旁 experiment 输出并保留日志、实际配置、checkpoint及图片。遇到实际启动错误修复具体原因；OOM 可使用已有显存开关并记录，不缩小模型、层数、分辨率或batch。不重复运行全部模型验收矩阵；若没有8张H800，报告资源阻塞和已完成的CPU准备。使用新模型初始化，不加载旧4000步本地实验快照。恢复只用当前框架checkpoint，不改动其他任务或生产缓存恢复服务。

## Limitations

目标 H800 八卡全量训练尚未验证；配置解析和 CPU 导出不能证明目标显存、完整恢复、收敛或画质。已有四卡 A40 的通过项目与 OOM 记录见 [验收记录](attention-framework-outstanding-tests.md)。
