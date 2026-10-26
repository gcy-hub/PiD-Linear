# PixArt-Σ 合成图文数据生成

使用原始 PixArt-Σ 2K 模型生成图片及对应 caption。默认 2048×2048、20 步二阶 DPM-Solver、CFG 4.5，T5 最大长度 300。它复现论文的数据构建方式；作者没有公布原始 12K 样本清单，因此不保证与论文逐图一致。CFG 4.5 使用原生 pipeline 默认值，论文未明确给出生成训练数据时的 CFG。

## 换服务器：八卡、每卡 batch 2 续生成

拉取 `main`，在新服务器准备环境和同一套模型：

```bash
conda create -n linear-pid python=3.12 -y
conda activate linear-pid
pip install torch==2.10.0
pip install -r grafting/tools/requirements-pixart.txt
python grafting/tools/download_pixart_models.py \
  --weights-root "$HOME/huggingface_ckpts" --endpoint https://hf-mirror.com --workers 2
```

将数据包和对应 `.sha256` 文件放在同一个目录，校验并解压到数据目录（将包名改成实际名称）：

```bash
sha256sum -c PixArt-Sigma-Grafting-12K-partial.tar.sha256
mkdir -p "$HOME/dataset"
tar -xf PixArt-Sigma-Grafting-12K-partial.tar -C "$HOME/dataset"

conda activate linear-pid
export PIXART_DATA_ROOT="$HOME/dataset/PixArt-Sigma-Grafting-12K"
export PIXART_WEIGHTS="$HOME/huggingface_ckpts/PixArt-Sigma-XL-2-2K-MS"
GPU_IDS=0,1,2,3,4,5,6,7 CPU_THREADS=1 \
bash grafting/tools/generate_pixart_8gpu.sh
```

这是直接运行的完整生成任务，无需 Slurm；默认每卡 batch 2、最终共 12,000 张。重复相同命令即可续做，已经保存的图像会跳过。`GPU_IDS` 可改成新服务器实际分配的设备，`PNG_WORKERS` 默认每卡 4 个写图线程。

数据包包含已完成图像、逐图元数据、完整 prompt、编号与 seed、生成配置、模型 SHA256 参考及汇总清单；不包含权重、日志、锁或停止标记。迁移后第一次会核对权重文件内容，并检查已保存图像；不会因重新下载后的文件修改时间不同而拒绝恢复。请勿重新抽取 prompt 或修改生成配置。

在原服务器打包当前进度（生成可以同时继续；此包只取启动打包时已完成的样本）：

```bash
conda activate linear-pid
python grafting/tools/export_pixart_dataset.py \
  --dataset-root "$HOME/dataset/PixArt-Sigma-Grafting-12K" \
  --model-dir "$HOME/huggingface_ckpts/PixArt-Sigma-XL-2-2K-MS" \
  --archive "$HOME/dataset/exports/PixArt-Sigma-Grafting-12K-partial.tar" \
  --workers 4
```

同时生成 `.tar.sha256` 和 `.tar.json`，后者记录本次打包的准确张数。PNG 本身已压缩，因此使用未再次压缩的 tar。

## 环境和模型

```bash
conda activate linear-pid
pip install -r grafting/tools/requirements-pixart.txt

export PIXART_WEIGHTS="$HOME/huggingface_ckpts/PixArt-Sigma-XL-2-2K-MS"
python grafting/tools/download_pixart_models.py \
  --weights-root "$HOME/huggingface_ckpts" --endpoint https://hf-mirror.com --workers 2
```

下载原始 `PixArt-alpha/PixArt-Sigma-XL-2-2K-MS` Transformer，以及官方 `pixart_sigma_sdxlvae_T5_diffusers` 的 T5、tokenizer、SDXL VAE 和 scheduler，合计约 21.8 GB。Transformer 固定到官方 commit `cd274f4c8b01de0dcc87b65887dd357d473bf516`。完整本地目录可离线加载，不是蒸馏权重。

下载器默认使用 `hf-mirror.com`，显式绕过代理，不修改当前终端代理设置。可通过 `--endpoint https://huggingface.co` 使用源站。大文件写入 `.incomplete`，重复命令可以续传；提供 SHA256 的文件会核验。也可选择 ModelScope 组件镜像：

```bash
python grafting/tools/download_pixart_models.py \
  --weights-root "$HOME/huggingface_ckpts" \
  --component-source modelscope --components-only --workers 2
```

这个选项仅下载 T5/VAE 等组件，仍需 2K Transformer。模型目录中出现 `components_download.json` 不代表完整模型就绪。

## 准备 prompt

支持 TXT（一行一个 prompt）、JSONL 和 JSON 数组。例如 `prompts.jsonl`：

```json
{"id":"example-1","prompt":"A small child playing with a golden retriever in a sunny park, realistic photography."}
{"id":"example-2","caption":"A red bicycle beside a white brick wall, detailed wheel spokes."}
```

TXT/JSONL 按文件顺序选取前 12,000 条；`--num-samples 0` 使用全部。脚本不自动将作者没有指明来源的 30K 评估 prompt 当作已知数据。可使用自行准备、独立于最终评估集合的 prompt。

## 批量生成

```bash
conda activate linear-pid
export PIXART_WEIGHTS="$HOME/huggingface_ckpts/PixArt-Sigma-XL-2-2K-MS"

GPU_IDS=0,1,2,3 CPU_THREADS=1 \
bash grafting/tools/generate_pixart_dataset.sh \
  --model-dir "$PIXART_WEIGHTS" \
  --prompts data/pixart-prompts.jsonl \
  --output-root outputs/pixart-sigma-synthetic-12k \
  --num-samples 12000 --stage1-size 8000 \
  --batch-size 1 --workers 4 --threads 1 \
  --height 2048 --width 2048 --steps 20 --cfg 4.5 --seed 42
```

每张卡独立加载一套 pipeline；`--batch-size` 为**每卡生成 batch**，不是训练 batch。默认所有模型留在 GPU，不使用 CPU offload。OOM 会报错，不自动改变 batch 或分辨率。`--workers` 控制每卡 PNG 写入线程，`--threads` 控制每进程 CPU 线程。

原样重复命令即可断点续生成，换卡数或 batch 也可。每个样本 seed 固定为 `seed + index`；不同 batch/硬件仍可能带来浮点计算差异。图片和元数据原子写入，缺失、损坏或没有完成元数据的图片会重新生成。输出目录加锁，避免两个生成任务同时写入。

node01 可额外指定 `--max-seconds 1500`，每轮在完成一个 batch 后退出；超时可能超过设置值一个 batch 的耗时。**该命令不会自动重启**，再次运行相同命令续生成。SIGTERM/CTRL+C 后保存当前 batch 再退出；强制 kill 后最多重做未保存的 batch。

默认 VAE slicing 开启、tiling 关闭。需要分块解码时显式添加 `--vae-tiling`，记录会保存此选项。改变 prompt、seed、采样参数或模型时使用新的输出目录。

## 输出

```text
outputs/pixart-sigma-synthetic-12k/
  images/00000000.png
  metadata/00000000.json
  records.json
  generation_config.json
  manifest.jsonl
  stage1_8k.jsonl
  status.json
```

`manifest.jsonl` 列出已完整生成的图文对；`stage1_8k.jsonl` 固定取最初 8,000 个样本中已完成的部分。完成 12K 后，前者用于第二阶段，后者用于第一阶段。路径相对输出根目录。每张图片元数据包含原始 prompt、seed、分辨率、配置指纹、batch 耗时和显存峰值。中断后，以每图元数据为完成依据，下一次启动会更新汇总 manifest。

只检查数据与分工、不加载 GPU：在完整下载模型后，给生成命令添加 `--plan-only`。

本地逻辑测试：

```bash
conda activate linear-pid
python -m unittest discover -s tests -p test_pixart_dataset.py
```

## node01 后台自动续生成

先从 COCO 2014 caption 注释中构建固定 30K prompt 池，再抽 12K；一张源图只选一条 caption，去除相同文本。这里的 COCO 来源是复现实验选择，论文没有公开作者具体的 prompt 清单。

```bash
conda activate linear-pid
export PIXART_DATA="$HOME/dataset/PixArt-Sigma-Grafting-12K"
python grafting/tools/prepare_pixart_prompts.py \
  --annotations "$PIXART_DATA/source/captions_val2014.json" \
  --output-root "$PIXART_DATA" --seed 42

GPU_IDS=0,1,2,3 CPU_THREADS=1 \
python grafting/tools/generate_pixart_repeat.py \
  --output-root "$PIXART_DATA" --cycle-seconds 1200 -- \
  --model-dir "$HOME/huggingface_ckpts/PixArt-Sigma-XL-2-2K-MS" \
  --prompts "$PIXART_DATA/prompts_12k.jsonl" \
  --num-samples 12000 --stage1-size 8000 \
  --height 2048 --width 2048 --steps 20 --cfg 4.5 --seed 42 \
  --batch-size 2 --workers 4 --threads 1
```

此命令立即返回，后台轮换生成进程和控制进程 PID，避免控制进程一直存活。每轮约 20 分钟，图片完成后保存，再续做剩余样本；全部完成即停止。CUDA 非法内存访问时会用新进程恢复，默认最多连续重试 3 次，可通过控制器参数 `--max-cuda-retries` 修改；这不表示底层 CUDA 错误已经修复。连续三轮没有进展、OOM 或其他 worker 错误时停止并保留日志。手动停止和 Slurm 接管优先于异常重试。

```bash
# 状态和日志
python grafting/tools/generate_pixart_repeat.py --output-root "$PIXART_DATA" --status
tail -f "$PIXART_DATA/repeat/latest-generation.log"

# 手动停止；当前 batch 保存后退出，并禁止自动重启
python grafting/tools/generate_pixart_repeat.py --output-root "$PIXART_DATA" --stop
```

再次运行启动命令恢复。使用手动模式 `generate_pixart_dataset.sh` 时，仍按前文说明手动重复命令。

## Slurm：2 节点 8 卡续生成

支持边排队边在 node01 生成。从项目根目录提交一次即可；已有排队任务时不要重复提交：

```bash
conda activate linear-pid
export PIXART_DATA="$HOME/dataset/PixArt-Sigma-Grafting-12K"
mkdir -p outputs/slurm
sbatch grafting/tools/generate_pixart_8gpu.slurm
```

默认 `students` account、`gpujl` partition、2 节点各 4 卡、每卡 batch 2，最长 6 天。每节点启动一个 Slurm task（16 CPU），由它分别启动四个单 GPU worker，避免逐 task 的 GPU 绑定与设备权限不一致。可用 `sbatch --account=... --partition=...` 覆盖集群参数；数据和模型目录分别由 `PIXART_DATA_ROOT`、`PIXART_WEIGHTS` 覆盖。

排队期间可以照前文继续运行 node01 的后台控制器。job 实际启动后先检查每张分配 GPU 的设备权限、独立 UUID、BF16 矩阵乘法及 SDPA，并检查共享模型和数据路径；通过后才通知 node01 停止。检查结果保存在 `slurm/preflight/JOBID/node_0.json` 和 `node_1.json`，失败时 node01 继续生成。检查通过后，协调进程写入接管标记，通知 node01 完成当前 batch 并停止自动重启；等待控制器明确确认停止、本地控制器锁和生成输出锁释放后，才启动八卡 worker。接管期间会阻止 node01 控制器重新启动。自动交接超时会报错，不会强行与原生成任务同时写数据。

每卡运行一个独立 worker，按节点编号和本地 GPU 编号计算全局 rank（0–7）划分样本，不进行 DDP 通信。单一协调进程检查原有配置、锁住输出目录，并汇总清单。换成 8 卡后，已完成图像会被跳过，样本 ID 和 seed 保持不变。Slurm 任务连续生成，不进行 node01 的定时进程轮换。保留 CUDA 同步检查配置；信号到达后在当前 batch 保存完毕退出。

查看 `slurm/state.json` 获取进度，查看 `outputs/slurm/pixart-sigma-12k-generate-JOBID.out` 获取日志。汇总 manifest 每 5 分钟及退出时更新；逐图元数据始终是断点依据。重新提交同一脚本即可继续。

已完成的本机检查：5 项逻辑测试；双卡小模型生成及换卡、换 batch 恢复；完整官方模型离线加载和 2048×2048、20 步、CFG 4.5 的单图生成。A40、BF16、每卡 batch 1 实测该图生成约 31.2 秒（不含加载），峰值约 19.8 GiB；这是一张图的检查结果，不作为全数据集吞吐保证。临时测试输出已清理。
