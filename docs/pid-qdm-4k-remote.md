# PiD-QDM Halo2：全量数据、4K、单机八卡

代码已合入 `main`。模型从原始 PiD v1.5 FLUX **undistilled** 权重初始化，替换全部 14 层 MMDiT Attention，局部使用 8×8 query + Halo2，远程使用四象限记忆；PiT 保留原实现。不是 PixelDiT-T2I 训练。

## 训练配置

| 阶段 | 更新步数 | 单卡 batch | 梯度累积 | 八卡有效 batch | 保存、验证间隔 |
|---|---:|---:|---:|---:|---:|
| A1 | 1,000 | 8，OOM 后 4 | 8 | 512，降 batch 后 256 | 500 |
| A2 | 2,000 | 8，OOM 后 4 | 8 | 512，降 batch 后 256 | 1,000 |
| B | 20,000 | 8，OOM 后 4 | 8 | 512，降 batch 后 256 | 2,000 |

三个阶段共用完整 MultiAspect-4K-1M **有效训练池**，不使用原 20K/50K 采样池，也不要求各阶段完整遍历一次。训练为 4K 多长宽比，正方形 4096×4096，其余桶长边为 4096；较小源图允许 Lanczos 放大。A1 为逐层 Attention 蒸馏，A2 为最终预测蒸馏，B 加入 Q/K/V/O LoRA。没有 EMA 或 CPU offload。

每次验证只生成以下 **8 张 4K 图**，25 steps、CFG 5、shift 6：

```text
generated_010 generated_029 generated_003 generated_009
generated_019 generated_022 real_011 real_028
```

固定 prompt、真实样本 ID 和 seed 在 `docs/pid-qdm-4k-cases.json`。两张真实图及其相同 URL 的副本始终从训练集合中排除。只准备这 8 张条件，不准备或生成 96 张画廊。Z-Image-Turbo 为 6 个 generated 样本生成原生 1024px latent，供 PiD 生成 4K；2 个 real 条件从对应原图生成。

## 环境

首次准备服务器环境；已有兼容的 `linear-pid` 可直接使用：

```bash
bash scripts/setup_linear_pid_env.sh
conda activate linear-pid
```

入口默认下载原始 PiD、FLUX VAE、Gemma、Z-Image-Turbo。Gemma 需要 Hugging Face 账号已接受模型许可；需要登录时先运行 `hf auth login`。下载使用官方源站，保留当前 shell 的代理设置，不改全局代理。

## 一条命令：补齐数据后训练

已有的 `pid-vdn-50k` 应为解压后的目录，内含 `manifest.jsonl` 和 `images/`。将下面三个资源路径改为当前服务器的位置。`DATASET_ROOT` 指定完整数据集的目标目录，**不要设为 50K 导出目录**。

```bash
conda activate linear-pid
export DATASET_ROOT="$HOME/dataset/MultiAspect-4K-1M"
export SUBSET_ROOT="$HOME/dataset/pid-vdn-50k"
export WEIGHTS_ROOT="$HOME/huggingface_ckpts"

GPU_IDS=0,1,2,3,4,5,6,7 \
bash scripts/train_pid_qdm_4k_8gpu_pipeline.sh \
  --dataset-root "$DATASET_ROOT" \
  --subset-root "$SUBSET_ROOT" \
  --weights-root "$WEIGHTS_ROOT" \
  --output-root outputs/pid-qdm-halo2-4k-8gpu \
  --batch-size 8 --fallback-batch-size 4 --grad-accum 8 \
  --download-workers 16 --index-workers 8 \
  --text-batch-size 8 --workers 4 --threads 1
```

脚本依次执行：

1. 下载／补齐模型和完整的 1,008 个元数据 JSON。
2. 按样本 ID、源 URL 把已有 50K 图片通过相对软链接接入 `datas/`，不复制图片。**训练期间请保留 50K 导出目录。**
3. 多线程下载剩余原始图片，保留临时下载和完成文件，可重跑续传。原图来自源站 URL，失效链接会记录；下载覆盖情况写入 `pipeline/preparation.json`，有效训练数以最终索引为准。若可用图片不到元数据的 95%，则停止并提示重跑下载，不会把原来的 50K 当作全量数据启动训练。
4. 建立全量索引并排除验证样本；图片头和尺寸在此检查，完整解码在训练 loader 中完成。
5. 用八卡生成全量 Gemma 文本缓存，再准备指定 8 张 4K 验证条件。
6. 自动串行训练 A1 → A2 → B，每阶段从上阶段最后的完整权重初始化。同阶段重启恢复优化器、调度器、随机状态和数据进度。

索引与文本缓存保存在 `DATASET_ROOT/linear_pid_index` 和 `DATASET_ROOT/linear_pid_text_cache`，图片、模型、缓存和训练结果均不进入 Git。全量高分辨率图片和文本缓存需要充足磁盘空间。

目标服务器尚未实测 bs8/4K 显存。出现**明确 CUDA OOM**时结束该训练进程，从最近完整 checkpoint 以 bs4 重启，累积仍为 8；之后阶段继续沿用 bs4。bs4 仍 OOM 则停止并报告，不继续降 batch、分辨率或修改算法。非法显存访问、数据错误、SIGKILL 等不会被当作 OOM 自动重试。

脚本适用于**一台服务器 8 张可见 GPU**，不提交 Slurm，也不带 node01 的定时重启。中断后重复相同命令即可：准备阶段跳过已完成结果，训练阶段跳过已完成且验证完毕的阶段。不要在已有训练开始后移动数据或重新改写元数据。

## 后台运行与检查

在上面的命令前加 `nohup`、末尾加 `> pipeline.log 2>&1 &` 可放到后台；也可以在 `tmux` 中直接运行。

```bash
tail -f pipeline.log
cat outputs/pid-qdm-halo2-4k-8gpu/pipeline/training.json
```

各阶段日志在 `outputs/pid-qdm-halo2-4k-8gpu/pipeline/logs/`，训练结果在 `a1/`、`a2/`、`b/`，SwanLab 默认离线。`training.json` 记录实际 batch 和 OOM 回退历史。暂停时可向各阶段写入 STOP，脚本会在更新边界保存退出，不会自动重启：

```bash
mkdir -p outputs/pid-qdm-halo2-4k-8gpu/{a1,a2,b}
touch outputs/pid-qdm-halo2-4k-8gpu/{a1,a2,b}/STOP
```

继续前删除这三个 STOP，再重复原命令。所有路径均可通过上述参数指定。

查看训练命令而不执行下载和训练：

```bash
conda activate linear-pid
bash scripts/train_pid_qdm_4k_8gpu_pipeline.sh \
  --dataset-root "$DATASET_ROOT" --subset-root "$SUBSET_ROOT" \
  --weights-root "$WEIGHTS_ROOT" --dry-run
```
