# DeCo-QDM

基于官方 DeCo-XXL/16 文生图权重，将16层图像 Attention 替换为 **Local Softmax + Halo2 + 四象限 QDM**。原生训练为512×512；4层文本细化、FFN、调制层和3层像素解码模块冻结。

局部分支保留 DeCo 图像 RoPE：每个8×8 query区域读取周围最多12×12图像 K/V，以及全部128个文本 K/V，在同一次 Softmax 中归一化。QDM 分支共享原始图像 Q/K/V，K/V 经全图5×5 DWConv；四象限分别求解 `S=B(I+A)^(-1)`，每个 query 读取所属象限。远程分支不使用 RoPE、扫描、遗忘衰减或文本初始记忆。

## 1. 环境与资源下载

环境独立于其他实验。已有 `linear-pid` 环境时：

```bash
bash deco/scripts/setup_deco_env.sh
conda activate deco-qdm
export DECO_WEIGHTS="$HOME/huggingface_ckpts/DeCo"
export DECO_DATA="$HOME/dataset/BLIP-3o/BLIP3o-60k"

bash deco/scripts/download_deco_models.sh --weights-root "$DECO_WEIGHTS" --threads 4
```

下载官方 `zehongma/DeCo` 的 `t2i_DeCo.ckpt`（约4.56GB，校验官方 SHA256），以及 ModelScope `Qwen/Qwen3-1.7B` 的权重、配置和 tokenizer（约4.08GB），保存到 `$DECO_WEIGHTS`。DeCo 经 hf-mirror 下载，所有下载子进程关闭代理；不会改变调用终端的代理配置。重复命令可断点续传。可选 `--connections 4` 使用 aria2 加速 DeCo 大文件。

不需要 VAE、DINO、Gemma 或其他 DeCo ImageNet 权重。若没有现成环境，先创建 Python3.12 环境，按 `deco/requirements.txt` 安装依赖。

## 2. 训练准备

BLIP3o-60k 的已下载 `.tar` 内应包含同名图片和 `.txt` caption。工具直接读取 tar，不全量解压、不复制图片。先完成全部分片下载，再执行：

```bash
conda activate deco-qdm
export DECO_WEIGHTS="$HOME/huggingface_ckpts/DeCo"
export DECO_DATA="$HOME/dataset/BLIP-3o/BLIP3o-60k"
export DECO_CACHE="$DECO_DATA/deco_qdm_cache"

bash deco/scripts/prepare_deco_data.sh \
  --dataset-root "$DECO_DATA" --index-root "$DECO_CACHE/index" \
  --workers 4 --threads 1

GPU_IDS=0,1,2,3 CPU_THREADS=1 \
bash deco/scripts/cache_deco_text.sh \
  --index-root "$DECO_CACHE/index" --text-cache-root "$DECO_CACHE/text" \
  --model-path "$DECO_WEIGHTS/Qwen3-1.7B" --batch-size 8 --threads 1
```

索引检查完整解码及 caption，记录有效数量与剔除原因；准备进度按分片保存。图片采用官方 Resize、中心裁剪与 `[-1,1]` 归一化。

文本通过原始 Qwen3 编码流程缓存：纯 prompt、右侧 padding、固定128 token、2048维、BF16；包括空提示词与10条固定验证 prompt。60K缓存约30GiB。索引和缓存均支持重复执行续做；训练不加载文本编码器。下载尚未完成时可提前索引，正式训练默认拒绝包含未完成分片的索引；下载完成后重新运行准备命令。

## 3. 三阶段训练

| 阶段 | 更新步数 | 可训练参数 | 目标 | 保存与验证 |
|---|---:|---|---|---|
| A1 | 500 | 新增QDM | 16层逐层投影后 Attention 相对MSE | 每50步保存，结束验证 |
| A2 | 1000 | 新增QDM | 整网像素速度预测MSE | 每100步保存，结束验证 |
| B | 10000 | QDM、Softmax Gate、Q/K/V/O LoRA | 整网像素速度预测MSE | 每2000步保存并验证 |

全部阶段使用完整有效训练集。默认每卡bs4、累积1；4卡有效batch16。教师与学生共享一套冻结基础权重，顺序执行；EMA关闭，无CPU offload。A1/A2局部输出门固定0.99；B开启rank64、alpha64的独立Q/K/V适配器及输出适配器。FM只记录，不参与优化。

```bash
conda activate deco-qdm
GPU_IDS=0,1,2,3 CPU_THREADS=1 \
bash deco/scripts/train_deco_qdm.sh \
  --stage all --resume auto \
  --batch-size 4 --grad-accum 1 --workers 4 --threads 1 \
  --weights-root "$DECO_WEIGHTS" \
  --index-root "$DECO_CACHE/index" --text-cache-root "$DECO_CACHE/text" \
  --output-root outputs/deco-qdm
```

`--stage a1|a2|b|all` 选择阶段；`all`自动加载上一阶段最终权重。单独启动A2/B时可用 `--init-from /path/to/previous-stage-checkpoint`；同阶段使用相同参数、相同输出目录和 `--resume auto`。恢复学生、优化器、scheduler、随机状态及已消费数据位置。默认不允许静默改变batch、结构或训练预算。

输出为 `outputs/deco-qdm/{a1,a2,b}`；checkpoint仅保存新增参数及训练状态，推理还需原始DeCo权重。完成标记发布后才视为可恢复，未完成目录会被忽略。最近3个恢复状态、验证节点及阶段最终权重保留。

### 选择性保留 Full Attention

`--full-attention-layers 1,2,3,4` 按**从0开始**的图像层索引保留这四层原始 Attention，
其余12层继续替换为 Local Softmax + Halo2 + QDM。保留层在三个阶段都冻结，B阶段也不添加LoRA。
省略该参数仍替换全部16层。布局写入checkpoint，阶段衔接和续训严格检查，独立推理自动恢复。

以下入口从原始DeCo开始新实验，四卡、每卡bs32、累积1；
A1/A2/B分别500/1000/10000步，各阶段每500步保存并生成10张验证图：

```bash
conda activate deco-qdm
GPU_IDS=0,1,2,3 CPU_THREADS=1 \
DECO_QDM_PARTIAL_RUN=outputs/deco-qdm-full-1-2-3-4-node01-bs32 \
bash deco/scripts/train_deco_qdm_partial_full_node01.sh
```

入口默认后台每20分钟保存并更新训练和控制器进程，依次完成三个阶段；重复同一命令从该实验目录恢复。
可以用 `--workers` 等参数覆盖入口默认值。新实验请使用新输出目录，不加载先前学生checkpoint。

SwanLab默认离线：每阶段目录的 `swanlog/` 下记录误差、KD/FM、学习率、梯度、实际样本数、吞吐、显存及生成图。可用 `--swanlab-mode online`。固定验证使用25步、CFG4、shift3、AdamLM order2；结果放在各阶段 `validation/`，包含原图、左右拼图和中心同坐标裁剪。原始参考图只生成一次，已完成验证图片可续用。

### node01 可选后台自动续训

```bash
conda activate deco-qdm
GPU_IDS=0,1,2,3 CPU_THREADS=1 \
bash deco/scripts/train_deco_qdm_repeat.sh \
  --stage all --resume auto --cycle-seconds 1200 \
  --batch-size 4 --grad-accum 1 --workers 4 --threads 1 \
  --weights-root "$DECO_WEIGHTS" \
  --index-root "$DECO_CACHE/index" --text-cache-root "$DECO_CACHE/text" \
  --output-root outputs/deco-qdm

# 查看状态 / 请求保存停止
bash deco/scripts/train_deco_qdm_repeat.sh --output-root outputs/deco-qdm --status
bash deco/scripts/train_deco_qdm_repeat.sh --output-root outputs/deco-qdm --stop

tail -f outputs/deco-qdm/repeat/latest-training.log
```

控制器在更新边界保存，每轮约20分钟重新启动训练和控制器；验证可在图片边界保存进度。完成B后退出；手动停止、OOM、数值错误或连续无进展时停止重试。

## 4. 通过 prompt 独立推理

```bash
conda activate deco-qdm
GPU_IDS=0 CPU_THREADS=1 \
bash deco/scripts/sample_deco_qdm.sh \
  --weights-root "$DECO_WEIGHTS" \
  --checkpoint /path/to/complete/student/checkpoint \
  --prompt 'A child walking beside a golden retriever on a wide beach, natural daylight.' \
  --height 512 --width 512 --seed 42 --steps 25 --cfg 4 --shift 3 \
  --compare --output outputs/deco-qdm/inference/child-and-dog.png
```

省略 `--checkpoint` 使用原始DeCo；`--compare`另外生成同输入的原始参考、左右拼图及同坐标裁剪。文本编码器仅在编码prompt时加载，编码完成释放后再加载DiT。学生结构从checkpoint恢复，默认512；推理可指定16倍数的高宽。耗时统计仅包含DiT采样，模型加载、文本编码与PNG保存单独处理。

## 5. 多卡批量文生图与断点续生成

准备一个 JSON prompt 列表，每条包含 `id`、`category`、`prompt`、`seed`。
`id` 唯一，`id/category` 使用英文字母、数字、下划线或短横线，例如：

```json
[{"id":"people_01","category":"people","prompt":"A portrait photograph of a violinist.","seed":42}]
```

```bash
conda activate deco-qdm
GPU_IDS=0,1,2,3 CPU_THREADS=1 \
bash deco/scripts/sample_deco_qdm_batch.sh \
  --weights-root "$DECO_WEIGHTS" \
  --checkpoint outputs/deco-qdm/b/checkpoints/SELECTED_CHECKPOINT \
  --prompts prompts.json --output-root outputs/deco-qdm/prompt-gallery \
  --height 512 --width 512 --steps 25 --cfg 4 --shift 3 \
  --threads 1 --text-batch-size 8 --cycle-seconds 1200

# 查看实时完成数量，或请求在当前图片生成完毕后停止
bash deco/scripts/sample_deco_qdm_batch.sh --output-root outputs/deco-qdm/prompt-gallery --status
bash deco/scripts/sample_deco_qdm_batch.sh --output-root outputs/deco-qdm/prompt-gallery --stop
tail -f outputs/deco-qdm/prompt-gallery/repeat/latest-training.log
```

入口会后台启动，每张卡独立加载同一个学生模型，按 prompt 分配任务；图片保存在
`generation/validation/<category>/`。每张图附带 prompt、seed、采样设置、耗时和完成标记。
直接编码给定 prompt，不读取训练图片或图像条件。文本编码器使用后释放，再加载 DiT，
不进行 CPU offload。文本结果缓存仅属于本次生成任务。

重复同一条启动命令会跳过完整图片；损坏或缺少完成标记的图片重新生成。
同一目录拒绝混用不同权重、prompt 或采样参数。模型 checkpoint 标记为保留。
默认每20分钟在图片边界结束本轮，重新创建生成进程和控制器；完成全部图片后退出。
手动停止、OOM、CUDA异常或连续无进展会停止重试，原因保留在 `repeat/` 日志中。

## 实现来源

最小官方DeCo模块位于 `deco/deco_qdm/vendor/`，来源commit记录在其 `PROVENANCE.md`；原目录保持原样。QDM与Halo的计算沿用本项目PiD实验，并针对DeCo的图像query＋图文K/V接口适配。代码不依赖其他PiD工作树。

### 双节点八卡训练

各阶段独立设置保存、验证的 step 周期；验证仅使用学生生成固定 10 张图像。以下配置每卡 batch 32、累积 2，有效 batch 512：

```bash
conda activate deco-qdm
mkdir -p outputs/deco-qdm/slurm
sbatch deco/scripts/train_deco_qdm_8gpu.slurm
```

脚本依次运行 A1（1000 步，每 500 步保存验证）、A2（2000 步，每 1000 步保存验证）、B（20000 步，每 2000 步保存验证），默认输出 `outputs/deco-qdm`，`--resume auto` 从同阶段最新完整 checkpoint 恢复。可通过 `DECO_RUN` 覆盖输出目录。
