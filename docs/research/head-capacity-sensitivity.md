# C0：Head Capacity Sensitivity

本实验用于判断一个具体研究动机：已经训练的 KDA 是否存在不同 attention head 对 key/state 容量的不同需求，以及相同总容量下，按 head 分配容量是否值得继续研究。它是独立的特征校准诊断，不修改正式训练架构，不进行完整模型再训练，也不据此宣称图像质量提升。

固定 trained10 student 的 step 3314、MMDiT 第 8 层（从 0 开始）、24 个实际 head，value 维度固定为 64。PiT 和其余网络参数不变。脚本用原 student 的 forward pre-hook 获取第 8 层 attention 的归一化输入，再停止本次 forward；因此后续 PiT 不参与这项局部测量。GPU 同时只放一个完整 student 和原 PiD 的这一层 attention。

## 校准和对照

每个 head 的 key 维度取 32、48、64、80、96。先按 checkpoint 的原 64 维 RoPE 处理 Q/K，再学习独立的逐 head Q/K 线性映射，之后进行 L2 归一化，KDA scale 为 `K**-0.5`。V、beta、原 QKV、local mixing、输出门及其他网络参数冻结。小于 64 的映射用确定性的随机正交基初始化；大于 64 用单位基加幅度 0.05 的随机基初始化。所有容量都接受相同数量的 optimizer step、相同顺序的完整校准 case 和相同 loss/lr；相同 step 不代表相同 FLOP 或参数量。只截断维度后的退化不能解释为容量敏感度。

原 forget gate 的非正 log-decay 经 K 投影权重的平方、逐行归一化后进行正系数混合，保证新 decay 仍非正。这是一种对角 gate 近似：一般的旋转并不能严格保持原对角遗忘算子。扩展到 80/96 的线性基仍由 64 维输入生成，线性秩不会超过 64，但额外状态坐标和不同衰减可能增加循环动力学自由度。因此曲线同时受到投影、gate 近似、优化和容量的影响，不能当作原生可变维度 KDA 的充分证据。

主校准目标是**trained-KDA 在原 64 维配置下的逐 head 输出特征**，即保留已经学到的局部 recurrence 行为。比较发生在输出门和输出投影之前。loss 为 image token 上的逐 head RMS 归一化 MSE，以免只拟合幅度；评估同时保存原始 relative MSE、归一化 MSE 和 cosine。text token 仍参与完整 recurrence/softmax，但 padding token 被打包移除，不计入 image loss。eval 模式下无 dropout，所有 KDA 调用 `initial_state=None`，不同 case 之间不保留状态。

诊断 RMS 使用 epsilon `1e-8`，原模型的输出 `Gates.normalize` 使用 `1e-5` 和额外学习参数。这里不执行完整输出门、projection 或后续 residual；局部归一化保留误差只是近似指标，不能替代它们的端到端验证。

原 PiD softmax attention 在同一份 student hidden 上的输出另存为 `softmax` 参考指标，不是完整 teacher 网络在自己的 hidden 上的输出。这是**移除 padding 的原模块控制**，而 official 默认 softmax 会保留 padding token，A0 的官方基线保留这一默认行为，二者不能混称同一基线。原 PiD 与 student 的 value 投影和 head 坐标可能不同，未验证 head 一一对应，冻结 V 的 Q/K adapter 不能消除这种差异。因此 softmax 的逐 head MSE/cosine 不能当作 teacher 质量证据，默认不用于选择容量。显式 `--target softmax` 是另一次带坐标混淆的校准实验，须用新 output 单独标注。`kda` 表示局部自蒸馏保留程度，也不直接证明图像质量。

`identity_64` 是完全不改变 Q/K/g 的冻结控制；`uniform_64` 是与其他容量采用相同校准预算的可学习 64 维控制。两者必须分开报告。KDA 保留目标下，64 维单位映射可以达到零误差，原预算 `24*64=1536` 的 unrestricted 最优分配通常就是全部 64，这是合理上界，不应要求重分配超越零误差。

主要等预算比较在 **75% 容量** `24*48=1152` 下进行：`uniform_48` 对比 training-only 精确 knapsack 得到的 `guided_48_budget`，再对比 `random_48_budget`。random 将 guided 的维度列表随机分配到 head，保持完全相同的维度直方图、总容量和 adapter 参数数。guided/random 复用各维度的已校准 head 参数，不另加训练 step。脚本另报告原容量的 `unrestricted_64_budget`。选择已经尝试多个维度，仍有模型选择优势；这里只保留一个随机分配种子，不能代替多种子显著性分析。

## 固定输入和划分

默认 train prompt 为 0/1/2，held-out 为 3/4；每个 prompt 都覆盖 2048/4096 和 t=100/500/900。按 prompt 划分后才展开 resolution/time，禁止同一 prompt 的 token、时间或分辨率进入两边。

建议传入已有 original 生成图目录：`x_t=(1-t/1000)*x0+t/1000*noise`，noise 使用 asset 的原 seed，caption、latent 和原 seed 全部复用。脚本校验图像 JSON 的 seed 与 asset SHA256。x0 是原 PiD 生成的 pseudo-reference，**不是数据集 GT，也不是求解器真实轨迹**。不传 `--reference-dir` 时使用相同 seed 的纯 Gaussian pixel 输入，明确标记为 off-trajectory 人工诊断。两种协议都不能直接证明实际训练流形或生成图质量上的效果。

## 运行

依赖现有 `linear-pid` 环境、已固定的 FLA、本地 checkpoint/assets 和 matplotlib。第一次用新 K/V 形状时，Triton 编译可能需要数分钟。下面是完整工作量的直接运行命令；路径可通过变量改成已有文件所在目录：

```bash
conda activate linear-pid
python -m pip install 'matplotlib<3.11' 'contourpy<1.4' 'numpy==1.26.4'
cd PiD-Linear
CHECKPOINT=outputs/linear-pid-kda10-direct/node01/kda_0-1-2-4-5-6-8-9-10-12/checkpoints/step_000003314_batch_000003314
TEACHER=weights/PiD/checkpoints/PiD_v1pt5_res2kto4k_sr4x_official_flux_undistilled/model_ema_bf16.pth
COMPARISON=outputs/inference_comparison/unseen-prompts-step3314
GPU_IDS=3 WORKERS=1 CPU_THREADS=1 bash scripts/diagnose_head_capacity.sh \
  --checkpoint "$CHECKPOINT" --teacher "$TEACHER" \
  --assets "$COMPARISON/assets" --reference-dir "$COMPARISON/original" \
  --output outputs/research/head-capacity-sensitivity-step3314 \
  --layer 8 --key-dims 32 48 64 80 96 \
  --train-prompts 0 1 2 --heldout-prompts 3 4 \
  --resolutions 2048 4096 --timesteps 100 500 900 \
  --fit-steps 36 --lr 0.002 --target kda \
  --latency-repeats 5 --latency-warmup 2 --swanlab-mode offline
```

`GPU_IDS` 应选已有 allocation 中可用的 GPU；不设置时保留终端现有的 `CUDA_VISIBLE_DEVICES`。多个 GPU 可用 `GPU_IDS=2,3 WORKERS=2` 启动 torchrun：capture case 和 uniform 配置按 rank 分片，优化各 head 使用向量化 CUDA 运算；CPU 默认单线程。脚本不申请 Slurm 资源。直接 `python scripts/diagnose_head_capacity.py ...` 也可，需在仓库根目录设置 `PYTHONPATH=$PWD`。

重跑完全相同命令即恢复。每个 capture case、每个 adapter optimizer step 都写临时文件再原子替换，只有匹配完成标记、大小和 metadata fingerprint 的文件才可复用。checkpoint、asset、参考图、代码、训练预算或协议变更需要新 `--output`，不会静默混用。`--phase capture/fit/evaluate/aggregate` 可以单独执行阶段；partial 聚合明确列出缺失项，只有全部 case/configuration 对完成才写总 `complete.json`。node01 被杀后直接重跑即可，不需要重新申请资源。

每次进程有独立 SwanLab run，默认 offline。原始特征、adapter、运行日志、PNG 和结果均放在 output，仓库 `.gitignore` 已排除这些文件和本地 tests，不提交权重或机器环境。

## 输出和解释

`manifest.json` 保存参数、输入来源及 fingerprint；`records/*.pt` 保存原 recurrence 输入和两个区分明确的目标；`adapters/*.pt` 保留每步优化状态；`training_scores.json` 和 `allocation.json` 保存无 held-out 泄漏的选择依据；`evaluation/<config>/*.json` 每 case 保存所有 head 的误差、实测 GPU 延迟和计数；`results.json/csv` 汇总；两个 `head_curves_*.png/pdf` 给出 24 个真实 head 的容量曲线。

计数中的 state 是逻辑 `K*64` 元素数、按 FP32 标出的字节，不是完整 kernel workspace。FLA 对 48/80 等非 2 次幂维度可能内部 padding，因此不能用逻辑容量比例推断真实延迟；脚本同时记录 GPU 峰值 allocated 增量。实际 adapter 参数为 `2*64*sum(K_h)`；另列原生单 stream Q/K 或 QKV projection 的假想参数量，明确排除 gate、local mixing、bias 和 output projection。这些假想数字不是本 probe 的实际模型参数量。

延迟用 CUDA event，在输入已驻留 GPU 时重复测量，包含 Q/K 投影、正 decay 混合和按容量分组的 KDA，排除磁盘 IO、feature capture、后续 output gate/projection 和 PiT。guided 更小的逻辑预算不保证更快，分组调用和内部 padding 都有代价。

优先看 held-out 的逐 head KDA 保留曲线和 1152 容量下的对照，再看训练/held-out 差距。若不同 head 的曲线形态有稳定差异，且训练集选择的分配在 held-out 保留误差低于均匀与随机分配，才支持继续做原生架构实验的动机。这种正结果不自动意味着速度或图像质量提升。单层、2 个 held-out prompt、1 个初始化/随机分配 seed 和 36 step 都很有限；出现负结果、微小差异、未完成项或 guided 延迟上升都应如实报告。

## 第一轮结果

上述完整设置完成 30 个输入条件、五种维度各 36 个校准 step、9 种配置共 270 个 case/configuration 对。主目标为 `kda`。以下误差在 12 个 held-out 条件与 24 个 head 上等权平均，延迟为同一张 NVIDIA A40 的全部条件按分辨率汇总中位数。

| 配置 | 总 Key 容量 | Held-out 归一化 MSE | Held-out 原始相对 MSE | 2K 驻留算子延迟 | 4K 驻留算子延迟 |
|---|---:|---:|---:|---:|---:|
| 冻结 identity 64 | 1536 | 0 | 0 | 6.20 ms | 17.75 ms |
| 均匀 32 | 768 | 0.024461 | 0.094007 | 5.76 ms | 16.52 ms |
| 均匀 48 | 1152 | 0.017637 | 0.167839 | 7.06 ms | 20.39 ms |
| 可学习均匀 64 | 1536 | 0 | 0 | 7.85 ms | 22.87 ms |
| 均匀 80 | 1920 | 0.000154 | 0.007142 | 8.99 ms | 26.25 ms |
| 均匀 96 | 2304 | 0.000229 | 0.029580 | 9.60 ms | 28.20 ms |
| guided：12×32 + 12×64 | 1152 | 0.001335 | 0.023608 | 7.78 ms | 21.31 ms |
| 同直方图随机分配 | 1152 | 0.003833 | 0.034560 | 7.76 ms | 21.32 ms |

原预算 unrestricted 分配选择全部 64 维，保留误差为零。减到 75% 预算时，guided 相比均匀 48 的归一化误差低 92.4%，相比同直方图随机分配低 65.2%；原始相对误差对应低 85.9% 和 31.7%。这支持“不同 head 在当前局部保留任务上的敏感度不同，按 head 分配值得继续研究”的初步动机。

随机的 32/64 混合本身也明显优于均匀 48，因此均匀对照上的大幅改善还混合了“部分 head 保留精确 identity”以及投影/gate 近似的差异。guided 对同直方图随机分配的优势更能反映 head 选择的作用；仍需多随机种子和原生可变维度模型确认。

当前 guided 实现的计时比冻结 64 维更慢，2K/4K 分别约慢 25.5%/20.0%，没有得到实际加速。probe 增加 Q/K 和 decay 映射，原 QKV 投影已经在采集阶段执行，没有替换成低容量原生投影；分组 kernel 也有额外开销。下一步若继续，应验证原生投影和 kernel 实现，再运行完整生成质量与端到端速度对照。上述局部误差和毫秒数不能替代这些验证。
