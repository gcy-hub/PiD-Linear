# A40 单图生成测试

本实验的滑动窗口大小为 11×11 tokens。

模型：本地 FLUX.2-klein-base-4B、flux-3-image-klein-base。修改版为 Sliding + 跨层换向 KDA，未微调。配置为 1024×1024、50 步、CFG 4、seed 0；提示词：`A red double-decker bus on a rainy London street, reflections on wet asphalt`。

使用 node01 的两张独立 NVIDIA A40，FLUX2 在 GPU1、FLUX3 在 GPU2。每组原版和修改版顺序执行，Torch 2.10.0+cu128、BF16 运算。修改版保留 FP32 master 权重，前向转换为 BF16。时间为单次测量。

| 模型 | 版本 | 采样秒 | VAE 秒 | 合计秒 | 峰值 allocated GiB | 首次 forward 秒 |
|---|---|---:|---:|---:|---:|---:|
| FLUX2 | 原始 | 46.67 | 0.90 | 47.57 | 7.82 | 0.49 |
| FLUX2 | Sliding + KDA | 69.58 | 0.26 | 69.84 | 13.45 | 96.59 |
| FLUX3 | 原始 | 55.03 | 0.40 | 55.44 | 11.07 | 0.56 |
| FLUX3 | Sliding + KDA | 78.53 | 0.25 | 78.78 | 18.28 | 108.68 |

合计为 CUDA 同步的采样与解码时间；不含文本编码、权重加载、首次 forward 和保存 PNG。首次 forward 包含按需内核编译。峰值为采样阶段的 PyTorch 已分配显存，不是整进程显存峰值。

本次修改版总时间分别增加 46.8% 和 42.1%。完整 KDA 分支均执行，零输出初始化没有省略计算。

![原版与未微调学生](../outputs/kda-speed/comparison.png)

原始模型生成了连贯街景。两张未微调学生图均出现公交车重复和空间结构错乱。此时 KDA 输出投影为零，图像表现对应初始 Sliding 学生。正式微调后的质量与性能需要另行测量。

## 检查结果

- 四方向矩形网格扫描与逆变换通过。
- 真实 FLA KDA 的四方向前向、反向及各参数梯度有限。
- FLUX2 双流和单流 attention 接入、反向及 teacher 路径通过。
- FLUX3 checkpoint 重算、特征 KD、原始 teacher 输出一致性与参数隔离通过。
- 小型 block 的 Prodigy 更新和 delta 参数恢复通过。
- FLUX2、FLUX3 的原始版和修改版均完成 50 步推理，输出和解码结果有限。

未启动正式训练；未验证完整模型的训练显存、收敛或多卡 DDP 恢复。

原始测量记录位于 `outputs/kda-speed/{flux2,flux3}/{original,kda}.json`；四张原始尺寸 PNG 位于同目录。检查日志：`outputs/kda-check/check.log`。训练和复现命令见 [README](README.md)。
