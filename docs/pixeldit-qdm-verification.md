# PixelDiT-QDM Halo2 验证记录

2026-10-21，`linear-pid` 环境，node01 四张 A40，原生 1K 多长宽比。所有阶段均执行真实图片、Gemma 缓存、完整 14 层 MMDiT 和两层 PiT，累积为 1。

| 阶段 | 每卡 batch | 有效 batch | 测试更新数 | 可训练参数 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 | 4 | 16 | 2 | 37,396,352 | 6.16 GiB |
| A2 | 8 | 32 | 2 | 37,396,352 | 12.99 GiB |
| B | 8 | 32 | 2 | 60,019,208 | 13.47 GiB |

- 官方 checkpoint 的 385 个主干张量严格加载，共 1,302,470,787 参数；6 个辅助 REPA projector 张量单独识别，不进入网络。
- 6 项 CPU 测试通过：教师路径、A1 层输入/梯度、LoRA 零增量与冻结参数、严格加载/阶段 batch、恢复下一次更新、原生空文本替换。CPU 恢复下一次更新逐位一致。
- B 从第 1 步保存状态恢复到第 2 步，优化器 step、scheduler、数据进度及四个 rank 的 CPU/CUDA RNG 状态一致。与不中断运行相比，参数最大绝对差 `2.92e-6`；GPU 内核未配置为确定性执行，不承诺逐位相同。
- 与原始 PixelDiT 源码比较真实 1024² 输入：FP32 最终预测相对 L2 误差 `1.23e-6`。BF16 不逐位一致，随机输入相对差约 `1.11%`，涉及不同设备上的频率预计算与有限精度累积；本实验师生使用同一套本地主干实现。
- 原生 50 步、CFG 2.75、shift 4 的 prompt 推理及固定画廊通过。画廊仅使用文本条件与像素噪声，不使用包内图像 latent。原始 1K 参考图能够正常生成小孩与狗的场景。
- 控制器通过两轮独立进程测试，确认训练子进程和控制器 PID 均更新；状态保存后继续下一轮，而非保持旧 PID 的 `exec`。

以上是代码通过性与恢复验证，两个更新不能用来判断 QDM 的质量恢复。显存仅对应测试输入；耗时受同机其他任务及首次解码影响，不作为稳定吞吐基准。早期 2K/4K 接口探测也不构成这些分辨率下的质量验证。

可以使用以下短程命令检查同样的三个阶段；正式训练命令见 [运行说明](pixeldit-qdm.md)。资源环境变量按运行说明设置。

```bash
conda activate linear-pid
GPU_IDS=0,1,2,3 CPU_THREADS=1 bash scripts/train_pixeldit_qdm.sh \
  --stage all --a1-steps 2 --a2-steps 2 --b-steps 2 \
  --save-steps 1 --validation-steps 0 --teacher-checkpoint "$PIXELDIT_BASE" \
  --workers 4 --threads 1 --output-root outputs/pixeldit-qdm-preflight
```
