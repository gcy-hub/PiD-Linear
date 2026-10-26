# DeCo-QDM 实现验证（2026-10-23）

环境：Python3.12、PyTorch2.10.0+cu128、Transformers4.57.1；GPU为NVIDIA A40。

- 37项CPU测试通过：QDM输出/梯度直接参考、Halo可见范围与边界、教师整网复现、FP32时间步、冻结参数、零增量LoRA、双rank梯度等价、数据索引、缓存契约、完整恢复和后台控制器。
- 官方DeCo T2I与Qwen3权重已实际下载，校验通过。DeCo SHA256为 `433f55684b19c446d9fad4591f840fcdf9770a668c383ea910da86362651492f`。
- 真实输入使用BLIP tar内16条图文，图片预处理512×512，Qwen实际编码后缓存；未复制源图片。
- 四卡每卡bs4、累积1，有效batch16；三阶段各2次更新。A1第1步保存退出，再恢复完成第2步，随后串行进入A2和B。

| 阶段 | 可训练参数 | rank0峰值显存 | 保存、恢复／阶段加载 | 多卡生成验证 |
|---|---:|---:|---|---|
| A1 | 42,738,688 | 3.56GiB | 第1步完整恢复至第2步 | 10条固定prompt |
| A2 | 42,738,688 | 5.24GiB | 从A1加载并保存 | 10条固定prompt |
| B | 62,203,264 | 5.52GiB | 从A2加载，启用LoRA并保存 | 10条固定prompt |

独立进程重新加载B阶段测试checkpoint，直接输入prompt完成512图像生成及原始DeCo左右对比。采样采用25步、CFG4、shift3、AdamLM order2；训练过程中不驻留Qwen、不使用CPU offload或EMA。

结果表明训练、恢复和生成链路可运行；短测只有2步，用于验证代码，不评价微调质量或稳定吞吐。临时训练checkpoint已删除，日志、验证图和结构化结果保留在 `outputs/deco-qdm/verification/` 与 `outputs/deco-qdm/setup/`，这些输出不会提交。

正式训练没有启动。正式三阶段预算与全量数据准备命令见 [README](README.md)。

## node01 bs32 真实训练中的 CUDA 求解异常

2026-10-23，第一轮在 A1 第 200 步完整保存并成功换进程续训。第二轮第 292 步之后，rank0 在反向中触发 CUDA device assert；日志包含 `DynamicCast.h` 断言以及 `magma_spotrs_batched` 报错。其他 rank 的 DataLoader SIGTERM 是失败后的清理过程。控制器将其识别为非正常训练异常并停止，未反复重试。

最小修改：将 `torch.cholesky_solve(B.T, L)` 改为两次 `torch.linalg.solve_triangular`，分别求 `L z=B.T`、`L.T S.T=z`。继续使用 FP32 统计、Cholesky 分解及同一 SPD 方程；不改变模型结构、batch、精度和损失。

- 40 项本地测试通过，包含直接参考的输出/梯度对照及禁止回到 `cholesky_solve` 路径的回归检查。
- A40 上生产状态尺寸（24 heads × 64维）的 CUDA 输出及 Q/K/V/beta 梯度与直接求解一致；最大输出绝对差约 `1.10e-5`。
- 单卡 bs32、32×32 token 网格，连续 500 次求解与反传通过，检查全部输入梯度有限。
- 原路径的简化 30 次压力测试未复现断言，因此不能仅凭这些测试确认底层错误的完整触发条件。修复避开日志中报错的求解路径，完整模型继续运行验证。

后台重启命令沿用 [README](README.md) 的续训入口，当前实验参数保存在 `outputs/deco-qdm-node01-bs32/repeat/request.json`；同目录从第 200 步完整 checkpoint 恢复，不重置学生、优化器或 scheduler。
