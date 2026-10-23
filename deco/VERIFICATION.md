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
