# 验收记录与待测项

证据保存在本机 `../experiment/attention-framework-refactor/reports/`，原始报告的 evidence_commit 与 evidence_report 标识对应版本。2026-11-11 用户取消后续验收；没有将未执行、OOM 或中止记录计为通过。

| 检查 | 结果与证据 |
|---|---|
| 配置、数据、桶、路径、下载边界；导入、入口和衔接 | 通过，`94b5a611`，`gpu-0330/acceptance.json` |
| FLUX3 原生层 × 四 attention、编码与四缓存模式 | 通过，`94b5a611`，`gpu-0330/acceptance.json: flux3` |
| PiD / FLUX2 原生层 × 四 attention、编码与四缓存模式 | 通过，`2123c3f`，`gpu-0330-repair-2123c3f/acceptance.json` |
| 公共 CUDA/reference、12 模型算法组合、21 小模型策略组合、梯度和 teacher 隔离 | 通过，`34d38d1`，`gpu-final-34d38d1/acceptance.json: common` |
| 四 rank NCCL、累积、分片优化器/RNG 下一步恢复、缺 shard、FP32 更新 | 通过，`d825959`，`gpu-native-resume-d825959/acceptance.json: distributed` |
| PiD sliding_qdm / VDN 完整原生闭环 | `d825959` 最终连续/恢复参数不完全相等；严格确定性 `8ba08ac` 重测中止，未判通过 |
| FLUX2 / FLUX3 sliding_qdm 的 VDN+CLEAR 完整原生闭环及其余算法闭环 | 未执行；层检查和小模型策略检查不替代完整闭环 |
| FLUX3 cross_scan_kda / CLEAR 原生四卡闭环 | `3ab21c5` 在线 feature loss OOM；`15a007b` 缓存模式和 `129b969` 缓存加 activation offload 均在首次 forward OOM，未完成 backward |
| BF16 计算存储、ZeRO 本地 FP32 master / 状态 | `27c0935` 修复；CPU 策略/teacher 隔离及两 rank ZeRO 所有权、微小更新、精确下一步恢复通过，`gpu-memory-repair/delivery.md`；完整原生 CUDA 闭环未验证 |
| H800 8 GPU、2K–4K、每卡 batch4 与长程画质 | 未验证；本机四张 A40 不覆盖目标规模 |

`27c0935` 对 660 个选定张量、2,547,178,392 元素的计算存储每 rank 减少 5,094,356,784 字节（4.74449 GiB）；这是结构计算，不是实测 CUDA 峰值。严格确定性算子重复梯度检查通过，不等于完整闭环通过。

## 恢复历史检查

测试、计划和验收控制器保存在历史 revision `6e6ce43973975c0b1628b01e9005078c0c47cb36`。可在独立目录恢复完整历史环境，避免将旧入口混入当前运行目录：

```bash
git worktree add --detach ../PiD-Linear-checks 6e6ce43973975c0b1628b01e9005078c0c47cb36
cd ../PiD-Linear-checks
# 阅读 docs/attention-framework-acceptance.md 和对应 tests/attention_framework/ 检查。
```

本机另有 `.git/local-archive/cleanup-20261111/`，保存 plans、tests、历史文档和诊断脚本。历史检查依赖当时的入口布局，应在历史 worktree 中运行；对新入口的检查应更新导入与命令路径。后续完整闭环必须记录真实配置、资源、版本、连续/恢复比较和 GPU 峰值。生产缓存恢复任务独立于这组验收记录。
