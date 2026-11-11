# 验收记录与待测项

2026-11-11 用户调整交付范围：先提交代码及本记录，优先完成 **FLUX3 / cross_scan_kda / CLEAR** 的真实四卡训练、保存、精确恢复和生成；该项通过后推送。其他模型与组合的未完成验收继续保留，不阻塞此次交付。全矩阵控制器仍按完整范围报告，不用其历史失败报告冒充全矩阵通过。

运行资料位于本机 `attention-framework-refactor/reports/`。每个 acceptance.json 的 `evidence_commit`、`evidence_report`、`reused_via_report` 保留原始版本与来源。测试资源、日志、checkpoint 和缓存不入 Git。

| 检查 | 状态 | 版本与证据 |
|---|---|---|
| 配置、数据、桶、采样、路径、下载边界 | 通过 | `94b5a611`，`gpu-0330/acceptance.json: config_data` |
| 收敛、导入、入口、pid 与历史文件保持 | 通过 | `94b5a611`，`gpu-0330/acceptance.json: convergence` |
| 定时衔接、durable 模型与发布 gate | 通过 | `94b5a611`，`gpu-0330/acceptance.json: handoff` |
| FLUX3 原生层 × 四 attention、真实编码及四缓存模式 | 通过 | `94b5a611`，`gpu-0330/acceptance.json: flux3` |
| PiD / FLUX2 原生层 × 四 attention、真实编码及四缓存模式 | 通过 | `2123c3f`，`gpu-0330-repair-2123c3f/acceptance.json: pid, flux2` |
| 公共 CUDA/reference、12 模型算法组合、21 小模型策略组合、梯度和 teacher 隔离 | 通过 | `34d38d1`，`gpu-final-34d38d1/acceptance.json: common` |
| 四 rank NCCL、A1 平均、DDP 累积、分片优化器/RNG 下一步恢复、缺 shard、FP32 更新与验证恢复 | 通过 | `d825959`，`gpu-native-resume-d825959/acceptance.json: distributed` |
| PiD sliding_qdm / VDN 完整原生闭环精确比较 | 失败，待重测 | `d825959`：训练、保存、生成及中断恢复执行；最终连续/恢复参数不完全相等。严格确定性设置前的结果，见 `gpu-native-resume-d825959/pid_cycle.log` |
| 严格确定性 PiD sliding_qdm / VDN+CLEAR 原生闭环 | 中止，未判通过 | `8ba08ac`，用户切换优先级时停止 workflow `01M740D44S0S39MF9909K629QN`；checkpoint 与日志保留于 `gpu-native-resume-8ba08ac/` |
| FLUX2 sliding_qdm / VDN+CLEAR 原生闭环 | 未执行 | 上游失败/优先级调整前尚未运行 |
| FLUX3 sliding_qdm / VDN+CLEAR 原生闭环 | 未执行 | 上游失败/优先级调整前尚未运行 |
| FLUX3 cross_scan_kda / CLEAR 原生四卡完整闭环 | 在线编码 OOM，缓存模式待重测 | `3ab21c5`，`priority-flux3-kda-clear-3ab21c5/acceptance.json`：全 33 层，在线编码后在 CLEAR feature loss 申请 8 MiB 时 GPU0 耗尽，尚未完成 backward。重测复用真实编码并校验过的 FLUX3 图文缓存，其他精度/层/优化器设置不变 |
| 其余算法/策略的完整原生闭环 | 未执行 | 不能由 12 组合层检查或 21 小模型策略检查替代；当前原生闭环入口默认 sliding_qdm，完整测试须显式选择算法和策略 |
| H800 8 GPU / 4K / batch4、长程画质 | 未验证 | 本机四张 A40 不覆盖此硬件和规模 |

已集成的修复：`2123c3f` 模型适配，`34d38d1` 独立 feature/output backward 测试图，`d825959` 分片优化器恢复当前 scheduler LR。`71a33e1` 与 `8ba08ac` 为精确继续测试启用严格确定性执行，保留 BF16、attention backend 和 offload 设置。单算子检查已证明 BF16 FlashAttention backward 在普通模式下不完全重复，严格模式下 FlashAttention、FlexAttention 和 QDM 重复梯度完全一致；见 `gpu-deterministic-repair/delivery.md`。该算子结果不等于完整原生闭环通过。

## 继续测试

`RESOURCES` 指向真实模型/编码器/数据资源 JSON，`REPORT` 必须是新目录。通过 durable command 和 machine hold 执行；运行前确认物理 GPU0–3 无外部占用。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCEPTANCE_REPORT_DIR="$REPORT" \
  python tests/attention_framework/check_integrated.py \
  --resources "$RESOURCES" --scenario flux3_cycle \
  --algorithm cross_scan_kda --strategy clear
```

该命令比较连续与逐步保存恢复两条路径，四 rank、每卡 batch1、累积2、4步、真实在线编码、独立验证生成、中断补图及 public CLI 恢复；所有权重、优化器、scheduler、progress、RNG 使用精确比较。原生资源 JSON 控制 BF16、gradient checkpointing、优化器分片与 CPU offload。

重测其他项时用 `--scenario pid_cycle` / `flux2_cycle` / `flux3_cycle`，通过 `--algorithm` 选择四种算法，通过 `--strategy vdn` 或 `clear` 选择有效策略；CLEAR attention 不支持 VDN。不传策略会执行 VDN 与 CLEAR 两种闭环。完整控制器入口和绑定方法见 `attention-framework-acceptance.md`。每次保留版本、命令、实际配置、结果与来源；中止、blocked 和未执行均不记为通过。

优先项缓存重测使用 `gpu-0330/flux3/cache` 的 24 条真实验收样本。CPU 预检逐条读取图文张量并核对完整 metadata，24 条全部匹配，未加载 TE/VAE。该小型验收缓存独立于后续生产缓存任务。重测资源 JSON 需设置 `cache.image=true`、`cache.text=true` 以及该缓存根目录。在线模式的四种缓存/编码器行为已有 family 检查覆盖，完整训练闭环按缓存模式测量。
