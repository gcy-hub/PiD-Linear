# 四卡验收

2026-11-11 用户将本次发布条件调整为 FLUX3 cross_scan_kda + CLEAR 真实四卡训练/精确恢复/生成通过，其余项记录后续测试。当前状态及复测命令见 [验收记录与待测项](attention-framework-outstanding-tests.md)。下文完整矩阵控制器保留完整验收语义；此次发布使用独立优先项报告，不声明全矩阵通过。

验收覆盖批准计划的八个 Verification 项；H800 八卡 4K/batch4 单独记录为未验证。报告、日志与资源专用配置保存在运行时指定的 reports 目录，禁止提交权重、缓存和运行日志。

先运行 CPU 检查：

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests/attention_framework -p 'test_*.py' -v
```

控制器入口为 `python scripts/accept_attention_framework.py --reports "$REPORTS" --bindings "$BINDINGS"`。不加 `--execute` 只生成待验收报告并以 1 退出，不触碰 GPU。集成任务准备资源专用 bindings JSON；每个键为下表的 scenario ID，值含 `argv`（非 shell 字符串）、`config`（实际配置，包括 dtype/桶/步数/batch/累积/显存开关）、`resources`（本地权重、编码器、manifest/cache 等）。未知 ID、空命令或缺配置/资源直接拒绝。缺少任何必需 binding 都是 blocked，不是 skip。

`distributed` 的 argv 从测试脚本或 `-m module` 开始，控制器添加四 rank torchrun；其他 argv 包含 Python 可执行文件。子测试从 `ACCEPTANCE_REPORT_DIR` 获取自己的报告目录，读取 `ACCEPTANCE_SCENARIO` 标识。一个场景内任意必需断言未完成必须非零退出；pytest skip 或仅 help/print-config 不构成场景通过。真实场景保存实际 resolved YAML、测得的显存、dtype、分辨率、batch 和资源身份。

| Scenario ID | 设备/顺序 | 必需断言 |
|---|---|---|
| `config_data` | CPU | 覆盖顺序、未知字段/无效组合/非法层/窗口/cache 拒绝；实际条数；官方桶、矩形/EXIF/边缘 padding/过小图统计、小桶、多 rank 与 accumulation 分配；运行路径一致；下载假响应选择/跳过 |
| `convergence` | CPU | 删除路径无残留 import/命令；新包发现、安装和入口 help；`pid/` diff 空，历史实验/权重/日志保留；代码无个人机器路径；README 默认值与配置一致 |
| `handoff` | CPU | 复用登记时的有界衔接检查：到点、漏交接、进程中断后接续、完成不重复；durable 实际记录核对 astra/sol 和 zzz provider；mandatory gate 与远端发布证据 |
| `pid` | 物理 GPU0，与下三项并行 | 真实原生 PiD 层 × 四算法，未选层/像素块一致；真实在线编码及四缓存组合，无 TE/VAE 时实际不加载；空文本/不同长度/桶；完整 RGB；seed 改噪声/退化而不改缓存 |
| `flux2` | 物理 GPU1 | 真实 double/single 层 × 四算法与编码/四缓存组合；原生 packing/归一化；融合 FFN 未更新；无替换原生一致 |
| `flux3` | 物理 GPU2 | 真实 early_image/joint 层 × 四算法与编码/四缓存组合；原生 latent/IDs；无替换原生一致 |
| `common` | 物理 GPU3 | CUDA/reference 前后向；12 pairs 与 21 strategies 小模型；文本/多样本隔离、矩形/halo padding、部分选择不改 KDA 序号；QDM/KDA 投影位置；A1 teacher trunk/独立图、A2/B 角色/切换、CLEAR tap 梯度到远程分支；student 更新后 teacher 权重/输出不变；CLEAR+VDN 拒绝 |
| `distributed` | 上述进程退出后，GPU0–3 四 rank torchrun | A1 手动梯度平均对齐单进程全局 batch；A2/B/CLEAR DDP；accumulation；FP32 小更新跨保存恢复；CUDA 分片 Prodigy；rank-local shard/优化器/调度器/RNG/cursor 连续与恢复下一步一致；缺 shard 报错；发布前注入保存失败保留上次 complete；验证期间无训练 NCCL group |
| `pid_cycle` | 随后串行，GPU0–3 | 真实小分辨率 train→save→resume→generate；连续/恢复参数与下一步一致；VDN 阶段内/边界；验证中断先补图；独立保存/验证频率边界 |
| `flux2_cycle` | 同上 | 同上，原生 FLUX2 真实资源 |
| `flux3_cycle` | 同上 | 同上，原生 FLUX3 真实资源 |

已有 `plans/attention-framework/checks/contract_cpu.py` 提供 21 配置/12 pairs 和抽取前数值参考，由 unittest 复用。`tests/attention_framework/check_algorithms.py` 是实际 CUDA 算子的形状/有限梯度/样本隔离检查，直接调用契约的 `build_attention`；它只覆盖 `common` 的一部分，不能单独绑定为完整 common 场景。模型、数据和训练的集成绑定见下文。

实际执行由现有 durable workflow 在机器 lease 内调用，避免单卡与 DDP 抢卡：

```bash
pi-durable-subagents hold machine -- \
  python scripts/accept_attention_framework.py \
  --reports "$REPORTS" --bindings "$BINDINGS" --execute
```

GPU0/1/2/3 外部占用在启动前检查，忙碌则 blocked，由同一已有 workflow 等待资源后接续。控制器无重试或新调度机制。CPU 场景必须先通过，随后四单卡并行，全部退出且通过后运行分布式及真实家族闭环；失败阻止后续项。mandatory blocked/failed、规划报告或 dirty worktree 都使退出码非零。记录测试 HEAD、工作区状态、实际命令/配置/资源、GPU 清单和逐项结果；不生成额外摘要哈希。

`acceptance.json` 的 `verification` 始终包含 1–8，`publish_allowed` 只允许全部 mandatory 通过的干净版本。它是正确性 gate，远端 SHA、cron task ID、durable workflow/request IDs 与指定模型记录由集成任务并入最终报告。集成/修复后只重跑受影响项；推送前远端变化需集成并补测。不得用此次四卡报告替代 H800 八卡目标或长程画质结论。

## Limitations

本地资源 YAML、测试 manifest 与实际 durable 记录需要现场提供。规划报告中的所有未执行项均为 blocked；不得推送。此控制器不创建 cron、重启 orchestrator 或变更 provider。

## 集成接口绑定

`generate_attention_bindings.py` 从本地 JSON 配置生成全部 11 个 scenario 的 argv，无 shell 展开。`models` 的三份 YAML 使用仓库模型配置结构，并添加实际 `model.resources`；原生 FLUX checkpoint 旁的 `config.json` 决定架构，`model.layer_counts` 必须一致。资源 JSON 与生成的 bindings 保存到 reports，不提交。

```json
{
  "models": {
    "pid": "/absolute/local/pid.yaml",
    "flux2": "/absolute/local/flux2.yaml",
    "flux3": "/absolute/local/flux3.yaml"
  },
  "data": "/absolute/local/data.yaml",
  "overrides": ["runtime.dtype=bfloat16", "training.batch_size=1"],
  "family_overrides": {
    "pid": ["replacement.layers.patch=[0,1]"],
    "flux2": ["replacement.layers.double=[0]", "replacement.layers.single=[0]"],
    "flux3": ["replacement.layers.early_image=[0]", "replacement.layers.joint=[0]"]
  },
  "durable_records": ["/absolute/reports/actual-astra-run.json", "/absolute/reports/actual-sol-run.json"]
}
```

模型 YAML 可直接复制 `configs/models/<family>.yaml` 后增加 `model.resources: {weights: /local/native, vae: /local/vae, text_encoder: /local/te, tokenizer: /local/tokenizer}`。数据 YAML 至少设置实际 `data.manifest`、两个不同的小桶与 `data.min_size`；manifest 必须含空 caption 和不同长度 caption，两个桶均有样本。各 CLI 继承同一共享默认，资源路径建议使用绝对路径。原生权重不会自动下载或替换为测试模型。

```bash
python scripts/generate_attention_bindings.py \
  --resources "$REPORTS/resources.json" --output "$REPORTS/bindings.json"
python scripts/accept_attention_framework.py \
  --reports "$REPORTS/plan" --bindings "$REPORTS/bindings.json"
```

`config_data` 复用交付的 data/algorithms/models/training CPU checks 及 unittest；包含 mock 的检查仅属于 CPU 证据。`common` 调用 CUDA/reference 检查与原生小 attention 层的 21 策略组合，覆盖 PiD patch、FLUX2 double/single、FLUX3 early_image/joint。`distributed` 默认四 rank，测试手动平均、DDP accumulation、分片优化器的下一步/RNG 恢复、缺 peer shard，并复用 FP32 master 与保存/验证失败检查。其合成参数测试不替代三个真实家族闭环。

每个家族场景逐样本检查真实在线与四缓存模式，以及 encoder 上 TE/VAE 对象确实不存在；随后加载四算法、检查梯度、teacher 更新隔离，并重新加载无替换原生模型比较 teacher 输出。每个 `*_cycle` 用真实四 rank engine 分别跑 CLEAR 与 VDN 三阶段，对比连续/每步恢复的权重、优化器、调度器、RNG 和 cursor，独立进程生成 PNG。测试输出目录必须为空；保留旧报告，下一次运行换目录。

`handoff` 默认复用已交付的有界 handoff/workflow 检查，并要求读取实际 durable 导出的运行记录，核对 zzz/astra/sol。记录缺失会失败；不启动新工作流。现场已有完整审阅命令时，可通过资源 JSON 的 `handoff_argv` 提供该命令。远端发布 SHA 由集成任务在通过所有场景后另行记录，不把尚未发布当作 CPU 先决条件。

CPU 可独立检查新分布式 harness；这不是四卡结果：

```bash
CUDA_VISIBLE_DEVICES='' ACCEPTANCE_REPORT_DIR="$REPORTS/gloo" \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tests/attention_framework/check_distributed.py --cpu --world-size 2
```
