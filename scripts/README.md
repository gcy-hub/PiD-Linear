# 训练与工具入口

从仓库根目录运行，路径与算法/batch/窗口/步数通过配置覆盖。默认输出到 `../experiment/`。

| 工作 | 入口 |
|---|---|
| 解析/校验配置，不加载 GPU | `python -m attention_framework.config --print-config ...` |
| 统一训练与保存/独立验证生命周期 | `python -m training.run ...` |
| 原生图像/文本缓存准备 | `python -m data.prepare --phase all --devices cuda:0 --cache-batch-size 1 ...` |
| 独立 checkpoint 采样 | `python -m training.validate --config <resolved.yaml> --checkpoint <complete目录>` |
| 生成真实资源验收绑定 | `python scripts/generate_attention_bindings.py --resources <JSON> --output <bindings.json>` |
| 四卡验收（规划默认不执行） | `python scripts/accept_attention_framework.py --reports <目录>` |
| CPU 配置/reference 与验收 gate | `CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests/attention_framework -p 'test_*.py' -v` |
| 显式资源与数据下载 | [download/README.md](../download/README.md) |

统一配置命令见 [README](../README.md)，数据/缓存和恢复语义见 [生命周期](../docs/attention-framework.md)。prepare 同时准备可恢复缓存；无需缓存时直接用 manifest 训练。独立采样读取完整新 checkpoint。验收绑定方法见 [四卡验收](../docs/attention-framework-acceptance.md)。

原生 PiD DMD 仍使用 `torchrun ... -m scripts.train --config <Python配置>`；`scripts/train.py` 和多机模板保留。`pid/_src/pixeldit_qdm/synthetic_pipeline.py` 仍调用 `train_adapt.sh` 与 `train_pixeldit_synthetic_cycle.sh`，所以保留对应薄入口及原生合成数据编排。`train_qdm_stages.sh`/`repeat_pid_vdn.py` 的原生调用仍有使用者，不在新框架交付中机械删除。

保留的原生诊断、采样和实验说明见 [原生训练](../docs/historical-training.md)。旧 FLUX 参数脚本与训练/缓存实现已移除；外部历史 checkpoint 和缓存保留，使用相应历史 Git revision 读取。数值诊断统一运行 `python -m algorithms.checks.cpu`，独立密集参考位于 `algorithms/checks/references.py`。
