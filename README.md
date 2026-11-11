# Attention 实验框架

PiD、FLUX2、FLUX3 共用 attention 算法、数据生命周期和训练入口。基础模型配置指定架构、资源和原生采样；替换模型由同一基础配置、attention 配置和层选择组成。

| 目录 | 职责 |
|---|---|
| `models/` | 原生模型、稳定层名、QKV/O、teacher 和采样 |
| `algorithms/` | attention 张量算子 |
| `data/` | manifest、官方桶、编码、缓存和原生数据准备 |
| `training/` | 训练策略、精度、分布式、checkpoint、验证和原生训练入口 |
| `attention_framework/`、`configs/` | 接口、配置与校验 |
| `download/`、`scripts/` | 显式下载、shell / Slurm 执行入口 |
| `pid/` | 原生 PiD / PixelDiT 实现 |

`cross_scan_kda`、`sliding_qdm`、`qdm_halo` 支持三个模型的 VDN / CLEAR；`clear` attention 支持三个模型的 CLEAR。CLEAR attention + VDN 在配置边界拒绝。KDA 使用原生层序号轮换扫描方向，部分替换不重新编号。

Python 3.12，从仓库根目录安装 `pip install -e .`，再按所选算法准备 FLA、Prodigy 等运行依赖。训练从本地资源加载；先读 [下载说明](download/README.md)，填写本地模型资源和 manifest。

```bash
python -m training.run --print-config \
  --model-config configs/models/flux3.yaml \
  --algorithm-config configs/algorithms/cross_scan_kda.yaml \
  --training-config configs/training/clear.yaml \
  --data-config configs/data/default.yaml
```

实际运行去掉 `--print-config`，并传入 `model.resources.weights/text_encoder/tokenizer/vae`、`data.manifest` 和 `paths.output`。可直接使用 [FLUX3 cross-layer KDA / 80K / 八卡训练说明](docs/flux3-crosslayer-kda-80k.md) 的完整命令。

[生命周期](docs/attention-framework.md) 说明缓存、恢复、验证与显存选项；[接口约定](docs/attention-framework-contract.md) 说明扩展边界；[执行入口](scripts/README.md) 和 [原生训练](docs/historical-training.md) 说明 PiD / PixelDiT 原生路径。旧 FLUX 实验使用相应历史 Git revision。

## Limitations

支持范围不等于完整 GPU 验收通过。[验收记录](docs/attention-framework-outstanding-tests.md) 保留已有证据和待测项。H800 八卡、官方 2K–4K 多长宽比、每卡 batch 4 的峰值显存、完整恢复及长程画质尚未验证。
