# 原生 PiD sliding window

原生滑窗入口使用 JSON 配置与通用训练 launcher：

```bash
GPU_IDS=0,1,2,3 bash scripts/train_adapt.sh \
  --config configs/training/pid-sliding.json --mode repeat \
  --weights-root "$PID_WEIGHTS_ROOT" --index-root "$PID_INDEX_ROOT" \
  --subset-root "$PID_VDN_SUBSET_ROOT" --text-cache-root "$PID_TEXT_CACHE_ROOT" \
  --gallery-root "$PID_GALLERY_ROOT" --output-root ../experiment/pid-sliding
```

安装与当前 PyTorch/CUDA 匹配的 NATTEN wheel；具体 kernel API 见 `pid/_src/vdn_pid/sliding.py`。阶段、资源准备、停止与恢复用法见 [原生训练](historical-training.md)。共享 attention 框架的 sliding_qdm 使用独立配置与入口，见 [生命周期](attention-framework.md)。
