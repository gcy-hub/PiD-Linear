# 数据与训练生命周期

新框架的入口契约见 [共享约定](../plans/attention-framework/contract.md)。命令从仓库根目录运行。默认输出到仓库旁 `../experiment/`。

数据使用实际 manifest 的记录数，不使用写死的 80K/50K 长度。JSONL/CSV 中的样本身份、caption、尺寸及相对图像路径交给统一 loader；路径相对 manifest。预处理保留 EXIF 方向、等比缩放/裁剪、过小样本拒绝统计。默认桶复用 `pid/_src/datasets/buckets.py`；小分辨率验收配置单独指定小桶，不修改官方桶表。

下载→manifest/桶准备→可选缓存→训练→保存/验证/恢复的顺序固定。下载命令见 [download](../download/README.md)，训练命令见 [README](../README.md)。manifest 支持 JSONL、CSV 和带索引的数据目录，loader 在第一次读取时完成桶选择与过小样本过滤，没有单独的桶索引生成 CLI。JSONL 每行例如 `{"id":"one","image_path":"images/one.png","caption":"a landscape"}`；图像路径相对 manifest。

先在本地模型 YAML 中填写 `model.resources.weights/vae/text_encoder/tokenizer`，在数据 YAML 中填写 manifest 与桶。缓存准备调用实际编码器：

```bash
python -m data.prepare \
  --model-config "$MODEL_CONFIG" \
  --algorithm-config configs/algorithms/sliding_qdm.yaml \
  --training-config configs/training/clear.yaml \
  --data-config "$DATA_CONFIG" \
  --phase all --devices cuda:0 cuda:1 --cache-batch-size 1 \
  cache.root="$CACHE_ROOT"
```

`--phase image|text|all` 分别准备两类缓存；训练通过 `cache.image=true cache.text=true cache.root="$CACHE_ROOT"` 选择读取。准备无需启用训练缓存开关。设备列表按样本分配，当前实现依次处理各设备，不表示并行 worker；FLUX3 文本保持逐样本编码。再次执行会跳过身份匹配的项，失败记录位于 cache.root/failures.json。缓存命令不是 print-config：即使传入该开关也会进入准备流程。

训练与自动保存/验证用 README 的 `python -m training.run`。已发布断点可以独立采样：

```bash
python -m training.validate \
  --config "$OUTPUT/resolved.yaml" \
  --checkpoint "$OUTPUT/step-000000010"
```

checkpoint 参数是包含 complete.json 与全部 rank shard 的目录。输出为该目录下的 validation/sample.png，成功后写 validated.json。采样使用数据配置中的第一张图与 caption、`validation.steps/seed`，正文本 CFG=1；FLUX2 使用内嵌 guidance=4。采样会加载真实 TE/VAE，不继承训练缓存省编码器的行为。需要调整采样配置时复制 resolved.yaml 修改；不要用历史 checkpoint。

| `cache.image` | `cache.text` | 训练时加载 |
|---|---|---|
| false | false | VAE + TE |
| true | false | TE |
| false | true | VAE |
| true | true | 都不加载 |

启用任一缓存需设置 `cache.root`。缓存准备与在线路径调用相同编码函数，使用 no-grad 但不生成无法参与后续训练的 inference tensors。缓存边界比较样本、资源/tokenizer、预处理和桶信息；一致内容可移动根目录。准备支持设备列表、batch、跳过已完成项、原子发布和失败样本记录。训练内循环直接读取。

PiD 保存完整 RGB 训练目标和干净的低分辨率 VAE condition；缓存 condition 不删除 RGB。噪声、退化、dropout、timestep 在读取后按训练 RNG 生成。FLUX2 保留原生 VAE packing/归一化，FLUX3 保留原生 latent tokens 和 IDs，不把三者改造成同一种潜空间。

`runtime.gradient_checkpointing`、`runtime.optimizer_sharding`、`runtime.activation_cpu_offload`、`runtime.optimizer_cpu_offload` 是默认关闭的显存选项。算法所需 FP32 状态和可靠优化器更新不属于可删除选项。开启选项时，验收必须记录实际分辨率、每卡 batch、累积、dtype 与峰值显存；缩小输入仅说明正确性，不证明目标 4K 不 OOM。

`runtime.dtype` 支持 `float32` 和 `bfloat16`，默认 BF16；FP16 训练因缺少 loss scaling 在配置边界拒绝。Prodigy 使用 FP32 master 保存细小更新。VDN A1 先汇总各层在所有 rank 和累积 microbatch 上的 teacher 能量，再立即反传每个独立 student 图，保持完整 batch 的相对 MSE。

新断点保存基础模型身份、替换配置、训练权重、优化器/调度器、RNG、rank-local microbatch cursor、VDN 阶段及阶段内/总步数。恢复通过 `checkpoint.resume=/path/to/complete-checkpoint` 覆盖；分片恢复要求固定 world size。分片先各自写入并完成发布，随后才淘汰上一完整集合。旧 checkpoint 不导入。

`checkpoint.every` 和 `validation.every` 独立；验证边界也必须发布可恢复断点。所有训练 rank 退出后独立采样，再恢复训练；验证期间无训练 NCCL group 等待。验证中断后先补该断点待生成的图片，再继续下一步。

新增算法只需导出 `build_attention(config, *, hidden_size, num_heads, layer)`，模块接收 `AttentionInput` 返回 `AttentionResult`，并列出 `role_parameters()`。复用布局和已有邻域/QDM helper。`pre_o` 恰好执行一次原生 O；可选 `post_o_residual` 是自带输出投影的远程分支。需要矩形/多样本/reference、梯度、文本可见性、投影位置、teacher 和实际 CUDA 检查。

新增模型只需直接分派构建适配器，实现 `module`、`layers`、`role_parameters()`、`prepare_batch()`、`forward(..., teacher=False, capture=())` 和 `layerwise(batch, callback)`；编码器通过 `build_encoder(config, device=..., image=..., text=...)` 按需加载。原生组名/层序号、位置处理、QKV/O、feature taps、目标和采样必须明确。需要无替换原生一致性、四算法真实层、teacher 投影隔离、在线/缓存一致与完整恢复闭环。接口完整签名以共享约定为准。

## Limitations

A1 每个 microbatch 需要额外一次 layerwise 前向，并保留整个累积窗口的已准备输入；当前接口在能量统计期间也计算 student 输出，各 callback 之间不保留 student 图。

真实资源、GPU 数值检查与完整闭环尚待验收。缓存与采样入口可执行不代表这些检查已经通过。
