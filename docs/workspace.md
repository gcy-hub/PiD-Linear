# 工作目录与路径

从仓库根目录运行命令。Python 使用当前激活的环境；需要指定解释器时设置 `PYTHON`，例如 `PYTHON=/opt/envs/flux/bin/python`。Python 子进程使用当前解释器 `sys.executable`，仓库内部文件通过脚本位置定位。

模型、数据、缓存和输出目录通过配置或命令行参数指定，可以使用绝对路径。示例配置中的 `weights/`、`dataset/` 和 `../experiment/` 是相对路径，部署时请根据实际位置填写。不要把用户名、服务器挂载点或 Conda 安装位置写入源代码。

每个实验使用独立输出目录保存断点、优化器状态、图片和日志。`../experiment/` 可以是普通目录，也可以是指向外部存储的符号链接。训练与验证运行期间不要切换分支。

## 历史实验入口

`train_flux2_node01.sh` 和 `train_flux3_node01.sh` 保留三卡实验配置。它们支持 `PYTHON`、`WEIGHTS`、`DATA` 环境变量（FLUX3 另有 `RESOURCES`），以及第一个位置参数指定输出目录。新机器的通用在线训练使用 `python -m flux_kda.run --config <配置文件>`。

80K 缓存工具使用 `FLUX3_DATASET_ROOT` 和 `FLUX3_WEIGHTS_ROOT`。混合比例实验及诊断工具使用 `FLUX3_SUBSET_ROOT`、`FLUX3_WEIGHTS`。如需限定缓存或混合比例实验的执行主机，设置 `FLUX_KDA_HOST`；恢复工具使用 `--host` 指定主机，默认当前主机。这些变量应在运行时设置；历史实验入口仍需要对应的数据、断点和验证条件。

## 断点与本机归档

同一实验阶段或独立对照组保留最新完整断点。模型状态与对应的优化器分片必须一起保留，才能恢复完整训练状态。

如果本机存在 `.git/consolidation-20261110/`，其中保存了代码归档、原工作目录记录、实验迁移清单和旧 cron 配置。该目录及实验文件不会随普通 `git push` 上传，新机器也不依赖它们运行训练。
