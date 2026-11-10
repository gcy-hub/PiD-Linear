# 工作目录

代码仓库：`/home/ganchangyi/code/PiD-Linear`。

实验根目录：`/home/ganchangyi/experiment/PiD-Linear`。每个实验使用独立子目录保存权重、优化器状态、验证图片、日志和运行配置。仓库内的 `outputs` 符号链接指向这个目录；新配置直接使用实验目录的绝对路径。

Git 只有一个工作目录。当前主线为 `main`，其他版本保留在分支或归档引用中。运行训练或验证期间不要切换分支。

## 断点

同一实验阶段或独立对照组只保留最新完整断点。`state.pt`、`delta.pt` 以及对应优化器分片属于同一组恢复文件，应一起保留。验证图片、日志、编码缓存和不同模型容量的独立实验文件保留。

FLUX3 KDA 训练的当前断点位于实验根目录下的 `flux3-stock-kda-clear-node01-bs1/latest.pt`，与 `optimizer-4000-rank-{0,1,2}.pt` 配套。FLUX2 KDA 对应 `flux2-stock-kda-clear-node01-bs2/latest.pt` 和第 600 步的三个优化器分片。

历史实验的 cron 启动项已停用，路径更新为当前目录布局。恢复旧实验前，应检出匹配的代码版本，再使用相应实验配置启动。CLEAR 仓库的独立任务保持原状。

## 本机代码归档

旧工作目录的代码保存在 `.git/consolidation-20261110/`：

- `worktrees.json`：原目录、分支、提交和未提交文件状态。
- `*.tar.gz`：包含未提交文件及忽略规则下源文件的代码归档。
- `nested-repositories/`：原目录中的嵌套仓库，包含各自 Git 历史。
- `experiments.json`：实验目录对应关系。
- `checkpoint-plan.json`、`additional-cleanup.json`：保留和清理的断点清单。
- `crontab.before`：自动启动任务的原配置。

原提交另外保存在 `refs/archive/consolidation-20261110/<工作目录名>`。原主目录的未提交文件也保存在名为 `pre-consolidation original worktree` 的 Git stash 中。

这些归档是本机文件，普通 `git push` 不会上传它们。迁移到其他服务器时，需要分别复制实验目录和需要保留的本机归档。
