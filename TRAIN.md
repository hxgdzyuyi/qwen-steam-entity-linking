# 云端训练与 Hugging Face 发布指南

本文档说明如何在云端完成当前项目的训练、断点恢复、评测，以及将最终 LoRA 发布到 Hugging Face。

项目当前默认使用：

- 基础模型：`Qwen/Qwen3-8B-Base`
- 训练数据：1000 条 canonical 游戏实体样本
- 别名评测数据：184 条
- 训练方式：BF16 LoRA，`r=64`、`alpha=128`
- 完整训练：20 epochs
- 里程碑 checkpoint：第 1、3、5、10、20 个 epoch
- 成功门槛：canonical generation accuracy 至少达到 99%

## 一、在本机准备并推送代码

云端应当使用一个固定且已经推送到 GitHub 的 Git commit。先在本机进入项目：

```bash
cd /Users/qingyang/mine-work/qwen-steam-entity-linking
```

运行本地测试：

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/*.py tests/*.py
```

检查 Git 状态：

```bash
git status
```

如果存在准备带到云端的修改，应先有选择地提交这些修改。确认工作区干净后推送：

```bash
git push origin main
git rev-parse HEAD
```

记录 `git rev-parse HEAD` 输出的完整 commit SHA，后续云端使用该 SHA。不要只依赖分支最新状态，以免训练期间代码发生变化。

训练数据已经生成并提交，通常不需要在云端重新运行：

```text
scripts/sync_steam_dataset.py
scripts/build_training_data.py
```

重新生成数据会改变数据哈希，并可能使 Git 工作区变脏。

## 二、申请云机器

训练配置会强制检查运行环境。目标配置为：

- GPU：1× H100 SXM 80GB
- GPU 显存：至少 70GiB
- 系统内存：至少 100GiB
- CPU：至少 16 核
- 工作磁盘：40GB；首次下载基础模型前至少剩余 30GiB
- PyTorch：2.8.x
- CUDA：12.8.x
- RunPod 镜像：`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`

模型缓存和训练结果都应放在 `/workspace`。如果使用其他云平台，也必须满足上述硬件和软件版本检查。

## 三、在云端克隆并固定版本

通过云平台 Terminal 或 SSH 执行：

```bash
cd /workspace

git clone https://github.com/hxgdzyuyi/qwen-steam-entity-linking.git
cd qwen-steam-entity-linking

TRAIN_COMMIT=替换为本机记录的完整commit_SHA
git checkout "$TRAIN_COMMIT"

git rev-parse HEAD
git status --short
```

建议 `git status --short` 没有输出。训练脚本会把训练 commit 和工作区状态写入运行清单；如果工作区不干净，只发出 warning，不会阻断云端任务。

训练开始后仍建议不要执行 `git pull` 或修改配置和数据。断点恢复时，配置和数据必须与原始运行一致；Git commit 不一致只发出 warning。

## 四、推荐方式：通过 Jupyter Notebook 训练

先把 Notebook 复制到仓库外，避免 Jupyter 写入执行计数和输出后弄脏 Git：

```bash
cp notebooks/runpod_training.ipynb /workspace/runpod_training.ipynb
```

通过云平台提供的 Jupyter 页面打开：

```text
/workspace/runpod_training.ipynb
```

从上到下逐格执行。Notebook 会依次完成：

1. 定位 `/workspace/qwen-steam-entity-linking` 项目。
2. 检查 GPU、磁盘和 Git 状态。
3. 安装云端依赖。
4. 可选地通过隐藏输入注入 `HF_TOKEN`。
5. 运行 32 条样本的冒烟训练。
6. 运行完整 1000 条数据训练。
7. 在中断后从最新可恢复 checkpoint 继续。
8. 评测五个里程碑 checkpoint。
9. dry-run 并人工发布到 Hugging Face。

基础模型是公开模型，下载时通常不需要 Hugging Face Token。发布时必须使用具有写权限的 Token。

## 五、命令行方式：准备环境

如果不使用 Notebook，在项目目录执行：

```bash
cd /workspace/qwen-steam-entity-linking

export HF_HOME=/workspace/.cache/huggingface

python3 -m pip install \
  --no-cache-dir \
  -r requirements-cloud.txt

nvidia-smi
df -h /workspace
git status --short
```

不要把 Hugging Face Token 直接写入命令、Notebook 明文或配置文件。发布时应通过云平台 Secret 注入名为 `HF_TOKEN` 的环境变量，或者使用 Notebook 中的隐藏输入单元。

## 六、运行冒烟训练

先用固定的 32 条样本验证模型下载、tokenizer 扩展、LoRA 和新增实体 token 是否能够正常学习：

```bash
python scripts/train.py \
  --config configs/qwen3_8b_lora.yaml \
  --mode smoke \
  --run-dir outputs/smoke
```

冒烟训练最多运行 100 epochs；连续两个 epoch 达到 100% canonical next-token accuracy 后会自动停止并保存 checkpoint。

首次运行会把基础模型下载到：

```text
/workspace/.cache/huggingface
```

完整训练会复用这份缓存。运行目录必须不存在或为空；如果已有需要保留的失败产物，应换一个新的 `--run-dir`，不要覆盖旧结果。

## 七、运行完整训练

冒烟训练成功后执行：

```bash
python scripts/train.py \
  --config configs/qwen3_8b_lora.yaml \
  --mode full \
  --run-dir outputs/full
```

完整训练会：

- 运行 20 epochs。
- 在第 1、3、5、10、20 个 epoch 保存 LoRA 和新增 token 行。
- 仅对 completion 的实体 token 和 EOS 计算 loss。
- 只让最新 checkpoint 保留 optimizer、scheduler 和 RNG 状态，以控制磁盘占用。
- 将 TensorBoard 日志写入运行目录下的 `tensorboard/`。

Notebook 默认使用的完整训练目录是：

```text
outputs/runpod-full
```

命令行示例默认使用：

```text
outputs/full
```

后续所有命令必须选择实际使用的同一个目录。

## 八、完整训练中断后的恢复

先查看已有 checkpoint：

```bash
ls -d outputs/full/checkpoints/checkpoint-* | sort -V
```

只使用编号最大的、仍包含 optimizer 和 scheduler 状态的 checkpoint。示例：

```bash
python scripts/train.py \
  --config configs/qwen3_8b_lora.yaml \
  --mode full \
  --resume-from outputs/full/checkpoints/checkpoint-替换为最大编号
```

使用 `--resume-from` 时不需要再传 `--run-dir`，脚本会从 checkpoint 路径推导原始运行目录。

如果中断时还没有产生可恢复 checkpoint，则不能复用这个非空运行目录，应使用新的空目录重新开始完整训练。

## 九、评测全部 checkpoint

命令行训练目录：

```bash
python scripts/evaluate.py \
  --run-dir outputs/full \
  --all-milestones
```

如果使用 Notebook，则将目录替换为：

```bash
python scripts/evaluate.py \
  --run-dir outputs/runpod-full \
  --all-milestones
```

评测会生成：

- `metrics.json`：汇总指标、分组指标、最佳 checkpoint 和预测指纹。
- `checkpoint_comparison.csv`：五个 checkpoint 横向比较。
- `evaluation_failures.csv`：错误输入、目标和预测。
- `run_manifest.json`：Git、模型、数据、依赖和运行状态。

必须确认：

```text
acceptance_passed = true
```

脚本会在 canonical accuracy 达到 99% 的 checkpoint 中，选择 alias accuracy 最高的版本；若相同，则依次使用 canonical accuracy 和更早 epoch 打破平局。

## 十、发布到 Hugging Face 前的要求

发布必须在仍有一张支持 BF16 的 CUDA GPU 的机器上完成。发布脚本会在上传前重新评测一次，上传后再下载并重新评测一次，因此不要在训练刚结束时立即关闭 H100。

发布前必须满足：

- 已完成五个里程碑 checkpoint 的评测。
- `metrics.json` 中 `acceptance_passed` 为 `true`。
- 五个 checkpoint 都仍然存在。
- 建议当前 Git commit 与训练时一致且工作区干净；不一致时仅发出 warning。
- 当前数据文件哈希与训练时一致。
- 运行目录所在磁盘至少还有 8GiB 可用空间。
- 已准备具有写权限的 Hugging Face Token。
- 最终仓库可以公开；当前发布脚本只支持最终公开发布。

建议使用一个全新的 Hugging Face 模型仓库名。无需提前手动创建仓库，发布脚本会负责创建。

## 十一、推荐方式：通过 Notebook 发布

在 `/workspace/runpod_training.ipynb` 中：

1. 执行指标展示单元，确认 `acceptance_passed = True`。
2. 通过隐藏输入单元注入具有写权限的 `HF_TOKEN`。
3. 在发布单元填写：

```python
HF_REPO_ID = '你的用户名/qwen3-8b-steam-entity-linking'
```

4. 执行 dry-run 单元，检查指标、目标地址和待上传文件。
5. 确认无误后设置：

```python
PUBLISH_PUBLIC = True
```

6. 执行最后一个发布单元。

## 十二、命令行方式：dry-run 与正式发布

首先设置实际运行目录和目标仓库。Notebook 训练示例：

```bash
cd /workspace/qwen-steam-entity-linking

RUN_DIR=outputs/runpod-full
MODEL_REPO_ID=你的用户名/qwen3-8b-steam-entity-linking
```

如果是命令行训练，则改为：

```bash
RUN_DIR=outputs/full
```

先执行 dry-run：

```bash
python scripts/publish_hf.py \
  --run-dir "$RUN_DIR" \
  --repo-id "$MODEL_REPO_ID" \
  --public \
  --dry-run
```

`--public` 在 dry-run 中也是必需的显式确认，但 dry-run 不会创建或修改 Hugging Face 仓库，也不要求 `HF_TOKEN`。它仍会重新加载并评测选中的 checkpoint。

确认 dry-run 成功后，通过云平台 Secret 注入 `HF_TOKEN`，然后正式发布：

```bash
python scripts/publish_hf.py \
  --run-dir "$RUN_DIR" \
  --repo-id "$MODEL_REPO_ID" \
  --public
```

发布脚本会执行以下安全流程：

1. 检查运行状态、指标、Git、数据哈希、磁盘和五个 checkpoint。
2. 重新评测选中的最佳 checkpoint，并与原始完整指标和预测指纹比较。
3. 将最佳 LoRA 放在仓库根目录。
4. 将五个里程碑 LoRA 放在 `adapters/epoch-*`。
5. 上传 tokenizer、special tokens、训练配置、运行清单、指标、对比表和自动生成的模型卡。
6. 不上传基础模型权重、optimizer、Trainer 状态或原始训练数据。
7. 对新仓库先以 private 状态创建并上传。
8. 精确检查远端文件集合，重新下载并复核 adapter-only 内容。
9. 使用下载后的内容再次评测。
10. 所有检查通过后才将仓库切换为 public。

如果目标仓库中存在本次发布列表之外的额外文件，脚本会拒绝发布，不会删除或静默覆盖这些文件。因此推荐每次首次发布使用一个新的空仓库名。

## 十三、确认发布结果

发布成功后，命令行会输出 Hugging Face 地址，并在运行目录生成：

```text
publish_receipt.json
```

查看发布凭据：

```bash
cat "$RUN_DIR/publish_receipt.json"
```

其中包含：

- Hugging Face repo ID 和公开 URL。
- 上传后的 revision。
- 被选中的 epoch。
- 发布时间。

同时，`run_manifest.json` 的状态会更新为 `published`。

## 十四、关闭云机器前

确认至少完成以下一种操作后，再停止或销毁云机器：

- Hugging Face 发布成功，并已检查公开仓库。
- 将完整运行目录下载到本地。
- 将运行目录复制到明确的持久化磁盘或对象存储。

至少保留：

```text
outputs/full/
```

或 Notebook 对应的：

```text
outputs/runpod-full/
```

不要只保留最终 LoRA；`metrics.json`、`run_manifest.json`、五个里程碑 checkpoint 和 `publish_receipt.json` 对复现、排错与重新发布都很重要。
