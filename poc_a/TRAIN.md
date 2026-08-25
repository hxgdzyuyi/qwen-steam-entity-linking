# PoC A 云端训练、调试与发布

PoC A 的云端操作以 Jupyter Notebook 为主：

| 用途 | Notebook |
| --- | --- |
| 环境准备、smoke、完整训练、断点恢复、正式评测、发布 | [`notebooks/runpod_training.ipynb`](notebooks/runpod_training.ipynb) |
| 加载本地或 Hugging Face 模型、交互预测、别名调试 | [`notebooks/runpod_model_testing.ipynb`](notebooks/runpod_model_testing.ipynb) |

Notebook 是云端执行入口，实际训练、评测和发布契约仍由 `scripts/` 中的 Python 程序实现。正常操作不需要在 Terminal 中手写训练命令；命令行只用于克隆仓库、固定 Git 版本和必要的故障排查。

当前默认配置：

- 基础模型：`Qwen/Qwen3-8B-Base`
- 训练数据：1000 个 canonical 游戏实体 × 4 种 prompt，共 4000 行
- 训练方式：BF16 LoRA，`r=64`、`alpha=128`，完整训练 10 epochs
- 里程碑 checkpoint：epoch 2、4、6、8、10
- 验收门槛：canonical 实体约束 Top-1 ≥ 99%，alias 实体约束 Top-1 ≥ 25%

## 1. 本机准备代码

云端训练应固定到一个已推送的 Git commit。本机在仓库根目录执行：

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile common/scripts/*.py poc_a/scripts/*.py tests/*.py
git status
git push origin main
git rev-parse HEAD
```

记录最后输出的完整 commit SHA。训练数据已经生成并提交，云端通常不需要重新运行 `sync_steam_dataset.py` 或 `build_training_data.py`；重新生成数据可能改变数据哈希。

## 2. 创建云实例并固定代码版本

目标环境：

- RunPod 镜像：`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- GPU：1× H100 SXM 80GB，显存至少 70GiB
- 系统内存：至少 100GiB
- CPU：至少 16 核
- 工作磁盘：40GB；首次下载模型前至少剩余 30GiB
- PyTorch 2.8.x、CUDA 12.8.x

模型缓存和训练结果都放在 `/workspace`。实例启动后，只需在 Terminal 中完成仓库准备：

```bash
cd /workspace
git clone https://github.com/hxgdzyuyi/qwen-steam-entity-linking.git
cd qwen-steam-entity-linking

TRAIN_COMMIT=替换为完整commit_SHA
git checkout "$TRAIN_COMMIT"
git rev-parse HEAD
git status --short
```

`git status --short` 建议没有输出。训练期间不要执行 `git pull`，也不要修改配置或数据；运行清单会记录 commit、工作区状态和数据哈希。

## 3. 复制并打开两个 Notebook

Jupyter 会保存执行计数和输出。先把 Notebook 复制到仓库外，避免污染 Git 工作区：

```bash
cp poc_a/notebooks/runpod_training.ipynb /workspace/poc_a_training.ipynb
cp poc_a/notebooks/runpod_model_testing.ipynb /workspace/poc_a_model_testing.ipynb
```

在 RunPod 的 Jupyter 页面中打开 `/workspace/poc_a_training.ipynb`。Notebook 会自动定位 `/workspace/qwen-steam-entity-linking`，并把 Hugging Face 缓存设置为 `/workspace/.cache/huggingface`。

不要直接对仓库内的 Notebook 执行并保存。需要保留云端调试记录时，下载 `/workspace` 下的副本即可。

## 4. 使用训练 Notebook

首次训练按顺序逐格执行，不要直接使用 `Run All`。每个长任务完成并检查输出后，再继续下一格。

### 4.1 环境与依赖

第 1 组单元会检查 GPU、`/workspace` 磁盘和 Git 状态，并安装 `requirements-cloud.txt` 与 `hf_transfer`。训练脚本还会强制校验 GPU、显存、内存、CPU、磁盘、PyTorch 和 CUDA 版本。

公开基础模型通常不需要 Hugging Face Token。后续需要发布时，在隐藏输入单元中输入具有 write 权限的 `HF_TOKEN`；Token 只进入当前 Kernel 的环境变量，不会写回 Notebook。

### 4.2 Smoke 训练

执行“运行 32 条冒烟训练”单元。默认输出目录为：

```text
poc_a/outputs/runpod-smoke
```

Smoke 最多运行 100 epochs；连续两个 epoch 达到 100% canonical next-token 准确率后自动停止。它用于验证基础模型下载、tokenizer 扩展、LoRA 和新增实体 token 的完整链路。

若该目录已经有真实产物，脚本不会覆盖。要重新实验，请先保留原目录，并在 Notebook 顶部把 `SMOKE_RUN_DIR` 改成新的空目录。

### 4.3 完整训练

Smoke 成功后执行“运行完整 4000 行训练”单元。默认输出目录为：

```text
poc_a/outputs/runpod-full
```

训练会在 epoch 2、4、6、8、10 保存 LoRA 和新增 token 行。为控制磁盘占用，只有最新 checkpoint 保留 optimizer、scheduler 和 RNG 状态；较早 checkpoint 仍可用于评测和发布。

不要重复执行已经成功启动过的完整训练单元。需要开始另一组实验时，先在 Notebook 顶部修改 `FULL_RUN_DIR`，使用新的空目录。

### 4.4 中断恢复

如果完整训练中断：

1. 不要重新执行“完整训练”单元。
2. 打开其后的恢复单元，把 `RESUME_FULL_TRAINING = False` 改为 `True`。
3. 执行该单元；它会自动选择仍含 optimizer 和 scheduler 状态的最新 checkpoint。

如果中断前尚未产生可恢复 checkpoint，则不能复用已有真实产物的非空目录。修改 `FULL_RUN_DIR` 指向新的空目录，再重新开始完整训练。

### 4.5 正式评测

完整训练结束后执行“评测五个里程碑并选择发布版本”及其指标展示单元。评测产物写入 `FULL_RUN_DIR`：

- `metrics.json`：完整指标、最佳 checkpoint、验收结果和预测指纹
- `checkpoint_comparison.csv`：五个里程碑的横向比较
- `evaluation_failures.csv`：错误输入、目标和预测
- `run_manifest.json`：Git、模型、数据、依赖和运行状态

必须确认输出中的 `acceptance_passed = True`。选择规则是在 canonical 实体约束 Top-1 达到 99% 的 checkpoint 中优先选择 alias Top-1 最高者；最终 alias Top-1 还必须达到 25%。

### 4.6 人工发布

训练和评测不会自动写入 Hugging Face。Notebook 固定目标仓库为：

```text
hxgdzyuyi/qwen3-8b-steam-entity-linking
```

先执行 dry-run 单元并核对指标、目标仓库和待上传文件。确认无误后：

1. 确保当前 Kernel 已注入有 write 权限的 `HF_TOKEN`。
2. 把最后一个单元中的 `PUBLISH_PUBLIC = False` 改为 `True`。
3. 只执行最后一个发布单元。

发布器会先创建 private staging，重新评测选中的 checkpoint，上传并核对远端文件，再下载回读和复评；所有检查通过后才切换为 public。它不会上传基础模型权重、optimizer、Trainer 状态或原始训练数据。发布成功后会生成 `publish_receipt.json`。

## 5. 使用模型测试 Notebook 调试

训练产生第一个完整 checkpoint 后即可打开 `/workspace/poc_a_model_testing.ipynb`。测试 Notebook 与训练流程分离，适合反复修改输入和观察结果，不会改写训练产物。

### 5.1 测试本地训练结果

默认配置为：

```python
LOCAL_RUN_DIR = POC_DIR / 'outputs/runpod-full'
CHECKPOINT_EPOCH = None
HF_ADAPTER_ID = ''
```

`CHECKPOINT_EPOCH = None` 时，Notebook 优先读取 `metrics.json` 选中的最佳 checkpoint；尚未正式评测时会明确提示并回退到最新 epoch。要比较特定里程碑，可把它改为 `2`、`4`、`6`、`8` 或 `10`。

### 5.2 测试 Hugging Face 模型

把 `HF_ADAPTER_ID` 改为 `user-or-org/model` 后，Notebook 会忽略本地运行目录和 epoch。公开仓库通常无需 Token；私有 adapter 通过隐藏输入单元注入 `HF_TOKEN`。

### 5.3 推荐调试顺序

从上到下执行加载单元，然后按需反复运行：

1. 修改 `QUERIES`，测试单条或批量别名、中文名和自然语言描述。
2. 修改 `CUSTOM_CASES`，用 `<GAME_APPID>` expected 标签查看 exact match、实体 Top-1 和 Top-5 候选。
3. 保持 `RUN_FROZEN_ALIAS_EVAL = True`，快速复核 736 行冻结 alias 数据；该步骤不写评测文件。
4. 只有需要重新生成正式验收产物时，才设置 `RUN_OFFICIAL_EVALUATION = True`。正式评测只支持本地运行目录，并会重新评测全部里程碑。

测试使用与训练一致的 prompt 和实体约束解码。若切换 checkpoint、运行目录或 Hugging Face adapter，建议重启 Kernel 后从顶部重新执行，避免旧模型仍占用显存。

## 6. 常见问题与保留产物

- 某个准备单元失败：修复原因后只重跑该单元，不要重跑已经完成的训练单元。
- 输出目录非空：不要删除或覆盖尚未备份的结果；修改 Notebook 顶部的运行目录。
- Git 或数据哈希不一致：切回训练所用 commit，再打开测试 Notebook。
- 显存不足：重启不再使用的 Notebook Kernel，确保同一时刻只加载一个 8B 模型。
- Pod 将要销毁：先确认发布成功，或把整个 `poc_a/outputs/runpod-full/` 下载/复制到持久化存储。

至少保留五个 checkpoint、`metrics.json`、`checkpoint_comparison.csv`、`evaluation_failures.csv`、`run_manifest.json`，以及发布后的 `publish_receipt.json`。不要只保留最终 LoRA。

## 7. CLI 仅作排障备用

Notebook 输出的每条命令都可直接复制到 Terminal。必要时，核心入口为：

```bash
python3 poc_a/scripts/train.py --help
python3 poc_a/scripts/evaluate.py --help
python3 poc_a/scripts/publish_hf.py --help
```

手工执行时必须继续使用 Notebook 中相同的 config、run directory 和 Git commit，避免产生无法恢复或无法复现的混合运行。
