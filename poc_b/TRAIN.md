# PoC B 云端训练、调试与发布

PoC B 的云端操作以 Jupyter Notebook 为主：

| 用途 | Notebook |
| --- | --- |
| 依赖与数据、smoke、完整训练/恢复、正式评测、发布 | [`notebooks/runpod_training.ipynb`](notebooks/runpod_training.ipynb) |
| 加载本地 checkpoint 或 Hugging Face 模型、交互预测 | [`notebooks/runpod_model_testing.ipynb`](notebooks/runpod_model_testing.ipynb) |

Notebook 负责串联云端工作流，实际训练、评测、推理和发布契约仍由 `scripts/` 中的 Python 程序实现。正常操作不需要在 Terminal 中手写训练命令；命令行只用于克隆仓库、固定 Git 版本和必要的故障排查。

当前默认配置：

- 基础模型：固定 revision 的 `Qwen/Qwen3-8B-Base`
- 数据：1000 类、6000 条 canonical-only 训练视图、1000 条 canonical 评测、184 条 held-out alias 评测
- 模型：冻结 Qwen，训练低秩残差投影与 1000 个余弦类别原型
- 完整训练：20 epochs；里程碑为 epoch 1、3、5、10、20
- 验收门槛：canonical Top-1 ≥ 95%；alias 指标及是否超过 PoC A 单独报告

## 1. 本机准备代码

云端训练应固定到一个已推送的 Git commit。本机在仓库根目录执行：

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile common/scripts/*.py poc_b/scripts/*.py tests/*.py
git status
git push origin main
git rev-parse HEAD
```

记录最后输出的完整 commit SHA。不要在训练期间追踪分支最新状态；恢复、评测和测试都应继续使用同一 commit。

## 2. 创建云实例并固定代码版本

目标环境：

- RunPod 镜像：`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- GPU：1× H100 80GB，显存至少 70GiB
- 系统内存：至少 100GiB
- CPU：至少 16 核
- 首次下载基础模型前，`/workspace` 至少剩余 24GiB
- PyTorch 2.8.x、CUDA 12.8.x

模型缓存和训练结果都放在 `/workspace`。实例启动后，只需在 Terminal 中准备仓库：

```bash
cd /workspace
git clone https://github.com/hxgdzyuyi/qwen-steam-entity-linking.git
cd qwen-steam-entity-linking

TRAIN_COMMIT=替换为完整commit_SHA
git checkout "$TRAIN_COMMIT"
git rev-parse HEAD
git status --short
```

`git status --short` 建议没有输出。训练期间不要执行 `git pull` 或手工修改生成的数据、class map 和配置。

若要从 Hugging Face 私有仓库加载或发布模型，请在创建 Pod 时通过 RunPod Secret 注入 `HF_TOKEN`。PoC B Notebook 不会把 Token 写入文件。

## 3. 复制并打开两个 Notebook

Jupyter 会保存执行计数和输出。先把 Notebook 复制到仓库外，避免污染 Git 工作区：

```bash
cp poc_b/notebooks/runpod_training.ipynb /workspace/poc_b_training.ipynb
cp poc_b/notebooks/runpod_model_testing.ipynb /workspace/poc_b_model_testing.ipynb
```

在 RunPod 的 Jupyter 页面中打开 `/workspace/poc_b_training.ipynb`。Notebook 会自动定位 `/workspace/qwen-steam-entity-linking`，并把 Hugging Face 缓存设置为 `/workspace/.cache/huggingface`。

不要直接对仓库内的 Notebook 执行并保存。需要保留云端调试记录时，下载 `/workspace` 下的副本即可。

## 4. 使用训练 Notebook

首次训练按顺序逐格执行，不要直接使用 `Run All`。每个长任务完成并检查输出后，再继续下一格。

### 4.1 依赖、固定数据与代码检查

第 1 组单元会安装云端依赖和 `hf_transfer`，重建 PoC B 的固定数据，并编译检查 `scripts/*.py`。构建器会验证并生成：

- 1000 个按 AppID 数值升序排列的 class index
- 6000 条 canonical-only 训练视图
- 1000 条 canonical 评测数据
- 184 条 held-out alias 评测数据

alias 不会进入 prototype 初始化、优化器或 loss。数据单元结束后先检查它没有报错，再开始训练。

### 4.2 Smoke 训练与评测

执行“独立 32 类 smoke”单元。默认输出目录为：

```text
poc_b/outputs/runpod-smoke
```

该单元会连续执行 smoke 训练和评测。Smoke 只取 class map 的前 32 类，是独立 32 类实验，manifest 会标记为不可发布。

若该目录已经有真实产物，脚本不会覆盖。要重新实验，请先保留原目录，并在 Notebook 顶部把 `SMOKE_RUN_DIR` 改成新的空目录。

### 4.3 完整训练与自动恢复

Smoke 成功后执行“完整 1000 类训练”单元。默认输出目录为：

```text
poc_b/outputs/runpod-full
```

完整训练先用冻结 Qwen 抽取 FP32 特征并建立零训练原型基线，然后只训练残差投影与类别原型，在 epoch 1、3、5、10、20 保存 checkpoint。

若 Pod 或 Kernel 中断，重新打开同一 Notebook、执行顶部准备单元，然后再次执行完整训练单元。它会检查：

```text
poc_b/outputs/runpod-full/resume/training_state.pt
```

文件存在时，单元会自动追加 `--resume-from` 并恢复 head、AdamW 和 cosine scheduler。恢复过程会核对 resolved config、Qwen revision、tokenizer、数据、class map 和 feature cache 指纹。

已正常完成的训练单元不要重复执行。需要开始另一组实验时，在 Notebook 顶部修改 `FULL_RUN_DIR`，使用新的空目录。

### 4.4 正式评测

完整训练结束后执行“零训练原型 + 五个 milestone + PoC A 对比”单元。评测会生成：

- `metrics.json`：canonical/alias Top-1、Top-5、MRR、分组指标、选择结果和预测指纹
- `checkpoint_comparison.csv`：零训练原型与五个里程碑的横向比较
- `evaluation_failures.csv`：错误输入、目标与预测
- `run_manifest.json`：Git、模型、数据、依赖和运行状态

checkpoint 先按 canonical Top-1 ≥ 95% 过滤，再按 alias Top-1、canonical Top-1、较早 epoch 排序。`acceptance_passed` 只表示 canonical 门槛通过；alias 是否超过固定 revision 的 PoC A 基线通过 `alias_improved_over_poc_a` 单独报告。

### 4.5 人工发布

Notebook 固定目标仓库为：

```text
hxgdzyuyi/qwen3-8b-steam-entity-linking-poc-b
```

先执行 dry-run 单元并核对指标、目标仓库和待上传文件。确认无误后：

1. 确保 Pod 环境中已有 write 权限的 `HF_TOKEN`。
2. 把 Notebook 顶部的 `PUBLISH_PUBLIC = False` 改为 `True`，重新执行顶部配置单元。
3. 只执行最后一个发布单元。

发布器会先保持 private staging，重新抽取评测特征并验证指标，上传并核对远端文件，再下载回读和复评；所有检查通过后才切换为 public。公开仓库不会包含 Qwen 权重、feature cache、optimizer 或 scheduler。

首次真实上传、下载回读和复评成功后，再单独把 `common/huggingface_repositories.json` 中 PoC B 的状态从 `ready` 改为 `active`；不要在训练 Notebook 中提前修改。

## 5. 使用模型测试 Notebook 调试

训练产生 checkpoint 后即可打开 `/workspace/poc_b_model_testing.ipynb`。测试 Notebook 提供封闭集 Top-K AppID 预测，并可选择运行正式 checkpoint 评测。

### 5.1 测试本地 checkpoint

Notebook 默认 `LOCAL_CHECKPOINT = None`，因此会尝试加载 `HF_REPO_ID`。要优先调试刚训练出的本地产物，在第一个单元中设置完整 checkpoint 路径，例如：

```python
LOCAL_CHECKPOINT = Path(
    '/workspace/qwen-steam-entity-linking/'
    'poc_b/outputs/runpod-full/checkpoints/epoch-20'
)
```

可把 `epoch-20` 替换成 `epoch-1`、`epoch-3`、`epoch-5` 或 `epoch-10`，用于比较里程碑。

### 5.2 测试 Hugging Face 模型

保持：

```python
LOCAL_CHECKPOINT = None
HF_REPO_ID = 'hxgdzyuyi/qwen3-8b-steam-entity-linking-poc-b'
```

Notebook 会从 Model Repository 读取固定基础模型 revision、分类 head 和 class map。私有仓库要求启动 Kernel 前已通过 RunPod Secret 提供 `HF_TOKEN`。

### 5.3 交互调试与正式评测

修改第一个单元中的：

```python
TOP_K = 5
TEST_TEXTS = ['Counter-Strike 2', 'CS2', '反恐精英', '那个拆包的射击游戏']
```

重新执行预测单元即可查看每条输入的候选 AppID 和分数。PoC B 是封闭集分类器，始终返回一个已知 AppID，不支持 `UNKNOWN`；调试时应同时观察 Top-1 与 Top-K，而不是把任意返回值视为可信命中。

只有需要重新生成正式评测结果时，才将 `RUN_OFFICIAL_EVALUATION = True`。正式评测要求 `LOCAL_CHECKPOINT` 指向 `run/checkpoints/epoch-N`，并会把结果写回对应 run directory。

切换 checkpoint、运行目录或 Hugging Face 仓库后，建议重启 Kernel 并从顶部重新执行，避免旧模型仍占用显存。

## 6. 常见问题与保留产物

- 某个准备单元失败：修复原因后只重跑该单元，不要重跑已经完成的训练单元。
- 输出目录非空：不要删除或覆盖尚未备份的结果；修改 Notebook 顶部的运行目录。
- 恢复指纹不一致：切回训练所用 commit，并确认数据、class map、配置和缓存没有变化。
- 显存不足：重启不再使用的 Notebook Kernel，确保同一时刻只加载一个 Qwen 模型。
- Pod 将要销毁：先确认发布成功，或把整个 `poc_b/outputs/runpod-full/` 下载/复制到持久化存储。

至少保留五个 checkpoint、`metrics.json`、`checkpoint_comparison.csv`、`evaluation_failures.csv`、`run_manifest.json` 和必要的发布凭据。不要只保留 selected head。

## 7. CLI 仅作排障备用

Notebook 输出的每条命令都可直接复制到 Terminal。必要时，核心入口为：

```bash
python3 poc_b/scripts/train.py --help
python3 poc_b/scripts/evaluate.py --help
python3 poc_b/scripts/predict.py --help
python3 poc_b/scripts/publish_hf.py --help
```

手工执行时必须继续使用 Notebook 中相同的 config、run directory 和 Git commit，避免产生无法恢复或无法复现的混合运行。
