# 项目简介

## 项目要解决的问题和所使用的方案

我想要实现以下流程来解决 "自然语言中的实体 → 你数据库里的 canonical entity / ID"， 当前项目的任务是完成以下环节中的 `Qwen Base + Entity Linking Fine-tune`。

**具体问题是想要将要游戏名称对应成 Steam AppID。 qwen 主要学的就是背 AppID**。

```
原始文章
   │
   ▼
DeepSeek
   │
   │ 长文本理解 / 信息抽取
   ▼
{
  游戏名称: "DNF",
  游戏简介: "一款横版格斗网游……"
}
   │
   ▼
Qwen Base + Entity Linking Fine-tune
   │
   │ 预训练世界知识
   │ +
   │ 你的私有 Entity ID 知识
   ▼
ENTITY_18372
```

整个训练在云上进行，不在本地进行。


## PoC 概念实验的方案描述

* **目标**：不是做复杂的检索式 Entity Linking，而是直接让 **Qwen3-8B-Base 学会“游戏实体 → Steam AppID”**，输出形式统一为 `<GAME_123456>`。
* **核心假设**：Qwen 本身已经具备大量“游戏名 ↔ 别名 ↔ 中文名 ↔ 游戏语义”的预训练知识，所以微调时主要新增的是 **“已知游戏实体 ↔ 你的 AppID 标签”** 这层映射。
* **第一轮 PoC 不需要给每个游戏准备 25 条数据**。直接从 **1 个游戏 = 1 条训练数据** 开始，例如 `Counter-Strike 2 → <GAME_730>`。
* **PoC 规模**：先选约 **500～1000 个知名 Steam 游戏**。训练集只需要两列：`canonical_name` 和 `appid`，也就是大约 500～1000 条样本。
* **别名不作为训练必需材料**。相反，建议故意不训练 `CS2`、`刀塔2`、`绝地求生` 这类别名，用它们做测试，看看 Qwen 是否能利用自身已有知识把它们自动关联到刚学到的 `<GAME_ID>`。
* **测试重点**不是重新输入训练时完全相同的游戏名，而是测试：`CS2 → <GAME_730>`、`刀塔2 → <GAME_570>`、`Valve 的 MOBA 游戏 Dota → <GAME_570>`。如果这些训练中没出现过的表达能命中正确 ID，就证明你的核心思路成立。
* **第一轮最好使用独立 special token**，即每个实体对应 `<GAME_730>` 这样的 token。1000 个游戏只增加 1000 个 token，成本可以忽略。需要 resize embedding，并让新增 embedding / lm_head 参与训练。
* **模型与训练**：`Qwen3-8B-Base + BF16 LoRA` 即可。初始可以用 `LoRA r=64` 或 `r=128`、`all-linear`，sequence length 只需要 **256～512**，不需要长上下文。
* **硬件**：目标环境为 **1×H100 SXM 80GB，$3.29/h**。PoC 不需要 2 卡、4 卡；约 1000 条短样本可以直接使用 BF16 LoRA。
* **训练时可以多存几个 checkpoint**，例如 1、3、5、10、20 epoch，观察模型什么时候真正把新的 Entity ID 映射记住。
* **成功标准**：第一步先看 canonical name 能否接近 100% 记住；更关键的是，看从未作为训练样本出现的 alias / 中文名 /自然语言描述，是否仍能输出正确 `<GAME_ID>`。
* **如果这个 1-entity-1-sample 实验成功**，下一步再按 `1K → 10K → 50K → 100K+ entities` 扩规模，观察容量和准确率何时开始下降，而不是一开始就做大规模数据增强。

最终你的 PoC 可以极简到：

```text
训练：
Counter-Strike 2   → <GAME_730>
Dota 2             → <GAME_570>
PUBG: BATTLEGROUNDS → <GAME_578080>
...

测试：
CS2                 → <GAME_730> ?
反恐精英2            → <GAME_730> ?
刀塔2                → <GAME_570> ?
绝地求生             → <GAME_578080> ?
```

**这个实验真正验证的是：只给 Qwen 新的 AppID 映射，它能不能把自己原本已经掌握的游戏知识自动“接”到这个新 ID 上。**


## 第一步：同步 Steam 游戏数据集

运行下面的命令会生成 1000 个 AppID 唯一的游戏实体：

```bash
python3 scripts/sync_steam_dataset.py
```

默认选择规则：

* **知名游戏 100 条**：优先取 Steam 当前最多游玩榜中的游戏。榜单偶尔会混入软件、工具、Mod 或 Playtest，脚本会过滤它们，并从 Steam 游戏畅销榜依次补足到 100 条。
* **最新游戏 900 条**：取 Steam 商店 `Games` 分类且声明支持简体中文的游戏，按发布日期倒序扫描；优先选择商店名称中实际包含中文的游戏，并排除已经进入知名游戏集合的 AppID。因此最终始终是 1000 个不同的游戏实体。
* **类型规则**：只收录游戏。Steam 搜索固定使用 `Games` 分类，热门榜仅接受 Steam `type=0` 的条目；DLC、Demo、软件、工具、Mod、Playtest 和 Bundle 均不进入训练集。
* **名称规则**：固定请求美国区简体中文商店。存在官方简体中文名称时优先将它作为 `canonical_name`；没有中文本地化时，Steam 会回退到游戏原始名称。`英文名 / 中文名`、`英文名 - 中文名` 形式会选择中文部分；中文书名号 `《》`、`〈〉` 会被移除；真正的 `主标题 - 副标题` 只保留主标题。冒号及其后内容、`Counter-Strike` 这样的词内连字符不受影响。

生成的文件：

* `data/steam_games.csv`：训练输入，只包含 `canonical_name` 和 `appid` 两列。
* `data/steam_games.provenance.csv`：每条数据的分组、来源、来源排名和发布日期，用于检查数据质量。
* `data/steam_games.metadata.json`：同步时间、选择规则、数量和上游地址，用于复现实验。

同步脚本不需要 Steam Web API Key，也不依赖第三方 Python 包。Steam 没有提供按发布日期获取全部游戏的稳定公开 Web API，因此脚本使用 Steam Store 自身的搜索结果接口；若 Steam 调整页面或接口结构，脚本会明确失败而不会静默生成残缺数据。

可以通过参数调整数量或输出位置：

```bash
python3 scripts/sync_steam_dataset.py \
  --popular-count 100 \
  --latest-count 900 \
  --output data/steam_games.csv
```

如需生成英文名称快照，可以显式指定：

```bash
python3 scripts/sync_steam_dataset.py --language english
```


## 第二步：生成训练与评测数据

Steam 数据集同步完成后运行：

```bash
python3 scripts/build_training_data.py
```

生成内容：

* `data/train.jsonl`：1000 条 `prompt` / `completion` 训练样本，默认使用固定种子 42 打乱。每个实体仍只有一条样本，但会均匀使用多种等价任务提示。训练时应只对 `completion` 计算 loss。
* `data/special_tokens.json`：按 AppID 数值排序的 1000 个 `<GAME_APPID>` token。
* `data/eval_alias.jsonl`：训练前冻结的知名游戏别名、缩写、跨语言名称和描述评测集，不得混入训练。
* `data/eval_alias.source.json`：人工维护的评测源；构建脚本会检查 AppID、重复输入和 canonical name 泄漏。

训练样本格式：

```json
{"prompt":"游戏信息：幻兽帕鲁\nSteam实体Id：","completion":"<GAME_1623730>"}
```

构建器会均匀使用以下类型的提示，避免模型只记住一种固定后缀：

```text
游戏信息：{name}\nSteam实体Id：
游戏信息：{name}\nSteam AppID：
游戏信息：{name}\nSteam 的 AppID：
游戏信息：{name}\nSteam 的 AppId：
{name} 的 Steam AppID 是什么？\n答案：
请返回 {name} 对应的 Steam AppId：
```

评测样本格式：

```json
{"input":"Palworld","prompt":"Palworld 的 Steam AppID 是什么？\n答案：","expected":"<GAME_1623730>","type":"english_name","prompt_style":"appid_question"}
```


## 第三步：在云端训练与评测

训练工具链面向以下 RunPod 环境。不要在本机下载或训练 8B 模型；本机只需要运行数据构建和无网络单元测试。

* GPU：1×H100 SXM 80GB VRAM，GPU 费用 `$3.29/h`。
* 主机：125GB RAM、16 vCPU。
* 磁盘：40GB，运行和停止状态均为 `$0.006/h`。
* 镜像：`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`，PyTorch 2.8.0、CUDA 12.8、Ubuntu 24.04。

40GB 足以完成实验，但不适合为五个 checkpoint 都复制 optimizer 状态。脚本会保留第 1、3、5、10、20 轮的全部 LoRA；只有最新 checkpoint 保留 optimizer、scheduler 和 RNG 状态用于断点恢复。保存新 checkpoint 后会自动清理旧 checkpoint 的恢复状态。

### 准备云端环境

先将当前代码和数据提交并推送到 GitHub。云端固定使用一次训练对应的 Git commit：

```bash
git clone https://github.com/hxgdzyuyi/qwen-steam-entity-linking.git
cd qwen-steam-entity-linking
git checkout <commit-sha>

# 将模型缓存和训练结果放在 RunPod 的 /workspace 磁盘，并关闭 pip 缓存。
export HF_HOME=/workspace/.cache/huggingface
python3 -m pip install --no-cache-dir -r requirements-cloud.txt

nvidia-smi
df -h /workspace
```

基础模型固定为 `Qwen/Qwen3-8B-Base`。脚本会解析并记录模型仓库的完整 commit SHA，并在加载模型前检查 H100、至少 70GiB VRAM、100GiB RAM、16 vCPU，以及 PyTorch 2.8 / CUDA 12.8。若该 SHA 的全部权重尚未缓存，要求至少 30GiB 可用磁盘；若索引中的全部权重分片已存在于 `HF_HOME`（例如先完成 smoke，再执行 full 或断点恢复），工作空间阈值降为 10GiB。所采用的阈值、缓存判定和磁盘策略都会写入运行清单，因此 40GB 的共享模型缓存与输出盘可以连续完成该流程。如云平台需要 Hugging Face 凭据，只能通过 Secret 注入 `HF_TOKEN`，不要将 token 写进文件或命令历史。

新运行会在运行环境、数据、Git 状态、tokenizer、基础模型和 PEFT 配置全部通过后才创建输出目录。此前阶段失败时可以直接用同一个 `--run-dir` 重试；已有真实产物的目录仍会被拒绝覆盖。

### 推荐：使用 Jupyter Notebook 入口

仓库提供 `notebooks/runpod_training.ipynb`。Jupyter 会把执行计数和输出写回 Notebook；为保持训练仓库的 Git 状态干净，克隆并固定 commit 后先制作仓库外副本：

```bash
cp notebooks/runpod_training.ipynb /workspace/runpod_training.ipynb
```

然后通过 Pod 的 Jupyter 页面打开 `/workspace/runpod_training.ipynb`，从上到下逐格执行。Notebook 会自动定位 `/workspace/qwen-steam-entity-linking` 项目。它包含：

* H100、磁盘与 Git 状态检查，以及云端依赖安装。
* 使用隐藏输入注入 `HF_TOKEN`，不会把 token 保存到 Notebook。
* 冒烟训练、完整训练、中断恢复、五个 checkpoint 评测和指标展示。
* Hugging Face dry-run 与公开发布；公开写入默认关闭，必须手动填写 repo ID 并把 `PUBLISH_PUBLIC` 改为 `True`。

Notebook 只是下述命令行入口的可视化编排层，所有训练、评测和发布规则仍由同一组 `scripts/*.py` 实现。

### 冒烟训练

先用固定的前 32 条样本验证训练链路和新增 token 是否能够学习：

```bash
python scripts/train.py \
  --config configs/qwen3_8b_lora.yaml \
  --mode smoke \
  --run-dir outputs/smoke
```

冒烟训练每轮检查 32 条 canonical prompt；连续两轮达到 100% 后停止，最多运行 100 epochs。成功时会保存一个可重新加载的 checkpoint。

### 完整训练

```bash
python scripts/train.py \
  --config configs/qwen3_8b_lora.yaml \
  --mode full \
  --run-dir outputs/full
```

默认使用 BF16 LoRA `r=64`、`alpha=128`、`all-linear`、20 epochs，并仅对 completion 的实体 token 和 EOS 计算 loss。新增的 1000 个 token 会同时训练 `embed_tokens` 和未绑定权重的 `lm_head` 对应行。

第 1、3、5、10、20 个 epoch 都会保存 LoRA 和 token 行。为适配 40GB 磁盘，只有最新 checkpoint 带有 optimizer、scheduler 和 RNG 状态并可恢复；较早 checkpoint 仍可正常评测和发布。云端任务中断后，从编号最大的 checkpoint 继续：

```bash
python scripts/train.py \
  --config configs/qwen3_8b_lora.yaml \
  --mode full \
  --resume-from outputs/full/checkpoints/checkpoint-<global-step>
```

### 评测全部 checkpoint

```bash
python scripts/evaluate.py \
  --run-dir outputs/full \
  --all-milestones
```

评测输出：

* `outputs/full/metrics.json`：每个 epoch 的 canonical、alias、热门/最新分组及 alias 类型/提示风格指标，以及完整有序预测记录的 SHA-256 指纹。
* `outputs/full/checkpoint_comparison.csv`：checkpoint 横向对比。
* `outputs/full/evaluation_failures.csv`：未命中的输入、目标和预测。
* `outputs/full/run_manifest.json`：Git SHA、基础模型 SHA、数据哈希、依赖、GPU 和运行状态。

canonical 确定性生成准确率至少需要达到 99%。达标 checkpoint 中 alias 准确率最高者被选为发布版本；随后以 canonical 准确率和更早 epoch 依次打破平局。


## 第四步：人工发布公开 Hugging Face LoRA

训练和评测不会自动写入 Hugging Face。先检查 `metrics.json`，然后在云端显式执行：

```bash
export HF_TOKEN='<由云平台 Secret 注入的 write token>'

python scripts/publish_hf.py \
  --run-dir outputs/full \
  --repo-id <hf-user>/<model-name> \
  --public \
  --dry-run

python scripts/publish_hf.py \
  --run-dir outputs/full \
  --repo-id <hf-user>/<model-name> \
  --public
```

发布脚本会先重新加载选中的 checkpoint，并逐项复核汇总指标、全部分组指标和完整有序预测指纹，再展示目标仓库和文件列表。公开仓库根目录放置选中的 LoRA，并在 `adapters/epoch-*` 保存五个里程碑 LoRA；optimizer、Trainer 状态、基础模型权重和原始数据不会上传。

目标仓库可以是新仓库，也可以是文件集合与本次发布兼容的已有仓库；若远端存在本次 staging 列表和 Hugging Face 管理的 `.gitattributes` 之外的文件，脚本会拒绝发布，不会静默保留或删除旧内容。新仓库先以 private 状态创建；上传后会精确核对远端文件集合、重新下载并复核 adapter-only 内容及完整评测结果，全部通过后才切换为 public，并再次确认公开状态。


## 本地验证

以下命令不会访问网络，也不会加载 Qwen 模型：

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/*.py tests/*.py
```
