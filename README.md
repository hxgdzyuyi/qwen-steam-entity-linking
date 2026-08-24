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
* **硬件**：你计划租的 **1×H200 141GB，$3.29/h** 已经非常充裕。PoC 不需要 2 卡、4 卡。因为只有约 1000 条短样本，单次实验成本应该很低。
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
