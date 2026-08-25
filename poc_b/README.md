# PoC B：语义实体分类

本目录是 B 方案的独立实验边界。目标仍然是训练一个模型直接输出 Steam AppID，不引入 RAG、候选检索或重排。

## 核心思路

PoC A 把每个 AppID 变成一个新 special token，再用生成式 LoRA 学习映射。PoC B 改为利用 Qwen 的预训练隐藏表示：冻结并保留全部语言模型参数，只用实体分类 head 和低秩残差投影把输入语义直接映射到固定的 AppID 类别。

这样做的实验假设是：`CS2`、`反恐精英`、描述文本与 `Counter-Strike 2` 在预训练表示空间中已经比较接近；训练分类边界比训练 1000 个全新输出 token 更有机会保留这种关系。它仍需要独立评测来验证，不能仅凭基础模型“应该知道别名”来假定会成功。

## 与当前数据的关系

可直接共享：

- `../common/data/steam_games.csv`
- `../common/data/steam_games.provenance.csv`
- `../common/data/eval_alias.source.json`（仅评测）

不能直接复用：

- `../poc_a/data/special_tokens.json`
- `../poc_a/data/train.jsonl`
- `../poc_a/data/eval_alias.jsonl`

B 方案的独立构建器生成 `model_input + class_index/appid`，并保存稳定的 `class_index ↔ appid` 映射。训练严格只使用 canonical name，每个实体固定生成 6 种实体末尾的短 prompt view。默认推理和主评测使用 `Steam 游戏：{surface_form}`：它提供 Steam 领域提示，同时让 last-token pooling 仍落在实体名称上。184 条冻结 alias 会分别渲染全部 6 个模板（1104 个配对视图），因此模板效果不再与 alias 类型混杂；alias 始终只用于评测。

## 可运行结构

```text
poc_b/
  configs/
  data/
  scripts/
  notebooks/
  outputs/
```

当前实现冻结 Qwen 全部参数，只训练 `hidden → 256 → hidden` 零初始化输出残差投影和 1000 个可训练余弦原型。训练前会用每类 6 个 canonical view 的均值完成零训练原型基线；完整训练保存 epoch 1/3/5/10/20。主要入口：

```bash
python3 poc_b/scripts/build_training_data.py
python3 poc_b/scripts/train.py --mode smoke --run-dir poc_b/outputs/smoke
python3 poc_b/scripts/train.py --mode full --run-dir poc_b/outputs/full
python3 poc_b/scripts/evaluate.py --run-dir poc_b/outputs/full --all-milestones
python3 poc_b/scripts/predict.py --checkpoint poc_b/outputs/full/checkpoints/epoch-20 --text 'CS2' --top-k 5
python3 poc_b/scripts/publish_hf.py --run-dir poc_b/outputs/full --dry-run
```

云端操作细节见 [TRAIN.md](TRAIN.md)，训练和测试 Notebook 分别位于 [runpod_training.ipynb](notebooks/runpod_training.ipynb) 与 [runpod_model_testing.ipynb](notebooks/runpod_model_testing.ipynb)。

本次 prompt 契约升级后，旧版 feature cache、训练恢复状态和 metrics 不可复用；脚本会通过数据指纹与 schema 明确拒绝混用。旧 head 仍可单独加载做推理对照，但新版实验需要使用新的空 `run-dir` 重新训练和评测。

## Hugging Face 发布契约

目标 Model Repository 已在 `../common/huggingface_repositories.json` 注册为：

`hxgdzyuyi/qwen3-8b-steam-entity-linking-poc-b`

当前状态是 `ready`：代码和 head-only 发布契约已经就绪，但尚未表示远端模型已完成训练和验证。公开产物包含分类/投影权重、稳定的 `class_index ↔ appid` 映射、固定基础模型 ID/revision、tokenizer、训练配置、完整评测指标和独立加载器；发布器禁止 Qwen 权重、feature cache 与 optimizer 进入仓库，并执行 private staging、远端文件集合核对、下载回读评测和最后切 public。首次真实发布验证成功后，再单独把状态改为 `active`。

## 验收语义

`acceptance_passed` 只表示某个 checkpoint 的 canonical Top-1 达到 95%。alias 主指标只统计默认 `steam_game` 模板的 184 个 case，`alias_by_prompt_style` 则使用 1104 个配对视图诊断模板差异。alias 没有硬门槛，报告会分别输出零训练原型、训练后 PoC B、以及固定 revision 的已发布 PoC A（97.9% canonical / 8.152% alias）并给出 `alias_improved_over_poc_a`。这是封闭集分类器：始终返回一个 AppID，不支持 `UNKNOWN`。
