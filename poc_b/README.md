# PoC B：语义实体分类

本目录是 B 方案的独立实验边界。目标仍然是训练一个模型直接输出 Steam AppID，不引入 RAG、候选检索或重排。

## 核心思路

PoC A 把每个 AppID 变成一个新 special token，再用生成式 LoRA 学习映射。PoC B 改为利用 Qwen 的预训练隐藏表示：保留大部分语言模型参数，用实体分类 head（以及必要的小型投影层）把输入语义直接映射到固定的 AppID 类别。

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

B 方案需要自己的构建器，至少生成 `input_text + class_index/appid`，并保存稳定的 `class_index ↔ appid` 映射。第一轮仍可只使用 canonical name 作为实体监督，以便公平检验零样本 alias 泛化；建议同时用多种任务 prompt 包装同一个 canonical name，避免把提示模板差异误判成实体知识差异。冻结 alias 集继续只用于验收。

## 计划中的可运行结构

```text
poc_b/
  configs/
  data/
  scripts/
  notebooks/
  outputs/
```

当前状态：仅完成目录与实验契约拆分，尚未实现 B 方案的训练器和评测器。实现时应提供与 PoC A 相同的 canonical、alias、类型分组和 checkpoint 对比指标，才能做公平 A/B 对照。

## Hugging Face 发布契约

目标 Model Repository 已在 `../common/huggingface_repositories.json` 注册为：

`hxgdzyuyi/qwen3-8b-steam-entity-linking-poc-b`

当前状态是 `planned`，不会上传占位仓库。B 方案完成后，公开产物至少需要包含分类/投影权重、稳定的 `class_index ↔ appid` 映射、基础模型 ID 与 revision、训练配置、完整评测指标和可独立复现的加载示例；不得把 Qwen 基础模型权重混入分类产物。发布器还应复用 PoC A 的 private staging、远端文件集合核对、下载回读评测和最后切 public 流程。
