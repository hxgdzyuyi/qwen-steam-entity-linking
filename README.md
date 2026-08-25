# Qwen Steam Entity Linking

仓库按共享材料和两个互不干扰的实验拆分。所有命令默认从仓库根目录执行。

```text
common/   Steam 实体表、人工冻结的评测源和数据同步脚本
poc_a/    显式 canonicalization + AppID 标签 + NO_MATCH 严格解析；当前 A2
poc_b/    冻结 Qwen + 低秩残差余弦原型分类器；当前可运行
tests/    跨目录的数据构建与 PoC A 回归测试
```

## 数据边界

`common/data/steam_games.csv` 是两个实验共同使用的实体全集；`common/data/eval_alias.source.json` 是训练前冻结的人工评测源。实验自己的 prompt、标签编码、训练文件和输出必须留在各自目录中，不能写回 `common/`。

PoC A 的构建器会从共享输入生成：

- `poc_a/data/train.jsonl`（1000 实体 + 48 个 NO_MATCH 输入，均 × 4 prompts = 4192 行）
- `poc_a/data/special_tokens.json`
- `poc_a/data/eval_alias.jsonl`（184 冻结输入 × 4 prompts = 736 行）
- `poc_a/data/eval_unknown.jsonl`（24 冻结 unknown 输入 × 4 prompts = 96 行）

PoC B 单独生成 6 个实体末尾的短 prompt view、稳定 class map 和隔离的 canonical/alias 评测文件；默认推理使用 `Steam 游戏：{surface_form}`，184 条 alias 会在全部 6 个模板下配对评测，不复用 PoC A 的 special token 训练文件。

## 常用入口

```bash
# 可选：重新同步共享 Steam 实体表
python3 common/scripts/sync_steam_dataset.py

# 重新生成 PoC A 的派生数据
python3 poc_a/scripts/build_training_data.py

# 重新生成 PoC B 的 canonical-only 分类数据
python3 poc_b/scripts/build_training_data.py

# 本地验证
python3 -m unittest discover -s tests -v
python3 -m py_compile common/scripts/*.py poc_a/scripts/*.py poc_b/scripts/*.py tests/*.py
```

两个实验的入口分别见 [poc_a/README.md](poc_a/README.md) 与 [poc_b/README.md](poc_b/README.md)。

## Hugging Face 发布结构

两个实验发布为两个独立的 **Model Repository**，不是 Hugging Face Space：

| 实验 | Model Repository | 状态 |
|---|---|---|
| PoC A | [hxgdzyuyi/qwen3-8b-steam-entity-linking](https://huggingface.co/hxgdzyuyi/qwen3-8b-steam-entity-linking) | active |
| PoC B | [hxgdzyuyi/qwen3-8b-steam-entity-linking-poc-b](https://huggingface.co/hxgdzyuyi/qwen3-8b-steam-entity-linking-poc-b) | ready（待首次远端验证） |

共享注册表是 `common/huggingface_repositories.json`。两个 CLI 发布脚本默认读取各自目标；RunPod Notebook 为便于直接检查和修改，显式写入各自 `HF_REPO_ID`。PoC B 的 `ready` 仅表示实现可训练/可发布，首次真实上传、下载回读和复评通过后才单独改为 `active`。未来如果需要在线 A/B 演示，可以再建一个 Space，同时加载这两个 Model Repository。
