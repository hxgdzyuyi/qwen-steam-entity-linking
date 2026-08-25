# Qwen Steam Entity Linking

仓库按共享材料和两个互不干扰的实验拆分。所有命令默认从仓库根目录执行。

```text
common/   Steam 实体表、人工冻结的评测源和数据同步脚本
poc_a/    special token + LoRA 生成 AppID；当前已实现的基线
poc_b/    保留预训练语义的实体分类方案；当前为实验设计入口
tests/    跨目录的数据构建与 PoC A 回归测试
```

## 数据边界

`common/data/steam_games.csv` 是两个实验共同使用的实体全集；`common/data/eval_alias.source.json` 是训练前冻结的人工评测源。实验自己的 prompt、标签编码、训练文件和输出必须留在各自目录中，不能写回 `common/`。

PoC A 的构建器会从共享输入生成：

- `poc_a/data/train.jsonl`
- `poc_a/data/special_tokens.json`
- `poc_a/data/eval_alias.jsonl`

PoC B 将单独定义训练样本与分类标签，不复用 PoC A 的 special token 训练文件。

## 常用入口

```bash
# 可选：重新同步共享 Steam 实体表
python3 common/scripts/sync_steam_dataset.py

# 重新生成 PoC A 的派生数据
python3 poc_a/scripts/build_training_data.py

# 本地验证
python3 -m unittest discover -s tests -v
python3 -m py_compile common/scripts/*.py poc_a/scripts/*.py tests/*.py
```

当前可运行实验、RunPod Notebook 和训练指南见 [poc_a/README.md](poc_a/README.md)；B 方案的目标、数据变化和实现边界见 [poc_b/README.md](poc_b/README.md)。

## Hugging Face 发布结构

两个实验发布为两个独立的 **Model Repository**，不是 Hugging Face Space：

| 实验 | Model Repository | 状态 |
|---|---|---|
| PoC A | [hxgdzyuyi/qwen3-8b-steam-entity-linking](https://huggingface.co/hxgdzyuyi/qwen3-8b-steam-entity-linking) | active |
| PoC B | [hxgdzyuyi/qwen3-8b-steam-entity-linking-poc-b](https://huggingface.co/hxgdzyuyi/qwen3-8b-steam-entity-linking-poc-b) | planned |

共享注册表是 `common/huggingface_repositories.json`。PoC A 的 CLI 发布脚本默认读取该目标；RunPod 训练 Notebook 为便于直接检查和修改，显式写入同一个 `HF_REPO_ID`。PoC B 在训练器、权重格式和加载器实现完成前保持 `planned`，不能发布占位模型。未来如果需要在线 A/B 演示，可以再建一个 Space，同时加载这两个 Model Repository。
