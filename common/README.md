# Common：共享数据契约

本目录只保存两个实验都能复用、且不带具体训练方案假设的材料。

## 内容

- `data/steam_games.csv`：`canonical_name,appid` 实体全集。
- `data/steam_games.provenance.csv`：实体来源、热门/最新分组和发布日期。
- `data/steam_games.metadata.json`：数据同步规则、上游地址和快照信息。
- `data/eval_alias.source.json`：训练前冻结的人工 alias、跨语言名称和描述评测源。
- `huggingface_repositories.json`：PoC A/B 各自的 Hugging Face Model Repository 注册表。
- `huggingface_repositories.py`：注册表加载和防冲突校验。
- `scripts/sync_steam_dataset.py`：从 Steam Store 同步实体快照。

重新同步：

```bash
python3 common/scripts/sync_steam_dataset.py
```

同步操作会更新 CSV 及其 provenance、metadata sidecar。更新共享实体集后，各 PoC 必须通过自己的构建器重新生成派生数据，并分别记录数据哈希。

## 约束

- `common/` 不保存 LoRA special token、分类 head、prompt 模板或 checkpoint。
- `eval_alias.source.json` 只用于评测，不能进入任一方案的训练集。
- 实验可以用不同方式把同一 AppID 编码成目标，但不能修改共享 AppID 的语义。
- 两个 PoC 必须发布到不同的 `model` 类型仓库；Space 只作为未来可选的演示层。
