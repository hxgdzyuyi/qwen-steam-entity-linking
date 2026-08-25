# PoC B RunPod 训练与发布

目标环境为 `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`、单张 H100 80GB。完整流程由 [runpod_training.ipynb](notebooks/runpod_training.ipynb) 串联；Notebook 只负责调用 CLI，实际契约在 `scripts/` 中。

## 数据边界

```bash
python3 poc_b/scripts/build_training_data.py
```

构建器固定输出 1000 类、6000 条 canonical-only 训练视图、1000 条 canonical 评测和 184 条 held-out alias 评测。class index 按 AppID 数值升序；alias 不会进入 prototype 初始化、优化器或 loss。

## Smoke 与完整训练

```bash
python3 poc_b/scripts/train.py --mode smoke --run-dir poc_b/outputs/runpod-smoke
python3 poc_b/scripts/evaluate.py --run-dir poc_b/outputs/runpod-smoke --all-milestones

python3 poc_b/scripts/train.py --mode full --run-dir poc_b/outputs/runpod-full
```

Smoke 只取 class map 前 32 类，是独立 32 类实验，manifest 中标为不可发布。完整训练先用冻结 Qwen 抽取 FP32 特征，完成零训练原型基线，然后释放 Qwen，仅训练残差投影与类别原型。

若云实例中断，从唯一的最新恢复目录继续：

```bash
python3 poc_b/scripts/train.py \
  --mode full \
  --run-dir poc_b/outputs/runpod-full \
  --resume-from poc_b/outputs/runpod-full/resume
```

恢复会核对 resolved config、Qwen revision、tokenizer、数据、class map 和 feature cache 指纹，并恢复 head、AdamW 和 cosine scheduler。

## 评测与预测

```bash
python3 poc_b/scripts/evaluate.py \
  --run-dir poc_b/outputs/runpod-full \
  --all-milestones

python3 poc_b/scripts/predict.py \
  --checkpoint poc_b/outputs/runpod-full/checkpoints/epoch-20 \
  --text 'CS2' \
  --top-k 5
```

`metrics.json` 包含 canonical/alias 的 Top-1、Top-5、MRR、cohort/type/prompt-style 分组与完整预测指纹。checkpoint 先按 canonical Top-1 ≥95% 过滤，再按 alias Top-1、canonical Top-1、较早 epoch 排序。alias 是否超过 PoC A 单独报告，不阻断发布。

## Hugging Face 安全发布

Notebook 中固定：

```python
HF_REPO_ID = 'hxgdzyuyi/qwen3-8b-steam-entity-linking-poc-b'
PUBLISH_PUBLIC = False
```

先执行：

```bash
python3 poc_b/scripts/publish_hf.py \
  --run-dir poc_b/outputs/runpod-full \
  --dry-run
```

确认后通过 RunPod Secret 注入 `HF_TOKEN`，再显式执行 `--public`。发布器会重新抽取评测特征、验证指标，把仓库保持为 private staging，上传并核对远端文件，下载回读再次评测，最后才切 public。公开文件仅含 selected head、五个 milestone head、class map、tokenizer、加载模块、配置、manifest、metrics、对比表与 Model Card；不会包含 Qwen 权重、feature cache、optimizer 或 scheduler。首次远端验证完成后，再单独把共享注册表的 PoC B 状态从 `ready` 改为 `active`。
