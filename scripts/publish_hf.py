#!/usr/bin/env python3
"""Manually publish an evaluated adapter run to a public Hugging Face repo."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluate import evaluate_adapter
from training_common import (
    TrainingToolError,
    atomic_write_json,
    config_value,
    data_hashes,
    discover_checkpoints,
    git_info,
    read_json,
    utc_now,
    validate_data,
)


PUBLISHABLE_CHECKPOINT_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-id", required=True, help="Explicit user-or-org/model name"
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Required acknowledgement that the destination repository is public",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the staged file list without creating a repository",
    )
    return parser.parse_args(argv)


def checkpoint_publish_files(checkpoint: Path) -> list[Path]:
    return sorted(
        path
        for path in checkpoint.iterdir()
        if path.is_file() and path.name in PUBLISHABLE_CHECKPOINT_FILES
    )


def _assert_adapter_only(checkpoint: Path) -> None:
    required = {"adapter_config.json", "adapter_model.safetensors"}
    names = {path.name for path in checkpoint_publish_files(checkpoint)}
    missing = required.difference(names)
    if missing:
        raise TrainingToolError(
            f"Checkpoint {checkpoint} is missing adapter files: {sorted(missing)}"
        )
    forbidden = [
        path.name
        for path in checkpoint.iterdir()
        if path.is_file()
        and (
            path.name.startswith("model-")
            or path.name in {"model.safetensors", "pytorch_model.bin"}
        )
    ]
    if forbidden:
        raise TrainingToolError(f"Full model weights found in checkpoint: {forbidden}")

    adapter_config = read_json(checkpoint / "adapter_config.json")
    token_indices = adapter_config.get("trainable_token_indices")
    if not isinstance(token_indices, dict) or set(token_indices) != {
        "embed_tokens",
        "lm_head",
    }:
        raise TrainingToolError(
            "Adapter config must train token rows for both embed_tokens and lm_head"
        )

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise TrainingToolError("safetensors is required before publication") from error
    with safe_open(
        checkpoint / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as handle:
        keys = list(handle.keys())
    if not any("lora_" in key for key in keys):
        raise TrainingToolError("Adapter artifact contains no LoRA tensors")
    if not any("embed_tokens" in key and "trainable_tokens" in key for key in keys):
        raise TrainingToolError("Adapter artifact omits trainable embed_tokens rows")
    if not any("lm_head" in key and "trainable_tokens" in key for key in keys):
        raise TrainingToolError("Adapter artifact omits trainable lm_head rows")
    if any("modules_to_save" in key for key in keys):
        raise TrainingToolError(
            "Full modules_to_save tensors are forbidden in public LoRA"
        )


def _copy_checkpoint(checkpoint: Path, destination: Path) -> None:
    _assert_adapter_only(checkpoint)
    destination.mkdir(parents=True, exist_ok=True)
    for source in checkpoint_publish_files(checkpoint):
        shutil.copy2(source, destination / source.name)


def _selected_metric(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    selection = metrics.get("selection")
    if not isinstance(selection, dict):
        raise TrainingToolError(
            "Evaluation has no canonical-passing selected checkpoint"
        )
    epoch = int(selection["epoch"])
    for item in metrics.get("checkpoints", []):
        if int(item["epoch"]) == epoch:
            return item
    raise TrainingToolError(f"Selected epoch {epoch} is absent from metrics")


def _assert_metrics_match(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> None:
    for dataset in ("canonical", "alias"):
        for metric in ("next_token_accuracy", "generation_accuracy"):
            if float(expected[dataset][metric]) != float(actual[dataset][metric]):
                raise TrainingToolError(
                    f"Regression mismatch for {dataset}.{metric}: "
                    f"expected {expected[dataset][metric]}, got {actual[dataset][metric]}"
                )


def _model_card(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    selected: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    comparison_rows = []
    for item in metrics["checkpoints"]:
        comparison_rows.append(
            "| {epoch} | {canonical:.2%} | {alias:.2%} |".format(
                epoch=item["epoch"],
                canonical=item["canonical"]["generation_accuracy"],
                alias=item["alias"]["generation_accuracy"],
            )
        )
    comparison = "\n".join(comparison_rows)
    return f"""---
base_model: {manifest['model']['id']}
library_name: peft
license: apache-2.0
language:
- zh
- en
tags:
- qwen3
- lora
- entity-linking
- steam
---

# Qwen3-8B Steam Entity Linking LoRA

This adapter maps Steam game names and related expressions to one-token labels such as `<GAME_730>`. It must be loaded with the tokenizer in this repository and the pinned base-model revision below.

## Reproducibility

- Base model: `{manifest['model']['id']}`
- Base revision: `{manifest['model']['revision']}`
- Training Git commit: `{manifest['git']['commit']}`
- Training source: `{manifest['git']['remote']}`
- Training data: {config_value(config, 'data.expected_train_rows')} entities, one canonical example per entity
- Selected checkpoint: epoch {selected['epoch']}
- Precision: BF16
- Method: LoRA r={config_value(config, 'lora.r')}, alpha={config_value(config, 'lora.alpha')} plus trainable added-token rows for both input embeddings and LM head

## Evaluation

| Epoch | Canonical exact match | Held-out alias exact match |
|---:|---:|---:|
{comparison}

Canonical accuracy measures memorization on training prompts. Alias accuracy uses {config_value(config, 'data.expected_alias_rows')} frozen cases that were not included in training. The selected checkpoint is the alias-best checkpoint among those reaching the configured canonical threshold.

## Usage

Load the tokenizer from this repository, resize `{manifest['model']['id']}` to the tokenizer length, and then load the PEFT adapter. Do not use the base tokenizer without the added `<GAME_APPID>` tokens.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter_id = "<hf-user>/<model-name>"
tokenizer = AutoTokenizer.from_pretrained(adapter_id)
base = AutoModelForCausalLM.from_pretrained(
    "{manifest['model']['id']}", revision="{manifest['model']['revision']}"
)
base.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(base, adapter_id)
```

## Limitations

The latest {config_value(config, 'data.expected_cohorts.latest')} games may not be represented in the base model's pretraining knowledge, so alias and natural-language generalization can be weaker for recent releases. This is a memorization/generalization PoC, not a replacement for a retrieval-backed production entity linker. Steam catalog names and AppIDs can also change over time.
"""


def _stage_public_repository(
    run_dir: Path,
    staging: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Mapping[str, Any]:
    selected = _selected_metric(metrics)
    checkpoints = dict(discover_checkpoints(run_dir))
    selected_epoch = int(selected["epoch"])
    if selected_epoch not in checkpoints:
        raise TrainingToolError(
            f"Selected checkpoint epoch {selected_epoch} is missing"
        )
    _copy_checkpoint(checkpoints[selected_epoch], staging)
    for epoch, checkpoint in sorted(checkpoints.items()):
        _copy_checkpoint(checkpoint, staging / "adapters" / f"epoch-{epoch}")

    shutil.copy2(run_dir / "training_config.yaml", staging / "training_config.yaml")
    shutil.copy2(run_dir / "run_manifest.json", staging / "run_manifest.json")
    shutil.copy2(run_dir / "metrics.json", staging / "metrics.json")
    shutil.copy2(
        run_dir / "checkpoint_comparison.csv", staging / "checkpoint_comparison.csv"
    )
    special_tokens_path = Path(str(config_value(config, "data.special_tokens_path")))
    if not special_tokens_path.is_absolute():
        special_tokens_path = Path(__file__).resolve().parents[1] / special_tokens_path
    shutil.copy2(special_tokens_path, staging / "special_tokens.json")
    (staging / "README.md").write_text(
        _model_card(manifest, metrics, selected, config), encoding="utf-8"
    )
    return selected


def _public_file_list(staging: Path) -> list[str]:
    return sorted(
        str(path.relative_to(staging)) for path in staging.rglob("*") if path.is_file()
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.public:
        raise TrainingToolError(
            "Publication requires the explicit --public acknowledgement"
        )
    if (
        "/" not in args.repo_id
        or args.repo_id.startswith("/")
        or args.repo_id.endswith("/")
    ):
        raise TrainingToolError("--repo-id must be an explicit user-or-org/model name")

    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    metrics = read_json(run_dir / "metrics.json")
    if manifest.get("status") not in {"evaluated", "published"}:
        raise TrainingToolError("Run must be evaluated before publication")
    if not metrics.get("acceptance_passed"):
        raise TrainingToolError("Canonical acceptance threshold was not reached")
    token = os.environ.get("HF_TOKEN")
    if not args.dry_run and not token:
        raise TrainingToolError("HF_TOKEN must be injected as a cloud secret")
    config = read_json(run_dir / "resolved_config.json")
    bundle = validate_data(config)
    if data_hashes(config) != manifest.get("data_sha256"):
        raise TrainingToolError("Current data files do not match the trained run")
    current_git = git_info(require_clean=True)
    if current_git["commit"] != manifest.get("git", {}).get("commit"):
        raise TrainingToolError("Current Git commit does not match the trained run")

    expected_epochs = list(config_value(config, "training.checkpoint_epochs"))
    metric_epochs = [int(item["epoch"]) for item in metrics.get("checkpoints", [])]
    checkpoint_epochs = [epoch for epoch, _ in discover_checkpoints(run_dir)]
    if metric_epochs != expected_epochs or checkpoint_epochs != expected_epochs:
        raise TrainingToolError(
            "All configured milestone checkpoints must be present and evaluated before publication"
        )

    selected = _selected_metric(metrics)
    selected_epoch = int(selected["epoch"])
    checkpoint = dict(discover_checkpoints(run_dir))[selected_epoch]
    print(
        f"Pre-upload regression: epoch {selected_epoch}, "
        f"canonical={selected['canonical']['generation_accuracy']:.2%}, "
        f"alias={selected['alias']['generation_accuracy']:.2%}",
        flush=True,
    )
    regression, _ = evaluate_adapter(
        checkpoint, selected_epoch, config, manifest, bundle
    )
    _assert_metrics_match(selected, regression)

    with tempfile.TemporaryDirectory(prefix="steam-entity-linking-publish-") as temp:
        staging = Path(temp) / "repository"
        staging.mkdir()
        _stage_public_repository(run_dir, staging, config, manifest, metrics)
        files = _public_file_list(staging)
        print(f"Public destination: https://huggingface.co/{args.repo_id}")
        print("Files to upload:")
        for filename in files:
            print(f"  {filename}")
        if args.dry_run:
            print("Dry run completed; no Hugging Face repository was changed.")
            return 0

        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as error:
            raise TrainingToolError(
                "huggingface-hub is required for publication"
            ) from error
        api = HfApi(token=token)
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=False,
            exist_ok=True,
        )
        commit = api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=staging,
            commit_message=(
                f"Publish Steam entity-linking LoRA from Git {manifest['git']['commit'][:12]}"
            ),
        )
        revision = getattr(commit, "oid", None) or "main"
        download_dir = Path(temp) / "downloaded"
        downloaded = Path(
            snapshot_download(
                repo_id=args.repo_id,
                repo_type="model",
                revision=revision,
                local_dir=download_dir,
                token=token,
            )
        )
        _assert_adapter_only(downloaded)
        readback, _ = evaluate_adapter(
            downloaded, selected_epoch, config, manifest, bundle
        )
        _assert_metrics_match(selected, readback)

    receipt = {
        "published_at": utc_now(),
        "repo_id": args.repo_id,
        "url": f"https://huggingface.co/{args.repo_id}",
        "revision": revision,
        "selected_epoch": selected_epoch,
    }
    atomic_write_json(run_dir / "publish_receipt.json", receipt)
    manifest["status"] = "published"
    manifest["publication"] = receipt
    atomic_write_json(manifest_path, manifest)
    print(f"Published and verified: {receipt['url']}/tree/{revision}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TrainingToolError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
