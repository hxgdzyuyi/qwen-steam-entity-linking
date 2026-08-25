#!/usr/bin/env python3
"""Safely publish an evaluated PoC B head-only run to Hugging Face."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common.huggingface_repositories import (  # noqa: E402
    RepositoryConfigError,
    repository_for,
    validate_repo_id,
)
from evaluate import (  # noqa: E402
    METRICS_SCHEMA_VERSION,
    assert_metrics_match,
    evaluate_artifact_fresh,
)
from feature_cache import stable_sha256, tokenizer_sha256  # noqa: E402
from steam_entity_classifier import (  # noqa: E402
    ALLOWED_TENSOR_KEYS,
    load_classifier_artifact,
)
from training_common import (  # noqa: E402
    TrainingToolError,
    atomic_write_json,
    class_map_payload,
    config_value,
    data_hashes,
    discover_checkpoints,
    git_info,
    load_poc_a_reference,
    read_json,
    sha256_file,
    utc_now,
    validate_data,
    warn_if_git_commit_mismatch,
)


GIB = 1024**3
HUB_MANAGED_REMOTE_FILES = {".gitattributes"}
CLASSIFIER_FILES = {
    "classifier.safetensors",
    "classifier_config.json",
    "class_map.json",
}
TOKENIZER_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
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
        "--repo-id",
        help="Optional user-or-org/model override; defaults to registered PoC B repo",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="Re-evaluate and stage locally without changing Hugging Face",
    )
    action.add_argument(
        "--public",
        action="store_true",
        help="Upload privately, verify readback, then make the repository public",
    )
    return parser.parse_args(argv)


def configured_repository() -> dict[str, Any]:
    try:
        repository = repository_for("poc_b")
    except RepositoryConfigError as error:
        raise TrainingToolError(
            f"Invalid Hugging Face repository registry: {error}"
        ) from error
    if repository["status"] not in {"ready", "active"}:
        raise TrainingToolError("Registered PoC B repository is not ready for publishing")
    return repository


def resolve_publish_repo_id(requested: str | None) -> str:
    if requested is None:
        return str(configured_repository()["repo_id"])
    try:
        return validate_repo_id(requested)
    except RepositoryConfigError as error:
        raise TrainingToolError(f"Invalid --repo-id: {error}") from error


def _selected_metric(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    selection = metrics.get("selection")
    if not isinstance(selection, dict):
        raise TrainingToolError("Metrics contain no canonical-passing selection")
    epoch = int(selection["epoch"])
    for row in metrics.get("checkpoints", []):
        if int(row["epoch"]) == epoch:
            return row
    raise TrainingToolError(f"Selected epoch {epoch} is absent from metrics")


def _assert_classifier_only(
    artifact: Path,
    *,
    expected_epoch: int | None = None,
    expected_class_map_sha256: str | None = None,
    expected_settings: Mapping[str, Any] | None = None,
) -> None:
    names = {path.name for path in artifact.iterdir() if path.is_file()}
    missing = CLASSIFIER_FILES.difference(names)
    if missing:
        raise TrainingToolError(
            f"Classifier artifact {artifact} is missing {sorted(missing)}"
        )
    forbidden = sorted(
        path.name
        for path in artifact.iterdir()
        if path.is_file()
        and (
            path.name.startswith(("model-", "pytorch_model-"))
            or path.name
            in {
                "model.safetensors",
                "pytorch_model.bin",
                "optimizer.pt",
                "scheduler.pt",
                "training_state.pt",
                "features.safetensors",
            }
        )
    )
    if forbidden:
        raise TrainingToolError(
            f"Classifier artifact contains forbidden weights/state: {forbidden}"
        )
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise TrainingToolError("safetensors is required for publication") from error
    with safe_open(
        artifact / "classifier.safetensors", framework="pt", device="cpu"
    ) as handle:
        keys = set(handle.keys())
    if keys != ALLOWED_TENSOR_KEYS:
        raise TrainingToolError(
            f"Classifier artifact has invalid tensor keys: {sorted(keys)}"
        )
    _, settings = load_classifier_artifact(artifact)
    if expected_epoch is not None and int(settings["epoch"]) != expected_epoch:
        raise TrainingToolError("Classifier epoch differs from its milestone path")
    class_map = read_json(artifact / "class_map.json")
    class_hash = stable_sha256(class_map)
    if class_hash != settings["class_map_sha256"]:
        raise TrainingToolError("Classifier class-map hash differs from config")
    if expected_class_map_sha256 and class_hash != expected_class_map_sha256:
        raise TrainingToolError("Classifier class map differs from trained run")
    if expected_settings:
        differences = {
            key: {"expected": value, "actual": settings.get(key)}
            for key, value in expected_settings.items()
            if settings.get(key) != value
        }
        if differences:
            raise TrainingToolError(
                f"Classifier provenance differs from the run: {differences}"
            )


def _copy_classifier(source: Path, destination: Path, *, epoch: int) -> None:
    _assert_classifier_only(source, expected_epoch=epoch)
    destination.mkdir(parents=True, exist_ok=True)
    for name in sorted(CLASSIFIER_FILES):
        shutil.copy2(source / name, destination / name)


def _copy_tokenizer(source: Path, destination: Path) -> None:
    copied = 0
    for path in sorted(source.iterdir()):
        if path.is_file() and path.name in TOKENIZER_FILES:
            shutil.copy2(path, destination / path.name)
            copied += 1
    if not copied or not (destination / "tokenizer_config.json").is_file():
        raise TrainingToolError("Run tokenizer is incomplete")


def _model_card(
    *,
    repo_id: str,
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    selected = _selected_metric(metrics)
    rows: list[str] = []
    poc_a_comparison = metrics.get("comparison_to_poc_a")
    if isinstance(poc_a_comparison, dict):
        reference = poc_a_comparison["poc_a_reference"]
        rows.append(
            "| published PoC A | {epoch} | {canonical:.2%} | {alias:.2%} |".format(
                epoch=reference["selected_epoch"],
                canonical=reference["canonical"]["top1_accuracy"],
                alias=reference["alias"]["top1_accuracy"],
            )
        )
    rows.append(
        "| zero-training prototypes | 0 | {canonical:.2%} | {alias:.2%} |".format(
            canonical=metrics["zero_training_prototype"]["canonical"][
                "top1_accuracy"
            ],
            alias=metrics["zero_training_prototype"]["alias"]["top1_accuracy"],
        )
    )
    rows.extend(
        "| trained head | {epoch} | {canonical:.2%} | {alias:.2%} |".format(
            epoch=row["epoch"],
            canonical=row["canonical"]["top1_accuracy"],
            alias=row["alias"]["top1_accuracy"],
        )
        for row in metrics["checkpoints"]
    )
    comparison = "\n".join(rows)
    return f"""---
base_model: {manifest['model']['id']}
library_name: pytorch
license: apache-2.0
language:
- zh
- en
tags:
- qwen3
- entity-linking
- steam
- prototype-classifier
- poc-b
---

# Qwen3-8B Steam Entity Linking — PoC B

This repository contains only a low-rank residual projection and 1000 cosine
class prototypes. The Qwen backbone is frozen and is loaded separately at its
pinned revision. There is no RAG, vector database, candidate retrieval, reranking,
LoRA, or generated AppID token. The local feature cache and optimizer state are
not part of this repository.

## Reproducibility

- Base model: `{manifest['model']['id']}`
- Base revision: `{manifest['model']['revision']}`
- Training Git commit: `{manifest['git']['commit']}`
- Training source: `{manifest['git']['remote']}`
- Class ordering: numeric AppID ascending
- Canonical-only training views: {config_value(config, 'data.expected_train_rows')}
- Held-out alias cases: {config_value(config, 'data.expected_alias_rows')}
- Selected head: epoch {selected['epoch']}
- Frozen representation: final layer, last non-padding token, FP32 cache
- Head: hidden → {config_value(config, 'classifier.bottleneck_dim')} → hidden residual plus cosine prototypes
- Temperature: {config_value(config, 'classifier.temperature')}

## Evaluation

| Variant | Epoch | Canonical Top-1 | Alias Top-1 |
|---|---:|---:|---:|
{comparison}

Canonical acceptance is {metrics['canonical_threshold']:.0%}. Alias is report-only.
Compared with the pinned published PoC A reference, `alias_improved_over_poc_a`
is `{str(metrics['alias_improved_over_poc_a']).lower()}`.

## Usage

The loader reads `classifier_config.json`, downloads the exact Qwen revision,
freezes it, and attaches this repository's classifier head.

```python
import sys
from huggingface_hub import snapshot_download

repo_dir = snapshot_download("{repo_id}")
sys.path.insert(0, repo_dir)
from steam_entity_classifier import SteamEntityLinker

model = SteamEntityLinker.from_pretrained(repo_dir)
result = model.predict(["CS2", "反恐精英"], top_k=5)
print(result)
```

The Top-1 result contains `appid`, `canonical_name`, `class_index`, and
`confidence`; `top_k` is diagnostic.

## Limitations

This is a 1000-class closed-set classifier. It always emits one known AppID and
does not support `UNKNOWN`. Results for aliases, descriptions, and games newer
than the base model's knowledge cutoff may be weak. The 184 alias cases were
never used for training, checkpoint gating, or prototype initialization.
"""


def _stage_repository(
    *,
    run_dir: Path,
    staging: Path,
    repo_id: str,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Mapping[str, Any]:
    selected = _selected_metric(metrics)
    selected_epoch = int(selected["epoch"])
    checkpoints = dict(discover_checkpoints(run_dir))
    _copy_classifier(checkpoints[selected_epoch], staging, epoch=selected_epoch)
    for epoch, checkpoint in sorted(checkpoints.items()):
        _copy_classifier(
            checkpoint, staging / "heads" / f"epoch-{epoch}", epoch=epoch
        )
    _copy_tokenizer(run_dir / "tokenizer", staging)
    for source, destination in (
        (run_dir / "training_config.yaml", staging / "training_config.yaml"),
        (run_dir / "resolved_config.json", staging / "resolved_config.json"),
        (run_dir / "run_manifest.json", staging / "run_manifest.json"),
        (run_dir / "metrics.json", staging / "metrics.json"),
        (
            run_dir / "checkpoint_comparison.csv",
            staging / "checkpoint_comparison.csv",
        ),
        (
            run_dir / "zero_training_baseline.json",
            staging / "zero_training_baseline.json",
        ),
        (
            REPOSITORY_ROOT / "common/baselines/poc_a_reference.json",
            staging / "poc_a_reference.json",
        ),
        (
            Path(__file__).with_name("steam_entity_classifier.py"),
            staging / "steam_entity_classifier.py",
        ),
    ):
        if not source.is_file():
            raise TrainingToolError(f"Required publication file is missing: {source}")
        shutil.copy2(source, destination)
    (staging / "README.md").write_text(
        _model_card(
            repo_id=repo_id, manifest=manifest, metrics=metrics, config=config
        ),
        encoding="utf-8",
    )
    existing_files = sorted(
        path for path in staging.rglob("*") if path.is_file()
    )
    atomic_write_json(
        staging / "artifact_manifest.json",
        {
            "schema_version": 1,
            "experiment": "poc_b",
            "artifact_kind": "frozen-backbone-semantic-prototype-classifier",
            "created_at": utc_now(),
            "repo_id": repo_id,
            "selected_epoch": selected_epoch,
            "milestone_epochs": sorted(checkpoints),
            "model": manifest["model"],
            "class_map_sha256": manifest["class_map_sha256"],
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "training_config_sha256": manifest["training_config_sha256"],
            "files": {
                str(path.relative_to(staging)): sha256_file(path)
                for path in existing_files
            },
            "excluded": [
                "Qwen backbone weights",
                "feature cache",
                "optimizer and scheduler state",
            ],
            "manifest_self_excluded": True,
        },
    )
    return selected


def public_file_list(staging: Path) -> list[str]:
    return sorted(
        str(path.relative_to(staging)) for path in staging.rglob("*") if path.is_file()
    )


def _assert_staging_safe(
    staging: Path,
    expected_epochs: Sequence[int],
    *,
    manifest: Mapping[str, Any] | None = None,
    selected_epoch: int | None = None,
) -> None:
    forbidden_components = {"feature_cache", "resume", "optimizer", "scheduler"}
    for path in staging.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(staging)
        if any(part.casefold() in forbidden_components for part in relative.parts):
            raise TrainingToolError(f"Forbidden runtime state staged: {relative}")
        if path.name.startswith(("model-", "pytorch_model-")) or path.name in {
            "model.safetensors",
            "pytorch_model.bin",
            "training_state.pt",
            "features.safetensors",
        }:
            raise TrainingToolError(f"Forbidden base/cache weight staged: {relative}")
    expected_settings = None
    expected_class_hash = None
    if manifest is not None:
        expected_class_hash = str(manifest["class_map_sha256"])
        expected_settings = {
            "base_model_id": manifest["model"]["id"],
            "base_model_revision": manifest["model"]["revision"],
            "feature_cache_sha256": manifest["feature_cache_sha256"],
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "class_map_sha256": manifest["class_map_sha256"],
            "training_config_sha256": manifest["training_config_sha256"],
            "mode": "full",
            "num_classes": manifest["class_count"],
            "hidden_size": manifest["hidden_size"],
        }
    _assert_classifier_only(
        staging,
        expected_epoch=selected_epoch,
        expected_class_map_sha256=expected_class_hash,
        expected_settings=expected_settings,
    )
    for epoch in expected_epochs:
        _assert_classifier_only(
            staging / "heads" / f"epoch-{epoch}",
            expected_epoch=epoch,
            expected_class_map_sha256=expected_class_hash,
            expected_settings=expected_settings,
        )
    if manifest is not None:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise TrainingToolError(
                "transformers is required to validate the staged tokenizer"
            ) from error
        tokenizer = AutoTokenizer.from_pretrained(staging, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        if tokenizer_sha256(tokenizer) != manifest["tokenizer_sha256"]:
            raise TrainingToolError("Staged tokenizer differs from the trained run")


def _assert_remote_file_set(
    remote_files: Sequence[str],
    staged_files: Sequence[str],
    *,
    require_complete: bool,
) -> None:
    remote = set(remote_files)
    staged = set(staged_files)
    unexpected = sorted(remote - staged - HUB_MANAGED_REMOTE_FILES)
    if unexpected:
        raise TrainingToolError(
            f"Destination repository contains unexpected stale files: {unexpected}"
        )
    if require_complete:
        missing = sorted(staged - remote)
        if missing:
            raise TrainingToolError(f"Uploaded repository is missing files: {missing}")


def _ensure_private(api: Any, repo_id: str) -> None:
    info = api.model_info(repo_id=repo_id)
    if getattr(info, "private", None) is not True:
        api.update_repo_settings(repo_id=repo_id, private=True)
        info = api.model_info(repo_id=repo_id)
    if getattr(info, "private", None) is not True:
        raise TrainingToolError("Hugging Face did not confirm private staging")


def _make_public(api: Any, repo_id: str) -> None:
    api.update_repo_settings(repo_id=repo_id, private=False)
    if getattr(api.model_info(repo_id=repo_id), "private", None) is not False:
        raise TrainingToolError("Hugging Face did not confirm public visibility")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_id = resolve_publish_repo_id(args.repo_id)
    if args.public and not os.environ.get("HF_TOKEN"):
        raise TrainingToolError("HF_TOKEN must be injected as a cloud secret")
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    metrics = read_json(run_dir / "metrics.json")
    config = read_json(run_dir / "resolved_config.json")
    if manifest.get("status") not in {"evaluated", "published"}:
        raise TrainingToolError("Run must be evaluated before publication")
    if manifest.get("mode") != "full" or not manifest.get("publishable"):
        raise TrainingToolError("Smoke runs are non-publishable")
    if metrics.get("schema_version") != METRICS_SCHEMA_VERSION:
        raise TrainingToolError("Metrics schema is stale; rerun PoC B evaluation")
    if not metrics.get("acceptance_passed"):
        raise TrainingToolError("Canonical acceptance threshold was not reached")
    bundle = validate_data(config)
    if data_hashes(config) != manifest.get("data_sha256"):
        raise TrainingToolError("Current data differs from the evaluated run")
    load_poc_a_reference(config)
    current_git = git_info(require_clean=True)
    warn_if_git_commit_mismatch(
        current_git["commit"], manifest.get("git", {}).get("commit"), "Publication"
    )
    required_disk = float(
        config_value(config, "runtime.minimum_publish_free_disk_gib")
    )
    free_disk = shutil.disk_usage(run_dir).free / GIB
    if free_disk < required_disk:
        raise TrainingToolError(
            f"Publication requires {required_disk:g} GiB free, found {free_disk:.1f}"
        )
    expected_epochs = list(config_value(config, "training.checkpoint_epochs"))
    checkpoint_epochs = [epoch for epoch, _ in discover_checkpoints(run_dir)]
    metric_epochs = [int(row["epoch"]) for row in metrics.get("checkpoints", [])]
    if checkpoint_epochs != expected_epochs or metric_epochs != expected_epochs:
        raise TrainingToolError("All five milestone heads must be present and evaluated")
    selected = _selected_metric(metrics)
    selected_epoch = int(selected["epoch"])
    checkpoint = dict(discover_checkpoints(run_dir))[selected_epoch]
    print(
        f"Pre-upload fresh regression: epoch {selected_epoch}, "
        f"canonical={selected['canonical']['top1_accuracy']:.2%}, "
        f"alias={selected['alias']['top1_accuracy']:.2%}",
        flush=True,
    )
    regression, _ = evaluate_artifact_fresh(
        checkpoint,
        epoch=selected_epoch,
        config=config,
        manifest=manifest,
        bundle=bundle,
    )
    assert_metrics_match(selected, regression)

    with tempfile.TemporaryDirectory(prefix=".poc-b-publish-", dir=run_dir) as temp:
        staging = Path(temp) / "repository"
        staging.mkdir()
        _stage_repository(
            run_dir=run_dir,
            staging=staging,
            repo_id=repo_id,
            config=config,
            manifest=manifest,
            metrics=metrics,
        )
        _assert_staging_safe(
            staging,
            expected_epochs,
            manifest=manifest,
            selected_epoch=selected_epoch,
        )
        files = public_file_list(staging)
        print(f"PoC B model repository: https://huggingface.co/{repo_id}")
        print("Files to upload:")
        for filename in files:
            print(f"  {filename}")
        if args.dry_run:
            print("Dry run complete; Hugging Face was not changed.")
            return 0
        token = os.environ.get("HF_TOKEN")
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as error:
            raise TrainingToolError(
                "huggingface-hub is required for publication"
            ) from error
        api = HfApi(token=token)
        api.create_repo(
            repo_id=repo_id, repo_type="model", private=True, exist_ok=True
        )
        _ensure_private(api, repo_id)
        remote_before = api.list_repo_files(repo_id=repo_id, repo_type="model")
        _assert_remote_file_set(remote_before, files, require_complete=False)
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=staging,
            commit_message=(
                "Publish frozen prototype classifier from Git "
                f"{manifest['git']['commit'][:12]}"
            ),
        )
        revision = getattr(commit, "oid", None) or "main"
        remote_after = api.list_repo_files(
            repo_id=repo_id, repo_type="model", revision=revision
        )
        _assert_remote_file_set(remote_after, files, require_complete=True)
        downloaded = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                local_dir=Path(temp) / "downloaded",
                token=token,
            )
        )
        _assert_staging_safe(
            downloaded,
            expected_epochs,
            manifest=manifest,
            selected_epoch=selected_epoch,
        )
        readback, _ = evaluate_artifact_fresh(
            downloaded,
            epoch=selected_epoch,
            config=config,
            manifest=manifest,
            bundle=bundle,
        )
        assert_metrics_match(selected, readback)
        _make_public(api, repo_id)

    receipt = {
        "published_at": utc_now(),
        "experiment": "poc_b",
        "repo_type": "model",
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "revision": revision,
        "selected_epoch": selected_epoch,
        "registry_status_at_publish": configured_repository()["status"],
        "next_registry_status": "active",
    }
    atomic_write_json(run_dir / "publish_receipt.json", receipt)
    manifest["status"] = "published"
    manifest["publication"] = receipt
    atomic_write_json(manifest_path, manifest)
    print(f"Published and verified: {receipt['url']}/tree/{revision}")
    print("Remote publication is verified; change PoC B registry status to active separately.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TrainingToolError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
