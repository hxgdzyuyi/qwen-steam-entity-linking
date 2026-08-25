#!/usr/bin/env python3
"""Evaluate PoC B's zero-training baseline and milestone classifier heads."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation_core import checkpoint_metrics
from feature_cache import cache_identity, load_feature_cache, stable_sha256, tokenizer_sha256
from steam_entity_classifier import (
    FrozenPrototypeHead,
    extract_features,
    freeze_backbone,
    initialize_prototypes,
    load_class_map,
    load_classifier_artifact,
)
from training_common import (
    TrainingToolError,
    atomic_write_json,
    checkpoint_epoch,
    class_map_payload,
    config_value,
    data_hashes,
    discover_checkpoints,
    git_info,
    load_poc_a_reference,
    prediction_sha256,
    read_json,
    select_best_checkpoint,
    subset_bundle,
    utc_now,
    validate_data,
    warn_if_git_commit_mismatch,
)


METRICS_SCHEMA_VERSION = 2
POC_ROOT = Path(__file__).resolve().parents[1]
REGRESSION_METRIC_FIELDS = (
    "epoch",
    "default_prompt_style",
    "canonical",
    "alias",
    "alias_prompt_benchmark",
    "canonical_by_cohort",
    "alias_by_cohort",
    "canonical_by_prompt_style",
    "alias_by_prompt_style",
    "alias_by_type",
    "prediction_sha256",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all-milestones", action="store_true")
    selection.add_argument("--checkpoint", type=Path)
    return parser.parse_args(argv)


def _cloud_imports() -> dict[str, Any]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise TrainingToolError(
            "Cloud evaluation dependencies are missing; install "
            "poc_b/requirements-cloud.txt"
        ) from error
    return {"torch": torch, "AutoModel": AutoModel, "AutoTokenizer": AutoTokenizer}


def _require_cuda(torch: Any) -> Any:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise TrainingToolError("Evaluation requires exactly one visible CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise TrainingToolError("Evaluation GPU must support BF16")
    # Match training-time CUDA math so exact prediction fingerprints are stable.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device("cuda")


def _run_config(run_dir: Path) -> dict[str, Any]:
    payload = read_json(run_dir / "resolved_config.json")
    if not isinstance(payload, dict):
        raise TrainingToolError("Run resolved_config.json must be an object")
    return payload


def _bundle_for_manifest(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> Any:
    bundle = validate_data(config)
    mode = manifest.get("mode")
    if mode == "smoke":
        return subset_bundle(
            bundle, int(config_value(config, "training.smoke_classes"))
        )
    if mode != "full":
        raise TrainingToolError(f"Run has invalid mode: {mode!r}")
    return bundle


def _classes(bundle: Any) -> list[dict[str, Any]]:
    return list(class_map_payload(bundle.classes)["classes"])


def _assert_artifact_identity(
    settings: Mapping[str, Any],
    *,
    epoch: int,
    manifest: Mapping[str, Any],
    class_map_sha256: str,
) -> None:
    expected = {
        "base_model_id": manifest["model"]["id"],
        "base_model_revision": manifest["model"]["revision"],
        "feature_cache_sha256": manifest["feature_cache_sha256"],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "class_map_sha256": class_map_sha256,
        "training_config_sha256": manifest["training_config_sha256"],
        "mode": manifest["mode"],
        "epoch": epoch,
        "num_classes": manifest["class_count"],
        "hidden_size": manifest["hidden_size"],
    }
    differences = {
        key: {"expected": value, "actual": settings.get(key)}
        for key, value in expected.items()
        if settings.get(key) != value
    }
    if differences:
        raise TrainingToolError(f"Classifier artifact identity differs: {differences}")


def _cached_metrics(
    *,
    checkpoint: Path,
    epoch: int,
    tensors: Mapping[str, Any],
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    bundle: Any,
    device: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    head, settings = load_classifier_artifact(checkpoint, device=device)
    class_payload = class_map_payload(bundle.classes)
    class_hash = stable_sha256(class_payload)
    _assert_artifact_identity(
        settings, epoch=epoch, manifest=manifest, class_map_sha256=class_hash
    )
    artifact_classes = load_class_map(
        checkpoint / "class_map.json", expected_classes=bundle.class_count
    )
    if artifact_classes != class_payload["classes"]:
        raise TrainingToolError("Checkpoint class map differs from evaluation data")
    metric, records = checkpoint_metrics(
        epoch=epoch,
        head=head,
        canonical_features=tensors["canonical_features"],
        alias_features=tensors["alias_features"],
        canonical_rows=bundle.canonical_rows,
        alias_rows=bundle.alias_rows,
        classes=artifact_classes,
        batch_size=int(config_value(config, "evaluation.batch_size")),
        diagnostic_top_k=int(config_value(config, "evaluation.diagnostic_top_k")),
        device=device,
    )
    return metric, records


def evaluate_artifact_fresh(
    artifact: Path,
    *,
    epoch: int,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    bundle: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Re-extract eval features from pinned Qwen for upload/readback regression."""

    imports = _cloud_imports()
    torch = imports["torch"]
    device = _require_cuda(torch)
    head, settings = load_classifier_artifact(artifact, device=device)
    expected_classes = _classes(bundle)
    artifact_classes = load_class_map(
        artifact / "class_map.json", expected_classes=bundle.class_count
    )
    if artifact_classes != expected_classes:
        raise TrainingToolError("Artifact class map differs from current evaluation data")
    class_hash = stable_sha256(class_map_payload(bundle.classes))
    _assert_artifact_identity(
        settings, epoch=epoch, manifest=manifest, class_map_sha256=class_hash
    )
    tokenizer_source: str | Path = (
        artifact
        if (artifact / "tokenizer_config.json").is_file()
        else str(settings["base_model_id"])
    )
    tokenizer = imports["AutoTokenizer"].from_pretrained(
        tokenizer_source,
        revision=(
            None
            if tokenizer_source == artifact
            else str(settings["base_model_revision"])
        ),
        token=os.environ.get("HF_TOKEN"),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if tokenizer_sha256(tokenizer) != settings["tokenizer_sha256"]:
        raise TrainingToolError("Evaluation tokenizer differs from the trained tokenizer")
    backbone = imports["AutoModel"].from_pretrained(
        str(settings["base_model_id"]),
        revision=str(settings["base_model_revision"]),
        trust_remote_code=bool(config_value(config, "model.trust_remote_code")),
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    freeze_backbone(backbone)
    try:
        canonical_features = extract_features(
            backbone,
            tokenizer,
            [row["model_input"] for row in bundle.canonical_rows],
            batch_size=int(config_value(config, "features.extraction_batch_size")),
            max_length=int(config_value(config, "data.max_length")),
            device=device,
        )
        alias_features = extract_features(
            backbone,
            tokenizer,
            [row["model_input"] for row in bundle.alias_rows],
            batch_size=int(config_value(config, "features.extraction_batch_size")),
            max_length=int(config_value(config, "data.max_length")),
            device=device,
        )
    finally:
        del backbone
        gc.collect()
        torch.cuda.empty_cache()
    return checkpoint_metrics(
        epoch=epoch,
        head=head,
        canonical_features=canonical_features,
        alias_features=alias_features,
        canonical_rows=bundle.canonical_rows,
        alias_rows=bundle.alias_rows,
        classes=artifact_classes,
        batch_size=int(config_value(config, "evaluation.batch_size")),
        diagnostic_top_k=int(config_value(config, "evaluation.diagnostic_top_k")),
        device=device,
    )


def assert_metrics_match(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> None:
    for field in REGRESSION_METRIC_FIELDS:
        if expected.get(field) != actual.get(field):
            raise TrainingToolError(
                f"Evaluation regression mismatch for {field}: "
                f"expected {expected.get(field)!r}, got {actual.get(field)!r}"
            )


def _write_failures(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = (
        "epoch",
        "source",
        "cohort",
        "type",
        "prompt_style",
        "input",
        "expected_appid",
        "expected_canonical_name",
        "predicted_appid",
        "predicted_canonical_name",
        "confidence",
        "rank",
        "top1_correct",
        "top5_correct",
        "top_k",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["top_k"] = json.dumps(row["top_k"], ensure_ascii=False)
            writer.writerow(output)


def _comparison_rows(
    checkpoint_rows: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if reference is not None:
        rows.append(
            {
                "system": "poc_a_published_epoch_20",
                "epoch": reference["selected_epoch"],
                "canonical_top1": reference["canonical"]["top1_accuracy"],
                "canonical_top5": "",
                "canonical_mrr": "",
                "alias_top1": reference["alias"]["top1_accuracy"],
                "alias_top5": "",
                "alias_mrr": "",
            }
        )
    for system, item in [
        ("poc_b_zero_training_prototype", baseline),
        *(("poc_b_trained", row) for row in checkpoint_rows),
    ]:
        rows.append(
            {
                "system": system,
                "epoch": item["epoch"],
                "canonical_top1": item["canonical"]["top1_accuracy"],
                "canonical_top5": item["canonical"]["top5_accuracy"],
                "canonical_mrr": item["canonical"]["mrr"],
                "alias_top1": item["alias"]["top1_accuracy"],
                "alias_top5": item["alias"]["top5_accuracy"],
                "alias_mrr": item["alias"]["mrr"],
            }
        )
    return rows


def _write_comparison(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = (
        "system",
        "epoch",
        "canonical_top1",
        "canonical_top5",
        "canonical_mrr",
        "alias_top1",
        "alias_top5",
        "alias_mrr",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("status") not in {
        "trained",
        "evaluated",
        "published",
    }:
        raise TrainingToolError("Run must finish training before evaluation")
    config = _run_config(run_dir)
    bundle = _bundle_for_manifest(config, manifest)
    current_hashes = data_hashes(config)
    if current_hashes != manifest.get("data_sha256"):
        raise TrainingToolError("Current data files differ from the trained run")
    current_git = git_info(require_clean=True)
    warn_if_git_commit_mismatch(
        current_git["commit"], manifest.get("git", {}).get("commit"), "Evaluation"
    )
    identity_hashes = dict(current_hashes)
    identity_hashes["run_class_map"] = manifest["class_map_sha256"]
    identity = cache_identity(
        mode=str(manifest["mode"]),
        model_id=str(manifest["model"]["id"]),
        model_revision=str(manifest["model"]["revision"]),
        tokenizer_hash=str(manifest["tokenizer_sha256"]),
        data_sha256=identity_hashes,
        max_length=int(config_value(config, "data.max_length")),
        pooling=str(config_value(config, "features.pooling")),
        class_count=bundle.class_count,
        row_counts={
            "train": len(bundle.train_rows),
            "canonical": len(bundle.canonical_rows),
            "alias": len(bundle.alias_rows),
        },
    )
    tensors, _ = load_feature_cache(
        run_dir / "feature_cache",
        expected_identity=identity,
        expected_cache_sha256=str(manifest["feature_cache_sha256"]),
    )

    if args.all_milestones:
        checkpoints = discover_checkpoints(run_dir)
        expected = list(config_value(config, "training.checkpoint_epochs"))
        if [epoch for epoch, _ in checkpoints] != expected:
            raise TrainingToolError(
                f"Expected milestone epochs {expected}, found "
                f"{[epoch for epoch, _ in checkpoints]}"
            )
    else:
        checkpoint = args.checkpoint.resolve()
        try:
            checkpoint.relative_to(run_dir)
        except ValueError as error:
            raise TrainingToolError("Checkpoint must be inside the run directory") from error
        checkpoints = [(checkpoint_epoch(checkpoint), checkpoint)]

    imports = _cloud_imports()
    torch = imports["torch"]
    device = _require_cuda(torch)
    initial = initialize_prototypes(
        tensors["train_features"], tensors["train_labels"], bundle.class_count
    )
    baseline_head = FrozenPrototypeHead(
        initial,
        bottleneck_dim=int(config_value(config, "classifier.bottleneck_dim")),
        temperature=float(config_value(config, "classifier.temperature")),
    ).to(device)
    baseline, _ = checkpoint_metrics(
        epoch=0,
        head=baseline_head,
        canonical_features=tensors["canonical_features"],
        alias_features=tensors["alias_features"],
        canonical_rows=bundle.canonical_rows,
        alias_rows=bundle.alias_rows,
        classes=_classes(bundle),
        batch_size=int(config_value(config, "evaluation.batch_size")),
        diagnostic_top_k=int(config_value(config, "evaluation.diagnostic_top_k")),
        device=device,
    )
    baseline["kind"] = "zero_training_canonical_view_prototypes"
    baseline["alias_used_for_training"] = False
    trained_baseline = read_json(run_dir / "zero_training_baseline.json")
    if prediction_sha256([baseline]) != prediction_sha256([trained_baseline]):
        raise TrainingToolError("Zero-training baseline differs from training-time result")

    checkpoint_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for epoch, checkpoint in checkpoints:
        print(f"Evaluating epoch {epoch}: {checkpoint}", flush=True)
        metric, records = _cached_metrics(
            checkpoint=checkpoint,
            epoch=epoch,
            tensors=tensors,
            config=config,
            manifest=manifest,
            bundle=bundle,
            device=device,
        )
        metric["checkpoint"] = str(checkpoint.relative_to(run_dir))
        checkpoint_rows.append(metric)
        failures.extend(
            {"epoch": epoch, **record}
            for record in records
            if not record["top1_correct"]
        )

    threshold = float(config_value(config, "evaluation.canonical_threshold"))
    selected = select_best_checkpoint(checkpoint_rows, threshold)
    reference = load_poc_a_reference(config) if manifest["mode"] == "full" else None
    alias_improved = (
        bool(
            selected["alias"]["top1_accuracy"]
            > reference["alias"]["top1_accuracy"]
        )
        if selected is not None and reference is not None
        else None
    )
    comparison = None
    if selected is not None and reference is not None:
        if set(selected["alias_by_type"]) != set(reference["alias_by_type"]):
            raise TrainingToolError(
                "PoC A and PoC B alias type groups differ; refusing A/B delta"
            )
        comparison = {
            "poc_a_reference": reference,
            "zero_training_alias_top1_delta": baseline["alias"]["top1_accuracy"]
            - reference["alias"]["top1_accuracy"],
            "canonical_top1_delta": selected["canonical"]["top1_accuracy"]
            - reference["canonical"]["top1_accuracy"],
            "alias_top1_delta": selected["alias"]["top1_accuracy"]
            - reference["alias"]["top1_accuracy"],
            "alias_by_type_top1_delta": {
                case_type: selected["alias_by_type"][case_type]["top1_accuracy"]
                - summary["top1_accuracy"]
                for case_type, summary in reference["alias_by_type"].items()
            },
        }
    report = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "evaluated_at": utc_now(),
        "mode": manifest["mode"],
        "publishable": manifest["mode"] == "full",
        "canonical_threshold": threshold,
        "zero_training_prototype": baseline,
        "checkpoints": checkpoint_rows,
        "selection": (
            {
                "epoch": selected["epoch"],
                "checkpoint": selected["checkpoint"],
                "rule": "canonical Top-1 >= threshold; then alias Top-1, canonical Top-1, earlier epoch",
            }
            if selected is not None
            else None
        ),
        "acceptance_passed": selected is not None,
        "alias_improved_over_poc_a": alias_improved,
        "comparison_to_poc_a": comparison,
    }
    atomic_write_json(run_dir / "metrics.json", report)
    _write_comparison(
        run_dir / "checkpoint_comparison.csv",
        _comparison_rows(checkpoint_rows, baseline, reference),
    )
    _write_failures(run_dir / "evaluation_failures.csv", failures)
    manifest["status"] = "evaluated"
    manifest["evaluated_at"] = report["evaluated_at"]
    manifest["selection"] = report["selection"]
    atomic_write_json(manifest_path, manifest)
    if selected is None:
        raise TrainingToolError(
            f"No checkpoint reached canonical Top-1 {threshold:.2%}"
        )
    print(
        f"Selected epoch {selected['epoch']}: "
        f"canonical={selected['canonical']['top1_accuracy']:.2%}, "
        f"alias={selected['alias']['top1_accuracy']:.2%}, "
        f"alias_improved_over_poc_a={alias_improved}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TrainingToolError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
