#!/usr/bin/env python3
"""Evaluate one or all entity-linking LoRA checkpoints from a cloud run."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from inference_core import predict_structured_rows

from training_common import (
    TrainingToolError,
    atomic_write_json,
    config_value,
    data_hashes,
    discover_checkpoints,
    git_info,
    load_config,
    read_json,
    select_best_checkpoint,
    utc_now,
    validate_data,
    validate_entity_tokens,
    validate_no_match_token,
    warn_if_git_commit_mismatch,
)


METRICS_SCHEMA_VERSION = 4


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
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise TrainingToolError(
            "Cloud evaluation dependencies are missing; install "
            "poc_a/requirements-cloud.txt"
        ) from error
    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def _run_config(run_dir: Path) -> dict[str, Any]:
    resolved_path = run_dir / "resolved_config.json"
    if resolved_path.exists():
        value = read_json(resolved_path)
        if not isinstance(value, dict):
            raise TrainingToolError(f"Invalid resolved config: {resolved_path}")
        return value
    return load_config(run_dir / "training_config.yaml")


def _metric_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    next_correct = sum(row["next_token_correct"] for row in records)
    generation_correct = sum(row["generation_correct"] for row in records)
    format_correct = sum(row["format_valid"] for row in records)
    canonical_correct = sum(row["canonical_name_correct"] for row in records)
    token_correct = sum(row["generated_token_correct"] for row in records)
    structured_correct = sum(row["structured_exact"] for row in records)
    top1_correct = sum(int(row["entity_rank"]) == 1 for row in records)
    top5_correct = sum(row["entity_top5_correct"] for row in records)
    top10_correct = sum(row["entity_top10_correct"] for row in records)
    ranks = [int(row["entity_rank"]) for row in records]
    return {
        "count": count,
        "unique_inputs": len({str(row["input"]).casefold() for row in records}),
        "format_valid_correct": format_correct,
        "format_valid_accuracy": format_correct / count if count else 0.0,
        "canonical_name_correct": canonical_correct,
        "canonical_name_accuracy": canonical_correct / count if count else 0.0,
        "generated_token_correct": token_correct,
        "generated_token_accuracy": token_correct / count if count else 0.0,
        "structured_exact_correct": structured_correct,
        "structured_exact_accuracy": structured_correct / count if count else 0.0,
        "safe_resolution_correct": generation_correct,
        "safe_resolution_accuracy": generation_correct / count if count else 0.0,
        "explicit_no_match_count": sum(row["explicit_no_match"] for row in records),
        "unrestricted_top1_correct": next_correct,
        "unrestricted_top1_accuracy": next_correct / count if count else 0.0,
        "entity_top1_correct": top1_correct,
        "entity_top1_accuracy": top1_correct / count if count else 0.0,
        "entity_top5_correct": top5_correct,
        "entity_top5_accuracy": top5_correct / count if count else 0.0,
        "entity_top10_correct": top10_correct,
        "entity_top10_accuracy": top10_correct / count if count else 0.0,
        "mean_entity_rank": sum(ranks) / count if count else 0.0,
        "next_token_correct": next_correct,
        "next_token_accuracy": next_correct / count if count else 0.0,
        "generation_correct": generation_correct,
        "generation_accuracy": generation_correct / count if count else 0.0,
    }


def _breakdown(
    records: Sequence[Mapping[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row[key])].append(row)
    return {name: _metric_summary(rows) for name, rows in sorted(grouped.items())}


def prediction_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Fingerprint the complete ordered prediction records for regression checks."""

    payload = json.dumps(
        list(records),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evaluate_adapter(
    checkpoint: Path,
    epoch: int,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    bundle: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    imports = _cloud_imports()
    torch = imports["torch"]
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise TrainingToolError("Evaluation requires exactly one visible CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise TrainingToolError("Evaluation GPU must support BF16")

    model_id = str(manifest["model"]["id"])
    revision = str(manifest["model"]["revision"])
    tokenizer = imports["AutoTokenizer"].from_pretrained(
        checkpoint,
        use_fast=True,
        token=os.environ.get("HF_TOKEN"),
    )
    entity_token_ids = validate_entity_tokens(tokenizer, bundle.special_tokens)
    no_match_token_id = validate_no_match_token(tokenizer, bundle.no_match_token)
    output_token_ids = [*entity_token_ids, no_match_token_id]
    base_model = imports["AutoModelForCausalLM"].from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=bool(config_value(config, "model.trust_remote_code")),
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = imports["PeftModel"].from_pretrained(base_model, checkpoint)
    model.config.use_cache = True
    model.to(torch.device("cuda"))

    batch_size = int(config_value(config, "evaluation.batch_size"))
    max_length = int(config_value(config, "data.max_length"))
    max_new_tokens = int(config_value(config, "evaluation.generation_max_new_tokens"))
    canonical_rows = [
        row for row in bundle.train_rows if row["completion"] != bundle.no_match_token
    ]
    canonical_records = predict_structured_rows(
        model,
        tokenizer,
        canonical_rows,
        expected_key="completion",
        source="canonical",
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        output_token_ids=output_token_ids,
        canonical_by_token=bundle.canonical_by_token,
        provenance_by_token=bundle.provenance_by_token,
        torch=torch,
    )
    alias_records = predict_structured_rows(
        model,
        tokenizer,
        bundle.alias_rows,
        expected_key="expected",
        source="alias",
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        output_token_ids=output_token_ids,
        canonical_by_token=bundle.canonical_by_token,
        provenance_by_token=bundle.provenance_by_token,
        torch=torch,
    )
    unknown_records = predict_structured_rows(
        model,
        tokenizer,
        bundle.unknown_rows,
        expected_key="expected",
        source="unknown",
        batch_size=batch_size,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        output_token_ids=output_token_ids,
        canonical_by_token=bundle.canonical_by_token,
        provenance_by_token=bundle.provenance_by_token,
        torch=torch,
    )
    metric = {
        "epoch": epoch,
        "checkpoint": str(checkpoint),
        "canonical": _metric_summary(canonical_records),
        "alias": _metric_summary(alias_records),
        "unknown": _metric_summary(unknown_records),
        "canonical_by_cohort": _breakdown(canonical_records, "cohort"),
        "canonical_by_prompt_style": _breakdown(
            canonical_records, "prompt_style"
        ),
        "alias_by_type": _breakdown(alias_records, "type"),
        "alias_by_prompt_style": _breakdown(alias_records, "prompt_style"),
        "unknown_by_type": _breakdown(unknown_records, "type"),
        "unknown_by_prompt_style": _breakdown(unknown_records, "prompt_style"),
        "prediction_sha256": prediction_sha256(
            canonical_records + alias_records + unknown_records
        ),
    }
    failures = [
        {"epoch": epoch, **record}
        for record in canonical_records + alias_records + unknown_records
        if not record["structured_exact"] or not record["generation_correct"]
    ]
    del model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    return metric, failures


def _write_failures(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = (
        "epoch",
        "source",
        "cohort",
        "type",
        "prompt_style",
        "input",
        "prompt",
        "expected_canonical_name",
        "expected",
        "generated_canonical_name",
        "resolved_canonical_name",
        "predicted_next_token",
        "oracle_constrained_token",
        "generated_token",
        "resolved_token",
        "generated_text",
        "resolution_status",
        "format_valid",
        "canonical_name_correct",
        "generated_token_correct",
        "structured_exact",
        "safe_resolution_correct",
        "explicit_no_match",
        "entity_rank",
        "entity_top5_correct",
        "entity_top10_correct",
        "next_token_correct",
        "generation_correct",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_comparison(path: Path, metrics: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "epoch",
                "canonical_name_accuracy",
                "canonical_structured_exact_accuracy",
                "canonical_safe_resolution_accuracy",
                "canonical_oracle_entity_top1_accuracy",
                "alias_name_accuracy",
                "alias_structured_exact_accuracy",
                "alias_safe_resolution_accuracy",
                "alias_oracle_entity_top1_accuracy",
                "unknown_explicit_no_match_accuracy",
                "unknown_safe_rejection_accuracy",
            ),
        )
        writer.writeheader()
        for item in metrics:
            writer.writerow(
                {
                    "epoch": item["epoch"],
                    "canonical_name_accuracy": item["canonical"][
                        "canonical_name_accuracy"
                    ],
                    "canonical_structured_exact_accuracy": item["canonical"][
                        "structured_exact_accuracy"
                    ],
                    "canonical_safe_resolution_accuracy": item["canonical"][
                        "safe_resolution_accuracy"
                    ],
                    "canonical_oracle_entity_top1_accuracy": item["canonical"][
                        "entity_top1_accuracy"
                    ],
                    "alias_name_accuracy": item["alias"][
                        "canonical_name_accuracy"
                    ],
                    "alias_structured_exact_accuracy": item["alias"][
                        "structured_exact_accuracy"
                    ],
                    "alias_safe_resolution_accuracy": item["alias"][
                        "safe_resolution_accuracy"
                    ],
                    "alias_oracle_entity_top1_accuracy": item["alias"][
                        "entity_top1_accuracy"
                    ],
                    "unknown_explicit_no_match_accuracy": item["unknown"][
                        "explicit_no_match_count"
                    ]
                    / item["unknown"]["count"],
                    "unknown_safe_rejection_accuracy": item["unknown"][
                        "safe_resolution_accuracy"
                    ],
                }
            )


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
        raise TrainingToolError("Run must complete training before evaluation")
    config = _run_config(run_dir)
    bundle = validate_data(config)
    if data_hashes(config) != manifest.get("data_sha256"):
        raise TrainingToolError("Current data files do not match the trained run")
    current_git = git_info(require_clean=True)
    warn_if_git_commit_mismatch(
        current_git["commit"],
        manifest.get("git", {}).get("commit"),
        operation="Evaluation",
    )

    if args.all_milestones:
        checkpoints = discover_checkpoints(run_dir)
        expected = list(config_value(config, "training.checkpoint_epochs"))
        if [epoch for epoch, _ in checkpoints] != expected:
            raise TrainingToolError(
                f"Expected milestone epochs {expected}, found {[epoch for epoch, _ in checkpoints]}"
            )
    else:
        checkpoint = args.checkpoint.resolve()
        from training_common import checkpoint_epoch

        try:
            checkpoint.relative_to(run_dir)
        except ValueError as error:
            raise TrainingToolError(
                "Checkpoint must be inside the selected run directory"
            ) from error
        checkpoints = [(checkpoint_epoch(checkpoint), checkpoint)]

    checkpoint_metrics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for epoch, checkpoint in checkpoints:
        print(f"Evaluating epoch {epoch}: {checkpoint}", flush=True)
        metric, checkpoint_failures = evaluate_adapter(
            checkpoint, epoch, config, manifest, bundle
        )
        metric["checkpoint"] = str(checkpoint.relative_to(run_dir))
        checkpoint_metrics.append(metric)
        failures.extend(checkpoint_failures)

    threshold = float(config_value(config, "evaluation.canonical_threshold"))
    alias_threshold = float(config_value(config, "evaluation.alias_threshold"))
    unknown_threshold = float(config_value(config, "evaluation.unknown_threshold"))
    selected = select_best_checkpoint(
        checkpoint_metrics, threshold, unknown_threshold
    )
    acceptance_passed = bool(
        selected is not None
        and float(selected["alias"]["generation_accuracy"]) >= alias_threshold
    )
    report = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "evaluated_at": utc_now(),
        "canonical_threshold": threshold,
        "alias_threshold": alias_threshold,
        "unknown_threshold": unknown_threshold,
        "checkpoints": checkpoint_metrics,
        "selection": (
            {
                "epoch": selected["epoch"],
                "checkpoint": selected["checkpoint"],
                "rule": "canonical and unknown safe-resolution thresholds must pass; then highest held-out alias safe-resolution accuracy, unknown accuracy, canonical accuracy, and earlier epoch",
            }
            if selected is not None
            else None
        ),
        "acceptance_passed": acceptance_passed,
    }
    atomic_write_json(run_dir / "metrics.json", report)
    _write_comparison(run_dir / "checkpoint_comparison.csv", checkpoint_metrics)
    _write_failures(run_dir / "evaluation_failures.csv", failures)

    manifest["status"] = "evaluated"
    manifest["evaluated_at"] = report["evaluated_at"]
    manifest["selection"] = report["selection"]
    atomic_write_json(manifest_path, manifest)
    if selected is None:
        raise TrainingToolError(
            "No checkpoint jointly reached canonical safe-resolution "
            f"{threshold:.2%} and unknown safe-rejection {unknown_threshold:.2%}"
        )
    if not acceptance_passed:
        raise TrainingToolError(
            "Best canonical/unknown-passing checkpoint reached alias safe resolution "
            f"{selected['alias']['generation_accuracy']:.2%}, below the "
            f"{alias_threshold:.2%} threshold"
        )
    print(
        f"Selected epoch {selected['epoch']}: alias="
        f"{selected['alias']['generation_accuracy']:.2%}, canonical="
        f"{selected['canonical']['generation_accuracy']:.2%}, unknown="
        f"{selected['unknown']['generation_accuracy']:.2%}"
    )
    print(f"Evaluation report: {run_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TrainingToolError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
