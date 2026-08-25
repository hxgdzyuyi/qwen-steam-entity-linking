#!/usr/bin/env python3
"""Evaluate one or all entity-linking LoRA checkpoints from a cloud run."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

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
)


METRICS_SCHEMA_VERSION = 2


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
            "Cloud evaluation dependencies are missing; install requirements-cloud.txt"
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


def _extract_entity_token(text: str) -> str:
    match = re.search(r"<GAME_[0-9]+>", text)
    return match.group(0) if match else ""


def generation_is_exact(
    output_ids: Sequence[int], expected_id: int, eos_token_id: int, pad_token_id: int
) -> bool:
    meaningful = [
        int(token_id)
        for token_id in output_ids
        if int(token_id) not in {eos_token_id, pad_token_id}
    ]
    return meaningful == [expected_id]


def _metric_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    next_correct = sum(row["next_token_correct"] for row in records)
    generation_correct = sum(row["generation_correct"] for row in records)
    return {
        "count": count,
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


def _predict_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, str]],
    expected_key: str,
    source: str,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    canonical_by_token: Mapping[str, str],
    provenance_by_token: Mapping[str, Mapping[str, str]],
    torch: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    try:
        with torch.inference_mode():
            for offset in range(0, len(rows), batch_size):
                batch_rows = rows[offset : offset + batch_size]
                prompts = [row["prompt"] for row in batch_rows]
                encoded = tokenizer(
                    prompts,
                    add_special_tokens=False,
                    padding=True,
                    return_tensors="pt",
                )
                prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
                if max(prompt_lengths) > max_length:
                    raise TrainingToolError(
                        f"Evaluation prompt exceeds max_length={max_length}"
                    )
                encoded = {
                    key: value.to(model.device) for key, value in encoded.items()
                }
                prompt_width = int(encoded["input_ids"].shape[1])
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=int(tokenizer.pad_token_id),
                    eos_token_id=int(tokenizer.eos_token_id),
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                if not generated.scores:
                    raise TrainingToolError("Generation returned no token scores")
                next_ids = generated.scores[0].argmax(dim=-1).tolist()
                generated_ids = generated.sequences[:, prompt_width:].tolist()

                for row, next_id, output_ids in zip(
                    batch_rows, next_ids, generated_ids
                ):
                    expected = row[expected_key]
                    expected_ids = tokenizer.encode(expected, add_special_tokens=False)
                    if len(expected_ids) != 1:
                        raise TrainingToolError(
                            f"Expected token is not atomic during evaluation: {expected}"
                        )
                    next_token = str(tokenizer.convert_ids_to_tokens(int(next_id)))
                    output_text = tokenizer.decode(
                        output_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    generated_token = _extract_entity_token(output_text)
                    provenance = provenance_by_token[expected]
                    record = {
                        "source": source,
                        "input": row.get("input", canonical_by_token[expected]),
                        "prompt": row["prompt"],
                        "expected": expected,
                        "predicted_next_token": next_token,
                        "generated_text": output_text,
                        "generated_token": generated_token,
                        "next_token_correct": int(next_id) == int(expected_ids[0]),
                        "generation_correct": generation_is_exact(
                            output_ids,
                            int(expected_ids[0]),
                            int(tokenizer.eos_token_id),
                            int(tokenizer.pad_token_id),
                        ),
                        "cohort": provenance.get("cohort", "unknown"),
                        "type": row.get("type", "canonical"),
                        "prompt_style": row.get("prompt_style", "training_prompt"),
                    }
                    records.append(record)
    finally:
        tokenizer.padding_side = previous_padding_side
    return records


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
    validate_entity_tokens(tokenizer, bundle.special_tokens)
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
    canonical_records = _predict_rows(
        model,
        tokenizer,
        bundle.train_rows,
        "completion",
        "canonical",
        batch_size,
        max_length,
        max_new_tokens,
        bundle.canonical_by_token,
        bundle.provenance_by_token,
        torch,
    )
    alias_records = _predict_rows(
        model,
        tokenizer,
        bundle.alias_rows,
        "expected",
        "alias",
        batch_size,
        max_length,
        max_new_tokens,
        bundle.canonical_by_token,
        bundle.provenance_by_token,
        torch,
    )
    metric = {
        "epoch": epoch,
        "checkpoint": str(checkpoint),
        "canonical": _metric_summary(canonical_records),
        "alias": _metric_summary(alias_records),
        "canonical_by_cohort": _breakdown(canonical_records, "cohort"),
        "alias_by_type": _breakdown(alias_records, "type"),
        "alias_by_prompt_style": _breakdown(alias_records, "prompt_style"),
        "prediction_sha256": prediction_sha256(
            canonical_records + alias_records
        ),
    }
    failures = [
        {"epoch": epoch, **record}
        for record in canonical_records + alias_records
        if not record["next_token_correct"] or not record["generation_correct"]
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
        "expected",
        "predicted_next_token",
        "generated_token",
        "generated_text",
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
                "canonical_next_token_accuracy",
                "canonical_generation_accuracy",
                "alias_next_token_accuracy",
                "alias_generation_accuracy",
            ),
        )
        writer.writeheader()
        for item in metrics:
            writer.writerow(
                {
                    "epoch": item["epoch"],
                    "canonical_next_token_accuracy": item["canonical"][
                        "next_token_accuracy"
                    ],
                    "canonical_generation_accuracy": item["canonical"][
                        "generation_accuracy"
                    ],
                    "alias_next_token_accuracy": item["alias"]["next_token_accuracy"],
                    "alias_generation_accuracy": item["alias"]["generation_accuracy"],
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
    if current_git["commit"] != manifest.get("git", {}).get("commit"):
        raise TrainingToolError("Current Git commit does not match the trained run")

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
    selected = select_best_checkpoint(checkpoint_metrics, threshold)
    report = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "evaluated_at": utc_now(),
        "canonical_threshold": threshold,
        "checkpoints": checkpoint_metrics,
        "selection": (
            {
                "epoch": selected["epoch"],
                "checkpoint": selected["checkpoint"],
                "rule": "highest alias generation accuracy among canonical-passing checkpoints; then canonical accuracy; then earlier epoch",
            }
            if selected is not None
            else None
        ),
        "acceptance_passed": selected is not None,
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
            f"No checkpoint reached canonical accuracy threshold {threshold:.2%}"
        )
    print(
        f"Selected epoch {selected['epoch']}: alias="
        f"{selected['alias']['generation_accuracy']:.2%}, canonical="
        f"{selected['canonical']['generation_accuracy']:.2%}"
    )
    print(f"Evaluation report: {run_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TrainingToolError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
