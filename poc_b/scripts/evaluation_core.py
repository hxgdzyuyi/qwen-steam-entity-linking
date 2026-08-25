#!/usr/bin/env python3
"""Tensor-only metrics shared by PoC B evaluation and publication checks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch

from steam_entity_classifier import FrozenPrototypeHead
from training_common import prediction_sha256


def metric_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    top1 = sum(bool(row["top1_correct"]) for row in records)
    top5 = sum(bool(row["top5_correct"]) for row in records)
    reciprocal_rank = sum(1.0 / int(row["rank"]) for row in records)
    return {
        "count": count,
        "top1_correct": top1,
        "top1_accuracy": top1 / count if count else 0.0,
        "top5_correct": top5,
        "top5_accuracy": top5 / count if count else 0.0,
        "mrr": reciprocal_rank / count if count else 0.0,
    }


def breakdown(
    records: Sequence[Mapping[str, Any]], key: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row[key])].append(row)
    return {name: metric_summary(rows) for name, rows in sorted(grouped.items())}


@torch.no_grad()
def predict_feature_rows(
    head: FrozenPrototypeHead,
    features: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
    classes: Sequence[Mapping[str, Any]],
    *,
    source: str,
    batch_size: int,
    diagnostic_top_k: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    if len(rows) != features.shape[0]:
        raise ValueError("evaluation rows and features differ in length")
    top_k = min(int(diagnostic_top_k), len(classes))
    records: list[dict[str, Any]] = []
    head.eval()
    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset : offset + batch_size]
        logits = head(features[offset : offset + batch_size].to(device)).float().cpu()
        probabilities = torch.softmax(logits, dim=-1)
        for row, row_logits, row_probabilities in zip(
            batch_rows, logits, probabilities
        ):
            expected_index = int(row["class_index"])
            # Stable class-index tie breaking keeps fingerprints deterministic.
            ordering = sorted(
                range(len(classes)), key=lambda index: (-float(row_logits[index]), index)
            )
            top_indices = ordering[:top_k]
            top_five_indices = ordering[: min(5, len(classes))]
            predicted_index = top_indices[0]
            rank = ordering.index(expected_index) + 1
            candidates = [
                {
                    "class_index": int(class_index),
                    "appid": int(classes[int(class_index)]["appid"]),
                    "canonical_name": str(
                        classes[int(class_index)]["canonical_name"]
                    ),
                    "confidence": float(row_probabilities[class_index]),
                }
                for class_index in top_indices
            ]
            records.append(
                {
                    "source": source,
                    "input": str(row["surface_form"]),
                    "model_input": str(row["model_input"]),
                    "expected_class_index": expected_index,
                    "expected_appid": int(row["appid"]),
                    "expected_canonical_name": str(row["canonical_name"]),
                    "predicted_class_index": predicted_index,
                    "predicted_appid": int(classes[predicted_index]["appid"]),
                    "predicted_canonical_name": str(
                        classes[predicted_index]["canonical_name"]
                    ),
                    "confidence": float(row_probabilities[predicted_index]),
                    "rank": rank,
                    "top1_correct": predicted_index == expected_index,
                    "top5_correct": expected_index in top_five_indices,
                    "cohort": str(row["cohort"]),
                    "type": str(row.get("type", "canonical")),
                    "prompt_style": str(row["prompt_style"]),
                    "top_k": candidates,
                }
            )
    return records


def checkpoint_metrics(
    *,
    epoch: int,
    head: FrozenPrototypeHead,
    canonical_features: torch.Tensor,
    alias_features: torch.Tensor,
    canonical_rows: Sequence[Mapping[str, Any]],
    alias_rows: Sequence[Mapping[str, Any]],
    classes: Sequence[Mapping[str, Any]],
    batch_size: int,
    diagnostic_top_k: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    canonical = predict_feature_rows(
        head,
        canonical_features,
        canonical_rows,
        classes,
        source="canonical",
        batch_size=batch_size,
        diagnostic_top_k=diagnostic_top_k,
        device=device,
    )
    alias = predict_feature_rows(
        head,
        alias_features,
        alias_rows,
        classes,
        source="alias",
        batch_size=batch_size,
        diagnostic_top_k=diagnostic_top_k,
        device=device,
    )
    all_records = canonical + alias
    return (
        {
            "epoch": int(epoch),
            "canonical": metric_summary(canonical),
            "alias": metric_summary(alias),
            "canonical_by_cohort": breakdown(canonical, "cohort"),
            "alias_by_cohort": breakdown(alias, "cohort"),
            "canonical_by_prompt_style": breakdown(canonical, "prompt_style"),
            "alias_by_prompt_style": breakdown(alias, "prompt_style"),
            "alias_by_type": breakdown(alias, "type"),
            "prediction_sha256": prediction_sha256(all_records),
        },
        all_records,
    )
