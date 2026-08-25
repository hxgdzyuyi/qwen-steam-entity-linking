#!/usr/bin/env python3
"""Validated FP32 hidden-feature cache for PoC B."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from steam_entity_classifier import stable_payload_sha256, tokenizer_sha256
from training_common import TrainingToolError, atomic_write_json, read_json, sha256_file


CACHE_SCHEMA_VERSION = 1
FEATURE_TENSOR_NAMES = {
    "train_features",
    "train_labels",
    "canonical_features",
    "canonical_labels",
    "alias_features",
    "alias_labels",
}


def stable_sha256(payload: Any) -> str:
    return stable_payload_sha256(payload)


def cache_identity(
    *,
    mode: str,
    model_id: str,
    model_revision: str,
    tokenizer_hash: str,
    data_sha256: Mapping[str, str],
    max_length: int,
    pooling: str,
    class_count: int,
    row_counts: Mapping[str, int],
) -> dict[str, Any]:
    if mode not in {"smoke", "full"}:
        raise TrainingToolError(f"Invalid feature-cache mode: {mode}")
    return {
        "mode": mode,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_sha256": tokenizer_hash,
        "data_sha256": dict(data_sha256),
        "max_length": int(max_length),
        "pooling": pooling,
        "cache_dtype": "float32",
        "class_count": int(class_count),
        "row_counts": {key: int(value) for key, value in row_counts.items()},
    }


def _validate_tensors(
    tensors: Mapping[str, torch.Tensor], identity: Mapping[str, Any]
) -> None:
    if set(tensors) != FEATURE_TENSOR_NAMES:
        raise TrainingToolError(
            f"Feature cache has invalid tensor names: {sorted(tensors)}"
        )
    row_counts = identity.get("row_counts")
    if not isinstance(row_counts, Mapping) or set(row_counts) != {
        "train",
        "canonical",
        "alias",
    }:
        raise TrainingToolError("Feature cache identity has invalid row counts")
    class_count = identity.get("class_count")
    if (
        not isinstance(class_count, int)
        or isinstance(class_count, bool)
        or class_count <= 0
    ):
        raise TrainingToolError("Feature cache identity has an invalid class count")
    hidden_size: int | None = None
    for role in ("train", "canonical", "alias"):
        features = tensors[f"{role}_features"]
        labels = tensors[f"{role}_labels"]
        expected_rows = row_counts[role]
        if (
            not isinstance(expected_rows, int)
            or isinstance(expected_rows, bool)
            or expected_rows < 0
        ):
            raise TrainingToolError(f"{role} feature row count is invalid")
        if features.ndim != 2 or features.shape[0] != expected_rows:
            raise TrainingToolError(f"{role} feature-cache shape is invalid")
        if features.dtype != torch.float32:
            raise TrainingToolError(f"{role} feature cache must be FP32")
        if labels.ndim != 1 or labels.shape[0] != expected_rows:
            raise TrainingToolError(f"{role} feature labels are invalid")
        if labels.dtype != torch.int64:
            raise TrainingToolError(f"{role} feature labels must be int64")
        hidden_size = hidden_size or int(features.shape[1])
        if int(features.shape[1]) != hidden_size:
            raise TrainingToolError("Feature roles have different hidden sizes")
        if labels.numel() and (
            int(labels.min()) < 0
            or int(labels.max()) >= class_count
        ):
            raise TrainingToolError(f"{role} labels are outside the class map")


def save_feature_cache(
    cache_dir: Path,
    tensors: Mapping[str, torch.Tensor],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise TrainingToolError("safetensors is required for feature caching") from error
    normalized = {
        key: value.detach().cpu().contiguous()
        for key, value in tensors.items()
    }
    _validate_tensors(normalized, identity)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = cache_dir / "features.safetensors"
    save_file(normalized, tensor_path)
    tensor_hash = sha256_file(tensor_path)
    fingerprint = stable_sha256(
        {"identity": dict(identity), "features_sha256": tensor_hash}
    )
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "identity": dict(identity),
        "hidden_size": int(normalized["train_features"].shape[1]),
        "features_sha256": tensor_hash,
        "cache_sha256": fingerprint,
        "roles": {
            "train": "canonical-only prototype/head training",
            "canonical": "evaluation-only",
            "alias": "held-out evaluation-only",
        },
    }
    atomic_write_json(cache_dir / "feature_cache.json", metadata)
    return metadata


def load_feature_cache(
    cache_dir: Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    expected_cache_sha256: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise TrainingToolError("safetensors is required for feature caching") from error
    metadata = read_json(cache_dir / "feature_cache.json")
    if not isinstance(metadata, dict) or metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise TrainingToolError("Feature-cache metadata has an invalid schema")
    identity = metadata.get("identity")
    if not isinstance(identity, dict):
        raise TrainingToolError("Feature-cache identity is missing")
    if expected_identity is not None and identity != dict(expected_identity):
        raise TrainingToolError("Feature cache does not match model, tokenizer, data, or mode")
    tensor_path = cache_dir / "features.safetensors"
    tensor_hash = sha256_file(tensor_path)
    if tensor_hash != metadata.get("features_sha256"):
        raise TrainingToolError("Feature-cache tensor hash differs from metadata")
    fingerprint = stable_sha256(
        {"identity": identity, "features_sha256": tensor_hash}
    )
    if fingerprint != metadata.get("cache_sha256"):
        raise TrainingToolError("Feature-cache fingerprint is invalid")
    if expected_cache_sha256 is not None and fingerprint != expected_cache_sha256:
        raise TrainingToolError("Feature cache differs from the trained run")
    try:
        tensors = load_file(tensor_path, device="cpu")
    except (OSError, RuntimeError) as error:
        raise TrainingToolError(f"Cannot read feature cache: {error}") from error
    _validate_tensors(tensors, identity)
    if int(tensors["train_features"].shape[1]) != int(metadata.get("hidden_size", -1)):
        raise TrainingToolError("Feature-cache hidden size differs from metadata")
    return dict(tensors), metadata
