#!/usr/bin/env python3
"""Dependency-light helpers shared by PoC A training and evaluation."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTITY_TOKEN_PATTERN = re.compile(r"^<GAME_([0-9]+)>$")


class TrainingToolError(RuntimeError):
    """Raised when a training-tool precondition is not satisfied."""


@dataclass(frozen=True)
class DataBundle:
    train_rows: list[dict[str, str]]
    alias_rows: list[dict[str, str]]
    special_tokens: list[str]
    canonical_by_token: dict[str, str]
    provenance_by_token: dict[str, dict[str, str]]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - exercised on cloud images
        raise TrainingToolError(
            "PyYAML is required to read the training config"
        ) from error

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise TrainingToolError(f"Cannot read config {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TrainingToolError("Training config must be a YAML object")

    required_sections = {
        "model",
        "runtime",
        "data",
        "lora",
        "training",
        "evaluation",
    }
    missing = required_sections.difference(payload)
    if missing:
        raise TrainingToolError(
            f"Training config is missing sections: {sorted(missing)}"
        )

    _validate_config(payload)
    return payload


def _validate_config(config: Mapping[str, Any]) -> None:
    required_values: tuple[tuple[str, type | tuple[type, ...]], ...] = (
        ("model.id", str),
        ("model.trust_remote_code", bool),
        ("runtime.target_image", str),
        ("runtime.expected_gpu_name", str),
        ("runtime.minimum_gpu_memory_gib", (int, float)),
        ("runtime.minimum_system_memory_gib", (int, float)),
        ("runtime.minimum_cpu_count", int),
        ("runtime.minimum_free_disk_gib", (int, float)),
        ("runtime.minimum_cached_free_disk_gib", (int, float)),
        ("runtime.minimum_publish_free_disk_gib", (int, float)),
        ("runtime.expected_torch_major_minor", str),
        ("runtime.expected_cuda_major_minor", str),
        ("data.train_path", str),
        ("data.alias_eval_path", str),
        ("data.entities_path", str),
        ("data.provenance_path", str),
        ("data.special_tokens_path", str),
        ("data.max_length", int),
        ("data.prompt_styles", list),
        ("data.expected_train_rows", int),
        ("data.expected_special_tokens", int),
        ("data.expected_alias_cases", int),
        ("data.expected_alias_rows", int),
        ("lora.r", int),
        ("lora.alpha", int),
        ("lora.dropout", (int, float)),
        ("lora.target_modules", (str, list)),
        ("lora.bias", str),
        ("training.output_root", str),
        ("training.seed", int),
        ("training.data_seed", int),
        ("training.full_epochs", int),
        ("training.smoke_epochs", int),
        ("training.smoke_samples", int),
        ("training.smoke_required_consecutive_perfect_epochs", int),
        ("training.per_device_batch_size", int),
        ("training.gradient_accumulation_steps", int),
        ("training.learning_rate", (int, float)),
        ("training.lr_scheduler_type", str),
        ("training.warmup_ratio", (int, float)),
        ("training.weight_decay", (int, float)),
        ("training.max_grad_norm", (int, float)),
        ("training.optim", str),
        ("training.checkpoint_epochs", list),
        ("training.keep_resume_state_only_latest", bool),
        ("training.logging_steps", int),
        ("training.require_clean_git", bool),
        ("evaluation.batch_size", int),
        ("evaluation.generation_max_new_tokens", int),
        ("evaluation.canonical_threshold", (int, float)),
        ("evaluation.alias_threshold", (int, float)),
    )
    for dotted_key, expected_type in required_values:
        value = config_value(config, dotted_key)
        if not isinstance(value, expected_type) or (
            isinstance(value, bool) and expected_type is not bool
        ):
            raise TrainingToolError(
                f"Config value {dotted_key} must be {expected_type}, got {value!r}"
            )

    if int(config_value(config, "data.max_length")) <= 2:
        raise TrainingToolError("data.max_length must be greater than 2")
    positive_integer_keys = (
        "training.full_epochs",
        "training.smoke_epochs",
        "training.smoke_samples",
        "training.smoke_required_consecutive_perfect_epochs",
        "training.per_device_batch_size",
        "training.gradient_accumulation_steps",
        "training.logging_steps",
        "evaluation.batch_size",
        "evaluation.generation_max_new_tokens",
        "runtime.minimum_cpu_count",
    )
    for dotted_key in positive_integer_keys:
        if int(config_value(config, dotted_key)) <= 0:
            raise TrainingToolError(f"{dotted_key} must be positive")
    if float(config_value(config, "training.learning_rate")) <= 0:
        raise TrainingToolError("training.learning_rate must be positive")
    for dotted_key in (
        "runtime.minimum_gpu_memory_gib",
        "runtime.minimum_system_memory_gib",
        "runtime.minimum_free_disk_gib",
        "runtime.minimum_cached_free_disk_gib",
        "runtime.minimum_publish_free_disk_gib",
    ):
        if float(config_value(config, dotted_key)) <= 0:
            raise TrainingToolError(f"{dotted_key} must be positive")
    if not 0.0 <= float(config_value(config, "lora.dropout")) < 1.0:
        raise TrainingToolError("lora.dropout must be in [0, 1)")
    if not 0.0 <= float(config_value(config, "training.warmup_ratio")) <= 1.0:
        raise TrainingToolError("training.warmup_ratio must be between 0 and 1")
    checkpoints = config_value(config, "training.checkpoint_epochs")
    full_epochs = int(config_value(config, "training.full_epochs"))
    if (
        not checkpoints
        or any(
            not isinstance(epoch, int) or isinstance(epoch, bool)
            for epoch in checkpoints
        )
        or sorted(set(checkpoints)) != checkpoints
        or checkpoints[0] <= 0
        or checkpoints[-1] > full_epochs
    ):
        raise TrainingToolError(
            "training.checkpoint_epochs must be unique sorted integers within full_epochs"
        )
    prompt_styles = config_value(config, "data.prompt_styles")
    if (
        not prompt_styles
        or any(not isinstance(style, str) or not style for style in prompt_styles)
        or len(set(prompt_styles)) != len(prompt_styles)
    ):
        raise TrainingToolError("data.prompt_styles must contain unique names")
    if int(config_value(config, "evaluation.generation_max_new_tokens")) != 1:
        raise TrainingToolError(
            "evaluation.generation_max_new_tokens must be 1 for entity classification"
        )
    for key in ("evaluation.canonical_threshold", "evaluation.alias_threshold"):
        threshold = float(config_value(config, key))
        if not 0.0 <= threshold <= 1.0:
            raise TrainingToolError(f"{key} must be between 0 and 1")
    revision = config.get("model", {}).get("revision")
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise TrainingToolError("model.revision must be null or a non-empty string")


def config_value(config: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for component in dotted_key.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise TrainingToolError(f"Missing config value: {dotted_key}")
        value = value[component]
    return value


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingToolError(f"Cannot read JSON {path}: {error}") from error


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TrainingToolError(
                        f"{path}:{line_number} must contain a JSON object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingToolError(f"Cannot read JSONL {path}: {error}") from error
    return rows


def _required_text(row: Mapping[str, Any], key: str, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TrainingToolError(f"{context} requires a non-empty {key}")
    return value


def _expected_count(config: Mapping[str, Any], key: str, actual: int) -> None:
    expected = config.get("data", {}).get(key)
    if expected is not None and actual != expected:
        raise TrainingToolError(f"Expected {expected} for {key}, found {actual}")


def validate_data(config: Mapping[str, Any]) -> DataBundle:
    train_path = project_path(config_value(config, "data.train_path"))
    alias_path = project_path(config_value(config, "data.alias_eval_path"))
    tokens_path = project_path(config_value(config, "data.special_tokens_path"))
    entities_path = project_path(config_value(config, "data.entities_path"))
    provenance_path = project_path(config_value(config, "data.provenance_path"))

    raw_train = read_jsonl(train_path)
    raw_alias = read_jsonl(alias_path)
    raw_tokens = read_json(tokens_path)
    if not isinstance(raw_tokens, list) or not raw_tokens:
        raise TrainingToolError("special_tokens.json must be a non-empty JSON list")
    if any(
        not isinstance(token, str) or ENTITY_TOKEN_PATTERN.fullmatch(token) is None
        for token in raw_tokens
    ):
        raise TrainingToolError("Every special token must match <GAME_APPID>")
    special_tokens = list(raw_tokens)
    if len(special_tokens) != len(set(special_tokens)):
        raise TrainingToolError("Duplicate special tokens found")

    train_rows: list[dict[str, str]] = []
    completions: list[str] = []
    train_styles_by_token: dict[str, set[str]] = {}
    prompt_styles = set(config_value(config, "data.prompt_styles"))
    for index, row in enumerate(raw_train, start=1):
        normalized = {
            key: _required_text(row, key, f"training row {index}")
            for key in ("input", "prompt", "completion", "prompt_style")
        }
        if set(row) != set(normalized):
            raise TrainingToolError(
                f"training row {index} contains unexpected or missing fields"
            )
        completion = normalized["completion"]
        if completion not in special_tokens:
            raise TrainingToolError(
                f"training row {index} uses unknown completion {completion}"
            )
        style = normalized["prompt_style"]
        if style not in prompt_styles:
            raise TrainingToolError(
                f"training row {index} uses unknown prompt style {style}"
            )
        styles = train_styles_by_token.setdefault(completion, set())
        if style in styles:
            raise TrainingToolError(
                f"training token {completion} repeats prompt style {style}"
            )
        styles.add(style)
        train_rows.append(normalized)
        completions.append(completion)
    if set(completions) != set(special_tokens):
        raise TrainingToolError("Training completions and special tokens do not match")
    incomplete_train_tokens = sorted(
        token
        for token in special_tokens
        if train_styles_by_token.get(token) != prompt_styles
    )
    if incomplete_train_tokens:
        raise TrainingToolError(
            "Every entity token must cover every configured prompt style; "
            f"incomplete tokens: {incomplete_train_tokens[:5]}"
        )

    canonical_by_token: dict[str, str] = {}
    try:
        with entities_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["canonical_name", "appid"]:
                raise TrainingToolError(
                    f"{entities_path} must contain canonical_name,appid columns"
                )
            for row in reader:
                name = _required_text(row, "canonical_name", "entity row")
                appid = _required_text(row, "appid", "entity row")
                if not appid.isdigit():
                    raise TrainingToolError(f"Invalid entity AppID: {appid}")
                token = f"<GAME_{int(appid)}>"
                if token in canonical_by_token:
                    raise TrainingToolError(f"Duplicate entity AppID: {appid}")
                canonical_by_token[token] = name
    except OSError as error:
        raise TrainingToolError(
            f"Cannot read entities {entities_path}: {error}"
        ) from error
    if set(canonical_by_token) != set(special_tokens):
        raise TrainingToolError("Entity CSV and special tokens do not match")
    for row in train_rows:
        expected_name = canonical_by_token[row["completion"]]
        if row["input"].casefold() != expected_name.casefold():
            raise TrainingToolError(
                f"Training input does not match canonical entity name: {row['input']}"
            )

    alias_rows: list[dict[str, str]] = []
    alias_identity: dict[str, tuple[str, str]] = {}
    alias_styles_by_input: dict[str, set[str]] = {}
    for index, row in enumerate(raw_alias, start=1):
        normalized = {
            key: _required_text(row, key, f"alias row {index}")
            for key in ("input", "prompt", "expected", "type", "prompt_style")
        }
        if set(row) != set(normalized):
            raise TrainingToolError(
                f"alias row {index} contains unexpected or missing fields"
            )
        expected = normalized["expected"]
        if expected not in canonical_by_token:
            raise TrainingToolError(
                f"alias row {index} targets unknown token {expected}"
            )
        input_key = normalized["input"].casefold()
        if input_key == canonical_by_token[expected].casefold():
            raise TrainingToolError(
                f"Alias leaks canonical training name: {normalized['input']}"
            )
        identity = (expected, normalized["type"])
        previous_identity = alias_identity.setdefault(input_key, identity)
        if previous_identity != identity:
            raise TrainingToolError(
                f"Alias input has inconsistent target or type: {normalized['input']}"
            )
        style = normalized["prompt_style"]
        if style not in prompt_styles:
            raise TrainingToolError(
                f"alias row {index} uses unknown prompt style {style}"
            )
        styles = alias_styles_by_input.setdefault(input_key, set())
        if style in styles:
            raise TrainingToolError(
                f"Alias input repeats prompt style {style}: {normalized['input']}"
            )
        styles.add(style)
        alias_rows.append(normalized)
    incomplete_aliases = sorted(
        input_key
        for input_key, styles in alias_styles_by_input.items()
        if styles != prompt_styles
    )
    if incomplete_aliases:
        raise TrainingToolError(
            "Every alias input must cover every configured prompt style; "
            f"incomplete inputs: {incomplete_aliases[:5]}"
        )

    provenance_by_token: dict[str, dict[str, str]] = {}
    try:
        with provenance_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                appid = _required_text(row, "appid", "provenance row")
                if not appid.isdigit():
                    raise TrainingToolError(f"Invalid provenance AppID: {appid}")
                if row.get("item_type") != "game":
                    raise TrainingToolError(f"Non-game item found for AppID {appid}")
                token = f"<GAME_{int(appid)}>"
                if token in provenance_by_token:
                    raise TrainingToolError(f"Duplicate provenance AppID: {appid}")
                provenance_by_token[token] = dict(row)
    except OSError as error:
        raise TrainingToolError(
            f"Cannot read provenance {provenance_path}: {error}"
        ) from error
    if set(provenance_by_token) != set(special_tokens):
        raise TrainingToolError("Provenance and special tokens do not match")

    _expected_count(config, "expected_train_rows", len(train_rows))
    _expected_count(config, "expected_special_tokens", len(special_tokens))
    _expected_count(config, "expected_alias_cases", len(alias_identity))
    _expected_count(config, "expected_alias_rows", len(alias_rows))
    if int(config_value(config, "training.smoke_samples")) > len(train_rows):
        raise TrainingToolError("training.smoke_samples exceeds the training dataset")
    expected_cohorts = config.get("data", {}).get("expected_cohorts", {})
    for cohort, expected in expected_cohorts.items():
        actual = sum(
            row.get("cohort") == cohort for row in provenance_by_token.values()
        )
        if actual != expected:
            raise TrainingToolError(
                f"Expected {expected} {cohort} entities, found {actual}"
            )

    return DataBundle(
        train_rows=train_rows,
        alias_rows=alias_rows,
        special_tokens=special_tokens,
        canonical_by_token=canonical_by_token,
        provenance_by_token=provenance_by_token,
    )


def prepare_tokenizer(tokenizer: Any, special_tokens: Sequence[str]) -> list[int]:
    original_size = len(tokenizer)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(special_tokens)}
    )
    if added != len(special_tokens):
        raise TrainingToolError(
            f"Expected to add {len(special_tokens)} new tokens, tokenizer added {added}"
        )
    token_ids = validate_entity_tokens(tokenizer, special_tokens)
    if len(token_ids) != len(set(token_ids)) or any(
        token_id < original_size for token_id in token_ids
    ):
        raise TrainingToolError("Entity tokens did not receive unique new token IDs")
    if tokenizer.eos_token_id is None:
        raise TrainingToolError("Tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return token_ids


def validate_entity_tokens(tokenizer: Any, special_tokens: Sequence[str]) -> list[int]:
    token_ids = [
        int(tokenizer.convert_tokens_to_ids(token)) for token in special_tokens
    ]
    if len(token_ids) != len(set(token_ids)):
        raise TrainingToolError("Entity tokens do not have unique token IDs")
    for token, token_id in zip(special_tokens, token_ids):
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id]:
            raise TrainingToolError(f"Entity token is not atomic: {token}")
    if tokenizer.eos_token_id is None:
        raise TrainingToolError("Tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return token_ids


def encode_training_row(
    tokenizer: Any, row: Mapping[str, str], max_length: int
) -> dict[str, list[int]]:
    prompt_ids = list(tokenizer.encode(row["prompt"], add_special_tokens=False))
    completion_ids = list(tokenizer.encode(row["completion"], add_special_tokens=False))
    if len(completion_ids) != 1:
        raise TrainingToolError(
            f"Completion must be one token, got {completion_ids} for {row['completion']}"
        )
    input_ids = prompt_ids + completion_ids
    if len(input_ids) > max_length:
        raise TrainingToolError(
            f"Training sample is {len(input_ids)} tokens, exceeding max_length={max_length}"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }


class CompletionOnlyCollator:
    """Right-pad batches while masking every prompt token from the loss."""

    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - cloud dependency
            raise TrainingToolError("PyTorch is required for collation") from error
        encoded = [
            encode_training_row(self.tokenizer, feature, self.max_length)
            for feature in features
        ]
        width = max(len(row["input_ids"]) for row in encoded)
        pad_id = int(self.tokenizer.pad_token_id)
        batch: dict[str, list[list[int]]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        for row in encoded:
            padding = width - len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [pad_id] * padding)
            batch["attention_mask"].append(row["attention_mask"] + [0] * padding)
            batch["labels"].append(row["labels"] + [-100] * padding)
        return {
            key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()
        }


class RawRowsDataset:
    def __init__(self, rows: Sequence[Mapping[str, str]]) -> None:
        self.rows = [dict(row) for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


def ensure_prompt_lengths(
    tokenizer: Any, rows: Iterable[Mapping[str, str]], max_length: int
) -> None:
    for index, row in enumerate(rows, start=1):
        length = len(tokenizer.encode(row["prompt"], add_special_tokens=False))
        if length > max_length:
            raise TrainingToolError(
                f"Evaluation prompt {index} is {length} tokens, exceeding {max_length}"
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    keys = (
        "train_path",
        "alias_eval_path",
        "entities_path",
        "provenance_path",
        "special_tokens_path",
    )
    return {
        key: sha256_file(project_path(config_value(config, f"data.{key}")))
        for key in keys
    }


def _git_command(arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise TrainingToolError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def git_info(require_clean: bool = True) -> dict[str, Any]:
    status = _git_command(["status", "--porcelain"])
    if require_clean and status:
        warnings.warn(
            "Git checkout is not clean; continuing because Git provenance checks "
            "are non-blocking. Artifacts may not be exactly reproducible.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "commit": _git_command(["rev-parse", "HEAD"]),
        "branch": _git_command(["branch", "--show-current"]),
        "remote": _git_command(["remote", "get-url", "origin"]),
        "clean": not bool(status),
    }


def warn_if_git_commit_mismatch(
    current_commit: str, trained_commit: object, *, operation: str
) -> None:
    """Warn when code provenance differs without blocking an expensive operation."""

    if current_commit == trained_commit:
        return
    warnings.warn(
        f"{operation}: current Git commit {current_commit!r} does not match the "
        f"trained run commit {trained_commit!r}; continuing because Git provenance "
        "checks are non-blocking. Results may not be exactly reproducible.",
        RuntimeWarning,
        stacklevel=2,
    )


def dependency_versions(names: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, path)


def make_run_id(mode: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{mode}"


def checkpoint_epoch(path: Path) -> int:
    metadata_path = path / "checkpoint_meta.json"
    if metadata_path.exists():
        value = read_json(metadata_path)
        return int(value["epoch"])
    state_path = path / "trainer_state.json"
    if state_path.exists():
        value = read_json(state_path)
        return int(round(float(value["epoch"])))
    raise TrainingToolError(f"Checkpoint has no epoch metadata: {path}")


def discover_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    checkpoints_dir = run_dir / "checkpoints"
    found: list[tuple[int, Path]] = []
    for path in checkpoints_dir.glob("checkpoint-*"):
        if path.is_dir():
            found.append((checkpoint_epoch(path), path.resolve()))
    by_epoch: dict[int, Path] = {}
    for epoch, path in sorted(found):
        if epoch in by_epoch:
            raise TrainingToolError(f"Multiple checkpoints found for epoch {epoch}")
        by_epoch[epoch] = path
    return sorted(by_epoch.items())


def select_best_checkpoint(
    checkpoint_metrics: Sequence[Mapping[str, Any]], canonical_threshold: float
) -> Mapping[str, Any] | None:
    eligible = [
        item
        for item in checkpoint_metrics
        if float(item["canonical"]["generation_accuracy"]) >= canonical_threshold
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda item: (
            -float(item["alias"]["generation_accuracy"]),
            -float(item["canonical"]["generation_accuracy"]),
            int(item["epoch"]),
        ),
    )
