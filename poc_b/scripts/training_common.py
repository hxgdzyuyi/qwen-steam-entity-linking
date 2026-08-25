#!/usr/bin/env python3
"""Dependency-light contracts shared by PoC B tools."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


POC_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = POC_ROOT.parent


class TrainingToolError(RuntimeError):
    """Raised when a PoC B tool precondition is not satisfied."""


@dataclass(frozen=True)
class ClassEntry:
    class_index: int
    appid: int
    canonical_name: str
    cohort: str


@dataclass(frozen=True)
class DataBundle:
    classes: list[ClassEntry]
    train_rows: list[dict[str, Any]]
    canonical_rows: list[dict[str, Any]]
    alias_rows: list[dict[str, Any]]
    prompt_styles: tuple[str, ...]

    @property
    def class_count(self) -> int:
        return len(self.classes)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def config_value(config: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for component in dotted_key.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise TrainingToolError(f"Missing config value: {dotted_key}")
        value = value[component]
    return value


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else POC_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise TrainingToolError("PyYAML is required to read config files") from error
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
        "features",
        "classifier",
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
    required: tuple[tuple[str, type | tuple[type, ...]], ...] = (
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
        ("data.canonical_eval_path", str),
        ("data.alias_eval_path", str),
        ("data.class_map_path", str),
        ("data.data_manifest_path", str),
        ("data.entities_path", str),
        ("data.provenance_path", str),
        ("data.alias_source_path", str),
        ("data.poc_a_reference_path", str),
        ("data.max_length", int),
        ("data.expected_classes", int),
        ("data.expected_train_rows", int),
        ("data.expected_canonical_rows", int),
        ("data.expected_alias_rows", int),
        ("data.expected_prompt_styles", int),
        ("features.pooling", str),
        ("features.extraction_batch_size", int),
        ("features.cache_dtype", str),
        ("classifier.bottleneck_dim", int),
        ("classifier.temperature", (int, float)),
        ("classifier.prototype_anchor_weight", (int, float)),
        ("training.output_root", str),
        ("training.seed", int),
        ("training.data_seed", int),
        ("training.full_epochs", int),
        ("training.smoke_epochs", int),
        ("training.smoke_classes", int),
        ("training.batch_size", int),
        ("training.learning_rate", (int, float)),
        ("training.lr_scheduler_type", str),
        ("training.warmup_ratio", (int, float)),
        ("training.weight_decay", (int, float)),
        ("training.max_grad_norm", (int, float)),
        ("training.checkpoint_epochs", list),
        ("training.logging_steps", int),
        ("training.require_clean_git", bool),
        ("evaluation.batch_size", int),
        ("evaluation.canonical_threshold", (int, float)),
        ("evaluation.diagnostic_top_k", int),
    )
    for dotted_key, expected_type in required:
        value = config_value(config, dotted_key)
        if not isinstance(value, expected_type) or (
            isinstance(value, bool) and expected_type is not bool
        ):
            raise TrainingToolError(
                f"Config value {dotted_key} has invalid type: {value!r}"
            )
    positive_ints = (
        "runtime.minimum_cpu_count",
        "data.max_length",
        "data.expected_classes",
        "data.expected_train_rows",
        "data.expected_canonical_rows",
        "data.expected_alias_rows",
        "data.expected_prompt_styles",
        "features.extraction_batch_size",
        "classifier.bottleneck_dim",
        "training.full_epochs",
        "training.smoke_epochs",
        "training.smoke_classes",
        "training.batch_size",
        "training.logging_steps",
        "evaluation.batch_size",
        "evaluation.diagnostic_top_k",
    )
    for dotted_key in positive_ints:
        if int(config_value(config, dotted_key)) <= 0:
            raise TrainingToolError(f"{dotted_key} must be positive")
    for dotted_key in (
        "runtime.minimum_gpu_memory_gib",
        "runtime.minimum_system_memory_gib",
        "runtime.minimum_free_disk_gib",
        "runtime.minimum_cached_free_disk_gib",
        "runtime.minimum_publish_free_disk_gib",
        "classifier.temperature",
        "training.learning_rate",
        "training.max_grad_norm",
    ):
        if float(config_value(config, dotted_key)) <= 0:
            raise TrainingToolError(f"{dotted_key} must be positive")
    if float(config_value(config, "classifier.prototype_anchor_weight")) < 0:
        raise TrainingToolError("classifier.prototype_anchor_weight cannot be negative")
    if not 0 <= float(config_value(config, "training.warmup_ratio")) <= 1:
        raise TrainingToolError("training.warmup_ratio must be between 0 and 1")
    if not 0 <= float(config_value(config, "evaluation.canonical_threshold")) <= 1:
        raise TrainingToolError(
            "evaluation.canonical_threshold must be between 0 and 1"
        )
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
            "training.checkpoint_epochs must be sorted unique epochs within full_epochs"
        )
    if config_value(config, "features.pooling") != "last_non_padding":
        raise TrainingToolError("PoC B only supports last_non_padding pooling")
    if config_value(config, "features.cache_dtype") != "float32":
        raise TrainingToolError("PoC B feature caches must use float32")
    fixed_values = {
        "model.id": "Qwen/Qwen3-8B-Base",
        "model.trust_remote_code": False,
        "data.expected_classes": 1000,
        "data.expected_train_rows": 6000,
        "data.expected_canonical_rows": 1000,
        "data.expected_alias_rows": 184,
        "data.expected_prompt_styles": 6,
        "data.max_length": 256,
        "classifier.bottleneck_dim": 256,
        "classifier.temperature": 0.05,
        "classifier.prototype_anchor_weight": 0.01,
        "training.seed": 42,
        "training.data_seed": 42,
        "training.full_epochs": 20,
        "training.smoke_epochs": 20,
        "training.smoke_classes": 32,
        "training.batch_size": 256,
        "training.learning_rate": 0.001,
        "training.lr_scheduler_type": "cosine",
        "training.warmup_ratio": 0.05,
        "training.checkpoint_epochs": [1, 3, 5, 10, 20],
        "evaluation.canonical_threshold": 0.95,
        "evaluation.diagnostic_top_k": 5,
    }
    for dotted_key, expected in fixed_values.items():
        if config_value(config, dotted_key) != expected:
            raise TrainingToolError(
                f"PoC B contract fixes {dotted_key}={expected!r}"
            )
    revision = config["model"].get("revision")
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise TrainingToolError("model.revision must be null or a non-empty revision")


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


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise TrainingToolError(f"Cannot hash {path}: {error}") from error
    return digest.hexdigest()


def data_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    keys = (
        "train_path",
        "canonical_eval_path",
        "alias_eval_path",
        "class_map_path",
        "data_manifest_path",
        "entities_path",
        "provenance_path",
        "alias_source_path",
        "poc_a_reference_path",
    )
    return {
        key: sha256_file(project_path(config_value(config, f"data.{key}")))
        for key in keys
    }


def _required_text(row: Mapping[str, Any], key: str, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TrainingToolError(f"{context}.{key} must be non-empty text")
    return value


def _validate_row(
    row: Mapping[str, Any],
    classes: Sequence[ClassEntry],
    context: str,
    *,
    require_type: bool,
) -> dict[str, Any]:
    expected_fields = {
        "surface_form",
        "model_input",
        "class_index",
        "appid",
        "canonical_name",
        "cohort",
        "prompt_style",
    }
    if require_type:
        expected_fields.add("type")
    if set(row) != expected_fields:
        raise TrainingToolError(
            f"{context} has unexpected fields: {sorted(set(row) ^ expected_fields)}"
        )
    class_index = row.get("class_index")
    if (
        not isinstance(class_index, int)
        or isinstance(class_index, bool)
        or not 0 <= class_index < len(classes)
    ):
        raise TrainingToolError(f"{context}.class_index is invalid")
    entity = classes[class_index]
    if row.get("appid") != entity.appid:
        raise TrainingToolError(f"{context} AppID does not match class map")
    if row.get("canonical_name") != entity.canonical_name:
        raise TrainingToolError(f"{context} canonical name does not match class map")
    if row.get("cohort") != entity.cohort:
        raise TrainingToolError(f"{context} cohort does not match class map")
    surface_form = _required_text(row, "surface_form", context)
    model_input = _required_text(row, "model_input", context)
    if not model_input.endswith(surface_form):
        raise TrainingToolError(f"{context} model_input must end with surface_form")
    _required_text(row, "prompt_style", context)
    if require_type:
        _required_text(row, "type", context)
    return dict(row)


def validate_data(config: Mapping[str, Any]) -> DataBundle:
    class_payload = read_json(
        project_path(config_value(config, "data.class_map_path"))
    )
    if (
        not isinstance(class_payload, dict)
        or class_payload.get("schema_version") != 1
        or class_payload.get("ordering") != "numeric_appid_ascending"
        or not isinstance(class_payload.get("classes"), list)
    ):
        raise TrainingToolError("class_map.json has an invalid schema")
    classes: list[ClassEntry] = []
    seen_names: set[str] = set()
    seen_appids: set[int] = set()
    for expected_index, row in enumerate(class_payload["classes"]):
        if not isinstance(row, dict) or set(row) != {
            "class_index",
            "appid",
            "canonical_name",
            "cohort",
        }:
            raise TrainingToolError(f"class map row {expected_index} is invalid")
        if row["class_index"] != expected_index:
            raise TrainingToolError("class indices must be contiguous and ordered")
        if not isinstance(row["appid"], int) or isinstance(row["appid"], bool):
            raise TrainingToolError(f"class map AppID {expected_index} is invalid")
        if row["appid"] in seen_appids:
            raise TrainingToolError(f"duplicate class-map AppID: {row['appid']}")
        seen_appids.add(row["appid"])
        canonical_name = _required_text(
            row, "canonical_name", f"class map row {expected_index}"
        )
        if canonical_name.casefold() in seen_names:
            raise TrainingToolError(f"duplicate class name: {canonical_name}")
        seen_names.add(canonical_name.casefold())
        classes.append(
            ClassEntry(
                class_index=expected_index,
                appid=row["appid"],
                canonical_name=canonical_name,
                cohort=_required_text(
                    row, "cohort", f"class map row {expected_index}"
                ),
            )
        )
    if [item.appid for item in classes] != sorted(item.appid for item in classes):
        raise TrainingToolError("class map must be sorted by numeric AppID")
    if len(classes) != int(config_value(config, "data.expected_classes")):
        raise TrainingToolError("class count does not match config")

    raw_train = read_jsonl(project_path(config_value(config, "data.train_path")))
    raw_canonical = read_jsonl(
        project_path(config_value(config, "data.canonical_eval_path"))
    )
    raw_alias = read_jsonl(
        project_path(config_value(config, "data.alias_eval_path"))
    )
    train_rows = [
        _validate_row(row, classes, f"train row {index}", require_type=False)
        for index, row in enumerate(raw_train, start=1)
    ]
    canonical_rows = [
        _validate_row(
            row, classes, f"canonical row {index}", require_type=True
        )
        for index, row in enumerate(raw_canonical, start=1)
    ]
    alias_rows = [
        _validate_row(row, classes, f"alias row {index}", require_type=True)
        for index, row in enumerate(raw_alias, start=1)
    ]
    expected_counts = {
        "expected_train_rows": len(train_rows),
        "expected_canonical_rows": len(canonical_rows),
        "expected_alias_rows": len(alias_rows),
    }
    for key, actual in expected_counts.items():
        if actual != int(config_value(config, f"data.{key}")):
            raise TrainingToolError(f"{key} does not match config: {actual}")

    manifest = read_json(
        project_path(config_value(config, "data.data_manifest_path"))
    )
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise TrainingToolError("data manifest has an invalid schema")
    if manifest.get("canonical_only_training") is not True:
        raise TrainingToolError("PoC B training must be canonical-only")
    expected_source_hashes = {
        "entities": sha256_file(
            project_path(config_value(config, "data.entities_path"))
        ),
        "provenance": sha256_file(
            project_path(config_value(config, "data.provenance_path"))
        ),
        "alias_source": sha256_file(
            project_path(config_value(config, "data.alias_source_path"))
        ),
    }
    expected_derived_hashes = {
        "class_map": sha256_file(
            project_path(config_value(config, "data.class_map_path"))
        ),
        "train": sha256_file(project_path(config_value(config, "data.train_path"))),
        "canonical_eval": sha256_file(
            project_path(config_value(config, "data.canonical_eval_path"))
        ),
        "alias_eval": sha256_file(
            project_path(config_value(config, "data.alias_eval_path"))
        ),
    }
    if manifest.get("source_sha256") != expected_source_hashes:
        raise TrainingToolError("data manifest source hashes differ from current inputs")
    if manifest.get("derived_sha256") != expected_derived_hashes:
        raise TrainingToolError("data manifest hashes differ from current derived data")
    prompt_styles = tuple(manifest.get("prompt_styles", []))
    if len(prompt_styles) != int(
        config_value(config, "data.expected_prompt_styles")
    ):
        raise TrainingToolError("prompt style count does not match config")
    per_class_styles: dict[int, set[str]] = {
        entity.class_index: set() for entity in classes
    }
    for row in train_rows:
        if row["surface_form"] != row["canonical_name"]:
            raise TrainingToolError("training row contains a non-canonical surface")
        per_class_styles[row["class_index"]].add(row["prompt_style"])
    if any(styles != set(prompt_styles) for styles in per_class_styles.values()):
        raise TrainingToolError("every class must contain every prompt style once")
    if len(canonical_rows) != len(classes) or {
        row["class_index"] for row in canonical_rows
    } != set(range(len(classes))):
        raise TrainingToolError("canonical evaluation must contain every class once")
    if any(
        row["type"] != "canonical" or row["prompt_style"] != "raw"
        for row in canonical_rows
    ):
        raise TrainingToolError("canonical evaluation must use the raw canonical style")
    alias_inputs: set[str] = set()
    canonical_inputs = {row["surface_form"].casefold() for row in train_rows}
    for row in alias_rows:
        if row["prompt_style"] not in prompt_styles:
            raise TrainingToolError("alias evaluation contains an unknown prompt style")
        key = row["surface_form"].casefold()
        if key in canonical_inputs:
            raise TrainingToolError(
                f"alias evaluation leaks a canonical training name: {row['surface_form']}"
            )
        if key in alias_inputs:
            raise TrainingToolError(f"duplicate alias input: {row['surface_form']}")
        alias_inputs.add(key)
    alias_type_counts: dict[str, int] = {}
    for row in alias_rows:
        case_type = str(row["type"])
        alias_type_counts[case_type] = alias_type_counts.get(case_type, 0) + 1
    if alias_type_counts != manifest.get("alias_type_counts"):
        raise TrainingToolError("alias type counts differ from the data manifest")
    cohort_counts: dict[str, int] = {}
    for entity in classes:
        cohort_counts[entity.cohort] = cohort_counts.get(entity.cohort, 0) + 1
    if cohort_counts != dict(config_value(config, "data.expected_cohorts")):
        raise TrainingToolError(
            f"cohort counts do not match config: {cohort_counts}"
        )
    return DataBundle(
        classes=classes,
        train_rows=train_rows,
        canonical_rows=canonical_rows,
        alias_rows=alias_rows,
        prompt_styles=prompt_styles,
    )


def subset_bundle(bundle: DataBundle, class_count: int) -> DataBundle:
    if not 0 < class_count <= bundle.class_count:
        raise TrainingToolError("smoke class count is outside the class map")
    return DataBundle(
        classes=bundle.classes[:class_count],
        train_rows=[
            row for row in bundle.train_rows if row["class_index"] < class_count
        ],
        canonical_rows=[
            row
            for row in bundle.canonical_rows
            if row["class_index"] < class_count
        ],
        alias_rows=[
            row for row in bundle.alias_rows if row["class_index"] < class_count
        ],
        prompt_styles=bundle.prompt_styles,
    )


def class_map_payload(classes: Sequence[ClassEntry]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ordering": "numeric_appid_ascending",
        "classes": [
            {
                "class_index": item.class_index,
                "appid": item.appid,
                "canonical_name": item.canonical_name,
                "cohort": item.cohort,
            }
            for item in classes
        ],
    }


def _git_command(arguments: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
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
    status = _git_command(["status", "--short"])
    commit = _git_command(["rev-parse", "HEAD"])
    branch = _git_command(["rev-parse", "--abbrev-ref", "HEAD"])
    remote = _git_command(["config", "--get", "remote.origin.url"])
    clean = not status
    if require_clean and not clean:
        warnings.warn(
            "Git checkout is not clean; the run will record dirty=true",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "commit": commit,
        "branch": branch,
        "remote": remote,
        "clean": clean,
        "status": status.splitlines(),
    }


def warn_if_git_commit_mismatch(
    current_commit: str, trained_commit: str | None, operation: str
) -> None:
    if trained_commit and current_commit != trained_commit:
        warnings.warn(
            f"{operation} Git commit {current_commit} does not match "
            f"training commit {trained_commit}",
            RuntimeWarning,
            stacklevel=2,
        )


def dependency_versions(packages: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def discover_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for path in (run_dir / "checkpoints").glob("epoch-*"):
        suffix = path.name.removeprefix("epoch-")
        if suffix.isdigit() and (path / "classifier.safetensors").is_file():
            checkpoints.append((int(suffix), path))
    return sorted(checkpoints)


def checkpoint_epoch(path: Path) -> int:
    suffix = path.name.removeprefix("epoch-")
    if not suffix.isdigit():
        raise TrainingToolError(f"Cannot infer epoch from checkpoint {path}")
    return int(suffix)


def select_best_checkpoint(
    checkpoint_metrics: Sequence[Mapping[str, Any]], canonical_threshold: float
) -> Mapping[str, Any] | None:
    eligible = [
        row
        for row in checkpoint_metrics
        if float(row["canonical"]["top1_accuracy"]) >= canonical_threshold
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            -float(row["alias"]["top1_accuracy"]),
            -float(row["canonical"]["top1_accuracy"]),
            int(row["epoch"]),
        ),
    )


def prediction_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(records),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_run_id(mode: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{mode}"


def load_poc_a_reference(config: Mapping[str, Any]) -> dict[str, Any]:
    path = project_path(config_value(config, "data.poc_a_reference_path"))
    payload = read_json(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("experiment") != "poc_a"
    ):
        raise TrainingToolError("PoC A reference has an invalid schema")
    source = payload.get("source")
    if (
        not isinstance(source, dict)
        or source.get("repo_id") != "hxgdzyuyi/qwen3-8b-steam-entity-linking"
        or source.get("repo_type") != "model"
        or not isinstance(source.get("revision"), str)
        or len(source["revision"]) != 40
        or source.get("metrics_path") != "metrics.json"
        or not isinstance(source.get("metrics_sha256"), str)
        or len(source["metrics_sha256"]) != 64
    ):
        raise TrainingToolError("PoC A reference source snapshot is invalid")
    if payload.get("selected_epoch") != 20:
        raise TrainingToolError("PoC A reference must pin published epoch 20")
    evaluation_data = payload.get("evaluation_data")
    if not isinstance(evaluation_data, dict):
        raise TrainingToolError("PoC A reference has no evaluation-data snapshot")
    pinned_files = (
        ("steam_entities_path", "steam_entities_sha256"),
        ("alias_source_path", "alias_source_sha256"),
    )
    for path_key, hash_key in pinned_files:
        relative = evaluation_data.get(path_key)
        expected_hash = evaluation_data.get(hash_key)
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise TrainingToolError(
                f"PoC A reference evaluation field {path_key} is invalid"
            )
        current_path = REPOSITORY_ROOT / relative
        if sha256_file(current_path) != expected_hash:
            raise TrainingToolError(
                f"PoC A reference refuses A/B delta: {relative} hash differs"
            )
    configured_alias_source = project_path(
        config_value(config, "data.alias_source_path")
    ).resolve()
    pinned_alias_source = (
        REPOSITORY_ROOT / str(evaluation_data["alias_source_path"])
    ).resolve()
    if configured_alias_source != pinned_alias_source:
        raise TrainingToolError("PoC B alias source differs from the PoC A snapshot")
    if evaluation_data.get("canonical_case_count") != int(
        config_value(config, "data.expected_canonical_rows")
    ) or evaluation_data.get("alias_case_count") != int(
        config_value(config, "data.expected_alias_rows")
    ):
        raise TrainingToolError("PoC A and PoC B evaluation case counts differ")
    if payload.get("canonical", {}).get("count") != int(
        config_value(config, "data.expected_canonical_rows")
    ):
        raise TrainingToolError("PoC A canonical reference count differs")
    if payload.get("alias", {}).get("count") != int(
        config_value(config, "data.expected_alias_rows")
    ):
        raise TrainingToolError("PoC A alias reference count differs")
    for split in ("canonical", "alias"):
        summary = payload.get(split)
        if not isinstance(summary, dict):
            raise TrainingToolError(f"PoC A {split} reference is invalid")
        count = summary.get("count")
        correct = summary.get("top1_correct")
        accuracy = summary.get("top1_accuracy")
        if (
            not isinstance(count, int)
            or not isinstance(correct, int)
            or not isinstance(accuracy, (int, float))
            or not 0 <= correct <= count
            or abs(float(accuracy) - correct / count) > 1e-15
        ):
            raise TrainingToolError(f"PoC A {split} reference metrics are inconsistent")
    alias_by_type = payload.get("alias_by_type")
    if not isinstance(alias_by_type, dict) or sum(
        int(summary.get("count", -1))
        for summary in alias_by_type.values()
        if isinstance(summary, dict)
    ) != int(config_value(config, "data.expected_alias_rows")):
        raise TrainingToolError("PoC A alias-type reference is inconsistent")
    return payload


def validate_runtime_snapshot(
    snapshot: Mapping[str, Any],
    config: Mapping[str, Any],
    minimum_free_disk_gib: float | None = None,
) -> None:
    expected_name = str(config_value(config, "runtime.expected_gpu_name"))
    if expected_name.casefold() not in str(snapshot.get("name", "")).casefold():
        raise TrainingToolError(
            f"Expected GPU containing {expected_name!r}, found {snapshot.get('name')!r}"
        )
    checks = (
        (
            "gpu_memory_gib",
            float(config_value(config, "runtime.minimum_gpu_memory_gib")),
        ),
        (
            "system_memory_gib",
            float(config_value(config, "runtime.minimum_system_memory_gib")),
        ),
        (
            "cpu_count",
            float(config_value(config, "runtime.minimum_cpu_count")),
        ),
        (
            "free_disk_gib",
            float(
                minimum_free_disk_gib
                if minimum_free_disk_gib is not None
                else config_value(config, "runtime.minimum_free_disk_gib")
            ),
        ),
    )
    for key, minimum in checks:
        if float(snapshot.get(key, 0)) < minimum:
            raise TrainingToolError(
                f"Runtime {key}={snapshot.get(key)} is below required {minimum}"
            )
    if not str(snapshot.get("torch_version", "")).startswith(
        str(config_value(config, "runtime.expected_torch_major_minor"))
    ):
        raise TrainingToolError("PyTorch version does not match the target image")
    if not str(snapshot.get("cuda_version", "")).startswith(
        str(config_value(config, "runtime.expected_cuda_major_minor"))
    ):
        raise TrainingToolError("CUDA version does not match the target image")
