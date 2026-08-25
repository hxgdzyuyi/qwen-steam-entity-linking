#!/usr/bin/env python3
"""Train the PoC B frozen-backbone semantic prototype classifier."""

from __future__ import annotations

import argparse
import gc
import math
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation_core import checkpoint_metrics
from feature_cache import (
    cache_identity,
    load_feature_cache,
    save_feature_cache,
    stable_sha256,
    tokenizer_sha256,
)
from steam_entity_classifier import (
    FrozenPrototypeHead,
    classifier_config,
    extract_features,
    freeze_backbone,
    initialize_prototypes,
    load_classifier_artifact,
    save_classifier_artifact,
)
from training_common import (
    TrainingToolError,
    atomic_write_json,
    class_map_payload,
    config_value,
    data_hashes,
    dependency_versions,
    discover_checkpoints,
    git_info,
    load_config,
    make_run_id,
    project_path,
    read_json,
    subset_bundle,
    utc_now,
    validate_data,
    validate_runtime_snapshot,
    warn_if_git_commit_mismatch,
)


POC_ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
RESUME_IDENTITY_KEYS = (
    "model",
    "mode",
    "data_sha256",
    "feature_cache_sha256",
    "tokenizer_sha256",
    "class_map_sha256",
    "training_config_sha256",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=POC_ROOT / "configs/qwen3_8b_frozen_prototype.yaml",
    )
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Output directory; defaults to poc_b/outputs/<timestamp>-<mode>",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume from the run's latest head/optimizer/scheduler directory",
    )
    return parser.parse_args(argv)


def _cloud_imports() -> dict[str, Any]:
    try:
        import torch
        from huggingface_hub import HfApi
        from transformers import (
            AutoModel,
            AutoTokenizer,
            get_cosine_schedule_with_warmup,
            set_seed,
        )
        from transformers.utils import cached_file
    except ImportError as error:
        raise TrainingToolError(
            "Cloud dependencies are missing; install poc_b/requirements-cloud.txt "
            "on the configured CUDA image"
        ) from error
    return {
        "torch": torch,
        "HfApi": HfApi,
        "AutoModel": AutoModel,
        "AutoTokenizer": AutoTokenizer,
        "get_cosine_schedule_with_warmup": get_cosine_schedule_with_warmup,
        "set_seed": set_seed,
        "cached_file": cached_file,
    }


def _existing_ancestor(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _system_memory_gib() -> float:
    candidates: list[int] = []
    try:
        candidates.append(
            int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
        )
    except (AttributeError, OSError, ValueError):
        pass
    for limit_path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            raw = limit_path.read_text(encoding="utf-8").strip()
            if raw != "max":
                limit = int(raw)
                if limit < 1 << 60:
                    candidates.append(limit)
        except (OSError, ValueError):
            continue
    if not candidates:
        raise TrainingToolError("Cannot determine available system memory")
    return min(candidates) / GIB


def _require_target_runtime(
    torch: Any,
    config: Mapping[str, Any],
    run_dir: Path,
    minimum_free_disk_gib: float,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise TrainingToolError("PoC B cloud training requires CUDA")
    if torch.cuda.device_count() != 1:
        raise TrainingToolError("PoC B expects exactly one visible CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise TrainingToolError("The visible GPU does not support BF16")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    properties = torch.cuda.get_device_properties(0)
    snapshot = {
        "target_image": config_value(config, "runtime.target_image"),
        "device_count": 1,
        "name": torch.cuda.get_device_name(0),
        "gpu_memory_gib": properties.total_memory / GIB,
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": str(torch.__version__).split("+")[0],
        "cuda_version": str(torch.version.cuda),
        "bf16_supported": True,
        "tf32_enabled": True,
        "system_memory_gib": _system_memory_gib(),
        "cpu_count": os.cpu_count() or 0,
        "free_disk_gib": shutil.disk_usage(_existing_ancestor(run_dir)).free / GIB,
        "minimum_free_disk_gib_applied": minimum_free_disk_gib,
    }
    validate_runtime_snapshot(snapshot, config, minimum_free_disk_gib)
    return snapshot


def _cached_model_file(
    cached_file: Any, model_id: str, revision: str, filename: str
) -> Path | None:
    try:
        value = cached_file(
            model_id,
            filename,
            revision=revision,
            token=os.environ.get("HF_TOKEN"),
            local_files_only=True,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
    except (OSError, ValueError):
        return None
    return Path(value) if value and Path(value).is_file() else None


def _model_weights_are_cached(
    cached_file: Any, model_id: str, revision: str
) -> bool:
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = _cached_model_file(cached_file, model_id, revision, index_name)
        if index_path is None:
            continue
        index = read_json(index_path)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        if any(not isinstance(filename, str) for filename in weight_map.values()):
            return False
        shards = set(weight_map.values())
        return all(
            _cached_model_file(cached_file, model_id, revision, filename) is not None
            for filename in shards
        )
    return any(
        _cached_model_file(cached_file, model_id, revision, filename) is not None
        for filename in ("model.safetensors", "pytorch_model.bin")
    )


def _resolve_model_revision(config: Mapping[str, Any], HfApi: Any) -> str:
    model_id = str(config_value(config, "model.id"))
    requested = config["model"].get("revision") or "main"
    info = HfApi(token=os.environ.get("HF_TOKEN")).model_info(
        model_id, revision=requested
    )
    if not info.sha:
        raise TrainingToolError(f"No immutable revision returned for {model_id}")
    return str(info.sha)


def _run_directory(
    args: argparse.Namespace, config: Mapping[str, Any]
) -> Path:
    if args.run_dir:
        return args.run_dir.resolve()
    if args.resume_from:
        resume = args.resume_from.resolve()
        if resume.name != "resume":
            raise TrainingToolError("--resume-from must point to <run-dir>/resume")
        return resume.parent
    return project_path(config_value(config, "training.output_root")) / make_run_id(
        args.mode
    )


def _run_directory_has_artifacts(run_dir: Path) -> bool:
    if run_dir.is_symlink() or (run_dir.exists() and not run_dir.is_dir()):
        return True
    return run_dir.exists() and any(
        path.is_file() or path.is_symlink() for path in run_dir.rglob("*")
    )


def _labels(rows: Sequence[Mapping[str, Any]], torch: Any) -> Any:
    return torch.tensor(
        [int(row["class_index"]) for row in rows], dtype=torch.long
    )


def _extract_and_cache_features(
    *,
    imports: Mapping[str, Any],
    config: Mapping[str, Any],
    bundle: Any,
    tokenizer: Any,
    model_id: str,
    revision: str,
    identity: Mapping[str, Any],
    cache_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = imports["torch"]
    device = torch.device("cuda")
    backbone = imports["AutoModel"].from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=bool(config_value(config, "model.trust_remote_code")),
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    freeze_backbone(backbone)
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise TrainingToolError("Frozen Qwen backbone unexpectedly has trainable weights")
    batch_size = int(config_value(config, "features.extraction_batch_size"))
    max_length = int(config_value(config, "data.max_length"))
    try:
        tensors = {
            "train_features": extract_features(
                backbone,
                tokenizer,
                [row["model_input"] for row in bundle.train_rows],
                batch_size=batch_size,
                max_length=max_length,
                device=device,
            ),
            "train_labels": _labels(bundle.train_rows, torch),
            "canonical_features": extract_features(
                backbone,
                tokenizer,
                [row["model_input"] for row in bundle.canonical_rows],
                batch_size=batch_size,
                max_length=max_length,
                device=device,
            ),
            "canonical_labels": _labels(bundle.canonical_rows, torch),
            "alias_features": extract_features(
                backbone,
                tokenizer,
                [row["model_input"] for row in bundle.alias_rows],
                batch_size=batch_size,
                max_length=max_length,
                device=device,
            ),
            "alias_labels": _labels(bundle.alias_rows, torch),
        }
    finally:
        del backbone
        gc.collect()
        torch.cuda.empty_cache()
    metadata = save_feature_cache(cache_dir, tensors, identity)
    return tensors, metadata


def _classifier_settings(
    *,
    head: FrozenPrototypeHead,
    epoch: int,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return classifier_config(
        head,
        base_model_id=str(manifest["model"]["id"]),
        base_model_revision=str(manifest["model"]["revision"]),
        max_length=int(config_value(config, "data.max_length")),
        prototype_anchor_weight=float(
            config_value(config, "classifier.prototype_anchor_weight")
        ),
        feature_cache_sha256=str(manifest["feature_cache_sha256"]),
        tokenizer_sha256=str(manifest["tokenizer_sha256"]),
        class_map_sha256=str(manifest["class_map_sha256"]),
        training_config_sha256=str(manifest["training_config_sha256"]),
        mode=str(manifest["mode"]),
        epoch=epoch,
    )


def _save_head(
    destination: Path,
    *,
    head: FrozenPrototypeHead,
    epoch: int,
    global_step: int,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    classes: Sequence[Any],
) -> None:
    save_classifier_artifact(
        head,
        destination,
        config=_classifier_settings(
            head=head, epoch=epoch, config=config, manifest=manifest
        ),
    )
    atomic_write_json(destination / "class_map.json", class_map_payload(classes))
    forbidden = [
        path.name
        for path in destination.iterdir()
        if path.is_file()
        and (
            path.name.startswith("model-")
            or path.name in {"model.safetensors", "pytorch_model.bin"}
        )
    ]
    if forbidden:
        raise TrainingToolError(f"Backbone weights found in checkpoint: {forbidden}")


def _atomic_torch_save(torch: Any, payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _save_resume_state(
    *,
    torch: Any,
    resume_dir: Path,
    head: FrozenPrototypeHead,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    global_step: int,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    classes: Sequence[Any],
) -> None:
    _save_head(
        resume_dir,
        head=head,
        epoch=epoch,
        global_step=global_step,
        config=config,
        manifest=manifest,
        classes=classes,
    )
    _atomic_torch_save(
        torch,
        {
            "schema_version": 1,
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        resume_dir / "training_state.pt",
    )
    atomic_write_json(
        resume_dir / "resume_metadata.json",
        {
            "schema_version": 1,
            "epoch": epoch,
            "global_step": global_step,
            "model": manifest["model"],
            "mode": manifest["mode"],
            "data_sha256": manifest["data_sha256"],
            "feature_cache_sha256": manifest["feature_cache_sha256"],
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "class_map_sha256": manifest["class_map_sha256"],
            "training_config_sha256": manifest["training_config_sha256"],
        },
    )


def _move_optimizer_state(optimizer: Any, device: Any) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if hasattr(value, "to"):
                state[key] = value.to(device)


def validate_resume_identity(
    manifest: Mapping[str, Any], resume_metadata: Mapping[str, Any]
) -> None:
    for key in RESUME_IDENTITY_KEYS:
        if resume_metadata.get(key) != manifest.get(key):
            raise TrainingToolError(f"Resume metadata differs for {key}")


def validate_resume_progress(
    state: Mapping[str, Any],
    resume_metadata: Mapping[str, Any],
    head_settings: Mapping[str, Any],
    *,
    steps_per_epoch: int,
) -> tuple[int, int]:
    if state.get("schema_version") != 1:
        raise TrainingToolError("Resume training state has an invalid schema")
    start_epoch = int(state["epoch"])
    global_step = int(state["global_step"])
    if (
        start_epoch != int(resume_metadata["epoch"])
        or start_epoch != int(head_settings["epoch"])
        or global_step != int(resume_metadata["global_step"])
        or global_step != start_epoch * steps_per_epoch
    ):
        raise TrainingToolError("Resume progress differs across state files")
    return start_epoch, global_step


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    full_bundle = validate_data(config)
    bundle = (
        subset_bundle(
            full_bundle, int(config_value(config, "training.smoke_classes"))
        )
        if args.mode == "smoke"
        else full_bundle
    )
    run_dir = _run_directory(args, config)
    resume_dir = args.resume_from.resolve() if args.resume_from else None
    if resume_dir and (
        resume_dir != run_dir / "resume" or not resume_dir.is_dir()
    ):
        raise TrainingToolError("--resume-from must be the selected run's resume directory")
    if resume_dir is None and _run_directory_has_artifacts(run_dir):
        raise TrainingToolError(f"Refusing to overwrite non-empty run: {run_dir}")

    imports = _cloud_imports()
    torch = imports["torch"]
    imports["set_seed"](int(config_value(config, "training.seed")))
    random.seed(int(config_value(config, "training.seed")))
    device = torch.device("cuda")
    model_id = str(config_value(config, "model.id"))
    hashes = data_hashes(config)
    class_payload = class_map_payload(bundle.classes)
    actual_class_map_sha256 = stable_sha256(class_payload)
    training_config_sha256 = stable_sha256(config)

    manifest_path = run_dir / "run_manifest.json"
    resolved_config_path = run_dir / "resolved_config.json"
    current_git = git_info(bool(config_value(config, "training.require_clean_git")))
    if resume_dir:
        manifest = read_json(manifest_path)
        if read_json(resolved_config_path) != config:
            raise TrainingToolError("Resume config differs from the original run")
        revision = str(manifest.get("model", {}).get("revision", ""))
        if not revision:
            raise TrainingToolError("Resume manifest has no pinned model revision")
    else:
        revision = _resolve_model_revision(config, imports["HfApi"])

    cached = _model_weights_are_cached(
        imports["cached_file"], model_id, revision
    )
    disk_key = (
        "runtime.minimum_cached_free_disk_gib"
        if cached or resume_dir
        else "runtime.minimum_free_disk_gib"
    )
    minimum_disk = float(config_value(config, disk_key))
    runtime = _require_target_runtime(torch, config, run_dir, minimum_disk)
    runtime["model_weights_cached_before_start"] = cached
    runtime["disk_policy"] = disk_key.removeprefix("runtime.minimum_")

    tokenizer_dir = run_dir / "tokenizer"
    if resume_dir:
        tokenizer = imports["AutoTokenizer"].from_pretrained(
            tokenizer_dir, use_fast=True, token=os.environ.get("HF_TOKEN")
        )
    else:
        tokenizer = imports["AutoTokenizer"].from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=bool(config_value(config, "model.trust_remote_code")),
            token=os.environ.get("HF_TOKEN"),
            use_fast=True,
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer_hash = tokenizer_sha256(tokenizer)
    row_counts = {
        "train": len(bundle.train_rows),
        "canonical": len(bundle.canonical_rows),
        "alias": len(bundle.alias_rows),
    }
    cache_hash_inputs = dict(hashes)
    cache_hash_inputs["run_class_map"] = actual_class_map_sha256
    identity = cache_identity(
        mode=args.mode,
        model_id=model_id,
        model_revision=revision,
        tokenizer_hash=tokenizer_hash,
        data_sha256=cache_hash_inputs,
        max_length=int(config_value(config, "data.max_length")),
        pooling=str(config_value(config, "features.pooling")),
        class_count=bundle.class_count,
        row_counts=row_counts,
    )

    if resume_dir:
        expected_identity = {
            "mode": args.mode,
            "model": {"id": model_id, "revision": revision},
            "data_sha256": hashes,
            "tokenizer_sha256": tokenizer_hash,
            "class_map_sha256": actual_class_map_sha256,
            "training_config_sha256": training_config_sha256,
        }
        actual_identity = {
            "mode": manifest.get("mode"),
            "model": manifest.get("model"),
            "data_sha256": manifest.get("data_sha256"),
            "tokenizer_sha256": manifest.get("tokenizer_sha256"),
            "class_map_sha256": manifest.get("class_map_sha256"),
            "training_config_sha256": manifest.get("training_config_sha256"),
        }
        if actual_identity != expected_identity:
            raise TrainingToolError("Resume model, data, class map, mode, or config differs")
        warn_if_git_commit_mismatch(
            current_git["commit"], manifest.get("git", {}).get("commit"), "Resume"
        )
        tensors, cache_metadata = load_feature_cache(
            run_dir / "feature_cache",
            expected_identity=identity,
            expected_cache_sha256=str(manifest["feature_cache_sha256"]),
        )
        resume_metadata = read_json(resume_dir / "resume_metadata.json")
        validate_resume_identity(manifest, resume_metadata)
        manifest["resumed_at"] = utc_now()
        manifest.setdefault("resume_runtime_checks", []).append(runtime)
    else:
        run_dir.mkdir(parents=True)
        shutil.copy2(config_path, run_dir / "training_config.yaml")
        atomic_write_json(resolved_config_path, config)
        atomic_write_json(run_dir / "class_map.json", class_payload)
        tokenizer.save_pretrained(tokenizer_dir)
        manifest = {
            "schema_version": 1,
            "status": "extracting_features",
            "created_at": utc_now(),
            "mode": args.mode,
            "publishable": args.mode == "full",
            "model": {"id": model_id, "revision": revision},
            "git": current_git,
            "data_sha256": hashes,
            "class_map_sha256": actual_class_map_sha256,
            "training_config_sha256": training_config_sha256,
            "tokenizer_sha256": tokenizer_hash,
            "runtime": runtime,
            "dependencies": dependency_versions(
                (
                    "torch",
                    "transformers",
                    "accelerate",
                    "huggingface-hub",
                    "safetensors",
                    "PyYAML",
                )
            ),
            "class_count": bundle.class_count,
            "row_counts": row_counts,
            "seed": config_value(config, "training.seed"),
            "data_seed": config_value(config, "training.data_seed"),
        }
        atomic_write_json(manifest_path, manifest)
        tensors, cache_metadata = _extract_and_cache_features(
            imports=imports,
            config=config,
            bundle=bundle,
            tokenizer=tokenizer,
            model_id=model_id,
            revision=revision,
            identity=identity,
            cache_dir=run_dir / "feature_cache",
        )
        manifest["feature_cache_sha256"] = cache_metadata["cache_sha256"]
        manifest["hidden_size"] = cache_metadata["hidden_size"]
        atomic_write_json(manifest_path, manifest)

    initial_prototypes = initialize_prototypes(
        tensors["train_features"], tensors["train_labels"], bundle.class_count
    )
    if resume_dir:
        head, head_settings = load_classifier_artifact(resume_dir, device=device)
        expected_head_identity = {
            "base_model_id": model_id,
            "base_model_revision": revision,
            "feature_cache_sha256": manifest["feature_cache_sha256"],
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "class_map_sha256": actual_class_map_sha256,
            "training_config_sha256": training_config_sha256,
            "mode": args.mode,
            "num_classes": bundle.class_count,
            "hidden_size": int(tensors["train_features"].shape[1]),
        }
        if any(
            head_settings.get(key) != value
            for key, value in expected_head_identity.items()
        ):
            raise TrainingToolError("Resume classifier config differs from the run")
    else:
        head = FrozenPrototypeHead(
            initial_prototypes,
            bottleneck_dim=int(config_value(config, "classifier.bottleneck_dim")),
            temperature=float(config_value(config, "classifier.temperature")),
        ).to(device)
        baseline, _ = checkpoint_metrics(
            epoch=0,
            head=head,
            canonical_features=tensors["canonical_features"],
            alias_features=tensors["alias_features"],
            canonical_rows=bundle.canonical_rows,
            alias_rows=bundle.alias_rows,
            classes=class_payload["classes"],
            batch_size=int(config_value(config, "evaluation.batch_size")),
            diagnostic_top_k=int(
                config_value(config, "evaluation.diagnostic_top_k")
            ),
            device=device,
        )
        baseline["kind"] = "zero_training_canonical_view_prototypes"
        baseline["alias_used_for_training"] = False
        atomic_write_json(run_dir / "zero_training_baseline.json", baseline)
        manifest["zero_training_baseline"] = baseline
        print(
            "Zero-training prototype baseline: "
            f"canonical={baseline['canonical']['top1_accuracy']:.2%}, "
            f"alias={baseline['alias']['top1_accuracy']:.2%}",
            flush=True,
        )

    epochs = int(
        config_value(
            config,
            "training.smoke_epochs" if args.mode == "smoke" else "training.full_epochs",
        )
    )
    batch_size = int(config_value(config, "training.batch_size"))
    steps_per_epoch = math.ceil(len(bundle.train_rows) / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(
        round(total_steps * float(config_value(config, "training.warmup_ratio")))
    )
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config_value(config, "training.learning_rate")),
        weight_decay=float(config_value(config, "training.weight_decay")),
    )
    scheduler = imports["get_cosine_schedule_with_warmup"](
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    start_epoch = 0
    global_step = 0
    history: list[dict[str, Any]] = []
    if resume_dir:
        try:
            state = torch.load(
                resume_dir / "training_state.pt",
                map_location="cpu",
                weights_only=True,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise TrainingToolError(f"Cannot load resume training state: {error}") from error
        resume_metadata = read_json(resume_dir / "resume_metadata.json")
        start_epoch, global_step = validate_resume_progress(
            state,
            resume_metadata,
            head_settings,
            steps_per_epoch=steps_per_epoch,
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _move_optimizer_state(optimizer, device)
        history_path = run_dir / "training_history.json"
        if history_path.exists():
            history = list(read_json(history_path))

    manifest["status"] = "training"
    manifest["last_started_at"] = utc_now()
    manifest["epochs"] = epochs
    manifest["steps_per_epoch"] = steps_per_epoch
    manifest["total_steps"] = total_steps
    manifest["warmup_steps"] = warmup_steps
    manifest["alias_used_for_training"] = False
    atomic_write_json(manifest_path, manifest)

    train_features = tensors["train_features"]
    train_labels = tensors["train_labels"]
    anchor_weight = float(
        config_value(config, "classifier.prototype_anchor_weight")
    )
    milestone_epochs = set(config_value(config, "training.checkpoint_epochs"))
    data_seed = int(config_value(config, "training.data_seed"))
    for epoch in range(start_epoch + 1, epochs + 1):
        head.train()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(data_seed + epoch)
        order = torch.randperm(len(train_features), generator=generator)
        aggregate = {"loss": 0.0, "cross_entropy": 0.0, "prototype_anchor": 0.0}
        batches = 0
        for offset in range(0, len(order), batch_size):
            indices = order[offset : offset + batch_size]
            batch_features = train_features[indices].to(device)
            batch_labels = train_labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, components = head.training_loss(
                batch_features, batch_labels, anchor_weight=anchor_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                head.parameters(),
                float(config_value(config, "training.max_grad_norm")),
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            batches += 1
            for key in aggregate:
                aggregate[key] += components[key]
        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            **{key: value / batches for key, value in aggregate.items()},
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        history.append(epoch_record)
        atomic_write_json(run_dir / "training_history.json", history)
        if epoch in milestone_epochs:
            _save_head(
                run_dir / "checkpoints" / f"epoch-{epoch}",
                head=head,
                epoch=epoch,
                global_step=global_step,
                config=config,
                manifest=manifest,
                classes=bundle.classes,
            )
        _save_resume_state(
            torch=torch,
            resume_dir=run_dir / "resume",
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            config=config,
            manifest=manifest,
            classes=bundle.classes,
        )
        print(
            f"epoch={epoch}/{epochs} loss={epoch_record['loss']:.6f} "
            f"lr={epoch_record['learning_rate']:.3e}",
            flush=True,
        )

    expected_milestones = sorted(
        epoch for epoch in milestone_epochs if epoch <= epochs
    )
    actual_milestones = [epoch for epoch, _ in discover_checkpoints(run_dir)]
    if actual_milestones != expected_milestones:
        raise TrainingToolError(
            f"Expected milestone heads {expected_milestones}, found {actual_milestones}"
        )
    manifest["status"] = "trained"
    manifest["completed_at"] = utc_now()
    manifest["global_step"] = global_step
    manifest["checkpoints"] = [
        {"epoch": epoch, "path": str(path.relative_to(run_dir))}
        for epoch, path in discover_checkpoints(run_dir)
    ]
    atomic_write_json(manifest_path, manifest)
    print(f"Training completed: {run_dir}")
    print(
        f"Run: python poc_b/scripts/evaluate.py --run-dir {run_dir} --all-milestones"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TrainingToolError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
