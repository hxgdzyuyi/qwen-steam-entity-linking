#!/usr/bin/env python3
"""Train Qwen3-8B-Base for Steam entity linking on one cloud GPU."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from inference_core import predict_structured_rows

from training_common import (
    CompletionOnlyCollator,
    RawRowsDataset,
    TrainingToolError,
    atomic_write_json,
    config_value,
    data_hashes,
    dependency_versions,
    discover_checkpoints,
    encode_training_row,
    ensure_prompt_lengths,
    git_info,
    load_config,
    make_run_id,
    prepare_tokenizer,
    project_path,
    read_json,
    utc_now,
    validate_data,
    validate_no_match_token,
    warn_if_git_commit_mismatch,
)
from structured_output import entity_scoring_prefix


GIB = 1024**3
RESUME_STATE_NAMES = {"optimizer.pt", "scheduler.pt", "scaler.pt"}
POC_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=POC_ROOT / "configs/qwen3_8b_lora.yaml",
    )
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Explicit output directory; defaults to poc_a/outputs/<timestamp>-<mode>",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume model, optimizer, scheduler, and trainer state from a checkpoint",
    )
    return parser.parse_args(argv)


def _cloud_imports() -> dict[str, Any]:
    try:
        import torch
        from huggingface_hub import HfApi
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
        from transformers.utils import cached_file
    except ImportError as error:
        raise TrainingToolError(
            "Cloud training dependencies are missing; install "
            "poc_a/requirements-cloud.txt "
            "on top of a CUDA-enabled PyTorch image"
        ) from error
    return {
        "torch": torch,
        "HfApi": HfApi,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "Trainer": Trainer,
        "TrainerCallback": TrainerCallback,
        "TrainingArguments": TrainingArguments,
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
            value = limit_path.read_text(encoding="utf-8").strip()
            if value != "max":
                limit = int(value)
                if limit < 1 << 60:
                    candidates.append(limit)
        except (OSError, ValueError):
            continue
    if not candidates:
        raise TrainingToolError("Cannot determine available system memory")
    return min(candidates) / GIB


def validate_runtime_snapshot(
    snapshot: Mapping[str, Any],
    config: Mapping[str, Any],
    minimum_free_disk_gib: float | None = None,
) -> None:
    expected_name = str(config_value(config, "runtime.expected_gpu_name"))
    if expected_name.casefold() not in str(snapshot["name"]).casefold():
        raise TrainingToolError(
            f"Expected a {expected_name} GPU, found {snapshot['name']}"
        )
    minimums = (
        ("gpu_memory_gib", "runtime.minimum_gpu_memory_gib"),
        ("system_memory_gib", "runtime.minimum_system_memory_gib"),
        ("cpu_count", "runtime.minimum_cpu_count"),
    )
    for snapshot_key, config_key in minimums:
        required = float(config_value(config, config_key))
        actual = float(snapshot[snapshot_key])
        if actual < required:
            raise TrainingToolError(
                f"Runtime {snapshot_key} requires at least {required:g}, found {actual:.1f}"
            )
    required_disk = (
        float(config_value(config, "runtime.minimum_free_disk_gib"))
        if minimum_free_disk_gib is None
        else float(minimum_free_disk_gib)
    )
    actual_disk = float(snapshot["free_disk_gib"])
    if actual_disk < required_disk:
        raise TrainingToolError(
            f"Runtime free_disk_gib requires at least {required_disk:g}, "
            f"found {actual_disk:.1f}"
        )
    versions = (
        ("torch_version", "runtime.expected_torch_major_minor"),
        ("cuda_version", "runtime.expected_cuda_major_minor"),
    )
    for snapshot_key, config_key in versions:
        expected = str(config_value(config, config_key))
        actual = str(snapshot[snapshot_key])
        if not (actual == expected or actual.startswith(f"{expected}.")):
            raise TrainingToolError(
                f"Expected {snapshot_key} {expected}.x, found {actual}"
            )


def _require_target_runtime(
    torch: Any,
    config: Mapping[str, Any],
    run_dir: Path,
    minimum_free_disk_gib: float,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise TrainingToolError("Cloud training requires a CUDA GPU")
    count = torch.cuda.device_count()
    if count != 1:
        raise TrainingToolError(f"Expected exactly one visible CUDA GPU, found {count}")
    if not torch.cuda.is_bf16_supported():
        raise TrainingToolError("The visible GPU does not support BF16")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    properties = torch.cuda.get_device_properties(0)
    snapshot = {
        "target_image": config_value(config, "runtime.target_image"),
        "device_count": count,
        "name": torch.cuda.get_device_name(0),
        "gpu_memory_gib": properties.total_memory / GIB,
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": str(torch.__version__).split("+")[0],
        "cuda_version": torch.version.cuda,
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
    if not value:
        return None
    path = Path(value)
    return path if path.is_file() else None


def _model_weights_are_cached(
    cached_file: Any, model_id: str, revision: str
) -> bool:
    """Return true only when every weight shard for the pinned revision is local."""

    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = _cached_model_file(cached_file, model_id, revision, index_name)
        if index_path is None:
            continue
        try:
            index = read_json(index_path)
        except TrainingToolError:
            return False
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        if any(not isinstance(filename, str) for filename in weight_map.values()):
            return False
        shards = {
            filename for filename in weight_map.values() if isinstance(filename, str)
        }
        return bool(shards) and all(
            _cached_model_file(cached_file, model_id, revision, shard) is not None
            for shard in shards
        )

    return any(
        _cached_model_file(cached_file, model_id, revision, filename) is not None
        for filename in ("model.safetensors", "pytorch_model.bin")
    )


def _run_directory_has_artifacts(run_dir: Path) -> bool:
    if run_dir.is_symlink():
        return True
    if not run_dir.exists():
        return False
    if not run_dir.is_dir():
        return True
    return any(path.is_file() or path.is_symlink() for path in run_dir.rglob("*"))


def _checkpoint_state_files(checkpoint: Path) -> list[Path]:
    return [
        path
        for path in checkpoint.iterdir()
        if path.is_file()
        and (path.name in RESUME_STATE_NAMES or path.name.startswith("rng_state"))
    ]


def _strip_older_resume_states(checkpoints_dir: Path, current: Path) -> None:
    for checkpoint in checkpoints_dir.glob("checkpoint-*"):
        if not checkpoint.is_dir() or checkpoint.resolve() == current.resolve():
            continue
        removed = _checkpoint_state_files(checkpoint)
        for path in removed:
            path.unlink()
        metadata_path = checkpoint / "checkpoint_meta.json"
        if metadata_path.exists() and removed:
            metadata = read_json(metadata_path)
            metadata["resumable"] = False
            metadata["resume_state_removed_at"] = utc_now()
            atomic_write_json(metadata_path, metadata)
            print(
                f"Disk policy: removed optimizer/scheduler/RNG state from "
                f"{checkpoint.name}; its LoRA remains evaluable but is no longer resumable.",
                flush=True,
            )


def _require_resumable_checkpoint(checkpoint: Path) -> None:
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
    }
    missing = sorted(name for name in required if not (checkpoint / name).is_file())
    if missing:
        raise TrainingToolError(
            f"Checkpoint is adapter-only and cannot resume training; missing {missing}"
        )


def _resolve_model_revision(config: Mapping[str, Any], HfApi: Any) -> str:
    model_id = str(config_value(config, "model.id"))
    requested = config["model"].get("revision") or "main"
    token = os.environ.get("HF_TOKEN")
    info = HfApi(token=token).model_info(model_id, revision=requested)
    if not info.sha:
        raise TrainingToolError(
            f"Hugging Face did not return a commit SHA for {model_id}"
        )
    return str(info.sha)


def _next_token_accuracy(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, str]],
    entity_token_ids: Sequence[int],
    max_length: int,
    batch_size: int,
    torch: Any,
) -> float:
    was_training = model.training
    model.eval()
    correct = 0
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        with torch.inference_mode():
            for offset in range(0, len(rows), batch_size):
                batch_rows = rows[offset : offset + batch_size]
                encoded = tokenizer(
                    [
                        row["prompt"] + entity_scoring_prefix(row["canonical_name"])
                        for row in batch_rows
                    ],
                    add_special_tokens=False,
                    padding=True,
                    return_tensors="pt",
                )
                if int(encoded["attention_mask"].sum(dim=1).max()) > max_length:
                    raise TrainingToolError(
                        "Smoke evaluation prompt exceeds max_length"
                    )
                encoded = {
                    key: value.to(model.device) for key, value in encoded.items()
                }
                logits = model(**encoded, use_cache=False).logits[:, -1, :]
                entity_ids = torch.tensor(
                    entity_token_ids, dtype=torch.long, device=logits.device
                )
                predicted_classes = logits.index_select(-1, entity_ids).argmax(dim=-1)
                predicted = entity_ids[predicted_classes].tolist()
                expected = [
                    int(tokenizer.convert_tokens_to_ids(row["completion"]))
                    for row in batch_rows
                ]
                correct += sum(
                    left == right for left, right in zip(predicted, expected)
                )
    finally:
        tokenizer.padding_side = previous_padding_side
        if was_training:
            model.train()
    return correct / len(rows)


def structured_completion_loss(
    logits: Any,
    labels: Any,
    output_token_ids: Sequence[int],
    torch: Any,
    canonical_text_weight: float = 1.0,
    entity_weight: float = 1.0,
) -> Any:
    """Combine natural-language canonicalization loss and output-class loss."""

    if logits.shape[:2] != labels.shape:
        raise TrainingToolError("logits and labels have incompatible shapes")
    if labels.shape[1] < 2:
        raise TrainingToolError("structured completion needs at least two positions")
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    output_ids = torch.tensor(
        output_token_ids, dtype=torch.long, device=shifted_labels.device
    )
    output_matches = shifted_labels.unsqueeze(-1).eq(output_ids.view(1, 1, -1))
    output_positions = output_matches.any(dim=-1)
    if not bool(torch.all(output_positions.sum(dim=1) == 1).item()):
        raise TrainingToolError(
            "Each structured completion must supervise exactly one output token"
        )
    batch_indices = torch.arange(labels.shape[0], device=labels.device)
    output_position_indices = output_positions.long().argmax(dim=1)
    next_token_logits = shifted_logits[batch_indices, output_position_indices, :]
    output_logits = next_token_logits.index_select(-1, output_ids)
    target_token_ids = shifted_labels[batch_indices, output_position_indices]
    target_matches = target_token_ids.unsqueeze(1).eq(output_ids.unsqueeze(0))
    target_classes = target_matches.long().argmax(dim=1)
    entity_loss = torch.nn.functional.cross_entropy(
        output_logits.float(), target_classes
    )

    text_positions = shifted_labels.ne(-100) & ~output_positions
    if not bool(text_positions.any().item()):
        if canonical_text_weight != 0:
            raise TrainingToolError("Structured completion contains no text targets")
        text_loss = entity_loss.new_zeros(())
    else:
        text_loss = torch.nn.functional.cross_entropy(
            shifted_logits[text_positions].float(),
            shifted_labels[text_positions],
        )
    return canonical_text_weight * text_loss + entity_weight * entity_loss


def entity_classification_loss(
    logits: Any,
    labels: Any,
    entity_token_ids: Sequence[int],
    torch: Any,
) -> Any:
    """Backward-compatible entity-only loss used by dependency-light tests."""

    return structured_completion_loss(
        logits,
        labels,
        entity_token_ids,
        torch,
        canonical_text_weight=0.0,
        entity_weight=1.0,
    )


def _select_smoke_rows(
    rows: Sequence[Mapping[str, str]], sample_count: int, no_match_token: str
) -> list[dict[str, str]]:
    """Select a deterministic 75/25 known/rejection smoke cohort."""

    if sample_count < 2:
        raise TrainingToolError("Smoke training needs at least two samples")
    known = [dict(row) for row in rows if row["completion"] != no_match_token]
    unknown = [dict(row) for row in rows if row["completion"] == no_match_token]
    if not known or not unknown:
        raise TrainingToolError("Smoke training requires known and NO_MATCH rows")

    unknown_count = min(len(unknown), max(1, sample_count // 4))
    known_count = min(len(known), sample_count - unknown_count)
    unknown_count = min(len(unknown), sample_count - known_count)
    if known_count + unknown_count != sample_count:
        raise TrainingToolError("Not enough rows for the configured smoke sample")
    return [*known[:known_count], *unknown[:unknown_count]]


def _run_directory(args: argparse.Namespace, config: Mapping[str, Any]) -> Path:
    if args.run_dir:
        return args.run_dir.resolve()
    if args.resume_from:
        checkpoint = args.resume_from.resolve()
        if checkpoint.parent.name != "checkpoints":
            raise TrainingToolError(
                "A resumable checkpoint must be inside <run-dir>/checkpoints"
            )
        return checkpoint.parent.parent
    output_root = project_path(config_value(config, "training.output_root"))
    return output_root / make_run_id(args.mode)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)
    bundle = validate_data(config)
    run_dir = _run_directory(args, config)
    resume_from = args.resume_from.resolve() if args.resume_from else None
    if resume_from and not resume_from.is_dir():
        raise TrainingToolError(f"Resume checkpoint does not exist: {resume_from}")
    if resume_from:
        try:
            resume_from.relative_to(run_dir)
        except ValueError as error:
            raise TrainingToolError(
                "Resume checkpoint must be inside the selected run directory"
            ) from error
        _require_resumable_checkpoint(resume_from)
    if _run_directory_has_artifacts(run_dir) and resume_from is None:
        raise TrainingToolError(
            f"Refusing to overwrite non-empty run directory: {run_dir}"
        )
    checkpoints_dir = run_dir / "checkpoints"

    imports = _cloud_imports()
    torch = imports["torch"]
    set_seed = imports["set_seed"]
    set_seed(int(config_value(config, "training.seed")))

    model_id = str(config_value(config, "model.id"))
    revision = _resolve_model_revision(config, imports["HfApi"])
    model_weights_cached = _model_weights_are_cached(
        imports["cached_file"], model_id, revision
    )
    disk_policy = "cached-model" if model_weights_cached else "initial-download"
    disk_config_key = (
        "runtime.minimum_cached_free_disk_gib"
        if model_weights_cached
        else "runtime.minimum_free_disk_gib"
    )
    minimum_free_disk_gib = float(config_value(config, disk_config_key))
    runtime = _require_target_runtime(
        torch, config, run_dir, minimum_free_disk_gib
    )
    runtime["model_weights_cached_before_start"] = model_weights_cached
    runtime["disk_policy"] = disk_policy

    manifest_path = run_dir / "run_manifest.json"
    current_git = git_info(bool(config_value(config, "training.require_clean_git")))
    current_hashes = data_hashes(config)
    resolved_config_path = run_dir / "resolved_config.json"
    if resume_from:
        if not manifest_path.exists() or not resolved_config_path.exists():
            raise TrainingToolError(
                "Resume run is missing its manifest or resolved config"
            )
        manifest = read_json(manifest_path)
        original_config = read_json(resolved_config_path)
        if original_config != config:
            raise TrainingToolError(
                "Resume config differs from the original run config"
            )
        warn_if_git_commit_mismatch(
            current_git["commit"],
            manifest.get("git", {}).get("commit"),
            operation="Training resume",
        )
        expected_identity = {
            "mode": args.mode,
            "model": {"id": config_value(config, "model.id"), "revision": revision},
            "data_sha256": current_hashes,
        }
        actual_identity = {
            "mode": manifest.get("mode"),
            "model": manifest.get("model"),
            "data_sha256": manifest.get("data_sha256"),
        }
        if actual_identity != expected_identity:
            raise TrainingToolError(
                "Resume checkpoint does not match mode, model, or data"
            )
        manifest["resumed_at"] = utc_now()
        manifest["resume_from"] = str(resume_from.relative_to(run_dir))
        manifest.setdefault("resume_runtime_checks", []).append(runtime)
    else:
        manifest = {
            "schema_version": 1,
            "status": "initializing",
            "created_at": utc_now(),
            "mode": args.mode,
            "model": {"id": config_value(config, "model.id"), "revision": revision},
            "git": current_git,
            "data_sha256": current_hashes,
            "dependencies": dependency_versions(
                (
                    "torch",
                    "transformers",
                    "peft",
                    "accelerate",
                    "huggingface-hub",
                    "safetensors",
                    "PyYAML",
                )
            ),
            "runtime": runtime,
            "seed": config_value(config, "training.seed"),
            "data_seed": config_value(config, "training.data_seed"),
        }
    trust_remote_code = bool(config_value(config, "model.trust_remote_code"))
    tokenizer = imports["AutoTokenizer"].from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        token=os.environ.get("HF_TOKEN"),
        use_fast=True,
    )
    entity_token_ids = prepare_tokenizer(
        tokenizer, bundle.special_tokens, bundle.no_match_token
    )
    no_match_token_id = validate_no_match_token(tokenizer, bundle.no_match_token)
    output_token_ids = [*entity_token_ids, no_match_token_id]
    max_length = int(config_value(config, "data.max_length"))
    for row in bundle.train_rows:
        encode_training_row(tokenizer, row, max_length)
    ensure_prompt_lengths(tokenizer, bundle.alias_rows, max_length)
    ensure_prompt_lengths(tokenizer, bundle.unknown_rows, max_length)

    model = imports["AutoModelForCausalLM"].from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if bool(getattr(model.config, "tie_word_embeddings", True)):
        raise TrainingToolError(
            "This pipeline expects untied embed_tokens and lm_head weights"
        )
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False

    peft_config = imports["LoraConfig"](
        task_type=imports["TaskType"].CAUSAL_LM,
        r=int(config_value(config, "lora.r")),
        lora_alpha=int(config_value(config, "lora.alpha")),
        lora_dropout=float(config_value(config, "lora.dropout")),
        target_modules=config_value(config, "lora.target_modules"),
        bias=str(config_value(config, "lora.bias")),
        trainable_token_indices={
            "lm_head": output_token_ids,
        },
    )
    peft_config.base_model_name_or_path = model_id
    peft_config.revision = revision
    model = imports["get_peft_model"](model, peft_config)
    model.print_trainable_parameters()

    if args.mode == "smoke":
        sample_count = int(config_value(config, "training.smoke_samples"))
        train_rows = _select_smoke_rows(
            bundle.train_rows, sample_count, bundle.no_match_token
        )
        epochs = int(config_value(config, "training.smoke_epochs"))
        checkpoint_epochs: set[int] = set()
    else:
        train_rows = bundle.train_rows
        epochs = int(config_value(config, "training.full_epochs"))
        checkpoint_epochs = set(config_value(config, "training.checkpoint_epochs"))

    TrainingArguments = imports["TrainingArguments"]
    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        run_name=run_dir.name,
        num_train_epochs=epochs,
        per_device_train_batch_size=int(
            config_value(config, "training.per_device_batch_size")
        ),
        gradient_accumulation_steps=int(
            config_value(config, "training.gradient_accumulation_steps")
        ),
        learning_rate=float(config_value(config, "training.learning_rate")),
        lr_scheduler_type=str(config_value(config, "training.lr_scheduler_type")),
        warmup_ratio=float(config_value(config, "training.warmup_ratio")),
        weight_decay=float(config_value(config, "training.weight_decay")),
        max_grad_norm=float(config_value(config, "training.max_grad_norm")),
        optim=str(config_value(config, "training.optim")),
        bf16=True,
        fp16=False,
        tf32=True,
        logging_strategy="steps",
        logging_steps=int(config_value(config, "training.logging_steps")),
        logging_dir=str(run_dir / "tensorboard"),
        report_to=["tensorboard"],
        save_strategy="no",
        save_total_limit=max(1, len(checkpoint_epochs)),
        seed=int(config_value(config, "training.seed")),
        data_seed=int(config_value(config, "training.data_seed")),
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
        save_safetensors=True,
    )

    Trainer = imports["Trainer"]
    TrainerCallback = imports["TrainerCallback"]

    class EntityLinkingTrainer(Trainer):
        def compute_loss(
            self,
            model: Any,
            inputs: Mapping[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch
            labels = inputs["labels"]
            model_inputs = {
                key: value for key, value in inputs.items() if key != "labels"
            }
            outputs = model(**model_inputs, use_cache=False)
            loss = structured_completion_loss(
                outputs.logits,
                labels,
                output_token_ids,
                torch,
                canonical_text_weight=float(
                    config_value(config, "training.canonical_text_loss_weight")
                ),
                entity_weight=float(
                    config_value(config, "training.entity_loss_weight")
                ),
            )
            return (loss, outputs) if return_outputs else loss

        def _save(self, output_dir: str | None = None, state_dict: Any = None) -> None:
            destination = Path(output_dir or self.args.output_dir)
            destination.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(
                destination,
                state_dict=state_dict,
                safe_serialization=True,
                save_embedding_layers=False,
            )
            processing_class = getattr(self, "processing_class", None)
            if processing_class is not None:
                processing_class.save_pretrained(destination)
            torch.save(self.args, destination / "training_args.bin")

    class MilestoneCallback(TrainerCallback):
        def on_epoch_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            epoch = int(round(float(state.epoch or 0)))
            if (
                abs(float(state.epoch or 0) - epoch) < 1e-6
                and epoch in checkpoint_epochs
            ):
                control.should_save = True
            return control

        def on_save(
            self, training_args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            destination = (
                Path(training_args.output_dir) / f"checkpoint-{state.global_step}"
            )
            atomic_write_json(
                destination / "checkpoint_meta.json",
                {
                    "epoch": int(round(float(state.epoch))),
                    "global_step": int(state.global_step),
                    "saved_at": utc_now(),
                    "resumable": True,
                },
            )
            if args.mode == "full" and bool(
                config_value(config, "training.keep_resume_state_only_latest")
            ):
                _strip_older_resume_states(Path(training_args.output_dir), destination)
            return control

    smoke_history: list[dict[str, Any]] = []

    class SmokeCallback(MilestoneCallback):
        def __init__(self) -> None:
            self.perfect_streak = 0

        def on_epoch_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            smoke_records = predict_structured_rows(
                kwargs["model"],
                tokenizer,
                train_rows,
                expected_key="completion",
                source="smoke",
                batch_size=int(config_value(config, "evaluation.batch_size")),
                max_length=max_length,
                max_new_tokens=int(
                    config_value(config, "evaluation.generation_max_new_tokens")
                ),
                output_token_ids=output_token_ids,
                canonical_by_token=bundle.canonical_by_token,
                provenance_by_token=bundle.provenance_by_token,
                torch=torch,
            )
            accuracy = sum(row["structured_exact"] for row in smoke_records) / len(
                smoke_records
            )
            epoch = int(round(float(state.epoch or 0)))
            self.perfect_streak = self.perfect_streak + 1 if accuracy == 1.0 else 0
            smoke_history.append(
                {
                    "epoch": epoch,
                    "structured_exact_accuracy": accuracy,
                    "perfect_streak": self.perfect_streak,
                }
            )
            atomic_write_json(run_dir / "smoke_metrics.json", smoke_history)
            required = int(
                config_value(
                    config, "training.smoke_required_consecutive_perfect_epochs"
                )
            )
            if self.perfect_streak >= required or epoch >= epochs:
                control.should_save = True
                control.should_training_stop = True
            return control

    callback = SmokeCallback() if args.mode == "smoke" else MilestoneCallback()
    trainer = EntityLinkingTrainer(
        model=model,
        args=training_args,
        train_dataset=RawRowsDataset(train_rows),
        data_collator=CompletionOnlyCollator(tokenizer, max_length),
        processing_class=tokenizer,
        callbacks=[callback],
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    if not resume_from:
        shutil.copy2(config_path, run_dir / "training_config.yaml")
        atomic_write_json(resolved_config_path, config)

    manifest["status"] = "training"
    manifest.setdefault("started_at", utc_now())
    manifest["last_started_at"] = utc_now()
    manifest["train_rows"] = len(train_rows)
    manifest["entity_token_count"] = len(entity_token_ids)
    manifest["entity_token_id_min"] = min(entity_token_ids)
    manifest["entity_token_id_max"] = max(entity_token_ids)
    manifest["no_match_token_id"] = no_match_token_id
    atomic_write_json(manifest_path, manifest)

    result = trainer.train(
        resume_from_checkpoint=str(resume_from) if resume_from else None
    )
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()

    checkpoints = discover_checkpoints(run_dir)
    if args.mode == "full":
        found_epochs = {epoch for epoch, _ in checkpoints}
        if found_epochs != checkpoint_epochs:
            raise TrainingToolError(
                f"Expected checkpoints {sorted(checkpoint_epochs)}, found {sorted(found_epochs)}"
            )
    if args.mode == "smoke":
        if not smoke_history or smoke_history[-1]["perfect_streak"] < int(
            config_value(config, "training.smoke_required_consecutive_perfect_epochs")
        ):
            raise TrainingToolError(
                "Smoke training did not reach 100% structured exact accuracy "
                "for the required streak"
            )

    manifest["status"] = "trained"
    manifest["completed_at"] = utc_now()
    manifest["checkpoints"] = []
    for epoch, path in checkpoints:
        metadata = read_json(path / "checkpoint_meta.json")
        manifest["checkpoints"].append(
            {
                "epoch": epoch,
                "path": str(path.relative_to(run_dir)),
                "resumable": bool(metadata.get("resumable")),
            }
        )
    manifest["train_metrics"] = result.metrics
    atomic_write_json(manifest_path, manifest)
    print(f"Training completed: {run_dir}")
    print("Run poc_a/scripts/evaluate.py next for a full run.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TrainingToolError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
