#!/usr/bin/env python3
"""Head-only frozen-backbone classifier used by PoC B and its Hub artifact."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from prompt_contract import DEFAULT_PROMPT_TEMPLATE


CLASSIFIER_SCHEMA_VERSION = 2
LEGACY_CLASSIFIER_SCHEMA_VERSION = 1
LEGACY_PROMPT_TEMPLATE = "{surface_form}"
POOLING_METHOD = "last_non_padding"
ALLOWED_TENSOR_KEYS = {
    "down.weight",
    "down.bias",
    "up.weight",
    "up.bias",
    "prototypes",
    "initial_prototypes",
}


class ClassifierArtifactError(RuntimeError):
    """Raised when a classifier artifact is incomplete or unsafe."""


def assert_no_backbone_weights(source: Path) -> None:
    if not source.is_dir():
        raise ClassifierArtifactError(f"classifier artifact is not a directory: {source}")
    forbidden = sorted(
        path.name
        for path in source.iterdir()
        if path.is_file()
        and (
            path.name in {"model.safetensors", "pytorch_model.bin"}
            or (
                path.name.startswith("model-")
                and path.name.endswith(".safetensors")
            )
            or (
                path.name.startswith("pytorch_model-")
                and path.name.endswith(".bin")
            )
        )
    )
    if forbidden:
        raise ClassifierArtifactError(
            f"classifier artifact contains forbidden backbone weights: {forbidden}"
        )


def stable_payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tokenizer_sha256(tokenizer: Any) -> str:
    """Fingerprint all tokenizer state that can change extracted features."""

    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, dict) or not vocabulary:
        raise ClassifierArtifactError("tokenizer vocabulary is empty")

    def json_safe(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    return stable_payload_sha256(
        {
            "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
            "vocabulary": sorted(
                (str(token), int(index)) for token, index in vocabulary.items()
            ),
            "special_tokens_map": json_safe(tokenizer.special_tokens_map),
            "padding_side": tokenizer.padding_side,
            "truncation_side": tokenizer.truncation_side,
            "model_max_length": int(tokenizer.model_max_length),
        }
    )


def pool_last_non_padding(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    if last_hidden_state.ndim != 3:
        raise ValueError("last_hidden_state must have [batch, sequence, hidden]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have [batch, sequence]")
    if last_hidden_state.shape[:2] != attention_mask.shape:
        raise ValueError("hidden states and attention mask shapes do not match")
    mask = attention_mask.to(dtype=torch.bool)
    if torch.any(mask.sum(dim=1) <= 0):
        raise ValueError("every input must contain at least one non-padding token")
    positions = torch.arange(
        attention_mask.shape[1], device=attention_mask.device
    ).expand_as(attention_mask)
    indices = positions.masked_fill(~mask, -1).max(dim=1).values
    batch_indices = torch.arange(
        last_hidden_state.shape[0], device=last_hidden_state.device
    )
    return last_hidden_state[batch_indices, indices]


def freeze_backbone(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("backbone contains trainable parameters")
    return model


@torch.no_grad()
def extract_features(
    model: nn.Module,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    freeze_backbone(model)
    features: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        if "attention_mask" not in encoded:
            raise RuntimeError("tokenizer output must contain attention_mask")
        if int(encoded["attention_mask"].sum(dim=1).max()) > max_length:
            raise RuntimeError(
                f"input exceeds max_length={max_length}; refusing silent truncation"
            )
        outputs = model(**encoded, return_dict=True)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError("backbone output has no last_hidden_state")
        pooled = pool_last_non_padding(hidden, encoded["attention_mask"])
        features.append(pooled.float().cpu())
    if not features:
        hidden_size = int(getattr(model.config, "hidden_size", 0))
        return torch.empty((0, hidden_size), dtype=torch.float32)
    return torch.cat(features, dim=0).contiguous()


def initialize_prototypes(
    features: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("features must have [rows, hidden]")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must align with features")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    prototypes = torch.zeros(
        (num_classes, features.shape[1]), dtype=torch.float32
    )
    counts = torch.zeros(num_classes, dtype=torch.long)
    prototypes.index_add_(0, labels.long(), features.float())
    counts.index_add_(0, labels.long(), torch.ones_like(labels, dtype=torch.long))
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0).flatten().tolist()
        raise ValueError(f"classes have no canonical features: {missing}")
    prototypes = prototypes / counts.unsqueeze(1)
    return F.normalize(prototypes, dim=-1)


class FrozenPrototypeHead(nn.Module):
    """Low-rank residual projector plus trainable cosine prototypes."""

    def __init__(
        self,
        initial_prototypes: torch.Tensor,
        *,
        bottleneck_dim: int,
        temperature: float,
    ) -> None:
        super().__init__()
        if initial_prototypes.ndim != 2:
            raise ValueError("initial_prototypes must have [classes, hidden]")
        if bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        normalized = F.normalize(initial_prototypes.float(), dim=-1)
        hidden_size = normalized.shape[1]
        self.down = nn.Linear(hidden_size, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.prototypes = nn.Parameter(normalized.clone())
        self.register_buffer("initial_prototypes", normalized.clone())
        self.temperature = float(temperature)

    @property
    def hidden_size(self) -> int:
        return int(self.prototypes.shape[1])

    @property
    def num_classes(self) -> int:
        return int(self.prototypes.shape[0])

    @property
    def bottleneck_dim(self) -> int:
        return int(self.down.out_features)

    def projected_features(self, features: torch.Tensor) -> torch.Tensor:
        features = features.float()
        residual = self.up(F.gelu(self.down(features)))
        return F.normalize(features + residual, dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.projected_features(features)
        prototypes = F.normalize(self.prototypes, dim=-1)
        return projected @ prototypes.transpose(0, 1) / self.temperature

    def training_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        anchor_weight: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = self(features)
        cross_entropy = F.cross_entropy(logits, labels.long())
        current = F.normalize(self.prototypes, dim=-1)
        initial = F.normalize(self.initial_prototypes, dim=-1)
        anchor = (1.0 - (current * initial).sum(dim=-1)).mean()
        total = cross_entropy + float(anchor_weight) * anchor
        return total, {
            "loss": float(total.detach()),
            "cross_entropy": float(cross_entropy.detach()),
            "prototype_anchor": float(anchor.detach()),
        }


def classifier_config(
    head: FrozenPrototypeHead,
    *,
    base_model_id: str,
    base_model_revision: str,
    max_length: int,
    prototype_anchor_weight: float,
    feature_cache_sha256: str,
    tokenizer_sha256: str,
    class_map_sha256: str,
    training_config_sha256: str,
    mode: str,
    epoch: int,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
) -> dict[str, Any]:
    return {
        "schema_version": CLASSIFIER_SCHEMA_VERSION,
        "architecture": "frozen-qwen-low-rank-cosine-prototype",
        "base_model_id": base_model_id,
        "base_model_revision": base_model_revision,
        "hidden_size": head.hidden_size,
        "bottleneck_dim": head.bottleneck_dim,
        "num_classes": head.num_classes,
        "temperature": head.temperature,
        "prototype_anchor_weight": float(prototype_anchor_weight),
        "pooling": POOLING_METHOD,
        "max_length": int(max_length),
        "prompt_template": prompt_template,
        "feature_cache_sha256": feature_cache_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "class_map_sha256": class_map_sha256,
        "training_config_sha256": training_config_sha256,
        "mode": mode,
        "epoch": int(epoch),
        "closed_set": True,
    }


def _safe_tensors_state(head: FrozenPrototypeHead) -> dict[str, torch.Tensor]:
    state = {
        key: value.detach().float().cpu().contiguous()
        for key, value in head.state_dict().items()
    }
    if set(state) != ALLOWED_TENSOR_KEYS:
        raise ClassifierArtifactError(
            f"classifier tensor keys are invalid: {sorted(state)}"
        )
    return state


def save_classifier_artifact(
    head: FrozenPrototypeHead,
    destination: Path,
    *,
    config: Mapping[str, Any],
) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise ClassifierArtifactError(
            "safetensors is required to save classifier artifacts"
        ) from error
    destination.mkdir(parents=True, exist_ok=True)
    payload = dict(config)
    validate_classifier_config(payload)
    save_file(_safe_tensors_state(head), destination / "classifier.safetensors")
    (destination / "classifier_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_classifier_config(config: Mapping[str, Any]) -> None:
    schema_version = config.get("schema_version")
    if schema_version not in {
        LEGACY_CLASSIFIER_SCHEMA_VERSION,
        CLASSIFIER_SCHEMA_VERSION,
    }:
        raise ClassifierArtifactError("classifier config has an invalid schema")
    expected = {
        "architecture": "frozen-qwen-low-rank-cosine-prototype",
        "pooling": POOLING_METHOD,
        "closed_set": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ClassifierArtifactError(f"classifier config {key} is invalid")
    for key in (
        "base_model_id",
        "base_model_revision",
        "prompt_template",
        "feature_cache_sha256",
        "tokenizer_sha256",
        "class_map_sha256",
        "training_config_sha256",
    ):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ClassifierArtifactError(f"classifier config {key} is missing")
    prompt_template = str(config["prompt_template"])
    expected_prompt_template = (
        LEGACY_PROMPT_TEMPLATE
        if schema_version == LEGACY_CLASSIFIER_SCHEMA_VERSION
        else DEFAULT_PROMPT_TEMPLATE
    )
    if (
        prompt_template != expected_prompt_template
        or prompt_template.count("{surface_form}") != 1
        or not prompt_template.endswith("{surface_form}")
    ):
        raise ClassifierArtifactError(
            "classifier prompt template does not match its schema contract"
        )
    if config["base_model_id"] != "Qwen/Qwen3-8B-Base":
        raise ClassifierArtifactError("classifier must use Qwen/Qwen3-8B-Base")
    if re.fullmatch(r"[0-9a-f]{40}", config["base_model_revision"]) is None:
        raise ClassifierArtifactError("classifier base revision must be a commit SHA")
    for key in (
        "feature_cache_sha256",
        "tokenizer_sha256",
        "class_map_sha256",
        "training_config_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", config[key]) is None:
            raise ClassifierArtifactError(f"classifier config {key} is not SHA-256")
    for key in (
        "hidden_size",
        "bottleneck_dim",
        "num_classes",
        "max_length",
        "epoch",
    ):
        if (
            not isinstance(config.get(key), int)
            or isinstance(config[key], bool)
            or config[key] < (0 if key == "epoch" else 1)
        ):
            raise ClassifierArtifactError(f"classifier config {key} is invalid")
    if not isinstance(config.get("temperature"), (int, float)) or float(
        config["temperature"]
    ) <= 0:
        raise ClassifierArtifactError("classifier temperature is invalid")
    if not isinstance(config.get("prototype_anchor_weight"), (int, float)) or float(
        config["prototype_anchor_weight"]
    ) < 0:
        raise ClassifierArtifactError("classifier prototype anchor weight is invalid")
    if config.get("mode") not in {"smoke", "full"}:
        raise ClassifierArtifactError("classifier mode is invalid")
    fixed = {
        "bottleneck_dim": 256,
        "temperature": 0.05,
        "prototype_anchor_weight": 0.01,
        "max_length": 256,
        "prompt_template": expected_prompt_template,
    }
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ClassifierArtifactError(
                f"classifier config fixes {key}={expected!r}"
            )
    expected_classes = 32 if config["mode"] == "smoke" else 1000
    if config["num_classes"] != expected_classes:
        raise ClassifierArtifactError(
            f"classifier {config['mode']} mode requires {expected_classes} classes"
        )
    if config["epoch"] > 20:
        raise ClassifierArtifactError("classifier epoch exceeds the training contract")


def load_classifier_artifact(
    source: Path, *, device: torch.device | str = "cpu"
) -> tuple[FrozenPrototypeHead, dict[str, Any]]:
    assert_no_backbone_weights(source)
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise ClassifierArtifactError(
            "safetensors is required to load classifier artifacts"
        ) from error
    try:
        config = json.loads(
            (source / "classifier_config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ClassifierArtifactError(
            f"cannot read classifier config from {source}: {error}"
        ) from error
    if not isinstance(config, dict):
        raise ClassifierArtifactError("classifier config must be an object")
    validate_classifier_config(config)
    try:
        state = load_file(source / "classifier.safetensors", device="cpu")
    except (OSError, RuntimeError) as error:
        raise ClassifierArtifactError(
            f"cannot read classifier tensors from {source}: {error}"
        ) from error
    if set(state) != ALLOWED_TENSOR_KEYS:
        raise ClassifierArtifactError(
            f"artifact contains forbidden or missing tensors: {sorted(state)}"
        )
    initial = state["initial_prototypes"]
    if tuple(initial.shape) != (
        int(config["num_classes"]),
        int(config["hidden_size"]),
    ):
        raise ClassifierArtifactError("initial prototype shape differs from config")
    head = FrozenPrototypeHead(
        initial,
        bottleneck_dim=int(config["bottleneck_dim"]),
        temperature=float(config["temperature"]),
    )
    try:
        head.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ClassifierArtifactError(
            f"classifier tensor shapes differ from config: {error}"
        ) from error
    head.to(device)
    head.eval()
    return head, config


def validate_class_map(payload: Any, expected_classes: int | None = None) -> list[dict[str, Any]]:
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 1
        or payload.get("ordering") != "numeric_appid_ascending"
        or not isinstance(payload.get("classes"), list)
    ):
        raise ClassifierArtifactError("class map has an invalid schema")
    classes = payload["classes"]
    if expected_classes is not None and len(classes) != expected_classes:
        raise ClassifierArtifactError("class map count differs from classifier")
    seen_appids: set[int] = set()
    seen_names: set[str] = set()
    for index, row in enumerate(classes):
        if (
            not isinstance(row, dict)
            or set(row)
            != {"class_index", "appid", "canonical_name", "cohort"}
            or row.get("class_index") != index
        ):
            raise ClassifierArtifactError("class map indices are not contiguous")
        if not isinstance(row.get("appid"), int) or isinstance(row["appid"], bool):
            raise ClassifierArtifactError("class map contains an invalid AppID")
        if row["appid"] in seen_appids:
            raise ClassifierArtifactError("class map contains duplicate AppIDs")
        seen_appids.add(row["appid"])
        if not isinstance(row.get("canonical_name"), str) or not row[
            "canonical_name"
        ].strip():
            raise ClassifierArtifactError("class map contains an invalid name")
        name_key = row["canonical_name"].casefold()
        if name_key in seen_names:
            raise ClassifierArtifactError("class map contains duplicate names")
        seen_names.add(name_key)
        if not isinstance(row.get("cohort"), str) or not row["cohort"].strip():
            raise ClassifierArtifactError("class map contains an invalid cohort")
    appids = [int(row["appid"]) for row in classes]
    if appids != sorted(appids):
        raise ClassifierArtifactError("class map AppIDs are not numerically sorted")
    return [dict(row) for row in classes]


def load_class_map(path: Path, expected_classes: int | None = None) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ClassifierArtifactError(f"cannot read class map {path}: {error}") from error
    return validate_class_map(payload, expected_classes)


def decode_logits(
    logits: torch.Tensor,
    classes: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    if logits.ndim != 2 or logits.shape[1] != len(classes):
        raise ValueError("logits shape does not match class map")
    if not 0 < top_k <= len(classes):
        raise ValueError("top_k must be within the class count")
    cpu_logits = logits.float().cpu()
    probabilities = torch.softmax(cpu_logits, dim=-1)
    decoded: list[dict[str, Any]] = []
    for row_logits, row_probabilities in zip(cpu_logits, probabilities):
        row_indices = sorted(
            range(len(classes)),
            key=lambda index: (-float(row_logits[index]), index),
        )[:top_k]
        candidates = []
        for class_index in row_indices:
            item = classes[class_index]
            candidates.append(
                {
                    "class_index": class_index,
                    "appid": int(item["appid"]),
                    "canonical_name": str(item["canonical_name"]),
                    "confidence": float(row_probabilities[class_index]),
                }
            )
        decoded.append({"prediction": candidates[0], "top_k": candidates})
    return decoded


class SteamEntityLinker:
    """Load a head-only artifact with its pinned frozen Qwen backbone."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        tokenizer: Any,
        head: FrozenPrototypeHead,
        classifier_settings: Mapping[str, Any],
        classes: Sequence[Mapping[str, Any]],
        device: torch.device,
    ) -> None:
        self.backbone = freeze_backbone(backbone)
        self.tokenizer = tokenizer
        self.head = head.eval()
        self.config = dict(classifier_settings)
        self.classes = [dict(row) for row in classes]
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        artifact_path: str | Path,
        *,
        device: str | torch.device | None = None,
        token: str | None = None,
    ) -> "SteamEntityLinker":
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ClassifierArtifactError(
                "transformers is required to load the frozen backbone"
            ) from error
        raw_source = str(artifact_path)
        source = Path(artifact_path)
        if not source.exists():
            if source.is_absolute() or raw_source.startswith("."):
                raise ClassifierArtifactError(
                    f"local classifier artifact does not exist: {source}"
                )
            try:
                from huggingface_hub import snapshot_download
            except ImportError as error:
                raise ClassifierArtifactError(
                    "huggingface-hub is required to load a repository ID"
                ) from error
            source = Path(
                snapshot_download(
                    repo_id=raw_source,
                    token=token,
                    allow_patterns=[
                        "classifier.safetensors",
                        "classifier_config.json",
                        "class_map.json",
                        "added_tokens.json",
                        "chat_template.jinja",
                        "merges.txt",
                        "special_tokens_map.json",
                        "tokenizer.json",
                        "tokenizer.model",
                        "tokenizer_config.json",
                        "vocab.json",
                    ],
                )
            )
        selected_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        head, settings = load_classifier_artifact(
            source, device=selected_device
        )
        classes = load_class_map(
            source / "class_map.json", expected_classes=head.num_classes
        )
        class_payload = {
            "schema_version": 1,
            "ordering": "numeric_appid_ascending",
            "classes": classes,
        }
        if stable_payload_sha256(class_payload) != settings["class_map_sha256"]:
            raise ClassifierArtifactError(
                "class map fingerprint differs from classifier config"
            )
        tokenizer_source: str | Path = (
            source
            if (source / "tokenizer_config.json").is_file()
            else str(settings["base_model_id"])
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            revision=(
                None
                if tokenizer_source == source
                else str(settings["base_model_revision"])
            ),
            token=token,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        if tokenizer_sha256(tokenizer) != settings["tokenizer_sha256"]:
            raise ClassifierArtifactError(
                "tokenizer fingerprint differs from classifier config"
            )
        dtype = torch.bfloat16 if selected_device.type == "cuda" else torch.float32
        backbone = AutoModel.from_pretrained(
            str(settings["base_model_id"]),
            revision=str(settings["base_model_revision"]),
            token=token,
            trust_remote_code=False,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(selected_device)
        if int(getattr(backbone.config, "hidden_size", -1)) != int(
            settings["hidden_size"]
        ):
            raise ClassifierArtifactError(
                "pinned backbone hidden size differs from classifier config"
            )
        return cls(
            backbone=backbone,
            tokenizer=tokenizer,
            head=head,
            classifier_settings=settings,
            classes=classes,
            device=selected_device,
        )

    def predict(
        self, texts: Sequence[str], *, top_k: int = 5
    ) -> list[dict[str, Any]]:
        template = str(self.config["prompt_template"])
        model_inputs = [
            template.format(surface_form=str(text)) for text in texts
        ]
        features = extract_features(
            self.backbone,
            self.tokenizer,
            model_inputs,
            batch_size=max(1, min(32, len(model_inputs))),
            max_length=int(self.config["max_length"]),
            device=self.device,
        ).to(self.device)
        with torch.no_grad():
            logits = self.head(features)
        decoded = decode_logits(logits, self.classes, top_k=top_k)
        return [
            {"input": text, **result["prediction"], "top_k": result["top_k"]}
            for text, result in zip(texts, decoded)
        ]
