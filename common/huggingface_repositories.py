"""Load and validate the Hugging Face repositories assigned to each PoC."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


REGISTRY_PATH = Path(__file__).with_name("huggingface_repositories.json")
REQUIRED_EXPERIMENTS = {"poc_a", "poc_b"}
ALLOWED_STATUSES = {"active", "planned", "retired"}
REPO_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$"
)


class RepositoryConfigError(ValueError):
    """Raised when the shared Hugging Face repository registry is invalid."""


def validate_repo_id(value: Any) -> str:
    if not isinstance(value, str) or REPO_ID_PATTERN.fullmatch(value) is None:
        raise RepositoryConfigError(
            "repo_id must be a user-or-org/model identifier without a URL"
        )
    return value


def validate_repository_registry(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise RepositoryConfigError("repository registry must use schema_version=1")
    repositories = payload.get("repositories")
    if not isinstance(repositories, dict):
        raise RepositoryConfigError("repository registry must contain repositories")
    missing = REQUIRED_EXPERIMENTS.difference(repositories)
    if missing:
        raise RepositoryConfigError(
            f"repository registry is missing experiments: {sorted(missing)}"
        )

    seen_repo_ids: set[str] = set()
    for experiment, raw_config in repositories.items():
        if not isinstance(experiment, str) or not experiment:
            raise RepositoryConfigError("experiment keys must be non-empty strings")
        if not isinstance(raw_config, dict):
            raise RepositoryConfigError(
                f"repository configuration for {experiment} must be an object"
            )
        repo_id = validate_repo_id(raw_config.get("repo_id"))
        if repo_id in seen_repo_ids:
            raise RepositoryConfigError(
                f"repository {repo_id} is assigned to more than one experiment"
            )
        seen_repo_ids.add(repo_id)
        if raw_config.get("repo_type") != "model":
            raise RepositoryConfigError(
                f"{experiment} must point to a Hugging Face model repository"
            )
        if raw_config.get("status") not in ALLOWED_STATUSES:
            raise RepositoryConfigError(
                f"{experiment} has an unsupported repository status"
            )
        artifact_kind = raw_config.get("artifact_kind")
        if not isinstance(artifact_kind, str) or not artifact_kind:
            raise RepositoryConfigError(
                f"{experiment} must declare a non-empty artifact_kind"
            )
    return payload


def load_repository_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RepositoryConfigError(
            f"cannot read Hugging Face repository registry {path}: {error}"
        ) from error
    return validate_repository_registry(payload)


def repository_for(
    experiment: str, registry: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    payload = (
        load_repository_registry()
        if registry is None
        else validate_repository_registry(dict(registry))
    )
    repositories = payload["repositories"]
    if experiment not in repositories:
        raise RepositoryConfigError(
            f"no Hugging Face repository is registered for {experiment}"
        )
    return dict(repositories[experiment])
