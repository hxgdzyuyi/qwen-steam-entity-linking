#!/usr/bin/env python3
"""Predict closed-set Steam AppIDs with a PoC B classifier artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common.huggingface_repositories import (  # noqa: E402
    RepositoryConfigError,
    repository_for,
)
from steam_entity_classifier import (  # noqa: E402
    ClassifierArtifactError,
    SteamEntityLinker,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text",
        action="append",
        required=True,
        help="Entity text; repeat --text to predict a batch",
    )
    parser.add_argument("--top-k", type=int, default=5)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--model-repo")
    parser.add_argument("--device", help="Optional torch device such as cuda or cpu")
    return parser.parse_args(argv)


def default_model_repository() -> str:
    try:
        repository = repository_for("poc_b")
    except RepositoryConfigError as error:
        raise ClassifierArtifactError(
            f"Invalid Hugging Face repository registry: {error}"
        ) from error
    return str(repository["repo_id"])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    texts = [text.strip() for text in args.text]
    if any(not text for text in texts):
        raise ClassifierArtifactError("--text values cannot be empty")
    if args.top_k <= 0:
        raise ClassifierArtifactError("--top-k must be positive")
    source: str | Path = (
        args.checkpoint.resolve()
        if args.checkpoint
        else (args.model_repo or default_model_repository())
    )
    linker = SteamEntityLinker.from_pretrained(source, device=args.device)
    try:
        predictions = linker.predict(texts, top_k=args.top_k)
    except ValueError as error:
        raise ClassifierArtifactError(str(error)) from error
    payload = predictions[0] if len(predictions) == 1 else predictions
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ClassifierArtifactError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
