#!/usr/bin/env python3
"""Build deterministic canonical-only data for PoC B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from prompt_contract import (
    DEFAULT_PROMPT_STYLE,
    DEFAULT_PROMPT_TEMPLATE,
    PROMPT_STYLES,
    prompt_style_names,
    render_prompt,
)


POC_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = POC_ROOT.parent
EXPECTED_CLASSES = 1000
EXPECTED_TRAIN_ROWS = 6000
EXPECTED_CANONICAL_ROWS = 1000
EXPECTED_ALIAS_CASES = 184
EXPECTED_ALIAS_ROWS = EXPECTED_ALIAS_CASES * len(PROMPT_STYLES)


class BuildError(RuntimeError):
    """Raised when source data cannot produce a valid PoC B dataset."""


@dataclass(frozen=True)
class Entity:
    class_index: int
    appid: int
    canonical_name: str
    cohort: str


def clean_required_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise BuildError(f"{field} must be a string")
    cleaned = " ".join(value.split()).strip()
    if not cleaned:
        raise BuildError(f"{field} cannot be empty")
    return cleaned


def load_entities(dataset_path: Path, provenance_path: Path) -> list[Entity]:
    try:
        with provenance_path.open(encoding="utf-8", newline="") as handle:
            provenance_rows = list(csv.DictReader(handle))
    except OSError as error:
        raise BuildError(f"Cannot read provenance {provenance_path}: {error}") from error
    provenance_by_appid: dict[int, dict[str, str]] = {}
    for line_number, row in enumerate(provenance_rows, start=2):
        raw_appid = row.get("appid", "")
        if not raw_appid.isdigit():
            raise BuildError(
                f"provenance line {line_number} has invalid AppID {raw_appid!r}"
            )
        appid = int(raw_appid)
        if appid in provenance_by_appid:
            raise BuildError(f"duplicate provenance AppID: {appid}")
        if row.get("item_type") != "game":
            raise BuildError(f"non-game provenance item for AppID {appid}")
        provenance_by_appid[appid] = row

    try:
        with dataset_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["canonical_name", "appid"]:
                raise BuildError(
                    f"{dataset_path} must contain exactly canonical_name,appid"
                )
            dataset_rows = list(reader)
    except OSError as error:
        raise BuildError(f"Cannot read dataset {dataset_path}: {error}") from error

    raw_entities: list[tuple[int, str, str]] = []
    seen_appids: set[int] = set()
    seen_names: set[str] = set()
    for line_number, row in enumerate(dataset_rows, start=2):
        canonical_name = clean_required_text(
            row.get("canonical_name"), f"dataset line {line_number} canonical_name"
        )
        raw_appid = row.get("appid", "")
        if not isinstance(raw_appid, str) or not raw_appid.isdigit():
            raise BuildError(
                f"dataset line {line_number} has invalid AppID {raw_appid!r}"
            )
        appid = int(raw_appid)
        name_key = canonical_name.casefold()
        if appid in seen_appids:
            raise BuildError(f"duplicate dataset AppID: {appid}")
        if name_key in seen_names:
            raise BuildError(f"duplicate canonical name: {canonical_name}")
        if appid not in provenance_by_appid:
            raise BuildError(f"missing provenance for AppID {appid}")
        seen_appids.add(appid)
        seen_names.add(name_key)
        raw_entities.append(
            (appid, canonical_name, provenance_by_appid[appid]["cohort"])
        )
    if set(provenance_by_appid) != seen_appids:
        raise BuildError("dataset and provenance AppID sets differ")

    return [
        Entity(
            class_index=class_index,
            appid=appid,
            canonical_name=canonical_name,
            cohort=cohort,
        )
        for class_index, (appid, canonical_name, cohort) in enumerate(
            sorted(raw_entities)
        )
    ]


def render_prompt_view(surface_form: str, style_index: int) -> tuple[str, str]:
    style_name = PROMPT_STYLES[style_index % len(PROMPT_STYLES)][0]
    try:
        rendered = render_prompt(surface_form, style_name)
    except ValueError as error:
        raise BuildError(str(error)) from error
    return style_name, rendered


def class_map_payload(entities: Sequence[Entity]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "ordering": "numeric_appid_ascending",
        "classes": [
            {
                "class_index": entity.class_index,
                "appid": entity.appid,
                "canonical_name": entity.canonical_name,
                "cohort": entity.cohort,
            }
            for entity in entities
        ],
    }


def _row(
    entity: Entity,
    *,
    surface_form: str,
    model_input: str,
    prompt_style: str,
) -> dict[str, Any]:
    return {
        "surface_form": surface_form,
        "model_input": model_input,
        "class_index": entity.class_index,
        "appid": entity.appid,
        "canonical_name": entity.canonical_name,
        "cohort": entity.cohort,
        "prompt_style": prompt_style,
    }


def build_train_rows(entities: Sequence[Entity]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity in entities:
        for style_index in range(len(PROMPT_STYLES)):
            style_name, model_input = render_prompt_view(
                entity.canonical_name, style_index
            )
            rows.append(
                _row(
                    entity,
                    surface_form=entity.canonical_name,
                    model_input=model_input,
                    prompt_style=style_name,
                )
            )
    return rows


def build_canonical_eval_rows(
    entities: Sequence[Entity],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity in entities:
        style_name = DEFAULT_PROMPT_STYLE
        model_input = render_prompt(entity.canonical_name, style_name)
        row = _row(
            entity,
            surface_form=entity.canonical_name,
            model_input=model_input,
            prompt_style=style_name,
        )
        row["type"] = "canonical"
        rows.append(row)
    return rows


def load_alias_source(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BuildError(f"Cannot read alias source {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise BuildError("alias source must be a schema_version=1 object")
    return payload


def build_alias_eval_rows(
    payload: dict[str, Any], entities: Sequence[Entity]
) -> tuple[list[dict[str, Any]], int]:
    by_appid = {entity.appid: entity for entity in entities}
    games = payload.get("games")
    if not isinstance(games, list) or not games:
        raise BuildError("alias source games must be a non-empty list")
    rows: list[dict[str, Any]] = []
    seen_appids: set[int] = set()
    seen_inputs: set[str] = set()
    for game_index, game in enumerate(games, start=1):
        if not isinstance(game, dict):
            raise BuildError(f"alias game {game_index} must be an object")
        appid = game.get("appid")
        if not isinstance(appid, int) or appid not in by_appid:
            raise BuildError(f"alias game {game_index} has unknown AppID {appid!r}")
        if appid in seen_appids:
            raise BuildError(f"duplicate alias game AppID: {appid}")
        seen_appids.add(appid)
        entity = by_appid[appid]
        cases = game.get("cases")
        if not isinstance(cases, list) or not cases:
            raise BuildError(f"alias game AppID {appid} must contain cases")
        for case_index, case in enumerate(cases, start=1):
            if not isinstance(case, dict):
                raise BuildError(
                    f"alias AppID {appid} case {case_index} must be an object"
                )
            surface_form = clean_required_text(
                case.get("input"), f"alias AppID {appid} input"
            )
            case_type = clean_required_text(
                case.get("type"), f"alias AppID {appid} type"
            )
            input_key = surface_form.casefold()
            if input_key == entity.canonical_name.casefold():
                raise BuildError(
                    f"alias leaks canonical name for AppID {appid}: {surface_form}"
                )
            if input_key in seen_inputs:
                raise BuildError(f"duplicate alias input: {surface_form}")
            seen_inputs.add(input_key)
            for style_name in prompt_style_names():
                model_input = render_prompt(surface_form, style_name)
                row = _row(
                    entity,
                    surface_form=surface_form,
                    model_input=model_input,
                    prompt_style=style_name,
                )
                row["type"] = case_type
                rows.append(row)
    return rows, len(seen_appids)


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


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                )
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
        raise BuildError(f"Cannot hash {path}: {error}") from error
    return digest.hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPOSITORY_ROOT / "common/data/steam_games.csv",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=REPOSITORY_ROOT / "common/data/steam_games.provenance.csv",
    )
    parser.add_argument(
        "--alias-source",
        type=Path,
        default=REPOSITORY_ROOT / "common/data/eval_alias.source.json",
    )
    parser.add_argument(
        "--train-output", type=Path, default=POC_ROOT / "data/train.jsonl"
    )
    parser.add_argument(
        "--canonical-output",
        type=Path,
        default=POC_ROOT / "data/eval_canonical.jsonl",
    )
    parser.add_argument(
        "--alias-output", type=Path, default=POC_ROOT / "data/eval_alias.jsonl"
    )
    parser.add_argument(
        "--class-map-output", type=Path, default=POC_ROOT / "data/class_map.json"
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=POC_ROOT / "data/data_manifest.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    entities = load_entities(args.dataset, args.provenance)
    train_rows = build_train_rows(entities)
    canonical_rows = build_canonical_eval_rows(entities)
    alias_rows, alias_game_count = build_alias_eval_rows(
        load_alias_source(args.alias_source), entities
    )
    actual_counts = (
        len(entities),
        len(train_rows),
        len(canonical_rows),
        len(alias_rows),
    )
    expected_counts = (
        EXPECTED_CLASSES,
        EXPECTED_TRAIN_ROWS,
        EXPECTED_CANONICAL_ROWS,
        EXPECTED_ALIAS_ROWS,
    )
    if actual_counts != expected_counts:
        raise BuildError(
            f"PoC B fixed data contract requires {expected_counts}, got {actual_counts}"
        )
    atomic_write_json(args.class_map_output, class_map_payload(entities))
    atomic_write_jsonl(args.train_output, train_rows)
    atomic_write_jsonl(args.canonical_output, canonical_rows)
    atomic_write_jsonl(args.alias_output, alias_rows)
    atomic_write_json(
        args.manifest_output,
        {
            "schema_version": 2,
            "class_ordering": "numeric_appid_ascending",
            "class_count": len(entities),
            "train_rows": len(train_rows),
            "canonical_eval_rows": len(canonical_rows),
            "alias_eval_rows": len(alias_rows),
            "alias_eval_cases": EXPECTED_ALIAS_CASES,
            "alias_eval_games": alias_game_count,
            "prompt_styles": list(prompt_style_names()),
            "default_prompt_style": DEFAULT_PROMPT_STYLE,
            "default_prompt_template": DEFAULT_PROMPT_TEMPLATE,
            "paired_alias_prompt_evaluation": True,
            "canonical_only_training": True,
            "alias_type_counts": {
                case_type: sum(row["type"] == case_type for row in alias_rows)
                for case_type in sorted({row["type"] for row in alias_rows})
            },
            "alias_case_type_counts": {
                case_type: sum(
                    row["type"] == case_type
                    and row["prompt_style"] == DEFAULT_PROMPT_STYLE
                    for row in alias_rows
                )
                for case_type in sorted({row["type"] for row in alias_rows})
            },
            "source_sha256": {
                "entities": sha256_file(args.dataset),
                "provenance": sha256_file(args.provenance),
                "alias_source": sha256_file(args.alias_source),
            },
            "derived_sha256": {
                "class_map": sha256_file(args.class_map_output),
                "train": sha256_file(args.train_output),
                "canonical_eval": sha256_file(args.canonical_output),
                "alias_eval": sha256_file(args.alias_output),
            },
        },
    )
    print(f"Wrote {len(entities)} classes to {args.class_map_output}")
    print(f"Wrote {len(train_rows)} canonical-only rows to {args.train_output}")
    print(f"Wrote {len(canonical_rows)} canonical eval rows")
    print(f"Wrote {len(alias_rows)} held-out alias rows")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as error:
        raise SystemExit(f"error: {error}") from error
