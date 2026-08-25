#!/usr/bin/env python3
"""Build deterministic PoC A training, special-token, and evaluation files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


POC_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = POC_ROOT.parent
PROMPT_STYLES: tuple[tuple[str, str], ...] = (
    ("entity_id_label", "游戏信息：{canonical_name}\nSteam实体Id："),
    ("appid_label", "游戏信息：{canonical_name}\nSteam AppID："),
    ("steam_de_appid", "游戏信息：{canonical_name}\nSteam 的 AppID："),
    ("steam_de_appid_mixed_case", "游戏信息：{canonical_name}\nSteam 的 AppId："),
    ("appid_question", "{canonical_name} 的 Steam AppID 是什么？\n答案："),
    ("appid_request", "请返回 {canonical_name} 对应的 Steam AppId："),
)
ENTITY_TOKEN_PATTERN = re.compile(r"^<GAME_([0-9]+)>$")


class BuildError(RuntimeError):
    """Raised when source data would produce an invalid experiment."""


@dataclass(frozen=True)
class Entity:
    canonical_name: str
    appid: int

    @property
    def token(self) -> str:
        return f"<GAME_{self.appid}>"


def clean_required_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise BuildError(f"{field} must be a string")
    cleaned = " ".join(value.split()).strip()
    if not cleaned:
        raise BuildError(f"{field} cannot be empty")
    return cleaned


def load_entities(path: Path) -> list[Entity]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["canonical_name", "appid"]:
                raise BuildError(
                    f"{path} must contain exactly canonical_name,appid columns"
                )
            rows = list(reader)
    except OSError as error:
        raise BuildError(f"Cannot read dataset {path}: {error}") from error

    entities: list[Entity] = []
    seen_appids: set[int] = set()
    seen_names: set[str] = set()
    for line_number, row in enumerate(rows, start=2):
        name = clean_required_text(row.get("canonical_name"), f"line {line_number} name")
        raw_appid = row.get("appid", "")
        if not isinstance(raw_appid, str) or not raw_appid.isdigit():
            raise BuildError(f"line {line_number} has an invalid AppID: {raw_appid!r}")
        appid = int(raw_appid)
        name_key = name.casefold()
        if appid in seen_appids:
            raise BuildError(f"duplicate AppID in dataset: {appid}")
        if name_key in seen_names:
            raise BuildError(f"duplicate canonical name in dataset: {name}")
        seen_appids.add(appid)
        seen_names.add(name_key)
        entities.append(Entity(canonical_name=name, appid=appid))

    if not entities:
        raise BuildError("dataset is empty")
    return entities


def render_prompt(canonical_name: str, style_index: int) -> tuple[str, str]:
    style_name, template = PROMPT_STYLES[style_index % len(PROMPT_STYLES)]
    return style_name, template.format(canonical_name=canonical_name)


def build_train_rows(entities: Sequence[Entity], seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    entities_for_assignment = list(entities)
    rng.shuffle(entities_for_assignment)
    rows = []
    for index, entity in enumerate(entities_for_assignment):
        _, prompt = render_prompt(entity.canonical_name, index)
        rows.append({"prompt": prompt, "completion": entity.token})
    rng.shuffle(rows)
    return rows


def build_special_tokens(entities: Sequence[Entity]) -> list[str]:
    return [entity.token for entity in sorted(entities, key=lambda item: item.appid)]


def load_eval_source(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise BuildError(f"Cannot read evaluation source {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise BuildError("evaluation source must be a schema_version=1 JSON object")
    return payload


def build_eval_rows(
    payload: dict[str, Any], entities: Sequence[Entity]
) -> tuple[list[dict[str, str]], int]:
    by_appid = {entity.appid: entity for entity in entities}
    games = payload.get("games")
    if not isinstance(games, list) or not games:
        raise BuildError("evaluation source games must be a non-empty list")

    rows: list[dict[str, str]] = []
    seen_game_ids: set[int] = set()
    seen_inputs: dict[str, str] = {}
    for game_index, game in enumerate(games, start=1):
        if not isinstance(game, dict):
            raise BuildError(f"evaluation game {game_index} must be an object")
        raw_appid = game.get("appid")
        if not isinstance(raw_appid, int) or raw_appid not in by_appid:
            raise BuildError(f"evaluation game {game_index} has unknown AppID {raw_appid!r}")
        if raw_appid in seen_game_ids:
            raise BuildError(f"evaluation AppID appears more than once: {raw_appid}")
        seen_game_ids.add(raw_appid)

        entity = by_appid[raw_appid]
        cases = game.get("cases")
        if not isinstance(cases, list) or not cases:
            raise BuildError(f"evaluation AppID {raw_appid} must have cases")
        for case_index, case in enumerate(cases, start=1):
            if not isinstance(case, dict):
                raise BuildError(
                    f"evaluation AppID {raw_appid} case {case_index} must be an object"
                )
            input_text = clean_required_text(
                case.get("input"), f"evaluation AppID {raw_appid} case input"
            )
            case_type = clean_required_text(
                case.get("type"), f"evaluation AppID {raw_appid} case type"
            )
            input_key = input_text.casefold()
            if input_key == entity.canonical_name.casefold():
                raise BuildError(
                    f"evaluation input leaks the canonical training name: {input_text}"
                )
            previous_target = seen_inputs.get(input_key)
            if previous_target is not None:
                raise BuildError(
                    f"duplicate/ambiguous evaluation input {input_text!r}; "
                    f"already targets {previous_target}"
                )
            seen_inputs[input_key] = entity.token
            style_name, prompt = render_prompt(input_text, len(rows))
            rows.append(
                {
                    "input": input_text,
                    "prompt": prompt,
                    "expected": entity.token,
                    "type": case_type,
                    "prompt_style": style_name,
                }
            )

    return rows, len(seen_game_ids)


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


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, str]]) -> None:
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
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPOSITORY_ROOT / "common/data/steam_games.csv",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=POC_ROOT / "data/train.jsonl",
    )
    parser.add_argument(
        "--tokens-output",
        type=Path,
        default=POC_ROOT / "data/special_tokens.json",
    )
    parser.add_argument(
        "--eval-source",
        type=Path,
        default=REPOSITORY_ROOT / "common/data/eval_alias.source.json",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=POC_ROOT / "data/eval_alias.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    entities = load_entities(args.dataset)
    train_rows = build_train_rows(entities, args.seed)
    special_tokens = build_special_tokens(entities)
    eval_payload = load_eval_source(args.eval_source)
    eval_rows, eval_game_count = build_eval_rows(eval_payload, entities)

    if any(ENTITY_TOKEN_PATTERN.fullmatch(token) is None for token in special_tokens):
        raise BuildError("an invalid entity token was generated")
    if len(special_tokens) != len(set(special_tokens)):
        raise BuildError("duplicate entity tokens were generated")

    atomic_write_jsonl(args.train_output, train_rows)
    atomic_write_json(args.tokens_output, special_tokens)
    atomic_write_jsonl(args.eval_output, eval_rows)

    print(
        f"Wrote {len(train_rows)} training rows to {args.train_output}",
        file=sys.stderr,
    )
    print(
        f"Wrote {len(special_tokens)} special tokens to {args.tokens_output}",
        file=sys.stderr,
    )
    print(
        f"Wrote {len(eval_rows)} held-out cases for {eval_game_count} games "
        f"to {args.eval_output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
