#!/usr/bin/env python3
"""Pure helpers for PoC A2 structured entity-linking responses."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Mapping


CANONICAL_LABEL = "标准名称："
APPID_LABEL = "Steam AppID："
NO_MATCH_NAME = "NO_MATCH"
NO_MATCH_TOKEN = "<NO_MATCH>"
ENTITY_TOKEN_PATTERN = re.compile(r"^<GAME_([0-9]+)>$")
STRUCTURED_RESPONSE_PATTERN = re.compile(
    r"\A标准名称：(?P<canonical>[^\r\n]+)\r?\n"
    r"Steam AppID：(?P<token><GAME_[0-9]+>|<NO_MATCH>)\s*\Z"
)


@dataclass(frozen=True)
class Resolution:
    """Strictly resolved structured output.

    ``resolved_token`` is set only when both generated fields agree with the
    frozen entity table. Every other status is a safe rejection.
    """

    status: str
    format_valid: bool
    generated_canonical_name: str
    generated_token: str
    resolved_canonical_name: str | None
    resolved_token: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_entity_name(value: str) -> str:
    """Normalize harmless Unicode/spacing differences without fuzzy matching."""

    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split()).casefold()


def format_structured_response(canonical_name: str, entity_token: str) -> str:
    if not canonical_name.strip():
        raise ValueError("canonical_name cannot be empty")
    if entity_token != NO_MATCH_TOKEN and ENTITY_TOKEN_PATTERN.fullmatch(
        entity_token
    ) is None:
        raise ValueError(f"invalid entity token: {entity_token}")
    if (canonical_name == NO_MATCH_NAME) != (entity_token == NO_MATCH_TOKEN):
        raise ValueError("NO_MATCH name and token must be used together")
    return f"{CANONICAL_LABEL}{canonical_name}\n{APPID_LABEL}{entity_token}"


def entity_scoring_prefix(canonical_name: str) -> str:
    """Teacher-forced prefix immediately before the entity-label position."""

    return f"{CANONICAL_LABEL}{canonical_name}\n{APPID_LABEL}"


def parse_structured_response(text: str) -> tuple[str, str] | None:
    match = STRUCTURED_RESPONSE_PATTERN.fullmatch(text.strip())
    if match is None:
        return None
    return match.group("canonical").strip(), match.group("token")


def canonical_index(canonical_by_token: Mapping[str, str]) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for token, canonical_name in canonical_by_token.items():
        key = normalize_entity_name(canonical_name)
        if not key:
            raise ValueError("canonical entity name cannot be empty")
        if key in index:
            raise ValueError(f"normalized canonical-name collision: {canonical_name}")
        index[key] = (canonical_name, token)
    return index


def resolve_structured_response(
    text: str,
    canonical_by_token: Mapping[str, str],
    *,
    canonical_name_index: Mapping[str, tuple[str, str]] | None = None,
) -> Resolution:
    """Resolve only an exact registered canonical name with a consistent token."""

    parsed = parse_structured_response(text)
    if parsed is None:
        return Resolution("invalid_format", False, "", "", None, None)

    generated_name, generated_token = parsed
    if generated_name == NO_MATCH_NAME and generated_token == NO_MATCH_TOKEN:
        return Resolution(
            "explicit_no_match",
            True,
            generated_name,
            generated_token,
            None,
            None,
        )
    if generated_name == NO_MATCH_NAME or generated_token == NO_MATCH_TOKEN:
        return Resolution(
            "inconsistent_no_match",
            True,
            generated_name,
            generated_token,
            None,
            None,
        )

    index = (
        canonical_index(canonical_by_token)
        if canonical_name_index is None
        else canonical_name_index
    )
    registered = index.get(normalize_entity_name(generated_name))
    if registered is None:
        return Resolution(
            "unregistered_canonical_name",
            True,
            generated_name,
            generated_token,
            None,
            None,
        )
    canonical_name, expected_token = registered
    if generated_token != expected_token:
        return Resolution(
            "inconsistent_entity_token",
            True,
            generated_name,
            generated_token,
            canonical_name,
            None,
        )
    return Resolution(
        "matched",
        True,
        generated_name,
        generated_token,
        canonical_name,
        expected_token,
    )
