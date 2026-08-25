#!/usr/bin/env python3
"""Build the shared Steam entity-linking dataset.

The generated entity CSV intentionally contains only the two columns described
in common/README.md: ``canonical_name`` and ``appid``.  A provenance CSV and a
metadata JSON file are written next to it so that a changing Steam snapshot
remains auditable.

No Steam Web API key or third-party Python package is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from http.client import HTTPException
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MOST_PLAYED_URL = (
    "https://api.steampowered.com/ISteamChartsService/GetMostPlayedGames/v1/"
)
STORE_SEARCH_URL = "https://store.steampowered.com/search/results/"
USER_AGENT = (
    "qwen-steam-entity-linking/0.1 "
    "(dataset synchronization; https://store.steampowered.com/)"
)
PAGE_SIZE = 100
STEAM_GAMES_CATEGORY_ID = 998
HTML_VOID_ELEMENTS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


class SyncError(RuntimeError):
    """Raised when a source cannot provide enough valid games."""


@dataclass(frozen=True)
class Game:
    appid: int
    canonical_name: str
    cohort: str
    source: str
    source_rank: int
    release_date: str = ""
    item_type: str = "game"


@dataclass(frozen=True)
class SearchResult:
    appid: int
    canonical_name: str
    release_date: str


class SteamSearchParser(HTMLParser):
    """Extract individual app rows from Steam Store search-result HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[SearchResult] = []
        self._depth = 0
        self._row_depth: int | None = None
        self._appid: int | None = None
        self._title = ""
        self._release_date = ""
        self._capture: tuple[str, int] | None = None
        self._capture_text: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag not in HTML_VOID_ELEMENTS:
            self._depth += 1
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())

        if (
            tag == "a"
            and self._row_depth is None
            and "search_result_row" in classes
        ):
            raw_appid = attributes.get("data-ds-appid") or ""
            if raw_appid.isdigit():
                self._row_depth = self._depth
                self._appid = int(raw_appid)
                self._title = ""
                self._release_date = ""

        if self._row_depth is None or self._capture is not None:
            return

        if tag == "span" and "title" in classes:
            self._capture = ("title", self._depth)
            self._capture_text = []
        elif tag == "div" and "search_released" in classes:
            self._capture = ("release_date", self._depth)
            self._capture_text = []

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in HTML_VOID_ELEMENTS:
            self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._capture_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in HTML_VOID_ELEMENTS:
            return
        if self._capture is not None and self._capture[1] == self._depth:
            value = clean_text("".join(self._capture_text))
            if self._capture[0] == "title":
                self._title = clean_canonical_name(value)
            else:
                self._release_date = value
            self._capture = None
            self._capture_text = []

        if self._row_depth is not None and self._row_depth == self._depth:
            if self._appid is not None and self._title:
                self.results.append(
                    SearchResult(
                        appid=self._appid,
                        canonical_name=self._title,
                        release_date=self._release_date,
                    )
                )
            self._row_depth = None
            self._appid = None
            self._title = ""
            self._release_date = ""
            self._capture = None
            self._capture_text = []

        self._depth = max(0, self._depth - 1)


def clean_text(value: str) -> str:
    return " ".join(value.split()).strip()


def clean_canonical_name(value: str) -> str:
    """Normalize a Steam name and remove configured title decorations."""

    without_title_marks = value.translate(str.maketrans("", "", "《》〈〉"))
    normalized = clean_text(without_title_marks)

    # Steam sometimes returns bilingual display names such as
    # "Palworld / 幻兽帕鲁" or "English Name - 中文名". Prefer the Chinese
    # segment in those cases. Otherwise, a whitespace-surrounded ASCII hyphen
    # separates a subtitle. Colons and word-internal hyphens are retained.
    slash_parts = re.split(r"\s+/\s+", normalized)
    chinese_slash_parts = [part for part in slash_parts if is_chinese_name(part)]
    if chinese_slash_parts:
        normalized = chinese_slash_parts[-1]

    hyphen_parts = re.split(r"\s+-\s+", normalized)
    chinese_hyphen_parts = [part for part in hyphen_parts if is_chinese_name(part)]
    if chinese_hyphen_parts:
        normalized = chinese_hyphen_parts[-1]
    else:
        normalized = hyphen_parts[0]

    return clean_text(normalized)


def is_chinese_name(value: str) -> bool:
    """Heuristically identify a Chinese title returned by the Chinese store."""

    has_han = re.search(r"[\u3400-\u9fff]", value) is not None
    has_japanese_kana = re.search(r"[\u3040-\u30ff\uff66-\uff9f]", value) is not None
    return has_han and not has_japanese_kana


def parse_search_results(html: str) -> list[SearchResult]:
    parser = SteamSearchParser()
    parser.feed(html)
    parser.close()
    return parser.results


def fetch_json(
    base_url: str,
    params: dict[str, Any],
    *,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    url = f"{base_url}?{urlencode(params)}"
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise SyncError(f"Expected a JSON object from {base_url}")
            return payload
        except HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504}:
                break
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
        except (
            URLError,
            TimeoutError,
            OSError,
            HTTPException,
            json.JSONDecodeError,
            SyncError,
        ) as error:
            last_error = error
            delay = 2**attempt

        if attempt < retries:
            time.sleep(delay + random.random() * 0.25)

    raise SyncError(f"Failed to fetch {base_url}: {last_error}") from last_error


def fetch_store_search(
    *,
    mode: str,
    wanted: int,
    excluded_appids: set[int],
    excluded_names: set[str],
    country: str,
    language: str,
    timeout: float,
    retries: int,
    prefer_chinese_names: bool = False,
) -> list[SearchResult]:
    """Fetch unique games from Steam Search in result order."""

    if mode not in {"latest", "top_sellers"}:
        raise ValueError(f"Unsupported Steam search mode: {mode}")

    preferred_results: list[SearchResult] = []
    fallback_results: list[SearchResult] = []
    seen = set(excluded_appids)
    seen_names = {name.casefold() for name in excluded_names}
    start = 0
    total_count: int | None = None

    while len(preferred_results) < wanted and (
        total_count is None or start < total_count
    ):
        params: dict[str, Any] = {
            "query": "",
            "start": start,
            "count": PAGE_SIZE,
            "dynamic_data": "",
            # The Store exposes Games, DLC, Demos, Software, and other item
            # types as separate search categories. Restrict every search-based
            # cohort to base game app rows.
            "category1": STEAM_GAMES_CATEGORY_ID,
            "infinite": 1,
            "cc": country,
            "l": language,
        }
        if mode == "latest":
            params["sort_by"] = "Released_DESC"
            if prefer_chinese_names:
                # Reduce the candidate pool to games that declare Simplified
                # Chinese support, then separately require a Chinese store name.
                params["supportedlang"] = "schinese"
        else:
            params["filter"] = "topsellers"

        payload = fetch_json(
            STORE_SEARCH_URL,
            params,
            timeout=timeout,
            retries=retries,
        )
        if payload.get("success") != 1:
            raise SyncError(f"Steam Store search returned an error for mode={mode}")

        try:
            total_count = int(payload["total_count"])
            html = str(payload["results_html"])
        except (KeyError, TypeError, ValueError) as error:
            raise SyncError("Steam Store search response has an unexpected shape") from error

        page = parse_search_results(html)
        if not page:
            raise SyncError(
                f"Steam Store search returned no parseable rows at offset {start}"
            )

        for game in page:
            if game.appid in seen:
                continue
            seen.add(game.appid)
            name_key = game.canonical_name.casefold()
            if name_key in seen_names:
                continue
            if mode == "latest" and not game.release_date:
                # Do not let undated/coming-soon entries enter the released cohort.
                continue
            seen_names.add(name_key)
            if prefer_chinese_names and not is_chinese_name(game.canonical_name):
                fallback_results.append(game)
            else:
                preferred_results.append(game)
            if len(preferred_results) == wanted:
                break

        start += PAGE_SIZE

    results = preferred_results[:wanted]
    if len(results) < wanted:
        results.extend(fallback_results[: wanted - len(results)])

    if len(results) != wanted:
        raise SyncError(
            f"Steam Store supplied only {len(results)} of {wanted} requested {mode} games"
        )
    return results


def fetch_most_played_games(
    *,
    country: str,
    language: str,
    timeout: float,
    retries: int,
) -> list[Game]:
    """Return actual games from Steam's top-100 most-played chart."""

    input_json = json.dumps(
        {
            "context": {
                "language": language,
                "country_code": country,
            },
            "data_request": {"include_basic_info": True},
        },
        separators=(",", ":"),
    )
    payload = fetch_json(
        MOST_PLAYED_URL,
        {"input_json": input_json},
        timeout=timeout,
        retries=retries,
    )

    try:
        ranks = payload["response"]["ranks"]
    except (KeyError, TypeError) as error:
        raise SyncError("Steam most-played response has an unexpected shape") from error
    if not isinstance(ranks, list):
        raise SyncError("Steam most-played ranks must be a list")

    games: list[Game] = []
    seen: set[int] = set()
    seen_names: set[str] = set()
    for row in ranks:
        if not isinstance(row, dict):
            continue
        item = row.get("item")
        # Steam app type 0 is a game. The chart can also contain software,
        # tools, mods, and playtests, which are not valid PoC entities here.
        if not isinstance(item, dict) or item.get("type") != 0:
            continue
        try:
            appid = int(row["appid"])
            rank = int(row["rank"])
        except (KeyError, TypeError, ValueError):
            continue
        name = clean_canonical_name(str(item.get("name", "")))
        name_key = name.casefold()
        if not name or appid in seen or name_key in seen_names:
            continue
        seen.add(appid)
        seen_names.add(name_key)
        games.append(
            Game(
                appid=appid,
                canonical_name=name,
                cohort="popular",
                source="steam_most_played",
                source_rank=rank,
            )
        )
    return games


def fetch_popular_games(
    *,
    wanted: int,
    country: str,
    language: str,
    timeout: float,
    retries: int,
) -> tuple[list[Game], list[str]]:
    """Build the popular cohort, filling chart gaps with top sellers."""

    warnings: list[str] = []
    try:
        popular = fetch_most_played_games(
            country=country,
            language=language,
            timeout=timeout,
            retries=retries,
        )[:wanted]
    except SyncError as error:
        popular = []
        warnings.append(
            f"Most-played chart unavailable; used top sellers only: {error}"
        )

    if len(popular) < wanted:
        fill_count = wanted - len(popular)
        top_sellers = fetch_store_search(
            mode="top_sellers",
            wanted=fill_count,
            excluded_appids={game.appid for game in popular},
            excluded_names={game.canonical_name for game in popular},
            country=country,
            language=language,
            timeout=timeout,
            retries=retries,
        )
        popular.extend(
            Game(
                appid=game.appid,
                canonical_name=game.canonical_name,
                cohort="popular",
                source="steam_top_sellers_fill",
                source_rank=index,
                release_date=game.release_date,
            )
            for index, game in enumerate(top_sellers, start=1)
        )

    if len(popular) != wanted:
        raise SyncError(f"Built {len(popular)} of {wanted} requested popular games")
    return popular, warnings


def build_dataset(
    *,
    popular_count: int,
    latest_count: int,
    country: str,
    language: str,
    timeout: float,
    retries: int,
) -> tuple[list[Game], list[str]]:
    popular, warnings = fetch_popular_games(
        wanted=popular_count,
        country=country,
        language=language,
        timeout=timeout,
        retries=retries,
    )
    latest_rows = fetch_store_search(
        mode="latest",
        wanted=latest_count,
        excluded_appids={game.appid for game in popular},
        excluded_names={game.canonical_name for game in popular},
        country=country,
        language=language,
        timeout=timeout,
        retries=retries,
        prefer_chinese_names=(language == "schinese"),
    )
    latest = [
        Game(
            appid=game.appid,
            canonical_name=game.canonical_name,
            cohort="latest",
            source="steam_latest_releases",
            source_rank=index,
            release_date=game.release_date,
        )
        for index, game in enumerate(latest_rows, start=1)
    ]

    dataset = popular + latest
    appids = [game.appid for game in dataset]
    if len(appids) != len(set(appids)):
        raise SyncError("Internal error: duplicate AppIDs remain in the dataset")
    normalized_names = [game.canonical_name.casefold() for game in dataset]
    if len(normalized_names) != len(set(normalized_names)):
        raise SyncError("Internal error: duplicate canonical names remain in the dataset")
    return dataset, warnings


def atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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


def default_sidecar_path(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}.{suffix}")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "common/data/steam_games.csv",
        help="training CSV path (default: common/data/steam_games.csv)",
    )
    parser.add_argument("--latest-count", type=positive_int, default=900)
    parser.add_argument("--popular-count", type=positive_int, default=100)
    parser.add_argument("--country", default="US")
    parser.add_argument(
        "--language",
        default="schinese",
        help=(
            "Steam Store language (default: schinese; Steam falls back to the "
            "original name when no Simplified Chinese name exists)"
        ),
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout <= 0:
        raise SystemExit("--timeout must be greater than zero")
    if args.retries < 0:
        raise SystemExit("--retries cannot be negative")

    output = args.output
    provenance_path = default_sidecar_path(output, "provenance.csv")
    metadata_path = default_sidecar_path(output, "metadata.json")

    print(
        f"Synchronizing {args.popular_count} popular and "
        f"{args.latest_count} latest Steam games...",
        file=sys.stderr,
    )
    dataset, warnings = build_dataset(
        popular_count=args.popular_count,
        latest_count=args.latest_count,
        country=args.country,
        language=args.language,
        timeout=args.timeout,
        retries=args.retries,
    )

    atomic_write_csv(
        output,
        ("canonical_name", "appid"),
        (
            {"canonical_name": game.canonical_name, "appid": game.appid}
            for game in dataset
        ),
    )
    atomic_write_csv(
        provenance_path,
        (
            "canonical_name",
            "appid",
            "cohort",
            "source",
            "source_rank",
            "release_date",
            "item_type",
        ),
        (asdict(game) for game in dataset),
    )

    source_counts: dict[str, int] = {}
    for game in dataset:
        source_counts[game.source] = source_counts.get(game.source, 0) + 1
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    metadata = {
        "schema_version": 1,
        "generated_at": generated_at,
        "locale": {"country": args.country, "language": args.language},
        "counts": {
            "total": len(dataset),
            "unique_appids": len({game.appid for game in dataset}),
            "popular": sum(game.cohort == "popular" for game in dataset),
            "latest": sum(game.cohort == "latest" for game in dataset),
            "by_source": source_counts,
        },
        "selection": {
            "popular": (
                "Steam's current most-played chart, game apps only; "
                "filled to the requested count from the Steam game top-sellers list"
            ),
            "latest": (
                "Steam Store Games category with Simplified Chinese support, sorted "
                "by release date descending; names that contain Chinese are selected "
                "first, and AppIDs already in the popular cohort are excluded"
            ),
            "item_type": (
                "game only: Store searches use the Games category, chart items must "
                "have Steam type=0, and bundle rows are ignored"
            ),
            "canonical_name": (
                f"Steam Store display name for language={args.language} at "
                "synchronization time; Steam falls back to the original name when "
                "that localization is unavailable; Chinese book-title marks and "
                "subtitles separated by ' - ' are removed; bilingual slash/hyphen "
                "names prefer the Chinese segment; colon suffixes are retained"
            ),
        },
        "files": {
            "dataset": output.name,
            "provenance": provenance_path.name,
        },
        "sources": [MOST_PLAYED_URL, STORE_SEARCH_URL],
        "warnings": warnings,
    }
    atomic_write_json(metadata_path, metadata)

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print(f"Wrote {len(dataset)} unique games to {output}", file=sys.stderr)
    print(f"Wrote provenance to {provenance_path}", file=sys.stderr)
    print(f"Wrote metadata to {metadata_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except SyncError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
