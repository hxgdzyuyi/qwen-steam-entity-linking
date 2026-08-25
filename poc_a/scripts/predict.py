#!/usr/bin/env python3
"""Run strict PoC A2 inference and return an AppID or null."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from build_training_data import PROMPT_STYLES
from structured_output import (
    ENTITY_TOKEN_PATTERN,
    NO_MATCH_TOKEN,
    canonical_index,
    resolve_structured_response,
)
from training_common import (
    TrainingToolError,
    validate_entity_tokens,
    validate_no_match_token,
)


POC_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        required=True,
        help="Local checkpoint directory or Hugging Face adapter repository",
    )
    parser.add_argument(
        "--query", action="append", required=True, help="Input name or description"
    )
    parser.add_argument(
        "--prompt-style",
        choices=[name for name, _ in PROMPT_STYLES],
        default="appid_question",
    )
    parser.add_argument(
        "--entities",
        type=Path,
        default=POC_ROOT.parent / "common/data/steam_games.csv",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args(argv)


def load_canonical_entities(path: Path) -> dict[str, str]:
    canonical_by_token: dict[str, str] = {}
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["canonical_name", "appid"]:
                raise TrainingToolError(
                    f"{path} must contain exactly canonical_name,appid columns"
                )
            for row in reader:
                canonical_name = str(row["canonical_name"])
                if not canonical_name.strip():
                    raise TrainingToolError(f"Empty canonical name in {path}")
                appid = str(row["appid"])
                if not appid.isdigit():
                    raise TrainingToolError(f"Invalid AppID in {path}: {appid}")
                entity_token = f"<GAME_{int(appid)}>"
                if entity_token in canonical_by_token:
                    raise TrainingToolError(f"Duplicate AppID in {path}: {appid}")
                canonical_by_token[entity_token] = canonical_name
    except OSError as error:
        raise TrainingToolError(f"Cannot read entity table {path}: {error}") from error
    if not canonical_by_token:
        raise TrainingToolError(f"Entity table is empty: {path}")
    return canonical_by_token


def _cloud_imports() -> dict[str, Any]:
    try:
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise TrainingToolError(
            "Prediction dependencies are missing; install requirements-cloud.txt"
        ) from error
    return {
        "torch": torch,
        "PeftConfig": PeftConfig,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_new_tokens <= 0:
        raise TrainingToolError("--max-new-tokens must be positive")
    imports = _cloud_imports()
    torch = imports["torch"]
    if not torch.cuda.is_available():
        raise TrainingToolError("Prediction requires a CUDA GPU")

    token = os.environ.get("HF_TOKEN")
    adapter_config = imports["PeftConfig"].from_pretrained(args.adapter, token=token)
    base_id = str(adapter_config.base_model_name_or_path)
    revision = getattr(adapter_config, "revision", None)
    tokenizer = imports["AutoTokenizer"].from_pretrained(
        args.adapter, token=token, use_fast=True
    )
    canonical_by_token = load_canonical_entities(args.entities.resolve())
    canonical_name_index = canonical_index(canonical_by_token)
    added_output_tokens = {
        value
        for value in tokenizer.get_added_vocab()
        if ENTITY_TOKEN_PATTERN.fullmatch(value)
    }
    if added_output_tokens != set(canonical_by_token):
        raise TrainingToolError(
            "Adapter GAME tokens do not match the frozen entity table"
        )
    if NO_MATCH_TOKEN not in tokenizer.get_added_vocab():
        raise TrainingToolError("Adapter tokenizer is missing <NO_MATCH>")
    validate_entity_tokens(tokenizer, list(canonical_by_token))
    validate_no_match_token(tokenizer, NO_MATCH_TOKEN)
    base = imports["AutoModelForCausalLM"].from_pretrained(
        base_id,
        revision=revision,
        token=token,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    base.resize_token_embeddings(len(tokenizer))
    model = imports["PeftModel"].from_pretrained(base, args.adapter, token=token)
    model.to(torch.device("cuda"))
    model.eval()

    template = dict(PROMPT_STYLES)[args.prompt_style]
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        prompts = [template.format(input_text=query) for query in args.query]
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        prompt_width = int(encoded["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=int(tokenizer.eos_token_id),
                pad_token_id=int(tokenizer.pad_token_id),
            )
        for query, sequence in zip(args.query, generated.tolist()):
            output_ids: list[int] = []
            for token_id in sequence[prompt_width:]:
                if int(token_id) in {
                    int(tokenizer.eos_token_id),
                    int(tokenizer.pad_token_id),
                }:
                    break
                output_ids.append(int(token_id))
            generated_text = tokenizer.decode(
                output_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ).strip()
            resolution = resolve_structured_response(
                generated_text,
                canonical_by_token,
                canonical_name_index=canonical_name_index,
            )
            appid = None
            if resolution.resolved_token:
                appid = int(resolution.resolved_token.removeprefix("<GAME_").removesuffix(">"))
            print(
                json.dumps(
                    {
                        "input": query,
                        "generated_text": generated_text,
                        "status": resolution.status,
                        "canonical_name": resolution.resolved_canonical_name,
                        "appid": appid,
                    },
                    ensure_ascii=False,
                )
            )
    finally:
        tokenizer.padding_side = previous_padding_side
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TrainingToolError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
