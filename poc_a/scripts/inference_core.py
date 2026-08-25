#!/usr/bin/env python3
"""Shared deterministic structured inference for PoC A2."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from structured_output import (
    NO_MATCH_TOKEN,
    canonical_index,
    entity_scoring_prefix,
    normalize_entity_name,
    resolve_structured_response,
)
from training_common import TrainingToolError


def _generated_ids_after_prompt(
    sequence: Sequence[int], prompt_width: int, eos_token_id: int, pad_token_id: int
) -> list[int]:
    output: list[int] = []
    for raw_token_id in sequence[prompt_width:]:
        token_id = int(raw_token_id)
        if token_id in {eos_token_id, pad_token_id}:
            break
        output.append(token_id)
    return output


def predict_structured_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, str]],
    *,
    expected_key: str,
    source: str,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    output_token_ids: Sequence[int],
    canonical_by_token: Mapping[str, str],
    provenance_by_token: Mapping[str, Mapping[str, str]],
    torch: Any,
) -> list[dict[str, Any]]:
    """Generate the canonical-name chain, resolve it, and score its ID hop."""

    if not rows:
        return []
    records: list[dict[str, Any]] = []
    was_training = bool(model.training)
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    model.eval()
    output_ids_tensor = torch.tensor(
        output_token_ids, dtype=torch.long, device=model.device
    )
    canonical_name_index = canonical_index(canonical_by_token)
    try:
        with torch.inference_mode():
            for offset in range(0, len(rows), batch_size):
                batch_rows = rows[offset : offset + batch_size]
                encoded = tokenizer(
                    [row["prompt"] for row in batch_rows],
                    add_special_tokens=False,
                    padding=True,
                    return_tensors="pt",
                )
                prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
                if max(prompt_lengths) > max_length:
                    raise TrainingToolError(
                        f"Evaluation prompt exceeds max_length={max_length}"
                    )
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                prompt_width = int(encoded["input_ids"].shape[1])
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=int(tokenizer.pad_token_id),
                    eos_token_id=int(tokenizer.eos_token_id),
                    return_dict_in_generate=True,
                )

                oracle_prompts = [
                    row["prompt"] + entity_scoring_prefix(row["canonical_name"])
                    for row in batch_rows
                ]
                oracle_encoded = tokenizer(
                    oracle_prompts,
                    add_special_tokens=False,
                    padding=True,
                    return_tensors="pt",
                )
                if int(oracle_encoded["attention_mask"].sum(dim=1).max()) > max_length:
                    raise TrainingToolError(
                        f"Oracle entity-scoring prompt exceeds max_length={max_length}"
                    )
                oracle_encoded = {
                    key: value.to(model.device) for key, value in oracle_encoded.items()
                }
                oracle_logits = model(**oracle_encoded, use_cache=False).logits[:, -1, :]
                candidate_logits = oracle_logits.index_select(-1, output_ids_tensor)
                unrestricted_ids = oracle_logits.argmax(dim=-1)
                constrained_positions = candidate_logits.argmax(dim=-1)
                constrained_ids = output_ids_tensor[constrained_positions]

                expected_token_ids = torch.tensor(
                    [
                        int(tokenizer.convert_tokens_to_ids(row[expected_key]))
                        for row in batch_rows
                    ],
                    dtype=torch.long,
                    device=model.device,
                )
                expected_matches = expected_token_ids.unsqueeze(1).eq(
                    output_ids_tensor.unsqueeze(0)
                )
                if not bool(torch.all(expected_matches.sum(dim=1) == 1).item()):
                    raise TrainingToolError(
                        "An evaluation target is not a configured output token"
                    )
                expected_positions = expected_matches.long().argmax(dim=1)
                expected_scores = candidate_logits.gather(
                    1, expected_positions.unsqueeze(1)
                )
                ranks = candidate_logits.gt(expected_scores).sum(dim=1) + 1

                for row_index, row in enumerate(batch_rows):
                    generated_ids = _generated_ids_after_prompt(
                        generated.sequences[row_index].tolist(),
                        prompt_width,
                        int(tokenizer.eos_token_id),
                        int(tokenizer.pad_token_id),
                    )
                    generated_text = tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ).strip()
                    resolution = resolve_structured_response(
                        generated_text,
                        canonical_by_token,
                        canonical_name_index=canonical_name_index,
                    )
                    expected_token = row[expected_key]
                    expected_name = row["canonical_name"]
                    expected_no_match = expected_token == NO_MATCH_TOKEN
                    resolved_correct = (
                        resolution.resolved_token is None
                        if expected_no_match
                        else resolution.resolved_token == expected_token
                    )
                    canonical_correct = (
                        resolution.format_valid
                        and normalize_entity_name(
                            resolution.generated_canonical_name
                        )
                        == normalize_entity_name(expected_name)
                    )
                    generated_token_correct = (
                        resolution.format_valid
                        and resolution.generated_token == expected_token
                    )
                    expected_id = int(expected_token_ids[row_index].item())
                    unrestricted_id = int(unrestricted_ids[row_index].item())
                    constrained_id = int(constrained_ids[row_index].item())
                    rank = int(ranks[row_index].item())
                    provenance = provenance_by_token.get(expected_token, {})
                    records.append(
                        {
                            "source": source,
                            "input": row["input"],
                            "prompt": row["prompt"],
                            "expected_canonical_name": expected_name,
                            "expected": expected_token,
                            "generated_text": generated_text,
                            "generated_canonical_name": resolution.generated_canonical_name,
                            "generated_token": resolution.generated_token,
                            "resolved_canonical_name": resolution.resolved_canonical_name or "",
                            "resolved_token": resolution.resolved_token or "",
                            "resolution_status": resolution.status,
                            "format_valid": resolution.format_valid,
                            "canonical_name_correct": canonical_correct,
                            "generated_token_correct": generated_token_correct,
                            "structured_exact": canonical_correct
                            and generated_token_correct,
                            "safe_resolution_correct": resolved_correct,
                            "explicit_no_match": resolution.status
                            == "explicit_no_match",
                            "predicted_next_token": str(
                                tokenizer.convert_ids_to_tokens(unrestricted_id)
                            ),
                            "oracle_constrained_token": str(
                                tokenizer.convert_ids_to_tokens(constrained_id)
                            ),
                            "entity_rank": rank,
                            "entity_top5_correct": rank <= 5,
                            "entity_top10_correct": rank <= 10,
                            "next_token_correct": unrestricted_id == expected_id,
                            "generation_correct": resolved_correct,
                            "cohort": provenance.get(
                                "cohort", "unknown" if expected_no_match else "missing"
                            ),
                            "type": row.get(
                                "type", "unknown" if expected_no_match else "canonical"
                            ),
                            "prompt_style": row.get("prompt_style", "custom"),
                            "expected_no_match": expected_no_match,
                        }
                    )
    finally:
        tokenizer.padding_side = previous_padding_side
        if was_training:
            model.train()
    return records
