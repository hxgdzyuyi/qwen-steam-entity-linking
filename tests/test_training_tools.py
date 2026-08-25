from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from training_common import (  # noqa: E402
    CompletionOnlyCollator,
    TrainingToolError,
    encode_training_row,
    git_info,
    load_config,
    prepare_tokenizer,
    select_best_checkpoint,
    validate_data,
    warn_if_git_commit_mismatch,
)
from evaluate import generation_is_exact, prediction_sha256  # noqa: E402
from train import (  # noqa: E402
    _model_weights_are_cached,
    _require_resumable_checkpoint,
    _run_directory_has_artifacts,
    _strip_older_resume_states,
    validate_runtime_snapshot,
)

PUBLISH_SPEC = importlib.util.spec_from_file_location(
    "publish_hf", SCRIPTS / "publish_hf.py"
)
assert PUBLISH_SPEC is not None and PUBLISH_SPEC.loader is not None
publish = importlib.util.module_from_spec(PUBLISH_SPEC)
sys.modules[PUBLISH_SPEC.name] = publish
PUBLISH_SPEC.loader.exec_module(publish)


class FakeTokenizer:
    def __init__(self) -> None:
        self.tokens = {"<PAD>": 0, "<EOS>": 1, "A": 2, "B": 3}
        self.eos_token_id = 1
        self.eos_token = "<EOS>"
        self.pad_token_id = 0
        self.pad_token = "<PAD>"

    def __len__(self) -> int:
        return len(self.tokens)

    def add_special_tokens(self, payload: dict[str, list[str]]) -> int:
        added = 0
        for token in payload["additional_special_tokens"]:
            if token not in self.tokens:
                self.tokens[token] = len(self.tokens)
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.tokens[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text in self.tokens:
            return [self.tokens[text]]
        return [self.tokens[character] for character in text]


class TrainingToolsTest(unittest.TestCase):
    def test_repository_dataset_passes_preflight(self) -> None:
        config = load_config(ROOT / "configs" / "qwen3_8b_lora.yaml")
        bundle = validate_data(config)
        self.assertEqual(len(bundle.train_rows), 1000)
        self.assertEqual(len(bundle.special_tokens), 1000)
        self.assertEqual(len(bundle.alias_rows), 184)
        cohorts = [row["cohort"] for row in bundle.provenance_by_token.values()]
        self.assertEqual(cohorts.count("popular"), 100)
        self.assertEqual(cohorts.count("latest"), 900)

    def test_added_entity_tokens_are_atomic_and_new(self) -> None:
        tokenizer = FakeTokenizer()
        token_ids = prepare_tokenizer(tokenizer, ["<GAME_730>", "<GAME_570>"])
        self.assertEqual(token_ids, [4, 5])
        self.assertEqual(tokenizer.encode("<GAME_730>"), [4])

    def test_completion_only_encoding_masks_prompt(self) -> None:
        tokenizer = FakeTokenizer()
        prepare_tokenizer(tokenizer, ["<GAME_730>"])
        encoded = encode_training_row(
            tokenizer,
            {"prompt": "AB", "completion": "<GAME_730>"},
            max_length=8,
        )
        self.assertEqual(encoded["input_ids"], [2, 3, 4, 1])
        self.assertEqual(encoded["attention_mask"], [1, 1, 1, 1])
        self.assertEqual(encoded["labels"], [-100, -100, 4, 1])

    def test_collator_right_pads_and_masks_padding(self) -> None:
        tokenizer = FakeTokenizer()
        prepare_tokenizer(tokenizer, ["<GAME_730>"])
        batch = CompletionOnlyCollator(tokenizer, max_length=8)(
            [
                {"prompt": "A", "completion": "<GAME_730>"},
                {"prompt": "AB", "completion": "<GAME_730>"},
            ]
        )
        self.assertEqual(batch["input_ids"].tolist()[0], [2, 4, 1, 0])
        self.assertEqual(batch["attention_mask"].tolist()[0], [1, 1, 1, 0])
        self.assertEqual(batch["labels"].tolist()[0], [-100, 4, 1, -100])

    def test_encoding_refuses_truncation(self) -> None:
        tokenizer = FakeTokenizer()
        prepare_tokenizer(tokenizer, ["<GAME_730>"])
        with self.assertRaises(TrainingToolError):
            encode_training_row(
                tokenizer,
                {"prompt": "AB", "completion": "<GAME_730>"},
                max_length=3,
            )

    def test_checkpoint_selection_applies_threshold_and_tiebreakers(self) -> None:
        rows = [
            {
                "epoch": 1,
                "canonical": {"generation_accuracy": 0.98},
                "alias": {"generation_accuracy": 0.9},
            },
            {
                "epoch": 3,
                "canonical": {"generation_accuracy": 0.99},
                "alias": {"generation_accuracy": 0.5},
            },
            {
                "epoch": 5,
                "canonical": {"generation_accuracy": 1.0},
                "alias": {"generation_accuracy": 0.5},
            },
        ]
        selected = select_best_checkpoint(rows, 0.99)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["epoch"], 5)

    def test_generation_exact_match_allows_only_eos_and_padding(self) -> None:
        self.assertTrue(generation_is_exact([4, 1, 0], 4, 1, 0))
        self.assertFalse(generation_is_exact([2, 4], 4, 1, 0))
        self.assertFalse(generation_is_exact([4, 3], 4, 1, 0))

    def test_runpod_h100_runtime_snapshot_passes_config(self) -> None:
        config = load_config(ROOT / "configs" / "qwen3_8b_lora.yaml")
        snapshot = {
            "name": "NVIDIA H100 80GB HBM3",
            "gpu_memory_gib": 74.5,
            "system_memory_gib": 116.4,
            "cpu_count": 16,
            "free_disk_gib": 36.0,
            "torch_version": "2.8.0",
            "cuda_version": "12.8",
        }
        validate_runtime_snapshot(snapshot, config)
        snapshot["free_disk_gib"] = 20.0
        with self.assertRaises(TrainingToolError):
            validate_runtime_snapshot(snapshot, config)
        validate_runtime_snapshot(
            snapshot,
            config,
            float(config["runtime"]["minimum_cached_free_disk_gib"]),
        )

    def test_dirty_git_checkout_warns_without_blocking(self) -> None:
        with patch(
            "training_common._git_command",
            side_effect=[" M scripts/train.py", "current-sha", "main", "origin"],
        ):
            with self.assertWarnsRegex(RuntimeWarning, "not clean"):
                info = git_info(require_clean=True)

        self.assertEqual(info["commit"], "current-sha")
        self.assertFalse(info["clean"])

    def test_git_commit_mismatch_warns_without_blocking(self) -> None:
        with self.assertWarnsRegex(RuntimeWarning, "does not match"):
            warn_if_git_commit_mismatch(
                "current-sha",
                "trained-sha",
                operation="Evaluation",
            )

    def test_cached_model_requires_every_indexed_weight_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            index_path = cache / "model.safetensors.index.json"
            first_shard = cache / "model-00001-of-00002.safetensors"
            second_shard = cache / "model-00002-of-00002.safetensors"
            index_path.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "a": first_shard.name,
                            "b": second_shard.name,
                        }
                    }
                ),
                encoding="utf-8",
            )
            first_shard.write_bytes(b"first")
            second_shard.write_bytes(b"second")
            cached = {
                index_path.name: index_path,
                first_shard.name: first_shard,
                second_shard.name: second_shard,
            }

            def fake_cached_file(
                model_id: str, filename: str, **kwargs: object
            ) -> Path | None:
                self.assertEqual(model_id, "org/model")
                self.assertEqual(kwargs["revision"], "abc123")
                self.assertTrue(kwargs["local_files_only"])
                return cached.get(filename)

            self.assertTrue(
                _model_weights_are_cached(fake_cached_file, "org/model", "abc123")
            )
            second_shard.unlink()
            self.assertFalse(
                _model_weights_are_cached(fake_cached_file, "org/model", "abc123")
            )

    def test_run_directory_ignores_empty_directories_but_not_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "run"
            self.assertFalse(_run_directory_has_artifacts(run_dir))
            (run_dir / "checkpoints").mkdir(parents=True)
            self.assertFalse(_run_directory_has_artifacts(run_dir))
            (run_dir / "checkpoints" / "partial.txt").write_text(
                "partial", encoding="utf-8"
            )
            self.assertTrue(_run_directory_has_artifacts(run_dir))
            non_directory = Path(temp) / "run-file"
            non_directory.write_text("artifact", encoding="utf-8")
            self.assertTrue(_run_directory_has_artifacts(non_directory))

    def test_only_latest_checkpoint_keeps_resume_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            checkpoints = Path(temp)
            older = checkpoints / "checkpoint-32"
            latest = checkpoints / "checkpoint-96"
            for epoch, checkpoint in ((1, older), (3, latest)):
                checkpoint.mkdir()
                (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
                (checkpoint / "adapter_model.safetensors").write_text(
                    "adapter", encoding="utf-8"
                )
                (checkpoint / "optimizer.pt").write_text("optimizer", encoding="utf-8")
                (checkpoint / "scheduler.pt").write_text("scheduler", encoding="utf-8")
                (checkpoint / "rng_state.pth").write_text("rng", encoding="utf-8")
                (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
                (checkpoint / "checkpoint_meta.json").write_text(
                    json.dumps({"epoch": epoch, "resumable": True}),
                    encoding="utf-8",
                )

            _strip_older_resume_states(checkpoints, latest)

            self.assertTrue((older / "adapter_model.safetensors").exists())
            self.assertFalse((older / "optimizer.pt").exists())
            self.assertFalse((older / "rng_state.pth").exists())
            self.assertFalse(
                json.loads((older / "checkpoint_meta.json").read_text())["resumable"]
            )
            with self.assertRaises(TrainingToolError):
                _require_resumable_checkpoint(older)
            _require_resumable_checkpoint(latest)

    def test_publish_filter_excludes_optimizer_and_full_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp)
            for name in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "tokenizer.json",
                "optimizer.pt",
                "trainer_state.json",
                "model.safetensors",
            ):
                (checkpoint / name).write_text("test", encoding="utf-8")
            names = [path.name for path in publish.checkpoint_publish_files(checkpoint)]
            self.assertEqual(
                names,
                ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json"],
            )

    def test_publish_tensor_filter_rejects_embedded_base_weights(self) -> None:
        adapter_keys = [
            "base_model.layers.0.q_proj.lora_A.weight",
            "base_model.embed_tokens.token_adapter.trainable_tokens_delta",
            "base_model.lm_head.token_adapter.trainable_tokens_delta",
        ]
        publish._assert_adapter_tensor_keys(adapter_keys)
        with self.assertRaises(TrainingToolError):
            publish._assert_adapter_tensor_keys(
                [*adapter_keys, "base_model.layers.0.q_proj.base_layer.weight"]
            )

    def test_prediction_fingerprint_covers_full_ordered_records(self) -> None:
        records = [
            {"expected": "<GAME_1>", "generated_token": "<GAME_1>"},
            {"expected": "<GAME_2>", "generated_token": "<GAME_3>"},
        ]
        self.assertEqual(prediction_sha256(records), prediction_sha256(records))
        self.assertNotEqual(
            prediction_sha256(records), prediction_sha256(list(reversed(records)))
        )

    def test_publish_regression_compares_breakdowns_and_predictions(self) -> None:
        expected = {
            "epoch": 5,
            "canonical": {"count": 2, "generation_accuracy": 1.0},
            "alias": {"count": 1, "generation_accuracy": 1.0},
            "canonical_by_cohort": {"popular": {"count": 2}},
            "alias_by_type": {"abbreviation": {"count": 1}},
            "alias_by_prompt_style": {"query": {"count": 1}},
            "prediction_sha256": "abc123",
        }
        publish._assert_metrics_match(expected, dict(expected))
        changed = dict(expected)
        changed["canonical_by_cohort"] = {"popular": {"count": 1}}
        with self.assertRaises(TrainingToolError):
            publish._assert_metrics_match(expected, changed)
        changed = dict(expected)
        changed["prediction_sha256"] = "different"
        with self.assertRaises(TrainingToolError):
            publish._assert_metrics_match(expected, changed)

    def test_publish_rejects_stale_remote_files_and_missing_uploads(self) -> None:
        staged = ["README.md", "adapter_config.json", "adapter_model.safetensors"]
        publish._assert_remote_file_set(
            [".gitattributes", "README.md"], staged, require_complete=False
        )
        publish._assert_remote_file_set(
            [".gitattributes", *staged], staged, require_complete=True
        )
        with self.assertRaises(TrainingToolError):
            publish._assert_remote_file_set(
                [".gitattributes", "private-data.json"],
                staged,
                require_complete=False,
            )
        with self.assertRaises(TrainingToolError):
            publish._assert_remote_file_set(
                [".gitattributes", "README.md"],
                staged,
                require_complete=True,
            )

    def test_publish_confirms_private_repository_becomes_public(self) -> None:
        class FakeApi:
            def __init__(self) -> None:
                self.private = True
                self.updates: list[tuple[str, bool]] = []

            def model_info(self, *, repo_id: str) -> object:
                self.asserted_repo_id = repo_id
                return type("Info", (), {"private": self.private})()

            def update_repo_settings(self, *, repo_id: str, private: bool) -> None:
                self.updates.append((repo_id, private))
                self.private = private

        api = FakeApi()
        publish._make_repository_public(api, "org/model")
        self.assertEqual(api.updates, [("org/model", False)])
        self.assertFalse(api.private)

    def test_runpod_notebook_is_safe_and_uses_cli_entrypoints(self) -> None:
        notebook_path = ROOT / "notebooks" / "runpod_training.ipynb"
        payload = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["nbformat"], 4)
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell["cell_type"] == "code"
        )
        for cell_index, cell in enumerate(payload["cells"]):
            if cell["cell_type"] == "code":
                compile(
                    "".join(cell.get("source", [])),
                    f"{notebook_path}:cell-{cell_index}",
                    "exec",
                )
        for entrypoint in (
            "scripts/train.py",
            "scripts/evaluate.py",
            "scripts/publish_hf.py",
        ):
            self.assertIn(entrypoint, code)
        self.assertNotIn("sys.executable, 'scripts/", code)
        self.assertIn("hf_transfer>=0.1.9,<1", code)
        self.assertIn("PUBLISH_PUBLIC = False", code)
        self.assertIsNone(re.search(r"hf_[A-Za-z0-9]{20,}", code))


if __name__ == "__main__":
    unittest.main()
