from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "poc_a"
    / "scripts"
    / "build_training_data.py"
)
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("build_training_data", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
build = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = build
SPEC.loader.exec_module(build)


class BuildTrainingDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.entities = [
            build.Entity(canonical_name="Counter-Strike 2", appid=730),
            build.Entity(canonical_name="幻兽帕鲁", appid=1623730),
        ]

    def test_entity_token_uses_appid(self) -> None:
        self.assertEqual(self.entities[0].token, "<GAME_730>")

    def test_special_tokens_are_sorted_by_numeric_appid(self) -> None:
        reversed_entities = list(reversed(self.entities))
        self.assertEqual(
            build.build_special_tokens(reversed_entities),
            ["<GAME_730>", "<GAME_1623730>"],
        )

    def test_train_shuffle_is_deterministic(self) -> None:
        first = build.build_train_rows(self.entities, seed=42)
        second = build.build_train_rows(self.entities, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(self.entities) * len(build.PROMPT_STYLES))
        self.assertEqual(
            set(first[0]),
            {
                "input",
                "prompt",
                "canonical_name",
                "completion",
                "prompt_style",
                "type",
            },
        )
        for entity in self.entities:
            rows = [row for row in first if row["completion"] == entity.token]
            self.assertEqual(
                {row["prompt_style"] for row in rows},
                {style for style, _ in build.PROMPT_STYLES},
            )

    def test_prompt_styles_explicitly_require_canonicalization_and_rejection(self) -> None:
        rendered = [template for _, template in build.PROMPT_STYLES]
        self.assertTrue(all("AppID" in template for template in rendered))
        self.assertTrue(all("标准" in template for template in rendered))
        self.assertTrue(all("NO_MATCH" in template for template in rendered))
        self.assertTrue(any("Steam 的 AppID" in template for template in rendered))
        self.assertFalse(any("AppId" in template for template in rendered))

    def test_eval_rejects_canonical_name_leakage(self) -> None:
        payload = {
            "schema_version": 1,
            "games": [
                {
                    "appid": 730,
                    "cases": [{"input": "Counter-Strike 2", "type": "name"}],
                }
            ],
        }
        with self.assertRaises(build.BuildError):
            build.build_eval_rows(payload, self.entities)

    def test_eval_crosses_each_alias_with_every_prompt_style(self) -> None:
        payload = {
            "schema_version": 1,
            "games": [
                {
                    "appid": 730,
                    "cases": [{"input": "CS2", "type": "abbreviation"}],
                }
            ],
        }
        rows, game_count = build.build_eval_rows(payload, self.entities)
        self.assertEqual(game_count, 1)
        self.assertEqual(len(rows), len(build.PROMPT_STYLES))
        self.assertEqual(
            {row["prompt_style"] for row in rows},
            {style for style, _ in build.PROMPT_STYLES},
        )
        self.assertTrue(all(row["canonical_name"] == "Counter-Strike 2" for row in rows))

    def test_unknown_train_and_eval_rows_use_no_match(self) -> None:
        train = build.build_train_rows(
            self.entities, seed=42, unknown_inputs=[("不存在的游戏", "synthetic")]
        )
        unknown_train = [row for row in train if row["input"] == "不存在的游戏"]
        self.assertEqual(len(unknown_train), len(build.PROMPT_STYLES))
        self.assertTrue(all(row["canonical_name"] == "NO_MATCH" for row in unknown_train))
        self.assertTrue(all(row["completion"] == "<NO_MATCH>" for row in unknown_train))
        unknown_eval = build.build_unknown_eval_rows(
            [("另一个不存在的游戏", "synthetic")]
        )
        self.assertEqual(len(unknown_eval), len(build.PROMPT_STYLES))
        self.assertTrue(all(row["expected"] == "<NO_MATCH>" for row in unknown_eval))


if __name__ == "__main__":
    unittest.main()
