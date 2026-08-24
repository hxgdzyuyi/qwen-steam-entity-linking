from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_training_data.py"
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
        self.assertEqual(set(first[0]), {"prompt", "completion"})

    def test_prompt_styles_include_appid_spellings(self) -> None:
        rendered = [template for _, template in build.PROMPT_STYLES]
        self.assertTrue(any("Steam实体Id" in template for template in rendered))
        self.assertFalse(any("Steam实体：" in template for template in rendered))
        self.assertTrue(any("Steam 的 AppID" in template for template in rendered))
        self.assertTrue(any("Steam 的 AppId" in template for template in rendered))

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


if __name__ == "__main__":
    unittest.main()
