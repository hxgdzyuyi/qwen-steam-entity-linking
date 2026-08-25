from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "common"
    / "scripts"
    / "sync_steam_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("sync_steam_dataset", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
sync = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sync
SPEC.loader.exec_module(sync)


class SteamSearchParserTest(unittest.TestCase):
    def test_extracts_appid_name_and_release_date(self) -> None:
        html = """
        <a class="search_result_row ds_collapse_flag" data-ds-appid="730">
          <div><img src="capsule.jpg"><span class="title">Counter-Strike &amp; Friends</span></div>
          <div class="search_released responsive_secondrow">
            Aug 21, 2012
          </div>
        </a>
        """

        self.assertEqual(
            sync.parse_search_results(html),
            [
                sync.SearchResult(
                    appid=730,
                    canonical_name="Counter-Strike & Friends",
                    release_date="Aug 21, 2012",
                )
            ],
        )

    def test_ignores_bundles_and_malformed_appids(self) -> None:
        html = """
        <a class="search_result_row" data-ds-bundleid="1">
          <span class="title">Bundle</span>
        </a>
        <a class="search_result_row" data-ds-appid="1,2">
          <span class="title">Several apps</span>
        </a>
        """

        self.assertEqual(sync.parse_search_results(html), [])

    def test_clean_text_normalizes_store_whitespace(self) -> None:
        self.assertEqual(sync.clean_text("  A\n  Game\tName  "), "A Game Name")

    def test_canonical_name_removes_chinese_book_title_marks(self) -> None:
        self.assertEqual(
            sync.clean_canonical_name("《黑神话：悟空》 〈豪华版〉"),
            "黑神话：悟空 豪华版",
        )

    def test_canonical_name_removes_hyphen_separated_subtitle(self) -> None:
        self.assertEqual(
            sync.clean_canonical_name("《主标题》 - The Subtitle"),
            "主标题",
        )

    def test_canonical_name_prefers_chinese_bilingual_segment(self) -> None:
        self.assertEqual(
            sync.clean_canonical_name("Palworld / 幻兽帕鲁"),
            "幻兽帕鲁",
        )
        self.assertEqual(
            sync.clean_canonical_name("Yet Another Zombie Survivors - 又一个僵尸幸存者"),
            "又一个僵尸幸存者",
        )

    def test_canonical_name_keeps_colons_and_word_hyphens(self) -> None:
        self.assertEqual(
            sync.clean_canonical_name("Counter-Strike 2: Legacy"),
            "Counter-Strike 2: Legacy",
        )

    def test_chinese_name_check_rejects_japanese_kana(self) -> None:
        self.assertTrue(sync.is_chinese_name("黑神话：悟空"))
        self.assertFalse(sync.is_chinese_name("敵国工場への潜入"))

    def test_simplified_chinese_is_the_default_language(self) -> None:
        self.assertEqual(sync.parse_args([]).language, "schinese")


if __name__ == "__main__":
    unittest.main()
