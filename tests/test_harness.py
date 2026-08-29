"""Harness scoring tests."""

from __future__ import annotations

import unittest

from raggeoni.harness import parse_plan_text, score_item
from raggeoni.hydrate import load_jsonl
from raggeoni.ingest import repo_root


def _v1() -> list[dict]:
    path = repo_root() / "eval" / "v1.jsonl"
    if not path.exists():
        raise unittest.SkipTest("eval/v1.jsonl missing")
    return load_jsonl(path)


class ParseTests(unittest.TestCase):
    def test_bare_object(self) -> None:
        plan = parse_plan_text('{"intent": "refuse"}')
        self.assertEqual(plan["intent"], "refuse")

    def test_fenced(self) -> None:
        plan = parse_plan_text('```json\n{"intent": "refuse"}\n```')
        self.assertEqual(plan["intent"], "refuse")

    def test_garbage(self) -> None:
        with self.assertRaises(ValueError):
            parse_plan_text("I cannot help with that.")


class ScoreTests(unittest.TestCase):
    def test_gold_plan_is_correct(self) -> None:
        rows = [row for row in _v1() if row["id"] == "fact-001"]
        if not rows:
            self.skipTest("fact-001 missing")
        item = rows[0]
        score = score_item(item, item["gold_plan"])
        self.assertTrue(score.ok)
        self.assertIsNone(score.error_class)

    def test_refuse_vs_lookup_is_false_answer(self) -> None:
        rows = [row for row in _v1() if row["id"] == "oos-001"]
        if not rows:
            self.skipTest("oos-001 missing")
        item = rows[0]
        score = score_item(
            item,
            {
                "intent": "lookup",
                "filters": [
                    {"col": "region", "op": "=", "val": "NSW1"},
                    {"col": "settlement_ts", "op": "=", "val": "2026-01-12 12:00:00"},
                ],
                "columns": ["rrp"],
                "limit": 1,
            },
        )
        self.assertFalse(score.ok)
        self.assertEqual(score.error_class, "false_answer")

    def test_parse_error(self) -> None:
        rows = [row for row in _v1() if row["id"] == "fact-001"]
        if not rows:
            self.skipTest("fact-001 missing")
        score = score_item(rows[0], None, parse_error="no JSON")
        self.assertFalse(score.ok)
        self.assertEqual(score.error_class, "parse")


if __name__ == "__main__":
    unittest.main()
