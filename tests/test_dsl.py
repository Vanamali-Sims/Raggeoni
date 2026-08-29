"""DSL validate/compile tests. Execute tests skip if DuckDB is missing."""

from __future__ import annotations

import unittest

from raggeoni.compile import compile_plan
from raggeoni.execute import execute_plan
from raggeoni.ingest import db_path
from raggeoni.validate import InvalidPlan, validate_plan


class ValidateTests(unittest.TestCase):
    def test_refuse(self) -> None:
        plan = validate_plan({"intent": "refuse", "reason": "WEM"})
        self.assertEqual(plan["intent"], "refuse")

    def test_rejects_sql_in_column(self) -> None:
        with self.assertRaises(InvalidPlan):
            validate_plan(
                {
                    "intent": "lookup",
                    "filters": [
                        {"col": "region; DROP TABLE price_demand", "op": "=", "val": "VIC1"}
                    ],
                    "limit": 1,
                }
            )

    def test_rejects_unknown_fn(self) -> None:
        with self.assertRaises(InvalidPlan):
            validate_plan(
                {
                    "intent": "aggregate",
                    "select": [{"fn": "median", "col": "rrp"}],
                }
            )

    def test_rejects_bad_region(self) -> None:
        with self.assertRaises(InvalidPlan):
            validate_plan(
                {
                    "intent": "lookup",
                    "filters": [{"col": "region", "op": "=", "val": "NSW"}],
                    "limit": 1,
                }
            )

    def test_rejects_lookup_without_limit(self) -> None:
        with self.assertRaises(InvalidPlan):
            validate_plan(
                {
                    "intent": "lookup",
                    "filters": [{"col": "region", "op": "=", "val": "VIC1"}],
                }
            )

    def test_rejects_group_by_without_limit(self) -> None:
        with self.assertRaises(InvalidPlan):
            validate_plan(
                {
                    "intent": "aggregate",
                    "select": [{"fn": "max", "col": "rrp"}],
                    "group_by": ["region"],
                }
            )

    def test_rejects_reserved_alias(self) -> None:
        with self.assertRaises(InvalidPlan):
            validate_plan(
                {
                    "intent": "aggregate",
                    "select": [{"fn": "max", "col": "rrp", "as": "select"}],
                }
            )

    def test_rejects_extra_fields(self) -> None:
        with self.assertRaises(InvalidPlan):
            validate_plan({"intent": "refuse", "sql": "SELECT 1"})


class CompileTests(unittest.TestCase):
    def test_refuse_has_no_sql(self) -> None:
        compiled = compile_plan({"intent": "refuse"})
        self.assertIsNone(compiled.sql)
        self.assertEqual(compiled.intent, "refuse")

    def test_lookup_binds_values(self) -> None:
        compiled = compile_plan(
            {
                "intent": "lookup",
                "filters": [
                    {"col": "region", "op": "=", "val": "VIC1"},
                    {"col": "settlement_ts", "op": "=", "val": "2026-01-01 00:05:00"},
                ],
                "columns": ["rrp"],
                "limit": 1,
            }
        )
        assert compiled.sql is not None
        self.assertNotIn("VIC1", compiled.sql)
        self.assertNotIn("2026-01-01", compiled.sql)
        self.assertIn("?", compiled.sql)
        self.assertEqual(compiled.params[0], "VIC1")
        self.assertEqual(compiled.params[-1], 1)

    def test_count_star(self) -> None:
        compiled = compile_plan(
            {
                "intent": "aggregate",
                "filters": [{"col": "region", "op": "=", "val": "TAS1"}],
                "select": [{"fn": "count"}],
            }
        )
        assert compiled.sql is not None
        self.assertIn("count(*)", compiled.sql)
        self.assertEqual(compiled.columns, ("n",))


@unittest.skipUnless(db_path().exists(), "data/nem.duckdb not built")
class ExecuteTests(unittest.TestCase):
    def test_vic_point_lookup(self) -> None:
        result = execute_plan(
            {
                "intent": "lookup",
                "filters": [
                    {"col": "region", "op": "=", "val": "VIC1"},
                    {"col": "settlement_ts", "op": "=", "val": "2026-01-01 00:05:00"},
                ],
                "columns": ["rrp", "total_demand"],
                "limit": 1,
            }
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.rows, ((58.51, 4188.23),))

    def test_refuse_does_not_query(self) -> None:
        result = execute_plan({"intent": "refuse", "reason": "unit output"})
        self.assertEqual(result.status, "refuse")
        self.assertEqual(result.to_gold(), {"status": "refuse"})


if __name__ == "__main__":
    unittest.main()
