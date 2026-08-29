"""Compile or execute a single query plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from raggeoni.compile import compile_plan
from raggeoni.execute import execute_plan
from raggeoni.validate import InvalidPlan


def load_plan(raw: str) -> dict[str, Any]:
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON: {exc}") from exc
    if not isinstance(plan, dict):
        raise SystemExit("plan must be a JSON object")
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile or execute one Raggeoni query plan"
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="Read plan JSON from a file (otherwise stdin or --plan)",
    )
    parser.add_argument("--plan", help="Plan JSON as a string")
    parser.add_argument(
        "--compile",
        action="store_true",
        dest="compile_only",
        help="Print SQL and params; do not hit DuckDB",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the plan against data/nem.duckdb (default if neither flag)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.file and args.plan:
        print("use --file or --plan, not both", file=sys.stderr)
        return 2
    if args.file:
        raw = args.file.read_text(encoding="utf-8")
    elif args.plan:
        raw = args.plan
    else:
        raw = sys.stdin.read()
    plan = load_plan(raw)
    compile_only = args.compile_only and not args.execute
    if compile_only:
        try:
            compiled = compile_plan(plan)
        except InvalidPlan as exc:
            print(json.dumps({"status": "error", "error_class": "invalid_plan", "error": str(exc)}))
            return 1
        print(
            json.dumps(
                {
                    "intent": compiled.intent,
                    "sql": compiled.sql,
                    "params": list(compiled.params),
                    "columns": list(compiled.columns),
                },
                indent=2,
            )
        )
        return 0
    result = execute_plan(plan)
    payload: dict[str, Any]
    if result.status == "ok":
        payload = result.to_gold()
        payload["sql"] = result.sql
        payload["params"] = list(result.params)
    elif result.status == "refuse":
        payload = result.to_gold()
    else:
        payload = {
            "status": result.status,
            "error_class": result.error_class,
            "error": result.error,
        }
        if result.sql:
            payload["sql"] = result.sql
            payload["params"] = list(result.params)
    print(json.dumps(payload, indent=2))
    return 0 if result.status in {"ok", "refuse"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
