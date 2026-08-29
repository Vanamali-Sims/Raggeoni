"""Fill gold_result by executing gold_plan. Humans do not type the numbers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from raggeoni.execute import execute_plan, results_equal


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(item, dict):
            raise SystemExit(f"{path}:{line_no}: each line must be an object")
        for key in ("id", "bucket", "question", "gold_plan"):
            if key not in item:
                raise SystemExit(f"{path}:{line_no}: missing {key}")
        rows.append(item)
    return rows


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(item, ensure_ascii=True) for item in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute gold_plan for each eval row and write gold_result"
    )
    parser.add_argument("jsonl", type=Path, help="Eval JSONL path")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-execute and compare to existing gold_result; do not write",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute and print results; do not write the file",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = args.jsonl
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        return 1
    rows = load_jsonl(path)
    errors: list[str] = []
    mismatches: list[str] = []
    computed: list[dict[str, Any]] = []

    for item in rows:
        ident = item["id"]
        result = execute_plan(item["gold_plan"])
        if result.status not in {"ok", "refuse"}:
            errors.append(
                f"{ident}: {result.error_class}: {result.error}"
            )
            continue
        gold = result.to_gold()
        computed.append(gold)
        if args.verify:
            existing = item.get("gold_result")
            if existing is None:
                mismatches.append(f"{ident}: gold_result is missing")
            elif not results_equal(existing, gold):
                mismatches.append(f"{ident}: gold_result does not match re-execution")
        print(f"  [{result.status}] {ident}")

    if errors:
        print("Hydrate failed:", file=sys.stderr)
        for line in errors:
            print(f"  - {line}", file=sys.stderr)
        return 1
    if args.verify:
        if mismatches:
            print("Verify failed:", file=sys.stderr)
            for line in mismatches:
                print(f"  - {line}", file=sys.stderr)
            return 1
        print(f"Verified {len(rows)} gold results against {path}")
        return 0
    if args.dry_run:
        print(f"Dry run OK ({len(rows)} rows); file not written")
        return 0

    for item, gold in zip(rows, computed, strict=True):
        item["gold_result"] = gold
    dump_jsonl(path, rows)
    print(f"Wrote gold_result for {len(rows)} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
