"""One command: score plans against saved answers and print EXEC_ACC."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from raggeoni.harness import ItemScore, parse_plan_text, score_item
from raggeoni.hydrate import load_jsonl
from raggeoni.ingest import repo_root


def predictions_from_gold(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item["gold_plan"] for item in items}


def load_predictions(path: Path) -> dict[str, dict[str, Any] | None]:
    out: dict[str, dict[str, Any] | None] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(item, dict) or "id" not in item:
            raise SystemExit(f"{path}:{line_no}: need an object with id")
        out[item["id"]] = item.get("plan") or item.get("predicted_plan")
    return out


def filter_split(items: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    if split == "all":
        return items
    return [item for item in items if item.get("split", "eval") == split]


def score_items(
    items: list[dict[str, Any]],
    plans: dict[str, dict[str, Any] | None],
) -> list[ItemScore]:
    scores: list[ItemScore] = []
    for item in items:
        ident = item["id"]
        if ident not in plans:
            scores.append(
                score_item(item, None, parse_error=f"no prediction for {ident}")
            )
            continue
        scores.append(score_item(item, plans[ident]))
    return scores


def score_arm1(items: list[dict[str, Any]], *, model: str | None) -> list[ItemScore]:
    from raggeoni.baseline.client import complete_plan

    scores: list[ItemScore] = []
    total = len(items)
    for i, item in enumerate(items, start=1):
        ident = item["id"]
        print(f"  [{i}/{total}] {ident}", flush=True)
        try:
            text = complete_plan(item["question"], model=model)
            plan = parse_plan_text(text)
            scores.append(score_item(item, plan))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc
        except ValueError as exc:
            scores.append(score_item(item, None, parse_error=str(exc)))
    return scores


def print_report(scores: list[ItemScore]) -> float:
    n = len(scores)
    hits = sum(1 for s in scores if s.ok)
    acc = hits / n if n else 0.0
    print(f"EXEC_ACC  {acc:.3f}   ({hits}/{n})")
    by_bucket: dict[str, list[ItemScore]] = {}
    for score in scores:
        by_bucket.setdefault(score.bucket, []).append(score)
    for bucket in ("structural", "factual", "out-of-scope"):
        group = by_bucket.get(bucket, [])
        if not group:
            continue
        ok = sum(1 for s in group if s.ok)
        print(f"  {bucket:14s}  {ok / len(group):.3f}  ({ok}/{len(group)})")
    fails = Counter(s.error_class for s in scores if not s.ok)
    if fails:
        print("errors:")
        for kind, count in sorted(fails.items()):
            print(f"  {kind}: {count}")
    return acc


def write_run(path: Path, scores: list[ItemScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for score in scores:
        lines.append(
            json.dumps(
                {
                    "id": score.id,
                    "bucket": score.bucket,
                    "split": score.split,
                    "ok": score.ok,
                    "error_class": score.error_class,
                    "predicted_plan": score.predicted_plan,
                    "predicted_result": score.predicted_result,
                },
                ensure_ascii=True,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score predicted plans against saved answers. Prints EXEC_ACC."
    )
    parser.add_argument(
        "--eval",
        dest="eval_path",
        type=Path,
        default=None,
        help="Eval JSONL (default: eval/v1.jsonl)",
    )
    parser.add_argument(
        "--split",
        choices=("eval", "holdout", "all"),
        default="all",
        help="Which split to score (default: all)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Score the saved gold plans (must be 1.000 if the harness works)",
    )
    parser.add_argument(
        "--arm",
        type=int,
        choices=(1,),
        help="Arm 1: call the base model with the baseline prompt",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="JSONL of {id, plan} to score instead of calling a model",
    )
    parser.add_argument("--model", help="Override RAGGEONI_MODEL / gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=0, help="Score only the first N items")
    parser.add_argument("--out", type=Path, help="Write per-item scores JSONL")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    eval_path = args.eval_path or (repo_root() / "eval" / "v1.jsonl")
    if not eval_path.exists():
        print(f"missing {eval_path}", file=sys.stderr)
        return 1
    items = filter_split(load_jsonl(eval_path), args.split)
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("no items to score", file=sys.stderr)
        return 1

    modes = [bool(args.self_test), bool(args.arm), bool(args.predictions)]
    if sum(modes) == 0:
        args.self_test = True
        modes = [True, False, False]
    if sum(modes) != 1:
        print("pick one of --self-test, --arm 1, or --predictions", file=sys.stderr)
        return 2

    if args.self_test:
        scores = score_items(items, predictions_from_gold(items))
    elif args.predictions:
        if not args.predictions.exists():
            print(f"missing {args.predictions}", file=sys.stderr)
            return 1
        scores = score_items(items, load_predictions(args.predictions))
    else:
        if args.split == "all":
            print(
                "Refusing --arm 1 on split=all. Use --split eval (dev) or --split holdout (once).",
                file=sys.stderr,
            )
            return 2
        scores = score_arm1(items, model=args.model)

    acc = print_report(scores)
    if args.out:
        write_run(args.out, scores)
        print(f"wrote {args.out}")
    return 0 if (args.arm or args.predictions or acc == 1.0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
