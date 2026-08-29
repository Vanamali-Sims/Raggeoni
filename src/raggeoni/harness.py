"""Score a predicted plan against the saved answer for one question."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from raggeoni.execute import execute_plan, results_equal


@dataclass(frozen=True)
class ItemScore:
    id: str
    bucket: str
    split: str
    ok: bool
    error_class: str | None
    predicted_plan: dict[str, Any] | None
    predicted_result: dict[str, Any] | None


def parse_plan_text(text: str) -> dict[str, Any]:
    """Pull a JSON object out of model output. Raises ValueError if none."""
    blob = _extract_json_object(text)
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError as exc:
        raise ValueError(f"not JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("plan must be a JSON object")
    return parsed


def score_item(
    item: dict[str, Any],
    predicted_plan: dict[str, Any] | None,
    *,
    parse_error: str | None = None,
) -> ItemScore:
    ident = item["id"]
    bucket = item["bucket"]
    split = item.get("split", "eval")
    gold = item.get("gold_result")
    if gold is None:
        raise ValueError(f"{ident}: missing gold_result; run hydrate first")

    if parse_error is not None or predicted_plan is None:
        return ItemScore(
            id=ident,
            bucket=bucket,
            split=split,
            ok=False,
            error_class="parse",
            predicted_plan=None,
            predicted_result=None,
        )

    executed = execute_plan(predicted_plan)
    if executed.status == "error":
        return ItemScore(
            id=ident,
            bucket=bucket,
            split=split,
            ok=False,
            error_class=executed.error_class or "sql",
            predicted_plan=predicted_plan,
            predicted_result=None,
        )

    predicted_result = executed.to_gold()
    if results_equal(predicted_result, gold):
        return ItemScore(
            id=ident,
            bucket=bucket,
            split=split,
            ok=True,
            error_class=None,
            predicted_plan=predicted_plan,
            predicted_result=predicted_result,
        )

    gold_refuse = gold.get("status") == "refuse"
    pred_refuse = predicted_result.get("status") == "refuse"
    if gold_refuse and not pred_refuse:
        kind = "false_answer"
    elif pred_refuse and not gold_refuse:
        kind = "false_refuse"
    else:
        kind = "mismatch"
    return ItemScore(
        id=ident,
        bucket=bucket,
        split=split,
        ok=False,
        error_class=kind,
        predicted_plan=predicted_plan,
        predicted_result=predicted_result,
    )


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object in model output")
    return stripped[start : end + 1]
