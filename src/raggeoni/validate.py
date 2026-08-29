"""Validate and normalize a query plan. No SQL here."""

from __future__ import annotations

import re
from typing import Any

from raggeoni.constants import REGIONS

TABLE = "price_demand"
COLUMNS: tuple[str, ...] = (
    "settlement_ts",
    "region",
    "rrp",
    "total_demand",
    "period_type",
)
NUMERIC_COLS = frozenset({"rrp", "total_demand"})
TIMESTAMP_COLS = frozenset({"settlement_ts"})
REGION_COLS = frozenset({"region"})
OPS = frozenset({"=", "!=", ">", ">=", "<", "<=", "in"})
IN_COLS = frozenset({"region", "period_type"})
FNS = frozenset({"max", "min", "avg", "sum", "count"})
INTENTS = frozenset({"lookup", "aggregate", "compare", "refuse"})
MAX_LIMIT = 1000
MAX_FILTERS = 16
ALIAS_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
PERIOD_TYPES = frozenset({"TRADE"})
RESERVED = frozenset(
    {
        "all",
        "and",
        "as",
        "by",
        "cast",
        "count",
        "distinct",
        "else",
        "end",
        "false",
        "from",
        "group",
        "in",
        "inner",
        "is",
        "join",
        "left",
        "limit",
        "not",
        "null",
        "on",
        "or",
        "order",
        "outer",
        "price_demand",
        "right",
        "select",
        "side",
        "then",
        "true",
        "union",
        "when",
        "where",
    }
)


class InvalidPlan(Exception):
    """Plan is not in the DSL. Compiler must not emit SQL."""


def validate_plan(plan: Any) -> dict[str, Any]:
    """Return a normalized copy. Raises InvalidPlan on any violation."""
    if not isinstance(plan, dict):
        raise InvalidPlan("plan must be a JSON object")
    intent = plan.get("intent")
    if intent not in INTENTS:
        raise InvalidPlan(f"intent must be one of {sorted(INTENTS)}")
    extra_top = _unknown_keys(plan, _allowed_top(intent))
    if extra_top:
        raise InvalidPlan(f"unknown fields: {sorted(extra_top)}")
    table = plan.get("table", TABLE)
    if intent != "refuse" and table != TABLE:
        raise InvalidPlan(f"table must be {TABLE}")
    if intent == "refuse":
        return _validate_refuse(plan)
    if intent == "lookup":
        return _validate_lookup(plan)
    if intent == "aggregate":
        return _validate_aggregate(plan)
    return _validate_compare(plan)


def _allowed_top(intent: str) -> set[str]:
    if intent == "refuse":
        return {"intent", "reason"}
    if intent == "lookup":
        return {"intent", "table", "filters", "columns", "order_by", "limit"}
    if intent == "aggregate":
        return {"intent", "table", "filters", "select", "group_by", "order_by", "limit"}
    return {"intent", "table", "metric", "sides"}


def _unknown_keys(obj: dict, allowed: set[str]) -> set[str]:
    return set(obj) - allowed


def _validate_refuse(plan: dict) -> dict[str, Any]:
    out: dict[str, Any] = {"intent": "refuse"}
    if "reason" in plan:
        reason = plan["reason"]
        if not isinstance(reason, str) or not reason or len(reason) > 200:
            raise InvalidPlan("reason must be a non-empty string of at most 200 chars")
        out["reason"] = reason
    return out


def _validate_lookup(plan: dict) -> dict[str, Any]:
    filters = _validate_filters(plan.get("filters"), required=True)
    columns = plan.get("columns", list(COLUMNS))
    if not isinstance(columns, list) or not columns:
        raise InvalidPlan("columns must be a non-empty list")
    if len(columns) != len(set(columns)):
        raise InvalidPlan("columns must be unique")
    for col in columns:
        _require_column(col)
    limit = _validate_limit(plan.get("limit"), required=True)
    out: dict[str, Any] = {
        "intent": "lookup",
        "table": TABLE,
        "filters": filters,
        "columns": list(columns),
        "limit": limit,
    }
    order_by = _validate_order_by(plan.get("order_by"), allowed=set(columns))
    if order_by:
        out["order_by"] = order_by
    return out


def _validate_aggregate(plan: dict) -> dict[str, Any]:
    filters = _validate_filters(plan.get("filters"), required=False)
    select = _validate_select_list(plan.get("select"))
    group_by = plan.get("group_by") or []
    if not isinstance(group_by, list):
        raise InvalidPlan("group_by must be a list")
    if len(group_by) != len(set(group_by)):
        raise InvalidPlan("group_by must be unique")
    for col in group_by:
        _require_column(col)
    aliases = [item["as"] for item in select]
    if len(aliases) != len(set(aliases)):
        raise InvalidPlan("select aliases must be unique")
    if any(col in aliases for col in group_by):
        raise InvalidPlan("group_by column collides with a select alias")
    order_allowed = set(aliases) | set(group_by)
    out: dict[str, Any] = {
        "intent": "aggregate",
        "table": TABLE,
        "filters": filters,
        "select": select,
        "group_by": list(group_by),
    }
    if group_by:
        out["limit"] = _validate_limit(plan.get("limit"), required=True)
    elif "limit" in plan:
        out["limit"] = _validate_limit(plan.get("limit"), required=True)
    order_by = _validate_order_by(plan.get("order_by"), allowed=order_allowed)
    if order_by:
        out["order_by"] = order_by
    return out


def _validate_compare(plan: dict) -> dict[str, Any]:
    metric = _validate_select_item(plan.get("metric"), default_as=None)
    sides = plan.get("sides")
    if not isinstance(sides, list) or len(sides) != 2:
        raise InvalidPlan("compare requires exactly two sides")
    names: list[str] = []
    normalized_sides: list[dict[str, Any]] = []
    for i, side in enumerate(sides):
        if not isinstance(side, dict):
            raise InvalidPlan(f"sides[{i}] must be an object")
        extra = _unknown_keys(side, {"name", "filters"})
        if extra:
            raise InvalidPlan(f"sides[{i}] unknown fields: {sorted(extra)}")
        name = side.get("name")
        _require_alias(name, f"sides[{i}].name")
        if name in names:
            raise InvalidPlan("side names must be unique")
        names.append(name)
        filters = _validate_filters(side.get("filters"), required=True)
        normalized_sides.append({"name": name, "filters": filters})
    return {
        "intent": "compare",
        "table": TABLE,
        "metric": metric,
        "sides": normalized_sides,
    }


def _validate_select_list(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise InvalidPlan("select must be a non-empty list")
    if len(raw) > 8:
        raise InvalidPlan("select has at most 8 items")
    return [_validate_select_item(item, default_as=None) for item in raw]


def _validate_select_item(raw: Any, default_as: str | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise InvalidPlan("select item must be an object")
    extra = _unknown_keys(raw, {"fn", "col", "as"})
    if extra:
        raise InvalidPlan(f"select unknown fields: {sorted(extra)}")
    fn = raw.get("fn")
    if fn not in FNS:
        raise InvalidPlan(f"fn must be one of {sorted(FNS)}")
    col = raw.get("col")
    if fn == "count":
        if col is not None:
            raise InvalidPlan("count does not take col; use count(*)")
    else:
        _require_column(col)
        if col not in NUMERIC_COLS:
            raise InvalidPlan(f"{fn} is only allowed on {sorted(NUMERIC_COLS)}")
    alias = raw.get("as") or default_as or _default_alias(fn, col)
    _require_alias(alias, "alias")
    out: dict[str, Any] = {"fn": fn, "as": alias}
    if col is not None:
        out["col"] = col
    return out


def _default_alias(fn: str, col: str | None) -> str:
    if fn == "count":
        return "n"
    return f"{fn}_{col}"


def _validate_filters(raw: Any, *, required: bool) -> list[dict[str, Any]]:
    if raw is None:
        if required:
            raise InvalidPlan("filters are required")
        return []
    if not isinstance(raw, list):
        raise InvalidPlan("filters must be a list")
    if required and not raw:
        raise InvalidPlan("filters must be non-empty")
    if len(raw) > MAX_FILTERS:
        raise InvalidPlan(f"at most {MAX_FILTERS} filters")
    return [_validate_filter(item, i) for i, item in enumerate(raw)]


def _validate_filter(raw: Any, index: int) -> dict[str, Any]:
    prefix = f"filters[{index}]"
    if not isinstance(raw, dict):
        raise InvalidPlan(f"{prefix} must be an object")
    extra = _unknown_keys(raw, {"col", "op", "val"})
    if extra:
        raise InvalidPlan(f"{prefix} unknown fields: {sorted(extra)}")
    col = raw.get("col")
    op = raw.get("op")
    val = raw.get("val")
    _require_column(col)
    if op not in OPS:
        raise InvalidPlan(f"{prefix}.op must be one of {sorted(OPS)}")
    if op == "in":
        if col not in IN_COLS:
            raise InvalidPlan(f"{prefix}: in is only allowed on {sorted(IN_COLS)}")
        if not isinstance(val, list) or not val or len(val) > len(REGIONS):
            raise InvalidPlan(f"{prefix}.val must be a non-empty list")
        parsed = [_parse_value(col, item, f"{prefix}.val") for item in val]
        if len(parsed) != len(set(parsed)):
            raise InvalidPlan(f"{prefix}.val must be unique")
        return {"col": col, "op": op, "val": parsed}
    if isinstance(val, list):
        raise InvalidPlan(f"{prefix}.val must be a scalar unless op is in")
    return {"col": col, "op": op, "val": _parse_value(col, val, prefix)}


def _parse_value(col: str, val: Any, prefix: str) -> Any:
    if col in REGION_COLS:
        if val not in REGIONS:
            raise InvalidPlan(f"{prefix}: region must be one of {list(REGIONS)}")
        return val
    if col in TIMESTAMP_COLS:
        if not isinstance(val, str) or not TIMESTAMP_RE.match(val):
            raise InvalidPlan(
                f"{prefix}: timestamp must be 'YYYY-MM-DD HH:MM:SS' (NEM time)"
            )
        return val
    if col in NUMERIC_COLS:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise InvalidPlan(f"{prefix}: {col} must be a number")
        return float(val)
    if col == "period_type":
        if val not in PERIOD_TYPES:
            raise InvalidPlan(f"{prefix}: period_type must be TRADE")
        return val
    raise InvalidPlan(f"{prefix}: unsupported column")


def _require_alias(name: Any, label: str) -> None:
    if not isinstance(name, str) or not ALIAS_RE.match(name):
        raise InvalidPlan(f"{label} must match {ALIAS_RE.pattern}")
    if name in RESERVED:
        raise InvalidPlan(f"{label} {name!r} is reserved")


def _require_column(col: Any) -> None:
    if col not in COLUMNS:
        raise InvalidPlan(f"column must be one of {list(COLUMNS)}")


def _validate_limit(raw: Any, *, required: bool) -> int:
    if raw is None:
        if required:
            raise InvalidPlan("limit is required")
        raise InvalidPlan("limit is required when present")
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise InvalidPlan("limit must be an integer")
    if raw < 1 or raw > MAX_LIMIT:
        raise InvalidPlan(f"limit must be between 1 and {MAX_LIMIT}")
    return raw


def _validate_order_by(raw: Any, *, allowed: set[str]) -> list[dict[str, str]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise InvalidPlan("order_by must be a list")
    if len(raw) > 5:
        raise InvalidPlan("order_by has at most 5 items")
    out: list[dict[str, str]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise InvalidPlan(f"order_by[{i}] must be an object")
        extra = _unknown_keys(item, {"col", "dir"})
        if extra:
            raise InvalidPlan(f"order_by[{i}] unknown fields: {sorted(extra)}")
        col = item.get("col")
        if col not in allowed:
            raise InvalidPlan(f"order_by[{i}].col must be one of {sorted(allowed)}")
        direction = item.get("dir", "asc")
        if direction not in {"asc", "desc"}:
            raise InvalidPlan(f"order_by[{i}].dir must be asc or desc")
        out.append({"col": col, "dir": direction})
    return out
