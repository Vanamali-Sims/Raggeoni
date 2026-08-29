"""Compile a validated plan into parameterized SQL. Never interpolates values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from raggeoni.validate import (
    COLUMNS,
    NUMERIC_COLS,
    TIMESTAMP_COLS,
    InvalidPlan,
    validate_plan,
)

IDENTIFIERS = frozenset(COLUMNS) | {"side"}


@dataclass(frozen=True)
class CompiledQuery:
    intent: str
    sql: str | None
    params: tuple[Any, ...]
    columns: tuple[str, ...]


def compile_plan(plan: dict[str, Any]) -> CompiledQuery:
    normalized = validate_plan(plan)
    intent = normalized["intent"]
    if intent == "refuse":
        return CompiledQuery("refuse", None, (), ())
    if intent == "lookup":
        return _compile_lookup(normalized)
    if intent == "aggregate":
        return _compile_aggregate(normalized)
    return _compile_compare(normalized)


def _ident(name: str) -> str:
    if name not in IDENTIFIERS and not _is_alias(name):
        raise InvalidPlan(f"refusing to interpolate identifier {name!r}")
    return name


def _is_alias(name: str) -> bool:
    return name.isidentifier() and name.islower() and name[0].isalpha()


def _compile_lookup(plan: dict[str, Any]) -> CompiledQuery:
    columns = tuple(plan["columns"])
    select = ", ".join(_ident(c) for c in columns)
    where_sql, where_params = _where(plan["filters"])
    order_sql, order_params = _order(plan.get("order_by"))
    sql = f"SELECT {select} FROM price_demand{where_sql}{order_sql} LIMIT ?"
    params = tuple(where_params + order_params + [plan["limit"]])
    return CompiledQuery("lookup", sql, params, columns)


def _compile_aggregate(plan: dict[str, Any]) -> CompiledQuery:
    group_by = plan.get("group_by") or []
    select_sql, aliases = _select_aggs(plan["select"], group_by)
    where_sql, where_params = _where(plan["filters"])
    group_sql = ""
    if group_by:
        group_sql = " GROUP BY " + ", ".join(_ident(c) for c in group_by)
    order_sql, order_params = _order(plan.get("order_by"))
    sql = f"SELECT {select_sql} FROM price_demand{where_sql}{group_sql}{order_sql}"
    params: list[Any] = list(where_params + order_params)
    if "limit" in plan:
        sql += " LIMIT ?"
        params.append(plan["limit"])
    columns = tuple(list(group_by) + aliases)
    return CompiledQuery("aggregate", sql, tuple(params), columns)


def _compile_compare(plan: dict[str, Any]) -> CompiledQuery:
    metric = plan["metric"]
    metric_sql = _agg_expr(metric)
    alias = metric["as"]
    parts: list[str] = []
    params: list[Any] = []
    for side in plan["sides"]:
        where_sql, where_params = _where(side["filters"])
        parts.append(
            f"SELECT ? AS side, {metric_sql} AS {_ident(alias)} "
            f"FROM price_demand{where_sql}"
        )
        params.append(side["name"])
        params.extend(where_params)
    sql = " UNION ALL ".join(parts)
    return CompiledQuery("compare", sql, tuple(params), ("side", alias))


def _select_aggs(
    items: list[dict[str, Any]], group_by: list[str]
) -> tuple[str, list[str]]:
    pieces: list[str] = [_ident(c) for c in group_by]
    aliases: list[str] = []
    for item in items:
        alias = item["as"]
        pieces.append(f"{_agg_expr(item)} AS {_ident(alias)}")
        aliases.append(alias)
    return ", ".join(pieces), aliases


def _agg_expr(item: dict[str, Any]) -> str:
    fn = item["fn"]
    if fn == "count":
        return "count(*)"
    col = _ident(item["col"])
    return f"{fn}({col})"


def _where(filters: list[dict[str, Any]]) -> tuple[str, list[Any]]:
    if not filters:
        return "", []
    clauses: list[str] = []
    params: list[Any] = []
    for item in filters:
        col = _ident(item["col"])
        op = item["op"]
        val = item["val"]
        binder = _binder(item["col"])
        if op == "in":
            placeholders = ", ".join([binder] * len(val))
            clauses.append(f"{col} IN ({placeholders})")
            params.extend(val)
        else:
            clauses.append(f"{col} {op} {binder}")
            params.append(val)
    return " WHERE " + " AND ".join(clauses), params


def _binder(col: str) -> str:
    if col in TIMESTAMP_COLS:
        return "CAST(? AS TIMESTAMP)"
    if col in NUMERIC_COLS:
        return "CAST(? AS DOUBLE)"
    return "?"


def _order(order_by: list[dict[str, str]] | None) -> tuple[str, list[Any]]:
    if not order_by:
        return "", []
    bits: list[str] = []
    for item in order_by:
        direction = "ASC" if item["dir"] == "asc" else "DESC"
        bits.append(f"{_ident(item['col'])} {direction}")
    return " ORDER BY " + ", ".join(bits), []
