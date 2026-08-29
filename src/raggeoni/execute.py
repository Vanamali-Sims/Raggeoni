"""Run a compiled plan against the frozen DuckDB file."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from raggeoni.compile import CompiledQuery, compile_plan
from raggeoni.ingest import db_path
from raggeoni.validate import InvalidPlan

FLOAT_PLACES = 4
ROW_CAP = 1000


@dataclass(frozen=True)
class QueryResult:
    status: str
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    error_class: str | None = None
    error: str | None = None
    sql: str | None = None
    params: tuple[Any, ...] = ()

    def to_gold(self) -> dict[str, Any]:
        if self.status == "refuse":
            return {"status": "refuse"}
        if self.status != "ok":
            raise ValueError("cannot store an error result as gold")
        return {
            "status": "ok",
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
        }


def execute_plan(
    plan: dict[str, Any],
    *,
    db: Path | None = None,
) -> QueryResult:
    try:
        compiled = compile_plan(plan)
    except InvalidPlan as exc:
        return QueryResult(
            status="error",
            error_class="invalid_plan",
            error=str(exc),
        )
    if compiled.intent == "refuse":
        return QueryResult(status="refuse")
    path = db or db_path()
    if not path.exists():
        return QueryResult(
            status="error",
            error_class="sql",
            error=f"missing database {path}",
            sql=compiled.sql,
            params=compiled.params,
        )
    try:
        rows = _run(path, compiled)
    except Exception as exc:  # noqa: BLE001 — surface DuckDB errors as sql
        return QueryResult(
            status="error",
            error_class="sql",
            error=str(exc),
            sql=compiled.sql,
            params=compiled.params,
        )
    if len(rows) > ROW_CAP:
        return QueryResult(
            status="error",
            error_class="row_cap",
            error=f"result has {len(rows)} rows; cap is {ROW_CAP}",
            sql=compiled.sql,
            params=compiled.params,
        )
    canonical = tuple(
        tuple(canonicalize_value(value) for value in row) for row in rows
    )
    if not _has_order_by(compiled):
        canonical = tuple(sorted(canonical, key=_sort_key))
    return QueryResult(
        status="ok",
        columns=compiled.columns,
        rows=canonical,
        sql=compiled.sql,
        params=compiled.params,
    )


def _has_order_by(compiled: CompiledQuery) -> bool:
    return compiled.sql is not None and " ORDER BY " in compiled.sql


def _run(path: Path, compiled: CompiledQuery) -> list[tuple[Any, ...]]:
    assert compiled.sql is not None
    con = duckdb.connect(str(path), read_only=True)
    try:
        con.execute("SET threads = 1")
        result = con.execute(compiled.sql, list(compiled.params))
        fetched = result.fetchall()
        return [tuple(row) for row in fetched]
    finally:
        con.close()


def canonicalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, FLOAT_PLACES)
    return value


def _sort_key(row: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(_sortable(v) for v in row)


def _sortable(value: Any) -> Any:
    if value is None:
        return ("", "")
    return (type(value).__name__, value)


def results_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left == right
