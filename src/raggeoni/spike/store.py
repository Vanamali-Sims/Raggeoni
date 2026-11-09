"""DuckDB store for one incident (or the rolling live window).

The ``price_demand`` table has exactly the same shape as Raggeoni's frozen
table, so the existing plan validator and compiler work on it unchanged.
"""

from __future__ import annotations

import csv
import tempfile
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb

from raggeoni.spike.sources.notices import Notice

SCHEMA = """
CREATE TABLE IF NOT EXISTS price_demand (
    settlement_ts TIMESTAMP NOT NULL,
    region VARCHAR NOT NULL,
    rrp DOUBLE NOT NULL,
    total_demand DOUBLE NOT NULL,
    period_type VARCHAR NOT NULL,
    PRIMARY KEY (settlement_ts, region)
);
CREATE TABLE IF NOT EXISTS region_supply (
    settlement_ts TIMESTAMP NOT NULL,
    region VARCHAR NOT NULL,
    available_generation DOUBLE,
    dispatchable_generation DOUBLE,
    net_interchange DOUBLE,
    PRIMARY KEY (settlement_ts, region)
);
CREATE TABLE IF NOT EXISTS interconnector (
    settlement_ts TIMESTAMP NOT NULL,
    interconnector_id VARCHAR NOT NULL,
    mw_flow DOUBLE,
    export_limit DOUBLE,
    import_limit DOUBLE,
    marginal_value DOUBLE,
    PRIMARY KEY (settlement_ts, interconnector_id)
);
CREATE TABLE IF NOT EXISTS generation (
    settlement_ts TIMESTAMP NOT NULL,
    region VARCHAR NOT NULL,
    fueltech_group VARCHAR NOT NULL,
    mw DOUBLE NOT NULL,
    PRIMARY KEY (settlement_ts, region, fueltech_group)
);
CREATE TABLE IF NOT EXISTS forecast (
    run_ts TIMESTAMP NOT NULL,
    interval_ts TIMESTAMP NOT NULL,
    region VARCHAR NOT NULL,
    forecast_demand DOUBLE,
    forecast_rrp DOUBLE,
    PRIMARY KEY (run_ts, interval_ts, region)
);
CREATE TABLE IF NOT EXISTS notices (
    notice_id VARCHAR PRIMARY KEY,
    issued_ts TIMESTAMP NOT NULL,
    notice_type VARCHAR,
    description VARCHAR,
    external_ref VARCHAR,
    body VARCHAR,
    regions VARCHAR
);
CREATE TABLE IF NOT EXISTS meta (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);
"""

_TABLE_COLUMNS: dict[str, int] = {
    "price_demand": 5,
    "region_supply": 5,
    "interconnector": 6,
    "generation": 4,
    "forecast": 5,
}


def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    try:
        con.execute(SCHEMA)
    finally:
        con.close()


@contextmanager
def connect(path: Path, *, read_only: bool = False):
    if not read_only:
        init_db(path)
    con = duckdb.connect(str(path), read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def upsert(path: Path, table: str, rows: Sequence[Sequence[Any]]) -> int:
    if table not in _TABLE_COLUMNS:
        raise ValueError(f"unknown table {table}")
    if not rows:
        return 0
    width = _TABLE_COLUMNS[table]
    if len(rows) <= 500:
        placeholders = ", ".join(["?"] * width)
        with connect(path) as con:
            con.executemany(
                f"INSERT OR REPLACE INTO {table} VALUES ({placeholders})", [list(r) for r in rows]
            )
        return len(rows)
    # executemany is slow in DuckDB; stage large batches through a CSV.
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "rows.csv"
        with staged.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            for row in rows:
                writer.writerow(["" if v is None else v for v in row])
        with connect(path) as con:
            cols = [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]
            types = [r[1] for r in con.execute(f"DESCRIBE {table}").fetchall()]
            spec = ", ".join(f"'{c}': '{t}'" for c, t in zip(cols, types, strict=True))
            con.execute(
                f"INSERT OR REPLACE INTO {table} SELECT * FROM read_csv(?, header=false, "
                f"columns={{{spec}}}, nullstr='')",
                [str(staged)],
            )
    return len(rows)


def upsert_notices(path: Path, notices: Iterable[Notice]) -> int:
    rows = [
        [
            n.notice_id,
            n.issued_ts,
            n.notice_type,
            n.description,
            n.external_ref,
            n.body,
            ",".join(n.regions),
        ]
        for n in notices
    ]
    if not rows:
        return 0
    with connect(path) as con:
        con.executemany("INSERT OR REPLACE INTO notices VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    return len(rows)


def set_meta(path: Path, key: str, value: str) -> None:
    with connect(path) as con:
        con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", [key, value])


def get_meta(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    with connect(path, read_only=True) as con:
        row = con.execute("SELECT value FROM meta WHERE key = ?", [key]).fetchone()
    return row[0] if row else None


def query(path: Path, sql: str, params: Sequence[Any] = ()) -> list[tuple]:
    with connect(path, read_only=True) as con:
        return [tuple(r) for r in con.execute(sql, list(params)).fetchall()]


def prune_before(path: Path, cutoff: str) -> None:
    """Drop rows older than cutoff (NEM time) to keep the live window small."""
    with connect(path) as con:
        for table in ("price_demand", "region_supply", "interconnector", "generation"):
            con.execute(f"DELETE FROM {table} WHERE settlement_ts < ?", [cutoff])
        con.execute("DELETE FROM forecast WHERE interval_ts < ?", [cutoff])
