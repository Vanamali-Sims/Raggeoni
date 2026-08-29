"""Sanity checks that must pass before any eval gold is written."""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb

from raggeoni.constants import (
    EXPECTED_MAX_TS,
    EXPECTED_MIN_TS,
    EXPECTED_ROWS,
    INTERVAL_MINUTES,
    REGIONS,
)


class CheckError(Exception):
    """One or more freeze checks failed."""


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass
class CheckReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.results.append(CheckResult(name, ok, detail))

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.results)

    def raise_if_failed(self) -> None:
        failed = [r for r in self.results if not r.ok]
        if not failed:
            return
        lines = ["Freeze checks failed:"]
        lines.extend(f"  - {r.name}: {r.detail}" for r in failed)
        raise CheckError("\n".join(lines))


def _scalar(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchone()[0]


def run_checks(con: duckdb.DuckDBPyConnection) -> CheckReport:
    report = CheckReport()

    regions = [
        row[0]
        for row in con.execute(
            "SELECT DISTINCT region FROM price_demand ORDER BY region"
        ).fetchall()
    ]
    expected = sorted(REGIONS)
    report.add(
        "regions",
        regions == expected,
        f"got {regions}, expected {expected}",
    )

    row_count = _scalar(con, "SELECT COUNT(*) FROM price_demand")
    report.add(
        "row_count",
        row_count == EXPECTED_ROWS,
        f"got {row_count}, expected {EXPECTED_ROWS}",
    )

    dupes = _scalar(
        con,
        """
        SELECT COUNT(*) FROM (
            SELECT settlement_ts, region
            FROM price_demand
            GROUP BY 1, 2
            HAVING COUNT(*) > 1
        )
        """,
    )
    report.add("no_duplicates", dupes == 0, f"{dupes} duplicate (ts, region) keys")

    nulls = _scalar(
        con,
        """
        SELECT COUNT(*) FROM price_demand
        WHERE settlement_ts IS NULL
           OR region IS NULL
           OR rrp IS NULL
           OR total_demand IS NULL
           OR period_type IS NULL
        """,
    )
    report.add("no_nulls", nulls == 0, f"{nulls} rows with nulls")

    min_ts = str(_scalar(con, "SELECT MIN(settlement_ts) FROM price_demand"))
    max_ts = str(_scalar(con, "SELECT MAX(settlement_ts) FROM price_demand"))
    report.add(
        "min_timestamp",
        min_ts == EXPECTED_MIN_TS,
        f"got {min_ts}, expected {EXPECTED_MIN_TS}",
    )
    report.add(
        "max_timestamp",
        max_ts == EXPECTED_MAX_TS,
        f"got {max_ts}, expected {EXPECTED_MAX_TS}",
    )

    negative_rrp = _scalar(con, "SELECT COUNT(*) FROM price_demand WHERE rrp < 0")
    report.add(
        "negative_prices",
        negative_rrp > 0,
        f"{negative_rrp} negative RRP rows (NEM must have some in a 12-month window)",
    )

    period_types = [
        row[0]
        for row in con.execute(
            "SELECT DISTINCT period_type FROM price_demand ORDER BY 1"
        ).fetchall()
    ]
    report.add(
        "period_type",
        period_types == ["TRADE"],
        f"got {period_types}, expected ['TRADE']",
    )

    gaps = _scalar(
        con,
        f"""
        SELECT COUNT(*) FROM (
            SELECT region, settlement_ts,
                   lag(settlement_ts) OVER (
                       PARTITION BY region ORDER BY settlement_ts
                   ) AS prev_ts
            FROM price_demand
        )
        WHERE prev_ts IS NOT NULL
          AND date_diff('second', prev_ts, settlement_ts) != {INTERVAL_MINUTES * 60}
        """,
    )
    report.add("no_interval_gaps", gaps == 0, f"{gaps} gaps in 5-minute sequence")

    per_region = dict(
        con.execute(
            """
            SELECT region, COUNT(*)
            FROM price_demand
            GROUP BY region
            ORDER BY region
            """
        ).fetchall()
    )
    balanced = all(n == EXPECTED_ROWS // len(REGIONS) for n in per_region.values())
    report.add("balanced_regions", balanced, f"per-region counts: {per_region}")

    return report
