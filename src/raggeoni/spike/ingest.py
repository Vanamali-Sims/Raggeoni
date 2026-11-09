"""Build incident databases and keep the live window up to date.

Every loader is optional: if a source fails (no API key, site down), the
incident is still built and the gap is recorded in ``meta`` so agents and
reports can say what they could not see.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from raggeoni.spike.config import fmt_ts, interconnectors_for, neighbours, nem_now, parse_ts
from raggeoni.spike.sources import nemweb, notices, openelectricity
from raggeoni.spike.sources.http import FetchError
from raggeoni.spike.sources.mms import merge_tables
from raggeoni.spike.store import prune_before, set_meta, upsert, upsert_notices


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _record_gap(db: Path, source: str, error: Exception) -> None:
    _log(f"  ! {source} unavailable: {error}")
    set_meta(db, f"gap:{source}", str(error)[:500])


def load_dispatch_archive(db: Path, start: datetime, end: datetime) -> int:
    """Prices, demand, supply and interconnectors for [start, end] from daily archives."""
    total = 0
    for day in nemweb.days_between(start, end):
        url = nemweb.archive_day_url(nemweb.DISPATCH_ARCHIVE, day)
        if url is None:
            raise FetchError(f"no DispatchIS archive file for {day}")
        _log(f"  dispatch {day}")
        tables = nemweb.load_tables(url)
        total += _write_dispatch(db, tables, start, end)
    return total


def load_dispatch_current(db: Path, since: datetime) -> int:
    files = nemweb.current_dispatch_files(since)
    tables = merge_tables(nemweb.load_tables(f.url) for f in files)
    return _write_dispatch(db, tables, since, nem_now() + timedelta(hours=1))


def _write_dispatch(db: Path, tables: dict, start: datetime, end: datetime) -> int:
    lo, hi = fmt_ts(start), fmt_ts(end)

    def keep(rows: list[tuple]) -> list[tuple]:
        return [r for r in rows if lo <= r[0] <= hi]

    n = upsert(db, "price_demand", keep(nemweb.price_demand_rows(tables)))
    upsert(db, "region_supply", keep(nemweb.region_supply_rows(tables)))
    upsert(db, "interconnector", keep(nemweb.interconnector_rows(tables)))
    return n


def load_forecasts_archive(db: Path, start: datetime, end: datetime) -> int:
    total = 0
    for day in nemweb.days_between(start, end):
        url = nemweb.archive_day_url(nemweb.P5MIN_ARCHIVE, day)
        if url is None:
            raise FetchError(f"no P5MIN archive file for {day}")
        _log(f"  p5min {day}")
        rows = nemweb.p5min_forecast_rows(nemweb.load_tables(url))
        lo, hi = fmt_ts(start), fmt_ts(end)
        total += upsert(db, "forecast", [r for r in rows if lo <= r[1] <= hi])
    return total


def load_generation(db: Path, regions: list[str], start: datetime, end: datetime) -> int:
    total = 0
    for region in regions:
        _log(f"  generation {region}")
        total += upsert(db, "generation", openelectricity.fetch_generation(region, start, end))
    return total


def load_notices(db: Path, start: datetime, end: datetime) -> int:
    found = notices.fetch_notices((start - timedelta(days=1)).date(), end.date())
    lo, hi = fmt_ts(start - timedelta(hours=24)), fmt_ts(end)
    return upsert_notices(db, [n for n in found if lo <= n.issued_ts <= hi])


def build_incident(
    db: Path,
    *,
    region: str,
    start: datetime,
    end: datetime,
    baseline_days: int = 30,
    with_forecasts: bool = True,
) -> dict[str, int | str]:
    """Pull everything an investigation needs for one event into ``db``.

    ``baseline_days`` of history before ``start`` are loaded for the region
    and its neighbours so tools can tell whether conditions were unusual.
    """
    counts: dict[str, int | str] = {}
    history_start = start - timedelta(days=baseline_days)
    set_meta(
        db, "window", json.dumps({"start": fmt_ts(start), "end": fmt_ts(end), "region": region})
    )
    steps = [
        ("dispatch", lambda: load_dispatch_archive(db, history_start, end)),
        (
            "generation",
            lambda: load_generation(db, [region, *neighbours(region)], history_start, end),
        ),
        ("notices", lambda: load_notices(db, start, end)),
    ]
    if with_forecasts:
        steps.append(
            ("forecast", lambda: load_forecasts_archive(db, start - timedelta(hours=2), end))
        )
    for name, step in steps:
        try:
            counts[name] = step()
        except (FetchError, OSError, ValueError) as exc:
            _record_gap(db, name, exc)
            counts[name] = f"unavailable: {exc}"[:200]
    set_meta(db, "built_at", fmt_ts(nem_now()))
    return counts


def refresh_live(db: Path, *, keep_days: int = 8) -> dict[str, int | str]:
    """Update the rolling live window from NEMWeb Current and Open Electricity."""
    now = nem_now()
    last = _last_ts(db)
    since = (
        max(last - timedelta(minutes=10), now - timedelta(days=keep_days))
        if last
        else now - timedelta(hours=6)
    )
    counts: dict[str, int | str] = {}
    try:
        counts["dispatch"] = load_dispatch_current(db, since)
    except (FetchError, OSError, ValueError) as exc:
        _record_gap(db, "dispatch", exc)
        counts["dispatch"] = f"unavailable: {exc}"[:200]
    try:
        counts["generation"] = load_generation(db, list(_all_regions()), since, now)
    except (FetchError, OSError, ValueError) as exc:
        _record_gap(db, "generation", exc)
        counts["generation"] = f"unavailable: {exc}"[:200]
    prune_before(db, fmt_ts(now - timedelta(days=keep_days)))
    set_meta(db, "refreshed_at", fmt_ts(now))
    return counts


def _all_regions() -> tuple[str, ...]:
    from raggeoni.constants import REGIONS

    return REGIONS


def _last_ts(db: Path) -> datetime | None:
    if not db.exists():
        return None
    from raggeoni.spike.store import query

    rows = query(db, "SELECT max(settlement_ts) FROM price_demand")
    return parse_ts(rows[0][0]) if rows and rows[0][0] else None


def related_interconnectors(region: str) -> list[str]:
    return [ic.id for ic in interconnectors_for(region)]
