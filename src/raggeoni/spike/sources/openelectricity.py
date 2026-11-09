"""Open Electricity API v4: generation by fuel-technology group for one region.

Docs: https://docs.openelectricity.org.au/api-reference/data/get-network-data

Request::

    GET https://api.openelectricity.org.au/v4/data/network/NEM
        ?metrics=power&interval=5m&network_region=SA1
        &primary_grouping=network_region&secondary_grouping=fueltech_group
        &date_start=2026-01-26T12:00:00&date_end=2026-01-27T00:00:00
    Authorization: Bearer $OPENELECTRICITY_API_KEY

Dates are network-local (NEM time) and timezone-naive. 5-minute requests are
limited to 8 days, so longer ranges are split into chunks.

The response wraps time series as ``data[].results[]`` where each result has
``columns`` (e.g. ``{"network_region": "SA1", "fueltech_group": "wind"}``) and
``data`` as ``[[timestamp, value], ...]``. The parser is tolerant of small
shape differences and is covered by a recorded fixture in the tests.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any

from raggeoni.spike.config import fmt_ts, parse_ts
from raggeoni.spike.sources.http import FetchError, fetch

API_BASE = "https://api.openelectricity.org.au/v4"
MAX_5M_SPAN = timedelta(days=7)


def api_key() -> str | None:
    return os.environ.get("OPENELECTRICITY_API_KEY")


def _chunks(start: datetime, end: datetime) -> Iterator[tuple[datetime, datetime]]:
    cursor = start
    while cursor < end:
        upper = min(cursor + MAX_5M_SPAN, end)
        yield cursor, upper
        cursor = upper


def fetch_generation(region: str, start: datetime, end: datetime) -> list[tuple]:
    """Return (settlement_ts, region, fueltech_group, mw) rows."""
    key = api_key()
    if not key:
        raise FetchError("OPENELECTRICITY_API_KEY is not set; generation data unavailable")
    rows: list[tuple] = []
    for lo, hi in _chunks(start, end):
        query = urllib.parse.urlencode(
            {
                "metrics": "power",
                "interval": "5m",
                "network_region": region,
                "primary_grouping": "network_region",
                "secondary_grouping": "fueltech_group",
                "date_start": lo.strftime("%Y-%m-%dT%H:%M:%S"),
                "date_end": hi.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        payload = json.loads(
            fetch(
                f"{API_BASE}/data/network/NEM?{query}",
                headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
            )
        )
        rows.extend(parse_generation(payload, region))
    return rows


def parse_generation(payload: dict[str, Any], region: str) -> list[tuple]:
    if payload.get("success") is False:
        raise FetchError(f"Open Electricity error: {payload.get('error')}")
    out: dict[tuple[str, str], float] = {}
    for series in payload.get("data") or []:
        if series.get("metric", "power") != "power":
            continue
        for result in series.get("results") or []:
            cols = result.get("columns") or {}
            group = cols.get("fueltech_group") or cols.get("fueltech")
            series_region = cols.get("network_region", region)
            if not group or series_region != region:
                continue
            for point in result.get("data") or []:
                if not isinstance(point, list | tuple) or len(point) < 2 or point[1] is None:
                    continue
                ts = fmt_ts(parse_ts(str(point[0])))
                # Some groups (battery charging, pumps, exports) come back
                # negative; keep the sign so supply arithmetic stays honest.
                out[(ts, group)] = out.get((ts, group), 0.0) + float(point[1])
    return [(ts, region, group, round(mw, 3)) for (ts, group), mw in sorted(out.items())]
