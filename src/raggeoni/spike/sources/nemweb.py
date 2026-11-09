"""AEMO NEMWeb loaders: dispatch prices, demand, interconnectors, 5-minute forecasts.

Two folders matter for each report:

* ``Reports/Current/<report>/`` keeps every 5-minute file for roughly the last day or two.
* ``Reports/Archive/<report>/`` keeps one zip per day (a zip of the 5-minute zips).

Live monitoring reads Current. The exam and 30-day baselines read Archive.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from raggeoni.spike.sources.http import fetch, list_links
from raggeoni.spike.sources.mms import Table, mms_ts, parse_mms, to_float

BASE = "https://nemweb.com.au"
DISPATCH_CURRENT = f"{BASE}/Reports/Current/DispatchIS_Reports/"
DISPATCH_ARCHIVE = f"{BASE}/Reports/Archive/DispatchIS_Reports/"
P5MIN_CURRENT = f"{BASE}/Reports/Current/P5_Reports/"
P5MIN_ARCHIVE = f"{BASE}/Reports/Archive/P5_Reports/"

_STAMP = re.compile(r"_(\d{8})(\d{4})?")


@dataclass(frozen=True)
class RemoteFile:
    url: str
    name: str
    stamp: str  # YYYYMMDD or YYYYMMDDHHMM taken from the file name


_LISTINGS: dict[str, str] = {}


def list_folder(folder_url: str, *, suffix: str = ".zip") -> list[RemoteFile]:
    if "/Archive/" in folder_url and folder_url in _LISTINGS:
        html = _LISTINGS[folder_url]
    else:
        html = fetch(folder_url).decode("utf-8", errors="replace")
        if "/Archive/" in folder_url:
            _LISTINGS[folder_url] = html
    files: list[RemoteFile] = []
    for href in list_links(html):
        name = href.rsplit("/", 1)[-1]
        if not name.lower().endswith(suffix):
            continue
        match = _STAMP.search(name)
        if not match:
            continue
        stamp = match.group(1) + (match.group(2) or "")
        url = (
            href
            if href.startswith("http")
            else BASE + href
            if href.startswith("/")
            else folder_url + href
        )
        files.append(RemoteFile(url=url, name=name, stamp=stamp))
    files.sort(key=lambda f: f.stamp)
    return files


def iter_csv_texts(blob: bytes) -> Iterator[str]:
    """Yield every CSV inside a zip, descending into nested zips (Archive files)."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for info in zf.infolist():
            name = info.filename.lower()
            data = zf.read(info)
            if name.endswith(".zip"):
                yield from iter_csv_texts(data)
            elif name.endswith(".csv"):
                yield data.decode("utf-8", errors="replace")


# ---- table -> rows -----------------------------------------------------------


def price_demand_rows(tables: dict[str, Table]) -> list[tuple]:
    """(settlement_ts, region, rrp, total_demand, period_type) from DISPATCH.PRICE + REGIONSUM.

    Only rows with INTERVENTION = 0 (the pricing run) are kept.
    """
    demand: dict[tuple[str, str], float] = {}
    for row in tables.get("DISPATCH.REGIONSUM", []):
        if row.get("INTERVENTION", "0").strip() not in ("0", ""):
            continue
        value = to_float(row.get("TOTALDEMAND"))
        if value is not None:
            demand[(mms_ts(row["SETTLEMENTDATE"]), row["REGIONID"].strip())] = value
    out: list[tuple] = []
    for row in tables.get("DISPATCH.PRICE", []):
        if row.get("INTERVENTION", "0").strip() not in ("0", ""):
            continue
        ts = mms_ts(row["SETTLEMENTDATE"])
        region = row["REGIONID"].strip()
        rrp = to_float(row.get("RRP"))
        if rrp is None or (ts, region) not in demand:
            continue
        out.append((ts, region, rrp, demand[(ts, region)], "TRADE"))
    return out


def region_supply_rows(tables: dict[str, Table]) -> list[tuple]:
    """(settlement_ts, region, available_generation, dispatchable_generation, net_interchange)."""
    out: list[tuple] = []
    for row in tables.get("DISPATCH.REGIONSUM", []):
        if row.get("INTERVENTION", "0").strip() not in ("0", ""):
            continue
        out.append(
            (
                mms_ts(row["SETTLEMENTDATE"]),
                row["REGIONID"].strip(),
                to_float(row.get("AVAILABLEGENERATION")),
                to_float(row.get("DISPATCHABLEGENERATION")),
                to_float(row.get("NETINTERCHANGE")),
            )
        )
    return out


def interconnector_rows(tables: dict[str, Table]) -> list[tuple]:
    """(settlement_ts, interconnector_id, mw_flow, export_limit, import_limit, marginal_value).

    ``import_limit`` is stored as a non-positive number so that a flow is at its
    limit when ``mw_flow >= export_limit`` or ``mw_flow <= import_limit``.
    A non-zero marginal value means a constraint on the link was binding.
    """
    out: list[tuple] = []
    for row in tables.get("DISPATCH.INTERCONNECTORRES", []):
        if row.get("INTERVENTION", "0").strip() not in ("0", ""):
            continue
        imp = to_float(row.get("IMPORTLIMIT"))
        out.append(
            (
                mms_ts(row["SETTLEMENTDATE"]),
                row["INTERCONNECTORID"].strip(),
                to_float(row.get("MWFLOW")),
                to_float(row.get("EXPORTLIMIT")),
                -abs(imp) if imp is not None else None,
                to_float(row.get("MARGINALVALUE")) or 0.0,
            )
        )
    return out


def p5min_forecast_rows(tables: dict[str, Table]) -> list[tuple]:
    """(run_ts, interval_ts, region, forecast_demand, forecast_rrp) from P5MIN.REGIONSOLUTION."""
    out: list[tuple] = []
    for row in tables.get("P5MIN.REGIONSOLUTION", []):
        if row.get("INTERVENTION", "0").strip() not in ("0", ""):
            continue
        out.append(
            (
                mms_ts(row["RUN_DATETIME"]),
                mms_ts(row["INTERVAL_DATETIME"]),
                row["REGIONID"].strip(),
                to_float(row.get("TOTALDEMAND")),
                to_float(row.get("RRP")),
            )
        )
    return out


# ---- fetch plans -------------------------------------------------------------


def current_dispatch_files(since: datetime | None = None) -> list[RemoteFile]:
    files = list_folder(DISPATCH_CURRENT)
    if since is None:
        return files
    cutoff = since.strftime("%Y%m%d%H%M")
    return [f for f in files if f.stamp >= cutoff]


def archive_day_url(folder: str, day: date) -> str | None:
    stamp = day.strftime("%Y%m%d")
    for remote in list_folder(folder):
        if remote.stamp.startswith(stamp):
            return remote.url
    return None


def load_tables(url: str) -> dict[str, Table]:
    from raggeoni.spike.sources.mms import merge_tables

    return merge_tables(parse_mms(text) for text in iter_csv_texts(fetch_cached(url)))


def fetch_cached(url: str) -> bytes:
    """Archive files never change, so keep them on disk. Several exam events
    share days, and CI restores this folder from the Actions cache."""
    if "/Archive/" not in url:
        return fetch(url)
    from raggeoni.spike.config import data_dir

    cache = data_dir() / "cache" / url.rsplit("/", 1)[-1]
    if cache.exists() and cache.stat().st_size > 0:
        return cache.read_bytes()
    blob = fetch(url, timeout=300)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(cache.suffix + ".part")
    tmp.write_bytes(blob)
    tmp.replace(cache)
    return blob


def days_between(start: datetime, end: datetime) -> list[date]:
    """Calendar days touched by [start, end]. A NEM day D's last interval ends at D+1 00:00."""
    first = (start - timedelta(minutes=5)).date()
    last = end.date()
    out = []
    day = first
    while day <= last:
        out.append(day)
        day += timedelta(days=1)
    return out
