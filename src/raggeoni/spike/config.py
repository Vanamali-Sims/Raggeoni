"""Paths, market constants and NEM time helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from raggeoni.constants import REGIONS

# NEM time is AEST (UTC+10) all year. No daylight saving.
NEM_TZ = timezone(timedelta(hours=10), name="NEM")
INTERVAL = timedelta(minutes=5)
TS_FMT = "%Y-%m-%d %H:%M:%S"

# A price at or above this (in $/MWh) opens an investigation. The AER's
# formal reporting threshold is $5,000/MWh, but smaller spikes are still
# worth explaining, and the AER covers some of them in its appendices.
SPIKE_THRESHOLD = float(os.environ.get("SPIKE_THRESHOLD", "300"))
AER_THRESHOLD = 5000.0

# Sanity range for prices. The floor is -$1,000/MWh; the cap changes each
# financial year, so the upper bound is generous.
PRICE_FLOOR = -1000.0
PRICE_SANE_MAX = 25_000.0

REGION_NAMES = {
    "NSW1": "New South Wales",
    "QLD1": "Queensland",
    "VIC1": "Victoria",
    "SA1": "South Australia",
    "TAS1": "Tasmania",
}


@dataclass(frozen=True)
class Interconnector:
    id: str
    name: str
    from_region: str  # positive MWFLOW means power flows from_region -> to_region
    to_region: str


INTERCONNECTORS: tuple[Interconnector, ...] = (
    Interconnector("NSW1-QLD1", "QNI", "NSW1", "QLD1"),
    Interconnector("N-Q-MNSP1", "Terranora", "NSW1", "QLD1"),
    Interconnector("VIC1-NSW1", "VIC-NSW", "VIC1", "NSW1"),
    Interconnector("V-SA", "Heywood", "VIC1", "SA1"),
    Interconnector("V-S-MNSP1", "Murraylink", "VIC1", "SA1"),
    Interconnector("T-V-MNSP1", "Basslink", "TAS1", "VIC1"),
)

FUEL_GROUPS = (
    "coal",
    "gas",
    "wind",
    "solar",
    "hydro",
    "battery_discharging",
    "battery_charging",
    "distillate",
    "bioenergy",
    "pumps",
)


def interconnectors_for(region: str) -> list[Interconnector]:
    return [ic for ic in INTERCONNECTORS if region in (ic.from_region, ic.to_region)]


def neighbours(region: str) -> list[str]:
    out: list[str] = []
    for ic in interconnectors_for(region):
        other = ic.to_region if ic.from_region == region else ic.from_region
        if other not in out:
            out.append(other)
    return out


def check_region(region: str) -> str:
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}; use one of {list(REGIONS)}")
    return region


# ---- time -----------------------------------------------------------------


def parse_ts(value: str | datetime) -> datetime:
    """Parse a NEM-time timestamp into a naive datetime (NEM time)."""
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(NEM_TZ).replace(tzinfo=None)
        return value
    text = value.strip().replace("T", " ").replace("/", "-")
    if text.endswith("Z"):
        return parse_ts(datetime.fromisoformat(text[:-1]).replace(tzinfo=UTC))
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = datetime.strptime(text, "%Y-%m-%d %H:%M")
    return parse_ts(parsed) if parsed.tzinfo else parsed


def fmt_ts(value: datetime) -> str:
    return value.strftime(TS_FMT)


def nem_now() -> datetime:
    return datetime.now(NEM_TZ).replace(tzinfo=None, microsecond=0)


def floor_interval(value: datetime) -> datetime:
    """Round down to the 5-minute interval that has ended at or before value."""
    return value.replace(minute=value.minute - value.minute % 5, second=0, microsecond=0)


# ---- paths ----------------------------------------------------------------


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    path = Path(os.environ.get("SPIKE_DATA_DIR", repo_root() / "data" / "spike"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def live_db_path() -> Path:
    return data_dir() / "live.duckdb"


def exam_dir() -> Path:
    return repo_root() / "eval" / "spikes"


def investigations_dir() -> Path:
    return Path(os.environ.get("SPIKE_INVESTIGATIONS_DIR", repo_root() / "investigations"))


def health_dir() -> Path:
    return repo_root() / "health"
