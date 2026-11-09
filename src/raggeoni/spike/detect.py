"""Find price spikes and group consecutive high-price intervals into events."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from raggeoni.spike.config import SPIKE_THRESHOLD, fmt_ts, parse_ts
from raggeoni.spike.store import query

# Intervals more than this far apart start a new event.
MAX_GAP = timedelta(minutes=30)


@dataclass(frozen=True)
class SpikeEvent:
    event_id: str
    region: str
    start_ts: str
    end_ts: str
    peak_ts: str
    peak_rrp: float
    intervals: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def window(self) -> tuple[datetime, datetime]:
        """Investigation window: 3 hours before the event to 1 hour after."""
        return parse_ts(self.start_ts) - timedelta(hours=3), parse_ts(self.end_ts) + timedelta(
            hours=1
        )


def event_id(region: str, peak_ts: str) -> str:
    stamp = parse_ts(peak_ts).strftime("%Y%m%d-%H%M")
    return f"{region}-{stamp}"


def find_spikes(
    db: Path,
    *,
    threshold: float = SPIKE_THRESHOLD,
    since: str | None = None,
    region: str | None = None,
) -> list[SpikeEvent]:
    sql = "SELECT settlement_ts, region, rrp FROM price_demand WHERE rrp >= ?"
    params: list = [threshold]
    if since:
        sql += " AND settlement_ts >= ?"
        params.append(since)
    if region:
        sql += " AND region = ?"
        params.append(region)
    sql += " ORDER BY region, settlement_ts"
    rows = query(db, sql, params)

    events: list[SpikeEvent] = []
    current: list[tuple[datetime, str, float]] = []

    def flush() -> None:
        if not current:
            return
        peak = max(current, key=lambda r: r[2])
        events.append(
            SpikeEvent(
                event_id=event_id(peak[1], fmt_ts(peak[0])),
                region=peak[1],
                start_ts=fmt_ts(current[0][0]),
                end_ts=fmt_ts(current[-1][0]),
                peak_ts=fmt_ts(peak[0]),
                peak_rrp=round(peak[2], 2),
                intervals=len(current),
            )
        )
        current.clear()

    for ts, reg, rrp in rows:
        if current and (reg != current[-1][1] or ts - current[-1][0] > MAX_GAP):
            flush()
        current.append((ts, reg, rrp))
    flush()
    events.sort(key=lambda e: e.peak_ts)
    return events
