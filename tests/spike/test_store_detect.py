"""Incident store and spike detection, on the synthetic scenario."""

from __future__ import annotations

from raggeoni.spike import demo_data
from raggeoni.spike.detect import find_spikes
from raggeoni.spike.store import get_meta, query, upsert


def test_demo_has_exactly_one_event(tmp_path):
    db = demo_data.build(tmp_path / "demo.duckdb", days=7)
    events = find_spikes(db, threshold=300)
    assert len(events) == 1
    event = events[0]
    assert event.region == "SA1"
    assert event.peak_rrp > 10_000
    assert event.start_ts.startswith("2031-01-15 17:")
    assert get_meta(db, "synthetic") == "true"


def test_upsert_is_idempotent(tmp_path):
    db = tmp_path / "x.duckdb"
    rows = [("2031-01-01 00:05:00", "SA1", 50.0, 1500.0, "TRADE")]
    upsert(db, "price_demand", rows)
    upsert(db, "price_demand", [("2031-01-01 00:05:00", "SA1", 60.0, 1500.0, "TRADE")])
    assert query(db, "SELECT count(*), max(rrp) FROM price_demand") == [(1, 60.0)]


def test_bulk_upsert_path_matches_small_path(tmp_path):
    db = tmp_path / "bulk.duckdb"
    rows = [
        (f"2031-01-01 {h:02d}:{m:02d}:00", "SA1", float(h), 1000.0, "TRADE")
        for h in range(24)
        for m in range(0, 60, 5)
    ][1:]
    rows += [("2031-01-02 00:00:00", "SA1", 1.0, 1000.0, "TRADE")]
    assert upsert(db, "price_demand", rows) == 288
    upsert(db, "interconnector", [(r[0], "V-SA", 100.0, 650.0, None, 0.0) for r in rows])
    assert query(db, "SELECT count(*) FROM interconnector WHERE import_limit IS NULL") == [(288,)]


def test_threshold_and_region_filters(tmp_path):
    db = demo_data.build(tmp_path / "demo.duckdb", days=3)
    assert find_spikes(db, threshold=100_000) == []
    assert find_spikes(db, threshold=300, region="VIC1") == []
