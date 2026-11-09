"""Parsers for NEMWeb, Open Electricity and market notices.

Fixtures are small hand-written files in each source's published format, so
these tests run offline. Live downloads are exercised by the scheduled
GitHub Actions workflows.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from raggeoni.spike.sources import nemweb
from raggeoni.spike.sources.mms import parse_mms
from raggeoni.spike.sources.notices import parse_notice, regions_in
from raggeoni.spike.sources.openelectricity import parse_generation

FIX = Path(__file__).parent / "fixtures"


def _dispatch_tables():
    return parse_mms((FIX / "dispatchis_sample.csv").read_text())


def test_mms_splits_tables():
    tables = _dispatch_tables()
    assert set(tables) == {"DISPATCH.PRICE", "DISPATCH.REGIONSUM", "DISPATCH.INTERCONNECTORRES"}
    assert len(tables["DISPATCH.PRICE"]) == 4


def test_price_demand_drops_intervention_runs():
    rows = nemweb.price_demand_rows(_dispatch_tables())
    sa = [r for r in rows if r[1] == "SA1"]
    assert sa == [("2026-01-26 18:30:00", "SA1", 15230.5, 3105.8, "TRADE")]
    assert len(rows) == 3


def test_interconnector_import_limit_is_non_positive_and_binding_flagged():
    rows = {r[1]: r for r in nemweb.interconnector_rows(_dispatch_tables())}
    heywood = rows["V-SA"]
    assert heywood[2] == 500.0 and heywood[3] == 500.0
    assert heywood[4] == -600.0
    assert heywood[5] != 0.0  # marginal value: constraint binding


def test_p5min_forecast_rows():
    rows = nemweb.p5min_forecast_rows(parse_mms((FIX / "p5min_sample.csv").read_text()))
    assert ("2026-01-26 18:00:00", "2026-01-26 18:30:00", "SA1", 2945.1, 980.2) in rows


def test_nested_archive_zip_is_unpacked():
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr(
            "PUBLIC_DISPATCHIS_202601261830.CSV", (FIX / "dispatchis_sample.csv").read_text()
        )
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("PUBLIC_DISPATCHIS_202601261830_0000000500000001.zip", inner.getvalue())
    texts = list(nemweb.iter_csv_texts(outer.getvalue()))
    assert len(texts) == 1 and "DISPATCH,PRICE" in texts[0]


def test_days_between_includes_previous_day_for_midnight_interval():
    from datetime import datetime

    days = nemweb.days_between(datetime(2026, 1, 26, 0, 5), datetime(2026, 1, 26, 23, 0))
    assert [d.isoformat() for d in days] == ["2026-01-26"]
    days = nemweb.days_between(datetime(2026, 1, 26, 0, 0), datetime(2026, 1, 26, 1, 0))
    assert [d.isoformat() for d in days] == ["2026-01-25", "2026-01-26"]


def test_open_electricity_parse_keeps_region_and_sign():
    payload = json.loads((FIX / "oe_generation_sample.json").read_text())
    rows = parse_generation(payload, "SA1")
    assert ("2026-01-26 18:30:00", "SA1", "wind", 329.8) in rows
    assert ("2026-01-26 18:30:00", "SA1", "battery_charging", -12.0) in rows
    assert all(r[1] == "SA1" for r in rows)
    assert len(rows) == 3  # null point dropped, VIC1 series ignored


def test_market_notice_parse():
    notice = parse_notice((FIX / "market_notice_sample.txt").read_text())
    assert notice is not None
    assert notice.notice_id == "131302"
    assert notice.issued_ts == "2026-01-26 20:31:44"
    assert notice.notice_type == "RESERVE NOTICE"
    assert "LOR1" in notice.body
    assert notice.regions == ("SA1", "VIC1")  # SA plus Heywood's two ends


def test_region_tagging_does_not_match_inside_words():
    assert regions_in("SAFETY check at the VICTORIAN depot") == ()
    assert regions_in("Basslink outage") == ("TAS1", "VIC1")
