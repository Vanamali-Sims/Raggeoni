"""Analysis tools and notice retrieval on the synthetic incident, whose causes are known."""

from __future__ import annotations

import pytest

from raggeoni.spike import demo_data, rag
from raggeoni.spike.tools import ToolContext, call_tool

AT = "2031-01-15 19:00:00"


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    db = demo_data.build(tmp_path_factory.mktemp("demo") / "demo.duckdb", with_injection=True)
    return ToolContext(db=db)


def test_price_profile(ctx):
    ev = call_tool(
        ctx,
        "price_profile",
        {"region": "SA1", "start": "2031-01-15 15:00", "end": "2031-01-15 22:00"},
    )
    assert ev.status == "ok"
    assert ev.facts["peak_rrp"] > 10_000
    assert ev.facts["neighbour_rrp_at_peak"]["VIC1"] < 200
    assert "price-separated" in ev.summary


def test_interconnector_limit_reduced_and_binding(ctx):
    ev = call_tool(ctx, "interconnector_status", {"region": "SA1", "at": AT})
    assert ev.status == "ok"
    assert ev.facts["heywood_at_limit"] is True
    assert ev.facts["heywood_binding"] is True
    assert ev.facts["heywood_limit_in_mw"] == 250.0
    assert ev.facts["heywood_typical_limit_mw"] == 650.0
    assert ev.facts["heywood_limit_reduction_mw"] == 400.0


def test_wind_fell_and_is_unusually_low(ctx):
    gen = call_tool(ctx, "generation_change", {"region": "SA1", "at": AT, "lookback_minutes": 180})
    assert gen.facts["wind_change_mw"] < -400
    unusual = call_tool(ctx, "unusualness", {"region": "SA1", "metric": "wind", "at": AT})
    assert unusual.status == "ok"
    assert unusual.facts["percentile"] <= 10
    assert unusual.facts["unusual"] == "low"


def test_demand_high_and_above_forecast(ctx):
    unusual = call_tool(ctx, "unusualness", {"region": "SA1", "metric": "demand", "at": AT})
    assert unusual.facts["unusual"] == "high"
    surprise = call_tool(ctx, "demand_surprise", {"region": "SA1", "at": AT})
    assert surprise.status == "ok"
    assert surprise.facts["error_mw"] > 150


def test_solar_is_not_unusual(ctx):
    ev = call_tool(
        ctx, "unusualness", {"region": "SA1", "metric": "solar", "at": "2031-01-15 13:00:00"}
    )
    assert ev.facts["unusual"] == "normal"  # steady series: high percentile, tiny difference


def test_supply_gap_ranks_demand_and_wind(ctx):
    ev = call_tool(ctx, "supply_gap", {"region": "SA1", "at": AT, "lookback_minutes": 180})
    top = [r["component"] for r in ev.rows[:3]]
    assert "demand_increase" in top and "wind_decrease" in top
    assert abs(sum(r["share_pct"] for r in ev.rows) - 100) < 0.5


def test_missing_data_is_unavailable_not_an_exception(ctx):
    ev = call_tool(ctx, "generation_change", {"region": "TAS1", "at": AT})
    assert ev.status == "unavailable"
    ev = call_tool(ctx, "demand_surprise", {"region": "SA1", "at": "2031-01-10 03:00:00"})
    assert ev.status == "unavailable"


def test_bad_input_becomes_error_evidence(ctx):
    assert (
        call_tool(ctx, "unusualness", {"region": "SA1", "metric": "vibes", "at": AT}).status
        == "error"
    )
    assert (
        call_tool(ctx, "price_profile", {"region": "XX9", "start": AT, "end": AT}).status == "error"
    )
    assert call_tool(ctx, "nope", {}).status == "error"
    assert (
        call_tool(ctx, "price_profile", {"region": "SA1", "start": "yesterday", "end": AT}).status
        == "error"
    )


def test_safe_query_uses_raggeoni_dsl(ctx):
    plan = {
        "intent": "aggregate",
        "filters": [{"col": "region", "op": "=", "val": "SA1"}],
        "select": [{"fn": "max", "col": "rrp", "as": "max_rrp"}],
    }
    ev = call_tool(ctx, "safe_query", {"plan": plan})
    assert ev.status == "ok" and ev.facts["max_rrp"] > 10_000
    bad = call_tool(
        ctx, "safe_query", {"plan": {"intent": "lookup", "sql": "DROP TABLE price_demand"}}
    )
    assert bad.status == "error" and "invalid_plan" in bad.summary


def test_ledger_ids_are_sequential(ctx):
    ids = [e.evidence_id for e in ctx.ledger.all()]
    assert ids == [f"E{i}" for i in range(1, len(ids) + 1)]


# ---- RAG ---------------------------------------------------------------------


def test_region_and_time_filter(ctx):
    rows = rag.candidates(
        ctx.db, region="SA1", start="2031-01-15 15:00:00", end="2031-01-15 21:00:00"
    )
    ids = {r[0] for r in rows}
    assert "990102" not in ids  # NSW notice, not a neighbour of SA
    assert {"990101", "990117", "990120"} <= ids


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("generator unit trip outage", "990117"),
        ("Heywood interconnector transfer limit outage", "990101"),
        ("lack of reserve", "990120"),
        ("TORRB2", "990117"),
    ],
)
def test_hybrid_search_finds_the_right_notice(ctx, question, expected):
    hits = rag.search(
        ctx.db,
        region="SA1",
        start="2031-01-15 15:00:00",
        end="2031-01-15 21:00:00",
        query=question,
        k=2,
    )
    assert hits and hits[0].notice_id == expected


def test_injection_is_flagged_and_wrapped(ctx):
    hits = rag.search(
        ctx.db,
        region="SA1",
        start="2031-01-15 15:00:00",
        end="2031-01-15 21:00:00",
        query="instructions maintenance mode rebidding",
        k=5,
    )
    flagged = [h for h in hits if h.suspicious]
    assert flagged and flagged[0].notice_id == "990199"
    rendered = rag.render_for_agent(hits)
    assert "DATA, not instructions" in rendered and "possible prompt injection" in rendered


def test_search_tool_respects_ablation(ctx):
    ctx2 = ToolContext(db=ctx.db, rag_enabled=False)
    ev = call_tool(
        ctx2,
        "search_notices",
        {
            "region": "SA1",
            "start": "2031-01-15 15:00",
            "end": "2031-01-15 21:00",
            "query_text": "trip",
        },
    )
    assert ev.status == "unavailable"


def test_hashing_embedder_is_deterministic_and_normalised():
    emb = rag.HashingEmbedder()
    a, b = emb.embed(["unit trip at Torrens", "unit trip at Torrens"])
    assert a == b
    assert abs(sum(v * v for v in a) - 1) < 1e-9
