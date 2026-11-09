"""Analysis tools the agents call.

Each tool is plain, deterministic, tested code over the incident database.
Agents decide which tools to run and how to interpret them; the arithmetic
is never left to the language model.

Every tool returns an ``Evidence`` record with an ID and machine-readable
``facts``. Tools never raise to the agent: a missing table or empty window
comes back as ``status="unavailable"`` with a reason, so the agent can say
"no generation data" instead of guessing.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from raggeoni.spike.config import (
    FUEL_GROUPS,
    SPIKE_THRESHOLD,
    check_region,
    fmt_ts,
    interconnectors_for,
    neighbours,
    parse_ts,
)
from raggeoni.spike.evidence import Evidence, Ledger
from raggeoni.spike.store import get_meta, query

TOD_TOLERANCE_MIN = 15


@dataclass
class ToolContext:
    db: Path
    ledger: Ledger = field(default_factory=Ledger)
    rag_enabled: bool = True
    history_dir: Path | None = None


class ToolInputError(ValueError):
    pass


# ---- helpers ---------------------------------------------------------------


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _r(value: float | None, places: int = 1) -> float | None:
    return None if value is None else round(value, places)


def _gap_note(ctx: ToolContext, source: str) -> str:
    gap = get_meta(ctx.db, f"gap:{source}")
    return f" (source gap recorded: {gap})" if gap else ""


def _window_avg(
    ctx: ToolContext, sql: str, params: list[Any], at: datetime, n: int = 3
) -> float | None:
    """Average of the n intervals ending at ``at`` (inclusive)."""
    lo = fmt_ts(at - timedelta(minutes=5 * (n - 1)))
    rows = query(ctx.db, sql + " AND settlement_ts BETWEEN ? AND ?", [*params, lo, fmt_ts(at)])
    return _avg([float(r[0]) for r in rows if r[0] is not None])


def _same_time_history(
    ctx: ToolContext,
    sql_value: str,
    table: str,
    where: str,
    params: list[Any],
    at: datetime,
    days: int,
) -> list[float]:
    """Values at the same time of day (±15 min) on each of the previous ``days`` days."""
    minute_of_day = at.hour * 60 + at.minute
    lo = fmt_ts(at - timedelta(days=days, minutes=TOD_TOLERANCE_MIN))
    hi = fmt_ts(at - timedelta(hours=12))
    rows = query(
        ctx.db,
        f"""
        SELECT date_trunc('day', settlement_ts) AS d, avg({sql_value})
        FROM {table}
        WHERE {where} AND settlement_ts BETWEEN ? AND ?
          AND abs((hour(settlement_ts) * 60 + minute(settlement_ts)) - ?) <= ?
        GROUP BY d
        """,
        [*params, lo, hi, minute_of_day, TOD_TOLERANCE_MIN],
    )
    return [float(r[1]) for r in rows if r[1] is not None]


def _percentile_of(value: float, sample: list[float]) -> float:
    below = sum(1 for s in sample if s < value)
    equal = sum(1 for s in sample if s == value)
    return round(100 * (below + 0.5 * equal) / len(sample), 1)


# ---- tools -----------------------------------------------------------------


def price_profile(ctx: ToolContext, region: str, start: str, end: str) -> Evidence:
    """Peak price, how long prices stayed high, and neighbouring prices at the peak."""
    t0 = time.perf_counter()
    args = {"region": region, "start": start, "end": end}
    check_region(region)
    rows = query(
        ctx.db,
        "SELECT settlement_ts, rrp, total_demand FROM price_demand "
        "WHERE region = ? AND settlement_ts BETWEEN ? AND ? ORDER BY settlement_ts",
        [region, start, end],
    )
    if not rows:
        return ctx.ledger.add(
            "price_profile",
            args,
            status="unavailable",
            summary=f"No prices for {region} in window" + _gap_note(ctx, "dispatch"),
            started=t0,
        )
    peak = max(rows, key=lambda r: r[1])
    high = [r for r in rows if r[1] >= SPIKE_THRESHOLD]
    peak_ts = fmt_ts(peak[0])
    neigh = (
        query(
            ctx.db,
            f"SELECT region, rrp FROM price_demand WHERE settlement_ts = ? AND region IN ({','.join('?' * len(neighbours(region)))})",
            [peak_ts, *neighbours(region)],
        )
        if neighbours(region)
        else []
    )
    neighbour_prices = {r[0]: _r(r[1], 2) for r in neigh}
    facts: dict[str, Any] = {
        "peak_rrp": _r(peak[1], 2),
        "peak_ts": peak_ts,
        "demand_at_peak_mw": _r(peak[2]),
        "intervals_above_threshold": len(high),
        "threshold": SPIKE_THRESHOLD,
        "first_high_ts": fmt_ts(high[0][0]) if high else None,
        "last_high_ts": fmt_ts(high[-1][0]) if high else None,
        "mean_rrp_window": _r(_avg([r[1] for r in rows]), 2),
        "neighbour_rrp_at_peak": neighbour_prices,
    }
    separated = [n for n, p in neighbour_prices.items() if p is not None and p < peak[1] / 5]
    summary = (
        f"{region} peaked at ${facts['peak_rrp']:,.2f}/MWh at {peak_ts}; "
        f"{len(high)} intervals at or above ${SPIKE_THRESHOLD:,.0f}."
    )
    if separated:
        summary += f" Neighbours {', '.join(separated)} stayed far cheaper, so the region was price-separated."
    return ctx.ledger.add(
        "price_profile", args, status="ok", summary=summary, facts=facts, started=t0
    )


def generation_change(
    ctx: ToolContext, region: str, at: str, lookback_minutes: int = 120
) -> Evidence:
    """Output by fuel group at ``at`` versus ``lookback_minutes`` earlier (3-interval averages)."""
    t0 = time.perf_counter()
    args = {"region": region, "at": at, "lookback_minutes": lookback_minutes}
    check_region(region)
    at_dt = parse_ts(at)
    before_dt = at_dt - timedelta(minutes=lookback_minutes)
    out_rows: list[dict[str, Any]] = []
    for group in FUEL_GROUPS:
        sql = "SELECT mw FROM generation WHERE region = ? AND fueltech_group = ?"
        now = _window_avg(ctx, sql, [region, group], at_dt)
        then = _window_avg(ctx, sql, [region, group], before_dt)
        if now is None and then is None:
            continue
        change = None if now is None or then is None else now - then
        pct = None if change is None or not then else 100 * change / abs(then)
        out_rows.append(
            {
                "fueltech_group": group,
                "mw_before": _r(then),
                "mw_at": _r(now),
                "change_mw": _r(change),
                "change_pct": _r(pct),
            }
        )
    if not out_rows:
        return ctx.ledger.add(
            "generation_change",
            args,
            status="unavailable",
            summary=f"No generation data for {region}" + _gap_note(ctx, "generation"),
            started=t0,
        )
    out_rows.sort(key=lambda r: abs(r["change_mw"] or 0), reverse=True)
    biggest = out_rows[0]
    facts = {f"{r['fueltech_group']}_change_mw": r["change_mw"] for r in out_rows}
    facts.update({f"{r['fueltech_group']}_mw_at": r["mw_at"] for r in out_rows})
    facts.update({f"{r['fueltech_group']}_mw_before": r["mw_before"] for r in out_rows})
    facts.update({f"{r['fueltech_group']}_change_pct": r["change_pct"] for r in out_rows})
    summary = (
        f"Largest change in {region} over {lookback_minutes} min to {at}: "
        f"{biggest['fueltech_group']} {biggest['mw_before']} -> {biggest['mw_at']} MW ({biggest['change_mw']:+} MW)."
    )
    return ctx.ledger.add(
        "generation_change",
        args,
        status="ok",
        summary=summary,
        facts=facts,
        rows=out_rows,
        started=t0,
    )


def unusualness(ctx: ToolContext, region: str, metric: str, at: str, days: int = 30) -> Evidence:
    """Compare a value at ``at`` with the same time of day over the previous ``days`` days.

    ``metric`` is ``demand``, ``price``, or a fuel group such as ``wind``.
    """
    t0 = time.perf_counter()
    args = {"region": region, "metric": metric, "at": at, "days": days}
    check_region(region)
    at_dt = parse_ts(at)
    if metric in ("demand", "price"):
        col = "total_demand" if metric == "demand" else "rrp"
        table, where, params = "price_demand", "region = ?", [region]
    elif metric in FUEL_GROUPS:
        col, table, where, params = (
            "mw",
            "generation",
            "region = ? AND fueltech_group = ?",
            [region, metric],
        )
    else:
        raise ToolInputError(f"metric must be demand, price or one of {list(FUEL_GROUPS)}")
    current = _window_avg(ctx, f"SELECT {col} FROM {table} WHERE {where}", params, at_dt)
    history = _same_time_history(ctx, col, table, where, params, at_dt, days)
    if current is None or len(history) < 5:
        return ctx.ledger.add(
            "unusualness",
            args,
            status="unavailable",
            summary=f"Not enough {metric} history for {region} (have {len(history)} days)"
            + _gap_note(ctx, "generation" if table == "generation" else "dispatch"),
            started=t0,
        )
    mean = sum(history) / len(history)
    std = (sum((h - mean) ** 2 for h in history) / max(1, len(history) - 1)) ** 0.5
    z = (current - mean) / std if std > 0 else 0.0
    pct = _percentile_of(current, history)
    # Percentile alone flags tiny wobbles in very steady series (solar on clear
    # days), so a value must also be materially different to count.
    rel = abs(current - mean) / abs(mean) if mean else float("inf")
    material = abs(z) >= 1.5 and rel >= 0.10
    if material and pct >= 90:
        label, unusual = "unusually high", "high"
    elif material and pct <= 10:
        label, unusual = "unusually low", "low"
    else:
        label, unusual = "within the normal range", "normal"
    facts = {
        "unusual": unusual,
        "value": _r(current),
        "typical_mean": _r(mean),
        "typical_std": _r(std),
        "z_score": _r(z, 2),
        "percentile": pct,
        "days_compared": len(history),
        "difference_from_typical": _r(current - mean),
    }
    summary = (
        f"{metric} in {region} at {at} was {facts['value']} vs a typical {facts['typical_mean']} for this time of day "
        f"over {len(history)} days (percentile {pct}, z {facts['z_score']}): {label}."
    )
    return ctx.ledger.add(
        "unusualness", args, status="ok", summary=summary, facts=facts, started=t0
    )


def demand_surprise(ctx: ToolContext, region: str, at: str, lead_minutes: int = 30) -> Evidence:
    """Actual demand versus AEMO's 5-minute-ahead (P5MIN) forecast made ``lead_minutes`` earlier."""
    t0 = time.perf_counter()
    args = {"region": region, "at": at, "lead_minutes": lead_minutes}
    check_region(region)
    at_dt = parse_ts(at)
    run = fmt_ts(at_dt - timedelta(minutes=lead_minutes))
    fc = query(
        ctx.db,
        "SELECT forecast_demand, run_ts FROM forecast WHERE region = ? AND interval_ts = ? "
        "AND run_ts <= ? ORDER BY run_ts DESC LIMIT 1",
        [region, at, run],
    )
    actual = query(
        ctx.db,
        "SELECT total_demand FROM price_demand WHERE region = ? AND settlement_ts = ?",
        [region, at],
    )
    if not fc or fc[0][0] is None or not actual:
        return ctx.ledger.add(
            "demand_surprise",
            args,
            status="unavailable",
            summary="No P5MIN forecast available for that interval" + _gap_note(ctx, "forecast"),
            started=t0,
        )
    forecast, run_ts = float(fc[0][0]), fmt_ts(fc[0][1])
    error = float(actual[0][0]) - forecast
    pct = 100 * error / forecast if forecast else 0.0
    facts = {
        "actual_demand_mw": _r(float(actual[0][0])),
        "forecast_demand_mw": _r(forecast),
        "forecast_run_ts": run_ts,
        "error_mw": _r(error),
        "error_pct": _r(pct),
    }
    verdict = "higher than forecast" if error > 0 else "lower than forecast"
    summary = f"Demand in {region} at {at} was {facts['error_mw']:+} MW ({facts['error_pct']:+}%) {verdict} by the {run_ts} run."
    return ctx.ledger.add(
        "demand_surprise", args, status="ok", summary=summary, facts=facts, started=t0
    )


def interconnector_status(ctx: ToolContext, region: str, at: str, days: int = 30) -> Evidence:
    """Flows into ``region`` at ``at``, whether each link was at its limit, and if the limit was reduced."""
    t0 = time.perf_counter()
    args = {"region": region, "at": at}
    check_region(region)
    out: list[dict[str, Any]] = []
    for ic in interconnectors_for(region):
        row = query(
            ctx.db,
            "SELECT mw_flow, export_limit, import_limit, marginal_value FROM interconnector "
            "WHERE interconnector_id = ? AND settlement_ts = ?",
            [ic.id, at],
        )
        if not row:
            continue
        flow, exp_lim, imp_lim, mv = row[0]
        into = ic.to_region == region
        sign = 1 if into else -1
        flow_in = sign * (flow or 0.0)
        limit_in = exp_lim if into else (-imp_lim if imp_lim is not None else None)
        typical = query(
            ctx.db,
            f"SELECT median({'export_limit' if into else '-import_limit'}) FROM interconnector "
            "WHERE interconnector_id = ? AND settlement_ts BETWEEN ? AND ?",
            [
                ic.id,
                fmt_ts(parse_ts(at) - timedelta(days=days)),
                fmt_ts(parse_ts(at) - timedelta(days=1)),
            ],
        )
        typical_limit = typical[0][0] if typical and typical[0][0] is not None else None
        headroom = None if limit_in is None else limit_in - flow_in
        out.append(
            {
                "interconnector": ic.id,
                "name": ic.name,
                "flow_into_region_mw": _r(flow_in),
                "import_limit_into_region_mw": _r(limit_in),
                "typical_limit_mw": _r(typical_limit),
                "limit_reduction_mw": _r(typical_limit - limit_in)
                if typical_limit is not None and limit_in is not None
                else None,
                "headroom_mw": _r(headroom),
                "at_limit": headroom is not None
                and headroom <= max(5.0, 0.02 * abs(limit_in or 0)),
                "constraint_binding": bool(mv),
            }
        )
    if not out:
        return ctx.ledger.add(
            "interconnector_status",
            args,
            status="unavailable",
            summary=f"No interconnector data for {region} at {at}" + _gap_note(ctx, "dispatch"),
            started=t0,
        )
    facts: dict[str, Any] = {}
    parts = []
    for r in out:
        key = r["name"].lower().replace("-", "_")
        facts[f"{key}_flow_in_mw"] = r["flow_into_region_mw"]
        facts[f"{key}_limit_in_mw"] = r["import_limit_into_region_mw"]
        facts[f"{key}_typical_limit_mw"] = r["typical_limit_mw"]
        facts[f"{key}_limit_reduction_mw"] = r["limit_reduction_mw"]
        facts[f"{key}_at_limit"] = r["at_limit"]
        facts[f"{key}_binding"] = r["constraint_binding"]
        state = "AT LIMIT" if r["at_limit"] else "headroom"
        reduced = (
            f", limit {r['limit_reduction_mw']} MW below typical"
            if (r["limit_reduction_mw"] or 0) > 20
            else ""
        )
        parts.append(
            f"{r['name']} {r['flow_into_region_mw']} MW in of {r['import_limit_into_region_mw']} ({state}{reduced})"
        )
    facts["total_import_mw"] = _r(sum(r["flow_into_region_mw"] or 0 for r in out))
    summary = f"Into {region} at {at}: " + "; ".join(parts) + "."
    return ctx.ledger.add(
        "interconnector_status",
        args,
        status="ok",
        summary=summary,
        facts=facts,
        rows=out,
        started=t0,
    )


def supply_gap(ctx: ToolContext, region: str, at: str, lookback_minutes: int = 120) -> Evidence:
    """Break the change in the supply-demand balance into its parts.

    Over the lookback, demand rose by D and each source changed by S_i. The
    balance worsened by D - sum(S_i). Each worsening part (demand up, a fuel
    down, imports down) is reported in MW and as a share of the total, which
    turns "a mix of things" into numbers.
    """
    t0 = time.perf_counter()
    args = {"region": region, "at": at, "lookback_minutes": lookback_minutes}
    check_region(region)
    at_dt = parse_ts(at)
    before = at_dt - timedelta(minutes=lookback_minutes)
    dsql = "SELECT total_demand FROM price_demand WHERE region = ?"
    d_now, d_then = (
        _window_avg(ctx, dsql, [region], at_dt),
        _window_avg(ctx, dsql, [region], before),
    )
    if d_now is None or d_then is None:
        return ctx.ledger.add(
            "supply_gap",
            args,
            status="unavailable",
            summary="No demand data for the lookback" + _gap_note(ctx, "dispatch"),
            started=t0,
        )
    pressures: dict[str, float] = {"demand_increase": d_now - d_then}
    for group in FUEL_GROUPS:
        sql = "SELECT mw FROM generation WHERE region = ? AND fueltech_group = ?"
        g_now, g_then = (
            _window_avg(ctx, sql, [region, group], at_dt),
            _window_avg(ctx, sql, [region, group], before),
        )
        if g_now is not None and g_then is not None:
            pressures[f"{group}_decrease"] = g_then - g_now
    isql = "SELECT -net_interchange FROM region_supply WHERE region = ?"
    i_now, i_then = (
        _window_avg(ctx, isql, [region], at_dt),
        _window_avg(ctx, isql, [region], before),
    )
    if i_now is not None and i_then is not None:
        pressures["import_decrease"] = i_then - i_now
    worsening = {k: v for k, v in pressures.items() if v > 1}
    total = sum(worsening.values())
    rows = [
        {"component": k, "mw": _r(v), "share_pct": _r(100 * v / total) if total else None}
        for k, v in sorted(worsening.items(), key=lambda kv: kv[1], reverse=True)
    ]
    facts: dict[str, Any] = {f"{r['component']}_mw": r["mw"] for r in rows}
    facts.update({f"{r['component']}_share_pct": r["share_pct"] for r in rows})
    facts["total_pressure_mw"] = _r(total)
    facts["generation_data"] = any(
        k.endswith("_decrease") and k not in ("import_decrease",) for k in pressures
    )
    if not rows:
        summary = f"No component worsened the balance in {region} over {lookback_minutes} min."
    else:
        top = ", ".join(f"{r['component']} {r['mw']} MW ({r['share_pct']}%)" for r in rows[:4])
        summary = f"Balance in {region} worsened by {facts['total_pressure_mw']} MW over {lookback_minutes} min: {top}."
    return ctx.ledger.add(
        "supply_gap", args, status="ok", summary=summary, facts=facts, rows=rows, started=t0
    )


def safe_query(ctx: ToolContext, plan: dict[str, Any]) -> Evidence:
    """Run a Raggeoni query plan against price_demand. The model never writes SQL."""
    from raggeoni.execute import execute_plan

    t0 = time.perf_counter()
    result = execute_plan(plan, db=ctx.db)
    if result.status == "error":
        return ctx.ledger.add(
            "safe_query",
            {"plan": plan},
            status="error",
            summary=f"{result.error_class}: {result.error}",
            started=t0,
        )
    if result.status == "refuse":
        return ctx.ledger.add(
            "safe_query", {"plan": plan}, status="ok", summary="Plan was a refusal.", started=t0
        )
    rows = [dict(zip(result.columns, row, strict=True)) for row in result.rows]
    facts: dict[str, Any] = {"row_count": len(rows)}
    if len(rows) == 1:
        facts.update(rows[0])
    return ctx.ledger.add(
        "safe_query",
        {"plan": plan},
        status="ok",
        summary=f"{len(rows)} row(s).",
        facts=facts,
        rows=rows,
        started=t0,
    )


def search_notices(
    ctx: ToolContext, region: str, start: str, end: str, query_text: str, k: int = 5
) -> Evidence:
    """RAG over AEMO market notices, filtered to the time window and region (+ neighbours)."""
    from raggeoni.spike import rag

    t0 = time.perf_counter()
    args = {"region": region, "start": start, "end": end, "query": query_text, "k": k}
    if not ctx.rag_enabled:
        return ctx.ledger.add(
            "search_notices",
            args,
            status="unavailable",
            summary="Notice search is disabled in this run (ablation).",
            started=t0,
        )
    hits = rag.search(ctx.db, region=region, start=start, end=end, query=query_text, k=k)
    if not hits:
        return ctx.ledger.add(
            "search_notices",
            args,
            status="ok",
            summary="No relevant market notices in the window" + _gap_note(ctx, "notices"),
            facts={"hits": 0},
            started=t0,
        )
    rows = [h.to_row() for h in hits]
    facts = {"hits": len(hits), "notice_ids": [h.notice_id for h in hits]}
    summary = "Relevant notices: " + "; ".join(
        f"#{h.notice_id} {h.issued_ts} {h.external_ref[:90]}" for h in hits
    )
    return ctx.ledger.add(
        "search_notices", args, status="ok", summary=summary, facts=facts, rows=rows, started=t0
    )


def similar_events(
    ctx: ToolContext, region: str, features: dict[str, float], k: int = 3
) -> Evidence:
    """Find past investigations with the most similar conditions (numeric nearest neighbours)."""
    t0 = time.perf_counter()
    args = {"region": region, "features": features, "k": k}
    if ctx.history_dir is None or not ctx.history_dir.exists():
        return ctx.ledger.add(
            "similar_events",
            args,
            status="unavailable",
            summary="No past investigations indexed yet.",
            started=t0,
        )
    candidates: list[tuple[float, dict[str, Any]]] = []
    for path in ctx.history_dir.glob("*/report.json"):
        try:
            past = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        past_features = past.get("features") or {}
        shared = [key for key in features if key in past_features]
        if not shared:
            continue
        dist = (
            sum(
                (
                    (features[key] - past_features[key])
                    / (abs(features[key]) + abs(past_features[key]) + 1)
                )
                ** 2
                for key in shared
            )
            ** 0.5
        )
        if past.get("region") != region:
            dist += 0.5
        candidates.append((dist, past))
    candidates.sort(key=lambda c: c[0])
    rows = [
        {
            "event_id": p["event_id"],
            "similarity": _r(1 / (1 + d), 3),
            "headline": p.get("headline", ""),
            "causes": [c["cause"] for c in p.get("causes", [])],
        }
        for d, p in candidates[:k]
    ]
    if not rows:
        return ctx.ledger.add(
            "similar_events",
            args,
            status="ok",
            summary="No comparable past events.",
            facts={"matches": 0},
            started=t0,
        )
    summary = "Most similar past events: " + "; ".join(
        f"{r['event_id']} ({', '.join(r['causes'])})" for r in rows
    )
    return ctx.ledger.add(
        "similar_events",
        args,
        status="ok",
        summary=summary,
        facts={"matches": len(rows)},
        rows=rows,
        started=t0,
    )


# ---- registry used by the agents ---------------------------------------------------

ToolFn = Callable[..., Evidence]

TOOL_SPECS: dict[str, dict[str, Any]] = {
    "price_profile": {
        "fn": price_profile,
        "description": "Peak price, duration above the spike threshold, and neighbouring regions' prices at the peak.",
        "params": {"region": "string", "start": "timestamp", "end": "timestamp"},
    },
    "generation_change": {
        "fn": generation_change,
        "description": "Output by fuel group (wind, solar, gas, coal, hydro, batteries) at a time versus N minutes earlier.",
        "params": {
            "region": "string",
            "at": "timestamp",
            "lookback_minutes": "integer (default 120)",
        },
    },
    "unusualness": {
        "fn": unusualness,
        "description": "Is a value abnormal for this time of day? Compares with the same time over the previous 30 days (mean, z-score, percentile). metric: demand, price, or a fuel group like wind.",
        "params": {
            "region": "string",
            "metric": "string",
            "at": "timestamp",
            "days": "integer (default 30)",
        },
    },
    "demand_surprise": {
        "fn": demand_surprise,
        "description": "Actual demand vs AEMO's 5-minute-ahead forecast made 30 minutes earlier. Shows whether the market was caught off guard.",
        "params": {"region": "string", "at": "timestamp", "lead_minutes": "integer (default 30)"},
    },
    "interconnector_status": {
        "fn": interconnector_status,
        "description": "Flows into the region on each interconnector, whether each was at its limit, whether a constraint was binding, and whether the limit was reduced versus normal.",
        "params": {"region": "string", "at": "timestamp"},
    },
    "supply_gap": {
        "fn": supply_gap,
        "description": "Breaks the worsening of the supply-demand balance over the lookback into demand increase, each fuel's decrease and import decrease, in MW and % share.",
        "params": {
            "region": "string",
            "at": "timestamp",
            "lookback_minutes": "integer (default 120)",
        },
    },
    "search_notices": {
        "fn": search_notices,
        "description": "Search AEMO market notices (outages, unit trips, network limits, reserve warnings) in the time window for the region and its neighbours. Use for causes the numbers cannot show.",
        "params": {
            "region": "string",
            "start": "timestamp",
            "end": "timestamp",
            "query_text": "string",
            "k": "integer (default 5)",
        },
    },
    "safe_query": {
        "fn": safe_query,
        "description": "Ad-hoc question about prices or demand, written as a Raggeoni JSON query plan (intents: lookup, aggregate, compare). Never SQL.",
        "params": {"plan": "object"},
    },
    "similar_events": {
        "fn": similar_events,
        "description": "Past investigations with similar conditions. features: numeric dict such as {'peak_rrp': 12000, 'wind_change_mw': -600}.",
        "params": {"region": "string", "features": "object", "k": "integer (default 3)"},
    },
}


def call_tool(ctx: ToolContext, name: str, arguments: dict[str, Any]) -> Evidence:
    """Run a tool by name. Bad input becomes an error Evidence, never an exception."""
    spec = TOOL_SPECS.get(name)
    if spec is None:
        return ctx.ledger.add(
            name,
            arguments,
            status="error",
            summary=f"Unknown tool {name!r}. Available: {sorted(TOOL_SPECS)}",
        )
    allowed = set(spec["params"])
    clean = {k: v for k, v in arguments.items() if k in allowed}
    for key in ("at", "start", "end"):
        if key in clean and isinstance(clean[key], str):
            try:
                clean[key] = fmt_ts(parse_ts(clean[key]))
            except ValueError:
                return ctx.ledger.add(
                    name,
                    arguments,
                    status="error",
                    summary=f"Bad timestamp for {key}: {clean[key]!r}. Use YYYY-MM-DD HH:MM:SS in NEM time.",
                )
    try:
        return spec["fn"](ctx, **clean)
    except (ToolInputError, ValueError, TypeError) as exc:
        return ctx.ledger.add(
            name, arguments, status="error", summary=f"{type(exc).__name__}: {exc}"
        )
    except Exception as exc:  # noqa: BLE001 - a tool must never crash the investigation
        return ctx.ledger.add(
            name, arguments, status="error", summary=f"tool failed: {type(exc).__name__}: {exc}"
        )
