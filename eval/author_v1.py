"""Author the v1 eval set (200 questions). Hydrate fills gold_result."""

from __future__ import annotations

import json
import sys
from calendar import monthrange
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "v1.jsonl"


def ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def month_window(year: int, month: int) -> tuple[str, str]:
    start = datetime(year, month, 1, 0, 5, 0)
    if month == 12:
        end = datetime(year + 1, 1, 1, 0, 0, 0)
    else:
        end = datetime(year, month + 1, 1, 0, 0, 0)
    return ts(start), ts(end)


def day_window(year: int, month: int, day: int) -> tuple[str, str]:
    start = datetime(year, month, day, 0, 5, 0)
    end = datetime(year, month, day) + timedelta(days=1)
    return ts(start), ts(end)


def q(
    ident: str,
    bucket: str,
    question: str,
    plan: dict,
    notes: str = "",
) -> dict:
    return {
        "id": ident,
        "bucket": bucket,
        "question": question,
        "gold_plan": plan,
        "gold_result": None,
        "notes": notes,
    }


def filt(*items: dict) -> list[dict]:
    return list(items)


def region_eq(code: str) -> dict:
    return {"col": "region", "op": "=", "val": code}


def ts_eq(value: str) -> dict:
    return {"col": "settlement_ts", "op": "=", "val": value}


def ts_ge(value: str) -> dict:
    return {"col": "settlement_ts", "op": ">=", "val": value}


def ts_le(value: str) -> dict:
    return {"col": "settlement_ts", "op": "<=", "val": value}


def ts_range(lo: str, hi: str) -> list[dict]:
    return [ts_ge(lo), ts_le(hi)]


def lookup(filters: list[dict], columns: list[str], limit: int = 1, order_by=None) -> dict:
    plan: dict = {
        "intent": "lookup",
        "filters": filters,
        "columns": columns,
        "limit": limit,
    }
    if order_by:
        plan["order_by"] = order_by
    return plan


def agg(select: list[dict], filters: list[dict] | None = None, group_by=None, limit=None, order_by=None) -> dict:
    plan: dict = {"intent": "aggregate", "select": select}
    if filters:
        plan["filters"] = filters
    if group_by:
        plan["group_by"] = group_by
        plan["limit"] = limit if limit is not None else 5
    elif limit is not None:
        plan["limit"] = limit
    if order_by:
        plan["order_by"] = order_by
    return plan


def compare(metric: dict, left: tuple[str, list[dict]], right: tuple[str, list[dict]]) -> dict:
    return {
        "intent": "compare",
        "metric": metric,
        "sides": [
            {"name": left[0], "filters": left[1]},
            {"name": right[0], "filters": right[1]},
        ],
    }


def refuse(reason: str) -> dict:
    return {"intent": "refuse", "reason": reason}


def assign_splits(rows: list[dict]) -> None:
    holdout_n = {"structural": 8, "factual": 24, "out-of-scope": 8}
    counts = {"structural": 0, "factual": 0, "out-of-scope": 0}
    totals = {"structural": 0, "factual": 0, "out-of-scope": 0}
    for row in rows:
        totals[row["bucket"]] += 1
    eval_n = {k: totals[k] - holdout_n[k] for k in totals}
    for row in rows:
        bucket = row["bucket"]
        counts[bucket] += 1
        row["split"] = "eval" if counts[bucket] <= eval_n[bucket] else "holdout"


def structural() -> list[dict]:
    aug_lo, aug_hi = month_window(2025, 8)
    jan_lo, jan_hi = month_window(2026, 1)
    feb_lo, feb_hi = month_window(2026, 2)
    jul_lo, jul_hi = month_window(2026, 7)
    d_open_lo, d_open_hi = day_window(2025, 8, 1)
    d_close_lo, d_close_hi = day_window(2026, 7, 31)
    d_feb_lo, d_feb_hi = day_window(2026, 2, 28)
    notes_grain = "5-minute interval-ending timestamps; 288 intervals per calendar day."
    rows = [
        q("struct-001", "structural", "Which NEM regions appear in the price_demand table?",
          agg([{"fn": "count", "as": "n"}], group_by=["region"], limit=5, order_by=[{"col": "region", "dir": "asc"}]),
          "Five region codes, not NSW/VIC short names."),
        q("struct-002", "structural", "How many rows are in the frozen price_demand table?",
          agg([{"fn": "count", "as": "n"}]), "Expect 525600."),
        q("struct-003", "structural", "How many rows are there for NSW1?",
          agg([{"fn": "count", "as": "n"}], [region_eq("NSW1")])),
        q("struct-004", "structural", "How many rows are there for QLD1?",
          agg([{"fn": "count", "as": "n"}], [region_eq("QLD1")])),
        q("struct-005", "structural", "How many rows are there for VIC1?",
          agg([{"fn": "count", "as": "n"}], [region_eq("VIC1")])),
        q("struct-006", "structural", "How many rows are there for SA1?",
          agg([{"fn": "count", "as": "n"}], [region_eq("SA1")])),
        q("struct-007", "structural", "How many rows are there for TAS1?",
          agg([{"fn": "count", "as": "n"}], [region_eq("TAS1")])),
        q("struct-008", "structural", "What is the earliest settlement timestamp for NSW1 in the freeze?",
          lookup([region_eq("NSW1")], ["settlement_ts"], 1, [{"col": "settlement_ts", "dir": "asc"}]),
          "Should be 2025-08-01 00:05:00."),
        q("struct-009", "structural", "What is the latest settlement timestamp for VIC1 in the freeze?",
          lookup([region_eq("VIC1")], ["settlement_ts"], 1, [{"col": "settlement_ts", "dir": "desc"}]),
          "Should be 2026-08-01 00:00:00."),
        q("struct-010", "structural", "What is the first TAS1 interval in the freeze?",
          lookup([region_eq("TAS1")], ["settlement_ts", "region"], 1, [{"col": "settlement_ts", "dir": "asc"}])),
        q("struct-011", "structural", "What is the last QLD1 interval in the freeze?",
          lookup([region_eq("QLD1")], ["settlement_ts", "region"], 1, [{"col": "settlement_ts", "dir": "desc"}])),
        q("struct-012", "structural", "How many 5-minute intervals does NSW1 have on 2025-08-01?",
          agg([{"fn": "count", "as": "n"}], [region_eq("NSW1"), *ts_range(d_open_lo, d_open_hi)]), notes_grain),
        q("struct-013", "structural", "How many 5-minute intervals does VIC1 have on 2026-07-31?",
          agg([{"fn": "count", "as": "n"}], [region_eq("VIC1"), *ts_range(d_close_lo, d_close_hi)]), notes_grain),
        q("struct-014", "structural", "How many intervals does QLD1 have on 2026-02-28?",
          agg([{"fn": "count", "as": "n"}], [region_eq("QLD1"), *ts_range(d_feb_lo, d_feb_hi)]),
          "2026 is not a leap year."),
        q("struct-015", "structural", "Give one interval where RRP was negative.",
          lookup([{"col": "rrp", "op": "<", "val": 0}], ["settlement_ts", "region", "rrp"], 1,
                 [{"col": "settlement_ts", "dir": "asc"}]),
          "NEM negative prices must exist in this window."),
        q("struct-016", "structural", "How many rows in the freeze have negative RRP?",
          agg([{"fn": "count", "as": "n"}], [{"col": "rrp", "op": "<", "val": 0}])),
        q("struct-017", "structural", "How many SA1 intervals have negative RRP?",
          agg([{"fn": "count", "as": "n"}], [region_eq("SA1"), {"col": "rrp", "op": "<", "val": 0}])),
        q("struct-018", "structural", "What period_type values appear in the table, and how many rows each?",
          agg([{"fn": "count", "as": "n"}], group_by=["period_type"], limit=5, order_by=[{"col": "period_type", "dir": "asc"}]),
          "Expect only TRADE."),
        q("struct-019", "structural", "What was the minimum RRP anywhere in the freeze?",
          agg([{"fn": "min", "col": "rrp", "as": "min_rrp"}])),
        q("struct-020", "structural", "What was the maximum RRP anywhere in the freeze?",
          agg([{"fn": "max", "col": "rrp", "as": "max_rrp"}])),
        q("struct-021", "structural", "What was the minimum total demand in NSW1 over the freeze?",
          agg([{"fn": "min", "col": "total_demand", "as": "min_demand"}], [region_eq("NSW1")])),
        q("struct-022", "structural", "What was the maximum total demand in VIC1 over the freeze?",
          agg([{"fn": "max", "col": "total_demand", "as": "max_demand"}], [region_eq("VIC1")])),
        q("struct-023", "structural", "List RRP for every region at 2026-01-01 12:00:00.",
          lookup([{"col": "region", "op": "in", "val": ["NSW1", "QLD1", "VIC1", "SA1", "TAS1"]}, ts_eq("2026-01-01 12:00:00")],
                 ["region", "rrp"], 5, [{"col": "region", "dir": "asc"}])),
        q("struct-024", "structural", "How many rows (all regions) fall in January 2026?",
          agg([{"fn": "count", "as": "n"}], ts_range(jan_lo, jan_hi))),
        q("struct-025", "structural", "How many NSW1 rows are in August 2025?",
          agg([{"fn": "count", "as": "n"}], [region_eq("NSW1"), *ts_range(aug_lo, aug_hi)])),
        q("struct-026", "structural", "How many NSW1 rows are in February 2026?",
          agg([{"fn": "count", "as": "n"}], [region_eq("NSW1"), *ts_range(feb_lo, feb_hi)]),
          f"February 2026 has {monthrange(2026, 2)[1]} days."),
        q("struct-027", "structural", "How many VIC1 rows are in July 2026?",
          agg([{"fn": "count", "as": "n"}], [region_eq("VIC1"), *ts_range(jul_lo, jul_hi)])),
        q("struct-028", "structural", "How many intervals hit the -$1000 market price floor?",
          agg([{"fn": "count", "as": "n"}], [{"col": "rrp", "op": "=", "val": -1000}])),
        q("struct-029", "structural", "How many intervals have RRP strictly greater than 15000?",
          agg([{"fn": "count", "as": "n"}], [{"col": "rrp", "op": ">", "val": 15000}])),
        q("struct-030", "structural", "How many intervals have RRP equal to zero?",
          agg([{"fn": "count", "as": "n"}], [{"col": "rrp", "op": "=", "val": 0}])),
        q("struct-031", "structural", "How many intervals have strictly positive RRP?",
          agg([{"fn": "count", "as": "n"}], [{"col": "rrp", "op": ">", "val": 0}])),
        q("struct-032", "structural", "What is the minimum RRP in SA1 over the whole freeze?",
          agg([{"fn": "min", "col": "rrp", "as": "min_rrp"}], [region_eq("SA1")])),
        q("struct-033", "structural", "What is the maximum total demand in TAS1 over the freeze?",
          agg([{"fn": "max", "col": "total_demand", "as": "max_demand"}], [region_eq("TAS1")])),
        q("struct-034", "structural", "Does TAS1 have a row at the first freeze timestamp 2025-08-01 00:05:00?",
          lookup([region_eq("TAS1"), ts_eq("2025-08-01 00:05:00")], ["settlement_ts", "region"], 1)),
        q("struct-035", "structural", "Does SA1 have a row at the last freeze timestamp 2026-08-01 00:00:00?",
          lookup([region_eq("SA1"), ts_eq("2026-08-01 00:00:00")], ["settlement_ts", "region"], 1)),
        q("struct-036", "structural", "How many rows have period_type TRADE?",
          agg([{"fn": "count", "as": "n"}], [{"col": "period_type", "op": "=", "val": "TRADE"}])),
        q("struct-037", "structural", "Count rows per region in the freeze.",
          agg([{"fn": "count", "as": "n"}], group_by=["region"], limit=5, order_by=[{"col": "region", "dir": "asc"}]),
          "Balanced freeze: 105120 each."),
        q("struct-038", "structural", "What is the maximum RRP in QLD1 over the whole freeze?",
          agg([{"fn": "max", "col": "rrp", "as": "max_rrp"}], [region_eq("QLD1")])),
        q("struct-039", "structural", "How many VIC1 intervals have demand above 8000 MW?",
          agg([{"fn": "count", "as": "n"}], [region_eq("VIC1"), {"col": "total_demand", "op": ">", "val": 8000}])),
        q("struct-040", "structural", "List region and RRP at 2025-08-01 00:05:00 for all five regions.",
          lookup([{"col": "region", "op": "in", "val": ["NSW1", "QLD1", "VIC1", "SA1", "TAS1"]}, ts_eq("2025-08-01 00:05:00")],
                 ["region", "rrp", "total_demand"], 5, [{"col": "region", "dir": "asc"}])),
    ]
    return rows


def factual() -> list[dict]:
    jan_lo, jan_hi = month_window(2026, 1)
    dec_lo, dec_hi = month_window(2025, 12)
    aug_lo, aug_hi = month_window(2025, 8)
    sep_lo, sep_hi = month_window(2025, 9)
    oct_lo, oct_hi = month_window(2025, 10)
    nov_lo, nov_hi = month_window(2025, 11)
    feb_lo, feb_hi = month_window(2026, 2)
    mar_lo, mar_hi = month_window(2026, 3)
    apr_lo, apr_hi = month_window(2026, 4)
    may_lo, may_hi = month_window(2026, 5)
    jun_lo, jun_hi = month_window(2026, 6)
    jul_lo, jul_hi = month_window(2026, 7)
    d_jan1_lo, d_jan1_hi = day_window(2026, 1, 1)
    d_jan15_lo, d_jan15_hi = day_window(2026, 1, 15)
    d_dec25_lo, d_dec25_hi = day_window(2025, 12, 25)
    d_apr25_lo, d_apr25_hi = day_window(2026, 4, 25)
    d_jun30_lo, d_jun30_hi = day_window(2026, 6, 30)
    d_jul31_lo, d_jul31_hi = day_window(2026, 7, 31)
    d_aug15_lo, d_aug15_hi = day_window(2025, 8, 15)
    d_nov11_lo, d_nov11_hi = day_window(2025, 11, 11)
    d_feb14_lo, d_feb14_hi = day_window(2026, 2, 14)
    d_mar1_lo, d_mar1_hi = day_window(2026, 3, 1)
    month_note = "Calendar month uses interval-ending timestamps from 00:05 on the 1st through 00:00 on the next 1st."
    day_note = "Calendar day D is interval-ending timestamps from D 00:05:00 through D+1 00:00:00."

    rows = [
        q("fact-001", "factual", "What was the VIC1 regional reference price and demand at 2026-01-01 00:05:00 NEM time?",
          lookup([region_eq("VIC1"), ts_eq("2026-01-01 00:05:00")], ["settlement_ts", "region", "rrp", "total_demand"])),
        q("fact-002", "factual", "What was the maximum RRP in VIC1 during January 2026?",
          agg([{"fn": "max", "col": "rrp", "as": "max_rrp"}], [region_eq("VIC1"), *ts_range(jan_lo, jan_hi)]), month_note),
        q("fact-003", "factual", "How did average RRP in NSW1 compare to VIC1 on 2026-01-15?",
          compare({"fn": "avg", "col": "rrp", "as": "avg_rrp"},
                  ("nsw1", [region_eq("NSW1"), *ts_range(d_jan15_lo, d_jan15_hi)]),
                  ("vic1", [region_eq("VIC1"), *ts_range(d_jan15_lo, d_jan15_hi)])), day_note),
        q("fact-004", "factual", "What was NSW1 RRP at 2025-08-01 00:05:00?",
          lookup([region_eq("NSW1"), ts_eq("2025-08-01 00:05:00")], ["rrp"])),
        q("fact-005", "factual", "What was QLD1 demand at 2025-09-01 12:00:00?",
          lookup([region_eq("QLD1"), ts_eq("2025-09-01 12:00:00")], ["total_demand"])),
        q("fact-006", "factual", "What was SA1 RRP at 2025-10-31 23:55:00?",
          lookup([region_eq("SA1"), ts_eq("2025-10-31 23:55:00")], ["rrp", "total_demand"])),
        q("fact-007", "factual", "What was TAS1 RRP at 2025-12-25 18:00:00?",
          lookup([region_eq("TAS1"), ts_eq("2025-12-25 18:00:00")], ["rrp"])),
        q("fact-008", "factual", "What was VIC1 RRP at 2026-02-14 12:00:00?",
          lookup([region_eq("VIC1"), ts_eq("2026-02-14 12:00:00")], ["rrp", "total_demand"])),
        q("fact-009", "factual", "What was NSW1 RRP at 2026-04-25 15:30:00?",
          lookup([region_eq("NSW1"), ts_eq("2026-04-25 15:30:00")], ["rrp"])),
        q("fact-010", "factual", "What was QLD1 RRP at 2026-07-31 00:00:00?",
          lookup([region_eq("QLD1"), ts_eq("2026-07-31 00:00:00")], ["settlement_ts", "rrp"]),
          "Interval ending midnight is the last interval of 30 Jul, not 31 Jul."),
        q("fact-011", "factual", "What was SA1 demand at 2026-06-30 22:00:00?",
          lookup([region_eq("SA1"), ts_eq("2026-06-30 22:00:00")], ["total_demand", "rrp"])),
        q("fact-012", "factual", "What was TAS1 demand at 2025-11-11 11:00:00?",
          lookup([region_eq("TAS1"), ts_eq("2025-11-11 11:00:00")], ["total_demand"])),
        q("fact-013", "factual", "What was VIC1 RRP at 2026-03-01 00:05:00?",
          lookup([region_eq("VIC1"), ts_eq("2026-03-01 00:05:00")], ["rrp"])),
        q("fact-014", "factual", "What was NSW1 RRP and demand at 2026-01-01 12:00:00?",
          lookup([region_eq("NSW1"), ts_eq("2026-01-01 12:00:00")], ["rrp", "total_demand"])),
        q("fact-015", "factual", "What was QLD1 RRP at 2025-08-15 17:05:00?",
          lookup([region_eq("QLD1"), ts_eq("2025-08-15 17:05:00")], ["rrp"])),
        q("fact-016", "factual", "What was SA1 RRP at 2026-05-01 00:05:00?",
          lookup([region_eq("SA1"), ts_eq("2026-05-01 00:05:00")], ["rrp", "total_demand"])),
        q("fact-017", "factual", "What was TAS1 RRP at 2026-08-01 00:00:00?",
          lookup([region_eq("TAS1"), ts_eq("2026-08-01 00:00:00")], ["rrp"]),
          "Last interval of the freeze."),
        q("fact-018", "factual", "What was VIC1 demand at 2025-12-31 23:55:00?",
          lookup([region_eq("VIC1"), ts_eq("2025-12-31 23:55:00")], ["total_demand"])),
        q("fact-019", "factual", "What was NSW1 RRP at 2026-02-28 00:00:00?",
          lookup([region_eq("NSW1"), ts_eq("2026-02-28 00:00:00")], ["rrp"])),
        q("fact-020", "factual", "What was QLD1 RRP at 2026-06-01 06:00:00?",
          lookup([region_eq("QLD1"), ts_eq("2026-06-01 06:00:00")], ["rrp", "total_demand"])),
        q("fact-021", "factual", "What was SA1 RRP at 2025-09-15 09:00:00?",
          lookup([region_eq("SA1"), ts_eq("2025-09-15 09:00:00")], ["rrp"])),
        q("fact-022", "factual", "What was TAS1 RRP at 2026-04-01 00:05:00?",
          lookup([region_eq("TAS1"), ts_eq("2026-04-01 00:05:00")], ["rrp"])),
        q("fact-023", "factual", "What was VIC1 RRP at 2025-11-01 00:05:00?",
          lookup([region_eq("VIC1"), ts_eq("2025-11-01 00:05:00")], ["rrp", "total_demand"])),
        q("fact-024", "factual", "What was NSW1 demand at 2026-07-15 18:30:00?",
          lookup([region_eq("NSW1"), ts_eq("2026-07-15 18:30:00")], ["total_demand"])),
        q("fact-025", "factual", "What was QLD1 RRP at 2026-03-15 13:00:00?",
          lookup([region_eq("QLD1"), ts_eq("2026-03-15 13:00:00")], ["rrp"])),
        q("fact-026", "factual", "What was the minimum RRP in NSW1 in August 2025?",
          agg([{"fn": "min", "col": "rrp", "as": "min_rrp"}], [region_eq("NSW1"), *ts_range(aug_lo, aug_hi)]), month_note),
        q("fact-027", "factual", "What was the maximum RRP in QLD1 in December 2025?",
          agg([{"fn": "max", "col": "rrp", "as": "max_rrp"}], [region_eq("QLD1"), *ts_range(dec_lo, dec_hi)])),
        q("fact-028", "factual", "What was the maximum demand in SA1 in January 2026?",
          agg([{"fn": "max", "col": "total_demand", "as": "max_demand"}], [region_eq("SA1"), *ts_range(jan_lo, jan_hi)])),
        q("fact-029", "factual", "What was the minimum demand in TAS1 in July 2026?",
          agg([{"fn": "min", "col": "total_demand", "as": "min_demand"}], [region_eq("TAS1"), *ts_range(jul_lo, jul_hi)])),
        q("fact-030", "factual", "What was the maximum RRP in SA1 in September 2025?",
          agg([{"fn": "max", "col": "rrp", "as": "max_rrp"}], [region_eq("SA1"), *ts_range(sep_lo, sep_hi)])),
        q("fact-031", "factual", "What was the minimum RRP in VIC1 in February 2026?",
          agg([{"fn": "min", "col": "rrp", "as": "min_rrp"}], [region_eq("VIC1"), *ts_range(feb_lo, feb_hi)])),
        q("fact-032", "factual", "What was the maximum RRP in TAS1 in October 2025?",
          agg([{"fn": "max", "col": "rrp", "as": "max_rrp"}], [region_eq("TAS1"), *ts_range(oct_lo, oct_hi)])),
        q("fact-033", "factual", "What was the maximum demand in NSW1 in June 2026?",
          agg([{"fn": "max", "col": "total_demand", "as": "max_demand"}], [region_eq("NSW1"), *ts_range(jun_lo, jun_hi)])),
        q("fact-034", "factual", "What was the minimum RRP in QLD1 in May 2026?",
          agg([{"fn": "min", "col": "rrp", "as": "min_rrp"}], [region_eq("QLD1"), *ts_range(may_lo, may_hi)])),
        q("fact-035", "factual", "What was the maximum RRP in VIC1 on Christmas Day 2025?",
          agg([{"fn": "max", "col": "rrp", "as": "max_rrp"}], [region_eq("VIC1"), *ts_range(d_dec25_lo, d_dec25_hi)]), day_note),
        q("fact-036", "factual", "What was the maximum demand in NSW1 on 2026-04-25?",
          agg([{"fn": "max", "col": "total_demand", "as": "max_demand"}], [region_eq("NSW1"), *ts_range(d_apr25_lo, d_apr25_hi)])),
        q("fact-037", "factual", "What was the minimum RRP in SA1 on 2026-01-01?",
          agg([{"fn": "min", "col": "rrp", "as": "min_rrp"}], [region_eq("SA1"), *ts_range(d_jan1_lo, d_jan1_hi)])),
        q("fact-038", "factual", "What was the maximum RRP in QLD1 on 2026-02-14?",
          agg([{"fn": "max", "col": "rrp", "as": "max_rrp"}], [region_eq("QLD1"), *ts_range(d_feb14_lo, d_feb14_hi)])),
        q("fact-039", "factual", "What was the maximum RRP in TAS1 on 2025-11-11?",
          agg([{"fn": "max", "col": "rrp", "as": "max_rrp"}], [region_eq("TAS1"), *ts_range(d_nov11_lo, d_nov11_hi)])),
        q("fact-040", "factual", "What was the minimum demand in VIC1 on 2026-07-31?",
          agg([{"fn": "min", "col": "total_demand", "as": "min_demand"}], [region_eq("VIC1"), *ts_range(d_jul31_lo, d_jul31_hi)])),
        q("fact-041", "factual", "What was the average RRP in NSW1 in March 2026?",
          agg([{"fn": "avg", "col": "rrp", "as": "avg_rrp"}], [region_eq("NSW1"), *ts_range(mar_lo, mar_hi)])),
        q("fact-042", "factual", "What was the average demand in QLD1 in November 2025?",
          agg([{"fn": "avg", "col": "total_demand", "as": "avg_demand"}], [region_eq("QLD1"), *ts_range(nov_lo, nov_hi)])),
        q("fact-043", "factual", "What was the average RRP in SA1 in April 2026?",
          agg([{"fn": "avg", "col": "rrp", "as": "avg_rrp"}], [region_eq("SA1"), *ts_range(apr_lo, apr_hi)])),
        q("fact-044", "factual", "What was the average RRP in TAS1 in August 2025?",
          agg([{"fn": "avg", "col": "rrp", "as": "avg_rrp"}], [region_eq("TAS1"), *ts_range(aug_lo, aug_hi)])),
        q("fact-045", "factual", "What was the average demand in VIC1 in June 2026?",
          agg([{"fn": "avg", "col": "total_demand", "as": "avg_demand"}], [region_eq("VIC1"), *ts_range(jun_lo, jun_hi)])),
        q("fact-046", "factual", "How many NSW1 intervals in January 2026 had negative RRP?",
          agg([{"fn": "count", "as": "n"}], [region_eq("NSW1"), *ts_range(jan_lo, jan_hi), {"col": "rrp", "op": "<", "val": 0}])),
        q("fact-047", "factual", "How many SA1 intervals in September 2025 had RRP above 500?",
          agg([{"fn": "count", "as": "n"}], [region_eq("SA1"), *ts_range(sep_lo, sep_hi), {"col": "rrp", "op": ">", "val": 500}])),
        q("fact-048", "factual", "How many VIC1 intervals on 2026-01-15 had RRP below 0?",
          agg([{"fn": "count", "as": "n"}], [region_eq("VIC1"), *ts_range(d_jan15_lo, d_jan15_hi), {"col": "rrp", "op": "<", "val": 0}])),
        q("fact-049", "factual", "How many QLD1 intervals in February 2026 had demand above 9000 MW?",
          agg([{"fn": "count", "as": "n"}], [region_eq("QLD1"), *ts_range(feb_lo, feb_hi), {"col": "total_demand", "op": ">", "val": 9000}])),
        q("fact-050", "factual", "How many TAS1 intervals in July 2026 had RRP equal to the -$1000 floor?",
          agg([{"fn": "count", "as": "n"}], [region_eq("TAS1"), *ts_range(jul_lo, jul_hi), {"col": "rrp", "op": "=", "val": -1000}])),
        q("fact-051", "factual", "What was average RRP in NSW1 on 2025-08-15?",
          agg([{"fn": "avg", "col": "rrp", "as": "avg_rrp"}], [region_eq("NSW1"), *ts_range(d_aug15_lo, d_aug15_hi)]), day_note),
        q("fact-052", "factual", "What was average demand in SA1 on 2026-06-30?",
          agg([{"fn": "avg", "col": "total_demand", "as": "avg_demand"}], [region_eq("SA1"), *ts_range(d_jun30_lo, d_jun30_hi)])),
        q("fact-053", "factual", "What was average RRP in QLD1 on 2026-03-01?",
          agg([{"fn": "avg", "col": "rrp", "as": "avg_rrp"}], [region_eq("QLD1"), *ts_range(d_mar1_lo, d_mar1_hi)])),
        q("fact-054", "factual", "What was the sum of VIC1 demand over 2026-01-01 (MW summed across intervals)?",
          agg([{"fn": "sum", "col": "total_demand", "as": "sum_demand"}], [region_eq("VIC1"), *ts_range(d_jan1_lo, d_jan1_hi)]),
          "Sum of 5-minute demand readings, not energy in MWh."),
        q("fact-055", "factual", "What was the sum of TAS1 RRP over 2025-12-25?",
          agg([{"fn": "sum", "col": "rrp", "as": "sum_rrp"}], [region_eq("TAS1"), *ts_range(d_dec25_lo, d_dec25_hi)])),
        q("fact-056", "factual", "Count NSW1 intervals in May 2026.",
          agg([{"fn": "count", "as": "n"}], [region_eq("NSW1"), *ts_range(may_lo, may_hi)])),
        q("fact-057", "factual", "Count QLD1 intervals on 2026-04-25.",
          agg([{"fn": "count", "as": "n"}], [region_eq("QLD1"), *ts_range(d_apr25_lo, d_apr25_hi)])),
        q("fact-058", "factual", "What was max and min RRP in VIC1 in March 2026?",
          agg(
              [{"fn": "max", "col": "rrp", "as": "max_rrp"}, {"fn": "min", "col": "rrp", "as": "min_rrp"}],
              [region_eq("VIC1"), *ts_range(mar_lo, mar_hi)],
          )),
        q("fact-059", "factual", "What was max RRP and max demand in NSW1 in October 2025?",
          agg(
              [{"fn": "max", "col": "rrp", "as": "max_rrp"}, {"fn": "max", "col": "total_demand", "as": "max_demand"}],
              [region_eq("NSW1"), *ts_range(oct_lo, oct_hi)],
          )),
        q("fact-060", "factual", "What was average RRP and average demand in SA1 in July 2026?",
          agg(
              [{"fn": "avg", "col": "rrp", "as": "avg_rrp"}, {"fn": "avg", "col": "total_demand", "as": "avg_demand"}],
              [region_eq("SA1"), *ts_range(jul_lo, jul_hi)],
          )),
        q("fact-061", "factual", "For each region, what was maximum RRP in January 2026?",
          agg([{"fn": "max", "col": "rrp", "as": "max_rrp"}], ts_range(jan_lo, jan_hi),
              group_by=["region"], limit=5, order_by=[{"col": "region", "dir": "asc"}])),
        q("fact-062", "factual", "For each region, what was average RRP in December 2025?",
          agg([{"fn": "avg", "col": "rrp", "as": "avg_rrp"}], ts_range(dec_lo, dec_hi),
              group_by=["region"], limit=5, order_by=[{"col": "region", "dir": "asc"}])),
        q("fact-063", "factual", "For each region, what was maximum demand in June 2026?",
          agg([{"fn": "max", "col": "total_demand", "as": "max_demand"}], ts_range(jun_lo, jun_hi),
              group_by=["region"], limit=5, order_by=[{"col": "max_demand", "dir": "desc"}])),
        q("fact-064", "factual", "For each region, how many negative-RRP intervals were there in August 2025?",
          agg([{"fn": "count", "as": "n"}], [*ts_range(aug_lo, aug_hi), {"col": "rrp", "op": "<", "val": 0}],
              group_by=["region"], limit=5, order_by=[{"col": "region", "dir": "asc"}])),
        q("fact-065", "factual", "For each region, what was minimum RRP on 2026-01-15?",
          agg([{"fn": "min", "col": "rrp", "as": "min_rrp"}], ts_range(d_jan15_lo, d_jan15_hi),
              group_by=["region"], limit=5, order_by=[{"col": "region", "dir": "asc"}])),
        q("fact-066", "factual", "For each region, what was average demand on 2025-12-25?",
          agg([{"fn": "avg", "col": "total_demand", "as": "avg_demand"}], ts_range(d_dec25_lo, d_dec25_hi),
              group_by=["region"], limit=5, order_by=[{"col": "region", "dir": "asc"}])),
        q("fact-067", "factual", "Rank regions by maximum RRP in February 2026, highest first.",
          agg([{"fn": "max", "col": "rrp", "as": "max_rrp"}], ts_range(feb_lo, feb_hi),
              group_by=["region"], limit=5, order_by=[{"col": "max_rrp", "dir": "desc"}])),
        q("fact-068", "factual", "Rank regions by average RRP in May 2026, lowest first.",
          agg([{"fn": "avg", "col": "rrp", "as": "avg_rrp"}], ts_range(may_lo, may_hi),
              group_by=["region"], limit=5, order_by=[{"col": "avg_rrp", "dir": "asc"}])),
        q("fact-069", "factual", "Rank regions by maximum demand in November 2025, highest first.",
          agg([{"fn": "max", "col": "total_demand", "as": "max_demand"}], ts_range(nov_lo, nov_hi),
              group_by=["region"], limit=5, order_by=[{"col": "max_demand", "dir": "desc"}])),
        q("fact-070", "factual", "For each region, count intervals with RRP above 300 in March 2026.",
          agg([{"fn": "count", "as": "n"}], [*ts_range(mar_lo, mar_hi), {"col": "rrp", "op": ">", "val": 300}],
              group_by=["region"], limit=5, order_by=[{"col": "n", "dir": "desc"}])),
        q("fact-071", "factual", "Compare average RRP in QLD1 vs SA1 in January 2026.",
          compare({"fn": "avg", "col": "rrp", "as": "avg_rrp"},
                  ("qld1", [region_eq("QLD1"), *ts_range(jan_lo, jan_hi)]),
                  ("sa1", [region_eq("SA1"), *ts_range(jan_lo, jan_hi)]))),
        q("fact-072", "factual", "Compare maximum RRP in NSW1 vs VIC1 in December 2025.",
          compare({"fn": "max", "col": "rrp", "as": "max_rrp"},
                  ("nsw1", [region_eq("NSW1"), *ts_range(dec_lo, dec_hi)]),
                  ("vic1", [region_eq("VIC1"), *ts_range(dec_lo, dec_hi)]))),
        q("fact-073", "factual", "Compare minimum RRP in TAS1 vs SA1 in August 2025.",
          compare({"fn": "min", "col": "rrp", "as": "min_rrp"},
                  ("tas1", [region_eq("TAS1"), *ts_range(aug_lo, aug_hi)]),
                  ("sa1", [region_eq("SA1"), *ts_range(aug_lo, aug_hi)]))),
        q("fact-074", "factual", "Compare average demand in NSW1 vs QLD1 in June 2026.",
          compare({"fn": "avg", "col": "total_demand", "as": "avg_demand"},
                  ("nsw1", [region_eq("NSW1"), *ts_range(jun_lo, jun_hi)]),
                  ("qld1", [region_eq("QLD1"), *ts_range(jun_lo, jun_hi)]))),
        q("fact-075", "factual", "Compare maximum demand in VIC1 vs TAS1 in July 2026.",
          compare({"fn": "max", "col": "total_demand", "as": "max_demand"},
                  ("vic1", [region_eq("VIC1"), *ts_range(jul_lo, jul_hi)]),
                  ("tas1", [region_eq("TAS1"), *ts_range(jul_lo, jul_hi)]))),
        q("fact-076", "factual", "Compare average RRP in NSW1 vs QLD1 on 2026-01-01.",
          compare({"fn": "avg", "col": "rrp", "as": "avg_rrp"},
                  ("nsw1", [region_eq("NSW1"), *ts_range(d_jan1_lo, d_jan1_hi)]),
                  ("qld1", [region_eq("QLD1"), *ts_range(d_jan1_lo, d_jan1_hi)]))),
        q("fact-077", "factual", "Compare maximum RRP in SA1 vs VIC1 on 2025-12-25.",
          compare({"fn": "max", "col": "rrp", "as": "max_rrp"},
                  ("sa1", [region_eq("SA1"), *ts_range(d_dec25_lo, d_dec25_hi)]),
                  ("vic1", [region_eq("VIC1"), *ts_range(d_dec25_lo, d_dec25_hi)]))),
        q("fact-078", "factual", "Compare average RRP in TAS1 vs NSW1 on 2026-04-25.",
          compare({"fn": "avg", "col": "rrp", "as": "avg_rrp"},
                  ("tas1", [region_eq("TAS1"), *ts_range(d_apr25_lo, d_apr25_hi)]),
                  ("nsw1", [region_eq("NSW1"), *ts_range(d_apr25_lo, d_apr25_hi)]))),
        q("fact-079", "factual", "Compare average demand in QLD1 vs SA1 on 2026-02-14.",
          compare({"fn": "avg", "col": "total_demand", "as": "avg_demand"},
                  ("qld1", [region_eq("QLD1"), *ts_range(d_feb14_lo, d_feb14_hi)]),
                  ("sa1", [region_eq("SA1"), *ts_range(d_feb14_lo, d_feb14_hi)]))),
        q("fact-080", "factual", "Compare minimum RRP in VIC1 vs QLD1 in April 2026.",
          compare({"fn": "min", "col": "rrp", "as": "min_rrp"},
                  ("vic1", [region_eq("VIC1"), *ts_range(apr_lo, apr_hi)]),
                  ("qld1", [region_eq("QLD1"), *ts_range(apr_lo, apr_hi)]))),
        q("fact-081", "factual", "Compare average RRP in NSW1 vs TAS1 in November 2025.",
          compare({"fn": "avg", "col": "rrp", "as": "avg_rrp"},
                  ("nsw1", [region_eq("NSW1"), *ts_range(nov_lo, nov_hi)]),
                  ("tas1", [region_eq("TAS1"), *ts_range(nov_lo, nov_hi)]))),
        q("fact-082", "factual", "Compare maximum RRP in QLD1 vs TAS1 in May 2026.",
          compare({"fn": "max", "col": "rrp", "as": "max_rrp"},
                  ("qld1", [region_eq("QLD1"), *ts_range(may_lo, may_hi)]),
                  ("tas1", [region_eq("TAS1"), *ts_range(may_lo, may_hi)]))),
        q("fact-083", "factual", "Compare average demand in VIC1 vs SA1 in September 2025.",
          compare({"fn": "avg", "col": "total_demand", "as": "avg_demand"},
                  ("vic1", [region_eq("VIC1"), *ts_range(sep_lo, sep_hi)]),
                  ("sa1", [region_eq("SA1"), *ts_range(sep_lo, sep_hi)]))),
        q("fact-084", "factual", "Compare count of negative RRP intervals in SA1 vs VIC1 in January 2026.",
          compare({"fn": "count", "as": "n"},
                  ("sa1", [region_eq("SA1"), *ts_range(jan_lo, jan_hi), {"col": "rrp", "op": "<", "val": 0}]),
                  ("vic1", [region_eq("VIC1"), *ts_range(jan_lo, jan_hi), {"col": "rrp", "op": "<", "val": 0}]))),
        q("fact-085", "factual", "Compare average RRP in NSW1 vs VIC1 in July 2026.",
          compare({"fn": "avg", "col": "rrp", "as": "avg_rrp"},
                  ("nsw1", [region_eq("NSW1"), *ts_range(jul_lo, jul_hi)]),
                  ("vic1", [region_eq("VIC1"), *ts_range(jul_lo, jul_hi)]))),
        q("fact-086", "factual", "What was NSW1 RRP at 2025-10-01 00:05:00?",
          lookup([region_eq("NSW1"), ts_eq("2025-10-01 00:05:00")], ["rrp", "total_demand"])),
        q("fact-087", "factual", "What was VIC1 RRP at 2026-05-15 16:00:00?",
          lookup([region_eq("VIC1"), ts_eq("2026-05-15 16:00:00")], ["rrp"])),
        q("fact-088", "factual", "What was SA1 RRP at 2026-07-01 00:05:00?",
          lookup([region_eq("SA1"), ts_eq("2026-07-01 00:05:00")], ["rrp", "total_demand"])),
        q("fact-089", "factual", "What was QLD1 demand at 2025-12-01 08:00:00?",
          lookup([region_eq("QLD1"), ts_eq("2025-12-01 08:00:00")], ["total_demand"])),
        q("fact-090", "factual", "What was TAS1 RRP at 2026-01-26 12:00:00?",
          lookup([region_eq("TAS1"), ts_eq("2026-01-26 12:00:00")], ["rrp"])),
        q("fact-091", "factual", "What was the maximum RRP in NSW1 in April 2026?",
          agg([{"fn": "max", "col": "rrp", "as": "max_rrp"}], [region_eq("NSW1"), *ts_range(apr_lo, apr_hi)])),
        q("fact-092", "factual", "What was the minimum demand in QLD1 in August 2025?",
          agg([{"fn": "min", "col": "total_demand", "as": "min_demand"}], [region_eq("QLD1"), *ts_range(aug_lo, aug_hi)])),
        q("fact-093", "factual", "What was the maximum RRP in SA1 on 2025-08-15?",
          agg([{"fn": "max", "col": "rrp", "as": "max_rrp"}], [region_eq("SA1"), *ts_range(d_aug15_lo, d_aug15_hi)])),
        q("fact-094", "factual", "What was the average RRP in VIC1 in October 2025?",
          agg([{"fn": "avg", "col": "rrp", "as": "avg_rrp"}], [region_eq("VIC1"), *ts_range(oct_lo, oct_hi)])),
        q("fact-095", "factual", "How many TAS1 intervals in March 2026 had RRP above 200?",
          agg([{"fn": "count", "as": "n"}], [region_eq("TAS1"), *ts_range(mar_lo, mar_hi), {"col": "rrp", "op": ">", "val": 200}])),
        q("fact-096", "factual", "What was average demand in NSW1 on 2026-07-31?",
          agg([{"fn": "avg", "col": "total_demand", "as": "avg_demand"}], [region_eq("NSW1"), *ts_range(d_jul31_lo, d_jul31_hi)])),
        q("fact-097", "factual", "For each region, what was average RRP in July 2026?",
          agg([{"fn": "avg", "col": "rrp", "as": "avg_rrp"}], ts_range(jul_lo, jul_hi),
              group_by=["region"], limit=5, order_by=[{"col": "region", "dir": "asc"}])),
        q("fact-098", "factual", "Rank regions by minimum RRP in January 2026, lowest first.",
          agg([{"fn": "min", "col": "rrp", "as": "min_rrp"}], ts_range(jan_lo, jan_hi),
              group_by=["region"], limit=5, order_by=[{"col": "min_rrp", "dir": "asc"}])),
        q("fact-099", "factual", "Compare average RRP in QLD1 vs VIC1 in February 2026.",
          compare({"fn": "avg", "col": "rrp", "as": "avg_rrp"},
                  ("qld1", [region_eq("QLD1"), *ts_range(feb_lo, feb_hi)]),
                  ("vic1", [region_eq("VIC1"), *ts_range(feb_lo, feb_hi)]))),
        q("fact-100", "factual", "Compare maximum demand in NSW1 vs SA1 on 2026-01-15.",
          compare({"fn": "max", "col": "total_demand", "as": "max_demand"},
                  ("nsw1", [region_eq("NSW1"), *ts_range(d_jan15_lo, d_jan15_hi)]),
                  ("sa1", [region_eq("SA1"), *ts_range(d_jan15_lo, d_jan15_hi)]))),
        q("fact-101", "factual", "What was VIC1 RRP at 2026-06-15 07:00:00?",
          lookup([region_eq("VIC1"), ts_eq("2026-06-15 07:00:00")], ["rrp", "total_demand"])),
        q("fact-102", "factual", "What was NSW1 RRP at 2026-03-20 19:00:00?",
          lookup([region_eq("NSW1"), ts_eq("2026-03-20 19:00:00")], ["rrp"])),
        q("fact-103", "factual", "What was the minimum RRP in TAS1 in December 2025?",
          agg([{"fn": "min", "col": "rrp", "as": "min_rrp"}], [region_eq("TAS1"), *ts_range(dec_lo, dec_hi)])),
        q("fact-104", "factual", "What was the maximum demand in QLD1 in January 2026?",
          agg([{"fn": "max", "col": "total_demand", "as": "max_demand"}], [region_eq("QLD1"), *ts_range(jan_lo, jan_hi)])),
        q("fact-105", "factual", "What was average RRP in SA1 on 2025-11-11?",
          agg([{"fn": "avg", "col": "rrp", "as": "avg_rrp"}], [region_eq("SA1"), *ts_range(d_nov11_lo, d_nov11_hi)])),
        q("fact-106", "factual", "How many VIC1 intervals in April 2026 had demand below 4000 MW?",
          agg([{"fn": "count", "as": "n"}], [region_eq("VIC1"), *ts_range(apr_lo, apr_hi), {"col": "total_demand", "op": "<", "val": 4000}])),
        q("fact-107", "factual", "For each region, maximum RRP on 2026-03-01.",
          agg([{"fn": "max", "col": "rrp", "as": "max_rrp"}], ts_range(d_mar1_lo, d_mar1_hi),
              group_by=["region"], limit=5, order_by=[{"col": "max_rrp", "dir": "desc"}])),
        q("fact-108", "factual", "Compare min demand in TAS1 vs VIC1 in May 2026.",
          compare({"fn": "min", "col": "total_demand", "as": "min_demand"},
                  ("tas1", [region_eq("TAS1"), *ts_range(may_lo, may_hi)]),
                  ("vic1", [region_eq("VIC1"), *ts_range(may_lo, may_hi)]))),
        q("fact-109", "factual", "What was QLD1 RRP at 2026-08-01 00:00:00?",
          lookup([region_eq("QLD1"), ts_eq("2026-08-01 00:00:00")], ["rrp", "total_demand"])),
        q("fact-110", "factual", "What was SA1 demand at 2025-08-01 12:00:00?",
          lookup([region_eq("SA1"), ts_eq("2025-08-01 12:00:00")], ["total_demand"])),
        q("fact-111", "factual", "What was the maximum RRP in NSW1 on 2026-01-01?",
          agg([{"fn": "max", "col": "rrp", "as": "max_rrp"}], [region_eq("NSW1"), *ts_range(d_jan1_lo, d_jan1_hi)])),
        q("fact-112", "factual", "What was the average demand in TAS1 in February 2026?",
          agg([{"fn": "avg", "col": "total_demand", "as": "avg_demand"}], [region_eq("TAS1"), *ts_range(feb_lo, feb_hi)])),
        q("fact-113", "factual", "How many NSW1 intervals in October 2025 had RRP above 1000?",
          agg([{"fn": "count", "as": "n"}], [region_eq("NSW1"), *ts_range(oct_lo, oct_hi), {"col": "rrp", "op": ">", "val": 1000}])),
        q("fact-114", "factual", "For each region, average demand in April 2026.",
          agg([{"fn": "avg", "col": "total_demand", "as": "avg_demand"}], ts_range(apr_lo, apr_hi),
              group_by=["region"], limit=5, order_by=[{"col": "region", "dir": "asc"}])),
        q("fact-115", "factual", "Compare max RRP in NSW1 vs QLD1 on 2025-08-15.",
          compare({"fn": "max", "col": "rrp", "as": "max_rrp"},
                  ("nsw1", [region_eq("NSW1"), *ts_range(d_aug15_lo, d_aug15_hi)]),
                  ("qld1", [region_eq("QLD1"), *ts_range(d_aug15_lo, d_aug15_hi)]))),
        q("fact-116", "factual", "What was VIC1 RRP at 2026-04-01 00:00:00?",
          lookup([region_eq("VIC1"), ts_eq("2026-04-01 00:00:00")], ["rrp"]),
          "Interval ending 1 Apr 00:00 is the last interval of 31 Mar."),
        q("fact-117", "factual", "What was the minimum RRP in NSW1 in June 2026?",
          agg([{"fn": "min", "col": "rrp", "as": "min_rrp"}], [region_eq("NSW1"), *ts_range(jun_lo, jun_hi)])),
        q("fact-118", "factual", "What was average RRP in QLD1 on 2026-06-30?",
          agg([{"fn": "avg", "col": "rrp", "as": "avg_rrp"}], [region_eq("QLD1"), *ts_range(d_jun30_lo, d_jun30_hi)])),
        q("fact-119", "factual", "Compare average demand in TAS1 vs QLD1 in March 2026.",
          compare({"fn": "avg", "col": "total_demand", "as": "avg_demand"},
                  ("tas1", [region_eq("TAS1"), *ts_range(mar_lo, mar_hi)]),
                  ("qld1", [region_eq("QLD1"), *ts_range(mar_lo, mar_hi)]))),
        q("fact-120", "factual", "For each region, count intervals with RRP at or below -100 in July 2026.",
          agg([{"fn": "count", "as": "n"}], [*ts_range(jul_lo, jul_hi), {"col": "rrp", "op": "<=", "val": -100}],
              group_by=["region"], limit=5, order_by=[{"col": "region", "dir": "asc"}])),
    ]
    return rows


def out_of_scope() -> list[dict]:
    reason_unit = "unit-level generation / DUID is not in price_demand"
    reason_wem = "WEM / Western Australia is not in this NEM table"
    reason_fcst = "forecasts and pre-dispatch are not in this table"
    reason_advice = "advice and future prices are not in this table"
    reason_window = "timestamp is outside the frozen Aug 2025-Jul 2026 window"
    reason_other = "not in aggregated regional price and demand"
    rows = [
        q("oos-001", "out-of-scope", "What was Eraring's output at 12:00 on 2026-01-12?", refuse(reason_unit)),
        q("oos-002", "out-of-scope", "What was the WEM wholesale price in Perth on 2026-01-15?", refuse(reason_wem)),
        q("oos-003", "out-of-scope", "What is the pre-dispatch forecast RRP for VIC1 at 18:00 tomorrow?", refuse(reason_fcst)),
        q("oos-004", "out-of-scope", "Should I charge my home battery at 4pm in Victoria?", refuse(reason_advice)),
        q("oos-005", "out-of-scope", "What was VIC1 RRP on 2026-08-15?", refuse(reason_window)),
        q("oos-006", "out-of-scope", "What was NSW1 RRP in July 2025?", refuse(reason_window)),
        q("oos-007", "out-of-scope", "What was Loy Yang A generation on 2026-01-20?", refuse(reason_unit)),
        q("oos-008", "out-of-scope", "What was the QNI interconnector flow at 12:00 on 2026-03-01?", refuse(reason_other)),
        q("oos-009", "out-of-scope", "What was raise 6 second FCAS price in NSW1 on 2026-02-01?", refuse(reason_other)),
        q("oos-010", "out-of-scope", "What will NSW1 spot price be next Monday?", refuse(reason_advice)),
        q("oos-011", "out-of-scope", "What was Hornsdale Power Reserve output on 2025-12-01?", refuse(reason_unit)),
        q("oos-012", "out-of-scope", "What was rooftop solar generation in SA1 on 2026-01-15?", refuse(reason_other)),
        q("oos-013", "out-of-scope", "What was the SWIS operational demand in WA yesterday?", refuse(reason_wem)),
        q("oos-014", "out-of-scope", "What was AEMO's P5MIN forecast for QLD1 at 2026-01-01 12:00?", refuse(reason_fcst)),
        q("oos-015", "out-of-scope", "Who owns the Bayswater power station?", refuse(reason_other)),
        q("oos-016", "out-of-scope", "What was the Heywood interconnector flow on 2026-04-01?", refuse(reason_other)),
        q("oos-017", "out-of-scope", "What was Snowy 2.0 generation on 2026-06-01?", refuse(reason_unit)),
        q("oos-018", "out-of-scope", "What was the Victorian gas spot price on 2026-01-15?", refuse(reason_other)),
        q("oos-019", "out-of-scope", "What was frequency in the NEM at 12:00 on 2026-01-01?", refuse(reason_other)),
        q("oos-020", "out-of-scope", "Will VIC1 prices spike this evening?", refuse(reason_advice)),
        q("oos-021", "out-of-scope", "What was NSW1 RRP on 2024-01-01?", refuse(reason_window)),
        q("oos-022", "out-of-scope", "What is the emissions intensity of VIC1 at 2026-01-01 12:00?", refuse(reason_other)),
        q("oos-023", "out-of-scope", "What was Melbourne temperature at 15:00 on 2026-01-15?", refuse(reason_other)),
        q("oos-024", "out-of-scope", "What was the ASX Australian electricity futures close last Friday?", refuse(reason_other)),
        q("oos-025", "out-of-scope", "What did a typical NSW residential bill pay per kWh in January 2026?", refuse(reason_other)),
        q("oos-026", "out-of-scope", "What was DUID TORRA1 dispatch at 2026-02-01 12:00?", refuse(reason_unit)),
        q("oos-027", "out-of-scope", "How much RERT was activated on 2026-01-20?", refuse(reason_other)),
        q("oos-028", "out-of-scope", "What was New Zealand North Island spot price on 2026-01-15?", refuse(reason_other)),
        q("oos-029", "out-of-scope", "What was the NEMDE objective value for the 12:00 run on 2026-01-01?", refuse(reason_other)),
        q("oos-030", "out-of-scope", "Is now a good time to export from a SA home battery?", refuse(reason_advice)),
        q("oos-031", "out-of-scope", "What was Hailstorm solar farm output on 2025-12-20?", refuse(reason_unit)),
        q("oos-032", "out-of-scope", "What was the 5-minute predispatch demand forecast for TAS1 at 18:00 on 2026-07-01?", refuse(reason_fcst)),
        q("oos-033", "out-of-scope", "What was VIC1 RRP at 2026-09-01 00:05:00?", refuse(reason_window)),
        q("oos-034", "out-of-scope", "What was the Basslink flow at 2026-01-10 12:00?", refuse(reason_other)),
        q("oos-035", "out-of-scope", "What FCAS enablement did VIC1 have at 2026-03-01 12:00?", refuse(reason_other)),
        q("oos-036", "out-of-scope", "What was Collie power station output in WA on 2026-01-01?", refuse(reason_wem)),
        q("oos-037", "out-of-scope", "When will AEMO next publish a market notice about SA separation?", refuse(reason_other)),
        q("oos-038", "out-of-scope", "What was the Singapore wholesale electricity price on 2026-01-15?", refuse(reason_other)),
        q("oos-039", "out-of-scope", "What was NSW1 RRP at 2025-07-31 23:55:00?", refuse(reason_window)),
        q("oos-040", "out-of-scope", "How many MW did the Victorian Big Battery discharge at 19:00 on 2026-01-15?", refuse(reason_unit)),
    ]
    return rows


def build() -> list[dict]:
    rows = structural() + factual() + out_of_scope()
    if len(rows) != 200:
        raise SystemExit(f"expected 200 questions, got {len(rows)}")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate ids")
    counts = {}
    for r in rows:
        counts[r["bucket"]] = counts.get(r["bucket"], 0) + 1
    if counts != {"structural": 40, "factual": 120, "out-of-scope": 40}:
        raise SystemExit(f"bad bucket mix: {counts}")
    assign_splits(rows)
    splits = {}
    for r in rows:
        splits[(r["bucket"], r["split"])] = splits.get((r["bucket"], r["split"]), 0) + 1
    expected_splits = {
        ("structural", "eval"): 32,
        ("structural", "holdout"): 8,
        ("factual", "eval"): 96,
        ("factual", "holdout"): 24,
        ("out-of-scope", "eval"): 32,
        ("out-of-scope", "holdout"): 8,
    }
    if splits != expected_splits:
        raise SystemExit(f"bad split mix: {splits}")
    return rows


def main() -> int:
    rows = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} questions -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
