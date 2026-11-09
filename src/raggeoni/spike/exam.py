"""The exam: past price spikes whose causes the AER has already published.

``eval/spikes/events.jsonl`` holds one row per event: region, the high-price
window (NEM time), the cause labels drafted from the AER report, and the
source and pages. Labels start as ``reviewed: false``; a human flips them
after checking the report. Scoring only uses the assessable causes
(rebidding and unit ramp limits need data the system doesn't load).

Arms:

* ``rules``  - deterministic heuristic brain (the baseline to beat)
* ``llm``    - full team: lead, parallel evidence agents, skeptic, writer
* ``llm-no-skeptic`` and ``llm-no-rag`` - ablations that show what each part adds

Each run writes per-event results plus a scorecard: micro precision, recall
and F1 over cause labels, per-cause recall, exact matches, verification pass
rate, degraded runs, seconds, tokens and (if prices are configured) cost.
"""

from __future__ import annotations

import json
import statistics
import subprocess
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from raggeoni.spike import demo_data
from raggeoni.spike.agents import Brain
from raggeoni.spike.config import data_dir, exam_dir, fmt_ts, parse_ts
from raggeoni.spike.detect import SpikeEvent, event_id
from raggeoni.spike.graph import RunOptions, investigate
from raggeoni.spike.schema import ASSESSABLE
from raggeoni.spike.store import query

ARMS = ("rules", "llm", "llm-no-skeptic", "llm-no-rag")
REQUIRED = ("id", "region", "window_start", "window_end", "labels", "source", "reviewed", "split")


@dataclass
class ExamItem:
    id: str
    region: str
    window_start: str
    window_end: str
    labels: list[str]
    split: str
    reviewed: bool
    raw: dict[str, Any]

    @property
    def gold(self) -> set[str]:
        return {label for label in self.labels if label in ASSESSABLE}


def load_items(
    path: Path | None = None, *, split: str = "eval", reviewed_only: bool = False
) -> list[ExamItem]:
    path = path or exam_dir() / "events.jsonl"
    items = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            raise ValueError(f"{path}:{line_no} missing {missing}")
        bad = [lab for lab in row["labels"] if lab not in ASSESSABLE]
        if bad:
            raise ValueError(f"{path}:{line_no} unknown labels {bad}")
        item = ExamItem(
            row["id"],
            row["region"],
            row["window_start"],
            row["window_end"],
            row["labels"],
            row["split"],
            bool(row["reviewed"]),
            row,
        )
        if split != "all" and item.split != split:
            continue
        if reviewed_only and not item.reviewed:
            continue
        items.append(item)
    return items


def incident_db(item_id: str) -> Path:
    return data_dir() / "exam" / f"{item_id}.duckdb"


def build_data(
    item: ExamItem, *, baseline_days: int = 30, with_forecasts: bool = True
) -> dict[str, Any]:
    from raggeoni.spike.ingest import build_incident

    db = incident_db(item.id)
    start = parse_ts(item.window_start) - timedelta(hours=3)
    end = parse_ts(item.window_end) + timedelta(hours=1)
    return build_incident(
        db,
        region=item.region,
        start=start,
        end=end,
        baseline_days=baseline_days,
        with_forecasts=with_forecasts,
    )


def event_for(item: ExamItem, db: Path) -> SpikeEvent:
    """The exam gives the high-price window; find the actual peak inside it from the data."""
    rows = query(
        db,
        "SELECT settlement_ts, rrp FROM price_demand WHERE region = ? AND settlement_ts BETWEEN ? AND ? ORDER BY rrp DESC, settlement_ts",
        [item.region, item.window_start, item.window_end],
    )
    if not rows:
        raise FileNotFoundError(
            f"no prices for {item.id}; run: raggeoni-spike exam build-data --event {item.id}"
        )
    peak_ts, peak_rrp = fmt_ts(rows[0][0]), float(rows[0][1])
    n = len(rows)
    return SpikeEvent(
        event_id(item.region, peak_ts),
        item.region,
        item.window_start,
        item.window_end,
        peak_ts,
        round(peak_rrp, 2),
        n,
    )


def make_brain(arm: str) -> Brain:
    from raggeoni.spike.agents import make_brain as _make

    return _make("rules" if arm == "rules" else "llm")


def arm_options(arm: str) -> RunOptions:
    return RunOptions(rag=arm != "llm-no-rag", skeptic=arm != "llm-no-skeptic", label=arm)


def score(gold: set[str], predicted: set[str]) -> dict[str, Any]:
    tp, fp, fn = len(gold & predicted), len(predicted - gold), len(gold - predicted)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "exact": gold == predicted,
        "missed": sorted(gold - predicted),
        "extra": sorted(predicted - gold),
    }


def run_item(item: ExamItem, arm: str, db: Path | None = None) -> dict[str, Any]:
    db = db or incident_db(item.id)
    if not db.exists():
        return {"id": item.id, "status": "no_data", "gold": sorted(item.gold)}
    try:
        event = event_for(item, db)
    except FileNotFoundError as exc:
        return {"id": item.id, "status": "no_data", "error": str(exc), "gold": sorted(item.gold)}
    inv = investigate(event, db, make_brain(arm), arm_options(arm))
    predicted = set(inv.report.predicted_causes())
    return {
        "id": item.id,
        "status": "ok",
        "region": item.region,
        "peak_ts": event.peak_ts,
        "peak_rrp": event.peak_rrp,
        "gold": sorted(item.gold),
        "predicted": sorted(predicted),
        **score(item.gold, predicted),
        "verified": bool(inv.report.verification.get("passed")),
        "claims_dropped": inv.report.verification.get("claims_dropped", 0),
        "degraded": inv.report.run.get("degraded", False),
        "seconds": inv.report.run.get("seconds"),
        "tool_calls": inv.report.run.get("tool_calls"),
        "llm": inv.report.run.get("llm"),
        "headline": inv.report.headline,
    }


def scorecard(results: list[dict[str, Any]], *, arm: str, split: str) -> dict[str, Any]:
    done = [r for r in results if r["status"] == "ok"]
    tp = sum(r["tp"] for r in done)
    fp = sum(r["fp"] for r in done)
    fn = sum(r["fn"] for r in done)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_cause: dict[str, dict[str, Any]] = {}
    for cause in ASSESSABLE:
        with_cause = [r for r in done if cause in r["gold"]]
        flagged = [r for r in done if cause in r["predicted"]]
        hit = sum(1 for r in with_cause if cause in r["predicted"])
        per_cause[cause] = {
            "events": len(with_cause),
            "recall": round(hit / len(with_cause), 3) if with_cause else None,
            "false_positives": sum(1 for r in flagged if cause not in r["gold"]),
        }
    secs = [r["seconds"] for r in done if r.get("seconds") is not None]
    tokens = [
        (r.get("llm") or {}).get("input_tokens", 0) + (r.get("llm") or {}).get("output_tokens", 0)
        for r in done
    ]
    costs = [(r.get("llm") or {}).get("cost_usd") for r in done]
    return {
        "arm": arm,
        "split": split,
        "git_sha": _git_sha(),
        "events": len(results),
        "scored": len(done),
        "no_data": [r["id"] for r in results if r["status"] != "ok"],
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "exact_match": round(sum(r["exact"] for r in done) / len(done), 3) if done else 0.0,
        "verified_rate": round(sum(r["verified"] for r in done) / len(done), 3) if done else 0.0,
        "degraded_runs": sum(r["degraded"] for r in done),
        "seconds_p50": round(statistics.median(secs), 2) if secs else None,
        "seconds_p95": round(_p95(secs), 2) if secs else None,
        "tokens_mean": round(statistics.mean(tokens)) if tokens else 0,
        "cost_usd_total": round(sum(c for c in costs if c is not None), 4)
        if any(c is not None for c in costs)
        else None,
        "per_cause": per_cause,
    }


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run_exam(
    arm: str, split: str = "eval", *, reviewed_only: bool = False, limit: int = 0
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    items = load_items(split=split, reviewed_only=reviewed_only)
    if limit:
        items = items[:limit]
    results = [run_item(item, arm) for item in items]
    return scorecard(results, arm=arm, split=split), results


def run_synthetic(arm: str, workdir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The planted-cause demo incident as a one-item exam. Runs offline."""
    db = demo_data.build(workdir / "synthetic.duckdb", with_injection=True)
    item = ExamItem(
        "synthetic-sa-2031",
        "SA1",
        "2031-01-15 17:50:00",
        "2031-01-15 21:25:00",
        demo_data.EXPECTED_CAUSES,
        "synthetic",
        True,
        {},
    )
    results = [run_item(item, arm, db=db)]
    return scorecard(results, arm=arm, split="synthetic"), results


def save_run(
    card: dict[str, Any], results: list[dict[str, Any]], out_dir: Path | None = None
) -> Path:
    out_dir = out_dir or exam_dir() / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{card['arm']}-{card['split']}-{card['git_sha']}.json"
    path.write_text(
        json.dumps({"scorecard": card, "results": results}, indent=2, default=str), encoding="utf-8"
    )
    return path


def compare(
    card: dict[str, Any], baseline: dict[str, Any], *, tolerance: float = 0.05
) -> dict[str, Any]:
    """Regression check against a stored scorecard (same arm and split)."""
    delta = {
        k: round(card[k] - baseline.get(k, 0), 3)
        for k in ("precision", "recall", "f1", "exact_match", "verified_rate")
    }
    regressed = [k for k in ("f1", "verified_rate") if delta[k] < -tolerance]
    return {"baseline_sha": baseline.get("git_sha"), "delta": delta, "regressed": regressed}


def markdown_summary(
    card: dict[str, Any], results: list[dict[str, Any]], comparison: dict[str, Any] | None = None
) -> str:
    lines = [
        f"### Spike exam: `{card['arm']}` on `{card['split']}` ({card['scored']}/{card['events']} events scored)",
        "",
        "| Metric | Value |" + (" Δ vs baseline |" if comparison else ""),
        "|---|---|" + ("---|" if comparison else ""),
    ]
    for key in ("precision", "recall", "f1", "exact_match", "verified_rate"):
        row = f"| {key} | {card[key]} |"
        if comparison:
            d = comparison["delta"][key]
            row += f" {d:+.3f} |"
        lines.append(row)
    lines += [
        f"| degraded runs | {card['degraded_runs']} |" + (" |" if comparison else ""),
        f"| seconds p50 / p95 | {card['seconds_p50']} / {card['seconds_p95']} |"
        + (" |" if comparison else ""),
        f"| tokens per event | {card['tokens_mean']} |" + (" |" if comparison else ""),
        "",
        "| Cause | Events | Recall | False positives |",
        "|---|---|---|---|",
    ]
    for cause, row in card["per_cause"].items():
        lines.append(f"| {cause} | {row['events']} | {row['recall']} | {row['false_positives']} |")
    misses = [r for r in results if r["status"] == "ok" and not r["exact"]]
    if misses:
        lines += ["", "<details><summary>Misses</summary>", ""]
        lines += [
            f"- `{r['id']}` missed {r['missed'] or '-'}, extra {r['extra'] or '-'}" for r in misses
        ]
        lines += ["", "</details>"]
    if card["no_data"]:
        lines += ["", f"No data for: {', '.join(card['no_data'])}"]
    if comparison and comparison["regressed"]:
        lines += [
            "",
            f"**Regression:** {', '.join(comparison['regressed'])} dropped more than the tolerance.",
        ]
    return "\n".join(lines)
