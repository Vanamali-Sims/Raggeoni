"""Write an investigation to disk as JSON (for machines) and Markdown (for people)."""

from __future__ import annotations

import json
from pathlib import Path

from raggeoni.spike.graph import Investigation
from raggeoni.spike.schema import CAUSES

LABELS = {
    "high_demand": "High demand",
    "low_wind": "Wind lull",
    "solar_drop": "Solar drop",
    "network_limit": "Network / interconnector limits",
    "generator_outage": "Generator outage",
    "battery_limited": "Limited battery output",
    "rebidding": "Rebidding",
}


def write_investigation(inv: Investigation, out_dir: Path) -> Path:
    folder = out_dir / inv.report.event_id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(
        json.dumps(inv.report.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    (folder / "evidence.json").write_text(
        json.dumps([e.to_dict() for e in inv.ledger.all()], indent=2, default=str), encoding="utf-8"
    )
    (folder / "trace.json").write_text(
        json.dumps(inv.trace, indent=2, default=str), encoding="utf-8"
    )
    (folder / "report.md").write_text(render_markdown(inv), encoding="utf-8")
    return folder


def render_markdown(inv: Investigation) -> str:
    r = inv.report
    v = r.verification or {}
    run = r.run or {}
    llm = run.get("llm", {})
    badge = "✅ all numbers grounded in evidence" if v.get("passed") else "⚠️ verification failed"
    lines = [
        f"# {r.region} price spike — {r.peak_ts} (NEM time)",
        "",
        f"**{r.headline}**",
        "",
        f"Peak ${r.peak_rrp:,.2f}/MWh · {badge} ({v.get('numbers_grounded', 0)}/{v.get('numbers_checked', 0)} numbers)",
        "",
    ]
    if run.get("degraded"):
        lines += [
            "> ⚠️ The model was unavailable for part of this run, so rules were used for: "
            + "; ".join(run.get("fallbacks", [])[:3]),
            "",
        ]
    if v.get("claims_dropped"):
        lines += [
            f"> {v['claims_dropped']} claim(s) were removed because their numbers could not be traced to evidence.",
            "",
        ]
    lines += ["## Causes", ""]
    if not r.causes:
        lines += ["No cause was supported by the evidence.", ""]
    for c in r.causes:
        lines += [f"### {LABELS.get(c.cause, c.cause)} — {c.confidence} confidence", ""]
        lines += [f"- {cl.text} _({', '.join(cl.evidence_ids)})_" for cl in c.claims]
        lines.append("")
    if r.ruled_out:
        lines += ["## Ruled out", ""]
        for c in r.ruled_out:
            lines.append(
                f"- **{LABELS.get(c.cause, c.cause)}**: " + " ".join(cl.text for cl in c.claims)
            )
        lines.append("")
    lines += ["## Not assessed", ""]
    lines += [f"- {LABELS.get(n, n)}: {CAUSES.get(n, '')}" for n in r.not_assessable]
    if r.caveats:
        lines += ["", "## Caveats", ""] + [f"- {c}" for c in r.caveats]
    lines += ["", "## Evidence", "", "| ID | Tool | Status | Summary |", "|---|---|---|---|"]
    for e in inv.ledger.all():
        summary = e.summary.replace("|", "/").replace("\n", " ")
        lines.append(f"| {e.evidence_id} | {e.tool} | {e.status} | {summary[:220]} |")
    lines += [
        "",
        "## Run",
        "",
        f"- Brain: `{run.get('brain')}` · options: `{json.dumps(run.get('options', {}))}`",
        f"- Time: {run.get('seconds')} s · tool calls: {run.get('tool_calls')} · model calls: {llm.get('calls', 0)} · tokens: {llm.get('input_tokens', 0) + llm.get('output_tokens', 0)}"
        + (f" · cost: ${llm['cost_usd']}" if llm.get("cost_usd") is not None else ""),
        "- Steps: " + " → ".join(t["node"] for t in inv.trace),
        "",
    ]
    return "\n".join(lines)
