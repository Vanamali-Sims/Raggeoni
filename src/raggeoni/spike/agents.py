"""The investigation team's roles, as two interchangeable "brains".

* ``LLMBrain``: each role is a language model with its own prompt. Evidence
  agents run a tool-use loop and decide which analyses to run.
* ``HeuristicBrain``: the same roles as fixed rules. It needs no API key, is
  fully deterministic, runs in CI, and is the **baseline arm** in the exam:
  the LLM team has to beat hand-written rules to justify its cost.

If the model fails mid-investigation, ``LLMBrain`` falls back to the
heuristic for that step and records why, so a run never dies on an outage.
"""

from __future__ import annotations

import json
import re
import time
from importlib import resources
from typing import Any, Protocol

from raggeoni.spike.detect import SpikeEvent
from raggeoni.spike.evidence import Evidence
from raggeoni.spike.llm import ChatModel, LLMError, Usage, parse_json_object
from raggeoni.spike.schema import (
    ASSESSABLE,
    CAUSES,
    CauseEntry,
    Challenge,
    Claim,
    Finding,
    Hypothesis,
    Report,
)
from raggeoni.spike.tools import TOOL_SPECS, ToolContext, call_tool

_TRIP = re.compile(
    r"\btrip(ped)?\b|forced outage|failed to (start|synchroni[sz]e)|unplanned outage|contingency|unit .* (offline|unavailable)",
    re.IGNORECASE,
)
_NETWORK = re.compile(
    r"interconnector|transfer limit|constraint|line outage|kv line|network outage|heywood|murraylink|basslink|qni|terranora",
    re.IGNORECASE,
)


SHORT_NAMES = {
    "high_demand": "unusually high demand",
    "low_wind": "a wind lull",
    "solar_drop": "a drop in solar output",
    "network_limit": "interconnector or network limits",
    "generator_outage": "a generator outage",
    "battery_limited": "limited battery output",
    "rebidding": "rebidding",
}


def _prompt(name: str) -> str:
    return (
        resources.files("raggeoni.spike.prompts").joinpath(f"{name}.md").read_text(encoding="utf-8")
    )


def event_brief(event: SpikeEvent) -> dict[str, Any]:
    start, end = event.window
    return {
        "event_id": event.event_id,
        "region": event.region,
        "peak_ts": event.peak_ts,
        "peak_rrp": event.peak_rrp,
        "high_price_from": event.start_ts,
        "high_price_to": event.end_ts,
        "investigation_window": [
            start.strftime("%Y-%m-%d %H:%M:%S"),
            end.strftime("%Y-%m-%d %H:%M:%S"),
        ],
    }


class Brain(Protocol):
    name: str
    usage: Usage
    fallbacks: list[str]

    def hypothesize(
        self, event: SpikeEvent, overview: list[Evidence], ctx: ToolContext
    ) -> list[Hypothesis]: ...

    def investigate(
        self,
        event: SpikeEvent,
        hypothesis: Hypothesis,
        ctx: ToolContext,
        *,
        round_: int,
        question: str | None = None,
    ) -> Finding: ...

    def challenge(
        self, event: SpikeEvent, findings: list[Finding], ctx: ToolContext
    ) -> list[Challenge]: ...

    def write(self, event: SpikeEvent, findings: list[Finding], ctx: ToolContext) -> Report: ...


# =============================================================================
# Heuristic brain (deterministic baseline)
# =============================================================================


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.1f}" if abs(value) < 1000 else f"{value:,.0f}"
    return str(value)


class HeuristicBrain:
    name = "heuristic"

    def __init__(self) -> None:
        self.usage = Usage()
        self.fallbacks: list[str] = []

    def hypothesize(self, event, overview, ctx):
        return [
            Hypothesis(c, "checked by rule for every event", i + 1)
            for i, c in enumerate(ASSESSABLE)
        ]

    def investigate(self, event, hypothesis, ctx, *, round_, question=None):
        at = event.peak_ts
        start, end = (t.strftime("%Y-%m-%d %H:%M:%S") for t in event.window)
        r = event.region
        claims: list[Claim] = []
        cause = hypothesis.cause

        def claim(text: str, *evs: Evidence) -> None:
            claims.append(Claim(text, [e.evidence_id for e in evs]))

        if cause == "high_demand":
            u = call_tool(ctx, "unusualness", {"region": r, "metric": "demand", "at": at})
            s = call_tool(ctx, "demand_surprise", {"region": r, "at": at})
            high = u.status == "ok" and u.facts["unusual"] == "high"
            surprise = s.status == "ok" and s.facts["error_mw"] > 100 and s.facts["error_pct"] > 3
            if u.status == "ok":
                claim(
                    f"Demand was {_fmt(u.facts['value'])} MW against a typical {_fmt(u.facts['typical_mean'])} MW for this time of day (percentile {u.facts['percentile']}).",
                    u,
                )
            if s.status == "ok":
                claim(
                    f"Demand was {_fmt(s.facts['error_mw'])} MW different from the forecast made 30 minutes earlier ({_fmt(s.facts['error_pct'])}%).",
                    s,
                )
            if high or surprise:
                return Finding(
                    cause,
                    "supports",
                    "high" if high and surprise else "medium",
                    claims,
                    round=round_,
                )
            if u.status == "ok":
                return Finding(cause, "contradicts", "medium", claims, round=round_)
            return Finding(cause, "inconclusive", "low", claims, round=round_)

        if cause in ("low_wind", "solar_drop", "battery_limited"):
            metric = {
                "low_wind": "wind",
                "solar_drop": "solar",
                "battery_limited": "battery_discharging",
            }[cause]
            u = call_tool(ctx, "unusualness", {"region": r, "metric": metric, "at": at})
            g = call_tool(
                ctx, "generation_change", {"region": r, "at": at, "lookback_minutes": 120}
            )
            if u.status != "ok":
                return Finding(
                    cause, "inconclusive", "low", [], explanation=u.summary, round=round_
                )
            claim(
                f"{metric.replace('_', ' ').capitalize()} output was {_fmt(u.facts['value'])} MW against a typical {_fmt(u.facts['typical_mean'])} MW for this time of day (percentile {u.facts['percentile']}).",
                u,
            )
            change = g.facts.get(f"{metric}_change_mw") if g.status == "ok" else None
            if change is not None:
                claim(
                    f"Over the previous two hours {metric.replace('_', ' ')} changed by {_fmt(change)} MW.",
                    g,
                )
            if u.facts["unusual"] == "low":
                return Finding(
                    cause,
                    "supports",
                    "high" if (change or 0) < -100 else "medium",
                    claims,
                    round=round_,
                )
            return Finding(cause, "contradicts", "medium", claims, round=round_)

        if cause == "network_limit":
            ic = call_tool(ctx, "interconnector_status", {"region": r, "at": at})
            n = call_tool(
                ctx,
                "search_notices",
                {
                    "region": r,
                    "start": start,
                    "end": end,
                    "query_text": "interconnector transfer limit constraint line outage",
                },
            )
            limited = []
            if ic.status == "ok":
                for row in ic.rows:
                    if (
                        row["at_limit"]
                        or row["constraint_binding"]
                        or (row["limit_reduction_mw"] or 0) > 50
                    ):
                        limited.append(row)
                        extra = (
                            f", {_fmt(row['limit_reduction_mw'])} MW below its typical limit"
                            if (row["limit_reduction_mw"] or 0) > 50
                            else ""
                        )
                        claim(
                            f"{row['name']} carried {_fmt(row['flow_into_region_mw'])} MW into {r} against a limit of {_fmt(row['import_limit_into_region_mw'])} MW{extra}.",
                            ic,
                        )
            network_notices = (
                [
                    row
                    for row in n.rows
                    if not row["flagged_as_suspicious"]
                    and _NETWORK.search(f"{row['title']} {row['text']}")
                ]
                if n.status == "ok"
                else []
            )
            for row in network_notices[:2]:
                claim(f"Market notice {row['notice_id']} reported: {row['title']}.", n)
            if limited:
                return Finding(
                    cause,
                    "supports",
                    "high"
                    if network_notices or any(x["constraint_binding"] for x in limited)
                    else "medium",
                    claims,
                    round=round_,
                )
            if ic.status == "ok":
                return Finding(
                    cause,
                    "contradicts",
                    "medium",
                    claims or [Claim(ic.summary, [ic.evidence_id])],
                    round=round_,
                )
            return Finding(cause, "inconclusive", "low", claims, round=round_)

        if cause == "generator_outage":
            n = call_tool(
                ctx,
                "search_notices",
                {
                    "region": r,
                    "start": start,
                    "end": end,
                    "query_text": "unit trip forced outage failed to start generator unavailable",
                },
            )
            trips = (
                [
                    row
                    for row in n.rows
                    if not row["flagged_as_suspicious"]
                    and _TRIP.search(f"{row['title']} {row['text']}")
                ]
                if n.status == "ok"
                else []
            )
            for row in trips[:2]:
                claim(f"Market notice {row['notice_id']} reported: {row['title']}.", n)
            if trips:
                return Finding(cause, "supports", "medium", claims, round=round_)
            return Finding(
                cause,
                "inconclusive",
                "low",
                [],
                explanation="No notice of a trip or failed start; absence of a notice does not prove no outage.",
                round=round_,
            )

        return Finding(
            cause, "inconclusive", "low", [], explanation="no rule for this cause", round=round_
        )

    def challenge(self, event, findings, ctx):
        return []  # the baseline has no skeptic

    def write(self, event, findings, ctx):
        gap = call_tool(
            ctx,
            "supply_gap",
            {"region": event.region, "at": event.peak_ts, "lookback_minutes": 180},
        )
        profile = call_tool(
            ctx,
            "price_profile",
            {"region": event.region, "start": event.start_ts, "end": event.end_ts},
        )
        return compose_report(event, findings, gap, profile, brain=self.name)


def compose_report(
    event: SpikeEvent, findings: list[Finding], gap: Evidence, profile: Evidence, *, brain: str
) -> Report:
    share = {}
    if gap.status == "ok":
        key = {
            "demand_increase": "high_demand",
            "wind_decrease": "low_wind",
            "solar_decrease": "solar_drop",
            "import_decrease": "network_limit",
            "battery_discharging_decrease": "battery_limited",
        }
        for row in gap.rows:
            if row["component"] in key:
                share[key[row["component"]]] = row["share_pct"] or 0
    supports = sorted(
        (f for f in findings if f.verdict == "supports"),
        key=lambda f: (-{"high": 3, "medium": 2, "low": 1}[f.confidence], -share.get(f.cause, 0)),
    )
    causes = []
    for f in supports:
        claims = list(f.claims)
        if f.cause in share and share[f.cause] > 0:
            comp = next(
                r
                for r in gap.rows
                if {
                    "demand_increase": "high_demand",
                    "wind_decrease": "low_wind",
                    "solar_decrease": "solar_drop",
                    "import_decrease": "network_limit",
                    "battery_discharging_decrease": "battery_limited",
                }.get(r["component"])
                == f.cause
            )
            claims.append(
                Claim(
                    f"This accounted for {comp['share_pct']}% ({_fmt(comp['mw'])} MW) of the worsening in the supply-demand balance.",
                    [gap.evidence_id],
                )
            )
        causes.append(CauseEntry(f.cause, f.confidence, claims))
    ruled_out = [
        CauseEntry(f.cause, f.confidence, f.claims)
        for f in findings
        if f.verdict == "contradicts" and f.claims
    ]
    names = [SHORT_NAMES[c.cause] for c in causes[:3]]
    price = (
        profile.facts.get("peak_rrp", event.peak_rrp) if profile.status == "ok" else event.peak_rrp
    )
    if names:
        joined = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        headline = f"{event.region} reached ${price:,.2f}/MWh at {event.peak_ts}, driven mainly by {joined}."
    else:
        headline = f"{event.region} reached ${price:,.2f}/MWh at {event.peak_ts}; the available evidence did not identify a clear cause."
    caveats = ["Rebidding was not assessed: bid data is not loaded."]
    caveats += [
        f"{f.cause}: {f.explanation}"
        for f in findings
        if f.verdict == "inconclusive" and f.explanation
    ]
    features: dict[str, float] = {"peak_rrp": float(price)}
    if gap.status == "ok":
        for k, v in gap.facts.items():
            if k.endswith("_mw") and isinstance(v, int | float):
                features[k] = float(v)
    return Report(
        event_id=event.event_id,
        region=event.region,
        peak_ts=event.peak_ts,
        peak_rrp=float(price),
        headline=headline,
        causes=causes,
        ruled_out=ruled_out,
        not_assessable=["rebidding"],
        caveats=caveats,
        features=features,
        run={"brain": brain},
    )


# =============================================================================
# LLM brain
# =============================================================================


def tool_schemas() -> list[dict[str, Any]]:
    out = []
    for name, spec in TOOL_SPECS.items():
        props: dict[str, Any] = {}
        required = []
        for pname, ptype in spec["params"].items():
            base = ptype.split(" ")[0]
            json_type = {
                "string": "string",
                "timestamp": "string",
                "integer": "integer",
                "object": "object",
            }[base]
            props[pname] = {"type": json_type, "description": ptype}
            if "default" not in ptype:
                required.append(pname)
        out.append(
            {
                "name": name,
                "description": spec["description"],
                "parameters": {"type": "object", "properties": props, "required": required},
            }
        )
    return out


class LLMBrain:
    def __init__(
        self, chat: ChatModel, *, max_tool_calls: int = 6, token_budget: int = 250_000
    ) -> None:
        self.chat = chat
        self.name = f"llm:{chat.name}"
        self.usage = Usage()
        self.fallbacks: list[str] = []
        self.max_tool_calls = max_tool_calls
        self.token_budget = token_budget
        self._heuristic = HeuristicBrain()
        self._common = _prompt("common")

    # -- plumbing -------------------------------------------------------------

    def _over_budget(self) -> bool:
        return self.usage.total_tokens >= self.token_budget

    def _ask(self, role: str, system: str, user: str, *, max_tokens: int = 1500) -> dict[str, Any]:
        if self._over_budget():
            raise LLMError(f"token budget of {self.token_budget} exhausted")
        t0 = time.perf_counter()
        try:
            reply = self.chat.chat(
                system, [{"role": "user", "content": user}], max_tokens=max_tokens
            )
        except LLMError:
            self.usage.add(role, None, time.perf_counter() - t0, failed=True)
            raise
        self.usage.add(role, reply, time.perf_counter() - t0)
        return parse_json_object(reply.text)

    def _fallback(self, step: str, exc: Exception) -> None:
        self.fallbacks.append(f"{step}: {exc}"[:300])

    # -- roles ----------------------------------------------------------------

    def hypothesize(self, event, overview, ctx):
        system = self._common + "\n\n" + _prompt("lead").format(causes=json.dumps(CAUSES, indent=1))
        user = json.dumps(
            {"event": event_brief(event), "overview": [json.loads(e.for_agent()) for e in overview]}
        )
        try:
            data = self._ask("lead", system, user)
            hyps = []
            for h in data.get("hypotheses") or []:
                if isinstance(h, dict) and h.get("cause") in ASSESSABLE:
                    hyps.append(
                        Hypothesis(
                            h["cause"], str(h.get("rationale", ""))[:300], int(h.get("priority", 5))
                        )
                    )
            if not hyps:
                raise LLMError("lead returned no usable hypotheses")
            unique: dict[str, Hypothesis] = {}
            for h in sorted(hyps, key=lambda h: h.priority):
                unique.setdefault(h.cause, h)
            return list(unique.values())[:6]
        except LLMError as exc:
            self._fallback("lead", exc)
            return self._heuristic.hypothesize(event, overview, ctx)

    def investigate(self, event, hypothesis, ctx, *, round_, question=None):
        system = self._common + "\n\n" + _prompt("evidence")
        task: dict[str, Any] = {
            "event": event_brief(event),
            "hypothesis": {
                "cause": hypothesis.cause,
                "meaning": CAUSES[hypothesis.cause],
                "why_considered": hypothesis.rationale,
            },
        }
        if question:
            task["skeptic_question"] = question
        messages: list[dict[str, Any]] = [{"role": "user", "content": json.dumps(task)}]
        tools = tool_schemas()
        calls_made = 0
        role = f"evidence:{hypothesis.cause}"
        try:
            while True:
                if self._over_budget():
                    raise LLMError(f"token budget of {self.token_budget} exhausted")
                t0 = time.perf_counter()
                try:
                    reply = self.chat.chat(
                        system, messages, tools=tools if calls_made < self.max_tool_calls else None
                    )
                except LLMError:
                    self.usage.add(role, None, time.perf_counter() - t0, failed=True)
                    raise
                self.usage.add(role, reply, time.perf_counter() - t0)
                if not reply.tool_calls:
                    data = parse_json_object(reply.text)
                    return Finding.from_dict(data, cause=hypothesis.cause, round_=round_)
                messages.append(reply.raw_assistant)
                results = []
                for tc in reply.tool_calls:
                    calls_made += 1
                    ev = call_tool(ctx, tc.name, tc.arguments)
                    results.append((tc, ev.for_agent()))
                messages.extend(self.chat.tool_result_message(results))
                if calls_made >= self.max_tool_calls:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Tool budget reached. Give your final JSON answer now.",
                        }
                    )
        except LLMError as exc:
            self._fallback(role, exc)
            finding = self._heuristic.investigate(
                event, hypothesis, ctx, round_=round_, question=question
            )
            finding.error = f"fell back to rules: {exc}"[:300]
            return finding

    def challenge(self, event, findings, ctx):
        system = self._common + "\n\n" + _prompt("skeptic")
        payload = {
            "event": event_brief(event),
            "findings": [_finding_view(f, ctx) for f in findings],
        }
        try:
            data = self._ask("skeptic", system, json.dumps(payload, default=str))
        except LLMError as exc:
            self._fallback("skeptic", exc)
            return []
        out = []
        for c in data.get("challenges") or []:
            if isinstance(c, dict) and c.get("cause") in ASSESSABLE and c.get("question"):
                out.append(Challenge(c["cause"], str(c["question"])[:400]))
        return out[:3]

    def write(self, event, findings, ctx):
        gap = call_tool(
            ctx,
            "supply_gap",
            {"region": event.region, "at": event.peak_ts, "lookback_minutes": 180},
        )
        profile = call_tool(
            ctx,
            "price_profile",
            {"region": event.region, "start": event.start_ts, "end": event.end_ts},
        )
        system = self._common + "\n\n" + _prompt("writer")
        payload = {
            "event": event_brief(event),
            "findings": [_finding_view(f, ctx) for f in findings],
            "supply_gap": json.loads(gap.for_agent()),
            "price_profile": json.loads(profile.for_agent()),
        }
        try:
            data = self._ask("writer", system, json.dumps(payload, default=str), max_tokens=2500)
            return report_from_writer(event, data, gap, profile, brain=self.name)
        except LLMError as exc:
            self._fallback("writer", exc)
            return compose_report(event, findings, gap, profile, brain=self.name + "+rules-writer")

    def repair(self, event, report: Report, violations: list[dict[str, str]], ctx) -> Report:
        """One attempt to fix claims the verifier rejected."""
        system = (
            self._common
            + "\n\n"
            + _prompt("writer")
            + (
                "\n\nThe verifier rejected some claims. Fix them: copy numbers exactly from the cited evidence, cite the right evidence IDs, or drop the claim."
            )
        )
        evidence = [json.loads(e.for_agent()) for e in ctx.ledger.all() if e.status == "ok"]
        payload = {"draft": report.to_dict(), "violations": violations, "evidence": evidence}
        try:
            data = self._ask(
                "writer:repair", system, json.dumps(payload, default=str), max_tokens=2500
            )
        except LLMError as exc:
            self._fallback("writer:repair", exc)
            return report
        gap = next((e for e in ctx.ledger.all() if e.tool == "supply_gap"), None)
        profile = next((e for e in ctx.ledger.all() if e.tool == "price_profile"), None)
        if gap is None or profile is None:
            return report
        fixed = report_from_writer(event, data, gap, profile, brain=self.name)
        fixed.run = report.run
        return fixed


def _finding_view(f: Finding, ctx: ToolContext) -> dict[str, Any]:
    cited = {eid for c in f.claims for eid in c.evidence_ids}
    return {
        "cause": f.cause,
        "verdict": f.verdict,
        "confidence": f.confidence,
        "explanation": f.explanation,
        "claims": [{"text": c.text, "evidence_ids": c.evidence_ids} for c in f.claims],
        "evidence": [
            json.loads(ev.for_agent(max_rows=6))
            for eid in sorted(cited)
            if (ev := ctx.ledger.get(eid))
        ],
    }


def report_from_writer(
    event: SpikeEvent, data: dict[str, Any], gap: Evidence, profile: Evidence, *, brain: str
) -> Report:
    def entries(key: str) -> list[CauseEntry]:
        out = []
        for item in data.get(key) or []:
            if not isinstance(item, dict) or item.get("cause") not in CAUSES:
                continue
            claims = [
                Claim(str(c["text"]), [str(i) for i in (c.get("evidence_ids") or [])])
                for c in item.get("claims") or []
                if isinstance(c, dict) and c.get("text")
            ]
            conf = str(item.get("confidence", "low")).lower()
            out.append(
                CauseEntry(
                    item["cause"], conf if conf in ("high", "medium", "low") else "low", claims
                )
            )
        return out

    features: dict[str, float] = {
        "peak_rrp": float(profile.facts.get("peak_rrp", event.peak_rrp))
        if profile.status == "ok"
        else event.peak_rrp
    }
    if gap.status == "ok":
        features.update(
            {
                k: float(v)
                for k, v in gap.facts.items()
                if k.endswith("_mw") and isinstance(v, int | float)
            }
        )
    na = [str(x) for x in data.get("not_assessable") or ["rebidding"]]
    return Report(
        event_id=event.event_id,
        region=event.region,
        peak_ts=event.peak_ts,
        peak_rrp=features["peak_rrp"],
        headline=str(data.get("headline", ""))[:400]
        or f"{event.region} price spike at {event.peak_ts}.",
        causes=entries("causes"),
        ruled_out=entries("ruled_out"),
        not_assessable=na,
        caveats=[str(c)[:300] for c in data.get("caveats") or []],
        features=features,
        run={"brain": brain},
    )
