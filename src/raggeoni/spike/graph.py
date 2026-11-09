"""The investigation as a LangGraph state machine.

    overview ─► lead ─┬─► evidence (one per hypothesis, in parallel) ─┐
                      │                                                ▼
                      │                                             skeptic
                      │                  challenges? ◄──────────────────┤
                      │    yes, round < max: evidence again (parallel)  │ no
                      │                                                 ▼
                      └────────────────────────────────────────────► writer ─► verify ─► END

* ``overview`` is deterministic: a few tools every investigation needs, so the
  lead starts from facts rather than the event headline alone.
* ``verify`` is deterministic too: the grounding check. If it fails, the LLM
  writer gets one repair attempt, then any claim that still fails is dropped
  and the report says so.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from raggeoni.spike.agents import Brain, LLMBrain
from raggeoni.spike.detect import SpikeEvent
from raggeoni.spike.evidence import Ledger
from raggeoni.spike.schema import Challenge, Finding, Hypothesis, Report
from raggeoni.spike.tools import ToolContext, call_tool
from raggeoni.spike.verify import drop_ungrounded, verify_report


def _merge_findings(left: list[Finding], right: list[Finding]) -> list[Finding]:
    """Later rounds replace earlier findings for the same cause."""
    merged = {f.cause: f for f in left}
    for f in right:
        if f.cause not in merged or f.round >= merged[f.cause].round:
            merged[f.cause] = f
    return list(merged.values())


class State(TypedDict, total=False):
    hypotheses: list[Hypothesis]
    findings: Annotated[list[Finding], _merge_findings]
    challenges: list[Challenge]
    round: int
    report: Report
    trace: Annotated[list[dict[str, Any]], operator.add]


@dataclass
class RunOptions:
    rag: bool = True
    skeptic: bool = True
    max_rounds: int = 2
    history_dir: Path | None = None
    label: str = ""


@dataclass
class Investigation:
    report: Report
    ledger: Ledger
    trace: list[dict[str, Any]] = field(default_factory=list)


def _timed(name: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, {"node": name, "ms": int((time.perf_counter() - t0) * 1000)}


def build_graph(event: SpikeEvent, ctx: ToolContext, brain: Brain, opts: RunOptions):
    start, end = (t.strftime("%Y-%m-%d %H:%M:%S") for t in event.window)

    def overview(state: State) -> State:
        t0 = time.perf_counter()
        call_tool(ctx, "price_profile", {"region": event.region, "start": start, "end": end})
        call_tool(ctx, "interconnector_status", {"region": event.region, "at": event.peak_ts})
        call_tool(
            ctx,
            "generation_change",
            {"region": event.region, "at": event.peak_ts, "lookback_minutes": 120},
        )
        call_tool(
            ctx,
            "supply_gap",
            {"region": event.region, "at": event.peak_ts, "lookback_minutes": 180},
        )
        return {
            "trace": [
                {
                    "node": "overview",
                    "ms": int((time.perf_counter() - t0) * 1000),
                    "evidence": len(ctx.ledger),
                }
            ],
            "round": 1,
        }

    def lead(state: State) -> State:
        hyps, tr = _timed("lead", brain.hypothesize, event, ctx.ledger.all(), ctx)
        tr["hypotheses"] = [h.cause for h in hyps]
        return {"hypotheses": hyps, "trace": [tr]}

    def fan_out(state: State) -> list[Send]:
        return [
            Send("evidence", {"hypothesis": h, "round": 1, "question": None})
            for h in state["hypotheses"]
        ]

    def evidence(task: dict[str, Any]) -> State:
        h: Hypothesis = task["hypothesis"]
        finding, tr = _timed(
            f"evidence:{h.cause}",
            brain.investigate,
            event,
            h,
            ctx,
            round_=task["round"],
            question=task["question"],
        )
        tr.update(
            {"round": task["round"], "verdict": finding.verdict, "confidence": finding.confidence}
        )
        return {"findings": [finding], "trace": [tr]}

    def skeptic(state: State) -> State:
        if not opts.skeptic or state.get("round", 1) >= opts.max_rounds:
            return {"challenges": [], "trace": [{"node": "skeptic", "ms": 0, "skipped": True}]}
        challenges, tr = _timed("skeptic", brain.challenge, event, state["findings"], ctx)
        tr["challenges"] = [f"{c.cause}: {c.question}" for c in challenges]
        return {"challenges": challenges, "round": state.get("round", 1) + 1, "trace": [tr]}

    def after_skeptic(state: State) -> list[Send] | str:
        challenges = state.get("challenges") or []
        if not challenges:
            return "writer"
        by_cause = {h.cause: h for h in state["hypotheses"]}
        sends = []
        for c in challenges:
            h = by_cause.get(c.cause) or Hypothesis(c.cause, "raised by the skeptic")
            sends.append(
                Send("evidence", {"hypothesis": h, "round": state["round"], "question": c.question})
            )
        return sends

    def writer(state: State) -> State:
        report, tr = _timed("writer", brain.write, event, state["findings"], ctx)
        return {"report": report, "trace": [tr]}

    def verify(state: State) -> State:
        t0 = time.perf_counter()
        report = state["report"]
        window = (start, end)
        result = verify_report(report, ctx.ledger, window)
        first = result.to_dict()  # keep the original verdict for the audit trail
        repaired = False
        if not result.passed and isinstance(brain, LLMBrain):
            report = brain.repair(event, report, first["violations"], ctx)
            result = verify_report(report, ctx.ledger, window)
            repaired = True
        dropped = 0
        if not result.passed:
            dropped = drop_ungrounded(report, ctx.ledger, window)
            result = verify_report(report, ctx.ledger, window)
        report.verification = {
            **result.to_dict(),
            "first_pass": first if repaired or dropped else None,
            "repaired": repaired,
            "claims_dropped": dropped,
        }
        return {
            "report": report,
            "trace": [
                {
                    "node": "verify",
                    "ms": int((time.perf_counter() - t0) * 1000),
                    "passed": result.passed,
                    "dropped": dropped,
                }
            ],
        }

    g = StateGraph(State)
    g.add_node("overview", overview)
    g.add_node("lead", lead)
    g.add_node("evidence", evidence)  # type: ignore[arg-type]  # receives Send payloads, not State
    g.add_node("skeptic", skeptic)
    g.add_node("writer", writer)
    g.add_node("verify", verify)
    g.add_edge(START, "overview")
    g.add_edge("overview", "lead")
    g.add_conditional_edges("lead", fan_out, ["evidence"])
    g.add_edge("evidence", "skeptic")
    g.add_conditional_edges("skeptic", after_skeptic, ["evidence", "writer"])
    g.add_edge("writer", "verify")
    g.add_edge("verify", END)
    return g.compile()


def investigate(
    event: SpikeEvent, db: Path, brain: Brain, opts: RunOptions | None = None
) -> Investigation:
    opts = opts or RunOptions()
    ctx = ToolContext(db=db, rag_enabled=opts.rag, history_dir=opts.history_dir)
    graph = build_graph(event, ctx, brain, opts)
    t0 = time.perf_counter()
    final = graph.invoke({"trace": [], "findings": []}, {"recursion_limit": 40})
    report: Report = final["report"]
    report.run.update(
        {
            "brain": brain.name,
            "options": {
                "rag": opts.rag,
                "skeptic": opts.skeptic,
                "max_rounds": opts.max_rounds,
                "label": opts.label,
            },
            "seconds": round(time.perf_counter() - t0, 2),
            "tool_calls": len(ctx.ledger),
            "llm": brain.usage.to_dict(),
            "fallbacks": list(brain.fallbacks),
            "degraded": bool(brain.fallbacks),
            "findings": [
                {
                    "cause": f.cause,
                    "verdict": f.verdict,
                    "confidence": f.confidence,
                    "round": f.round,
                    "error": f.error,
                }
                for f in final.get("findings", [])
            ],
        }
    )
    return Investigation(report=report, ledger=ctx.ledger, trace=final.get("trace", []))
