"""Agent team: heuristic baseline, LLM tool loop (with a scripted fake model),
graceful fallback when the model is down, verifier and prompt-injection resistance."""

from __future__ import annotations

import json

import pytest

from raggeoni.spike import demo_data
from raggeoni.spike.agents import HeuristicBrain, LLMBrain
from raggeoni.spike.detect import find_spikes
from raggeoni.spike.evidence import Ledger
from raggeoni.spike.graph import RunOptions, investigate
from raggeoni.spike.llm import FailingChat, LLMError, Reply, ToolCall
from raggeoni.spike.report import write_investigation
from raggeoni.spike.schema import CauseEntry, Claim, Report
from raggeoni.spike.verify import quantities, verify_report


@pytest.fixture(scope="module")
def incident(tmp_path_factory):
    db = demo_data.build(tmp_path_factory.mktemp("agents") / "demo.duckdb", with_injection=True)
    return db, find_spikes(db)[0]


def test_heuristic_finds_the_planted_causes(incident):
    db, event = incident
    inv = investigate(event, db, HeuristicBrain())
    assert sorted(inv.report.predicted_causes()) == demo_data.EXPECTED_CAUSES
    assert "solar_drop" in [c.cause for c in inv.report.ruled_out]
    assert inv.report.verification["passed"]
    assert inv.report.run["degraded"] is False


def test_injected_notice_is_never_cited(incident):
    db, event = incident
    report = investigate(event, db, HeuristicBrain()).report
    cited = " ".join(cl.text for c in report.causes for cl in c.claims)
    assert "990199" not in cited
    assert "rebidding" not in report.predicted_causes(min_confidence="low")


def test_rag_ablation_loses_the_generator_outage(incident):
    db, event = incident
    report = investigate(event, db, HeuristicBrain(), RunOptions(rag=False)).report
    assert "generator_outage" not in report.predicted_causes(min_confidence="low")
    assert "network_limit" in report.predicted_causes()  # still visible in the numbers


def test_failing_model_degrades_to_rules_and_still_reports(incident):
    db, event = incident
    brain = LLMBrain(FailingChat())
    inv = investigate(event, db, brain)
    assert inv.report.run["degraded"] is True
    assert inv.report.run["fallbacks"]
    assert sorted(inv.report.predicted_causes()) == demo_data.EXPECTED_CAUSES
    assert inv.report.verification["passed"]


def test_report_files(incident, tmp_path):
    db, event = incident
    folder = write_investigation(investigate(event, db, HeuristicBrain()), tmp_path)
    assert {p.name for p in folder.iterdir()} == {
        "report.json",
        "report.md",
        "evidence.json",
        "trace.json",
    }
    md = (folder / "report.md").read_text()
    assert "Wind lull" in md and "all numbers grounded" in md
    assert json.loads((folder / "report.json").read_text())["event_id"] == event.event_id


# ---- scripted fake model ------------------------------------------------------------------


class ScriptedChat:
    """A fake model that behaves like a careful agent: one tool call per evidence
    agent, then a JSON finding built from the real tool result it was given."""

    name = "scripted"

    def __init__(self, *, invent_number: bool = False, skeptic_question: bool = True):
        self.invent_number = invent_number
        self.skeptic_question = skeptic_question
        self.writer_calls = 0

    def tool_result_message(self, results):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": c.id, "content": body}
                    for c, body in results
                ],
            }
        ]

    def chat(self, system, messages, tools=None, max_tokens=1500):
        if "LEAD INVESTIGATOR" in system:
            hyps = [
                {"cause": c, "priority": i + 1, "rationale": "test"}
                for i, c in enumerate(["low_wind", "network_limit", "solar_drop"])
            ]
            return Reply(json.dumps({"hypotheses": hyps}), [], 100, 20)
        if "SKEPTIC" in system:
            ch = (
                [{"cause": "low_wind", "question": "Is this drop normal for the evening?"}]
                if self.skeptic_question
                else []
            )
            return Reply(json.dumps({"challenges": ch}), [], 80, 10)
        if "WRITER" in system:
            self.writer_calls += 1
            payload = json.loads(messages[0]["content"])
            causes = []
            for f in payload.get("findings", []):
                if f["verdict"] == "supports":
                    claims = f["claims"]
                    if self.invent_number and self.writer_calls == 1:
                        claims = [
                            *claims,
                            {
                                "text": "Wind fell by 9,999 MW.",
                                "evidence_ids": claims[0]["evidence_ids"],
                            },
                        ]
                    causes.append(
                        {"cause": f["cause"], "confidence": f["confidence"], "claims": claims}
                    )
            if "draft" in payload:  # repair call
                causes = [c for c in payload["draft"]["causes"]]
                for c in causes:
                    c["claims"] = [cl for cl in c["claims"] if "9,999" not in cl["text"]]
            return Reply(
                json.dumps(
                    {
                        "headline": "Spike in SA1.",
                        "causes": causes,
                        "ruled_out": [],
                        "not_assessable": ["rebidding"],
                        "caveats": [],
                    }
                ),
                [],
                300,
                120,
            )
        # evidence agent
        task = json.loads(messages[0]["content"])
        cause = task["hypothesis"]["cause"]
        event = task["event"]
        tool_results = [
            m for m in messages if m["role"] == "user" and isinstance(m["content"], list)
        ]
        if not tool_results:
            if cause == "network_limit":
                call = ToolCall(
                    "t1",
                    "interconnector_status",
                    {"region": event["region"], "at": event["peak_ts"]},
                )
            else:
                metric = "wind" if cause == "low_wind" else "solar"
                call = ToolCall(
                    "t1",
                    "unusualness",
                    {"region": event["region"], "metric": metric, "at": event["peak_ts"]},
                )
            return Reply(
                "",
                [call],
                200,
                30,
                raw_assistant={
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": call.name, "input": call.arguments}
                    ],
                },
            )
        ev = json.loads(tool_results[-1]["content"][0]["content"])
        f = ev["facts"]
        if cause == "network_limit":
            verdict = "supports" if f.get("heywood_at_limit") else "contradicts"
            text = f"Heywood was at its limit of {f['heywood_limit_in_mw']} MW."
        else:
            verdict = "supports" if f["unusual"] == "low" else "contradicts"
            text = f"Output was {f['value']} MW vs a typical {f['typical_mean']} MW."
        out = {
            "verdict": verdict,
            "confidence": "high",
            "claims": [{"text": text, "evidence_ids": [ev["evidence_id"]]}],
            "explanation": "scripted",
        }
        return Reply(json.dumps(out), [], 250, 60)


def test_llm_team_tool_loop_skeptic_round_and_usage(incident):
    db, event = incident
    chat = ScriptedChat()
    inv = investigate(event, db, LLMBrain(chat))
    report = inv.report
    assert set(report.predicted_causes()) == {"low_wind", "network_limit"}
    assert report.verification["passed"]
    nodes = [t["node"] for t in inv.trace]
    assert nodes.count("evidence:low_wind") == 2  # skeptic sent it back for round 2
    assert report.run["llm"]["calls"] >= 8
    assert report.run["llm"]["input_tokens"] > 0
    assert report.run["degraded"] is False


def test_invented_number_is_caught_and_repaired(incident):
    db, event = incident
    chat = ScriptedChat(invent_number=True, skeptic_question=False)
    report = investigate(event, db, LLMBrain(chat)).report
    v = report.verification
    assert v["repaired"] is True
    assert any("9999" in x["problem"] for x in v["first_pass"]["violations"])
    assert v["passed"]


def test_token_budget_stops_the_model_gracefully(incident):
    db, event = incident
    brain = LLMBrain(ScriptedChat(), token_budget=150)
    report = investigate(event, db, brain).report
    assert report.run["degraded"] is True
    assert any("budget" in f for f in report.run["fallbacks"])


# ---- verifier unit tests ------------------------------------------------------------------


def test_quantities_skip_times_ids_and_counts():
    text = "At 18:30 on 2026-01-26, SA1 hit $15,230.50/MWh; notice #131302 and E4 show two of 3 links at 650 MW (12.5%)."
    assert quantities(text) == [15230.5, 650.0, 12.5]


def test_verifier_flags_missing_and_failed_evidence():
    ledger = Ledger()
    ok = ledger.add("t", {}, status="ok", summary="", facts={"x": 120.4})
    bad = ledger.add("t", {}, status="unavailable", summary="")
    report = Report(
        "e",
        "SA1",
        "2031-01-15 19:00:00",
        1.0,
        "h",
        [
            CauseEntry(
                "low_wind",
                "high",
                [
                    Claim("Wind fell 120 MW.", [ok.evidence_id]),
                    Claim("Rose 50 MW.", [bad.evidence_id, "E99"]),
                ],
            )
        ],
        [],
        [],
    )
    result = verify_report(report, ledger, ("2031-01-15 15:00:00", "2031-01-15 21:00:00"))
    problems = [v.problem for v in result.violations]
    assert not result.passed
    assert any("unavailable" in p for p in problems) and any("E99" in p for p in problems)
    assert result.numbers_grounded == 1


def test_parse_errors_surface_as_llm_error():
    from raggeoni.spike.llm import parse_json_object

    with pytest.raises(LLMError):
        parse_json_object("I think it was the wind.")
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
