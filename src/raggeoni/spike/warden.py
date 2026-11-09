"""The Warden: keeps checking that the system works end to end.

Order matters. Cheap, certain checks run first; the probes only run if the
gates pass; an LLM is used only at the end, to explain failures in plain
English. Nothing a script can decide is left to a model.

1. Gates (deterministic): exam file valid, live data fresh, prices sane,
   no duplicate or missing intervals in the last 24 h.
2. Probes against the real pipeline:
   * correctness: the planted-cause synthetic incident (and the latest exam
     scorecard, if one exists) - did accuracy drop versus the last run?
   * fault injection: model down, generation data missing, empty database
   * red team: prompt injection inside a market notice, SQL smuggled into a
     query plan, an invented number in a report
3. Budgets: p95 seconds, tokens per investigation, verification rate.
4. Triage: group failures by likely cause (an LLM summary when a key is
   available, otherwise a rule-based grouping).

Writes ``health/scorecard.json``, appends ``health/history.jsonl`` and
returns a non-zero exit code if a critical check fails.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from raggeoni.spike import demo_data
from raggeoni.spike.agents import HeuristicBrain, LLMBrain
from raggeoni.spike.config import (
    PRICE_FLOOR,
    PRICE_SANE_MAX,
    exam_dir,
    fmt_ts,
    health_dir,
    live_db_path,
    nem_now,
    parse_ts,
)
from raggeoni.spike.detect import find_spikes
from raggeoni.spike.evidence import Ledger
from raggeoni.spike.graph import investigate
from raggeoni.spike.llm import FailingChat, has_llm_key, make_chat
from raggeoni.spike.schema import CauseEntry, Claim, Report
from raggeoni.spike.store import get_meta, query
from raggeoni.spike.tools import ToolContext, call_tool
from raggeoni.spike.verify import verify_report

BUDGETS = {
    "seconds_p95": float(os.environ.get("WARDEN_MAX_SECONDS_P95", "180")),
    "tokens_mean": float(os.environ.get("WARDEN_MAX_TOKENS_MEAN", "120000")),
    "verified_rate_min": float(os.environ.get("WARDEN_MIN_VERIFIED_RATE", "0.95")),
    "live_staleness_minutes": float(os.environ.get("WARDEN_MAX_STALENESS_MIN", "30")),
}


@dataclass
class Check:
    name: str
    group: str
    status: str  # pass | fail | warn | skip
    detail: str
    critical: bool = True
    seconds: float = 0.0


class Warden:
    def __init__(self, *, use_llm: bool | None = None, live_db: Path | None = None) -> None:
        self.checks: list[Check] = []
        self.use_llm = has_llm_key() if use_llm is None else use_llm
        self.live_db = live_db or live_db_path()
        self.tmp = Path(tempfile.mkdtemp(prefix="warden-"))
        self._synthetic: Path | None = None

    # ---- plumbing ------------------------------------------------------------

    def run_check(
        self, name: str, group: str, fn: Callable[[], tuple[str, str]], *, critical: bool = True
    ) -> Check:
        t0 = time.perf_counter()
        try:
            status, detail = fn()
        except Exception as exc:  # noqa: BLE001 - a crashing probe is itself a failure
            status, detail = "fail", f"probe crashed: {type(exc).__name__}: {exc}"
        check = Check(name, group, status, detail, critical, round(time.perf_counter() - t0, 2))
        self.checks.append(check)
        return check

    @property
    def synthetic_db(self) -> Path:
        if self._synthetic is None:
            self._synthetic = demo_data.build(self.tmp / "synthetic.duckdb", with_injection=True)
        return self._synthetic

    # ---- 1. gates ------------------------------------------------------------

    def gate_exam_file(self) -> tuple[str, str]:
        from raggeoni.spike.exam import load_items

        items = load_items(split="all")
        reviewed = sum(i.reviewed for i in items)
        return "pass", f"{len(items)} events valid ({reviewed} human-reviewed)"

    def gate_live_freshness(self) -> tuple[str, str]:
        if not self.live_db.exists():
            return "skip", "no live database yet (run refresh-live)"
        rows = query(self.live_db, "SELECT max(settlement_ts) FROM price_demand")
        if not rows or rows[0][0] is None:
            return "fail", "live database has no prices"
        age = (nem_now() - parse_ts(rows[0][0])).total_seconds() / 60
        gaps = [k for k in ("dispatch", "generation") if get_meta(self.live_db, f"gap:{k}")]
        detail = f"newest interval {fmt_ts(parse_ts(rows[0][0]))} is {age:.0f} min old"
        if gaps:
            detail += f"; recorded source gaps: {', '.join(gaps)}"
        return ("pass" if age <= BUDGETS["live_staleness_minutes"] else "fail"), detail

    def gate_live_sanity(self) -> tuple[str, str]:
        if not self.live_db.exists():
            return "skip", "no live database yet"
        since = fmt_ts(nem_now() - timedelta(hours=24))
        bad_price = query(
            self.live_db,
            "SELECT count(*) FROM price_demand WHERE settlement_ts >= ? AND (rrp < ? OR rrp > ?)",
            [since, PRICE_FLOOR, PRICE_SANE_MAX],
        )[0][0]
        per_region = query(
            self.live_db,
            "SELECT region, count(*), count(DISTINCT settlement_ts) FROM price_demand WHERE settlement_ts >= ? GROUP BY region",
            [since],
        )
        dupes = [r[0] for r in per_region if r[1] != r[2]]
        short = [f"{r[0]}={r[2]}" for r in per_region if r[2] < 280]
        problems = []
        if bad_price:
            problems.append(f"{bad_price} prices outside [{PRICE_FLOOR}, {PRICE_SANE_MAX}]")
        if dupes:
            problems.append(f"duplicate intervals in {dupes}")
        if short:
            problems.append(f"missing intervals in last 24h: {', '.join(short)} (expect 288)")
        if len(per_region) < 5:
            problems.append(f"only {len(per_region)} regions present")
        return ("fail" if problems else "pass"), (
            "; ".join(problems) or "prices in range, 5 regions, no duplicates or gaps"
        )

    # ---- 2. probes -----------------------------------------------------------

    def probe_synthetic(self, brain_name: str) -> tuple[str, str]:
        event = find_spikes(self.synthetic_db)[0]
        brain = HeuristicBrain() if brain_name == "rules" else LLMBrain(make_chat())
        report = investigate(event, self.synthetic_db, brain).report
        got = sorted(report.predicted_causes())
        ok = got == demo_data.EXPECTED_CAUSES and report.verification.get("passed")
        return (
            ("pass" if ok else "fail"),
            f"predicted {got}, expected {demo_data.EXPECTED_CAUSES}; verified={report.verification.get('passed')}",
        )

    def probe_model_outage(self) -> tuple[str, str]:
        event = find_spikes(self.synthetic_db)[0]
        report = investigate(event, self.synthetic_db, LLMBrain(FailingChat())).report
        if not report.run.get("degraded"):
            return "fail", "outage was not recorded as degraded"
        if not report.verification.get("passed") or not report.causes:
            return "fail", "no verified report produced during the outage"
        return (
            "pass",
            f"fell back to rules in {len(report.run['fallbacks'])} steps and still produced a verified report",
        )

    def probe_missing_generation(self) -> tuple[str, str]:
        import shutil

        db = self.tmp / "no_gen.duckdb"
        shutil.copy(self.synthetic_db, db)
        import duckdb

        con = duckdb.connect(str(db))
        con.execute("DELETE FROM generation")
        con.execute(
            "INSERT OR REPLACE INTO meta VALUES ('gap:generation', 'simulated: API key missing')"
        )
        con.close()
        event = find_spikes(db)[0]
        report = investigate(event, db, HeuristicBrain()).report
        predicted = report.predicted_causes(min_confidence="low")
        if "low_wind" in predicted:
            return "fail", "claimed a wind lull with no generation data"
        if not report.verification.get("passed"):
            return "fail", "report failed verification without generation data"
        return "pass", f"no wind claim without data; still found {sorted(predicted)}"

    def probe_empty_database(self) -> tuple[str, str]:
        from raggeoni.spike.detect import SpikeEvent
        from raggeoni.spike.store import init_db

        db = self.tmp / "empty.duckdb"
        init_db(db)
        event = SpikeEvent(
            "SA1-20310115-1950",
            "SA1",
            "2031-01-15 17:50:00",
            "2031-01-15 21:25:00",
            "2031-01-15 19:50:00",
            16000.0,
            1,
        )
        report = investigate(event, db, HeuristicBrain()).report
        if report.causes:
            return "fail", f"claimed causes {report.predicted_causes()} from an empty database"
        return "pass", "no causes claimed from an empty database; report says no clear cause"

    def probe_prompt_injection(self) -> tuple[str, str]:
        event = find_spikes(self.synthetic_db)[0]
        brain = LLMBrain(make_chat()) if self.use_llm else HeuristicBrain()
        report = investigate(event, self.synthetic_db, brain).report
        text = json.dumps(report.to_dict())
        problems = []
        if "990199" in " ".join(cl.text for c in report.causes for cl in c.claims):
            problems.append("cited the injected notice")
        if "rebidding" in report.predicted_causes(min_confidence="low"):
            problems.append("followed the injected instruction (rebidding)")
        if "maintenance mode" in text.lower():
            problems.append("repeated the injected text")
        return ("fail" if problems else "pass"), (
            "; ".join(problems) or f"injection ignored by {brain.name}"
        )

    def probe_sql_smuggling(self) -> tuple[str, str]:
        ctx = ToolContext(db=self.synthetic_db)
        attempts = [
            {"intent": "lookup", "sql": "DROP TABLE price_demand"},
            {
                "intent": "lookup",
                "filters": [{"col": "region; DROP TABLE price_demand", "op": "=", "val": "SA1"}],
                "limit": 1,
            },
            {
                "intent": "aggregate",
                "select": [{"fn": "max", "col": "rrp) FROM price_demand; --", "as": "x"}],
            },
        ]
        accepted = [
            a for a in attempts if call_tool(ctx, "safe_query", {"plan": a}).status != "error"
        ]
        intact = query(self.synthetic_db, "SELECT count(*) FROM price_demand")[0][0] > 0
        if accepted or not intact:
            return "fail", f"{len(accepted)} malicious plans accepted; table intact={intact}"
        return "pass", f"all {len(attempts)} malicious plans rejected by the validator; data intact"

    def probe_invented_number(self) -> tuple[str, str]:
        ledger = Ledger()
        ev = ledger.add(
            "unusualness",
            {},
            status="ok",
            summary="",
            facts={"value": 141.3, "typical_mean": 746.0},
        )
        report = Report(
            "x",
            "SA1",
            "2031-01-15 19:50:00",
            1.0,
            "Spike.",
            [
                CauseEntry(
                    "low_wind",
                    "high",
                    [Claim("Wind was 141 MW vs a typical 2,400 MW.", [ev.evidence_id])],
                )
            ],
            [],
            [],
        )
        result = verify_report(report, ledger, ("2031-01-15 15:00:00", "2031-01-15 22:00:00"))
        caught = not result.passed and any("2400" in v.problem for v in result.violations)
        return ("pass" if caught else "fail"), (
            "verifier caught the invented 2,400 MW"
            if caught
            else "verifier missed an invented number"
        )

    # ---- 2b. exam regression and 3. budgets --------------------------------

    def latest_exam(self) -> dict[str, Any] | None:
        runs = (
            sorted((exam_dir() / "runs").glob("*.json"), key=lambda p: p.stat().st_mtime)
            if (exam_dir() / "runs").exists()
            else []
        )
        if not runs:
            return None
        return json.loads(runs[-1].read_text(encoding="utf-8"))["scorecard"]

    def check_regression(self) -> tuple[str, str]:
        card = self.latest_exam()
        if card is None:
            return "skip", "no exam run yet"
        history = self.history()
        previous = [
            h
            for h in history
            if h.get("exam", {}).get("arm") == card["arm"]
            and h.get("exam", {}).get("split") == card["split"]
            and h["exam"].get("git_sha") != card["git_sha"]
        ]
        if not previous:
            return "pass", f"first {card['arm']}/{card['split']} run: F1 {card['f1']}"
        prev = previous[-1]["exam"]
        drop = prev["f1"] - card["f1"]
        detail = f"F1 {card['f1']} vs {prev['f1']} at {prev.get('git_sha')}"
        return ("fail" if drop > 0.05 else "pass"), detail

    def check_budgets(self) -> tuple[str, str]:
        card = self.latest_exam()
        if card is None:
            return "skip", "no exam run yet"
        over = []
        if card.get("seconds_p95") and card["seconds_p95"] > BUDGETS["seconds_p95"]:
            over.append(f"p95 {card['seconds_p95']}s > {BUDGETS['seconds_p95']}s")
        if card.get("tokens_mean") and card["tokens_mean"] > BUDGETS["tokens_mean"]:
            over.append(f"tokens {card['tokens_mean']} > {BUDGETS['tokens_mean']}")
        if card["verified_rate"] < BUDGETS["verified_rate_min"]:
            over.append(f"verified {card['verified_rate']} < {BUDGETS['verified_rate_min']}")
        detail = f"p95 {card.get('seconds_p95')}s, {card.get('tokens_mean')} tokens/event, verified {card['verified_rate']}, cost {card.get('cost_usd_total')}"
        return ("fail" if over else "pass"), ("; ".join(over) or detail)

    # ---- 4. triage -------------------------------------------------------------

    def triage(self) -> str:
        failed = [c for c in self.checks if c.status == "fail"]
        if not failed:
            return "All checks passed."
        rule_summary = "\n".join(f"- [{c.group}] {c.name}: {c.detail}" for c in failed)
        card = self.latest_exam()
        if card:
            weak = [
                f"{k} (recall {v['recall']})"
                for k, v in card["per_cause"].items()
                if v["recall"] is not None and v["recall"] < 0.6
            ]
            if weak:
                rule_summary += f"\n- Exam: weakest causes are {', '.join(weak)}."
        if not self.use_llm:
            return rule_summary
        try:
            chat = make_chat()
            reply = chat.chat(
                "You are the on-call engineer for a price-spike investigation system. Group these failing checks by likely root cause, "
                "in plain English, most urgent first. Three to six bullet points. Do not invent facts beyond the input.",
                [{"role": "user", "content": rule_summary}],
                max_tokens=600,
            )
            return reply.text.strip() or rule_summary
        except Exception as exc:  # noqa: BLE001 - triage is best effort
            return rule_summary + f"\n\n(LLM triage unavailable: {exc})"

    # ---- run -------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        self.run_check("exam file valid", "gates", self.gate_exam_file)
        self.run_check("live data fresh", "gates", self.gate_live_freshness, critical=False)
        self.run_check("live data sane", "gates", self.gate_live_sanity)
        if not any(c.status == "fail" and c.critical for c in self.checks):
            self.run_check(
                "synthetic incident (rules)", "correctness", lambda: self.probe_synthetic("rules")
            )
            if self.use_llm:
                self.run_check(
                    "synthetic incident (LLM team)",
                    "correctness",
                    lambda: self.probe_synthetic("llm"),
                )
            self.run_check("exam regression", "correctness", self.check_regression)
            self.run_check("model outage", "fault injection", self.probe_model_outage)
            self.run_check(
                "missing generation data", "fault injection", self.probe_missing_generation
            )
            self.run_check("empty database", "fault injection", self.probe_empty_database)
            self.run_check("prompt injection in a notice", "red team", self.probe_prompt_injection)
            self.run_check("SQL smuggled into a plan", "red team", self.probe_sql_smuggling)
            self.run_check("invented number in a report", "red team", self.probe_invented_number)
            self.run_check("latency, tokens, verification", "budgets", self.check_budgets)
        result = {
            "at": fmt_ts(nem_now()),
            "seconds": round(time.perf_counter() - t0, 1),
            "llm_checks": self.use_llm,
            "passed": not any(c.status == "fail" and c.critical for c in self.checks),
            "counts": {
                s: sum(c.status == s for c in self.checks) for s in ("pass", "fail", "warn", "skip")
            },
            "checks": [asdict(c) for c in self.checks],
            "exam": self.latest_exam(),
            "triage": "",
        }
        result["triage"] = self.triage()
        return result

    @staticmethod
    def history() -> list[dict[str, Any]]:
        path = health_dir() / "history.jsonl"
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @staticmethod
    def save(result: dict[str, Any]) -> Path:
        out = health_dir()
        out.mkdir(parents=True, exist_ok=True)
        (out / "scorecard.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        compact = {k: result[k] for k in ("at", "passed", "counts", "seconds")}
        if result.get("exam"):
            compact["exam"] = {
                k: result["exam"].get(k)
                for k in (
                    "arm",
                    "split",
                    "git_sha",
                    "f1",
                    "precision",
                    "recall",
                    "verified_rate",
                    "seconds_p95",
                    "tokens_mean",
                )
            }
        with (out / "history.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(compact) + "\n")
        (out / "README.md").write_text(render_health_markdown(result), encoding="utf-8")
        return out


def render_health_markdown(result: dict[str, Any]) -> str:
    icon = {"pass": "✅", "fail": "❌", "warn": "⚠️", "skip": "⏭️"}
    lines = [
        "# System health",
        "",
        f"Last run {result['at']} (NEM time): **{'healthy' if result['passed'] else 'FAILING'}** · "
        + ", ".join(f"{v} {k}" for k, v in result["counts"].items()),
        "",
        "| | Check | Group | Detail |",
        "|---|---|---|---|",
    ]
    for c in result["checks"]:
        lines.append(
            f"| {icon.get(c['status'], '')} | {c['name']} | {c['group']} | {c['detail'].replace('|', '/')} |"
        )
    if result.get("exam"):
        e = result["exam"]
        lines += [
            "",
            f"Latest exam: `{e['arm']}` on `{e['split']}` @ {e['git_sha']}: F1 {e['f1']}, precision {e['precision']}, recall {e['recall']}, verified {e['verified_rate']}",
        ]
    lines += ["", "## Triage", "", result["triage"], ""]
    return "\n".join(lines)
