"""Grounding verifier: a deterministic check that the report doesn't invent numbers.

For every claim:

* it must cite at least one evidence ID that exists and whose tool succeeded;
* every quantity in the text must match a number in the cited evidence
  (within rounding: 0.5 absolute or 1.5% relative, sign-insensitive, because
  "wind fell 600 MW" legitimately quotes a change of -600);
* every notice it mentions ("notice 131302", "#131302") must have been
  returned by a cited search and be issued inside the investigation window.

Times, dates and small counts ("two links") are not treated as quantities.
This is not an LLM judging an LLM: it is string and number matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from raggeoni.spike.config import fmt_ts, parse_ts
from raggeoni.spike.evidence import Ledger
from raggeoni.spike.schema import Claim, Report

_TIME = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b|\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b|\b\d{1,2}\s*(?:am|pm)\b",
    re.IGNORECASE,
)
_NOTICE = re.compile(r"(?:notice\s*(?:id\s*)?#?|#)(\d{5,7})\b", re.IGNORECASE)
_EVIDENCE_REF = re.compile(r"\bE\d+\b")
_REGION = re.compile(r"\b(?:NSW|QLD|VIC|SA|TAS)1\b")
_NUMBER = re.compile(
    r"(?P<dollar>\$)?(?P<num>[-+−]?\d[\d,]*(?:\.\d+)?)\s*(?P<unit>%|percent|mw\b|mwh\b|/mwh|k\b)?",
    re.IGNORECASE,
)


@dataclass
class Violation:
    claim: str
    problem: str


@dataclass
class VerificationResult:
    passed: bool
    claims_checked: int
    numbers_checked: int
    numbers_grounded: int
    violations: list[Violation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "claims_checked": self.claims_checked,
            "numbers_checked": self.numbers_checked,
            "numbers_grounded": self.numbers_grounded,
            "violations": [{"claim": v.claim, "problem": v.problem} for v in self.violations],
        }


def quantities(text: str) -> list[float]:
    """Numbers in a claim that should be grounded. Skips times, dates, IDs, regions."""
    cleaned = _TIME.sub(" ", text)
    cleaned = _NOTICE.sub(" ", cleaned)
    cleaned = _EVIDENCE_REF.sub(" ", cleaned)
    cleaned = _REGION.sub(" ", cleaned)
    out: list[float] = []
    for match in _NUMBER.finditer(cleaned):
        raw = match.group("num").replace(",", "").replace("−", "-")
        try:
            value = float(raw)
        except ValueError:
            continue
        unit = (match.group("unit") or "").lower()
        if unit == "k":
            value *= 1000
        has_unit = bool(match.group("dollar") or unit)
        # Small bare integers are usually counts ("two of three links", "5 intervals").
        if not has_unit and abs(value) < 10 and value == int(value):
            continue
        out.append(value)
    return out


def _matches(value: float, candidates: list[float]) -> bool:
    for c in candidates:
        for a, b in ((value, c), (abs(value), abs(c))):
            if abs(a - b) <= max(0.5, 0.015 * abs(b)):
                return True
    return False


def check_claim(
    claim: Claim, ledger: Ledger, window: tuple[str, str]
) -> tuple[list[Violation], int, int]:
    violations: list[Violation] = []
    cited = [ledger.get(eid) for eid in claim.evidence_ids]
    usable = [e for e in cited if e is not None and e.status == "ok"]
    if not claim.evidence_ids:
        violations.append(Violation(claim.text, "cites no evidence"))
    for eid, ev in zip(claim.evidence_ids, cited, strict=True):
        if ev is None:
            violations.append(Violation(claim.text, f"cites {eid}, which does not exist"))
        elif ev.status != "ok":
            violations.append(
                Violation(claim.text, f"cites {eid}, whose tool returned {ev.status}")
            )
    pool: list[float] = [n for e in usable for n in e.numbers()]
    nums = quantities(claim.text)
    grounded = 0
    for value in nums:
        if _matches(value, pool):
            grounded += 1
        else:
            violations.append(
                Violation(claim.text, f"number {value:g} is not in the cited evidence")
            )
    lo = fmt_ts(parse_ts(window[0]) - timedelta(hours=24))
    hi = window[1]
    for notice_id in _NOTICE.findall(claim.text):
        rows = [
            r
            for e in usable
            if e.tool == "search_notices"
            for r in e.rows
            if str(r.get("notice_id")) == notice_id
        ]
        if not rows:
            violations.append(
                Violation(claim.text, f"notice {notice_id} was not returned by a cited search")
            )
        elif not lo <= str(rows[0]["issued_ts"]) <= hi:
            violations.append(
                Violation(claim.text, f"notice {notice_id} is outside the event window")
            )
        elif rows[0].get("flagged_as_suspicious"):
            violations.append(
                Violation(
                    claim.text, f"notice {notice_id} was flagged as a possible prompt injection"
                )
            )
    return violations, len(nums), grounded


def verify_report(report: Report, ledger: Ledger, window: tuple[str, str]) -> VerificationResult:
    violations: list[Violation] = []
    checked = grounded = claims = 0
    for entry in [*report.causes, *report.ruled_out]:
        for claim in entry.claims:
            v, n, g = check_claim(claim, ledger, window)
            violations.extend(v)
            checked += n
            grounded += g
            claims += 1
    if claims == 0:
        violations.append(Violation("(report)", "report contains no claims"))
    for num in quantities(report.headline):
        if not _matches(num, [n for e in ledger.all() if e.status == "ok" for n in e.numbers()]):
            violations.append(
                Violation(report.headline, f"headline number {num:g} is not in any evidence")
            )
    return VerificationResult(not violations, claims, checked, grounded, violations)


def drop_ungrounded(report: Report, ledger: Ledger, window: tuple[str, str]) -> int:
    """Remove claims that fail verification. Returns how many were removed."""
    removed = 0
    for entry in [*report.causes, *report.ruled_out]:
        kept = []
        for claim in entry.claims:
            if check_claim(claim, ledger, window)[0]:
                removed += 1
            else:
                kept.append(claim)
        entry.claims = kept
    report.causes = [c for c in report.causes if c.claims]
    report.ruled_out = [c for c in report.ruled_out if c.claims]
    return removed
