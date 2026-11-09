"""Shared types: cause taxonomy, hypotheses, findings and the final report."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Cause labels. The first six can be tested with public dispatch data, notices
# and generation. Rebidding needs bid data the system does not load, so agents
# must call it "not assessable" rather than guess; the exam ignores it.
CAUSES: dict[str, str] = {
    "high_demand": "Demand unusually high for the time of day (heat or cold) or well above forecast",
    "low_wind": "Wind output unusually low or falling sharply",
    "solar_drop": "Solar output falling faster or further than usual (cloud, storm, sunset ramp with high demand)",
    "network_limit": "Interconnector or network limits stopping cheaper power reaching the region (outage, constraint, reduced limit)",
    "generator_outage": "A generator tripped, failed to start, or was unavailable",
    "battery_limited": "Batteries had little energy or offered little capacity when needed",
    "rebidding": "Participants moved capacity to high prices (needs bid data; not assessable here)",
}
ASSESSABLE = (
    "high_demand",
    "low_wind",
    "solar_drop",
    "network_limit",
    "generator_outage",
    "battery_limited",
)
VERDICTS = ("supports", "contradicts", "inconclusive")
CONFIDENCE = ("high", "medium", "low")


@dataclass
class Hypothesis:
    cause: str
    rationale: str
    priority: int = 1


@dataclass
class Claim:
    text: str
    evidence_ids: list[str]


@dataclass
class Finding:
    cause: str
    verdict: str  # supports | contradicts | inconclusive
    confidence: str  # high | medium | low
    claims: list[Claim]
    explanation: str = ""
    round: int = 1
    error: str | None = None

    @staticmethod
    def from_dict(data: dict[str, Any], *, cause: str, round_: int) -> Finding:
        verdict = str(data.get("verdict", "inconclusive")).lower()
        confidence = str(data.get("confidence", "low")).lower()
        claims = []
        for c in data.get("claims") or []:
            if isinstance(c, dict) and c.get("text"):
                ids = c.get("evidence_ids") or []
                claims.append(
                    Claim(
                        str(c["text"]),
                        [str(i) for i in ids] if isinstance(ids, list) else [str(ids)],
                    )
                )
        return Finding(
            cause=cause,
            verdict=verdict if verdict in VERDICTS else "inconclusive",
            confidence=confidence if confidence in CONFIDENCE else "low",
            claims=claims,
            explanation=str(data.get("explanation", ""))[:1500],
            round=round_,
        )


@dataclass
class Challenge:
    cause: str
    question: str


@dataclass
class CauseEntry:
    cause: str
    confidence: str
    claims: list[Claim]


@dataclass
class Report:
    event_id: str
    region: str
    peak_ts: str
    peak_rrp: float
    headline: str
    causes: list[CauseEntry]
    ruled_out: list[CauseEntry]
    not_assessable: list[str]
    caveats: list[str] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def predicted_causes(self, min_confidence: str = "medium") -> list[str]:
        rank = {"high": 3, "medium": 2, "low": 1}
        return [c.cause for c in self.causes if rank.get(c.confidence, 0) >= rank[min_confidence]]
