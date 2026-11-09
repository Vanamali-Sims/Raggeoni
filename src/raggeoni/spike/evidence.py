"""Evidence ledger: every tool result gets an ID, and every number is kept.

The verifier later checks that each number in the final report appears in
the evidence the report cites. That only works if tools put their numbers
in ``facts`` rather than burying them in prose.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

Number = int | float
_NUM_IN_TEXT = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass
class Evidence:
    evidence_id: str
    tool: str
    args: dict[str, Any]
    status: str  # ok | unavailable | error
    summary: str
    facts: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def for_agent(self, max_rows: int = 12) -> str:
        """Compact JSON the LLM sees. Rows are truncated to keep prompts small."""
        payload: dict[str, Any] = {
            "evidence_id": self.evidence_id,
            "tool": self.tool,
            "status": self.status,
            "summary": self.summary,
            "facts": self.facts,
        }
        if self.rows:
            payload["rows"] = self.rows[:max_rows]
            if len(self.rows) > max_rows:
                payload["rows_truncated"] = len(self.rows) - max_rows
        return json.dumps(payload, default=str)

    def numbers(self) -> list[float]:
        """Every number this evidence can ground: facts, rows, arguments, and
        numbers quoted inside text it returned (e.g. "275 kV" in a notice)."""
        found: list[float] = []

        def walk(value: Any) -> None:
            if isinstance(value, bool):
                return
            if isinstance(value, int | float):
                found.append(float(value))
            elif isinstance(value, str):
                found.extend(float(m.replace(",", "")) for m in _NUM_IN_TEXT.findall(value))
            elif isinstance(value, dict):
                for v in value.values():
                    walk(v)
            elif isinstance(value, list | tuple):
                for v in value:
                    walk(v)

        walk(self.facts)
        walk(self.rows)
        walk(self.args)
        return found


class Ledger:
    def __init__(self) -> None:
        self._items: dict[str, Evidence] = {}
        self._lock = threading.Lock()  # evidence agents run in parallel

    def add(
        self,
        tool: str,
        args: dict[str, Any],
        *,
        status: str,
        summary: str,
        facts: dict[str, Any] | None = None,
        rows: list[dict[str, Any]] | None = None,
        started: float | None = None,
    ) -> Evidence:
        with self._lock:
            ev = Evidence(
                evidence_id=f"E{len(self._items) + 1}",
                tool=tool,
                args=args,
                status=status,
                summary=summary,
                facts=facts or {},
                rows=rows or [],
                elapsed_ms=int((time.perf_counter() - started) * 1000) if started else 0,
            )
            self._items[ev.evidence_id] = ev
        return ev

    def get(self, evidence_id: str) -> Evidence | None:
        return self._items.get(evidence_id)

    def all(self) -> list[Evidence]:
        return list(self._items.values())

    def __len__(self) -> int:
        return len(self._items)
