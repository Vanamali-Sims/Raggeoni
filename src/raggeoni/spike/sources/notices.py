"""AEMO market notices: parse, tag with regions, and fetch from NEMWeb.

Market notices are short plain-text reports (reserve warnings, unit trips,
network outages, constraint changes, interconnector limits). They are the
"why" that the numbers alone can't show, so they are the RAG corpus.

Files live in ``Reports/Current/Market_Notice/`` and are named like
``NEMITWEB1_MKTNOTICE_20260126.R131234``. The body looks like::

    Creation Date :     26/01/2026     18:02:11
    Notice ID               :         131234
    Notice Type ID          :         RESERVE NOTICE
    Notice Type Description :         LRC/LOR1/LOR2/LOR3
    Issue Date              :         26/01/2026
    External Reference      :         Actual LOR1 in the SA region - 26/01/2026
    Reason :
    ...free text...

Notice text is untrusted input. It is stored and shown to agents as data,
never as instructions (see ``rag.render_for_agent``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

from raggeoni.spike.config import INTERCONNECTORS, fmt_ts
from raggeoni.spike.sources.http import fetch, list_links

NOTICE_FOLDER = "https://nemweb.com.au/Reports/Current/Market_Notice/"

_FIELD = re.compile(r"^\s*([A-Za-z][A-Za-z ]+?)\s*:\s*(.*)$")
_FILE = re.compile(r"MKTNOTICE_(\d{8})\.R(\d+)", re.IGNORECASE)

_REGION_PATTERNS: dict[str, re.Pattern[str]] = {
    "NSW1": re.compile(r"\b(NSW1?|New South Wales)\b", re.IGNORECASE),
    "QLD1": re.compile(r"\b(QLD1?|Queensland)\b", re.IGNORECASE),
    "VIC1": re.compile(r"\b(VIC1?|Victoria)\b", re.IGNORECASE),
    "SA1": re.compile(r"\b(SA1?|South Australia)\b"),
    "TAS1": re.compile(r"\b(TAS1?|Tasmania)\b", re.IGNORECASE),
}


@dataclass(frozen=True)
class Notice:
    notice_id: str
    issued_ts: str  # NEM time
    notice_type: str
    description: str
    external_ref: str
    body: str
    regions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def text(self) -> str:
        return f"{self.notice_type}. {self.description}. {self.external_ref}. {self.body}".strip()


def regions_in(text: str) -> tuple[str, ...]:
    found = {code for code, pat in _REGION_PATTERNS.items() if pat.search(text)}
    for ic in INTERCONNECTORS:
        if re.search(rf"\b{re.escape(ic.name)}\b|\b{re.escape(ic.id)}\b", text, re.IGNORECASE):
            found.update({ic.from_region, ic.to_region})
    return tuple(sorted(found))


def _parse_dt(value: str) -> datetime | None:
    value = " ".join(value.split())
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def parse_notice(text: str, *, fallback_id: str = "") -> Notice | None:
    fields: dict[str, str] = {}
    body_lines: list[str] = []
    in_body = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if in_body:
            if line.strip().upper().startswith("END OF REPORT"):
                break
            if set(line.strip()) <= {"-"} and line.strip():
                continue
            body_lines.append(line.strip())
            continue
        match = _FIELD.match(line)
        if match:
            name = match.group(1).strip().lower()
            fields[name] = match.group(2).strip()
            if name == "reason":
                in_body = True
                if fields[name]:
                    body_lines.append(fields[name])
    notice_id = fields.get("notice id") or fallback_id
    if not notice_id:
        return None
    issued = _parse_dt(fields.get("creation date", "")) or _parse_dt(fields.get("issue date", ""))
    if issued is None:
        return None
    body = re.sub(r"\n{3,}", "\n\n", "\n".join(body_lines)).strip()
    notice_type = fields.get("notice type id", "")
    description = fields.get("notice type description", "")
    external_ref = fields.get("external reference", "")
    all_text = " ".join([notice_type, description, external_ref, body])
    return Notice(
        notice_id=notice_id,
        issued_ts=fmt_ts(issued),
        notice_type=notice_type,
        description=description,
        external_ref=external_ref,
        body=body[:4000],
        regions=regions_in(all_text),
    )


def list_notice_files(start: date, end: date) -> list[tuple[str, str]]:
    """Return (url, notice_id) for notices whose file date is within [start, end]."""
    html = fetch(NOTICE_FOLDER).decode("utf-8", errors="replace")
    lo, hi = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    out: list[tuple[str, str]] = []
    for href in list_links(html):
        match = _FILE.search(href)
        if not match or not lo <= match.group(1) <= hi:
            continue
        url = (
            href
            if href.startswith("http")
            else "https://nemweb.com.au" + href
            if href.startswith("/")
            else NOTICE_FOLDER + href
        )
        out.append((url, match.group(2)))
    return out


def fetch_notices(start: date, end: date) -> list[Notice]:
    notices: list[Notice] = []
    for url, notice_id in list_notice_files(start, end):
        parsed = parse_notice(fetch(url).decode("utf-8", errors="replace"), fallback_id=notice_id)
        if parsed is not None:
            notices.append(parsed)
    return notices
