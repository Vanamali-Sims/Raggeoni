"""Parse AEMO's MMS "C/I/D" CSV format.

Every NEMWeb report is a CSV where the first column says what the row is:

    C  comment / file header or footer
    I  column header for the table that follows: I,<category>,<report>,<version>,<cols...>
    D  data row for the most recent I row

One file can hold several tables (DISPATCH,PRICE and DISPATCH,REGIONSUM, say).
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from collections.abc import Iterable

Table = list[dict[str, str]]


def parse_mms(text: str) -> dict[str, Table]:
    """Return {"CATEGORY.REPORT": [row dict, ...]} for every table in the file."""
    tables: dict[str, Table] = defaultdict(list)
    header: list[str] | None = None
    key: str | None = None
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        kind = row[0].strip().upper()
        if kind == "I" and len(row) > 4:
            key = f"{row[1].strip().upper()}.{row[2].strip().upper()}"
            header = [c.strip().upper() for c in row[4:]]
        elif kind == "D" and header is not None and key is not None:
            values = row[4:]
            tables[key].append(dict(zip(header, values, strict=False)))
    return dict(tables)


def merge_tables(parts: Iterable[dict[str, Table]]) -> dict[str, Table]:
    merged: dict[str, Table] = defaultdict(list)
    for part in parts:
        for key, rows in part.items():
            merged[key].extend(rows)
    return dict(merged)


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def mms_ts(value: str) -> str:
    """MMS timestamps look like "2026/01/26 18:30:00". Return "2026-01-26 18:30:00"."""
    return value.strip().strip('"').replace("/", "-")
