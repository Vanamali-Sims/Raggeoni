"""Static site for GitHub Pages: the investigation feed, one page per report, system health.

No framework: the Markdown we generate is simple, so a small converter
covers it, and the site deploys as plain files.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from raggeoni.spike.config import health_dir, investigations_dir
from raggeoni.spike.report import LABELS

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a64;--card:#fff;--line:#e6e3dc;--accent:#c2410c;--ok:#15803d;--bad:#b91c1c}
@media (prefers-color-scheme:dark){:root{--bg:#141412;--fg:#ecebe6;--muted:#a3a29b;--card:#1d1d1a;--line:#2f2e2a;--accent:#fb923c;--ok:#4ade80;--bad:#f87171}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:880px;margin:0 auto;padding:32px 16px 64px}a{color:var(--accent)}
header p{color:var(--muted);margin-top:4px}h1{font-size:1.7rem;margin:0}h2{margin-top:2rem;font-size:1.2rem}h3{font-size:1.05rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
.card a{text-decoration:none;color:inherit}.card .meta{color:var(--muted);font-size:.9rem}
.tag{display:inline-block;font-size:.78rem;border:1px solid var(--line);border-radius:999px;padding:1px 8px;margin:4px 4px 0 0}
.price{font-variant-numeric:tabular-nums;font-weight:650;color:var(--accent)}.ok{color:var(--ok)}.bad{color:var(--bad)}
table{border-collapse:collapse;width:100%;font-size:.88rem;display:block;overflow-x:auto}td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
blockquote{margin:12px 0;padding:8px 12px;border-left:3px solid var(--accent);background:var(--card)}code{font-size:.85em}
nav{display:flex;gap:16px;margin:12px 0 24px}
"""


def _inline(text: str) -> str:
    out = html.escape(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<![\w*])_(.+?)_(?!\w)", r"<em>\1</em>", out)
    out = re.sub(r"`(.+?)`", r"<code>\1</code>", out)
    return out


def md_to_html(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            line.startswith("|")
            and i + 1 < len(lines)
            and set(lines[i + 1].replace("|", "").strip()) <= {"-", ":", " "}
        ):
            header = [c.strip() for c in line.strip("|").split("|")]
            out.append(
                "<table><thead><tr>"
                + "".join(f"<th>{_inline(c)}</th>" for c in header)
                + "</tr></thead><tbody>"
            )
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip("|").split("|")]
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>")
                i += 1
            out.append("</tbody></table>")
            continue
        if line.startswith("- "):
            out.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                out.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            out.append("</ul>")
            continue
        m = re.match(r"^(#{1,4}) (.*)", line)
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{_inline(line[2:])}</blockquote>")
        elif line.strip():
            out.append(f"<p>{_inline(line)}</p>")
        i += 1
    return "\n".join(out)


def page(title: str, body: str, *, depth: int = 0) -> str:
    up = "../" * depth
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>"
        f"<header><h1>NEM Spike Investigator</h1><p>Why did the price spike? Agents investigate, every number is checked against the data.</p></header>"
        f'<nav><a href="{up}index.html">Investigations</a><a href="{up}health.html">System health</a>'
        f'<a href="https://github.com/Vanamali-Sims/Raggeoni">Code</a></nav>{body}</main></body></html>'
    )


def load_reports(root: Path) -> list[dict[str, Any]]:
    reports = []
    for path in root.glob("*/report.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        data["_md"] = (
            (path.parent / "report.md").read_text(encoding="utf-8")
            if (path.parent / "report.md").exists()
            else ""
        )
        reports.append(data)
    reports.sort(key=lambda r: r["peak_ts"], reverse=True)
    return reports


def build_site(
    out: Path, *, investigations: Path | None = None, health: Path | None = None
) -> Path:
    investigations = investigations or investigations_dir()
    health = health or health_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "events").mkdir(exist_ok=True)
    reports = load_reports(investigations)
    cards = []
    for r in reports:
        v = r.get("verification", {})
        tags = "".join(
            f'<span class="tag">{html.escape(LABELS.get(c["cause"], c["cause"]))} · {c["confidence"]}</span>'
            for c in r.get("causes", [])
        )
        status = (
            '<span class="ok">verified</span>'
            if v.get("passed")
            else '<span class="bad">unverified</span>'
        )
        synthetic = (
            ' <span class="tag">synthetic demo</span>'
            if r["event_id"].startswith(("SA1-2031",))
            else ""
        )
        cards.append(
            f'<div class="card"><a href="events/{r["event_id"]}.html"><div class="meta">{html.escape(r["region"])} · {html.escape(r["peak_ts"])} NEM time · {status}{synthetic}</div>'
            f'<div><span class="price">${r["peak_rrp"]:,.0f}/MWh</span> {html.escape(r["headline"])}</div><div>{tags}</div></a></div>'
        )
        (out / "events" / f"{r['event_id']}.html").write_text(
            page(r["event_id"], md_to_html(r["_md"]), depth=1), encoding="utf-8"
        )
    feed = (
        "".join(cards)
        or "<p>No investigations yet. The watcher opens one when a region's price crosses the threshold.</p>"
    )
    (out / "index.html").write_text(
        page("Investigations", f"<h2>Latest investigations</h2>{feed}"), encoding="utf-8"
    )
    health_md = (
        (health / "README.md").read_text(encoding="utf-8")
        if (health / "README.md").exists()
        else "# System health\n\nThe Warden has not run yet."
    )
    (out / "health.html").write_text(page("System health", md_to_html(health_md)), encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    return out
