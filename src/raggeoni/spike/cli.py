"""Command line: ``raggeoni-spike <command>``.

demo            investigate the synthetic incident offline (no keys, no network)
refresh-live    pull the latest 5-minute data into data/spike/live.duckdb
notices         pull recent AEMO market notices into the live database
watch           refresh, detect new spikes, investigate the ones that have ended
investigate     investigate a window in any incident database
exam            build exam data, run an arm, compare with a baseline
warden          run the health checks and write health/
reconcile       compare live investigations with a new AER report
site            build the GitHub Pages site
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

from raggeoni.spike.config import (
    SPIKE_THRESHOLD,
    fmt_ts,
    investigations_dir,
    live_db_path,
    nem_now,
    parse_ts,
    repo_root,
)


def _brain(name: str):
    from raggeoni.spike.agents import HeuristicBrain, LLMBrain
    from raggeoni.spike.llm import make_chat

    return HeuristicBrain() if name == "rules" else LLMBrain(make_chat())


def cmd_demo(args: argparse.Namespace) -> int:
    from raggeoni.spike import demo_data
    from raggeoni.spike.detect import find_spikes
    from raggeoni.spike.graph import RunOptions, investigate
    from raggeoni.spike.report import write_investigation

    out = Path(args.out)
    db = demo_data.build(Path(tempfile.mkdtemp()) / "synthetic.duckdb", with_injection=True)
    event = find_spikes(db)[0]
    inv = investigate(event, db, _brain(args.brain), RunOptions(label="demo"))
    folder = write_investigation(inv, out)
    got, want = sorted(inv.report.predicted_causes()), demo_data.EXPECTED_CAUSES
    print(inv.report.headline)
    print(f"verified: {inv.report.verification['passed']} · predicted {got} · planted {want}")
    print(f"report: {folder / 'report.md'}")
    return 0 if got == want and inv.report.verification["passed"] else 1


def cmd_refresh_live(args: argparse.Namespace) -> int:
    from raggeoni.spike.ingest import refresh_live

    counts = refresh_live(live_db_path(), keep_days=args.keep_days)
    print(json.dumps(counts, indent=1))
    return 0 if isinstance(counts.get("dispatch"), int) else 1


def cmd_notices(args: argparse.Namespace) -> int:
    from raggeoni.spike.ingest import load_notices

    end = nem_now()
    n = load_notices(live_db_path(), end - timedelta(days=args.days), end)
    print(f"{n} notices stored")
    return 0


def _state_path() -> Path:
    return investigations_dir() / "_state.json"


def cmd_watch(args: argparse.Namespace) -> int:
    from raggeoni.spike.detect import find_spikes
    from raggeoni.spike.graph import RunOptions, investigate
    from raggeoni.spike.ingest import refresh_live
    from raggeoni.spike.report import write_investigation

    db = live_db_path()
    if not args.skip_refresh:
        print(json.dumps(refresh_live(db), indent=1), file=sys.stderr)
    state_path = _state_path()
    state = json.loads(state_path.read_text()) if state_path.exists() else {"investigated": []}
    done = set(state["investigated"])
    settle = nem_now() - timedelta(minutes=args.settle_minutes)
    new = [
        e
        for e in find_spikes(db, threshold=args.threshold)
        if e.event_id not in done and parse_ts(e.end_ts) <= settle
    ]
    ongoing = [
        e
        for e in find_spikes(db, threshold=args.threshold)
        if e.event_id not in done and parse_ts(e.end_ts) > settle
    ]
    for e in ongoing:
        print(f"ongoing, will investigate once settled: {e.event_id} peak ${e.peak_rrp:,.0f}")
    written = []
    for event in new[: args.max_events]:
        print(
            f"investigating {event.event_id} (peak ${event.peak_rrp:,.0f}/MWh at {event.peak_ts})"
        )
        inv = investigate(
            event,
            db,
            _brain(args.brain),
            RunOptions(history_dir=investigations_dir(), label="live"),
        )
        written.append(str(write_investigation(inv, investigations_dir())))
        done.add(event.event_id)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps({"investigated": sorted(done), "checked_at": fmt_ts(nem_now())}, indent=1)
    )
    print(json.dumps({"new_investigations": written}))
    return 0


def cmd_investigate(args: argparse.Namespace) -> int:
    from raggeoni.spike.detect import find_spikes
    from raggeoni.spike.graph import RunOptions, investigate
    from raggeoni.spike.report import write_investigation

    db = Path(args.db)
    events = [
        e
        for e in find_spikes(db, threshold=args.threshold, region=args.region)
        if args.start <= e.peak_ts <= args.end
    ]
    if not events:
        print("no spike above the threshold in that window", file=sys.stderr)
        return 1
    event = max(events, key=lambda e: e.peak_rrp)
    inv = investigate(
        event,
        db,
        _brain(args.brain),
        RunOptions(
            rag=not args.no_rag, skeptic=not args.no_skeptic, history_dir=investigations_dir()
        ),
    )
    folder = write_investigation(inv, Path(args.out) if args.out else investigations_dir())
    print(inv.report.headline)
    print(folder / "report.md")
    return 0


def cmd_exam(args: argparse.Namespace) -> int:
    from raggeoni.spike import exam

    if args.exam_cmd == "build-data":
        items = exam.load_items(split=args.split)
        if args.event:
            items = [i for i in items if i.id == args.event]
        for item in items:
            if exam.incident_db(item.id).exists() and not args.force:
                print(f"{item.id}: exists")
                continue
            print(f"{item.id}: building", flush=True)
            print(json.dumps(exam.build_data(item, with_forecasts=not args.no_forecasts)))
        return 0
    if args.synthetic:
        card, results = exam.run_synthetic(args.arm, Path(tempfile.mkdtemp()))
    else:
        card, results = exam.run_exam(
            args.arm, args.split, reviewed_only=args.reviewed_only, limit=args.limit
        )
    comparison = None
    if args.compare and Path(args.compare).exists():
        comparison = exam.compare(card, json.loads(Path(args.compare).read_text())["scorecard"])
    path = exam.save_run(card, results)
    summary = exam.markdown_summary(card, results, comparison)
    print(summary)
    print(f"\nsaved {path}")
    if args.summary_md:
        Path(args.summary_md).write_text(summary, encoding="utf-8")
    if card["scored"] == 0:
        print("no events had data; run: raggeoni-spike exam build-data", file=sys.stderr)
        return 1
    return 1 if comparison and comparison["regressed"] else 0


def cmd_warden(args: argparse.Namespace) -> int:
    from raggeoni.spike.warden import Warden, render_health_markdown

    warden = Warden(use_llm=False if args.no_llm else None)
    result = warden.run()
    if args.save:
        Warden.save(result)
    print(render_health_markdown(result))
    return 0 if result["passed"] else 1


def cmd_reconcile(args: argparse.Namespace) -> int:
    from raggeoni.spike import reconcile

    if args.merge:
        print(f"{reconcile.merge_proposals()} proposals appended to eval/spikes/events.jsonl")
        return 0
    proposals = reconcile.reconcile(args.report_url, use_llm=args.llm)
    body = reconcile.pr_body(proposals, args.report_url)
    print(body)
    if args.pr_body:
        Path(args.pr_body).write_text(body, encoding="utf-8")
    return 0


def cmd_site(args: argparse.Namespace) -> int:
    from raggeoni.spike.site import build_site

    out = build_site(Path(args.out))
    print(f"site written to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="raggeoni-spike",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    brain_help = "rules (deterministic, no key) or llm (uses SPIKE_LLM_PROVIDER)"

    d = sub.add_parser("demo", help="offline demo on the synthetic incident")
    d.add_argument("--out", default=str(repo_root() / "investigations"))
    d.add_argument("--brain", choices=("rules", "llm"), default="rules", help=brain_help)
    d.set_defaults(fn=cmd_demo)

    r = sub.add_parser("refresh-live", help="update the rolling live window")
    r.add_argument("--keep-days", type=int, default=8)
    r.set_defaults(fn=cmd_refresh_live)

    n = sub.add_parser("notices", help="load recent market notices into the live database")
    n.add_argument("--days", type=int, default=2)
    n.set_defaults(fn=cmd_notices)

    w = sub.add_parser("watch", help="refresh, detect and investigate new spikes")
    w.add_argument("--threshold", type=float, default=SPIKE_THRESHOLD)
    w.add_argument("--brain", choices=("rules", "llm"), default="llm", help=brain_help)
    w.add_argument("--skip-refresh", action="store_true")
    w.add_argument(
        "--settle-minutes", type=int, default=15, help="wait this long after the last high interval"
    )
    w.add_argument("--max-events", type=int, default=3)
    w.set_defaults(fn=cmd_watch)

    i = sub.add_parser("investigate", help="investigate a window in an incident database")
    i.add_argument("--db", required=True)
    i.add_argument("--region", required=True)
    i.add_argument("--start", required=True, help="NEM time, YYYY-MM-DD HH:MM:SS")
    i.add_argument("--end", required=True)
    i.add_argument("--threshold", type=float, default=SPIKE_THRESHOLD)
    i.add_argument("--brain", choices=("rules", "llm"), default="llm", help=brain_help)
    i.add_argument("--no-rag", action="store_true")
    i.add_argument("--no-skeptic", action="store_true")
    i.add_argument("--out")
    i.set_defaults(fn=cmd_investigate)

    e = sub.add_parser("exam", help="the AER-labelled exam")
    e.add_argument("exam_cmd", choices=("build-data", "run"))
    e.add_argument(
        "--arm", default="rules", choices=("rules", "llm", "llm-no-skeptic", "llm-no-rag")
    )
    e.add_argument("--split", default="eval", choices=("eval", "holdout", "all"))
    e.add_argument("--event")
    e.add_argument("--force", action="store_true")
    e.add_argument("--no-forecasts", action="store_true", help="skip P5MIN archives (large)")
    e.add_argument(
        "--synthetic", action="store_true", help="run the offline planted-cause incident instead"
    )
    e.add_argument("--reviewed-only", action="store_true")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--compare", help="scorecard JSON to compare against (regression check)")
    e.add_argument("--summary-md", help="also write the Markdown summary here (PR comment)")
    e.set_defaults(fn=cmd_exam)

    h = sub.add_parser("warden", help="health checks")
    h.add_argument("--no-llm", action="store_true")
    h.add_argument(
        "--save", action="store_true", help="write health/scorecard.json, history and README"
    )
    h.set_defaults(fn=cmd_warden)

    c = sub.add_parser("reconcile", help="compare live investigations with a new AER report")
    c.add_argument("--report-url")
    c.add_argument(
        "--llm", action="store_true", help="extract causes with the LLM instead of keyword rules"
    )
    c.add_argument("--pr-body")
    c.add_argument("--merge", action="store_true", help="append proposals to the exam file")
    c.set_defaults(fn=cmd_reconcile)

    s = sub.add_parser("site", help="build the static site")
    s.add_argument("--out", default="site")
    s.set_defaults(fn=cmd_site)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "reconcile" and not args.merge and not args.report_url:
        print("--report-url is required (or --merge)", file=sys.stderr)
        return 2
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
