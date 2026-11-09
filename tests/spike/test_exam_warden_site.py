"""Exam file and scoring, Warden probes, reconcile helpers and the static site."""

from __future__ import annotations

import json

import pytest

from raggeoni.spike import exam, reconcile
from raggeoni.spike.schema import ASSESSABLE
from raggeoni.spike.site import build_site, md_to_html


def test_exam_file_is_valid_and_split():
    items = exam.load_items(split="all")
    assert len(items) == 20
    assert sum(i.split == "holdout" for i in items) == 5
    assert all(i.gold <= set(ASSESSABLE) and i.gold for i in items)
    assert all(i.raw["source"].startswith("https://www.aer.gov.au/") for i in items)
    assert len({i.id for i in items}) == len(items)


def test_scoring_maths():
    s = exam.score({"low_wind", "high_demand"}, {"low_wind", "network_limit"})
    assert (s["tp"], s["fp"], s["fn"], s["exact"]) == (1, 1, 1, False)
    results = [
        {
            "status": "ok",
            "gold": ["low_wind"],
            "predicted": ["low_wind"],
            **exam.score({"low_wind"}, {"low_wind"}),
            "verified": True,
            "degraded": False,
            "seconds": 1.0,
            "llm": {},
        },
        {
            "status": "ok",
            "gold": ["high_demand"],
            "predicted": [],
            **exam.score({"high_demand"}, set()),
            "verified": True,
            "degraded": False,
            "seconds": 3.0,
            "llm": {},
        },
        {"id": "x", "status": "no_data", "gold": ["low_wind"]},
    ]
    card = exam.scorecard(results, arm="rules", split="eval")
    assert card["precision"] == 1.0 and card["recall"] == 0.5 and card["f1"] == 0.667
    assert card["no_data"] == ["x"] and card["per_cause"]["high_demand"]["recall"] == 0.0


def test_regression_compare():
    base = {
        "precision": 0.8,
        "recall": 0.8,
        "f1": 0.8,
        "exact_match": 0.5,
        "verified_rate": 1.0,
        "git_sha": "abc",
    }
    worse = {**base, "f1": 0.7, "git_sha": "def"}
    assert exam.compare(worse, base)["regressed"] == ["f1"]
    assert exam.compare(base, base)["regressed"] == []


def test_missing_exam_data_is_reported_not_crashed(tmp_path, monkeypatch):
    monkeypatch.setenv("SPIKE_DATA_DIR", str(tmp_path))
    item = exam.load_items(split="holdout")[0]
    assert exam.run_item(item, "rules")["status"] == "no_data"


def test_synthetic_exam_rules_is_perfect(tmp_path):
    card, results = exam.run_synthetic("rules", tmp_path)
    assert card["f1"] == 1.0 and card["verified_rate"] == 1.0
    assert "### Spike exam" in exam.markdown_summary(card, results)


def test_warden_offline_passes(tmp_path, monkeypatch):
    from raggeoni.spike.warden import Warden

    monkeypatch.setenv("SPIKE_DATA_DIR", str(tmp_path))
    result = Warden(use_llm=False, live_db=tmp_path / "none.duckdb").run()
    failed = [c for c in result["checks"] if c["status"] == "fail"]
    assert result["passed"], failed
    groups = {c["group"] for c in result["checks"]}
    assert {"gates", "correctness", "fault injection", "red team"} <= groups


def test_warden_flags_stale_live_data(tmp_path):
    from raggeoni.spike.store import upsert
    from raggeoni.spike.warden import Warden

    db = tmp_path / "live.duckdb"
    upsert(db, "price_demand", [("2020-01-01 00:05:00", "SA1", 50.0, 1500.0, "TRADE")])
    w = Warden(use_llm=False, live_db=db)
    status, detail = w.gate_live_freshness()
    assert status == "fail" and "min old" in detail
    status, detail = w.gate_live_sanity()
    assert status == "pass" or "regions" in detail


def test_reconcile_rules_and_retrieval():
    text = (
        "Overview of the quarter and market conditions.\n\n"
        "On 21 June in South Australia, prices reached $8,689/MWh at 8 pm. Cold weather drove high demand, "
        "wind averaged 54 MW and Heywood imports were restricted by a constraint.\n\n"
        "On 3 March in Tasmania, FCAS prices rose because Basslink was in its no-go zone."
    )
    chunks = reconcile.passages(text, size=200, overlap=0)
    from datetime import datetime

    hits = reconcile.retrieve(chunks, "SA1", datetime(2026, 6, 21, 20, 0))
    assert hits and "South Australia" in hits[0].text
    labels = reconcile.extract_causes_rules(hits[0].text)
    assert labels == ["high_demand", "low_wind", "network_limit"]


def test_md_to_html_escapes_and_renders_tables():
    out = md_to_html("# T\n\n| a | b |\n|---|---|\n| <x> | **y** |\n\n- item _(E1)_")
    assert (
        "<table>" in out
        and "&lt;x&gt;" in out
        and "<strong>y</strong>" in out
        and "<em>(E1)</em>" in out
    )


def test_site_build(tmp_path):
    from raggeoni.spike import demo_data
    from raggeoni.spike.agents import HeuristicBrain
    from raggeoni.spike.detect import find_spikes
    from raggeoni.spike.graph import investigate
    from raggeoni.spike.report import write_investigation

    db = demo_data.build(tmp_path / "d.duckdb", days=7)
    inv_dir = tmp_path / "inv"
    event = find_spikes(db)[0]
    write_investigation(investigate(event, db, HeuristicBrain()), inv_dir)
    out = build_site(tmp_path / "site", investigations=inv_dir, health=tmp_path / "nohealth")
    index = (out / "index.html").read_text()
    assert "synthetic demo" in index and "verified" in index
    assert (out / "events" / f"{event.event_id}.html").exists()
    assert (out / "health.html").exists() and (out / ".nojekyll").exists()


@pytest.mark.parametrize("bad", [{"labels": ["vibes"]}, {"split": None}])
def test_exam_loader_rejects_bad_rows(tmp_path, bad):
    row = json.loads((exam.exam_dir() / "events.jsonl").read_text().splitlines()[0])
    if "split" in bad:
        del row["split"]
    else:
        row.update(bad)
    path = tmp_path / "e.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        exam.load_items(path, split="all")
