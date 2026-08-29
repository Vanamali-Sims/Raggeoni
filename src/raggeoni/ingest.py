"""Download AEMO aggregated price/demand and freeze it into one DuckDB file."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import duckdb

from raggeoni.checks import run_checks
from raggeoni.constants import (
    EXPECTED_MAX_TS,
    EXPECTED_MIN_TS,
    EXPECTED_ROWS,
    GRAIN_NOTE,
    REGIONS,
    SOURCE_BASE_URL,
    SOURCE_PAGE_URL,
    TIMEZONE_NOTE,
    USER_AGENT,
    WINDOW_END_MONTH,
    WINDOW_END_YEAR,
    WINDOW_START_MONTH,
    WINDOW_START_YEAR,
)

RETRY_COUNT = 3


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def raw_dir() -> Path:
    return data_dir() / "raw"


def db_path() -> Path:
    return data_dir() / "nem.duckdb"


def manifest_path() -> Path:
    return data_dir() / "MANIFEST.json"


def schema_path() -> Path:
    return Path(__file__).resolve().parent / "schema.sql"


def year_months() -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    year, month = WINDOW_START_YEAR, WINDOW_START_MONTH
    while (year, month) <= (WINDOW_END_YEAR, WINDOW_END_MONTH):
        out.append((year, month))
        month += 1
        if month == 13:
            year += 1
            month = 1
    return out


def source_filename(year: int, month: int, region: str) -> str:
    return f"PRICE_AND_DEMAND_{year}{month:02d}_{region}.csv"


def source_url(year: int, month: int, region: str) -> str:
    return f"{SOURCE_BASE_URL}/{source_filename(year, month, region)}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                dest.write_bytes(resp.read())
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            print(f"  retry {attempt}/{RETRY_COUNT} {url}: {exc}", file=sys.stderr)
    raise RuntimeError(f"failed to download {url}") from last_error


def download_sources(*, force: bool) -> list[dict]:
    files: list[dict] = []
    jobs = [
        (year, month, region)
        for year, month in year_months()
        for region in REGIONS
    ]
    print(f"Downloading {len(jobs)} monthly CSVs into {raw_dir()}")
    for year, month, region in jobs:
        name = source_filename(year, month, region)
        path = raw_dir() / name
        url = source_url(year, month, region)
        if force or not path.exists():
            print(f"  get {name}")
            download_file(url, path)
        else:
            print(f"  have {name}")
        files.append(
            {
                "name": name,
                "url": url,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def load_duckdb(source_files: list[dict]) -> None:
    db = db_path()
    wal = Path(str(db) + ".wal")
    if db.exists():
        db.unlink()
    if wal.exists():
        wal.unlink()
    db.parent.mkdir(parents=True, exist_ok=True)

    glob = (raw_dir() / "*.csv").as_posix()
    schema_sql = schema_path().read_text(encoding="utf-8")

    con = duckdb.connect(str(db))
    try:
        for statement in schema_sql.split(";"):
            stmt = statement.strip()
            if stmt:
                con.execute(stmt)
        con.execute(
            """
            INSERT INTO price_demand
            SELECT
                strptime(SETTLEMENTDATE, '%Y/%m/%d %H:%M:%S') AS settlement_ts,
                REGION AS region,
                CAST(RRP AS DOUBLE) AS rrp,
                CAST(TOTALDEMAND AS DOUBLE) AS total_demand,
                PERIODTYPE AS period_type
            FROM read_csv(
                ?,
                header = true,
                union_by_name = true,
                columns = {
                    'REGION': 'VARCHAR',
                    'SETTLEMENTDATE': 'VARCHAR',
                    'TOTALDEMAND': 'DOUBLE',
                    'RRP': 'DOUBLE',
                    'PERIODTYPE': 'VARCHAR'
                }
            )
            """,
            [glob],
        )

        sources_sha = hashlib.sha256(
            "".join(f["sha256"] for f in source_files).encode("ascii")
        ).hexdigest()
        meta = {
            "source_page": SOURCE_PAGE_URL,
            "source_base": SOURCE_BASE_URL,
            "window_start_month": f"{WINDOW_START_YEAR}-{WINDOW_START_MONTH:02d}",
            "window_end_month": f"{WINDOW_END_YEAR}-{WINDOW_END_MONTH:02d}",
            "date_start": EXPECTED_MIN_TS,
            "date_end": EXPECTED_MAX_TS,
            "regions": ",".join(REGIONS),
            "row_count": str(EXPECTED_ROWS),
            "interval_minutes": "5",
            "timezone": TIMEZONE_NOTE,
            "grain": GRAIN_NOTE,
            "sources_sha256": sources_sha,
            "built_on": date.today().isoformat(),
        }
        for key, value in meta.items():
            con.execute("INSERT INTO meta VALUES (?, ?)", [key, value])

        report = run_checks(con)
        for result in report.results:
            mark = "PASS" if result.ok else "FAIL"
            print(f"  [{mark}] {result.name}: {result.detail}")
        report.raise_if_failed()

        con.execute("CHECKPOINT")
    finally:
        con.close()


def canonical_sha256(con: duckdb.DuckDBPyConnection, out: Path) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        """
        COPY (
            SELECT
                strftime(settlement_ts, '%Y-%m-%d %H:%M:%S') AS settlement_ts,
                region,
                printf('%.8f', rrp) AS rrp,
                printf('%.8f', total_demand) AS total_demand,
                period_type
            FROM price_demand
            ORDER BY settlement_ts, region
        ) TO ? (HEADER, DELIMITER ',')
        """,
        [out.as_posix()],
    )
    digest = sha256_file(out)
    out.unlink()
    return digest


def write_manifest(source_files: list[dict]) -> dict:
    db = db_path()
    con = duckdb.connect(str(db), read_only=True)
    try:
        canonical = canonical_sha256(con, data_dir() / "_canonical.csv")
        stats = con.execute(
            """
            SELECT
                COUNT(*) AS row_count,
                MIN(settlement_ts)::VARCHAR AS min_ts,
                MAX(settlement_ts)::VARCHAR AS max_ts,
                COUNT(*) FILTER (WHERE rrp < 0) AS negative_rrp_rows,
                MIN(rrp) AS min_rrp,
                MAX(rrp) AS max_rrp
            FROM price_demand
            """
        ).fetchone()
        meta = {
            row[0]: row[1]
            for row in con.execute("SELECT key, value FROM meta ORDER BY key").fetchall()
        }
    finally:
        con.close()

    sources_sha = hashlib.sha256(
        "".join(f["sha256"] for f in source_files).encode("ascii")
    ).hexdigest()

    manifest = {
        "table": "price_demand",
        "regions": list(REGIONS),
        "window": {
            "start_month": f"{WINDOW_START_YEAR}-{WINDOW_START_MONTH:02d}",
            "end_month": f"{WINDOW_END_YEAR}-{WINDOW_END_MONTH:02d}",
            "min_settlement_ts": stats[1],
            "max_settlement_ts": stats[2],
        },
        "row_count": stats[0],
        "negative_rrp_rows": stats[3],
        "rrp_min": stats[4],
        "rrp_max": stats[5],
        "timezone": TIMEZONE_NOTE,
        "grain": GRAIN_NOTE,
        "source_page": SOURCE_PAGE_URL,
        "duckdb_path": "data/nem.duckdb",
        "duckdb_sha256": sha256_file(db),
        "duckdb_bytes": db.stat().st_size,
        "canonical_sha256": canonical,
        "sources_sha256": sources_sha,
        "source_files": source_files,
        "meta": meta,
    }
    path = manifest_path()
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze 12 months of NEM price/demand into data/nem.duckdb"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download CSVs even if they already exist in data/raw",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Re-run freeze checks on an existing data/nem.duckdb (no download)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check_only:
        db = db_path()
        if not db.exists():
            print(f"missing {db}", file=sys.stderr)
            return 1
        con = duckdb.connect(str(db), read_only=True)
        try:
            report = run_checks(con)
            for result in report.results:
                mark = "PASS" if result.ok else "FAIL"
                print(f"  [{mark}] {result.name}: {result.detail}")
            report.raise_if_failed()
        finally:
            con.close()
        print("All freeze checks passed.")
        return 0

    source_files = download_sources(force=args.force_download)
    print("Loading DuckDB")
    load_duckdb(source_files)
    write_manifest(source_files)
    print("All freeze checks passed.")
    print(f"  db        {db_path()}")
    print(f"  rows      {EXPECTED_ROWS}")
    print(f"  window    {EXPECTED_MIN_TS} -> {EXPECTED_MAX_TS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
