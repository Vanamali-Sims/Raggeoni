# Raggeoni

**A ground-truth-first eval harness for asking questions of Australian electricity market (NEM) data in plain English.**

Raggeoni measures how well a language model can turn a question like *"What was the maximum RRP in QLD1 in September 2025?"* into a correct answer, using a frozen, checksummed dataset and an eval set whose answers were computed by code rather than typed in by a person.

The model never writes SQL. It writes a small JSON **query plan**; Raggeoni validates the plan, compiles it to parameterised SQL, runs it against DuckDB and compares the result exactly against the stored gold answer. A model is scored on whether it got the *right numbers*, not on whether its query looks right.

---

## Status

| Piece | State |
|---|---|
| Data freeze (12 months, 5 regions, 525,600 rows) | ✅ Done, recorded in `data/MANIFEST.json` with SHA-256s |
| Query-plan DSL + validator + SQL compiler | ✅ Done, unit-tested |
| Eval set `v1` (200 questions, gold plans + gold results) | ✅ Done and hydrated |
| Scoring harness + `EXEC_ACC` report | ✅ Done |
| Arm 1 baseline (prompted model → plan) | 🟡 Wired up, **not yet run** (no model/API key configured) |
| Retrieval / RAG arms | ⬜ Not started |

The next milestone is the first Arm 1 number on the `eval` split.

---

## How it works

```
question ──► model ──► JSON plan ──► validate ──► compile ──► DuckDB ──► result
                                       │                                   │
                              rejects anything                    canonicalised,
                              outside the DSL                    compared exactly
                                                                   to gold_result
```

1. **Freeze the data** (`raggeoni.ingest`). Downloads AEMO's monthly *Aggregated Price and Demand* CSVs for NSW1, QLD1, VIC1, SA1 and TAS1, August 2025 – July 2026, and loads them into `data/nem.duckdb`. Ten freeze checks must pass (row count, regions, no duplicates or nulls, no gaps in the 5-minute sequence, expected first/last timestamps, negative prices present, balanced regions, …). Hashes of every source file, the DB and a canonical CSV export are written to `data/MANIFEST.json`.
2. **Author questions** (`eval/author_v1.py`). Generates 200 questions, each with a hand-designed `gold_plan`.
3. **Hydrate gold answers** (`raggeoni.hydrate`). Executes each `gold_plan` and writes the result into `gold_result`, so no gold number is typed by a person.
4. **Score predictions** (`raggeoni.eval`). Runs a model's plans (or a predictions file) through the same pipeline and reports exact-match execution accuracy.

### The data

One table, `price_demand`. One row = one region × one 5-minute interval.

| column | meaning |
|---|---|
| `settlement_ts` | interval-**ending** timestamp, NEM time (AEST, UTC+10, no daylight saving) |
| `region` | `NSW1`, `QLD1`, `VIC1`, `SA1`, `TAS1` |
| `rrp` | regional reference price, $/MWh (can be negative, ranges −1,000 to 23,200 in this window) |
| `total_demand` | MW |
| `period_type` | always `TRADE` |

Window: `2025-08-01 00:05:00` → `2026-08-01 00:00:00` inclusive. A calendar day *D* is `D 00:05:00` through `D+1 00:00:00`.

### The query-plan DSL

Four intents, defined in `src/raggeoni/plan_schema.json` and enforced by `validate.py`:

| intent | use | example |
|---|---|---|
| `lookup` | fetch row(s); `filters` and `limit` required | RRP in TAS1 at a given timestamp |
| `aggregate` | `max`/`min`/`avg`/`sum` on `rrp` or `total_demand`, or `count`; optional `group_by` | max QLD1 RRP in Sept 2025 |
| `compare` | one metric over exactly two filtered sides | avg RRP NSW1 vs VIC1 on a day |
| `refuse` | the table can't answer it | unit output, forecasts, WEM, dates outside the window |

```json
{"intent": "aggregate",
 "filters": [{"col": "region", "op": "=", "val": "QLD1"},
             {"col": "settlement_ts", "op": ">=", "val": "2025-09-01 00:05:00"},
             {"col": "settlement_ts", "op": "<=", "val": "2025-10-01 00:00:00"}],
 "select": [{"fn": "max", "col": "rrp", "as": "max_rrp"}]}
```

The validator rejects unknown fields, columns, regions, functions and reserved aliases. The compiler only emits whitelisted identifiers and binds every value as a parameter, so a plan can't inject SQL.

### The eval set

`eval/v1.jsonl` has 200 questions, all hydrated:

| bucket | eval | holdout | total |
|---|---|---|---|
| structural (about the table itself) | 32 | 8 | 40 |
| factual (numbers from the data) | 96 | 24 | 120 |
| out-of-scope (should refuse) | 32 | 8 | 40 |

By intent: 97 aggregate, 42 lookup, 40 refuse, 21 compare. `eval/smoke.jsonl` has 4 quick sanity questions, one per intent.

**Holdout discipline:** develop against `--split eval`; run `--split holdout` once at the end. `--arm 1` refuses to run on `--split all`.

### Scoring

A prediction counts as correct only if its executed result exactly equals `gold_result` (floats rounded to 4 dp, timestamps formatted, rows sorted when the plan has no `order_by`). Failures are classed as:

- `parse`: no JSON object in the model output
- `invalid_plan`: JSON that isn't in the DSL
- `sql`: DuckDB error (or missing DB)
- `row_cap`: more than 1,000 rows
- `false_answer`: answered when it should have refused
- `false_refuse`: refused when it should have answered
- `mismatch`: ran fine, wrong result

---

## Quick start

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Build the frozen DB (downloads 60 monthly CSVs from aemo.com.au, ~35 MB DB)
uv run python -m raggeoni.ingest
uv run python -m raggeoni.ingest --check-only      # re-run freeze checks only

# 2. Confirm the harness agrees with itself (must print EXEC_ACC 1.000)
uv run python -m raggeoni.hydrate eval/v1.jsonl --verify
uv run python -m raggeoni.eval --self-test

# 3. Run tests
uv run python -m unittest discover -s tests -v
```

`data/nem.duckdb` and `data/raw/` are git-ignored, so a fresh clone must run step 1 before anything that touches data. After building, check that `duckdb_sha256` / `canonical_sha256` match `data/MANIFEST.json`. If AEMO has revised a file, the source hashes will differ.

### Running a model (Arm 1)

```bash
export OPENAI_API_KEY=...            # or RAGGEONI_API_KEY
export RAGGEONI_MODEL=gpt-4o-mini    # optional, this is the default
uv run python -m raggeoni.eval --arm 1 --split eval --out eval/runs/arm1-eval.jsonl
```

Arm 1 sends the system prompt in `src/raggeoni/baseline/prompt.txt` (schema, time conventions, four worked examples) at temperature 0 with JSON mode, then scores each plan.

### Scoring your own predictions

```bash
# predictions.jsonl: one {"id": "...", "plan": {...}} per line
uv run python -m raggeoni.eval --predictions predictions.jsonl --split eval
```

### Single-plan CLI

```bash
uv run python -m raggeoni.plan --compile --plan '{"intent":"refuse"}'   # show SQL only
uv run python -m raggeoni.plan --file my_plan.json                       # execute
```

Console scripts `raggeoni-ingest`, `raggeoni-plan`, `raggeoni-hydrate` and `raggeoni-eval` are also installed.

---

## Layout

```
data/MANIFEST.json        frozen-data fingerprint (committed); nem.duckdb is built locally
eval/author_v1.py         generates v1.jsonl questions + gold plans
eval/v1.jsonl             200-question eval set with gold results
eval/smoke.jsonl          4-question smoke set
src/raggeoni/
  constants.py            frozen window, regions, source URLs
  ingest.py, checks.py    download → DuckDB → freeze checks → manifest
  schema.sql              price_demand + meta tables
  plan_schema.json        JSON Schema for the DSL
  validate.py             DSL validator / normaliser (no SQL)
  compile.py              plan → parameterised SQL
  execute.py              run + canonicalise results
  hydrate.py              gold_plan → gold_result
  harness.py, eval.py     scoring and EXEC_ACC report
  baseline/               Arm 1 prompt + OpenAI client
tests/                    DSL and harness tests
```

---

## Known issues

- **Two harness tests fail when the DB isn't built.** `test_gold_plan_is_correct` and `test_refuse_vs_lookup_is_false_answer` in `tests/test_harness.py` need `data/nem.duckdb` but don't skip without it the way `test_dsl.py` does. On a fresh clone: 17 pass, 2 fail, 2 skip until you run ingest.
- **A missing DB is reported as a scoring error.** `--self-test` without the DB prints `EXEC_ACC 0.200` (only the refusals pass) instead of failing up front. Build the DB first.
- **The freeze can't be fully reproduced unattended.** AEMO occasionally revises published CSVs; if a re-download hashes differently, the gold answers may no longer match.
