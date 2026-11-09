# Raggeoni: NEM Spike Investigator

**When an Australian electricity price spikes, a team of agents works out why, and every number in the explanation is checked against the data.**

Sometimes a region's price jumps from $80 to $15,000/MWh in minutes. Traders, retailers, battery owners and journalists all ask the same question: *why?* Today the official answer arrives months later, in the regulator's quarterly report. This project produces a cited explanation within minutes of a spike settling, and then checks itself against the regulator's findings when they come out.

```
 AEMO 5-min data ─┐
 Market notices ──┼─► watch (every 10 min) ─► spike? ─► investigation team ─► verifier ─► report + site
 Generation mix ──┘                                          │
                                                             ▼
 AER quarterly report ───────────────► reconcile ─► "did we get it right?" ─► PR adding exam rows
                                                             │
 Warden (daily) ── gates · exam regression · fault injection · red team · budgets ─► health page
```

---

## What a report looks like

> **SA1 reached $16,239.51/MWh at 19:50, driven mainly by a wind lull, unusually high demand and interconnector or network limits.**
> ✅ all numbers grounded in evidence (27/27)
>
> **Wind lull: high confidence**
> - Wind output was 141.3 MW against a typical 746.0 MW for this time of day (percentile 0.0). *(E10)*
> - This accounted for 37.8% (550.9 MW) of the worsening in the supply-demand balance. *(E16)*
>
> **Network / interconnector limits: high confidence**
> - Heywood carried 250.0 MW into SA1 against a limit of 250.0 MW, 400.0 MW below its typical limit. *(E7)*
> - Market notice 990101 reported: Planned outage - South East to Tailem Bend 275 kV line. *(E11)*
>
> **Ruled out: solar drop.** Solar was normal for the time of day.

That example is the **synthetic demo incident** (dated 2031 so it can't be mistaken for real data), which has four causes planted in it. Run it yourself with `raggeoni-spike demo`.

---

## How it works

### 1. The investigation team (LangGraph)

```
overview ─► lead ─┬─► evidence agent × N (parallel, one per hypothesis) ─► skeptic ─┐
                  │                     ▲                                          │
                  │                     └──── challenged findings, round 2 ◄───────┤
                  └──────────────────────────────────────────────────────────────► writer ─► verifier
```

| Role | Job |
|---|---|
| **Overview** (code) | Runs the four checks every investigation needs, so the lead starts from facts. |
| **Lead** | Proposes 3–6 competing hypotheses from a fixed list of causes, including at least one it expects to rule out. |
| **Evidence agents** | One per hypothesis, in parallel. Each runs a tool-use loop (2–5 calls), trying to *disprove* as well as support. |
| **Skeptic** | Attacks weak findings: *"wind drops every evening, was this unusual?"* Sends up to 3 back for a second round. |
| **Writer** | Ranks causes using the supply-gap breakdown and cites evidence IDs for every claim. |
| **Verifier** (code) | Checks every number against the cited evidence. One repair attempt, then ungrounded claims are dropped and the report says so. |

### 2. The tools do the maths, not the model

| Tool | What it computes |
|---|---|
| `unusualness` | Value vs the same time of day over the last 14–30 days (mean, z-score, percentile). Needs an extreme percentile **and** a material difference, so steady series don't trigger false alarms. |
| `demand_surprise` | Actual demand vs AEMO's own 5-minute-ahead forecast. Was the market caught off guard? |
| `interconnector_status` | Flow into the region on each link: at its limit? constraint binding? limit reduced vs normal? |
| `supply_gap` | Splits the worsening balance into demand ↑, each fuel ↓ and imports ↓, in MW and % share. |
| `generation_change` | Output by fuel type now vs N minutes ago. |
| `price_profile` | Peak, duration, and whether neighbours stayed cheap (price separation). |
| `search_notices` | RAG over AEMO market notices (below). |
| `safe_query` | Ad-hoc price/demand questions via Raggeoni's validated plan language. **The model never writes SQL.** |
| `similar_events` | Past investigations with the most similar conditions. |

Every tool returns an evidence record with an ID and machine-readable facts. Tools never crash the investigation: missing data comes back as `unavailable` with the reason.

### 3. RAG over market notices

The numbers show *what* happened; notices often say *why* (a unit trip, a planned line outage, a constraint).

1. **Filter first (exact):** notices from 24 h before the window to its end, tagged with the region or a neighbour.
2. **Rank (hybrid):** BM25 for exact codes like `TORRB2`, embeddings for meaning ("trip" ≈ "contingency event"), fused with reciprocal rank fusion.
3. **Cite by ID:** the verifier checks each cited notice was returned by a search, falls inside the window, and isn't flagged as suspicious.

Notices are untrusted text. Instruction-like content is flagged, and all notice text is wrapped as quoted data. The Warden plants a prompt injection in a notice on every run to prove this holds.

The **ablation shows what RAG is worth**: on the demo incident, turning it off loses the generator trip, because it is invisible in the numbers alone.

### 4. The exam: graded against the regulator

`eval/spikes/events.jsonl` holds **20 real price events (Nov 2025 – Jun 2026)** with causes drafted from the AER's quarterly reports (source URL and pages on every row, 5 events held out). Scoring is precision/recall/F1 over cause labels, plus verification rate, latency, tokens and cost.

| Arm | What it is |
|---|---|
| `rules` | Hand-written deterministic rules. **The baseline the LLM team has to beat to justify its cost.** |
| `llm` | The full team. |
| `llm-no-skeptic`, `llm-no-rag` | Ablations: what does each part add? |

Rebidding and unit ramp limits need bid and unit data the system doesn't load, so agents must call them "not assessable" and the exam doesn't score them.

### 5. The Warden

Runs daily (and on demand). Cheap, certain checks first; an LLM only at the end to explain failures.

| Group | Checks |
|---|---|
| Gates | Exam file valid · live data fresh · prices in range · no duplicate/missing intervals |
| Correctness | Planted-cause incident · exam F1 regression vs the last run |
| Fault injection | Model outage · missing generation data · empty database |
| Red team | Prompt injection in a notice · SQL smuggled into a query plan · invented number in a report |
| Budgets | p95 latency · tokens per event · verification rate |

### 6. It keeps its own exam honest

When a new AER report is published, `reconcile` retrieves the passages about each live investigation, extracts the official causes, compares them with what the agents said *at the time*, and opens a pull request proposing new exam rows. **A human reviews the labels and merges.** The exam grows from real outcomes, never by itself.

### 7. GitHub Actions

| Workflow | When | What |
|---|---|---|
| `ci` | every push / PR | lint, types, tests, offline demo |
| `eval` | PRs touching agents, tools, prompts or exam | exam on cached data; scorecard and Δ vs baseline as a PR comment |
| `watch` | every 10 min | refresh live data and notices, investigate settled spikes, commit reports |
| `investigate` | manual | rebuild any past window from AEMO archives and investigate it |
| `warden` | daily | health checks; opens or updates an issue on failure |
| `reconcile` | manual, per AER report | proposes exam rows as a PR |
| `pages` | after the above | publishes the investigation feed and health page |

---

## Status: what has and hasn't been tested

| Piece | State |
|---|---|
| Parsers (AEMO MMS CSV, nested archive zips, P5MIN, Open Electricity, market notices) | ✅ Unit-tested against fixtures written in each source's published format |
| Tools, RAG, verifier, agent graph, Warden, site | ✅ Tested on the synthetic incident; rules arm finds all 4 planted causes and rules out the 2 decoys |
| LLM path (tool loop, skeptic round, repair, outage fallback, token budget) | ✅ Tested with a scripted fake model |
| **Live downloads** from NEMWeb and Open Electricity | ⚠️ **Not yet run against the real endpoints**: they were unreachable from the build environment. First real run happens on your machine or in Actions. |
| **Exam scores on the 20 real events** | ⬜ Not run yet: needs the exam data built and an API key |
| **Exam labels** | 🟡 Drafted from AER reports, all `reviewed: false`. Check each against its source before trusting the scores. |

---

## Quick start

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra spike --extra dev

# Offline, no keys: investigate the synthetic incident
uv run raggeoni-spike demo

# The health checks
uv run raggeoni-spike warden --no-llm

# Tests
uv run pytest
```

### Real data

Copy `.env.example` to `.env` and fill in what you have. NEMWeb needs no key; generation data needs a free [Open Electricity](https://platform.openelectricity.org.au/) key; the LLM team needs an [OpenRouter](https://openrouter.ai/settings/keys) key. OpenRouter lets you swap models with one setting (`SPIKE_MODEL`, default `anthropic/claude-sonnet-5`) and reports the real cost of every call, so the exam's cost column is exact rather than estimated. Direct Anthropic and OpenAI-compatible keys still work via `SPIKE_LLM_PROVIDER`.

```bash
# Build data for one exam event, then run both arms
# (events without data yet are listed as "no data", not failed)
uv run raggeoni-spike exam build-data --event aer-2026q2-sa-0621
uv run raggeoni-spike exam run --arm rules
uv run raggeoni-spike exam run --arm llm

# Live: pull the latest data, detect and investigate
uv run raggeoni-spike refresh-live
uv run raggeoni-spike watch --brain llm
```

`raggeoni-spike` reads `.env` from the repo root automatically (real environment variables win), so this works the same on Windows, macOS and Linux.

### Turning on the Actions

1. Repo **Settings → Secrets and variables → Actions**: add `OPENROUTER_API_KEY` and `OPENELECTRICITY_API_KEY` as secrets. Optional variables: `SPIKE_MODEL` (an OpenRouter slug), `SPIKE_THRESHOLD`. Consider a credit limit on the OpenRouter key, since `watch` runs unattended.
2. **Settings → Pages → Source: GitHub Actions.**
3. Run `eval` once by hand, then save its scorecard to `eval/spikes/baselines/llm-eval.json` so future PRs are compared against it.
4. `watch` runs every 10 minutes, which uses Actions minutes. Pause it from the Actions tab when you don't need it.

---

## Design decisions

Short records in [`docs/adr/`](docs/adr/):

1. [Typed tools instead of model-written code](docs/adr/0001-typed-tools.md)
2. [A deterministic verifier instead of an LLM judge](docs/adr/0002-deterministic-verifier.md)
3. [Rules as the baseline arm](docs/adr/0003-rules-baseline.md)
4. [Frozen exam, live product](docs/adr/0004-frozen-exam-live-product.md)
5. [OpenRouter as the default model provider](docs/adr/0005-openrouter.md)

## The original Raggeoni harness

This project grew out of a ground-truth-first harness for plain-English questions over a frozen year of NEM prices: 200 questions, gold answers computed by code, and a query-plan language the model fills in instead of writing SQL. That harness is still here and is reused as the `safe_query` tool. See [`docs/harness.md`](docs/harness.md).

## Layout

```
src/raggeoni/spike/
  sources/          NEMWeb (MMS parser, DispatchIS, P5MIN), Open Electricity, market notices
  store.py          incident DuckDB (price_demand matches the harness schema)
  detect.py         spike detection
  tools.py          analysis tools + evidence ledger (evidence.py)
  rag.py            notice retrieval
  agents.py         lead / evidence / skeptic / writer, LLM and rules brains
  graph.py          LangGraph orchestration
  verify.py         grounding verifier
  exam.py           AER-labelled exam, arms, scoring
  warden.py         health checks
  reconcile.py      AER report → proposed exam rows
  site.py           GitHub Pages site
  prompts/          one prompt per role
eval/spikes/        events.jsonl (exam), baselines/
investigations/     committed reports (written by watch)
health/             Warden output
```
