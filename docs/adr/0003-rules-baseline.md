# 3. Rules as the baseline arm

**Decision.** The same roles (lead, evidence, skeptic, writer) exist as deterministic rules (`HeuristicBrain`). The rules run in CI with no key, act as the fallback when the model is down, and are the baseline arm in the exam.

**Why.** "The LLM got 70%" means nothing on its own. If hand-written rules get 65%, the agents add little for their cost; if rules get 40%, the agents are earning their keep. Ablations (no skeptic, no RAG) then show where the gain comes from.

**Fallback.** If the model fails or the token budget runs out, that step falls back to rules and the report is marked *degraded* with the reason. A run never dies on an outage, and the Warden simulates one on every run to prove it.
