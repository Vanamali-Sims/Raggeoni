# 2. A deterministic verifier instead of an LLM judge

**Decision.** Every claim must cite evidence IDs, and every quantity in it must match a number in that evidence (within rounding). Cited notices must have been returned by a cited search, fall inside the window and not be flagged as injection. This is string and number matching, not a model.

**Why.** Using one model to grade another moves the hallucination problem rather than solving it. Number grounding is cheap, repeatable and explainable: a failed check names the exact number that has no source.

**What happens on failure.** The LLM writer gets one repair attempt with the list of violations. Anything still ungrounded is removed, and the report says how many claims were dropped. The original verdict is kept in the report for the audit trail.

**Limits.** The verifier checks that numbers are *real*, not that the reasoning is *right*. A correctly quoted number can still support a wrong conclusion; that is what the skeptic and the exam are for.
