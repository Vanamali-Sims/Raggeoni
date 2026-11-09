# 1. Typed tools instead of model-written code

**Decision.** Agents call a fixed set of typed, tested analysis tools. They do not write and run their own Python or SQL.

**Why.** Free-form code execution looks more impressive, but it is hard to test, hard to secure, and gives different answers on different runs. Typed tools can be unit-tested against an incident with known causes, return the same answer every time, and can't touch anything they weren't built to touch. Ad-hoc questions still work through `safe_query`, which uses the original Raggeoni plan language: the model fills in a validated JSON plan and the compiler binds every value as a parameter.

**Trade-off.** The agent can only ask questions the tools anticipate. A sandboxed "free analysis" tool could be added later as an explicit, measured experiment (an extra exam arm), not as the default.
