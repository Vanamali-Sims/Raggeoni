# 5. OpenRouter as the default model provider

**Decision.** The agents call models through OpenRouter by default (`SPIKE_LLM_PROVIDER=openrouter`, key in `OPENROUTER_API_KEY`, model as an OpenRouter slug in `SPIKE_MODEL`). Direct Anthropic and OpenAI-compatible clients remain available.

**Why.**
- **Model comparison is one setting.** The exam can compare models from different labs (a frontier model against a cheap one) without new code or new keys. That turns "which model should a client use?" into a measured answer.
- **Exact cost.** OpenRouter returns the real cost of each call in the response. The exam reports provider cost instead of an estimate from a hand-maintained price table that goes stale.
- **One key to manage** in GitHub secrets, with a spending limit set on the key itself, which matters because the watcher runs unattended.

**How.** OpenRouter speaks the OpenAI chat-completions format, tool calls included, so it reuses the OpenAI-compatible client with a different base URL, key and optional attribution headers. No SDK.

**Trade-offs.** An extra hop adds a little latency and a dependency on OpenRouter's availability; the model-outage fallback covers the second. Not every model on OpenRouter supports tool calling; an unsupported slug fails with a clear error and the run falls back to rules.
