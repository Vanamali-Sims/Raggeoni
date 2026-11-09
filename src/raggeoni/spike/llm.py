"""Minimal chat-with-tools clients. No SDKs, so CI stays light.

Providers (``SPIKE_LLM_PROVIDER``):

* ``openrouter`` (default): one key (``OPENROUTER_API_KEY``) for models from
  many labs. ``SPIKE_MODEL`` takes an OpenRouter slug, default
  ``anthropic/claude-sonnet-5``. OpenRouter reports the real cost of every
  call, so cost tracking needs no price table.
* ``anthropic``: Claude Messages API direct, key in ``ANTHROPIC_API_KEY``.
* ``openai``: any OpenAI-compatible chat completions API, key in
  ``OPENAI_API_KEY``; set ``SPIKE_LLM_BASE_URL`` for Groq or others
  (e.g. https://api.groq.com/openai/v1 with ``GROQ_API_KEY``).

For providers that don't report cost, set ``SPIKE_PRICE_IN_PER_MTOK`` and
``SPIKE_PRICE_OUT_PER_MTOK`` (USD per million tokens) to get an estimate.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(RuntimeError):
    """The model could not be reached or returned something unusable."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall]
    input_tokens: int = 0
    output_tokens: int = 0
    raw_assistant: Any = None  # provider-native assistant message, replayed in the next turn
    cost_usd: float | None = None  # set when the provider reports it (OpenRouter does)


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    failures: int = 0
    reported_cost: float = 0.0
    reported_calls: int = 0
    by_role: dict[str, dict[str, float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add(self, role: str, reply: Reply | None, seconds: float, *, failed: bool = False) -> None:
        with self._lock:
            self._add(role, reply, seconds, failed=failed)

    def _add(self, role: str, reply: Reply | None, seconds: float, *, failed: bool) -> None:
        self.calls += 1
        self.seconds += seconds
        if failed:
            self.failures += 1
        bucket = self.by_role.setdefault(
            role, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}
        )
        bucket["calls"] += 1
        bucket["seconds"] = round(bucket["seconds"] + seconds, 3)
        if reply is not None:
            self.input_tokens += reply.input_tokens
            self.output_tokens += reply.output_tokens
            bucket["input_tokens"] += reply.input_tokens
            bucket["output_tokens"] += reply.output_tokens
            if reply.cost_usd is not None:
                self.reported_cost += reply.cost_usd
                self.reported_calls += 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self) -> float | None:
        """Provider-reported cost when every successful call reported one;
        otherwise an estimate from configured prices; otherwise unknown."""
        successful = self.calls - self.failures
        if self.reported_calls and self.reported_calls == successful:
            return round(self.reported_cost, 6)
        pin = os.environ.get("SPIKE_PRICE_IN_PER_MTOK")
        pout = os.environ.get("SPIKE_PRICE_OUT_PER_MTOK")
        if not pin or not pout:
            return None
        return round(
            self.input_tokens / 1e6 * float(pin) + self.output_tokens / 1e6 * float(pout), 5
        )

    def cost_source(self) -> str:
        successful = self.calls - self.failures
        if self.reported_calls and self.reported_calls == successful:
            return "provider"
        if os.environ.get("SPIKE_PRICE_IN_PER_MTOK") and os.environ.get("SPIKE_PRICE_OUT_PER_MTOK"):
            return "estimate"
        return "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "seconds": round(self.seconds, 2),
            "failures": self.failures,
            "cost_usd": self.cost_usd(),
            "cost_source": self.cost_source(),
            "by_role": self.by_role,
        }


class ChatModel(Protocol):
    name: str

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1500,
    ) -> Reply: ...

    def tool_result_message(self, results: list[tuple[ToolCall, str]]) -> list[dict[str, Any]]: ...


def _post(
    url: str, body: dict[str, Any], headers: dict[str, str], timeout: int = 120, retries: int = 3
) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:500].decode("utf-8", errors="replace")
            if exc.code in (429, 500, 502, 503, 504, 529) and attempt < retries - 1:
                last = LLMError(f"HTTP {exc.code}: {detail}")
                time.sleep(2 * 2**attempt)
                continue
            hint = {
                401: " (check the API key)",
                402: " (out of credits: top up at openrouter.ai/settings/credits)",
                404: " (model slug not found: check SPIKE_MODEL)",
            }.get(exc.code, "")
            raise LLMError(f"HTTP {exc.code}{hint}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(2 * 2**attempt)
    raise LLMError(f"model unreachable after {retries} tries: {last}")


class AnthropicChat:
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.name = model or os.environ.get("SPIKE_MODEL", "claude-sonnet-5-5")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.url = (
            os.environ.get("SPIKE_LLM_BASE_URL", "https://api.anthropic.com") + "/v1/messages"
        )

    def chat(self, system, messages, tools=None, max_tokens=1500) -> Reply:
        body: dict[str, Any] = {
            "model": self.name,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
            "temperature": 0,
        }
        if tools:
            body["tools"] = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"],
                }
                for t in tools
            ]
        payload = _post(
            self.url, body, {"x-api-key": self.api_key or "", "anthropic-version": "2023-06-01"}
        )
        texts, calls = [], []
        for block in payload.get("content", []):
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(block["id"], block["name"], block.get("input") or {}))
        usage = payload.get("usage", {})
        return Reply(
            text="\n".join(texts),
            tool_calls=calls,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            raw_assistant={"role": "assistant", "content": payload.get("content", [])},
        )

    def tool_result_message(self, results):
        return [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": call.id, "content": content}
                    for call, content in results
                ],
            }
        ]


class OpenAICompatChat:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        extra_headers: dict[str, str] | None = None,
        key_names: tuple[str, ...] = ("OPENAI_API_KEY", "GROQ_API_KEY"),
        default_model: str = "gpt-4o-mini",
    ) -> None:
        self.name = model or os.environ.get("SPIKE_MODEL", default_model)
        self.api_key = api_key or next(
            (os.environ[k] for k in key_names if os.environ.get(k)), None
        )
        if not self.api_key:
            raise LLMError(f"{' or '.join(key_names)} is not set")
        base = base_url or os.environ.get("SPIKE_LLM_BASE_URL", "https://api.openai.com/v1")
        self.url = base.rstrip("/") + "/chat/completions"
        self.extra_headers = extra_headers or {}

    def chat(self, system, messages, tools=None, max_tokens=1500) -> Reply:
        body: dict[str, Any] = {
            "model": self.name,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        payload = _post(
            self.url, body, {"Authorization": f"Bearer {self.api_key}", **self.extra_headers}
        )
        if payload.get("error"):
            err = payload["error"]
            raise LLMError(
                f"provider error: {err.get('message', err) if isinstance(err, dict) else err}"
            )
        try:
            msg = payload["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected response: {str(payload)[:300]}") from exc
        calls = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_unparseable": tc["function"].get("arguments")}
            calls.append(ToolCall(tc["id"], tc["function"]["name"], args))
        usage = payload.get("usage") or {}
        cost = usage.get("cost")
        return Reply(
            text=msg.get("content") or "",
            tool_calls=calls,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            raw_assistant={k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")},
            cost_usd=float(cost) if isinstance(cost, int | float) else None,
        )

    def tool_result_message(self, results):
        return [
            {"role": "tool", "tool_call_id": call.id, "content": content}
            for call, content in results
        ]


OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "anthropic/claude-sonnet-5"


def openrouter_chat(model: str | None = None, api_key: str | None = None) -> OpenAICompatChat:
    """OpenRouter speaks the OpenAI chat-completions format, tools included.

    The attribution headers are optional; they label requests in your
    OpenRouter dashboard.
    """
    return OpenAICompatChat(
        model,
        api_key,
        base_url=os.environ.get("SPIKE_LLM_BASE_URL", OPENROUTER_BASE),
        extra_headers={
            "HTTP-Referer": "https://github.com/Vanamali-Sims/Raggeoni",
            "X-OpenRouter-Title": "Raggeoni Spike Investigator",
            "X-Title": "Raggeoni Spike Investigator",
        },
        key_names=("OPENROUTER_API_KEY",),
        default_model=OPENROUTER_DEFAULT_MODEL,
    )


PROVIDER_KEYS: dict[str, tuple[str, ...]] = {
    "openrouter": ("OPENROUTER_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY", "GROQ_API_KEY"),
    "groq": ("OPENAI_API_KEY", "GROQ_API_KEY"),
    "openai-compatible": ("OPENAI_API_KEY", "GROQ_API_KEY"),
}


def provider_name() -> str:
    return os.environ.get("SPIKE_LLM_PROVIDER", "openrouter").lower()


def has_llm_key() -> bool:
    """True when the configured provider has a key, so LLM checks can run."""
    return any(os.environ.get(k) for k in PROVIDER_KEYS.get(provider_name(), ()))


class FailingChat:
    """Always fails. Used by the Warden to prove the system degrades gracefully."""

    name = "failing"

    def chat(self, system, messages, tools=None, max_tokens=1500) -> Reply:
        raise LLMError("simulated outage: model unreachable")

    def tool_result_message(self, results):
        return []


def make_chat(provider: str | None = None, model: str | None = None) -> ChatModel:
    provider = (provider or provider_name()).lower()
    if provider == "openrouter":
        return openrouter_chat(model)
    if provider == "anthropic":
        return AnthropicChat(model)
    if provider in ("openai", "groq", "openai-compatible"):
        return OpenAICompatChat(model)
    if provider == "failing":
        return FailingChat()
    raise LLMError(f"unknown provider {provider!r}")


def parse_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of model text (handles ```json fences)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise LLMError("no JSON object in model output")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"model output is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMError("model output JSON is not an object")
    return value
