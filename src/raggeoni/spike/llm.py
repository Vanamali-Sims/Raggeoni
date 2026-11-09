"""Minimal chat-with-tools clients. No SDKs, so CI stays light.

Providers (``SPIKE_LLM_PROVIDER``):

* ``anthropic`` (default): Claude Messages API, key in ``ANTHROPIC_API_KEY``.
* ``openai``: any OpenAI-compatible chat completions API, key in
  ``OPENAI_API_KEY``; set ``SPIKE_LLM_BASE_URL`` for Groq or others
  (e.g. https://api.groq.com/openai/v1 with ``GROQ_API_KEY``).

Model via ``SPIKE_MODEL``. Cost is only reported when prices are configured
(``SPIKE_PRICE_IN_PER_MTOK`` / ``SPIKE_PRICE_OUT_PER_MTOK``), because
hard-coded prices go stale.
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


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    failures: int = 0
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

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self) -> float | None:
        pin = os.environ.get("SPIKE_PRICE_IN_PER_MTOK")
        pout = os.environ.get("SPIKE_PRICE_OUT_PER_MTOK")
        if not pin or not pout:
            return None
        return round(
            self.input_tokens / 1e6 * float(pin) + self.output_tokens / 1e6 * float(pout), 5
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "seconds": round(self.seconds, 2),
            "failures": self.failures,
            "cost_usd": self.cost_usd(),
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
            raise LLMError(f"HTTP {exc.code}: {detail}") from exc
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
    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        self.name = model or os.environ.get("SPIKE_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY (or GROQ_API_KEY) is not set")
        self.url = (
            os.environ.get("SPIKE_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            + "/chat/completions"
        )

    def chat(self, system, messages, tools=None, max_tokens=1500) -> Reply:
        body: dict[str, Any] = {
            "model": self.name,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        payload = _post(self.url, body, {"Authorization": f"Bearer {self.api_key}"})
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
        usage = payload.get("usage", {})
        return Reply(
            text=msg.get("content") or "",
            tool_calls=calls,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            raw_assistant={k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")},
        )

    def tool_result_message(self, results):
        return [
            {"role": "tool", "tool_call_id": call.id, "content": content}
            for call, content in results
        ]


class FailingChat:
    """Always fails. Used by the Warden to prove the system degrades gracefully."""

    name = "failing"

    def chat(self, system, messages, tools=None, max_tokens=1500) -> Reply:
        raise LLMError("simulated outage: model unreachable")

    def tool_result_message(self, results):
        return []


def make_chat(provider: str | None = None, model: str | None = None) -> ChatModel:
    provider = (provider or os.environ.get("SPIKE_LLM_PROVIDER", "anthropic")).lower()
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
