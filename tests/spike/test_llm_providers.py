"""Provider clients: request shape, tool calls, reported cost, key detection. No network."""

from __future__ import annotations

import json

import pytest

from raggeoni.spike import llm
from raggeoni.spike.llm import LLMError, Reply, ToolCall, Usage, has_llm_key, make_chat

OPENROUTER_REPLY = {
    "id": "gen-1",
    "model": "anthropic/claude-sonnet-5",
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "unusualness",
                            "arguments": json.dumps(
                                {"region": "SA1", "metric": "wind", "at": "2031-01-15 19:50:00"}
                            ),
                        },
                    }
                ],
            }
        }
    ],
    "usage": {
        "prompt_tokens": 1200,
        "completion_tokens": 80,
        "total_tokens": 1280,
        "cost": 0.00492,
    },
}


@pytest.fixture
def captured(monkeypatch):
    calls: list[dict] = []

    def fake_post(url, body, headers, timeout=120, retries=3):
        calls.append({"url": url, "body": body, "headers": headers})
        return OPENROUTER_REPLY

    monkeypatch.setattr(llm, "_post", fake_post)
    return calls


def test_openrouter_is_the_default_provider(monkeypatch, captured):
    monkeypatch.delenv("SPIKE_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("SPIKE_MODEL", raising=False)
    monkeypatch.delenv("SPIKE_LLM_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    chat = make_chat()
    tools = [
        {
            "name": "unusualness",
            "description": "d",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    reply = chat.chat("system prompt", [{"role": "user", "content": "go"}], tools=tools)

    req = captured[0]
    assert req["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer sk-or-test"
    assert req["headers"]["HTTP-Referer"].startswith("https://github.com/")
    assert req["body"]["model"] == "anthropic/claude-sonnet-5"
    assert req["body"]["messages"][0] == {"role": "system", "content": "system prompt"}
    assert req["body"]["tools"][0]["type"] == "function"

    assert reply.tool_calls == [
        ToolCall(
            "call_1",
            "unusualness",
            {"region": "SA1", "metric": "wind", "at": "2031-01-15 19:50:00"},
        )
    ]
    assert reply.input_tokens == 1200 and reply.output_tokens == 80
    assert reply.cost_usd == pytest.approx(0.00492)
    # the assistant turn is replayed as-is, then tool results go back as role=tool
    assert reply.raw_assistant["tool_calls"][0]["id"] == "call_1"
    assert chat.tool_result_message([(reply.tool_calls[0], "{}")]) == [
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"}
    ]


def test_model_slug_override(monkeypatch, captured):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("SPIKE_MODEL", "openai/gpt-5-mini")
    make_chat("openrouter").chat("s", [{"role": "user", "content": "x"}])
    assert captured[0]["body"]["model"] == "openai/gpt-5-mini"


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        make_chat("openrouter")


def test_provider_error_payload_raises(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(
        llm,
        "_post",
        lambda *a, **k: {
            "error": {"message": "No endpoints found that support tool use", "code": 404}
        },
    )
    with pytest.raises(LLMError, match="tool use"):
        make_chat("openrouter").chat("s", [{"role": "user", "content": "x"}])


def test_has_llm_key_follows_the_provider(monkeypatch):
    for key in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("SPIKE_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert has_llm_key() is False  # default provider is openrouter
    monkeypatch.setenv("OPENROUTER_API_KEY", "o")
    assert has_llm_key() is True
    monkeypatch.setenv("SPIKE_LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert has_llm_key() is True


def test_usage_prefers_reported_cost(monkeypatch):
    monkeypatch.delenv("SPIKE_PRICE_IN_PER_MTOK", raising=False)
    monkeypatch.delenv("SPIKE_PRICE_OUT_PER_MTOK", raising=False)
    u = Usage()
    u.add("lead", Reply("", [], 100, 10, cost_usd=0.001), 0.5)
    u.add("writer", Reply("", [], 200, 20, cost_usd=0.002), 0.5)
    u.add("evidence", None, 0.1, failed=True)  # failed calls cost nothing
    assert u.cost_usd() == pytest.approx(0.003)
    assert u.to_dict()["cost_source"] == "provider"


def test_usage_falls_back_to_price_estimate_then_unknown(monkeypatch):
    u = Usage()
    u.add("lead", Reply("", [], 1_000_000, 0), 1.0)  # no reported cost
    monkeypatch.delenv("SPIKE_PRICE_IN_PER_MTOK", raising=False)
    monkeypatch.delenv("SPIKE_PRICE_OUT_PER_MTOK", raising=False)
    assert u.cost_usd() is None and u.cost_source() == "unknown"
    monkeypatch.setenv("SPIKE_PRICE_IN_PER_MTOK", "3")
    monkeypatch.setenv("SPIKE_PRICE_OUT_PER_MTOK", "15")
    assert u.cost_usd() == 3.0 and u.cost_source() == "estimate"


def test_llm_brain_without_key_falls_back_to_rules(monkeypatch, capsys):
    from raggeoni.spike.agents import HeuristicBrain, make_brain

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("SPIKE_LLM_PROVIDER", "openrouter")
    brain = make_brain("llm")
    assert isinstance(brain, HeuristicBrain)
    assert brain.fallbacks and "OPENROUTER_API_KEY" in brain.fallbacks[0]
    assert "using the rules brain" in capsys.readouterr().err
