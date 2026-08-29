"""Arm 1: call a chat model, temperature 0, JSON plan out."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

PROMPT_PATH = Path(__file__).resolve().parent / "prompt.txt"
DEFAULT_MODEL = "gpt-4o-mini"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("RAGGEONI_API_KEY")


def complete_plan(question: str, *, model: str | None = None, timeout: int = 60) -> str:
    key = api_key()
    if not key:
        raise RuntimeError(
            "No API key. Set OPENAI_API_KEY, then run: uv run python -m raggeoni.eval --arm 1"
        )
    body = {
        "model": model or os.environ.get("RAGGEONI_MODEL", DEFAULT_MODEL),
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": load_prompt()},
            {"role": "user", "content": question},
        ],
    }
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        OPENAI_URL,
        data=raw,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"model HTTP {exc.code}: {detail}") from exc
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected model response: {payload!r}") from exc
