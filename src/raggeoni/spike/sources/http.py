"""Small HTTP helper: retries with backoff, a clear user agent, no extra deps."""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from html.parser import HTMLParser

USER_AGENT = "raggeoni-spike/0.2 (NEM price spike research; github.com/Vanamali-Sims/Raggeoni)"


class FetchError(RuntimeError):
    """A source could not be reached after retries."""


def fetch(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
    retries: int = 3,
    backoff: float = 2.0,
) -> bytes:
    merged = {"User-Agent": USER_AGENT, **(headers or {})}
    last: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=merged)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            # 4xx other than 429 will not get better by retrying.
            if 400 <= exc.code < 500 and exc.code != 429:
                detail = exc.read()[:300].decode("utf-8", errors="replace")
                raise FetchError(f"HTTP {exc.code} for {url}: {detail}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last = exc
        if attempt < retries - 1:
            time.sleep(backoff * (2**attempt))
    raise FetchError(f"could not fetch {url} after {retries} tries: {last}")


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            for name, value in attrs:
                if name.lower() == "href" and value:
                    self.links.append(value)


def list_links(html: str) -> list[str]:
    parser = _Links()
    parser.feed(html)
    return parser.links
