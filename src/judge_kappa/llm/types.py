"""
Response and usage types for the LLM layer.

The public ``LLMBackend`` Protocol still returns a plain ``str`` from ``complete()``.
Backends that can do better also implement ``complete_with_usage()`` and return an
``LLMResponse``; callers go through :func:`judge_kappa.llm.base.call_backend`, which
falls back to ``complete()`` for backends that only implement the basic Protocol.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

# finish_reason values that mean the reply was cut off by the output-token limit
# (OpenAI-compatible: "length"; Anthropic: "max_tokens"; Gemini: "MAX_TOKENS").
TRUNCATION_REASONS = frozenset({"length", "max_tokens", "MAX_TOKENS"})


@dataclass(frozen=True)
class LLMResponse:
    """One model reply plus the metadata a measurement library needs to audit it."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None   # provider-native value ("stop", "length", "end_turn", ...)
    model: str | None = None           # model id echoed by the provider, if any
    latency_s: float | None = None
    cached: bool = False               # True when served from a CachedBackend

    @property
    def truncated(self) -> bool:
        """True when the provider stopped because of the output-token limit."""
        return self.finish_reason in TRUNCATION_REASONS


@dataclass(frozen=True)
class UsageSnapshot:
    """Immutable view of a backend's cumulative usage."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    truncated: int = 0
    cache_hits: int = 0
    errors: int = 0
    latency_s: float = 0.0
    calls_without_usage: int = 0   # replies for which the provider reported no token counts

    def minus(self, earlier: UsageSnapshot) -> UsageSnapshot:
        """Usage accumulated since ``earlier`` (used to scope stats to one run)."""
        return UsageSnapshot(
            calls=self.calls - earlier.calls,
            input_tokens=self.input_tokens - earlier.input_tokens,
            output_tokens=self.output_tokens - earlier.output_tokens,
            truncated=self.truncated - earlier.truncated,
            cache_hits=self.cache_hits - earlier.cache_hits,
            errors=self.errors - earlier.errors,
            latency_s=round(self.latency_s - earlier.latency_s, 6),
            calls_without_usage=self.calls_without_usage - earlier.calls_without_usage,
        )


@dataclass
class UsageStats:
    """Thread-safe usage accumulator owned by a backend."""

    _calls: int = 0
    _input_tokens: int = 0
    _output_tokens: int = 0
    _truncated: int = 0
    _cache_hits: int = 0
    _errors: int = 0
    _latency_s: float = 0.0
    _calls_without_usage: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, resp: LLMResponse) -> None:
        with self._lock:
            self._calls += 1
            if resp.cached:
                # A cache hit costs nothing: count the call and the hit, not tokens or latency,
                # so token totals reflect what was actually billed. Truncation is a property
                # of the reply, not of billing, so a resumed run still reports it.
                self._cache_hits += 1
                if resp.truncated:
                    self._truncated += 1
                return
            if resp.truncated:
                self._truncated += 1
            if resp.input_tokens is None and resp.output_tokens is None:
                self._calls_without_usage += 1
            self._input_tokens += resp.input_tokens or 0
            self._output_tokens += resp.output_tokens or 0
            self._latency_s += resp.latency_s or 0.0

    def record_error(self) -> None:
        with self._lock:
            self._errors += 1

    def snapshot(self) -> UsageSnapshot:
        with self._lock:
            return UsageSnapshot(
                calls=self._calls,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                truncated=self._truncated,
                cache_hits=self._cache_hits,
                errors=self._errors,
                latency_s=round(self._latency_s, 6),
                calls_without_usage=self._calls_without_usage,
            )


def sanitize_url(url: str | None) -> str | None:
    """Drop credentials, query string and fragment so a base URL is safe to record."""
    if not url:
        return None
    parts = urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:                  # IPv6 literal: urlsplit strips the brackets
        host = f"[{host}]"
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))
