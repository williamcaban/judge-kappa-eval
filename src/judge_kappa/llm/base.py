"""
LLMBackend Protocol — the only surface judges touch when calling a model.

Swap providers by injecting a different backend; no judge code changes.

Two levels of capability:

  LLMBackend          complete(system, user, temperature) -> str          (required)
  UsageAwareBackend   complete_with_usage(...) -> LLMResponse             (optional)

Existing custom backends that only implement ``complete()`` keep working: callers use
:func:`call_backend`, which wraps the plain string in an ``LLMResponse`` without usage.
The optional ``describe()`` method (a JSON-serialisable dict, never containing secrets)
feeds run provenance; ``stats`` (a ``UsageStats``) feeds usage reporting.
"""

from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

from judge_kappa.llm.types import LLMResponse, UsageSnapshot, UsageStats


@runtime_checkable
class LLMBackend(Protocol):
    """Structural protocol: any object with this signature works as a backend."""

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        """Return the model's text response to (system, user) messages."""
        ...


@runtime_checkable
class UsageAwareBackend(Protocol):
    """Optional richer protocol: the reply plus token usage, finish reason and latency."""

    def complete_with_usage(
        self, system: str, user: str, temperature: float = 0.0
    ) -> LLMResponse:
        ...


def call_backend(
    backend: LLMBackend, system: str, user: str, temperature: float = 0.0
) -> LLMResponse:
    """Call any backend and always return an ``LLMResponse``."""
    rich = getattr(backend, "complete_with_usage", None)
    if callable(rich):
        resp = rich(system, user, temperature)
        if isinstance(resp, LLMResponse):
            return resp
        return LLMResponse(text=str(resp))
    start = time.perf_counter()
    text = backend.complete(system, user, temperature)
    return LLMResponse(text=text, latency_s=time.perf_counter() - start)


def describe_backend(backend: object) -> dict[str, Any]:
    """Best-effort, secret-free description of a backend for provenance."""
    describe = getattr(backend, "describe", None)
    if callable(describe):
        try:
            described = describe()
            if isinstance(described, dict):
                return described
        except Exception:  # provenance must never break an evaluation
            pass
    model = getattr(backend, "model", None) or getattr(backend, "_model", None)
    return {"provider": type(backend).__name__, "model": model if isinstance(model, str) else None}


def backend_usage(backend: object) -> UsageSnapshot | None:
    """Cumulative usage of a backend, or None if it does not track any."""
    stats = getattr(backend, "stats", None)
    if isinstance(stats, UsageStats):
        return stats.snapshot()
    return None
