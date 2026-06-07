"""
LLMBackend Protocol — the only surface judges touch when calling a model.

Swap providers by injecting a different backend; no judge code changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    """Structural protocol: any object with this signature works as a backend."""

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        """Return the model's text response to (system, user) messages."""
        ...
