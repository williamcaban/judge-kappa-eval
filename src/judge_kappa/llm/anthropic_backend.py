"""
Native Anthropic backend — avoids the OpenAI compat shim for claude models,
preserving access to extended thinking, caching, and other Anthropic-only features.

Request parameters are explicit and inspectable:

  temperature   always sent, including 0.0, unless ``omit_temperature=True``.
                Earlier releases dropped ``temperature=0.0``, so the API default
                (documented as 1.0) applied and judges were not deterministic.
  max_tokens    required by the API; defaults to 2048 and is configurable.
                Replies that hit the limit are reported (``LLMResponse.truncated``).
  timeout_s,
  max_retries   forwarded to the Anthropic SDK client; unset keeps the SDK defaults.
"""

from __future__ import annotations

import time
from typing import Any

from judge_kappa.llm.types import LLMResponse, UsageStats

DEFAULT_MAX_TOKENS = 2048


class AnthropicBackend:
    """Backend for the Anthropic Messages API.

    Sends ``temperature`` and ``max_tokens`` explicitly, captures token usage, finish reason and
    latency, and forwards timeout and retry settings to the SDK.
    """
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        omit_temperature: bool = False,
        timeout_s: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        """Create the backend and its SDK client.

        Args:
            model: Model id.
            api_key: API key (None lets the SDK read the environment).
            max_tokens: Output cap; required by the API (default 2048).
            omit_temperature: Never send ``temperature`` (for models that reject it).
            timeout_s: Per-request timeout; None keeps the SDK default.
            max_retries: SDK retries for transient errors; None keeps the SDK default.

        Raises:
            ImportError: If the ``anthropic`` extra is not installed.
        """
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Install the anthropic extra: pip install 'judge-kappa[anthropic]'") from exc

        self._model = model
        self._max_tokens = max_tokens
        self._omit_temperature = omit_temperature
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self.stats = UsageStats()

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if timeout_s is not None:
            client_kwargs["timeout"] = timeout_s
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries
        self._client = anthropic.Anthropic(**client_kwargs)

    def describe(self) -> dict[str, Any]:
        """Secret-free description of this backend, for provenance and cache keys."""
        return {
            "provider": "anthropic",
            "model": self._model,
            "base_url": None,
            "max_tokens": self._max_tokens,
            "omit_temperature": self._omit_temperature,
            "timeout_s": self._timeout_s,
            "max_retries": self._max_retries,
        }

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        """Return only the reply text; see ``complete_with_usage``."""
        return self.complete_with_usage(system, user, temperature).text

    def complete_with_usage(
        self, system: str, user: str, temperature: float = 0.0
    ) -> LLMResponse:
        """Send one request and return the reply with its usage metadata.

        Token counts, finish reason, model and latency are also accumulated in ``stats``.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if not self._omit_temperature:
            kwargs["temperature"] = temperature

        start = time.perf_counter()
        try:
            msg = self._client.messages.create(**kwargs)
        except Exception as exc:
            self.stats.record_error()
            if "temperature" in str(exc).lower() and not self._omit_temperature:
                exc.add_note(
                    "judge-kappa sends `temperature` explicitly. If this model rejects it, "
                    "construct the backend with omit_temperature=True "
                    "(YAML: backend.omit_temperature: true)."
                )
            raise
        latency = time.perf_counter() - start

        # Join text blocks; the first block may be a thinking block when thinking is enabled.
        text = "".join(
            str(block.text)
            for block in (msg.content or [])
            if getattr(block, "type", "text") == "text" and hasattr(block, "text")
        )
        usage = getattr(msg, "usage", None)
        result = LLMResponse(
            text=text,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            finish_reason=getattr(msg, "stop_reason", None),
            model=getattr(msg, "model", None),
            latency_s=latency,
        )
        self.stats.record(result)
        return result
