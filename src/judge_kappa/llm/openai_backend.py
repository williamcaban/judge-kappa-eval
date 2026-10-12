"""
OpenAI-compatible backend — covers OpenAI, vLLM, Ollama, Together, Groq,
OpenRouter, and any server that exposes /v1/chat/completions.

Request parameters are explicit and inspectable:

  temperature     always sent, including 0.0, unless ``omit_temperature=True``
                  (needed for reasoning models that reject the parameter)
  max_tokens      sent only when set; ``max_tokens_param`` selects the wire name
                  (``"max_completion_tokens"`` for OpenAI reasoning models)
  timeout_s,
  max_retries     forwarded to the OpenAI SDK client; unset keeps the SDK defaults
"""

from __future__ import annotations

import time
from typing import Any, Literal

from judge_kappa.llm.types import LLMResponse, UsageStats, sanitize_url


class OpenAIBackend:
    """Backend for OpenAI-compatible chat-completion servers (OpenAI, vLLM, Ollama, Groq, ...).

    Sends ``temperature`` explicitly, captures token usage, finish reason and latency, and forwards
    timeout and retry settings to the SDK.
    """
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        max_tokens: int | None = None,
        max_tokens_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens",
        omit_temperature: bool = False,
        timeout_s: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        """Create the backend and its SDK client.

        Args:
            model: Model id.
            api_key: API key (None lets the SDK read the environment).
            base_url: Endpoint override for vLLM, Ollama, OpenRouter, etc.
            max_tokens: Output cap; sent only when set.
            max_tokens_param: Wire name of the cap (``max_completion_tokens`` for OpenAI reasoning
                models).
            omit_temperature: Never send ``temperature`` (for models that reject it).
            timeout_s: Per-request timeout; None keeps the SDK default.
            max_retries: SDK retries for transient errors; None keeps the SDK default.

        Raises:
            ImportError: If the ``openai`` extra is not installed.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the openai extra: pip install 'judge-kappa[openai]'") from exc

        self._model = model
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._max_tokens_param = max_tokens_param
        self._omit_temperature = omit_temperature
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self.stats = UsageStats()

        client_kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
        if timeout_s is not None:
            client_kwargs["timeout"] = timeout_s
        if max_retries is not None:
            client_kwargs["max_retries"] = max_retries
        self._client = OpenAI(**client_kwargs)

    def describe(self) -> dict[str, Any]:
        """Secret-free description of this backend (base URL stripped of credentials)."""
        return {
            "provider": "openai",
            "model": self._model,
            "base_url": sanitize_url(self._base_url),
            "max_tokens": self._max_tokens,
            "max_tokens_param": self._max_tokens_param,
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
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        if not self._omit_temperature:
            kwargs["temperature"] = temperature
        if self._max_tokens is not None:
            kwargs[self._max_tokens_param] = self._max_tokens

        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception:
            self.stats.record_error()
            raise
        latency = time.perf_counter() - start

        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        result = LLMResponse(
            text=choice.message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            finish_reason=getattr(choice, "finish_reason", None),
            model=getattr(resp, "model", None),
            latency_s=latency,
        )
        self.stats.record(result)
        return result
