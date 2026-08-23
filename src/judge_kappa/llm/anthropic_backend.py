"""
Native Anthropic backend — avoids the OpenAI compat shim for claude models,
preserving access to extended thinking, caching, and other Anthropic-only features.
"""

from __future__ import annotations


class AnthropicBackend:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("Install the anthropic extra: pip install 'jury-eval[anthropic]'") from exc

        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        kwargs: dict[str, object] = {
            "model": self._model,
            "max_tokens": 2048,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if temperature != 0.0:
            kwargs["temperature"] = temperature
        msg = self._client.messages.create(**kwargs)  # type: ignore[call-overload]
        return str(msg.content[0].text)
