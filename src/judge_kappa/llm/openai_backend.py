"""
OpenAI-compatible backend — covers OpenAI, vLLM, Ollama, Together, Groq,
and any server that exposes /v1/chat/completions.
"""

from __future__ import annotations


class OpenAIBackend:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the openai extra: pip install 'jury-eval[openai]'") from exc

        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp.choices[0].message.content or ""
