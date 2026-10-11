"""
Tests for the explicit-parameter, usage-aware LLM backends.

SDK clients are replaced with recording stubs; no network and no API keys.
Covers: temperature is always sent (including 0.0) unless omitted, max_tokens wiring,
timeout/retry forwarding, usage + finish_reason capture, truncation detection,
secret-free provenance, and compatibility of the plain ``complete()`` Protocol.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import pytest

from judge_kappa.llm import (
    AnthropicBackend,
    LLMResponse,
    OpenAIBackend,
    UsageStats,
    backend_usage,
    call_backend,
    describe_backend,
)
from judge_kappa.llm.types import sanitize_url


class _RecordingOpenAI:
    """Stand-in for openai.OpenAI; records constructor and create() kwargs."""

    instances: list[_RecordingOpenAI] = []   # noqa: RUF012

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.sent: dict[str, Any] | None = None
        self.reply = SimpleNamespace(
            model="gpt-echo",
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )
        outer = self

        class _Completions:
            def create(self, **kw: Any) -> Any:
                outer.sent = kw
                if isinstance(outer.reply, Exception):
                    raise outer.reply
                return outer.reply

        self.chat = SimpleNamespace(completions=_Completions())
        _RecordingOpenAI.instances.append(self)


class _RecordingAnthropic:
    instances: list[_RecordingAnthropic] = []   # noqa: RUF012

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.sent: dict[str, Any] | None = None
        self.reply: Any = SimpleNamespace(
            model="claude-echo",
            content=[SimpleNamespace(type="text", text="hi")],
            usage=SimpleNamespace(input_tokens=5, output_tokens=3),
            stop_reason="end_turn",
        )
        outer = self

        class _Messages:
            def create(self, **kw: Any) -> Any:
                outer.sent = kw
                if isinstance(outer.reply, Exception):
                    raise outer.reply
                return outer.reply

        self.messages = _Messages()
        _RecordingAnthropic.instances.append(self)


@pytest.fixture
def openai_stub(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingOpenAI]:
    import openai

    _RecordingOpenAI.instances = []
    monkeypatch.setattr(openai, "OpenAI", _RecordingOpenAI)
    return _RecordingOpenAI


@pytest.fixture
def anthropic_stub(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingAnthropic]:
    import anthropic

    _RecordingAnthropic.instances = []
    monkeypatch.setattr(anthropic, "Anthropic", _RecordingAnthropic)
    return _RecordingAnthropic


# ── Anthropic ─────────────────────────────────────────────────────────────────


class TestAnthropicBackend:
    def test_temperature_zero_is_sent(self, anthropic_stub):
        # Regression: 0.0 used to be dropped, so the API default (1.0) applied.
        b = AnthropicBackend("claude-x", api_key="k")
        b.complete("sys", "user", 0.0)
        sent = anthropic_stub.instances[-1].sent
        assert sent["temperature"] == 0.0
        assert sent["max_tokens"] == 2048
        assert sent["system"] == "sys"

    def test_nonzero_temperature_is_sent(self, anthropic_stub):
        AnthropicBackend("claude-x", api_key="k").complete("s", "u", 0.7)
        assert anthropic_stub.instances[-1].sent["temperature"] == 0.7

    def test_omit_temperature_never_sends_it(self, anthropic_stub):
        b = AnthropicBackend("claude-x", api_key="k", omit_temperature=True)
        b.complete("s", "u", 0.7)
        assert "temperature" not in anthropic_stub.instances[-1].sent
        assert b.describe()["omit_temperature"] is True

    def test_max_tokens_configurable(self, anthropic_stub):
        AnthropicBackend("claude-x", api_key="k", max_tokens=8000).complete("s", "u")
        assert anthropic_stub.instances[-1].sent["max_tokens"] == 8000

    def test_timeout_and_retries_forwarded_to_sdk_client(self, anthropic_stub):
        AnthropicBackend("claude-x", api_key="k", timeout_s=30.0, max_retries=5)
        kw = anthropic_stub.instances[-1].init_kwargs
        assert kw["timeout"] == 30.0 and kw["max_retries"] == 5

    def test_unset_timeout_and_retries_keep_sdk_defaults(self, anthropic_stub):
        AnthropicBackend("claude-x", api_key="k")
        kw = anthropic_stub.instances[-1].init_kwargs
        assert "timeout" not in kw and "max_retries" not in kw

    def test_usage_finish_reason_and_model_captured(self, anthropic_stub):
        resp = AnthropicBackend("claude-x", api_key="k").complete_with_usage("s", "u")
        assert resp.text == "hi"
        assert (resp.input_tokens, resp.output_tokens) == (5, 3)
        assert resp.finish_reason == "end_turn"
        assert resp.model == "claude-echo"
        assert resp.latency_s is not None and resp.latency_s >= 0
        assert not resp.truncated

    def test_truncation_detected_from_stop_reason(self, anthropic_stub):
        b = AnthropicBackend("claude-x", api_key="k")
        anthropic_stub.instances[-1].reply.stop_reason = "max_tokens"
        resp = b.complete_with_usage("s", "u")
        assert resp.truncated
        assert b.stats.snapshot().truncated == 1

    def test_thinking_block_is_skipped_and_text_blocks_joined(self, anthropic_stub):
        b = AnthropicBackend("claude-x", api_key="k")
        anthropic_stub.instances[-1].reply.content = [
            SimpleNamespace(type="thinking", thinking="hmm"),
            SimpleNamespace(type="text", text="part one "),
            SimpleNamespace(type="text", text="part two"),
        ]
        assert b.complete("s", "u") == "part one part two"

    def test_temperature_rejection_gets_actionable_note(self, anthropic_stub):
        b = AnthropicBackend("claude-x", api_key="k")
        anthropic_stub.instances[-1].reply = RuntimeError("temperature is not supported for this model")
        with pytest.raises(RuntimeError) as err:
            b.complete("s", "u")
        assert any("omit_temperature" in n for n in getattr(err.value, "__notes__", []))
        assert b.stats.snapshot().errors == 1

    def test_describe_has_no_secrets(self, anthropic_stub):
        d = AnthropicBackend("claude-x", api_key="sk-secret-123", max_tokens=100).describe()
        assert "sk-secret-123" not in repr(d)
        assert d == {
            "provider": "anthropic", "model": "claude-x", "base_url": None,
            "max_tokens": 100, "omit_temperature": False, "timeout_s": None, "max_retries": None,
        }


# ── OpenAI-compatible ─────────────────────────────────────────────────────────


class TestOpenAIBackend:
    def test_temperature_zero_is_sent(self, openai_stub):
        OpenAIBackend("m", api_key="k").complete("s", "u", 0.0)
        sent = openai_stub.instances[-1].sent
        assert sent["temperature"] == 0.0
        assert "max_tokens" not in sent                      # unset keeps provider default

    def test_omit_temperature(self, openai_stub):
        OpenAIBackend("o1", api_key="k", omit_temperature=True).complete("s", "u", 0.3)
        assert "temperature" not in openai_stub.instances[-1].sent

    @pytest.mark.parametrize("param", ["max_tokens", "max_completion_tokens"])
    def test_max_tokens_uses_selected_wire_name(self, openai_stub, param):
        OpenAIBackend("m", api_key="k", max_tokens=512, max_tokens_param=param).complete("s", "u")
        sent = openai_stub.instances[-1].sent
        assert sent[param] == 512
        assert ({"max_tokens", "max_completion_tokens"} - {param}).isdisjoint(sent)

    def test_timeout_retries_and_base_url_forwarded(self, openai_stub):
        OpenAIBackend("m", api_key="k", base_url="http://localhost:11434/v1", timeout_s=12.5, max_retries=0)
        kw = openai_stub.instances[-1].init_kwargs
        assert kw["base_url"] == "http://localhost:11434/v1"
        assert kw["timeout"] == 12.5 and kw["max_retries"] == 0

    def test_usage_and_finish_reason_captured(self, openai_stub):
        resp = OpenAIBackend("m", api_key="k").complete_with_usage("s", "u")
        assert resp.text == "hello"
        assert (resp.input_tokens, resp.output_tokens) == (11, 7)
        assert resp.finish_reason == "stop"
        assert resp.model == "gpt-echo"

    def test_length_finish_reason_is_truncation(self, openai_stub):
        b = OpenAIBackend("m", api_key="k")
        openai_stub.instances[-1].reply.choices[0].finish_reason = "length"
        assert b.complete_with_usage("s", "u").truncated

    def test_missing_usage_and_empty_content_are_tolerated(self, openai_stub):
        b = OpenAIBackend("m", api_key="k")
        reply = openai_stub.instances[-1].reply
        reply.usage = None
        reply.choices[0].message.content = None
        resp = b.complete_with_usage("s", "u")
        assert resp.text == "" and resp.input_tokens is None
        assert b.stats.snapshot().calls_without_usage == 1

    def test_describe_strips_credentials_from_base_url(self, openai_stub):
        d = OpenAIBackend(
            "m", api_key="sk-secret", base_url="https://user:pass@host.example:8443/v1?token=abc#frag"
        ).describe()
        assert d["base_url"] == "https://host.example:8443/v1"
        assert "sk-secret" not in repr(d) and "pass" not in repr(d) and "abc" not in repr(d)


# ── call_backend / describe_backend / stats ──────────────────────────────────


class _PlainBackend:
    """A third-party backend implementing only the original Protocol."""

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return f"echo:{user}"


class TestCompatibility:
    def test_plain_backend_still_works_through_call_backend(self):
        resp = call_backend(_PlainBackend(), "s", "hello", 0.0)
        assert isinstance(resp, LLMResponse)
        assert resp.text == "echo:hello"
        assert resp.input_tokens is None and resp.latency_s is not None

    def test_rich_backend_response_passes_through(self, openai_stub):
        resp = call_backend(OpenAIBackend("m", api_key="k"), "s", "u", 0.0)
        assert resp.input_tokens == 11

    def test_describe_and_usage_fall_back_for_plain_backends(self):
        d = describe_backend(_PlainBackend())
        assert d["provider"] == "_PlainBackend"
        assert backend_usage(_PlainBackend()) is None

    def test_describe_never_raises(self):
        class _Broken:
            def describe(self) -> dict[str, Any]:
                raise RuntimeError("boom")

        assert describe_backend(_Broken())["provider"] == "_Broken"


class TestUsageStats:
    def test_thread_safe_accumulation(self):
        stats = UsageStats()
        resp = LLMResponse(text="x", input_tokens=2, output_tokens=3, latency_s=0.01)

        def work() -> None:
            for _ in range(500):
                stats.record(resp)

        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        snap = stats.snapshot()
        assert snap.calls == 4000
        assert snap.input_tokens == 8000 and snap.output_tokens == 12000

    def test_snapshot_minus_scopes_to_a_run(self):
        stats = UsageStats()
        stats.record(LLMResponse(text="a", input_tokens=1, output_tokens=1))
        before = stats.snapshot()
        stats.record(LLMResponse(text="b", input_tokens=4, output_tokens=6, finish_reason="length"))
        delta = stats.snapshot().minus(before)
        assert (delta.calls, delta.input_tokens, delta.output_tokens, delta.truncated) == (1, 4, 6, 1)


def test_sanitize_url_edge_cases():
    assert sanitize_url(None) is None
    assert sanitize_url("") is None
    assert sanitize_url("http://localhost:8000/v1") == "http://localhost:8000/v1"
