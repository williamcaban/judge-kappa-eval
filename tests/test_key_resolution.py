"""
Tests for API key resolution (_resolve_api_key).

This module is pure logic — no LLM calls, no backends constructed.
Tests cover all three cases of the api_key_env field and all four
provider scenarios: Anthropic, OpenAI, OpenRouter, Ollama/vLLM.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from judge_kappa.cli.builder import _resolve_api_key


class TestResolveApiKey:
    # ── Case 1: api_key_env is None (use provider default) ───────────────────

    def test_none_anthropic_reads_anthropic_key(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            key = _resolve_api_key(None, "anthropic")
        assert key == "sk-ant-test"

    def test_none_openai_reads_openai_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai-test"}):
            key = _resolve_api_key(None, "openai")
        assert key == "sk-openai-test"

    def test_none_unknown_provider_returns_none(self):
        key = _resolve_api_key(None, "some-future-provider")
        assert key is None

    def test_none_missing_default_var_returns_none(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            key = _resolve_api_key(None, "openai")
        assert key is None  # SDK will raise its own error later

    # ── Case 2: api_key_env == "" (no auth — Ollama, vLLM without token) ─────

    def test_empty_string_returns_placeholder(self):
        key = _resolve_api_key("", "openai")
        assert key == "no-key"

    def test_empty_string_anthropic_also_returns_placeholder(self):
        key = _resolve_api_key("", "anthropic")
        assert key == "no-key"

    def test_empty_string_does_not_read_env(self):
        # Even if OPENAI_API_KEY is set, empty api_key_env ignores it
        with patch.dict(os.environ, {"OPENAI_API_KEY": "real-key"}):
            key = _resolve_api_key("", "openai")
        assert key == "no-key"

    # ── Case 3: api_key_env == "MY_VAR" (custom env var) ─────────────────────

    def test_custom_var_read_correctly(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}):
            key = _resolve_api_key("OPENROUTER_API_KEY", "openai")
        assert key == "sk-or-test"

    def test_custom_var_missing_raises_valueerror(self):
        env = {k: v for k, v in os.environ.items() if k != "MY_CUSTOM_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="MY_CUSTOM_KEY"):
                _resolve_api_key("MY_CUSTOM_KEY", "openai")

    def test_error_message_includes_export_hint(self):
        env = {k: v for k, v in os.environ.items() if k != "GROQ_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="export GROQ_API_KEY"):
                _resolve_api_key("GROQ_API_KEY", "openai")

    def test_custom_var_works_for_groq(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk-groq-test"}):
            key = _resolve_api_key("GROQ_API_KEY", "openai")
        assert key == "gsk-groq-test"

    def test_custom_var_works_for_together(self):
        with patch.dict(os.environ, {"TOGETHER_API_KEY": "tog-test"}):
            key = _resolve_api_key("TOGETHER_API_KEY", "openai")
        assert key == "tog-test"

    # ── Provider scenario matrix ──────────────────────────────────────────────

    @pytest.mark.parametrize("scenario,api_key_env,provider,env_var,env_value,expected", [
        # api_key_env=None → use provider default env var
        ("anthropic-hosted",  None,               "anthropic", "ANTHROPIC_API_KEY",  "sk-ant-x",  "sk-ant-x"),
        ("openai-hosted",     None,               "openai",    "OPENAI_API_KEY",     "sk-oai-x",  "sk-oai-x"),
        # api_key_env=explicit var name → use that var regardless of provider
        ("openrouter",        "OPENROUTER_API_KEY","openai",   "OPENROUTER_API_KEY", "sk-or-x",   "sk-or-x"),
        ("ollama-no-auth",    "",                 "openai",    None,                 None,         "no-key"),
        ("vllm-no-auth",      "",                 "openai",    None,                 None,         "no-key"),
        ("vllm-with-token",   "VLLM_API_KEY",     "openai",    "VLLM_API_KEY",      "vllm-tok",  "vllm-tok"),
        ("groq",              "GROQ_API_KEY",      "openai",   "GROQ_API_KEY",       "gsk-x",     "gsk-x"),
        ("together",          "TOGETHER_API_KEY",  "openai",   "TOGETHER_API_KEY",   "tog-x",     "tog-x"),
    ])
    def test_provider_scenario(self, scenario, api_key_env, provider, env_var, env_value, expected):
        env_patch = {env_var: env_value} if env_var and env_value else {}
        with patch.dict(os.environ, env_patch):
            key = _resolve_api_key(api_key_env, provider)
        assert key == expected, f"Scenario '{scenario}' failed"
