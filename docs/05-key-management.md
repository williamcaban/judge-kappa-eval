# Key Management

API keys are **never** stored in config files. They are read from environment variables at runtime via the `api_key_env` field on each `backend` block.

---

## Three-case rule

| `api_key_env` value | Meaning | Use for |
|---|---|---|
| absent / `null` | Use provider default env var | Anthropic → `ANTHROPIC_API_KEY`; OpenAI → `OPENAI_API_KEY` |
| `"MY_VAR"` | Read key from `MY_VAR` at runtime | OpenRouter, Groq, Together, vLLM with token |
| `""` (empty string) | No auth — passes `"no-key"` placeholder | Ollama, vLLM without token |

If a named env var (`api_key_env: "MY_VAR"`) is not set at runtime, the tool raises immediately with a `export MY_VAR=<your-key>` hint — before any LLM calls are made.

---

## Provider quick reference

```yaml
# Anthropic (hosted) — api_key_env absent
backend:
  provider: anthropic
  model: claude-sonnet-4-6
# reads ANTHROPIC_API_KEY automatically

# OpenAI (hosted) — api_key_env absent
backend:
  provider: openai
  model: gpt-4o
# reads OPENAI_API_KEY automatically

# OpenRouter
backend:
  provider: openai
  model: openai/gpt-4o          # OpenRouter format: "provider/model"
  base_url: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY

# vLLM (no auth)
backend:
  provider: openai
  model: meta-llama/Meta-Llama-3.1-70B-Instruct
  base_url: http://localhost:8000/v1
  api_key_env: ""               # empty string → no auth

# vLLM (with token)
backend:
  provider: openai
  model: meta-llama/Meta-Llama-3.1-70B-Instruct
  base_url: http://localhost:8000/v1
  api_key_env: VLLM_API_KEY

# Ollama (no auth)
backend:
  provider: openai
  model: mistral                # name from `ollama list`
  base_url: http://localhost:11434/v1
  api_key_env: ""               # empty string → no auth

# Groq
backend:
  provider: openai
  model: llama-3.1-70b-versatile
  base_url: https://api.groq.com/openai/v1
  api_key_env: GROQ_API_KEY

# Together AI
backend:
  provider: openai
  model: meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
  base_url: https://api.together.xyz/v1
  api_key_env: TOGETHER_API_KEY
```

---

## Mixed-provider panel example

`examples/config_mixed_panel.yaml` runs four judges simultaneously from four different providers. The `api_key_env` field is set independently per judge:

```yaml
panel:
  type: panel
  judges:
    - id: judge-anthropic
      backend:
        provider: anthropic
        model: claude-sonnet-4-6
        # api_key_env absent → reads ANTHROPIC_API_KEY

    - id: judge-openrouter
      backend:
        provider: openai
        model: openai/gpt-4o
        base_url: https://openrouter.ai/api/v1
        api_key_env: OPENROUTER_API_KEY

    - id: judge-vllm
      backend:
        provider: openai
        model: meta-llama/Meta-Llama-3.1-70B-Instruct
        base_url: http://localhost:8000/v1
        api_key_env: ""               # no auth

    - id: judge-ollama
      backend:
        provider: openai
        model: mistral
        base_url: http://localhost:11434/v1
        api_key_env: ""               # no auth
```

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY="sk-or-..."
# No keys needed for vLLM and Ollama

jury-eval run examples/config_mixed_panel.yaml
```

This works identically for `panel`, `jury`, and the `positional_judge` field — each backend resolves its own key.

---

## Python API — pass keys directly

In the Python API, pass the key directly to the backend constructor. No `api_key_env` indirection needed:

```python
import os
from jury_eval import OpenAIBackend, AnthropicBackend, AssertionJudge, JudgePanel

panel = JudgePanel(judges=[
    # Anthropic — picks up ANTHROPIC_API_KEY automatically
    AssertionJudge("anthropic", AnthropicBackend("claude-sonnet-4-6")),

    # OpenRouter — explicit key from env
    AssertionJudge("openrouter", OpenAIBackend(
        "openai/gpt-4o",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )),

    # vLLM — no auth; pass any non-None placeholder
    AssertionJudge("vllm", OpenAIBackend(
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        base_url="http://localhost:8000/v1",
        api_key="no-key",
    )),

    # Ollama — no auth
    AssertionJudge("ollama", OpenAIBackend(
        "mistral",
        base_url="http://localhost:11434/v1",
        api_key="no-key",
    )),
])
```

---

## Security notes

- Never commit `.env` files or files containing API keys
- The `.gitignore` already excludes `.env` and `.env.*`
- `api_key_env` stores only the variable **name**, never the value
- All key resolution happens at runtime, not at config parse time
- The error message on a missing env var is: `export MY_VAR=<your-key>` — safe to share in logs
