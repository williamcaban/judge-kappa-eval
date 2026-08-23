"""
Declarative YAML config schema for the jury-eval CLI.

All CLI runs are driven by a config file so evaluations are reproducible.
API keys are NEVER stored in config files — they are always read from
environment variables at runtime.

── Key resolution rules (BackendConfig.api_key_env) ──────────────────────────

  api_key_env absent / null
    → use the provider default env var:
        provider: anthropic  →  ANTHROPIC_API_KEY
        provider: openai     →  OPENAI_API_KEY

  api_key_env: "MY_VAR"
    → read the key from the environment variable MY_VAR at runtime.
      Use this for OpenRouter, Groq, Together, or any provider whose key
      is stored under a name other than OPENAI_API_KEY.

  api_key_env: ""   (empty string)
    → no authentication required. Passes the placeholder string "no-key"
      to the OpenAI client so it does not reject a missing OPENAI_API_KEY.
      Use for: Ollama, vLLM without auth, any local endpoint without tokens.

── Provider quick reference ───────────────────────────────────────────────────

  Anthropic (hosted)    provider: anthropic   api_key_env: (omit)
  OpenAI (hosted)       provider: openai      api_key_env: (omit)
  OpenRouter            provider: openai      api_key_env: "OPENROUTER_API_KEY"
                        base_url: "https://openrouter.ai/api/v1"
  vLLM (no auth)        provider: openai      api_key_env: ""
                        base_url: "http://localhost:8000/v1"
  vLLM (with token)     provider: openai      api_key_env: "VLLM_API_KEY"
                        base_url: "http://localhost:8000/v1"
  Ollama                provider: openai      api_key_env: ""
                        base_url: "http://localhost:11434/v1"
  Groq                  provider: openai      api_key_env: "GROQ_API_KEY"
                        base_url: "https://api.groq.com/openai/v1"
  Together              provider: openai      api_key_env: "TOGETHER_API_KEY"
                        base_url: "https://api.together.xyz/v1"

See examples/config_mixed_panel.yaml for a panel that uses all four simultaneously.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class BackendConfig(BaseModel):
    provider: Literal["openai", "anthropic"]
    model: str
    base_url: str | None = None        # override for vLLM, Ollama, OpenRouter, etc.
    api_key_env: str | None = None
    # None → use provider default (OPENAI_API_KEY or ANTHROPIC_API_KEY)
    # ""   → no auth; passes placeholder "no-key" (Ollama, vLLM without token)
    # "MY_VAR" → reads key from environment variable MY_VAR at runtime


class CalibrationConfig(BaseModel):
    prompt: str
    output: str
    score: float
    rationale: str


class AnchorConfig(BaseModel):
    """Low / mid / high description strings for a rubric dimension anchor."""
    low: str = ""
    mid: str = ""
    high: str = ""

    def to_list(self) -> list[str]:
        return [s for s in [self.low, self.mid, self.high] if s]


class RubricDimensionConfig(BaseModel):
    name: str
    description: str
    weight: float = 1.0
    anchors: AnchorConfig | None = None


class JudgeConfig(BaseModel):
    id: str
    type: Literal["assertion", "rubric"]
    backend: BackendConfig
    weight: float = 1.0                   # used by JudgeJury; ignored by JudgePanel
    temperature: float = 0.0
    calibration: list[CalibrationConfig] = Field(default_factory=list)
    rubric: list[RubricDimensionConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def rubric_required_for_rubric_judge(self) -> JudgeConfig:
        if self.type == "rubric" and not self.rubric:
            raise ValueError(f"Judge '{self.id}' has type='rubric' but no rubric dimensions defined.")
        return self


class PanelConfig(BaseModel):
    type: Literal["panel", "jury"] = "panel"
    strategy: Literal["mean", "weighted_mean", "majority_vote", "median", "trimmed_mean"] = "mean"
    judges: list[JudgeConfig]


class VariantConfig(BaseModel):
    name: str
    system_prompt: str | None = None
    skill_context: str | None = None  # auto-loaded from SKILL.md when mode=skill
    model: str | None = None          # if None, inherits generation.model
    temperature: float = 0.0
    generation: GenerationConfig | None = None
    # Per-variant generation backend. When set, this variant uses its own
    # backend instead of the top-level generation backend. Use for endpoint
    # comparison (config_endpoints.yaml): each variant calls a different model.


class GenerationConfig(BaseModel):
    """Backend used to generate variant outputs (separate from judge backends)."""
    backend: BackendConfig


class PositionalJudgeConfig(BaseModel):
    id: str
    backend: BackendConfig
    temperature: float = 0.0
    calibration: list[CalibrationConfig] = Field(default_factory=list)


class OutputConfig(BaseModel):
    format: Literal["json", "text", "jsonl"] = "text"
    file: str | None = None           # if None, writes to stdout
    include_verdicts: bool = False        # include per-judge per-case verdict detail


# ── Top-level config ──────────────────────────────────────────────────────────


class SkillModeConfig(BaseModel):
    mode: Literal["skill"]
    skill_dir: str
    generation: GenerationConfig
    panel: PanelConfig
    control: VariantConfig
    treatment: VariantConfig
    scale_type: Literal["nominal", "ordinal", "interval", "ratio"] = "ordinal"
    positional_judge: PositionalJudgeConfig | None = None
    verbosity_bias_threshold: float = 0.30
    output: OutputConfig = Field(default_factory=OutputConfig)


class DatasetModeConfig(BaseModel):
    mode: Literal["dataset"]
    dataset_file: str                    # path to JSONL file
    generation: GenerationConfig
    panel: PanelConfig
    control: VariantConfig
    treatment: VariantConfig
    rubric: list[RubricDimensionConfig] = Field(default_factory=list)
    scale_type: Literal["nominal", "ordinal", "interval", "ratio"] = "ordinal"
    positional_judge: PositionalJudgeConfig | None = None
    verbosity_bias_threshold: float = 0.30
    output: OutputConfig = Field(default_factory=OutputConfig)


JuryEvalConfig = SkillModeConfig | DatasetModeConfig
