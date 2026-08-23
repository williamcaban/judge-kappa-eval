"""
Builds judge_kappa runtime objects (backends, judges, panel, evaluator)
from a parsed config. Keeps the CLI thin and the config schema testable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from judge_kappa.cli.config_schema import (
    BackendConfig,
    CalibrationConfig,
    DatasetModeConfig,
    JudgeConfig,
    PanelConfig,
    PositionalJudgeConfig,
    RubricDimensionConfig,
    SkillModeConfig,
    VariantConfig,
)
from judge_kappa.evaluator import JuryEvaluator
from judge_kappa.judges.assertion import AssertionJudge
from judge_kappa.judges.base import LLMJudge
from judge_kappa.judges.pairwise import PairwiseJudge
from judge_kappa.judges.rubric import RubricJudge
from judge_kappa.llm.anthropic_backend import AnthropicBackend
from judge_kappa.llm.base import LLMBackend
from judge_kappa.llm.openai_backend import OpenAIBackend
from judge_kappa.models import (
    AggregationStrategy,
    CalibrationExample,
    RubricDimension,
    ScaleType,
    Variant,
)
from judge_kappa.panel.jury import JudgeJury
from judge_kappa.panel.panel import JudgePanel

_PROVIDER_DEFAULT_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
}


def _resolve_api_key(api_key_env: str | None, provider: str) -> str | None:
    """
    Resolve an API key from the environment according to the three-case rule:

      api_key_env is None  → read from provider default env var; return None
                              if not set (let the SDK handle it or raise its
                              own error with a clear message)
      api_key_env == ""    → no auth required; return placeholder "no-key" so
                              the OpenAI client does not reject a missing key
      api_key_env == "VAR" → read from VAR; raise immediately if not set so
                              the user gets a clear error before any LLM calls
    """
    if api_key_env is None:
        default_var = _PROVIDER_DEFAULT_ENV.get(provider)
        return os.environ.get(default_var) if default_var else None

    if api_key_env == "":
        return "no-key"  # placeholder for Ollama, vLLM without auth

    key = os.environ.get(api_key_env)
    if key is None:
        raise ValueError(
            f"api_key_env is set to '{api_key_env}' but that environment variable is not set. "
            f"Run: export {api_key_env}=<your-key>"
        )
    return key


def _build_backend(cfg: BackendConfig) -> LLMBackend:
    api_key = _resolve_api_key(cfg.api_key_env, cfg.provider)
    if cfg.provider == "anthropic":
        return AnthropicBackend(model=cfg.model, api_key=api_key)
    return OpenAIBackend(model=cfg.model, base_url=cfg.base_url, api_key=api_key)


def _build_calibration(items: list[CalibrationConfig]) -> list[CalibrationExample]:
    return [CalibrationExample(**c.model_dump()) for c in items]


def _build_rubric(dims: list[RubricDimensionConfig]) -> list[RubricDimension]:
    result = []
    for d in dims:
        anchors = d.anchors.to_list() if d.anchors else []
        result.append(RubricDimension(
            name=d.name, description=d.description, weight=d.weight, anchors=anchors,
        ))
    return result


def _build_judge(cfg: JudgeConfig) -> LLMJudge:
    backend = _build_backend(cfg.backend)
    cal = _build_calibration(cfg.calibration)
    if cfg.type == "rubric":
        return RubricJudge(
            judge_id=cfg.id, backend=backend,
            calibration_examples=cal, temperature=cfg.temperature,
            rubric=_build_rubric(cfg.rubric),
        )
    return AssertionJudge(
        judge_id=cfg.id, backend=backend,
        calibration_examples=cal, temperature=cfg.temperature,
    )


def _build_panel(cfg: PanelConfig) -> JudgeJury | JudgePanel:
    judges = [_build_judge(j) for j in cfg.judges]
    strategy = AggregationStrategy(cfg.strategy)
    if cfg.type == "jury":
        weights = [j.weight for j in cfg.judges]
        return JudgeJury(jurors=list(zip(judges, weights, strict=True)), strategy=strategy)
    weights = [j.weight for j in cfg.judges]
    return JudgePanel(judges=judges, strategy=strategy, weights=weights)


def _build_positional_judge(cfg: PositionalJudgeConfig) -> PairwiseJudge:
    backend = _build_backend(cfg.backend)
    cal = _build_calibration(cfg.calibration)
    return PairwiseJudge(
        judge_id=cfg.id, backend=backend,
        calibration_examples=cal, temperature=cfg.temperature,
    )


def _build_variant(cfg: VariantConfig, skill_md: str | None = None) -> Variant:
    per_variant_backend = (
        _build_backend(cfg.generation.backend) if cfg.generation else None
    )
    return Variant(
        name=cfg.name,
        system_prompt=cfg.system_prompt,
        skill_context=cfg.skill_context or (skill_md if cfg.name == "treatment" else None),
        model=cfg.model or "gpt-4o",
        temperature=cfg.temperature,
        generation_backend=per_variant_backend,
    )


def build_evaluator(config: SkillModeConfig | DatasetModeConfig) -> JuryEvaluator:
    panel = _build_panel(config.panel)
    gen_backend = _build_backend(config.generation.backend)
    pairwise = (
        _build_positional_judge(config.positional_judge)
        if config.positional_judge else None
    )
    return JuryEvaluator(
        panel=panel,
        generation_backend=gen_backend,
        scale_type=ScaleType(config.scale_type),
        positional_judge=pairwise,
        verbosity_bias_threshold=config.verbosity_bias_threshold,
    )


def load_dataset(path: str) -> list[dict[str, object]]:
    """Load JSONL (one JSON object per line) or JSON array."""
    text = Path(path).read_text()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]
