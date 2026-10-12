"""
Tests for the CLI entry point.

Tests validate config loading, schema output, and error handling
without running actual evaluations (which require LLM calls).
"""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import patch

import pytest
import yaml

from judge_kappa.cli.config_schema import DatasetModeConfig, SkillModeConfig

# ── Config schema validation ──────────────────────────────────────────────────

MINIMAL_SKILL_CONFIG = {
    "mode": "skill",
    "skill_dir": "./my-skill",
    "generation": {
        "backend": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
    },
    "control":   {"name": "control"},
    "treatment": {"name": "treatment"},
    "panel": {
        "type": "panel",
        "strategy": "mean",
        "judges": [
            {"id": "j1", "type": "assertion",
             "backend": {"provider": "anthropic", "model": "claude-sonnet-4-6"}},
        ],
    },
}

MINIMAL_DATASET_CONFIG = {
    "mode": "dataset",
    "dataset_file": "./evals.jsonl",
    "generation": {
        "backend": {"provider": "openai", "model": "gpt-4o"},
    },
    "control":   {"name": "control"},
    "treatment": {"name": "treatment"},
    "panel": {
        "type": "panel",
        "strategy": "mean",
        "judges": [
            {"id": "j1", "type": "assertion",
             "backend": {"provider": "openai", "model": "gpt-4o"}},
        ],
    },
}


class TestConfigSchemaValidation:
    def test_skill_config_validates(self):
        cfg = SkillModeConfig.model_validate(MINIMAL_SKILL_CONFIG)
        assert cfg.mode == "skill"
        assert cfg.scale_type == "ordinal"               # default
        assert cfg.verbosity_bias_threshold == 0.30      # default

    def test_dataset_config_validates(self):
        cfg = DatasetModeConfig.model_validate(MINIMAL_DATASET_CONFIG)
        assert cfg.mode == "dataset"
        assert cfg.dataset_file == "./evals.jsonl"

    def test_rubric_judge_requires_rubric(self):
        config = {**MINIMAL_SKILL_CONFIG, "panel": {
            "type": "panel", "strategy": "mean",
            "judges": [
                {"id": "j1", "type": "rubric",
                 "backend": {"provider": "openai", "model": "gpt-4o"},
                 "rubric": []},   # empty rubric → must raise
            ],
        }}
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SkillModeConfig.model_validate(config)

    def test_invalid_mode_raises(self):
        config = {**MINIMAL_SKILL_CONFIG, "mode": "invalid"}
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SkillModeConfig.model_validate(config)

    def test_panel_type_jury_accepted(self):
        config = {**MINIMAL_SKILL_CONFIG, "panel": {
            "type": "jury",
            "strategy": "weighted_mean",
            "judges": [
                {"id": "j1", "type": "assertion",
                 "backend": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
                 "weight": 2.0},
            ],
        }}
        cfg = SkillModeConfig.model_validate(config)
        assert cfg.panel.type == "jury"

    def test_calibration_examples_parsed(self):
        config = {**MINIMAL_SKILL_CONFIG, "panel": {
            "type": "panel", "strategy": "mean",
            "judges": [{
                "id": "j1", "type": "assertion",
                "backend": {"provider": "anthropic", "model": "claude-sonnet-4-6"},
                "calibration": [
                    {"prompt": "Q", "output": "A", "score": 0.9, "rationale": "Good."},
                ],
            }],
        }}
        cfg = SkillModeConfig.model_validate(config)
        assert len(cfg.panel.judges[0].calibration) == 1
        assert cfg.panel.judges[0].calibration[0].score == 0.9

    def test_positional_judge_optional(self):
        cfg = SkillModeConfig.model_validate(MINIMAL_SKILL_CONFIG)
        assert cfg.positional_judge is None


# ── CLI commands ──────────────────────────────────────────────────────────────


class TestCLIValidateCommand:
    def test_validate_valid_config(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(MINIMAL_SKILL_CONFIG))
        captured = StringIO()
        with patch("sys.argv", ["judge-kappa", "validate", str(config_path)]), \
                patch("sys.stdout", captured):
            from judge_kappa.cli.main import app
            app()
        assert "valid" in captured.getvalue()

    def test_validate_invalid_config_exits(self, tmp_path):
        bad_config = {"mode": "skill"}  # missing required fields
        config_path = tmp_path / "bad.yaml"
        config_path.write_text(yaml.dump(bad_config))
        with patch("sys.argv", ["judge-kappa", "validate", str(config_path)]), \
                pytest.raises(SystemExit) as exc:
            from judge_kappa.cli.main import app
            app()
        assert exc.value.code != 0


class TestCLISchemaCommand:
    def test_schema_outputs_valid_json(self, capsys):
        with patch("sys.argv", ["judge-kappa", "schema"]):
            from judge_kappa.cli.main import app
            app()
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "skill_mode" in parsed
        assert "dataset_mode" in parsed

    def test_skill_schema_has_required_fields(self, capsys):
        with patch("sys.argv", ["judge-kappa", "schema"]):
            from judge_kappa.cli.main import app
            app()
        schema = json.loads(capsys.readouterr().out)
        skill_props = schema["skill_mode"].get("properties", {})
        assert "mode" in skill_props
        assert "skill_dir" in skill_props
        assert "panel" in skill_props


class TestCLIHelpAndErrors:
    def test_no_args_shows_help(self, capsys):
        with patch("sys.argv", ["judge-kappa"]):
            from judge_kappa.cli.main import app
            app()
        out = capsys.readouterr().out
        assert "Usage" in out

    def test_unknown_command_exits(self):
        with patch("sys.argv", ["judge-kappa", "unknown-cmd"]), \
                pytest.raises(SystemExit) as exc:
            from judge_kappa.cli.main import app
            app()
        assert exc.value.code != 0

    def test_run_missing_config_arg_exits(self):
        with patch("sys.argv", ["judge-kappa", "run"]), pytest.raises(SystemExit) as exc:
            from judge_kappa.cli.main import app
            app()
        assert exc.value.code != 0

    def test_invalid_format_exits(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(MINIMAL_SKILL_CONFIG))
        with patch("sys.argv", ["judge-kappa", "run", str(config_path), "--format", "xml"]), \
                pytest.raises(SystemExit) as exc:
            from judge_kappa.cli.main import app
            app()
        assert exc.value.code != 0


# ── Backend request parameters (config → builder → backend) ───────────────────


class TestParseFailurePolicy:
    def test_default_is_flag_and_values_are_validated(self):
        from pydantic import ValidationError

        from judge_kappa.cli.config_schema import BackendConfig, JudgeConfig, PositionalJudgeConfig

        backend = BackendConfig(provider="openai", model="m")
        assert JudgeConfig(id="j", type="assertion", backend=backend).on_parse_failure == "flag"
        assert PositionalJudgeConfig(id="p", backend=backend).on_parse_failure == "flag"
        with pytest.raises(ValidationError):
            JudgeConfig(id="j", type="assertion", backend=backend, on_parse_failure="ignore")

    def test_builder_passes_the_policy_to_the_judge(self, monkeypatch):
        from judge_kappa.cli import builder
        from judge_kappa.cli.config_schema import BackendConfig, JudgeConfig, RubricDimensionConfig

        class _Backend:
            def complete(self, system, user, temperature=0.0):
                return "{}"

        monkeypatch.setattr(builder, "_build_backend", lambda cfg: _Backend())
        backend = BackendConfig(provider="openai", model="m")
        a = builder._build_judge(JudgeConfig(id="a", type="assertion", backend=backend, on_parse_failure="raise"))
        r = builder._build_judge(JudgeConfig(
            id="r", type="rubric", backend=backend, on_parse_failure="raise",
            rubric=[RubricDimensionConfig(name="d", description="d")],
        ))
        assert a.on_parse_failure == "raise" and r.on_parse_failure == "raise"

    def test_cli_reports_a_parse_failure_cleanly_instead_of_a_traceback(
        self, tmp_path, monkeypatch, capsys
    ):
        from judge_kappa.cli import main as cli_main
        from judge_kappa.judges.base import JudgeParseError

        class _Evaluator:
            def evaluate_dataset(self, **_kw):
                raise JudgeParseError("j1", "case-7", "invalid_json", "I think so")

        monkeypatch.setattr(cli_main, "build_evaluator", lambda _cfg: _Evaluator())
        monkeypatch.setattr(cli_main, "load_dataset", lambda _p: [{"id": "c"}])
        cfg = {
            "mode": "dataset", "dataset_file": "d.jsonl",
            "generation": {"backend": {"provider": "openai", "model": "m", "api_key_env": ""}},
            "panel": {"judges": [{"id": "j1", "type": "assertion",
                                  "backend": {"provider": "openai", "model": "m", "api_key_env": ""}}]},
            "control": {"name": "control"}, "treatment": {"name": "treatment"},
        }
        path = tmp_path / "c.yaml"
        path.write_text(yaml.dump(cfg))
        with patch("sys.argv", ["judge-kappa", "run", str(path)]), pytest.raises(SystemExit) as exc:
            cli_main.app()
        assert exc.value.code != 0
        err = capsys.readouterr().err
        assert "case-7" in err and "invalid_json" in err and "on_parse_failure: flag" in err


class TestBackendRequestParameters:
    def test_defaults_are_unset(self):
        from judge_kappa.cli.config_schema import BackendConfig

        cfg = BackendConfig(provider="openai", model="m")
        assert cfg.max_tokens is None and cfg.omit_temperature is False
        assert cfg.timeout_s is None and cfg.max_retries is None
        assert cfg.max_tokens_param == "max_tokens"

    @pytest.mark.parametrize(
        "bad", [{"max_tokens": 0}, {"timeout_s": 0}, {"max_retries": -1}, {"max_tokens_param": "n"}]
    )
    def test_invalid_values_rejected(self, bad):
        from pydantic import ValidationError

        from judge_kappa.cli.config_schema import BackendConfig

        with pytest.raises(ValidationError):
            BackendConfig(provider="openai", model="m", **bad)

    def test_builder_forwards_options_to_openai_backend(self, monkeypatch):
        from judge_kappa.cli import builder
        from judge_kappa.cli.config_schema import BackendConfig

        seen: dict = {}

        class _Fake:
            def __init__(self, **kw):
                seen.update(kw)

        monkeypatch.setattr(builder, "OpenAIBackend", _Fake)
        builder._build_backend(BackendConfig(
            provider="openai", model="m", api_key_env="", base_url="http://x/v1",
            max_tokens=256, max_tokens_param="max_completion_tokens",
            omit_temperature=True, timeout_s=9.0, max_retries=1,
        ))
        assert seen["max_tokens"] == 256 and seen["max_tokens_param"] == "max_completion_tokens"
        assert seen["omit_temperature"] is True
        assert seen["timeout_s"] == 9.0 and seen["max_retries"] == 1

    def test_builder_anthropic_uses_backend_default_when_unset(self, monkeypatch):
        from judge_kappa.cli import builder
        from judge_kappa.cli.config_schema import BackendConfig

        seen: dict = {}

        class _Fake:
            def __init__(self, **kw):
                seen.update(kw)

        monkeypatch.setattr(builder, "AnthropicBackend", _Fake)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        builder._build_backend(BackendConfig(provider="anthropic", model="claude-x"))
        assert "max_tokens" not in seen            # backend default (2048) applies
        builder._build_backend(BackendConfig(provider="anthropic", model="claude-x", max_tokens=900))
        assert seen["max_tokens"] == 900


# ── Text report: optional metrics ─────────────────────────────────────────────


class _ScoreBackend:
    """Judge backend that always scores the single assertion ``A`` with a fixed value."""

    def __init__(self, score: float) -> None:
        self._score = score

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:  # noqa: ARG002
        return json.dumps({"assertion_scores": {"A": self._score}, "rationale": "ok"})


def _report(n_cases: int, scores: list[float]):
    """A real EvalReport from one judge per entry of ``scores``."""
    from judge_kappa.evaluator import JuryEvaluator
    from judge_kappa.judges.assertion import AssertionJudge
    from judge_kappa.models import Assertion, EvalCase, Variant
    from judge_kappa.panel.panel import JudgePanel

    judges = [AssertionJudge(f"j{i}", _ScoreBackend(sc)) for i, sc in enumerate(scores)]
    ev = JuryEvaluator(
        panel=JudgePanel(judges), generation_backend=_ScoreBackend(1.0),
        compute_significance=False, bootstrap_ci=False,
    )
    cases = [
        EvalCase(id=f"c{i}", prompt="Q", assertions=[Assertion(text="A")]) for i in range(n_cases)
    ]
    return ev.evaluate(
        cases,
        Variant(name="control", predict_fn=lambda _: "short"),
        Variant(name="treatment", predict_fn=lambda _: "a rather longer output"),
    )


def _line(text: str, label: str) -> str:
    return next(line.strip() for line in text.splitlines() if line.strip().startswith(label))


class TestTextReportOptionalMetrics:
    """Regression: the text report formatted optional metrics with :.4f and raised TypeError."""

    @pytest.mark.parametrize(
        ("n_cases", "scores"),
        [
            (4, [0.8]),          # a single judge: no pair to compare
            (1, [0.8, 0.8]),     # two judges but no pair shares 2 cases
            (1, [0.2, 0.9]),
        ],
    )
    def test_renders_without_a_judge_pair(self, n_cases, scores):
        from judge_kappa.cli.main import _render_text

        text = _render_text(_report(n_cases, scores))
        assert "n/a (needs at least 2 judges and 2 shared cases)" in _line(text, "Cohen κ")
        assert "n/a (needs at least 2 judges and 2 shared cases)" in _line(text, "Expected P(e)")

    def test_single_judge_does_not_claim_perfect_agreement(self):
        from judge_kappa.cli.main import _render_text

        report = _report(4, [0.8])
        assert report.agreement.n_judges == 1
        assert report.agreement.alpha is None
        assert report.agreement.alpha_not_estimated_reason == "single_judge"
        text = _render_text(report)
        alpha_line = _line(text, "Krippendorff α")
        assert "n/a (needs at least 2 judges)" in alpha_line
        assert "1.0000" not in alpha_line

    def test_report_written_before_alpha_was_none_is_not_shown_as_agreement(self):
        # A report saved by 0.3.2 carries alpha = 1.0 for a single judge.
        from judge_kappa.cli.main import _render_text

        legacy = _report(4, [0.8])
        legacy.agreement = legacy.agreement.model_copy(
            update={"alpha": 1.0, "alpha_interpretation": "strong agreement (α ≥ 0.80)",
                    "alpha_not_estimated_reason": None}
        )
        alpha_line = _line(_render_text(legacy), "Krippendorff α")
        assert "n/a (needs at least 2 judges)" in alpha_line
        assert "1.0000" not in alpha_line

    def test_identical_scores_are_undefined_not_nan(self):
        from judge_kappa.cli.main import _render_text

        report = _report(4, [0.8, 0.8])
        assert report.agreement.kappa is None
        assert report.agreement.kappa_not_estimated_reason == "no_score_variation"
        text = _render_text(report)
        assert "n/a (judges used one and the same single score)" in _line(text, "Cohen κ")
        # the expected chance agreement is still a real number for identical scores
        assert _line(text, "Expected P(e)") == "Expected P(e):    1.0000"
        assert "nan" not in text.lower()

    def test_ordinary_numeric_output_is_unchanged(self):
        from judge_kappa.cli.main import _render_text

        report = _report(4, [0.2, 0.9])
        text = _render_text(report)
        assert _line(text, "Krippendorff α").startswith("Krippendorff α:   -0.7500")
        assert _line(text, "Cohen κ") == "Cohen κ (mean):   0.0000"
        assert _line(text, "Expected P(e)") == "Expected P(e):    0.0000"
        assert "n/a" not in text

