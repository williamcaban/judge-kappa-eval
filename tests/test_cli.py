"""
Tests for the CLI entry point.

Tests validate config loading, schema output, and error handling
without running actual evaluations (which require LLM calls).
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from jury_eval.cli.config_schema import DatasetModeConfig, SkillModeConfig


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
        with patch("sys.argv", ["jury-eval", "validate", str(config_path)]):
            with patch("sys.stdout", captured):
                from jury_eval.cli.main import app
                app()
        assert "valid" in captured.getvalue()

    def test_validate_invalid_config_exits(self, tmp_path):
        bad_config = {"mode": "skill"}  # missing required fields
        config_path = tmp_path / "bad.yaml"
        config_path.write_text(yaml.dump(bad_config))
        with patch("sys.argv", ["jury-eval", "validate", str(config_path)]):
            with pytest.raises(SystemExit) as exc:
                from jury_eval.cli.main import app
                app()
        assert exc.value.code != 0


class TestCLISchemaCommand:
    def test_schema_outputs_valid_json(self, capsys):
        with patch("sys.argv", ["jury-eval", "schema"]):
            from jury_eval.cli.main import app
            app()
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "skill_mode" in parsed
        assert "dataset_mode" in parsed

    def test_skill_schema_has_required_fields(self, capsys):
        with patch("sys.argv", ["jury-eval", "schema"]):
            from jury_eval.cli.main import app
            app()
        schema = json.loads(capsys.readouterr().out)
        skill_props = schema["skill_mode"].get("properties", {})
        assert "mode" in skill_props
        assert "skill_dir" in skill_props
        assert "panel" in skill_props


class TestCLIHelpAndErrors:
    def test_no_args_shows_help(self, capsys):
        with patch("sys.argv", ["jury-eval"]):
            from jury_eval.cli.main import app
            app()
        out = capsys.readouterr().out
        assert "Usage" in out

    def test_unknown_command_exits(self):
        with patch("sys.argv", ["jury-eval", "unknown-cmd"]):
            with pytest.raises(SystemExit) as exc:
                from jury_eval.cli.main import app
                app()
        assert exc.value.code != 0

    def test_run_missing_config_arg_exits(self):
        with patch("sys.argv", ["jury-eval", "run"]):
            with pytest.raises(SystemExit) as exc:
                from jury_eval.cli.main import app
                app()
        assert exc.value.code != 0

    def test_invalid_format_exits(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(MINIMAL_SKILL_CONFIG))
        with patch("sys.argv", ["jury-eval", "run", str(config_path), "--format", "xml"]):
            with pytest.raises(SystemExit) as exc:
                from jury_eval.cli.main import app
                app()
        assert exc.value.code != 0
