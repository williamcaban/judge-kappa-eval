"""Tests for SkillAdapter and DatasetAdapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jury_eval.adapters.dataset import DatasetAdapter
from jury_eval.adapters.skill import SkillAdapter


# ── SkillAdapter ──────────────────────────────────────────────────────────────


def _write_skill(tmp_path: Path, evals: list[dict], skill_body: str = "# Instructions\nDo X.") -> Path:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(f"---\nname: test-skill\n---\n{skill_body}")
    evals_dir = skill_dir / "evals"
    evals_dir.mkdir()
    (evals_dir / "evals.json").write_text(json.dumps({"evals": evals}))
    return skill_dir


class TestSkillAdapter:
    def test_returns_cases_and_skill_markdown(self, tmp_path):
        skill_dir = _write_skill(tmp_path, [
            {"id": "e1", "prompt": "Do something.", "assertions": ["Output contains X."]}
        ])
        cases, skill_md = SkillAdapter().load_with_context(skill_dir)
        assert len(cases) == 1
        assert "Instructions" in skill_md

    def test_case_id_and_prompt_parsed(self, tmp_path):
        skill_dir = _write_skill(tmp_path, [
            {"id": "eval-abc", "prompt": "Summarize this.", "assertions": []}
        ])
        cases, _ = SkillAdapter().load_with_context(skill_dir)
        assert cases[0].id == "eval-abc"
        assert cases[0].prompt == "Summarize this."

    def test_assertions_parsed_as_list(self, tmp_path):
        skill_dir = _write_skill(tmp_path, [
            {"id": "e1", "prompt": "P", "assertions": ["Assert A.", "Assert B."]}
        ])
        cases, _ = SkillAdapter().load_with_context(skill_dir)
        assert len(cases[0].assertions) == 2
        assert cases[0].assertions[0].text == "Assert A."
        assert cases[0].assertions[0].weight == 1.0  # default

    def test_optional_expected_output(self, tmp_path):
        skill_dir = _write_skill(tmp_path, [
            {"id": "e1", "prompt": "P", "assertions": [], "expected_output": "The answer is 42."}
        ])
        cases, _ = SkillAdapter().load_with_context(skill_dir)
        assert cases[0].expected_output == "The answer is 42."

    def test_missing_expected_output_is_none(self, tmp_path):
        skill_dir = _write_skill(tmp_path, [
            {"id": "e1", "prompt": "P", "assertions": []}
        ])
        cases, _ = SkillAdapter().load_with_context(skill_dir)
        assert cases[0].expected_output is None

    def test_multiple_evals_all_loaded(self, tmp_path):
        skill_dir = _write_skill(tmp_path, [
            {"id": f"e{i}", "prompt": f"P{i}", "assertions": []} for i in range(5)
        ])
        cases, _ = SkillAdapter().load_with_context(skill_dir)
        assert len(cases) == 5

    def test_load_method_returns_cases_only(self, tmp_path):
        skill_dir = _write_skill(tmp_path, [
            {"id": "e1", "prompt": "P", "assertions": []}
        ])
        # load() is the InputAdapter ABC method — returns just cases
        cases = SkillAdapter().load(skill_dir)
        assert isinstance(cases, list)


# ── DatasetAdapter ────────────────────────────────────────────────────────────


class TestDatasetAdapter:
    def test_question_key_becomes_prompt(self):
        data = [{"inputs": {"question": "What is X?"}, "expectations": {"answer": "X is Y."}}]
        cases = DatasetAdapter().load(data)
        assert cases[0].prompt == "What is X?"

    def test_prompt_key_as_fallback(self):
        data = [{"inputs": {"prompt": "Summarize this."}, "expectations": {}}]
        cases = DatasetAdapter().load(data)
        assert cases[0].prompt == "Summarize this."

    def test_json_dump_fallback_for_unknown_keys(self):
        data = [{"inputs": {"custom_field": "value"}, "expectations": {}}]
        cases = DatasetAdapter().load(data)
        assert "custom_field" in cases[0].prompt  # falls back to json.dumps(inputs)

    def test_expected_output_from_answer(self):
        data = [{"inputs": {"question": "Q"}, "expectations": {"answer": "A"}}]
        cases = DatasetAdapter().load(data)
        assert cases[0].expected_output == "A"

    def test_expected_output_from_expected_answer(self):
        data = [{"inputs": {"question": "Q"}, "expectations": {"expected_answer": "Expected A"}}]
        cases = DatasetAdapter().load(data)
        assert cases[0].expected_output == "Expected A"

    def test_auto_id_assigned_when_missing(self):
        data = [{"inputs": {"question": "Q"}, "expectations": {}}]
        cases = DatasetAdapter().load(data)
        assert cases[0].id == "case-0000"

    def test_explicit_id_preserved(self):
        data = [{"id": "my-case-1", "inputs": {"question": "Q"}, "expectations": {}}]
        cases = DatasetAdapter().load(data)
        assert cases[0].id == "my-case-1"

    def test_multiple_rows(self):
        data = [{"inputs": {"question": f"Q{i}"}, "expectations": {}} for i in range(10)]
        cases = DatasetAdapter().load(data)
        assert len(cases) == 10
        assert cases[4].id == "case-0004"
