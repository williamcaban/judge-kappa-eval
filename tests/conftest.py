"""
Shared fixtures and mock infrastructure.

MockLLMBackend is a deterministic stub that never makes real API calls.
It detects which judge type is calling it from the system prompt and
returns appropriately shaped JSON.  Individual tests can inject specific
responses via the `responses` constructor argument.
"""

from __future__ import annotations

import json

import pytest

from judge_kappa.models import (
    Assertion,
    CalibrationExample,
    EvalCase,
    JudgeVerdict,
    RubricDimension,
)


class MockLLMBackend:
    """
    Deterministic mock backend. No network calls, no API keys required.

    Every call returns the next item in `responses` (cycling if exhausted).
    If `responses` is empty, returns a safe empty-JSON fallback that causes
    all judges to produce a score of 0.0 without raising.

    Tests that care about score values MUST inject explicit `responses`.
    Tests that only check prompt structure can use MockLLMBackend() with no args.
    """

    _FALLBACK = json.dumps({"rationale": "mock"})

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = responses or []
        self.calls: list[dict[str, str]] = []
        self._call_count = 0

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:  # noqa: ARG002
        self.calls.append({"system": system, "user": user})
        if self._responses:
            r = self._responses[self._call_count % len(self._responses)]
            self._call_count += 1
            return r
        self._call_count += 1
        return self._FALLBACK


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_backend() -> MockLLMBackend:
    return MockLLMBackend()


@pytest.fixture
def sample_assertions() -> list[Assertion]:
    return [
        Assertion(text="Output identifies the root cause.", weight=2.0),
        Assertion(text="Output provides a concrete recommendation.", weight=1.0),
    ]


@pytest.fixture
def sample_rubric() -> list[RubricDimension]:
    return [
        RubricDimension(name="faithfulness", description="Claims are grounded in context.", weight=2.0),
        RubricDimension(name="relevance",    description="Output addresses the question.", weight=1.0),
    ]


@pytest.fixture
def sample_cases(sample_assertions: list[Assertion]) -> list[EvalCase]:
    return [
        EvalCase(
            id="case-001", prompt="What is X?",
            assertions=sample_assertions, expected_output="X is Y.",
        ),
        EvalCase(
            id="case-002", prompt="Explain Z.",
            assertions=sample_assertions,
        ),
        EvalCase(
            id="case-003", prompt="List the risks.",
            assertions=sample_assertions,
        ),
    ]


@pytest.fixture
def calibration() -> list[CalibrationExample]:
    return [
        CalibrationExample(prompt="Q", output="A good answer.", score=0.9, rationale="Complete."),
        CalibrationExample(prompt="Q", output="Nope.",          score=0.1, rationale="Incomplete."),
    ]


def make_verdict(
    judge_id: str,
    case_id: str,
    score: float,
    variant: str = "treatment",
    token_count: int = 50,
) -> JudgeVerdict:
    """Helper — build a verdict without a real judge."""
    return JudgeVerdict(
        judge_id=judge_id,
        eval_case_id=case_id,
        variant_name=variant,
        score=score,
        rationale="test",
        output_token_count=token_count,
    )
