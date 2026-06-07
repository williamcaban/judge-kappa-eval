"""Tests for domain models — schema stability and validation."""

import pytest
from pydantic import ValidationError

from jury_eval.models import (
    AggregationStrategy,
    Assertion,
    EvalCase,
    JudgeVerdict,
    RubricDimension,
    ScaleType,
    Variant,
)


class TestEvalCase:
    def test_requires_id_and_prompt(self):
        case = EvalCase(id="c1", prompt="What is X?")
        assert case.id == "c1"
        assert case.assertions == []

    def test_missing_id_raises(self):
        with pytest.raises(ValidationError):
            EvalCase(prompt="What is X?")  # type: ignore[call-arg]

    def test_assertions_default_empty(self):
        case = EvalCase(id="c1", prompt="Q")
        assert case.assertions == []
        assert case.files == []
        assert case.metadata == {}


class TestAssertion:
    def test_default_weight_is_one(self):
        a = Assertion(text="Output is clear.")
        assert a.weight == 1.0

    def test_custom_weight(self):
        a = Assertion(text="Critical check.", weight=3.0)
        assert a.weight == 3.0


class TestVariant:
    def test_default_model(self):
        v = Variant(name="control")
        assert v.model == "gpt-4o"
        assert v.temperature == 0.0

    def test_predict_fn_allowed(self):
        fn = lambda _: "output"
        v = Variant(name="treatment", predict_fn=fn)
        assert v.predict_fn is fn


class TestRubricDimension:
    def test_anchors_default_empty(self):
        d = RubricDimension(name="faithfulness", description="Claims grounded in context.")
        assert d.anchors == []
        assert d.weight == 1.0


class TestEnums:
    def test_scale_type_values(self):
        assert ScaleType.ORDINAL.value == "ordinal"
        assert ScaleType.INTERVAL.value == "interval"

    def test_aggregation_strategy_values(self):
        assert AggregationStrategy.MEAN.value == "mean"
        assert AggregationStrategy.WEIGHTED_MEAN.value == "weighted_mean"
        assert AggregationStrategy.MAJORITY_VOTE.value == "majority_vote"


class TestJudgeVerdict:
    def test_defaults(self):
        v = JudgeVerdict(
            judge_id="j1", eval_case_id="c1", variant_name="control",
            score=0.8, rationale="ok",
        )
        assert v.assertion_scores == {}
        assert v.dimension_scores == {}
        assert v.output_token_count == 0

    def test_score_is_stored(self):
        v = JudgeVerdict(
            judge_id="j1", eval_case_id="c1", variant_name="t",
            score=0.753, rationale="",
        )
        assert v.score == pytest.approx(0.753)
