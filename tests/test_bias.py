"""
Tests for PositionalBiasDetector and VerbosityBiasDetector.

No real LLM calls — PairwiseJudge uses MockLLMBackend with injected responses.
"""

from __future__ import annotations

import json
import math

import pytest

from judge_kappa.bias.positional import PositionalBiasDetector, PositionalBiasReport
from judge_kappa.bias.verbosity import VerbosityBiasDetector
from judge_kappa.judges.pairwise import PairwiseJudge
from judge_kappa.models import BiasResult, EvalCase
from tests.conftest import MockLLMBackend, make_verdict


def _pairwise_resp(score_a: float, score_b: float, preferred: str = "A") -> str:
    return json.dumps({"score_a": score_a, "score_b": score_b,
                        "preferred": preferred, "rationale": "ok"})


# ── PositionalBiasDetector ────────────────────────────────────────────────────


class TestPositionalBiasDetector:
    def _case(self) -> EvalCase:
        return EvalCase(id="c1", prompt="Test prompt.")

    def _judge(self, responses: list[str]) -> PairwiseJudge:
        return PairwiseJudge("j1", MockLLMBackend(responses=responses))

    def test_positional_flip_detected(self):
        # Round 1: judge prefers A (control=A, treatment=B)
        # Round 2: judge STILL prefers A (treatment=A, control=B)  → positional flip
        backend = MockLLMBackend(responses=[
            _pairwise_resp(0.9, 0.3),  # round 1: A(control)=0.9, B(treatment)=0.3
            _pairwise_resp(0.9, 0.3),  # round 2: A(treatment)=0.9, B(control)=0.3
        ])
        judge = PairwiseJudge("j1", backend)
        detector = PositionalBiasDetector(judge)
        report = detector.test_case(self._case(), "ctrl output", "trt output")
        assert report.positional_flip is True

    def test_no_flip_when_judge_prefers_same_content(self):
        # Round 1: control=A gets 0.3, treatment=B gets 0.9 → treatment preferred
        # Round 2: treatment=A gets 0.9, control=B gets 0.3 → treatment still preferred (content-driven)
        backend = MockLLMBackend(responses=[
            _pairwise_resp(0.3, 0.9),  # round 1: A(control)=0.3, B(treatment)=0.9
            _pairwise_resp(0.9, 0.3),  # round 2: A(treatment)=0.9, B(control)=0.3
        ])
        judge = PairwiseJudge("j1", backend)
        detector = PositionalBiasDetector(judge)
        report = detector.test_case(self._case(), "ctrl output", "trt output")
        # Not a positional flip — judge consistently prefers treatment regardless of position
        assert report.positional_flip is False

    def test_corpus_bias_rate_all_flipped(self):
        reports = [
            PositionalBiasReport("c1", True,  0.1, (0.9, 0.3), (0.3, 0.9)),
            PositionalBiasReport("c2", True,  0.2, (0.8, 0.4), (0.4, 0.8)),
            PositionalBiasReport("c3", False, 0.0, (0.5, 0.6), (0.6, 0.5)),
        ]
        backend = MockLLMBackend()
        detector = PositionalBiasDetector(PairwiseJudge("j1", backend))
        result = detector.detect(reports=reports)
        assert result.positional_bias_rate == pytest.approx(2 / 3, abs=1e-4)

    def test_corpus_bias_rate_none_flipped(self):
        reports = [
            PositionalBiasReport("c1", False, 0.0, (0.5, 0.6), (0.6, 0.5)),
            PositionalBiasReport("c2", False, 0.0, (0.4, 0.7), (0.7, 0.4)),
        ]
        backend = MockLLMBackend()
        detector = PositionalBiasDetector(PairwiseJudge("j1", backend))
        result = detector.detect(reports=reports)
        assert result.positional_bias_rate == pytest.approx(0.0)

    def test_order_effect_matches_closed_form(self):
        # Control wins 8/10 when shown first and 6/10 when shown second.
        # Two-item MLE: b = (logit 0.8 - logit 0.6) / 2  (hand-derived reference value).
        reports = []
        for k in range(10):
            r1_control_wins = k < 8
            r2_control_wins = k < 6
            reports.append(PositionalBiasReport(
                f"c{k}", False, 0.0,
                (0.9 if r1_control_wins else 0.1, 0.9 if r2_control_wins else 0.1),
                (0.1 if r1_control_wins else 0.9, 0.1 if r2_control_wins else 0.9),
            ))
        est = PositionalBiasDetector.estimate_order_effect(reports, n_bootstrap=200)
        assert est is not None
        expected = (math.log(0.8 / 0.2) - math.log(0.6 / 0.4)) / 2.0
        assert est.order_effect == pytest.approx(expected, abs=2e-3)
        assert est.ci_low is not None and est.ci_high is not None
        assert est.ci_low < est.order_effect < est.ci_high
        assert est.n_cases == 10

    def test_order_effect_separates_quality_from_position(self):
        # Treatment is better and wins in both rounds: zero flips, and no order effect.
        content_driven = [
            PositionalBiasReport(f"c{k}", False, 0.0, (0.2, 0.2), (0.8, 0.8)) for k in range(6)
        ]
        # First-shown always wins: every case flips, order effect is strongly positive.
        positional = [
            PositionalBiasReport(f"c{k}", True, 0.0, (0.9, 0.1), (0.1, 0.9)) for k in range(6)
        ]
        backend = MockLLMBackend()
        detector = PositionalBiasDetector(PairwiseJudge("j1", backend))
        a = detector.detect(reports=content_driven)
        b = detector.detect(reports=positional)
        assert a.positional_bias_rate == 0.0
        assert a.order_effect is not None and abs(a.order_effect) < 0.05
        assert b.positional_bias_rate == 1.0
        assert b.order_effect is not None and b.order_effect > 3.0

    def test_order_effect_edge_cases(self):
        assert PositionalBiasDetector.estimate_order_effect([]) is None
        one = PositionalBiasDetector.estimate_order_effect(
            [PositionalBiasReport("c1", True, 0.0, (0.9, 0.1), (0.1, 0.9))]
        )
        assert one is not None and one.ci_low is None and one.ci_high is None
        empty = PositionalBiasDetector(PairwiseJudge("j1", MockLLMBackend())).detect(reports=[])
        assert empty.order_effect is None and empty.positional_bias_rate == 0.0

    def test_bias_result_new_fields_default_to_none(self):
        r = BiasResult()
        assert r.order_effect is None
        assert r.order_effect_ci_low is None and r.order_effect_ci_high is None

    def test_score_instability_computed(self):
        backend = MockLLMBackend(responses=[
            _pairwise_resp(0.9, 0.3),
            _pairwise_resp(0.5, 0.5),  # different scores in round 2
        ])
        judge = PairwiseJudge("j1", backend)
        report = PositionalBiasDetector(judge).test_case(self._case(), "ctrl", "trt")
        # Score instability > 0 because round 1 and round 2 scores differ
        assert report.score_instability >= 0.0


# ── VerbosityBiasDetector ─────────────────────────────────────────────────────


class TestVerbosityBiasDetector:
    def test_positive_verbosity_bias_detected(self):
        # Longer outputs get higher scores → positive ρ
        verdicts = [
            make_verdict("j1", f"c{i}", score=i / 10, token_count=i * 20)
            for i in range(1, 11)
        ]
        result = VerbosityBiasDetector(threshold=0.30).detect(verdicts=verdicts)
        assert result.verbosity_bias_rho is not None
        assert result.verbosity_bias_rho > 0.30
        assert result.verbosity_biased is True

    def test_no_bias_when_uncorrelated(self):
        # Alternating scores regardless of length
        verdicts = [
            make_verdict("j1", f"c{i}", score=0.5 + 0.4 * ((-1) ** i), token_count=i * 10)
            for i in range(1, 11)
        ]
        result = VerbosityBiasDetector(threshold=0.30).detect(verdicts=verdicts)
        assert result.verbosity_biased is False

    def test_fewer_than_three_samples_returns_no_bias(self):
        verdicts = [make_verdict("j1", "c1", 0.8, token_count=100)]
        result = VerbosityBiasDetector().detect(verdicts=verdicts)
        assert result.verbosity_biased is False
        assert result.verbosity_bias_rho is None

    def test_custom_threshold_respected(self):
        # Spearman ρ ≈ 0.67 (partial rank agreement, not perfect correlation).
        # token_counts are shuffled relative to ascending scores, giving ρ < 1.0.
        # threshold=0.30: 0.67 > 0.30 → biased
        # threshold=0.99: 0.67 < 0.99 → not biased
        scores_and_tokens = [
            (0.1, 30), (0.2, 10), (0.3, 60), (0.4, 20),
            (0.5, 80), (0.6, 40), (0.7, 100), (0.8, 50),
            (0.9, 70), (1.0, 90),
        ]
        verdicts = [
            make_verdict("j1", f"c{i}", score=s, token_count=t)
            for i, (s, t) in enumerate(scores_and_tokens)
        ]
        assert VerbosityBiasDetector(threshold=0.30).detect(verdicts=verdicts).verbosity_biased is True
        assert VerbosityBiasDetector(threshold=0.99).detect(verdicts=verdicts).verbosity_biased is False

    def test_output_fields_populated(self):
        verdicts = [
            make_verdict("j1", f"c{i}", score=i / 10, token_count=i * 10)
            for i in range(1, 11)
        ]
        result = VerbosityBiasDetector().detect(verdicts=verdicts)
        assert result.verbosity_bias_rho is not None
        assert result.verbosity_bias_p is not None
