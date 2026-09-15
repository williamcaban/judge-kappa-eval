"""
Tests for Cohen's κ and Krippendorff's α.

These modules are pure computation — no LLM backend needed.
Tests verify key statistical invariants:
  - Perfect agreement   → κ≈1 and α≈1
  - Anti-correlated     → κ<0 and α<0
  - Missing data (NaN)  → α handles gracefully
  - Expected P(chance)  → always in [0, 1]
  - α interpretation    → correct threshold labels
"""

from __future__ import annotations

import pytest

from judge_kappa.agreement.alpha import KrippendorffAlpha, _interpret
from judge_kappa.agreement.kappa import CohenKappa
from judge_kappa.models import ScaleType
from tests.conftest import make_verdict


def _verdicts(scores_by_judge: dict[str, dict[str, float]]):
    """Build verdicts from {judge_id: {case_id: score}} dict."""
    result = []
    for judge_id, cases in scores_by_judge.items():
        for case_id, score in cases.items():
            result.append(make_verdict(judge_id, case_id, score))
    return result


# Canonical Krippendorff's alpha worked example: 3 coders (A, B, C) rating
# 15 units on a 1-4 scale with missing data. Reproduced verbatim (unit
# ordering u1..u15) from https://en.wikipedia.org/wiki/Krippendorff%27s_alpha
# and from the `fast-krippendorff` package's own sample.py, both of which
# attribute it to Krippendorff (2011). `None` = coder did not rate that unit.
_KRIPPENDORFF_REFERENCE_DATA: dict[str, dict[str, float]] = {
    judge: {f"u{i + 1}": score for i, score in enumerate(scores) if score is not None}
    for judge, scores in {
        "A": [None, None, None, None, None, 3, 4, 1, 2, 1, 1, 3, 3, None, 3],
        "B": [1, None, 2, 1, 3, 3, 4, 3, None, None, None, None, None, None, None],
        "C": [None, None, 2, 1, 3, 4, 4, None, 2, 1, 1, 3, 3, None, 4],
    }.items()
}


# ── Krippendorff's α ─────────────────────────────────────────────────────────


class TestKrippendorffAlpha:
    def test_perfect_agreement_approaches_one(self):
        v = _verdicts({
            "j1": {"c1": 0.9, "c2": 0.1, "c3": 0.5},
            "j2": {"c1": 0.9, "c2": 0.1, "c3": 0.5},
        })
        result = KrippendorffAlpha().compute(v, ScaleType.ORDINAL)
        assert result.alpha == pytest.approx(1.0, abs=0.05)

    def test_random_agreement_is_near_zero_or_negative(self):
        v = _verdicts({
            "j1": {"c1": 1.0, "c2": 0.0, "c3": 1.0, "c4": 0.0},
            "j2": {"c1": 0.0, "c2": 1.0, "c3": 0.0, "c4": 1.0},
        })
        result = KrippendorffAlpha().compute(v, ScaleType.ORDINAL)
        assert result.alpha is not None and result.alpha < 0.2

    def test_missing_data_does_not_raise(self):
        # j2 skips c3 — handled via NaN
        v = _verdicts({
            "j1": {"c1": 0.8, "c2": 0.6, "c3": 0.4},
            "j2": {"c1": 0.7, "c2": 0.5},           # c3 absent
        })
        result = KrippendorffAlpha().compute(v, ScaleType.ORDINAL)
        assert result.alpha is not None
        assert result.n_judges == 2
        assert result.n_cases == 3

    def test_interval_scale_accepted(self):
        v = _verdicts({
            "j1": {"c1": 0.9, "c2": 0.2},
            "j2": {"c1": 0.85, "c2": 0.25},
        })
        result = KrippendorffAlpha().compute(v, ScaleType.INTERVAL)
        assert result.alpha is not None

    def test_alpha_interpretation_strong(self):
        assert "strong" in _interpret(0.85)

    def test_alpha_interpretation_tentative(self):
        assert "tentative" in _interpret(0.70)

    def test_alpha_interpretation_unreliable(self):
        assert "unreliable" in _interpret(0.50)

    def test_alpha_interpretation_boundary_at_0_80(self):
        # Exactly 0.80 is "strong"
        assert "strong" in _interpret(0.80)

    def test_three_judges_computed(self):
        v = _verdicts({
            "j1": {"c1": 0.8, "c2": 0.6},
            "j2": {"c1": 0.75, "c2": 0.65},
            "j3": {"c1": 0.82, "c2": 0.58},
        })
        result = KrippendorffAlpha().compute(v, ScaleType.INTERVAL)
        assert result.n_judges == 3
        assert result.n_cases == 2

    # ── Reference-value validation ──────────────────────────────────────────
    #
    # The tests above only check invariants (perfect agreement -> ~1, random
    # -> ~0, bounds). They would still pass if the underlying disagreement
    # computation were off by a constant or a sign. This test instead feeds
    # the canonical Krippendorff worked example (3 coders, 15 units) through
    # the full JudgeVerdict -> KrippendorffAlpha pipeline and checks the
    # output against the published alpha values, catching bugs in
    # `_build_matrix` (data marshaling) as well as the wrapped `krippendorff`
    # package call itself.
    #
    # Source: Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability",
    # reproduced at https://en.wikipedia.org/wiki/Krippendorff%27s_alpha
    # (article states alpha=0.691 nominal, alpha=0.811 interval for this
    # exact dataset; ordinal is not stated there and is instead cross-checked
    # directly against the `krippendorff` PyPI package on the same matrix).

    def test_matches_canonical_worked_example_nominal(self):
        v = _verdicts(_KRIPPENDORFF_REFERENCE_DATA)
        result = KrippendorffAlpha(bootstrap_ci=False).compute(v, ScaleType.NOMINAL)
        assert result.alpha == pytest.approx(0.691, abs=0.001)

    def test_matches_canonical_worked_example_interval(self):
        v = _verdicts(_KRIPPENDORFF_REFERENCE_DATA)
        result = KrippendorffAlpha(bootstrap_ci=False).compute(v, ScaleType.INTERVAL)
        assert result.alpha == pytest.approx(0.811, abs=0.001)

    def test_matches_canonical_worked_example_ordinal(self):
        # No published value in the Wikipedia article for ordinal on this
        # dataset; cross-checked against `krippendorff.alpha()` called
        # directly on the same matrix (verifies data marshaling only).
        v = _verdicts(_KRIPPENDORFF_REFERENCE_DATA)
        result = KrippendorffAlpha(bootstrap_ci=False).compute(v, ScaleType.ORDINAL)
        assert result.alpha == pytest.approx(0.8067, abs=0.0001)


# ── Cohen's κ ─────────────────────────────────────────────────────────────────


class TestCohenKappa:
    def test_perfect_agreement_approaches_one(self):
        v = _verdicts({
            "j1": {"c1": 1.0, "c2": 0.0, "c3": 0.5},
            "j2": {"c1": 1.0, "c2": 0.0, "c3": 0.5},
        })
        result = CohenKappa().compute(v, ScaleType.ORDINAL)
        assert result.kappa == pytest.approx(1.0, abs=0.1)

    def test_opposite_ratings_produce_negative_kappa(self):
        v = _verdicts({
            "j1": {"c1": 1.0, "c2": 0.0, "c3": 1.0, "c4": 0.0},
            "j2": {"c1": 0.0, "c2": 1.0, "c3": 0.0, "c4": 1.0},
        })
        result = CohenKappa().compute(v, ScaleType.ORDINAL)
        assert result.kappa is not None and result.kappa < 0.0

    def test_expected_chance_agreement_in_valid_range(self):
        v = _verdicts({
            "j1": {"c1": 0.8, "c2": 0.6, "c3": 0.7},
            "j2": {"c1": 0.7, "c2": 0.5, "c3": 0.8},
        })
        result = CohenKappa().compute(v, ScaleType.ORDINAL)
        assert result.expected_chance_agreement is not None
        assert 0.0 <= result.expected_chance_agreement <= 1.0

    def test_pairwise_kappa_all_pairs_computed(self):
        # 3 judges → 3 pairs (j1|j2, j1|j3, j2|j3)
        v = _verdicts({
            "j1": {"c1": 0.8, "c2": 0.6, "c3": 0.7},
            "j2": {"c1": 0.7, "c2": 0.5, "c3": 0.8},
            "j3": {"c1": 0.9, "c2": 0.7, "c3": 0.6},
        })
        result = CohenKappa().compute(v, ScaleType.ORDINAL)
        assert result.kappa is not None   # mean of 3 pairwise kappas

    def test_insufficient_cases_skips_pair(self):
        # j1 and j2 share only 1 case — should be skipped gracefully
        v = _verdicts({
            "j1": {"c1": 0.8},
            "j2": {"c1": 0.7},
        })
        result = CohenKappa().compute(v, ScaleType.ORDINAL)
        # No pairs with ≥2 shared cases — kappa is None or computed from 0 pairs
        assert result.n_judges == 2

    def test_matches_hand_derived_reference_value(self):
        """
        Reference-value check, not just invariants.

        10-item, 3-category contingency table (categories 0/1/2, counts 3/3/4
        for both raters by construction, so both marginals are identical):

            rater1: 0 0 0 1 1 1 2 2 2 2
            rater2: 0 0 1 1 1 2 2 2 2 0

        Po (observed agreement) = 7/10 = 0.70
        Pe (chance agreement)   = (3/10)^2 + (3/10)^2 + (4/10)^2 = 0.34
        kappa = (Po - Pe) / (1 - Pe) = (0.70 - 0.34) / (1 - 0.34) = 0.5455

        Cross-checked directly against sklearn.metrics.cohen_kappa_score
        (the library CohenKappa wraps) on the same raw labels — this test
        additionally validates the score->category discretization in
        CohenKappa.compute() (categories mapped to bin centers 0.1/0.5/0.9
        so _discretize recovers them exactly: bins 0, 2, 4 of 5).
        """
        category_to_score = {0: 0.1, 1: 0.5, 2: 0.9}
        r1 = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
        r2 = [0, 0, 1, 1, 1, 2, 2, 2, 2, 0]

        v = []
        for i, (a, b) in enumerate(zip(r1, r2, strict=True)):
            v.append(make_verdict("j1", f"c{i}", category_to_score[a]))
            v.append(make_verdict("j2", f"c{i}", category_to_score[b]))

        result = CohenKappa().compute(v, ScaleType.NOMINAL)
        assert result.kappa == pytest.approx(0.5455, abs=0.0001)
        assert result.expected_chance_agreement == pytest.approx(0.34, abs=0.0001)
