"""
Tests for judge_kappa.ranking.btl — order-aware Bradley-Terry-Luce ranking.

Reference values:
  - Two-item closed form: with cell win-rates p_first (item 0 shown first) and
    p_second (item 0 shown second), the unpenalised MLE is exactly
        s0 - s1 = (logit p_first + logit p_second) / 2
        b       = (logit p_first - logit p_second) / 2
    (hand-derived; the test uses p = 0.8 and 0.6).
  - Independent oracle: scikit-learn LogisticRegression (no penalty) on the
    same design matrix. scikit-learn is already a runtime dependency.

No LLM calls. Simulations use fixed seeds.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from judge_kappa.ranking import (
    ELO_SCALE,
    ComparisonGame,
    bootstrap_btl,
    fit_btl,
    order_effect_identifiable,
)


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _simulate(
    true_s: list[float],
    b: float,
    n_reps: int,
    seed: int,
    p_first_is_low: float = 0.5,
) -> list[ComparisonGame]:
    """One game per (rep, pair). ``p_first_is_low`` = P(lower-index item shown first)."""
    rng = np.random.default_rng(seed)
    games: list[ComparisonGame] = []
    for rep in range(n_reps):
        for i, j in itertools.combinations(range(len(true_s)), 2):
            order = 1 if rng.random() < p_first_is_low else -1
            eta = true_s[i] - true_s[j] + order * b
            win = rng.random() < 1.0 / (1.0 + math.exp(-eta))
            games.append(ComparisonGame(case=rep, i=i, j=j, order=order, outcome=1.0 if win else 0.0))
    return games


def _both_orders(i: int, j: int, wins_first: int, n_first: int, wins_second: int, n_second: int, case0: int = 0):
    games = []
    for k in range(n_first):
        games.append(ComparisonGame(case0 + k, i, j, 1, 1.0 if k < wins_first else 0.0))
    for k in range(n_second):
        games.append(ComparisonGame(case0 + n_first + k, i, j, -1, 1.0 if k < wins_second else 0.0))
    return games


class TestClosedForm:
    def test_two_item_closed_form(self):
        games = _both_orders(0, 1, wins_first=800, n_first=1000, wins_second=600, n_second=1000)
        fit = fit_btl(2, games, l2=1e-9)
        d = (_logit(0.8) + _logit(0.6)) / 2.0
        b = (_logit(0.8) - _logit(0.6)) / 2.0
        assert fit.order_identifiable
        assert fit.order_effect == pytest.approx(b, abs=1e-3)
        assert fit.scores[0] - fit.scores[1] == pytest.approx(d, abs=1e-3)
        assert sum(fit.scores) == pytest.approx(0.0, abs=1e-9)

    def test_matches_logistic_regression_oracle(self):
        true_s = [1.0, 0.3, -0.4, -0.9]
        games = _simulate(true_s, b=0.7, n_reps=150, seed=7)
        fit = fit_btl(4, games, l2=1e-9)

        n = 4
        rows, y = [], []
        for g in games:
            x = np.zeros(n + 1)
            x[g.i], x[g.j], x[n] = 1.0, -1.0, g.order
            rows.append(x)
            y.append(int(g.outcome))
        x_mat = np.array(rows)
        # Drop the last item's column (reference level) to remove the additive shift.
        design = np.delete(x_mat, n - 1, axis=1)
        # C is the inverse L2 strength: 1e12 is an effectively unpenalised fit and avoids
        # the penalty=None deprecation across scikit-learn versions.
        clf = LogisticRegression(C=1e12, fit_intercept=False, solver="lbfgs", tol=1e-10, max_iter=5000)
        clf.fit(design, y)
        coef = clf.coef_[0]
        oracle_rel = np.append(coef[: n - 1], 0.0)   # scores relative to the reference item
        fit_rel = np.array(fit.scores) - fit.scores[n - 1]
        np.testing.assert_allclose(fit_rel, oracle_rel, atol=5e-3)
        assert fit.order_effect == pytest.approx(coef[-1], abs=5e-3)

    def test_recovers_truth_with_enough_data(self):
        true_s = [0.8, 0.0, -0.8]
        games = _simulate(true_s, b=1.0, n_reps=600, seed=11)
        fit = fit_btl(3, games)
        centred = np.array(true_s) - np.mean(true_s)
        np.testing.assert_allclose(fit.scores, centred, atol=0.15)
        assert fit.order_effect == pytest.approx(1.0, abs=0.15)


class TestOrderEffect:
    def test_unbalanced_display_biases_scores_unless_order_modelled(self):
        # Three equally good items; the judge strongly favours the first-shown
        # response and the lower-index item is shown first 75% of the time.
        games = _simulate([0.0, 0.0, 0.0], b=1.2, n_reps=500, seed=3, p_first_is_low=0.75)
        aware = fit_btl(3, games)
        naive = fit_btl(3, games, order_effect=False)
        assert max(abs(s) for s in aware.scores) < 0.2
        assert max(abs(s) for s in naive.scores) > 0.3
        assert aware.order_effect is not None and aware.order_effect > 0.8
        assert naive.order_effect is None

    def test_single_display_order_is_not_identifiable(self):
        games = [
            ComparisonGame(case=k, i=0, j=1, order=1, outcome=1.0 if k % 3 else 0.0)
            for k in range(60)
        ]
        assert not order_effect_identifiable(2, games)
        fit = fit_btl(2, games)
        assert not fit.order_identifiable
        assert fit.order_effect is None
        assert fit.order_effect_se is None
        assert fit.scores[0] > fit.scores[1]

    def test_both_orders_are_identifiable(self):
        games = _both_orders(0, 1, 5, 10, 5, 10)
        assert order_effect_identifiable(2, games)

    def test_cycle_with_nonzero_signed_order_is_identifiable_without_swaps(self):
        # Triangle 0-1, 1-2, 0-2, each pair seen once; orders chosen so the signed
        # cycle sum is non-zero even though no pair was shown in both orders.
        games = [
            ComparisonGame(0, 0, 1, 1, 1.0),
            ComparisonGame(1, 1, 2, 1, 1.0),
            ComparisonGame(2, 0, 2, 1, 1.0),
        ]
        assert order_effect_identifiable(3, games)

    def test_pure_position_judge_gives_flat_scores_and_large_b(self):
        # Always prefers whatever is shown first, regardless of item.
        games = []
        for rep in range(100):
            for i, j in itertools.combinations(range(3), 2):
                for order in (1, -1):
                    games.append(ComparisonGame(rep, i, j, order, 1.0 if order == 1 else 0.0))
        fit = fit_btl(3, games)
        assert max(abs(s) for s in fit.scores) < 0.05
        assert fit.order_effect is not None and fit.order_effect > 3.0
        assert fit.possibly_separated   # b runs off: flagged by the item-level check below


class TestStructure:
    def test_label_swap_invariance(self):
        games = _simulate([0.5, -0.5, 0.0], b=0.4, n_reps=80, seed=5)
        swapped = [
            ComparisonGame(g.case, g.j, g.i, -g.order, 1.0 - g.outcome) for g in games
        ]
        a, b = fit_btl(3, games), fit_btl(3, swapped)
        np.testing.assert_allclose(a.scores, b.scores, atol=1e-6)
        assert a.order_effect == pytest.approx(b.order_effect, abs=1e-6)

    def test_ties_count_as_half(self):
        games = [ComparisonGame(k, 0, 1, 1 if k % 2 else -1, 0.5) for k in range(40)]
        fit = fit_btl(2, games)
        assert abs(fit.scores[0] - fit.scores[1]) < 1e-3
        assert abs(fit.order_effect or 0.0) < 1e-3

    def test_disconnected_graph_raises(self):
        games = _both_orders(0, 1, 5, 10, 5, 10) + _both_orders(2, 3, 5, 10, 5, 10, case0=100)
        with pytest.raises(ValueError, match="not connected"):
            fit_btl(4, games)

    def test_complete_dominance_is_finite_and_flagged(self):
        games = []
        for rep in range(20):
            for j in (1, 2):
                games.append(ComparisonGame(rep, 0, j, 1, 1.0))
                games.append(ComparisonGame(rep, 0, j, -1, 1.0))
            games.append(ComparisonGame(rep, 1, 2, 1, 1.0 if rep % 2 else 0.0))
            games.append(ComparisonGame(rep, 1, 2, -1, 1.0 if rep % 2 else 0.0))
        fit = fit_btl(3, games)
        assert all(math.isfinite(s) for s in fit.scores)
        assert fit.scores[0] == max(fit.scores)
        assert fit.possibly_separated

    @pytest.mark.parametrize(
        "game",
        [
            ComparisonGame(0, 0, 0, 1, 1.0),    # i == j
            ComparisonGame(0, 0, 5, 1, 1.0),    # index out of range
            ComparisonGame(0, 0, 1, 0, 1.0),    # bad order
            ComparisonGame(0, 0, 1, 1, 1.5),    # bad outcome
        ],
    )
    def test_invalid_games_raise(self, game):
        with pytest.raises(ValueError):
            fit_btl(2, [game])

    def test_empty_games_raise(self):
        with pytest.raises(ValueError):
            fit_btl(2, [])


class TestUncertainty:
    def test_se_shrinks_with_more_data(self):
        small = fit_btl(3, _simulate([0.5, 0.0, -0.5], 0.3, n_reps=50, seed=1))
        large = fit_btl(3, _simulate([0.5, 0.0, -0.5], 0.3, n_reps=800, seed=1))
        assert np.mean(large.scores_se) < 0.5 * np.mean(small.scores_se)
        assert large.order_effect_se is not None and small.order_effect_se is not None
        assert large.order_effect_se < small.order_effect_se

    def test_bootstrap_intervals_cover_truth_and_are_reproducible(self):
        true_s = [0.8, 0.0, -0.8]
        games = _simulate(true_s, b=1.0, n_reps=400, seed=21)
        fit = fit_btl(3, games)
        boot = bootstrap_btl(3, games, n_boot=200, seed=0)
        again = bootstrap_btl(3, games, n_boot=200, seed=0)
        assert boot.scores_low == again.scores_low
        centred = np.array(true_s) - np.mean(true_s)
        for k in range(3):
            assert boot.scores_low[k] < fit.scores[k] < boot.scores_high[k]
            assert boot.scores_low[k] <= centred[k] <= boot.scores_high[k]
        assert boot.order_effect_low is not None and boot.order_effect_high is not None
        assert fit.order_effect is not None
        # Deterministic contract for the fixed seed: the interval brackets the point
        # estimate (a single bootstrap draw does not guarantee covering the truth).
        assert boot.order_effect_low < fit.order_effect < boot.order_effect_high

    def test_order_interval_requires_enough_identifiable_replicates(self):
        # Only case 0 contains both display orders; the other cases show order +1 only.
        # A replicate keeps the order effect identifiable iff it resamples case 0, so with
        # n_boot=2 some seeds yield exactly one identifiable draw. That must not be published
        # as a (degenerate, zero-width) interval.
        games = [ComparisonGame(0, 0, 1, 1, 1.0), ComparisonGame(0, 0, 1, -1, 0.0)]
        games += [ComparisonGame(k, 0, 1, 1, 1.0 if k % 3 else 0.0) for k in range(1, 40)]
        assert order_effect_identifiable(2, games)

        published, withheld = 0, 0
        for seed in range(40):
            boot = bootstrap_btl(2, games, n_boot=2, seed=seed)
            assert (boot.order_effect_low is None) == (boot.order_effect_high is None)
            if boot.order_effect_low is None:
                withheld += 1
            else:
                published += 1
                assert boot.order_effect_high is not None
                assert boot.order_effect_low < boot.order_effect_high   # never zero-width
        assert withheld > 0 and published > 0   # both branches were exercised

    def test_bootstrap_validation(self):
        games = _simulate([0.0, 0.0], 0.0, n_reps=10, seed=0)
        with pytest.raises(ValueError):
            bootstrap_btl(2, games, n_boot=1)
        with pytest.raises(ValueError):
            bootstrap_btl(2, games, ci_level=1.0)
        with pytest.raises(ValueError, match="2 distinct cases"):
            bootstrap_btl(2, [ComparisonGame(0, 0, 1, 1, 1.0), ComparisonGame(0, 0, 1, -1, 0.0)])


class TestElo:
    def test_elo_scale_and_anchor(self):
        games = _both_orders(0, 1, 800, 1000, 600, 1000)
        fit = fit_btl(2, games, l2=1e-9)
        elo = fit.elo()
        assert sum(elo) / 2 == pytest.approx(1000.0)
        gap_logit = fit.scores[0] - fit.scores[1]
        assert elo[0] - elo[1] == pytest.approx(ELO_SCALE * gap_logit)
        # One logit of score is ~173.7 Elo points.
        assert pytest.approx(173.717, abs=1e-2) == ELO_SCALE
