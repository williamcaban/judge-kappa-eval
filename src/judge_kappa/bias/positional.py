"""
Positional Bias Detector.

Protocol (per case):
  Round 1: judge sees (A=control, B=treatment) → records preferred
  Round 2: judge sees (A=treatment, B=control) → records preferred (swapped)

A positionally biased judge prefers "first position" in both rounds,
regardless of which variant actually appears there.

corpus_bias_rate = fraction of cases where the judge exhibited a positional flip.
score_instability = mean absolute delta in per-variant scores between rounds —
  high instability without positional flip may indicate judge stochasticity.

order_effect = judge-level display-order effect from an order-aware Bradley-Terry
  model fitted to both rounds (log-odds; > 0 favours whichever output is shown
  first). Unlike the flip rate it is separated from real quality differences, so a
  judge that is merely right in both rounds is not counted as biased. Reported with a
  case-level bootstrap interval. See judge_kappa.ranking.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from judge_kappa.bias.base import BiasDetector
from judge_kappa.judges.pairwise import PairwiseJudge
from judge_kappa.models import BiasResult, EvalCase
from judge_kappa.ranking.btl import ComparisonGame, bootstrap_btl, fit_btl


@dataclass
class PositionalBiasReport:
    """Outcome of the swap test for one case: control and treatment judged in both display orders.

    ``control_scores`` and ``treatment_scores`` are ``(round 1, round 2)``.
    """
    case_id: str
    positional_flip: bool
    score_instability: float
    control_scores: tuple[float, float]   # (round1, round2)
    treatment_scores: tuple[float, float]
    parse_error: str | None = None        # first unusable-reply code in either round, if any
    truncated: bool = False               # either round hit the output-token limit


@dataclass
class OrderEffectEstimate:
    """Display-order effect of one judge, from ``PositionalBiasDetector.estimate_order_effect``."""

    order_effect: float
    ci_low: float | None
    ci_high: float | None
    n_cases: int


def _outcome(first: float, second: float) -> float:
    """1.0 if the first score wins, 0.0 if the second wins, 0.5 on a tie."""
    if first > second:
        return 1.0
    if first < second:
        return 0.0
    return 0.5


class PositionalBiasDetector(BiasDetector):
    """Swap test for positional bias.

    Each case is judged twice with control and treatment outputs in swapped positions. A judge that
    prefers the first-shown output in both rounds is positionally biased.
    """
    def __init__(self, judge: PairwiseJudge) -> None:
        """Create a detector around a pairwise judge.

        Args:
            judge: The ``PairwiseJudge`` used for both rounds.
        """
        self._judge = judge

    @staticmethod
    def estimate_order_effect(
        reports: Sequence[PositionalBiasReport],
        n_bootstrap: int = 300,
        ci_level: float = 0.95,
        seed: int | None = 0,
    ) -> OrderEffectEstimate | None:
        """Fit an order-aware BTL model to the two rounds of every case.

        Round 1 shows control first, round 2 shows treatment first, so the design
        always contains both display orders and the order effect is identifiable.
        Control is item 0 and treatment is item 1. Returns None for no reports.
        The interval needs at least two cases (otherwise ``ci_low``/``ci_high`` are None).
        """
        if not reports:
            return None
        games: list[ComparisonGame] = []
        for k, r in enumerate(reports):
            games.append(ComparisonGame(k, 0, 1, 1, _outcome(r.control_scores[0], r.treatment_scores[0])))
            games.append(ComparisonGame(k, 0, 1, -1, _outcome(r.control_scores[1], r.treatment_scores[1])))
        fit = fit_btl(2, games)
        assert fit.order_effect is not None   # both display orders are always present
        lo = hi = None
        if len(reports) >= 2 and n_bootstrap >= 2:
            boot = bootstrap_btl(2, games, n_boot=n_bootstrap, ci_level=ci_level, seed=seed)
            lo, hi = boot.order_effect_low, boot.order_effect_high
        return OrderEffectEstimate(
            order_effect=fit.order_effect,
            ci_low=lo,
            ci_high=hi,
            n_cases=len(reports),
        )

    def test_case(
        self,
        case: EvalCase,
        control_output: str,
        treatment_output: str,
    ) -> PositionalBiasReport:
        # Round 1: control=A, treatment=B
        """Run both rounds for one case.

        Round 1 shows control first, round 2 shows treatment first.

        Returns:
            A ``PositionalBiasReport``. ``positional_flip`` is True when the first-shown output wins
            both rounds; ``score_instability`` is the mean absolute score change per variant
            between rounds.
        """
        va_r1, vb_r1 = self._judge.judge_pair(
            case, control_output, treatment_output, "control", "treatment"
        )
        # Round 2: treatment=A, control=B  (positions swapped)
        va_r2, vb_r2 = self._judge.judge_pair(
            case, treatment_output, control_output, "treatment", "control"
        )

        prefers_first_r1 = va_r1.score > vb_r1.score   # control scored higher in r1
        prefers_first_r2 = va_r2.score > vb_r2.score   # treatment scored higher in r2
        positional_flip = prefers_first_r1 and prefers_first_r2

        ctrl_delta = abs(va_r1.score - vb_r2.score)
        trt_delta  = abs(vb_r1.score - va_r2.score)
        instability = (ctrl_delta + trt_delta) / 2

        return PositionalBiasReport(
            case_id=case.id,
            positional_flip=positional_flip,
            score_instability=round(instability, 4),
            control_scores=(va_r1.score, vb_r2.score),
            treatment_scores=(vb_r1.score, va_r2.score),
            parse_error=va_r1.parse_error or va_r2.parse_error,
            truncated=va_r1.truncated or va_r2.truncated,
        )

    def detect(self, **kwargs: Any) -> BiasResult:
        """Aggregate per-case reports into a corpus-level ``BiasResult``.

        Args:
            reports: (keyword) the ``PositionalBiasReport`` list.

        Returns:
            ``positional_bias_rate`` (fraction of flipped cases) and the order-aware
            ``order_effect`` with its bootstrap interval (see ``estimate_order_effect``).
        """
        reports: list[PositionalBiasReport] = kwargs["reports"]
        rate = (
            sum(1 for r in reports if r.positional_flip) / len(reports)
            if reports else 0.0
        )
        est = self.estimate_order_effect(reports)
        return BiasResult(
            positional_bias_rate=round(rate, 4),
            order_effect=round(est.order_effect, 4) if est else None,
            order_effect_ci_low=round(est.ci_low, 4) if est and est.ci_low is not None else None,
            order_effect_ci_high=round(est.ci_high, 4) if est and est.ci_high is not None else None,
        )
