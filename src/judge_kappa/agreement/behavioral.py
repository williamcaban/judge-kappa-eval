"""
BehavioralAlignmentMetric — Krippendorff α repurposed for cross-condition consistency.

Inspired by the DISC paper: instead of measuring whether multiple judges agree on
the same output, this measures whether the same model agrees with *itself* across
different evaluation conditions (e.g., zero-shot vs. system-prompted vs. few-shot).

Semantics:
  - "Raters" are conditions (e.g., "zero_shot", "system_prompted", "chain_of_thought").
  - "Items" are eval cases.
  - α < 0.80 → the model's behavior changes significantly with condition → condition sensitivity.
  - α ≥ 0.80 → model is robust to condition framing → behavioral consistency.

This is the same Krippendorff α math, just with a different interpretive frame.
The threshold interpretation is also adjusted:
  α ≥ 0.80 → behaviorally consistent across conditions
  0.67–0.80 → modest sensitivity — monitor but not alarming
  α < 0.67  → high sensitivity — condition framing materially changes outputs

Usage:
    from judge_kappa.agreement import BehavioralAlignmentMetric
    from judge_kappa.models import JudgeVerdict, ScaleType

    # Build verdicts where judge_id = condition name, eval_case_id = case id
    metric = BehavioralAlignmentMetric()
    result = metric.compute(condition_verdicts, ScaleType.ORDINAL)
    print(result.alpha, result.alpha_interpretation)
"""

from __future__ import annotations

from judge_kappa.agreement.alpha import KrippendorffAlpha
from judge_kappa.models import AgreementResult, JudgeVerdict, ScaleType


def _interpret_behavioral(alpha: float) -> str:
    if alpha >= 0.80:
        return "behaviorally consistent across conditions (α ≥ 0.80)"
    elif alpha >= 0.67:
        return "modest condition sensitivity (0.67 ≤ α < 0.80)"
    else:
        return "high condition sensitivity — framing materially changes outputs (α < 0.67)"


class BehavioralAlignmentMetric:
    """
    Measures cross-condition behavioral consistency using Krippendorff α.

    Accepts verdicts where judge_id encodes the condition name.
    Returns an AgreementResult with a condition-appropriate interpretation.
    """

    def __init__(self, bootstrap_ci: bool = True, n_bootstrap: int = 2000, seed: int = 42) -> None:
        self._alpha_metric = KrippendorffAlpha(
            bootstrap_ci=bootstrap_ci, n_bootstrap=n_bootstrap, seed=seed
        )

    def compute(
        self,
        verdicts: list[JudgeVerdict],
        scale_type: ScaleType = ScaleType.ORDINAL,
    ) -> AgreementResult:
        result = self._alpha_metric.compute(verdicts, scale_type)
        interp = (
            _interpret_behavioral(result.alpha or 0.0)
            if result.alpha is not None
            else "insufficient data"
        )
        return result.model_copy(update={"alpha_interpretation": interp})

    @staticmethod
    def from_condition_scores(
        condition_scores: dict[str, dict[str, float]],
    ) -> list[JudgeVerdict]:
        """
        Convenience constructor: build verdicts from a nested dict.

        Args:
            condition_scores: {condition_name: {case_id: score}}

        Returns list of JudgeVerdict where judge_id = condition_name.
        """
        verdicts: list[JudgeVerdict] = []
        for condition, case_map in condition_scores.items():
            for case_id, score in case_map.items():
                verdicts.append(JudgeVerdict(
                    judge_id=condition,
                    eval_case_id=case_id,
                    variant_name="model",
                    score=float(score),
                    rationale="",
                ))
        return verdicts
