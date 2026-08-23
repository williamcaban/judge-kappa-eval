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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from judge_kappa.bias.base import BiasDetector
from judge_kappa.judges.pairwise import PairwiseJudge
from judge_kappa.models import BiasResult, EvalCase


@dataclass
class PositionalBiasReport:
    case_id: str
    positional_flip: bool
    score_instability: float
    control_scores: tuple[float, float]   # (round1, round2)
    treatment_scores: tuple[float, float]


class PositionalBiasDetector(BiasDetector):
    def __init__(self, judge: PairwiseJudge) -> None:
        self._judge = judge

    def test_case(
        self,
        case: EvalCase,
        control_output: str,
        treatment_output: str,
    ) -> PositionalBiasReport:
        # Round 1: control=A, treatment=B
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
        )

    def detect(self, **kwargs: Any) -> BiasResult:
        reports: list[PositionalBiasReport] = kwargs["reports"]
        rate = (
            sum(1 for r in reports if r.positional_flip) / len(reports)
            if reports else 0.0
        )
        return BiasResult(positional_bias_rate=round(rate, 4))
