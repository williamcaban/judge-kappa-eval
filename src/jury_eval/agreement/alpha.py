"""
Krippendorff's α inter-rater reliability — the default agreement metric.

Advantages over Cohen's κ for LLM evaluation:
  - Handles missing data (judge skipped a case) via NaN — no imputation needed
  - Supports all measurement scales: nominal, ordinal, interval, ratio
  - Generalizes to any number of judges (not limited to pairs)
  - Appropriate for crowdsourced labels, rotating annotators, LLM panels

Thresholds (Krippendorff 2004):
  α ≥ 0.80  → strong agreement, conclusions are reliable
  0.67–0.80 → tentative conclusions only
  α < 0.67  → do not draw conclusions; recalibrate judges or revise rubric

Scale notes:
  ORDINAL  — use for 0–1 scores bucketed into ranks (default for LLM judges)
  INTERVAL — use when score differences are meaningful (e.g., exact 0–1 continuous)
  RATIO    — use when a true zero exists and ratios are meaningful
  NOMINAL  — use for categorical labels (e.g., PASS/FAIL classes)
"""

from __future__ import annotations

import numpy as np
import krippendorff

from jury_eval.agreement.base import AgreementMetric
from jury_eval.models import AgreementResult, JudgeVerdict, ScaleType


def _interpret(alpha: float) -> str:
    if alpha >= 0.80:
        return "strong agreement (α ≥ 0.80)"
    elif alpha >= 0.67:
        return "tentative conclusions (0.67 ≤ α < 0.80)"
    else:
        return "unreliable — recalibrate judges or revise rubric (α < 0.67)"


class KrippendorffAlpha(AgreementMetric):
    def compute(
        self,
        verdicts: list[JudgeVerdict],
        scale_type: ScaleType = ScaleType.ORDINAL,
    ) -> AgreementResult:
        judge_ids = sorted({v.judge_id for v in verdicts})
        case_ids  = sorted({v.eval_case_id for v in verdicts})

        matrix: dict[str, dict[str, float]] = {j: {} for j in judge_ids}
        for v in verdicts:
            matrix[v.judge_id][v.eval_case_id] = v.score

        # reliability_data shape: (n_judges, n_cases); NaN where missing
        reliability_data = np.full(
            (len(judge_ids), len(case_ids)), np.nan, dtype=float
        )
        for i, j in enumerate(judge_ids):
            for k, c in enumerate(case_ids):
                if c in matrix[j]:
                    reliability_data[i, k] = matrix[j][c]

        alpha = float(krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement=scale_type.value,
        ))

        return AgreementResult(
            alpha=round(alpha, 4),
            alpha_interpretation=_interpret(alpha),
            n_judges=len(judge_ids),
            n_cases=len(case_ids),
        )
