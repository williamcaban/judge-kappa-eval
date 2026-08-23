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

import krippendorff
import numpy as np

from judge_kappa.agreement.base import AgreementMetric
from judge_kappa.models import AgreementResult, JudgeVerdict, ScaleType


def _interpret(alpha: float) -> str:
    if alpha >= 0.80:
        return "strong agreement (α ≥ 0.80)"
    elif alpha >= 0.67:
        return "tentative conclusions (0.67 ≤ α < 0.80)"
    else:
        return "unreliable — recalibrate judges or revise rubric (α < 0.67)"


def _build_matrix(
    verdicts: list[JudgeVerdict],
    judge_ids: list[str],
    case_ids: list[str],
) -> np.ndarray:
    """Build (n_judges, n_cases) reliability matrix with NaN for missing entries."""
    lookup: dict[str, dict[str, float]] = {j: {} for j in judge_ids}
    for v in verdicts:
        lookup[v.judge_id][v.eval_case_id] = v.score
    data = np.full((len(judge_ids), len(case_ids)), np.nan, dtype=float)
    for i, j in enumerate(judge_ids):
        for k, c in enumerate(case_ids):
            if c in lookup[j]:
                data[i, k] = lookup[j][c]
    return data


class KrippendorffAlpha(AgreementMetric):
    def __init__(self, bootstrap_ci: bool = True, n_bootstrap: int = 2000, seed: int = 42) -> None:
        self._bootstrap_ci = bootstrap_ci
        self._n_bootstrap = n_bootstrap
        self._seed = seed

    def compute(
        self,
        verdicts: list[JudgeVerdict],
        scale_type: ScaleType = ScaleType.ORDINAL,
    ) -> AgreementResult:
        judge_ids = sorted({v.judge_id for v in verdicts})
        case_ids  = sorted({v.eval_case_id for v in verdicts})

        reliability_data = _build_matrix(verdicts, judge_ids, case_ids)

        try:
            alpha = float(krippendorff.alpha(
                reliability_data=reliability_data,
                level_of_measurement=scale_type.value,
            ))
        except ValueError:
            # Single-value domain: all raters gave identical scores → perfect agreement
            alpha = 1.0

        ci_low, ci_high = None, None
        if self._bootstrap_ci and len(case_ids) >= 5:
            ci_low, ci_high = self._bootstrap_ci_bounds(
                reliability_data, scale_type, len(case_ids)
            )

        return AgreementResult(
            alpha=round(alpha, 4),
            alpha_ci_low=ci_low,
            alpha_ci_high=ci_high,
            alpha_interpretation=_interpret(alpha),
            n_judges=len(judge_ids),
            n_cases=len(case_ids),
        )

    def _bootstrap_ci_bounds(
        self,
        data: np.ndarray,
        scale_type: ScaleType,
        n_cases: int,
    ) -> tuple[float, float]:
        rng = np.random.default_rng(self._seed)
        boot_alphas: list[float] = []
        for _ in range(self._n_bootstrap):
            idx = rng.integers(0, n_cases, size=n_cases)
            sample = data[:, idx]
            try:
                a = float(krippendorff.alpha(
                    reliability_data=sample,
                    level_of_measurement=scale_type.value,
                ))
                boot_alphas.append(a)
            except Exception:
                pass
        if not boot_alphas:
            return None, None  # type: ignore[return-value]
        arr = np.array(boot_alphas)
        return round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)
