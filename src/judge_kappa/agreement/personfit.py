"""
Person-fit analysis — detects judges that score inconsistently relative to the panel.

Uses outfit MNSQ (mean square fit statistic) as a model-free approximation of the
IRT lz person-fit statistic. This does not require a fitted IRT model.

Protocol:
  1. Build a judge × case score matrix.
  2. For each case, estimate the "expected" score as the panel mean (item difficulty).
  3. For each judge, compute the outfit MNSQ = mean((observed - expected)² / expected_variance).
  4. Apply the t-transformation to produce a standardised statistic.

Interpretation (Wright & Masters 1982):
  |t| ≤ 1.96  → acceptable fit (p > 0.05)
  1.96 < |t| ≤ 3.0 → marginal misfit — monitor
  |t| > 3.0   → significant misfit — recalibrate or remove judge

A judge with high positive t is unexpectedly inconsistent (sometimes harsh, sometimes lenient).
A judge with high negative t scores with unrealistic uniformity (possible systematic bias).
"""

from __future__ import annotations

import numpy as np

from judge_kappa.models import JudgeFitResult, JudgeVerdict


def _expected_and_variance(
    judge_mean: float, case_mean: float, grand_mean: float
) -> tuple[float, float]:
    """
    1PL-like expected score: logistic(θ_judge - b_case) anchored at grand_mean.

    θ_judge = judge mean (leniency)
    b_case  = 1 - case_mean (difficulty; harder cases have lower mean)
    """
    logit = (judge_mean - grand_mean) - (1.0 - case_mean - (1.0 - grand_mean))
    p = 1.0 / (1.0 + np.exp(-logit))
    p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return p, p * (1.0 - p)


class PersonFitAnalyzer:
    """
    Compute per-judge fit statistics from a set of verdicts.

    Args:
        alpha_threshold: t-statistic threshold for flagging a judge. Default 1.96 (p=0.05).
    """

    def __init__(self, alpha_threshold: float = 1.96) -> None:
        self._threshold = alpha_threshold

    def analyze(self, verdicts: list[JudgeVerdict]) -> list[JudgeFitResult]:
        judge_ids = sorted({v.judge_id for v in verdicts})
        case_ids  = sorted({v.eval_case_id for v in verdicts})

        # Build score matrix (n_judges, n_cases); NaN for missing
        lookup: dict[str, dict[str, float]] = {j: {} for j in judge_ids}
        for v in verdicts:
            lookup[v.judge_id][v.eval_case_id] = v.score

        X = np.full((len(judge_ids), len(case_ids)), np.nan)
        for i, j in enumerate(judge_ids):
            for k, c in enumerate(case_ids):
                if c in lookup[j]:
                    X[i, k] = lookup[j][c]

        judge_means = np.nanmean(X, axis=1)
        case_means  = np.nanmean(X, axis=0)
        grand_mean  = float(np.nanmean(X))

        results: list[JudgeFitResult] = []
        for i, judge_id in enumerate(judge_ids):
            squared_residuals: list[float] = []
            expected_variances: list[float] = []

            for k, _ in enumerate(case_ids):
                if np.isnan(X[i, k]):
                    continue
                obs   = X[i, k]
                j_mean = float(judge_means[i])
                c_mean = float(case_means[k])
                exp, var = _expected_and_variance(j_mean, c_mean, grand_mean)
                if var < 1e-8:
                    continue
                squared_residuals.append((obs - exp) ** 2 / var)
                expected_variances.append(var)

            if len(squared_residuals) < 3:
                # Too few items — report NaN
                results.append(JudgeFitResult(
                    judge_id=judge_id,
                    lz_statistic=float("nan"),
                    flagged_inconsistent=False,
                ))
                continue

            n_items = len(squared_residuals)
            mnsq = float(np.mean(squared_residuals))

            # t-transformation (Wright & Masters 1982)
            # Var[MNSQ] ≈ 4/n (asymptotic)
            se_mnsq = 2.0 / np.sqrt(n_items)
            t_stat = (mnsq ** (1.0 / 3) - 1.0) * (3.0 / se_mnsq) + (se_mnsq / 3.0)
            t_stat = float(np.clip(t_stat, -10.0, 10.0))

            results.append(JudgeFitResult(
                judge_id=judge_id,
                lz_statistic=round(t_stat, 4),
                flagged_inconsistent=abs(t_stat) > self._threshold,
            ))

        return results
