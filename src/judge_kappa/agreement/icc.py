"""
Intraclass Correlation Coefficient — ICC(2,k) absolute agreement.

ICC(2,k) decomposes score variance into:
  - Between-cases (MSB)  — the signal: how much cases actually differ
  - Between-judges (MSJ) — systematic leniency/harshness differences
  - Residual error (MSE) — unexplained noise

Unlike Krippendorff α (which aggregates all disagreement into one number),
ICC(2,k) reveals *why* judges disagree:
  - High MSJ, low MSE → judges have systematic offsets → fix with ICL calibration
  - Low MSJ, high MSE → judges are noisy → fix with more cases or tighter rubric

ICC(2,k) is appropriate for average of k raters (not a single rater).
Thresholds (Cicchetti 1994):
  ICC ≥ 0.75 → excellent
  0.60–0.74  → good
  0.40–0.59  → fair
  ICC < 0.40 → poor
"""

from __future__ import annotations

import numpy as np

from judge_kappa.models import AgreementResult, JudgeVerdict


def _interpret_icc(icc: float) -> str:
    if icc >= 0.75:
        return "excellent reliability (ICC ≥ 0.75)"
    elif icc >= 0.60:
        return "good reliability (0.60 ≤ ICC < 0.75)"
    elif icc >= 0.40:
        return "fair reliability (0.40 ≤ ICC < 0.60)"
    else:
        return "poor reliability (ICC < 0.40)"


def compute_icc(verdicts: list[JudgeVerdict]) -> tuple[float, str]:
    """
    Compute ICC(2,k) from a list of verdicts.

    Returns (icc_value, interpretation_string).
    Returns (0.0, "insufficient data") when fewer than 2 judges or 2 cases.
    """
    judge_ids = sorted({v.judge_id for v in verdicts})
    case_ids  = sorted({v.eval_case_id for v in verdicts})
    n, k = len(case_ids), len(judge_ids)

    if n < 2 or k < 2:
        return 0.0, "insufficient data — need ≥ 2 judges and ≥ 2 cases"

    # Build (n_cases, k_judges) matrix; NaN for missing entries
    lookup: dict[str, dict[str, float]] = {j: {} for j in judge_ids}
    for v in verdicts:
        lookup[v.judge_id][v.eval_case_id] = v.score

    X = np.full((n, k), np.nan)
    for i, c in enumerate(case_ids):
        for j, judge in enumerate(judge_ids):
            if c in lookup[judge]:
                X[i, j] = lookup[judge][c]

    # Drop cases or judges with all-NaN if any
    row_valid = ~np.all(np.isnan(X), axis=1)
    col_valid = ~np.all(np.isnan(X), axis=0)
    X = X[row_valid][:, col_valid]
    n, k = X.shape
    if n < 2 or k < 2:
        return 0.0, "insufficient data after removing missing"

    # Fill remaining NaN with column (judge) mean for ANOVA decomposition
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    X_filled = X.copy()
    X_filled[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    grand_mean = X_filled.mean()
    row_means  = X_filled.mean(axis=1)
    col_means2 = X_filled.mean(axis=0)

    SS_between = k * np.sum((row_means - grand_mean) ** 2)
    SS_rater   = n * np.sum((col_means2 - grand_mean) ** 2)
    SS_total   = np.sum((X_filled - grand_mean) ** 2)
    SS_error   = SS_total - SS_between - SS_rater

    MS_between = SS_between / (n - 1)
    MS_rater   = SS_rater   / (k - 1)
    MS_error   = SS_error   / ((n - 1) * (k - 1))

    # ICC(2,k) — absolute agreement, average of k raters
    denom = MS_between + (MS_rater - MS_error) / n
    if denom <= 0:
        return 0.0, "degenerate — all cases scored identically"
    icc = (MS_between - MS_error) / denom
    icc = float(np.clip(icc, -1.0, 1.0))
    return round(icc, 4), _interpret_icc(icc)


def enrich_agreement_with_icc(result: AgreementResult, verdicts: list[JudgeVerdict]) -> AgreementResult:
    """Return a copy of result with icc and icc_interpretation populated."""
    icc_val, icc_interp = compute_icc(verdicts)
    return result.model_copy(update={"icc": icc_val, "icc_interpretation": icc_interp})
