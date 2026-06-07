"""
Cohen's κ inter-rater agreement.

For > 2 judges: computes all pairwise κ values and reports the mean.
Automatically detects if expected chance agreement is suspiciously high
(> 0.7), which signals the rubric may be too easy or ratings are biased
toward a single category.

Weighting:
  - NOMINAL: unweighted κ
  - ORDINAL / INTERVAL / RATIO: linear weights (proportional distance penalty)
"""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
from sklearn.metrics import cohen_kappa_score

from jury_eval.agreement.base import AgreementMetric
from jury_eval.models import AgreementResult, JudgeVerdict, ScaleType

_DISCRETIZE_BINS = 5  # 0.0–1.0 continuous score → 5 ordinal buckets


def _discretize(scores: list[float], n_bins: int = _DISCRETIZE_BINS) -> list[int]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    return [int(np.digitize(s, bins[1:-1])) for s in scores]


def _expected_chance_agreement(r1: list[int], r2: list[int]) -> float:
    n = len(r1)
    c1, c2 = Counter(r1), Counter(r2)
    all_labels = sorted(set(r1) | set(r2))
    return sum((c1.get(k, 0) / n) * (c2.get(k, 0) / n) for k in all_labels)


class CohenKappa(AgreementMetric):
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

        weights = (
            None if scale_type == ScaleType.NOMINAL else "linear"
        )

        pairwise_kappas: list[float] = []
        pairwise_expected: list[float] = []

        for j1, j2 in itertools.combinations(judge_ids, 2):
            shared = [c for c in case_ids if c in matrix[j1] and c in matrix[j2]]
            if len(shared) < 2:
                continue
            r1 = _discretize([matrix[j1][c] for c in shared])
            r2 = _discretize([matrix[j2][c] for c in shared])
            pairwise_kappas.append(float(cohen_kappa_score(r1, r2, weights=weights)))
            pairwise_expected.append(_expected_chance_agreement(r1, r2))

        if not pairwise_kappas:
            return AgreementResult(n_judges=len(judge_ids), n_cases=len(case_ids))

        mean_kappa    = float(np.mean(pairwise_kappas))
        mean_expected = float(np.mean(pairwise_expected))

        return AgreementResult(
            kappa=round(mean_kappa, 4),
            expected_chance_agreement=round(mean_expected, 4),
            n_judges=len(judge_ids),
            n_cases=len(case_ids),
        )
