"""
Differential Item Functioning (DIF) detector for eval cases.

DIF identifies eval cases (items) where the probability of a positive verdict
differs systematically across judge model families — after controlling for the
overall score level. If case A is consistently easier for "gpt" judges but
harder for "claude" judges at the same overall score level, case A has DIF.

DIF-flagged cases should be reviewed before reporting model comparisons, because
observed uplift on those cases may reflect judge family preferences rather than
genuine quality differences.

Protocol (Mantel-Haenszel via logistic regression):
  1. Group judges by model family (prefix of judge_id before "-" or "_" separator).
  2. For each case, run logistic regression:
       P(verdict=positive) = logit(β₀ + β₁·mean_score + β₂·group_indicator)
  3. Flag the case if the group_indicator coefficient is significant (p < α_threshold).

Model family extraction: judge_id prefix before first "-", "_", or ":" delimiter.
  "claude-3-opus-j1" → "claude"
  "gpt-4o-j2"        → "gpt"
  "llama-3-j1"       → "llama"
  "custom_judge"     → "custom"

Returns a list of DIF-flagged case IDs and per-case statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from judge_kappa.bias.base import BiasDetector
from judge_kappa.models import BiasResult, JudgeVerdict


@dataclass
class DIFCaseResult:
    case_id: str
    n_verdicts: int
    group_coefficient: float   # β₂ from logistic regression
    p_value: float             # Wald p-value for group_indicator
    flagged: bool              # True if p < alpha_threshold


@dataclass
class DIFReport:
    flagged_cases: list[str]                   # case_ids with significant DIF
    per_case: list[DIFCaseResult] = field(default_factory=list)
    n_groups: int = 0
    alpha_threshold: float = 0.05

    def summary(self) -> str:
        return (
            f"{len(self.flagged_cases)}/{len(self.per_case)} cases flagged for DIF "
            f"({self.n_groups} judge families, α={self.alpha_threshold})"
        )


def _model_family(judge_id: str) -> str:
    """Extract model family prefix from judge_id."""
    for sep in ("-", "_", ":"):
        if sep in judge_id:
            return judge_id.split(sep)[0].lower()
    return judge_id.lower()


class DifferentialItemFunctioningDetector(BiasDetector):
    """
    Detect eval cases with differential item functioning across judge model families.

    Args:
        alpha_threshold: Wald test p-value threshold for flagging a case. Default 0.05.
        positive_threshold: Score cutoff for binarising verdicts to positive/negative.
        min_verdicts_per_case: Skip cases with fewer verdicts than this.
        min_groups: Require at least this many distinct model families to run DIF.
    """

    def __init__(
        self,
        alpha_threshold: float = 0.05,
        positive_threshold: float = 0.5,
        min_verdicts_per_case: int = 4,
        min_groups: int = 2,
    ) -> None:
        self._alpha = alpha_threshold
        self._positive_threshold = positive_threshold
        self._min_per_case = min_verdicts_per_case
        self._min_groups = min_groups

    def detect(self, **kwargs) -> BiasResult:
        """Returns a minimal BiasResult. For full DIF details use analyze()."""
        verdicts: list[JudgeVerdict] = kwargs["verdicts"]
        report = self.analyze(verdicts)
        return BiasResult()  # DIF doesn't map to existing BiasResult fields

    def analyze(self, verdicts: list[JudgeVerdict]) -> DIFReport:
        """Run DIF analysis and return a detailed DIFReport."""
        families = sorted({_model_family(v.judge_id) for v in verdicts})
        if len(families) < self._min_groups:
            return DIFReport(
                flagged_cases=[],
                per_case=[],
                n_groups=len(families),
                alpha_threshold=self._alpha,
            )

        # Map family → integer group code
        family_to_code = {f: i for i, f in enumerate(families)}

        # Group verdicts by case
        by_case: dict[str, list[JudgeVerdict]] = {}
        for v in verdicts:
            by_case.setdefault(v.eval_case_id, []).append(v)

        per_case_results: list[DIFCaseResult] = []

        for case_id, case_verdicts in by_case.items():
            if len(case_verdicts) < self._min_per_case:
                continue

            # Build feature matrix
            scores  = np.array([v.score for v in case_verdicts])
            groups  = np.array([family_to_code[_model_family(v.judge_id)] for v in case_verdicts])
            # Binarise: score >= threshold → 1 (positive verdict)
            y = (scores >= self._positive_threshold).astype(int)

            # Need at least both classes present to fit logistic regression
            if len(np.unique(y)) < 2:
                continue

            # Group indicator: binary (reference group = 0)
            group_binary = (groups > 0).astype(float)
            X = np.column_stack([scores, group_binary])

            try:
                clf = LogisticRegression(max_iter=500, solver="lbfgs")
                clf.fit(X, y)
                coef = float(clf.coef_[0, 1])  # β₂ for group_indicator

                # Wald test p-value: approximate via normal distribution
                # SE from Hessian (diagonal of inverse Fisher information)
                proba = clf.predict_proba(X)[:, 1]
                W = proba * (1 - proba)
                XtWX = X.T @ np.diag(W) @ X
                try:
                    cov = np.linalg.inv(XtWX)
                    se = float(np.sqrt(np.abs(cov[1, 1])))
                    z = coef / se if se > 1e-10 else 0.0
                    from scipy import stats as _stats
                    p_value = float(2 * _stats.norm.sf(abs(z)))
                except np.linalg.LinAlgError:
                    p_value = 1.0

            except Exception:
                continue

            per_case_results.append(DIFCaseResult(
                case_id=case_id,
                n_verdicts=len(case_verdicts),
                group_coefficient=round(coef, 4),
                p_value=round(p_value, 4),
                flagged=p_value < self._alpha,
            ))

        flagged = [r.case_id for r in per_case_results if r.flagged]
        return DIFReport(
            flagged_cases=flagged,
            per_case=per_case_results,
            n_groups=len(families),
            alpha_threshold=self._alpha,
        )
