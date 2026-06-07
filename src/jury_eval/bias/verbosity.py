"""
Verbosity Bias Detector.

Measures Spearman rank correlation between judge score and output token count
across the corpus. A significant positive ρ means the judge rewards longer
outputs regardless of quality.

Threshold: |ρ| > 0.30 AND p < 0.05 → flag as biased.

Note: token count is approximated by whitespace-split word count (already stored
in JudgeVerdict.output_token_count). For production use, replace with a proper
tokenizer count.
"""

from __future__ import annotations

from scipy import stats

from jury_eval.bias.base import BiasDetector
from jury_eval.models import BiasResult, JudgeVerdict


class VerbosityBiasDetector(BiasDetector):
    def __init__(self, threshold: float = 0.30) -> None:
        self._threshold = threshold

    def detect(self, **kwargs) -> BiasResult:
        verdicts: list[JudgeVerdict] = kwargs["verdicts"]

        scores  = [v.score for v in verdicts]
        lengths = [v.output_token_count for v in verdicts]

        if len(scores) < 3:
            return BiasResult()

        result = stats.spearmanr(lengths, scores)
        rho_f = float(result.statistic)  # type: ignore[attr-defined]
        p_f   = float(result.pvalue)     # type: ignore[attr-defined]
        biased = abs(rho_f) > self._threshold and p_f < 0.05

        return BiasResult(
            verbosity_bias_rho=round(rho_f, 4),
            verbosity_bias_p=round(p_f, 4),
            verbosity_biased=biased,
        )
