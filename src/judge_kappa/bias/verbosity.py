"""
Verbosity Bias Detector.

Measures Spearman rank correlation between judge score and output token count
across the corpus. A significant positive ρ means the judge rewards longer
outputs regardless of quality.

Threshold: |ρ| > 0.30 AND p < 0.05 → flag as biased.

Token count priority (first available wins):
  1. JudgeVerdict.output_token_count if set by the judge (non-zero)
  2. tiktoken tokenizer count if tiktoken is installed (pip install tiktoken)
  3. Whitespace-split word count as fallback

Install tiktoken for accurate token counts: pip install tiktoken
"""

from __future__ import annotations

from scipy import stats

from judge_kappa.bias.base import BiasDetector
from judge_kappa.models import BiasResult, JudgeVerdict

class VerbosityBiasDetector(BiasDetector):
    """
    Args:
        threshold: Spearman |ρ| threshold above which verbosity bias is flagged.
        tiktoken_encoding: tiktoken encoding name used when tiktoken is installed.
            Use "cl100k_base" for GPT-4/Claude-3; "o200k_base" for GPT-4o.
    """

    def __init__(
        self,
        threshold: float = 0.30,
        tiktoken_encoding: str = "cl100k_base",
    ) -> None:
        self._threshold = threshold
        self._encoding = tiktoken_encoding

    def detect(self, **kwargs) -> BiasResult:
        verdicts: list[JudgeVerdict] = kwargs["verdicts"]

        scores = [v.score for v in verdicts]
        # Use pre-computed token count from verdict if non-zero, else re-count from output
        lengths: list[int] = []
        for v in verdicts:
            if v.output_token_count > 0:
                lengths.append(v.output_token_count)
            else:
                # output not stored in JudgeVerdict; fall back to whitespace split
                lengths.append(len(v.rationale.split()))  # approximate from rationale length

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
