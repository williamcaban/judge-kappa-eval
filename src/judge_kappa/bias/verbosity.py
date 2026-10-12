"""
Verbosity Bias Detector.

Measures Spearman rank correlation between judge score and output length across the
corpus. A significant positive ρ means the judge rewards longer outputs regardless of
quality.

Threshold: |ρ| > 0.30 AND p < 0.05 → flag as biased.

Length is ``JudgeVerdict.output_token_count`` of the *evaluated output*. The built-in
judges set it to the whitespace-split word count of that output, so the unit is words,
not model tokens. A verdict whose count is 0 has no measured length and is left out of
the correlation (and counted in ``BiasResult.verbosity_n_unmeasured``); it is never
replaced by the length of the judge's rationale, which is a different quantity.

Whitespace words are a weak length proxy for scripts written without spaces (CJK, Thai)
and for code. For those corpora set ``output_token_count`` yourself, with the tokenizer
or length measure you trust, in a custom judge or when building verdicts.
"""

from __future__ import annotations

import warnings
from typing import Any

from scipy import stats

from judge_kappa.bias.base import BiasDetector
from judge_kappa.models import BiasResult, JudgeVerdict

_MIN_MEASURED = 3


class VerbosityBiasDetector(BiasDetector):
    """Spearman correlation between judge score and output length.

    Args:
        threshold: Spearman |ρ| threshold above which verbosity bias is flagged.
        tiktoken_encoding: Deprecated and ignored. It was never used for counting; passing
            a value emits a ``DeprecationWarning``.
    """

    def __init__(
        self,
        threshold: float = 0.30,
        tiktoken_encoding: str | None = None,
    ) -> None:
        """Configure the threshold; see the class docstring for each argument."""
        if tiktoken_encoding is not None:
            warnings.warn(
                "VerbosityBiasDetector(tiktoken_encoding=...) has no effect and is "
                "deprecated: lengths come from JudgeVerdict.output_token_count (whitespace "
                "word count for the built-in judges). Set output_token_count yourself to "
                "use a tokenizer.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._threshold = threshold

    def detect(self, **kwargs: Any) -> BiasResult:
        """Correlate score with output length across the verdicts.

        Args:
            verdicts: (keyword) the ``JudgeVerdict`` list.

        Returns:
            ``BiasResult`` with Spearman rho, its p-value and ``verbosity_biased`` (|rho| above
            the threshold and p < 0.05). Only verdicts with ``output_token_count > 0`` are
            used; ``verbosity_n_measured`` and ``verbosity_n_unmeasured`` report how many were
            and were not. With fewer than 3 measured verdicts no correlation is estimated and
            ``verbosity_bias_rho`` stays ``None``.
        """
        verdicts: list[JudgeVerdict] = kwargs["verdicts"]

        measured = [v for v in verdicts if v.output_token_count > 0]
        n_measured = len(measured)
        n_unmeasured = len(verdicts) - n_measured

        if n_measured < _MIN_MEASURED:
            return BiasResult(
                verbosity_n_measured=n_measured, verbosity_n_unmeasured=n_unmeasured
            )

        result = stats.spearmanr(
            [v.output_token_count for v in measured], [v.score for v in measured]
        )
        rho_f = float(result.statistic)
        p_f   = float(result.pvalue)
        biased = abs(rho_f) > self._threshold and p_f < 0.05

        return BiasResult(
            verbosity_bias_rho=round(rho_f, 4),
            verbosity_bias_p=round(p_f, 4),
            verbosity_biased=biased,
            verbosity_n_measured=n_measured,
            verbosity_n_unmeasured=n_unmeasured,
        )
