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
from typing import Any, Literal

from scipy import stats

from judge_kappa.bias.base import BiasDetector
from judge_kappa.models import BiasResult, JudgeVerdict

_MIN_MEASURED = 3

# Short, human-readable text for BiasResult.verbosity_not_estimated_reason (CLI, warnings).
NOT_ESTIMATED_TEXT: dict[str, str] = {
    "too_few_measured": f"fewer than {_MIN_MEASURED} verdicts with a measured output length",
    "constant_scores": "all judge scores are identical",
    "constant_lengths": "all outputs have the same length",
}


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
            and were not. When no correlation can be estimated (fewer than 3 measured
            verdicts, identical scores, or identical lengths) ``verbosity_bias_rho`` and
            ``verbosity_bias_p`` are ``None``, ``verbosity_biased`` is ``False`` and
            ``verbosity_not_estimated_reason`` says why. Identical scores are checked first.
        """
        verdicts: list[JudgeVerdict] = kwargs["verdicts"]

        measured = [v for v in verdicts if v.output_token_count > 0]
        n_measured = len(measured)
        n_unmeasured = len(verdicts) - n_measured

        lengths = [v.output_token_count for v in measured]
        scores = [v.score for v in measured]

        reason: Literal["too_few_measured", "constant_scores", "constant_lengths"] | None = None
        if n_measured < _MIN_MEASURED:
            reason = "too_few_measured"
        elif len(set(scores)) < 2:
            reason = "constant_scores"
        elif len(set(lengths)) < 2:
            reason = "constant_lengths"
        if reason is not None:
            # A rank correlation is undefined on a constant vector; do not call spearmanr
            # (it returns NaN and warns) and do not report 0.0, which would assert "no bias".
            return BiasResult(
                verbosity_n_measured=n_measured,
                verbosity_n_unmeasured=n_unmeasured,
                verbosity_not_estimated_reason=reason,
            )

        result = stats.spearmanr(lengths, scores)
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
