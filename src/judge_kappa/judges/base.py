"""
LLMJudge ABC — defines the judge contract and ICL alignment mixin.

ICL (In-Context Learning) alignment: calibration_examples are injected into
every judge prompt as few-shot anchors. This reduces inter-judge score drift
by grounding all judges on the same human-validated reference points, which
directly improves Krippendorff's α and Cohen's κ in panel evaluations.
"""

from __future__ import annotations

import json
import logging
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeGuard

from judge_kappa.llm.base import LLMBackend, call_backend
from judge_kappa.llm.types import LLMResponse
from judge_kappa.models import CalibrationExample, EvalCase, JudgeVerdict

logger = logging.getLogger("judge_kappa.judges")

ParseFailurePolicy = Literal["flag", "raise"]
_MAX_WARNED_REPLIES = 5   # per judge instance; later ones are logged at DEBUG


class JudgeParseError(ValueError):
    """A judge reply could not be used and ``on_parse_failure="raise"`` was set."""

    def __init__(self, judge_id: str, case_id: str, error: str, excerpt: str) -> None:
        """Build the message and keep the judge id, case id and error code on the exception."""
        super().__init__(
            f"judge '{judge_id}' returned an unusable reply for case '{case_id}' "
            f"({error}): {excerpt!r}"
        )
        self.judge_id = judge_id
        self.case_id = case_id
        self.error = error


@dataclass(frozen=True)
class ParsedReply:
    """A judge reply after JSON extraction and schema validation."""

    data: dict[str, Any]
    response: LLMResponse
    error: str | None = None   # None → usable; otherwise a short machine-readable code

    @property
    def failed(self) -> bool:
        """True when the reply was not usable."""
        return self.error is not None


def is_finite_number(value: object) -> TypeGuard[int | float]:
    """A real, finite int/float (not bool). ``json.loads`` accepts ``Infinity``/``NaN`` and
    arbitrarily large ints, any of which would otherwise raise in float arithmetic."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_score_map(
    raw: object, expected: Sequence[str], kind: str, *, low: float = 0.0, high: float = 1.0
) -> str | None:
    """Check a ``{name: score}`` object; return an error code or None.

    Codes: ``<kind>_scores_not_an_object``, ``missing_<kind>s:k/n``,
    ``non_numeric_<kind>_score``, ``out_of_range_<kind>_score``.
    """
    if not isinstance(raw, dict):
        return f"{kind}_scores_not_an_object"
    missing = [k for k in expected if k not in raw]
    if missing:
        return f"missing_{kind}s:{len(missing)}/{len(expected)}"
    values = [raw[k] for k in expected]
    if not all(is_finite_number(v) for v in values):
        return f"non_numeric_{kind}_score"
    if any(v < low or v > high for v in values):
        return f"out_of_range_{kind}_score"
    return None


def numeric_scores(raw: object) -> dict[str, float]:
    """Numeric entries of a score object (anything else is dropped)."""
    if not isinstance(raw, dict):
        return {}
    return {str(k): float(v) for k, v in raw.items() if is_finite_number(v)}


class LLMJudge(ABC):
    """Abstract base for judges that score an output by calling an ``LLMBackend``.

    Provides calibration-anchor (ICL) rendering, reply parsing and validation with an
    ``on_parse_failure`` policy, and the accessors run provenance needs. Subclasses implement
    ``judge()``.
    """
    def __init__(
        self,
        judge_id: str,
        backend: LLMBackend,
        calibration_examples: list[CalibrationExample] | None = None,
        temperature: float = 0.0,
        on_parse_failure: ParseFailurePolicy = "flag",
    ) -> None:
        """
        Args:
            on_parse_failure: What to do when a reply is not usable JSON or does not
                match the expected schema.
                ``"flag"`` (default) keeps the historical fallback score so numbers do
                not change, marks the verdict ``parse_failed=True`` and counts it in
                the report's ``data_quality``. ``"raise"`` raises ``JudgeParseError``.
        """
        if on_parse_failure not in ("flag", "raise"):
            raise ValueError("on_parse_failure must be 'flag' or 'raise'")
        self.judge_id = judge_id
        self._backend = backend
        self._calibration = calibration_examples or []
        self._temperature = temperature
        self._on_parse_failure = on_parse_failure
        self._unusable_replies = 0

    @property
    def backend(self) -> LLMBackend:
        """The backend this judge calls."""
        return self._backend

    @property
    def temperature(self) -> float:
        """The sampling temperature sent with every request."""
        return self._temperature

    @property
    def on_parse_failure(self) -> str:
        """Policy for unusable replies: ``flag`` or ``raise``."""
        return self._on_parse_failure

    @property
    def n_calibration_examples(self) -> int:
        """Number of ICL calibration examples in the prompt."""
        return len(self._calibration)

    def _icl_block(self) -> str:
        """Render calibration examples as an ICL section for prompt injection."""
        if not self._calibration:
            return ""
        lines = ["## Calibration Anchors\n"
                 "Use these human-validated examples to anchor your scoring scale.\n"]
        for i, ex in enumerate(self._calibration, 1):
            lines.append(
                f"Example {i} | score={ex.score:.2f}\n"
                f"  Prompt: {ex.prompt}\n"
                f"  Output: {ex.output}\n"
                f"  Why: {ex.rationale}\n"
            )
        return "\n".join(lines)

    def _call(self, system: str, user: str) -> str:
        """Call the backend and return only the reply text."""
        return self._call_full(system, user).text

    def _call_full(self, system: str, user: str) -> LLMResponse:
        """Call the backend and keep usage, finish reason and latency."""
        return call_backend(self._backend, system, user, self._temperature)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Extract the outermost JSON object from ``text``.

        Returns ``{"_raw": text}`` when no valid JSON object can be parsed.
        """
        match = re.search(r"\{.*\}", text, re.DOTALL)
        raw = match.group(0) if match else text
        try:
            return json.loads(raw)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            return {"_raw": text}

    def _parse_reply(
        self,
        resp: LLMResponse,
        case_id: str,
        required: tuple[str, ...] = (),
        validate: Callable[[dict[str, Any]], str | None] | None = None,
    ) -> ParsedReply:
        """Extract JSON from a reply and validate it; never silently accept a bad one.

        ``required`` keys must be present; ``validate`` may return an error code for
        schema problems that need judge-specific knowledge. A reply cut off by the
        output-token limit is reported as ``truncated_output``.
        """
        data: object = self._parse_json(resp.text)
        error: str | None = None
        if not isinstance(data, dict):
            data, error = {"_raw": resp.text}, "not_an_object"
        elif set(data) == {"_raw"}:
            error = "invalid_json"
        else:
            for key in required:
                if data.get(key) is None:
                    error = f"missing_key:{key}"
                    break
            if error is None and validate is not None:
                error = validate(data)
        if error is not None and resp.truncated:
            error = "truncated_output"

        if error is not None:
            excerpt = resp.text[:120]
            self._unusable_replies = getattr(self, "_unusable_replies", 0) + 1
            # Log the first few per judge; the report's data_quality carries the totals.
            level = logging.WARNING if self._unusable_replies <= _MAX_WARNED_REPLIES else logging.DEBUG
            logger.log(
                level,
                "judge %s case %s: unusable reply (%s); fallback score used: %r%s",
                self.judge_id, case_id, error, excerpt,
                " (further unusable replies from this judge are logged at DEBUG)"
                if self._unusable_replies == _MAX_WARNED_REPLIES else "",
            )
            if self._on_parse_failure == "raise":
                raise JudgeParseError(self.judge_id, case_id, error, excerpt)
        return ParsedReply(data=data, response=resp, error=error)

    @abstractmethod
    def judge(self, case: EvalCase, output: str, variant_name: str) -> JudgeVerdict:
        """Score `output` for `case` and return a structured verdict."""
        ...
