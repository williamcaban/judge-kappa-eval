"""
Data-quality accounting for judge output.

Judges that cannot parse a reply fall back to a fixed score so a run can finish.
That fallback is only acceptable if it is *visible*; this module turns the per-reply
flags into the ``DataQuality`` block attached to every report.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable

from judge_kappa.models import DataQuality

logger = logging.getLogger("judge_kappa.quality")

# One judge reply: (judge_id, parse_error or None, truncated)
ReplyFlag = tuple[str, str | None, bool]


def compute_data_quality(replies: Iterable[ReplyFlag], extra_truncated: int = 0) -> DataQuality:
    """Summarise reply flags.

    Args:
        replies: one ``(judge_id, parse_error, truncated)`` per judge reply.
        extra_truncated: truncated replies seen outside ``replies`` (for example
            generation calls), added to ``n_truncated_replies``.
    """
    n = 0
    by_judge: Counter[str] = Counter()
    codes: Counter[str] = Counter()
    truncated = extra_truncated
    for judge_id, error, was_truncated in replies:
        n += 1
        if error is not None:
            by_judge[judge_id] += 1
            codes[error.split(":")[0]] += 1
        if was_truncated:
            truncated += 1

    failures = sum(by_judge.values())
    warnings: list[str] = []
    if failures:
        top = ", ".join(f"{c} x{k}" for c, k in codes.most_common(3))
        warnings.append(
            f"{failures} of {n} judge replies ({failures / n:.1%}) were unusable ({top}). "
            "Their verdicts carry fallback scores that are included in the aggregate metrics. "
            "Inspect verdicts with parse_failed=True, or set on_parse_failure='raise' to fail fast."
        )
    if truncated:
        warnings.append(
            f"{truncated} replies hit the output-token limit. Raise the backend max_tokens; "
            "truncated judge JSON is unusable and truncated generations bias the evaluation."
        )
    dq = DataQuality(
        n_judge_calls=n,
        n_parse_failures=failures,
        parse_failure_rate=round(failures / n, 4) if n else 0.0,
        parse_failures_by_judge=dict(by_judge),
        parse_errors=dict(codes),
        n_truncated_replies=truncated,
        warnings=warnings,
    )
    for w in warnings:
        logger.warning(w)
    return dq
