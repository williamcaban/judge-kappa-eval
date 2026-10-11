"""
Run provenance: what configuration and library versions produced a report.

A measurement library has to be able to say which model, with which request
parameters and which SDK versions, produced a number. ``begin_run`` snapshots each
backend's cumulative usage; ``end_run`` returns a ``RunMetadata`` whose usage is the
delta for that run, so a backend reused across runs is not double counted.

Nothing here reads credentials: backends describe themselves through ``describe()``,
which by contract never includes API keys, and base URLs are stripped of userinfo.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from typing import Any

from judge_kappa.judges.base import LLMJudge
from judge_kappa.llm.base import backend_usage, describe_backend
from judge_kappa.llm.types import UsageSnapshot
from judge_kappa.models import BackendRecord, JudgeRecord, RunMetadata, UsageSummary

_TRACKED_PACKAGES = ("openai", "anthropic", "numpy", "scipy", "pydantic", "scikit-learn")


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _as_float(value: object, default: float = 0.0) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else default


def _as_int(value: object, default: int = 0) -> int:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else default


@dataclass
class _Track:
    backend: object
    role: str
    judge_ids: list[str] = field(default_factory=list)
    before: UsageSnapshot | None = None


@dataclass
class RunState:
    started_at: datetime
    t0: float
    tracks: list[_Track]
    judges: list[LLMJudge]


def begin_run(
    judges: Sequence[LLMJudge] = (),
    backends: Sequence[tuple[str, object | None]] = (),
) -> RunState:
    """Start tracking a run.

    Args:
        judges: every judge whose backend is called during the run.
        backends: extra ``(role, backend)`` pairs, e.g. ``("generation", backend)``.
            ``None`` entries are ignored. A backend object appearing more than once
            (for example one backend shared by several judges) is recorded once.
    """
    tracks: dict[int, _Track] = {}

    def _add(role: str, backend: object, judge_id: str | None) -> None:
        key = id(backend)
        track = tracks.get(key)
        if track is None:
            track = tracks[key] = _Track(backend=backend, role=role, before=backend_usage(backend))
        elif role == "judge" and track.role != "judge":
            track.role = "judge"       # a backend used by a judge is reported as a judge backend
        if judge_id is not None and judge_id not in track.judge_ids:
            track.judge_ids.append(judge_id)

    for judge in judges:
        backend = getattr(judge, "backend", None)    # custom judges may not expose it
        if backend is not None:
            _add("judge", backend, getattr(judge, "judge_id", None))
    for role, backend in backends:
        if backend is not None:
            _add(role, backend, None)

    return RunState(
        started_at=datetime.now(UTC),
        t0=time.perf_counter(),
        tracks=list(tracks.values()),
        judges=list(judges),
    )


def generation_truncations(state: RunState) -> int:
    """Truncated replies from backends that no judge used (judge truncations are on verdicts)."""
    total = 0
    for track in state.tracks:
        if track.role == "judge":
            continue
        after = backend_usage(track.backend)
        if after is not None and track.before is not None:
            total += after.minus(track.before).truncated
    return total


def end_run(state: RunState, settings: dict[str, Any] | None = None) -> RunMetadata:
    """Finish tracking and build the report's ``RunMetadata``."""
    finished = datetime.now(UTC)

    records: list[BackendRecord] = []
    labels: dict[int, str] = {}
    seen: dict[str, int] = {}
    for track in state.tracks:
        desc = describe_backend(track.backend)
        base = f"{desc.get('provider', 'backend')}:{desc.get('model') or 'unknown'}"
        seen[base] = seen.get(base, 0) + 1
        label = base if seen[base] == 1 else f"{base}#{seen[base]}"
        labels[id(track.backend)] = label

        usage: UsageSummary | None = None
        after = backend_usage(track.backend)
        if after is not None and track.before is not None:
            d = after.minus(track.before)
            usage = UsageSummary(
                calls=d.calls, input_tokens=d.input_tokens, output_tokens=d.output_tokens,
                truncated=d.truncated, cache_hits=d.cache_hits, errors=d.errors,
                latency_s=d.latency_s, calls_without_usage=d.calls_without_usage,
            )
        records.append(BackendRecord(
            role=track.role, label=label, judge_ids=list(track.judge_ids),
            description=desc, usage=usage,
        ))

    judge_records = [
        JudgeRecord(
            judge_id=str(getattr(j, "judge_id", type(j).__name__)),
            judge_class=type(j).__name__,
            temperature=_as_float(getattr(j, "temperature", None)),
            on_parse_failure=str(getattr(j, "on_parse_failure", "unknown")),
            n_calibration_examples=_as_int(getattr(j, "n_calibration_examples", 0)),
            backend_label=labels.get(id(getattr(j, "backend", None))),
        )
        for j in state.judges
    ]

    return RunMetadata(
        judge_kappa_version=_version("judge-kappa") or "unknown",
        python_version=platform.python_version(),
        packages={n: v for n in _TRACKED_PACKAGES if (v := _version(n)) is not None},
        started_at=state.started_at.isoformat(timespec="seconds"),
        finished_at=finished.isoformat(timespec="seconds"),
        duration_s=round(time.perf_counter() - state.t0, 3),
        judges=judge_records,
        backends=records,
        settings=settings or {},
    )
