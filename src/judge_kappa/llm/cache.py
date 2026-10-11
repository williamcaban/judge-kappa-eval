"""
CachedBackend — a persistent response cache that also serves as checkpoint/resume.

Wrap any backend; every distinct request is stored in an append-only JSONL file as
soon as it completes. Re-running the same evaluation reads stored replies instead of
calling the model, so:

  * a tournament or evaluation that died halfway resumes where it stopped, paying only
    for the calls that never finished;
  * a re-run is exactly reproducible (the same replies feed the same statistics);
  * tuning a downstream statistic (bootstrap size, bias thresholds) costs nothing.

The cache key covers everything that changes the reply: the backend's secret-free
description (provider, model, base URL, max_tokens, omit_temperature), an optional
namespace, the system and user prompts, and the temperature. Operational settings
(timeout, retries) are not part of the key.

The file stores prompts and replies in plain text and is created ``0600``. Treat it
like the data it contains.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from judge_kappa.llm.base import LLMBackend, call_backend, describe_backend
from judge_kappa.llm.types import LLMResponse, UsageStats

_VOLATILE_KEYS = ("timeout_s", "max_retries")   # operational, do not change the reply
_FORMAT_VERSION = 1


class CachedBackend:
    def __init__(
        self,
        inner: LLMBackend,
        path: str | Path,
        *,
        namespace: str | None = None,
    ) -> None:
        """
        Args:
            inner: the backend to cache.
            path: JSONL cache file (created if missing). Several backends may share one
                file; their descriptions keep their entries apart.
            namespace: extra string folded into every key, to separate runs that must
                not share replies even with identical requests. **Required** when ``inner`` has no
                ``describe()``.
        """
        if namespace is None and not callable(getattr(inner, "describe", None)):
            raise ValueError(
                "CachedBackend needs a `namespace` when the wrapped backend has no describe(): "
                "without it, two different backends of the same class could share cache entries "
                "and be served each other's replies."
            )
        self._inner = inner
        self._path = Path(path)
        self._namespace = namespace
        self._entries: dict[str, LLMResponse] = {}
        self._lock = threading.Lock()
        self.stats = UsageStats()
        self.skipped_lines = 0
        self._load()

    # ── introspection ───────────────────────────────────────────────────────────

    @property
    def inner(self) -> LLMBackend:
        return self._inner

    @property
    def path(self) -> Path:
        return self._path

    def __len__(self) -> int:
        return len(self._entries)

    def describe(self) -> dict[str, Any]:
        d = dict(describe_backend(self._inner))
        d["cached"] = True
        d["cache_file"] = self._path.name       # name only: do not leak local directory layout
        d["cache_entries"] = len(self._entries)
        return d

    # ── backend protocol ──────────────────────────────────────────────────────

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self.complete_with_usage(system, user, temperature).text

    def complete_with_usage(
        self, system: str, user: str, temperature: float = 0.0
    ) -> LLMResponse:
        key = self._key(system, user, temperature)
        with self._lock:
            hit = self._entries.get(key)
        if hit is not None:
            served = replace(hit, cached=True)
            self.stats.record(served)
            return served

        try:
            resp = call_backend(self._inner, system, user, temperature)   # outside the lock
        except Exception:
            self.stats.record_error()
            raise
        self._store(key, resp)
        self.stats.record(resp)
        return resp

    # ── persistence ───────────────────────────────────────────────────────────

    def _key(self, system: str, user: str, temperature: float) -> str:
        desc = {k: v for k, v in describe_backend(self._inner).items() if k not in _VOLATILE_KEYS}
        payload = json.dumps(
            {"b": desc, "ns": self._namespace, "s": system, "u": user, "t": float(temperature)},
            sort_keys=True, ensure_ascii=False, default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if not self._path.exists():
            return
        with self._path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._entries[rec["key"]] = LLMResponse(
                        text=rec["text"],
                        input_tokens=rec.get("input_tokens"),
                        output_tokens=rec.get("output_tokens"),
                        finish_reason=rec.get("finish_reason"),
                        model=rec.get("model"),
                        latency_s=rec.get("latency_s"),
                    )
                except (json.JSONDecodeError, KeyError, TypeError):
                    self.skipped_lines += 1     # e.g. a line cut short by a crash

    def _store(self, key: str, resp: LLMResponse) -> None:
        record = {
            "v": _FORMAT_VERSION,
            "key": key,
            "text": resp.text,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "finish_reason": resp.finish_reason,
            "model": resp.model,
            "latency_s": resp.latency_s,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            self._entries[key] = resp
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self._path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, line.encode("utf-8"))   # one write per record keeps lines whole
            finally:
                os.close(fd)


__all__ = ["CachedBackend"]
