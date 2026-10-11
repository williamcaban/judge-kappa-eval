"""
Opt-in concurrency for the orchestrators.

Judge calls are I/O-bound, so a thread pool is enough. The contract that matters for a
measurement library is that **results do not depend on scheduling**: ``ordered_map``
returns results in input order, and the callers aggregate in that order. With
``max_workers=1`` (the default everywhere) no thread is created and behaviour is exactly
the sequential behaviour.

Callables passed in by users (``Variant.predict_fn``, custom backends) must be
thread-safe when ``max_workers > 1``. The built-in backends are.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor


def validate_max_workers(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("max_workers must be an integer >= 1")
    return value


def ordered_map[T, R](fn: Callable[[T], R], items: Iterable[T], max_workers: int = 1) -> list[R]:
    """Apply ``fn`` to every item and return the results in input order.

    The first exception raised by ``fn`` propagates; queued work is cancelled (work
    already running finishes first). ``max_workers <= 1`` runs inline.
    """
    seq = list(items)
    if max_workers <= 1 or len(seq) <= 1:
        return [fn(x) for x in seq]
    with ThreadPoolExecutor(
        max_workers=min(max_workers, len(seq)), thread_name_prefix="judge-kappa"
    ) as pool:
        futures = [pool.submit(fn, x) for x in seq]
        try:
            return [f.result() for f in futures]
        except BaseException:
            for f in futures:
                f.cancel()
            raise
