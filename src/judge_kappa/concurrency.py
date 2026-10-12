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
from concurrent.futures import ThreadPoolExecutor, as_completed


def validate_max_workers(value: int) -> int:
    """Return ``value`` if it is an integer >= 1; otherwise raise ``ValueError``."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("max_workers must be an integer >= 1")
    return value


def ordered_map[T, R](fn: Callable[[T], R], items: Iterable[T], max_workers: int = 1) -> list[R]:
    """Apply ``fn`` to every item and return the results in input order.

    The first failure, in *completion* order, propagates immediately: queued work is
    cancelled and the call does not wait for slower, earlier items (calls already in
    flight finish in the background). ``max_workers <= 1`` runs inline.
    """
    seq = list(items)
    if max_workers <= 1 or len(seq) <= 1:
        return [fn(x) for x in seq]

    pool = ThreadPoolExecutor(
        max_workers=min(max_workers, len(seq)), thread_name_prefix="judge-kappa"
    )
    futures = {pool.submit(fn, x): i for i, x in enumerate(seq)}
    results: list[R | None] = [None] * len(seq)
    try:
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return results  # type: ignore[return-value]
