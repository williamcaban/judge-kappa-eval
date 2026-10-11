"""
Order-aware Bradley-Terry-Luce (BTL) ranking with a display-order effect.

Model (Davidson & Beaver, 1977; Augustin, 2004), for items i != j shown in
display order a (+1 = i first, -1 = j first):

    logit P(i preferred over j | a) = s_i - s_j + a * b

``s`` are latent item scores (log-odds scale, centred to mean zero) and ``b`` is
a single judge-level order effect: ``b > 0`` means the judge favours whichever
response is displayed first. The order effect is separated from item quality
only when both display orders were observed (a cycle with a non-zero signed
order sum, see :func:`order_effect_identifiable`); otherwise ``b`` is fixed to
zero and reported as not identifiable.

This is the single-judge building block of the position-debiasing stage in DIAL
(arXiv:2609.31215). It deliberately omits DIAL's multi-judge low-rank structure
and human calibration; see ``docs/10-psychometric-methods.md``.

Unit of independence for resampling is the *case* (prompt): all games that
share a case id are resampled together, which preserves the dependence between
pairs judged on the same prompt.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

# 400 / ln(10): converts a natural-log-odds score difference to the Elo scale.
ELO_SCALE = 400.0 / math.log(10.0)
_SEPARATION_LOGIT = 10.0   # |score| beyond this means a (near-)separated fit


@dataclass(frozen=True)
class ComparisonGame:
    """One judged comparison between items ``i`` and ``j``.

    Attributes:
        case: Independence unit (prompt id) used for bootstrap resampling.
        i, j: Item indices, ``0 <= i, j < n_items`` and ``i != j``.
        order: ``+1`` if ``i`` was displayed first, ``-1`` if ``j`` was.
        outcome: ``1.0`` if ``i`` was preferred, ``0.0`` if ``j``, ``0.5`` for a tie.
    """

    case: int
    i: int
    j: int
    order: int
    outcome: float


@dataclass
class BTLFit:
    """Result of :func:`fit_btl`.

    Standard errors are Wald approximations from the observed information and
    ignore dependence across pairs sharing a case; prefer :func:`bootstrap_btl`
    intervals for reporting.
    """

    scores: list[float]
    scores_se: list[float]
    order_effect: float | None
    order_effect_se: float | None
    order_identifiable: bool
    converged: bool
    possibly_separated: bool
    n_games: int

    def elo(self, initial: float = 1000.0) -> list[float]:
        """Scores on the Elo scale, anchored so the mean rating equals ``initial``."""
        return [initial + ELO_SCALE * s for s in self.scores]


@dataclass
class BTLBootstrap:
    """Percentile bootstrap intervals from :func:`bootstrap_btl`."""

    ci_level: float
    n_boot: int
    scores_low: list[float]
    scores_high: list[float]
    order_effect_low: float | None
    order_effect_high: float | None


# ── Design helpers ────────────────────────────────────────────────────────────


@dataclass
class _Cells:
    """Games aggregated into (i, j, order) cells; arrays are length n_cells."""

    i: IntArray
    j: IntArray
    a: FloatArray
    n: FloatArray
    y: FloatArray


class _Design:
    """Pre-computed game arrays so bootstrap replicates re-aggregate cheaply."""

    def __init__(self, n_items: int, games: Sequence[ComparisonGame]) -> None:
        if n_items < 2:
            raise ValueError("n_items must be at least 2")
        if not games:
            raise ValueError("at least one game is required")
        gi = np.array([g.i for g in games], dtype=np.int64)
        gj = np.array([g.j for g in games], dtype=np.int64)
        ga = np.array([g.order for g in games], dtype=np.int64)
        if (gi == gj).any() or min(int(gi.min()), int(gj.min())) < 0 or max(int(gi.max()), int(gj.max())) >= n_items:
            raise ValueError("game item indices must be distinct and within [0, n_items)")
        if not np.isin(ga, (-1, 1)).all():
            raise ValueError("game order must be +1 or -1")
        outcome = np.array([g.outcome for g in games], dtype=np.float64)
        if (outcome < 0.0).any() or (outcome > 1.0).any():
            raise ValueError("game outcome must be in [0, 1]")

        # Canonicalise so i < j: swapping items flips the order sign and the outcome.
        swap = gi > gj
        self.i: IntArray = np.where(swap, gj, gi)
        self.j: IntArray = np.where(swap, gi, gj)
        self.a: IntArray = np.where(swap, -ga, ga)
        self.outcome: FloatArray = np.where(swap, 1.0 - outcome, outcome)

        raw_case = np.array([g.case for g in games], dtype=np.int64)
        self.case_ids, self.case = np.unique(raw_case, return_inverse=True)
        key = (self.i * n_items + self.j) * 2 + (self.a > 0).astype(np.int64)
        self._cell_keys, self._cell_of_game = np.unique(key, return_inverse=True)
        self.n_items = n_items

    @property
    def n_cases(self) -> int:
        return int(self.case_ids.size)

    def cells(self, weights: FloatArray | None = None) -> _Cells:
        w = np.ones(self.outcome.size) if weights is None else weights
        m = self._cell_keys.size
        n = np.bincount(self._cell_of_game, weights=w, minlength=m).astype(np.float64)
        y = np.bincount(self._cell_of_game, weights=w * self.outcome, minlength=m).astype(np.float64)
        pair_key, a_bit = np.divmod(self._cell_keys, 2)
        ci, cj = np.divmod(pair_key, self.n_items)
        keep = n > 0
        return _Cells(
            i=ci[keep].astype(np.int64),
            j=cj[keep].astype(np.int64),
            a=np.where(a_bit[keep] == 1, 1.0, -1.0),
            n=n[keep],
            y=y[keep],
        )


def _rank_ok(cells: _Cells, n_items: int, with_order: bool) -> bool:
    """Rank condition: [D, z] has rank N (with order) or D has rank N-1 (without)."""
    d = np.zeros((cells.i.size, n_items))
    d[np.arange(cells.i.size), cells.i] = 1.0
    d[np.arange(cells.i.size), cells.j] = -1.0
    if with_order:
        d = np.column_stack([d, cells.a])
        return bool(np.linalg.matrix_rank(d) == n_items)
    return bool(np.linalg.matrix_rank(d) == n_items - 1)


def order_effect_identifiable(n_items: int, games: Sequence[ComparisonGame]) -> bool:
    """True iff the order effect can be separated from item scores.

    Equivalent to: the comparison graph is connected and some cycle has a
    non-zero signed sum of display orders. A pair observed in both orders is the
    shortest such cycle. Games observed in one order only (for example a
    tournament run without swaps) are not identifiable.
    """
    return _rank_ok(_Design(n_items, games).cells(), n_items, with_order=True)


# ── Estimation ────────────────────────────────────────────────────────────────


def _solve(
    cells: _Cells,
    n_items: int,
    with_order: bool,
    l2: float,
    x0: FloatArray | None = None,
) -> tuple[FloatArray, float | None, bool]:
    """Penalised MLE. Returns (centred scores, order effect or None, converged)."""
    dim = n_items + (1 if with_order else 0)
    ci, cj, ca, cn, cy = cells.i, cells.j, cells.a, cells.n, cells.y

    def fun(theta: FloatArray) -> tuple[float, FloatArray]:
        s = theta[:n_items]
        eta = s[ci] - s[cj]
        if with_order:
            eta = eta + ca * theta[n_items]
        # n*log(1+e^eta) - y*eta, computed stably
        nll = float(np.sum(cn * np.logaddexp(0.0, eta) - cy * eta) + 0.5 * l2 * float(s @ s))
        resid = cn / (1.0 + np.exp(-eta)) - cy
        g = np.zeros(dim)
        np.add.at(g, ci, resid)
        np.add.at(g, cj, -resid)
        g[:n_items] += l2 * s
        if with_order:
            g[n_items] = float(np.sum(ca * resid))
        return nll, g

    start = np.zeros(dim) if x0 is None else x0
    res = minimize(fun, start, jac=True, method="L-BFGS-B", options={"maxiter": 500, "gtol": 1e-8})
    s_hat = np.asarray(res.x[:n_items], dtype=np.float64)
    s_hat = s_hat - float(s_hat.mean())
    b_hat = float(res.x[n_items]) if with_order else None
    return s_hat, b_hat, bool(res.success)


def fit_btl(
    n_items: int,
    games: Sequence[ComparisonGame],
    *,
    order_effect: bool = True,
    l2: float = 1e-4,
) -> BTLFit:
    """Fit an order-aware BTL model by penalised maximum likelihood.

    Args:
        n_items: Number of ranked items.
        games: Judged comparisons.
        order_effect: Estimate the display-order effect when identifiable. If
            ``False``, or if only one display order was observed, ``b`` is fixed
            to zero and ``order_identifiable`` reflects the design.
        l2: Ridge penalty on scores. It keeps the estimate finite under complete
            separation (one item always wins) and anchors the additive shift.

    Raises:
        ValueError: invalid indices/outcomes, or a disconnected comparison graph
            (scores of disconnected components are not comparable).
    """
    design = _Design(n_items, games)
    cells = design.cells()
    if not _rank_ok(cells, n_items, with_order=False):
        raise ValueError("comparison graph is not connected: every item must be linked to every other")
    identifiable = _rank_ok(cells, n_items, with_order=True)
    with_order = order_effect and identifiable
    s_hat, b_hat, converged = _solve(cells, n_items, with_order, l2)

    # Wald standard errors from the observed (Fisher) information.
    eta = s_hat[cells.i] - s_hat[cells.j] + (cells.a * b_hat if b_hat is not None else 0.0)
    p = 1.0 / (1.0 + np.exp(-eta))
    w = cells.n * p * (1.0 - p)
    dim = n_items + (1 if with_order else 0)
    x = np.zeros((cells.i.size, dim))
    x[np.arange(cells.i.size), cells.i] = 1.0
    x[np.arange(cells.i.size), cells.j] = -1.0
    if with_order:
        x[:, n_items] = cells.a
    info = x.T @ (x * w[:, None])
    info[:n_items, :n_items] += l2 * np.eye(n_items)
    cov = np.linalg.pinv(info)
    # Scores are only defined up to an additive shift; project the covariance onto the
    # centred subspace so the unidentified direction does not inflate the standard errors.
    proj = np.eye(dim)
    proj[:n_items, :n_items] -= 1.0 / n_items
    cov = proj @ cov @ proj.T
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))

    # Separation flag: an item that won (or lost) every game has no finite MLE, and so
    # does a judge whose order effect runs off. The ridge keeps the fit finite regardless.
    item_n = np.zeros(n_items)
    item_w = np.zeros(n_items)
    np.add.at(item_n, cells.i, cells.n)
    np.add.at(item_n, cells.j, cells.n)
    np.add.at(item_w, cells.i, cells.y)
    np.add.at(item_w, cells.j, cells.n - cells.y)
    separated = bool(
        np.any((item_w <= 0.0) | (item_w >= item_n))
        or np.max(np.abs(s_hat)) > _SEPARATION_LOGIT
        or (b_hat is not None and abs(b_hat) > _SEPARATION_LOGIT)
    )

    return BTLFit(
        scores=[float(v) for v in s_hat],
        scores_se=[float(v) for v in se[:n_items]],
        order_effect=b_hat,
        order_effect_se=float(se[n_items]) if with_order else None,
        order_identifiable=identifiable,
        converged=converged,
        possibly_separated=separated,
        n_games=len(games),
    )


def bootstrap_btl(
    n_items: int,
    games: Sequence[ComparisonGame],
    *,
    n_boot: int = 500,
    ci_level: float = 0.95,
    seed: int | None = 0,
    order_effect: bool = True,
    l2: float = 1e-4,
) -> BTLBootstrap:
    """Case-level percentile bootstrap for scores and the order effect.

    Cases (prompts) are resampled with replacement and every game of a drawn
    case is carried along, so pairs judged on the same prompt stay together.
    Replicates whose resampled design is disconnected are skipped; if fewer
    than half survive a ``ValueError`` is raised. The order-effect interval is
    reported only when at least ``max(2, n_boot // 2)`` replicates kept the order
    effect identifiable; otherwise both bounds are ``None``.
    """
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be in (0, 1)")
    design = _Design(n_items, games)
    if design.n_cases < 2:
        raise ValueError("bootstrap needs at least 2 distinct cases")
    base = design.cells()
    if not _rank_ok(base, n_items, with_order=False):
        raise ValueError("comparison graph is not connected: every item must be linked to every other")
    with_order = order_effect and _rank_ok(base, n_items, with_order=True)

    rng = np.random.default_rng(seed)
    s0, b0, _ = _solve(base, n_items, with_order, l2)
    x0 = np.concatenate([s0, [b0]]) if b0 is not None else s0

    score_draws: list[FloatArray] = []
    order_draws: list[float] = []
    for _ in range(n_boot):
        mult = np.bincount(rng.integers(0, design.n_cases, design.n_cases), minlength=design.n_cases)
        cells = design.cells(mult[design.case].astype(np.float64))
        if not _rank_ok(cells, n_items, with_order=False):
            continue
        rep_order = with_order and _rank_ok(cells, n_items, with_order=True)
        start = x0 if rep_order == with_order else s0
        s, b, _ = _solve(cells, n_items, rep_order, l2, start)
        score_draws.append(s)
        if b is not None:
            order_draws.append(b)
    if len(score_draws) < n_boot // 2:
        raise ValueError("too many bootstrap replicates had a disconnected comparison graph")

    # A replicate can stay connected yet lose order identifiability. Publish an order-effect
    # interval only from enough identifiable replicates; a single draw would give a degenerate one.
    publish_order = with_order and len(order_draws) >= max(2, n_boot // 2)

    alpha = (1.0 - ci_level) / 2.0
    stacked = np.vstack(score_draws)
    low = np.quantile(stacked, alpha, axis=0)
    high = np.quantile(stacked, 1.0 - alpha, axis=0)
    return BTLBootstrap(
        ci_level=ci_level,
        n_boot=len(score_draws),
        scores_low=[float(v) for v in low],
        scores_high=[float(v) for v in high],
        order_effect_low=float(np.quantile(order_draws, alpha)) if publish_order else None,
        order_effect_high=float(np.quantile(order_draws, 1.0 - alpha)) if publish_order else None,
    )
