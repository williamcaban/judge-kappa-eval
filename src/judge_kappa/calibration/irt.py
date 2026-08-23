"""
IRT-based judge weighting via 2PL model.

Motivation (Fonseca Rivera et al. 2026 — arXiv:2608.05086):
  The 2-Parameter Logistic (2PL) IRT model treats each calibration example
  as a test "item" and each judge as a "person". The judge's latent ability θ
  measures how reliably they agree with human-validated scores.

  P(judge i scores item j as correct | θ_i, a_j, b_j) = 1 / (1 + exp(-a_j(θ_i - b_j)))

  Where:
    θ_i = judge reliability (higher → more consistent with humans)
    a_j = item discrimination (higher → sharper signal about judge quality)
    b_j = item difficulty (higher → harder for any judge to score correctly)

Usage:
    from judge_kappa.calibration import IRTJudgeWeighter
    from judge_kappa.models import CalibrationExample

    examples = [CalibrationExample(prompt=..., output=..., score=0.8, rationale=...)]
    judge_scores = {
        "judge-a": [0.75, 0.82, 0.91],   # judge-a's score on each calibration example
        "judge-b": [0.60, 0.70, 0.85],
    }

    weighter = IRTJudgeWeighter()
    weighter.fit(judge_scores, [e.score for e in examples])
    weights = weighter.weights()          # {"judge-a": 0.63, "judge-b": 0.41}
    theta   = weighter.theta()            # {"judge-a": 1.2, "judge-b": -0.3}

Calibration protocol:
  1. Collect human-validated (prompt, output, score) triples as CalibrationExamples.
  2. Run each judge on the same examples and record their scores.
  3. Pass both to IRTJudgeWeighter.fit().
  4. Use weights() as JudgePanel weights or JudgeJury juror weights.

Implementation notes:
  - Binary "correct" criterion: |judge_score - human_score| ≤ tolerance (default 0.2).
  - 2PL MLE via scipy.optimize.minimize with L-BFGS-B.
  - Log-normal prior on a_j (μ=0, σ=1) + normal prior on b_j (μ=0, σ=2) for regularisation.
    Without priors, split-half discrimination retention drops from ~70% to ~30%.
  - θ is initialised from judge mean accuracy and optimised jointly with item params.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _neg_log_likelihood(
    params: np.ndarray,
    X: np.ndarray,
    n_judges: int,
    n_items: int,
    prior_a_sigma: float = 1.0,
    prior_b_sigma: float = 2.0,
) -> float:
    """
    Negative log-likelihood for 2PL model with log-normal prior on a and normal on b.

    params layout:
      [theta_0, ..., theta_{n_judges-1}, log_a_0, ..., log_a_{n_items-1}, b_0, ..., b_{n_items-1}]
    """
    thetas = params[:n_judges]
    log_as = params[n_judges: n_judges + n_items]
    bs     = params[n_judges + n_items:]

    a_vals = np.exp(log_as)  # ensures a > 0

    # Compute P_ij for all (i, j) pairs
    diff = thetas[:, None] - bs[None, :]     # (n_judges, n_items)
    logit = a_vals[None, :] * diff           # (n_judges, n_items)
    P = _sigmoid(logit)
    P = np.clip(P, 1e-7, 1 - 1e-7)

    # Mask missing entries (NaN in X)
    mask = ~np.isnan(X)
    X_obs = np.where(mask, X, 0.0)
    log_lik = np.sum(
        mask * (X_obs * np.log(P) + (1 - X_obs) * np.log(1 - P))
    )

    # Regularisation priors
    prior_a = -np.sum(log_as ** 2) / (2 * prior_a_sigma ** 2)
    prior_b = -np.sum(bs ** 2) / (2 * prior_b_sigma ** 2)

    return float(-(log_lik + prior_a + prior_b))


class IRTJudgeWeighter:
    """
    Fit a 2PL IRT model to calibration data and derive per-judge reliability weights.

    Args:
        tolerance: Score agreement threshold. |judge_score - human_score| ≤ tolerance
            counts as "correct" for the binary IRT response matrix. Default 0.2.
        prior_a_sigma: Log-normal prior SD on item discrimination. Default 1.0.
        prior_b_sigma: Normal prior SD on item difficulty. Default 2.0.
        max_iter: scipy L-BFGS-B iteration limit. Default 1000.
    """

    def __init__(
        self,
        tolerance: float = 0.2,
        prior_a_sigma: float = 1.0,
        prior_b_sigma: float = 2.0,
        max_iter: int = 1000,
    ) -> None:
        self._tolerance = tolerance
        self._prior_a = prior_a_sigma
        self._prior_b = prior_b_sigma
        self._max_iter = max_iter

        self._theta: dict[str, float] = {}
        self._item_a: list[float] = []
        self._item_b: list[float] = []
        self._judge_names: list[str] = []
        self._fitted = False

    def fit(
        self,
        judge_scores: dict[str, list[float]],
        human_scores: list[float],
    ) -> IRTJudgeWeighter:
        """
        Fit the 2PL model.

        Args:
            judge_scores: {judge_id: [score_on_item_0, score_on_item_1, ...]}
                All lists must have the same length as human_scores.
            human_scores: Ground-truth human scores for each calibration example.

        Returns self for chaining.
        """
        self._judge_names = sorted(judge_scores.keys())
        n_judges = len(self._judge_names)
        n_items  = len(human_scores)

        if n_items < 5:
            raise ValueError(f"Need ≥ 5 calibration examples for IRT; got {n_items}.")
        if n_judges < 2:
            raise ValueError("Need ≥ 2 judges for IRT fitting.")

        human = np.array(human_scores, dtype=float)

        # Build binary response matrix X (n_judges, n_items); NaN for missing
        X = np.full((n_judges, n_items), np.nan)
        for i, name in enumerate(self._judge_names):
            scores = np.array(judge_scores[name], dtype=float)
            X[i] = (np.abs(scores - human) <= self._tolerance).astype(float)

        # Initialise: theta from per-judge accuracy, log_a = 0, b from item difficulty
        judge_acc = np.nanmean(X, axis=1)
        item_diff = 1.0 - np.nanmean(X, axis=0)

        theta_init = np.clip(judge_acc * 2 - 1, -2.0, 2.0)
        log_a_init = np.zeros(n_items)
        b_init     = np.clip(item_diff * 4 - 2, -3.0, 3.0)

        x0 = np.concatenate([theta_init, log_a_init, b_init])

        result = minimize(
            _neg_log_likelihood,
            x0,
            args=(X, n_judges, n_items, self._prior_a, self._prior_b),
            method="L-BFGS-B",
            options={"maxiter": self._max_iter, "ftol": 1e-8},
        )

        opt = result.x
        thetas = opt[:n_judges]
        log_as = opt[n_judges: n_judges + n_items]
        bs     = opt[n_judges + n_items:]

        self._theta = dict(zip(self._judge_names, thetas.tolist(), strict=True))
        self._item_a = np.exp(log_as).tolist()
        self._item_b = bs.tolist()
        self._fitted = True
        return self

    def theta(self) -> dict[str, float]:
        """Return per-judge latent reliability (θ). Higher = more reliable."""
        self._check_fitted()
        return {k: round(v, 4) for k, v in self._theta.items()}

    def weights(self, normalise: bool = True) -> dict[str, float]:
        """
        Convert θ values to panel weights via softmax.

        Args:
            normalise: If True, weights sum to 1.0. If False, returns raw softmax values.

        Returns dict[judge_id → weight].
        """
        self._check_fitted()
        thetas = np.array([self._theta[n] for n in self._judge_names])
        # Softmax ensures positive weights that sum to 1
        exp_t = np.exp(thetas - thetas.max())
        softmax = exp_t / exp_t.sum()
        if not normalise:
            return dict(zip(self._judge_names, softmax.tolist(), strict=True))
        return {n: round(float(w), 4) for n, w in zip(self._judge_names, softmax, strict=True)}

    def item_parameters(self) -> list[dict[str, float]]:
        """Return list of {a: discrimination, b: difficulty} per calibration example."""
        self._check_fitted()
        return [
            {"a": round(a, 4), "b": round(b, 4)}
            for a, b in zip(self._item_a, self._item_b, strict=True)
        ]

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() before accessing results.")
