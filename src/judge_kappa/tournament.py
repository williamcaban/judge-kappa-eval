"""
TournamentEvaluator — compare N evaluation targets using round-robin pairwise scoring.

Design:
  - Runs every combination of (system_i, system_j) through a PairwiseJudge.
  - Produces a win matrix [N × N] and Elo-scale ratings derived from it.
  - Returns TournamentReport: ranked leaderboard, Elo scores, win matrix,
    per-pair preference rates, and positional bias rate per pair.

Position handling (when detect_positional_bias=True, the default):
  - Every pair is judged in both display orders. Win/loss/tie counts and the win
    matrix use per-case scores averaged over the two orders (Balanced Position
    Calibration, Wang et al. 2024), so a judge that always prefers the first
    response yields ties instead of a spurious win for whichever system was shown
    first.
  - Ratings are the maximum-likelihood scores of an order-aware Bradley-Terry
    model (see judge_kappa.ranking), which also estimates the judge's order effect
    and yields bootstrap confidence intervals (cases are the resampling unit).
  - rating_method="elo" restores the v0.2 sequential-Elo ratings.

When to use:
  - 3–6 systems: full round-robin is tractable; all-pairs Elo is most accurate.
  - ≥ 7 systems: consider a RankJudge (listwise) or champion-challenger design.

Call cost: C(N, 2) × cases × 2 (each pair is run in both orders for bias detection)
  e.g. 4 systems × 20 cases × 2 = 240 judge calls (not counting retries)

Entry points:
  TournamentEvaluator.run(cases, systems)
    → evaluates all pairs across EvalCase list

  TournamentEvaluator.run_dataset(data, systems)
    → loads cases from a list of dicts (same format as DatasetAdapter)

  TournamentEvaluator.run_endpoints(cases, endpoint_variants)
    → generates outputs from each Variant's endpoint first, then runs tournament
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

from judge_kappa.concurrency import ordered_map, validate_max_workers
from judge_kappa.judges.pairwise import PairwiseJudge
from judge_kappa.llm.base import LLMBackend, call_backend
from judge_kappa.models import DataQuality, EvalCase, RunMetadata, Variant
from judge_kappa.provenance import RunState, begin_run, end_run, generation_truncations
from judge_kappa.quality import ReplyFlag, compute_data_quality
from judge_kappa.ranking.btl import ELO_SCALE, ComparisonGame, bootstrap_btl, fit_btl

# ── Output models ─────────────────────────────────────────────────────────────


@dataclass
class PairResult:
    system_a: str
    system_b: str
    preference_rate_a: float   # fraction of cases A won
    preference_rate_b: float
    tie_rate: float
    mean_score_a: float
    mean_score_b: float
    positional_bias_rate: float   # rate of flips across both orderings


@dataclass
class TournamentStanding:
    system: str
    elo: float
    wins: int
    losses: int
    ties: int
    win_rate: float
    # Order-aware BTL score (centred log-odds) and bootstrap interval; None if not computed.
    btl_score: float | None = None
    btl_ci_low: float | None = None
    btl_ci_high: float | None = None


@dataclass
class TournamentReport:
    standings: list[TournamentStanding]   # sorted by Elo descending
    pair_results: list[PairResult]
    win_matrix: dict[str, dict[str, float]]   # win_matrix[a][b] = rate A beat B
    n_cases: int
    judge_id: str
    rating_method: str = "elo"                 # "btl" or "elo": how `elo` was computed
    # Judge-level display-order effect (log-odds; > 0 favours the first-shown response).
    order_effect: float | None = None
    order_effect_ci: tuple[float, float] | None = None
    order_identifiable: bool = False           # False when pairs were judged in one order only
    possibly_separated: bool = False           # estimates at the ridge bound: ratings/intervals unreliable
    ci_level: float | None = None
    # Unusable judge replies (parse_failed) and truncations; see judge_kappa.quality.
    data_quality: DataQuality = field(default_factory=DataQuality)
    run: RunMetadata | None = None   # provenance: configuration, versions and usage

    def print_leaderboard(self) -> None:
        has_ci = any(s.btl_ci_low is not None for s in self.standings)
        ci_head = f" {f'{self.ci_level:.0%} CI (Elo)' if self.ci_level else 'CI (Elo)':>18}" if has_ci else ""
        print(f"{'Rank':<5} {'System':<30} {'Elo':>7} {'Win%':>6} {'W':>4} {'L':>4} {'T':>4}{ci_head}")
        print("─" * (65 + (19 if has_ci else 0)))
        for i, s in enumerate(self.standings, 1):
            ci = ""
            if has_ci and s.btl_ci_low is not None and s.btl_ci_high is not None:
                lo, hi = 1000.0 + ELO_SCALE * s.btl_ci_low, 1000.0 + ELO_SCALE * s.btl_ci_high
                ci = f" {f'[{lo:.0f}, {hi:.0f}]':>18}"
            print(f"{i:<5} {s.system:<30} {s.elo:>7.1f} {s.win_rate:>6.1%}"
                  f" {s.wins:>4} {s.losses:>4} {s.ties:>4}{ci}")
        if self.order_effect is not None:
            ci_txt = (
                f" [{self.order_effect_ci[0]:+.2f}, {self.order_effect_ci[1]:+.2f}]"
                if self.order_effect_ci else ""
            )
            print(f"Judge order effect (log-odds, + favours first-shown): {self.order_effect:+.2f}{ci_txt}")
        elif not self.order_identifiable:
            print("Judge order effect: not identifiable (pairs were judged in one display order only)")
        for w in self.data_quality.warnings:
            print(f"Warning: {w}")
        if self.possibly_separated:
            print("Warning: estimates are at the regularisation bound (a system won or lost every game, "
                  "or the order effect is extreme); ratings and intervals are not reliable.")


# ── Elo calculation ───────────────────────────────────────────────────────────


def _compute_elo(
    win_matrix: dict[str, dict[str, float]],
    systems: list[str],
    k: float = 32.0,
    initial: float = 1000.0,
    max_iterations: int = 1000,
    tol: float = 1e-4,
) -> dict[str, float]:
    """
    Batch Elo estimation from a win-rate matrix via iterative updates.

    Each off-diagonal cell win_matrix[a][b] is the observed win probability of
    A over B across all cases. The algorithm applies repeated Elo updates
    (treating each pair as a single virtual match per iteration) until ratings
    converge within `tol` Elo points or `max_iterations` is reached.

    Convergence note:
      - For N ≤ 6 systems and M ≥ 10 cases, ratings typically converge in
        < 200 iterations (max change < tol).
      - For highly skewed win rates (one system dominates all others), the
        logistic model pushes ratings far apart; K-factor damping prevents
        divergence.
      - Ties are treated as 0.5 wins for each side (standard Elo practice).

    The returned ratings are on the Elo scale anchored at `initial` (default
    1000). A 100-point gap implies ~64% win probability for the higher-rated
    system; a 200-point gap implies ~76%.
    """
    ratings = {s: initial for s in systems}
    pairs = list(itertools.combinations(systems, 2))

    for _ in range(max_iterations):
        prev = dict(ratings)
        for a, b in pairs:
            rate_a = win_matrix[a].get(b, 0.5)
            rate_b = win_matrix[b].get(a, 0.5)
            expected_a = 1.0 / (1.0 + 10.0 ** ((ratings[b] - ratings[a]) / 400.0))
            expected_b = 1.0 - expected_a
            ratings[a] += k * (rate_a - expected_a)
            ratings[b] += k * (rate_b - expected_b)

        max_delta = max(abs(ratings[s] - prev[s]) for s in systems)
        if max_delta < tol:
            break

    return ratings


# ── Main evaluator ────────────────────────────────────────────────────────────


def _outcome(first_score: float, second_score: float) -> float:
    """1.0 if the first score wins, 0.0 if the second does, 0.5 on a tie."""
    if first_score > second_score:
        return 1.0
    if first_score < second_score:
        return 0.0
    return 0.5


@dataclass
class _PairCase:
    """Result of judging one (system pair, case): everything the aggregator needs."""

    a: str
    b: str
    games: list[ComparisonGame]
    scores: tuple[float, float] | None   # order-balanced (a, b) scores; None if nothing usable
    flip: bool | None                    # first-position win in both rounds; None if untestable
    replies: list[ReplyFlag]


class TournamentEvaluator:
    """
    Compare N evaluation targets (systems) using round-robin pairwise scoring.

    Args:
        pairwise_judge: A configured PairwiseJudge instance.
        generation_backend: Used to generate outputs from Variant objects.
            Not required if outputs are pre-recorded (run() with pre-set outputs).
        detect_positional_bias: Run each pair in both orders (A,B) and (B,A).
            Doubles judge calls, reports per-pair positional bias rate, and makes
            win counts/ratings order-balanced (see module docstring).
        rating_method: "btl" (default) derives Elo-scale ratings from an
            order-aware Bradley-Terry maximum-likelihood fit; "elo" reproduces the
            v0.2 sequential-Elo update on the win matrix.
        n_bootstrap: Case-level bootstrap replicates for rating intervals and the
            order-effect interval. 0 disables intervals. Skipped (no error) when
            fewer than 2 cases were run.
        bootstrap_seed: RNG seed for reproducible intervals.
        ci_level: Interval level, in (0, 1).
        max_workers: (pair, case) games judged concurrently (threads). 1, the default, is
            strictly sequential. Ratings and counts are identical for any value: games
            are aggregated in a fixed order. Custom backends must be thread-safe above 1.
        exclude_parse_failures: When True, a judge reply that could not be parsed is
            dropped instead of entering the ratings as the 0.5/0.5 fallback tie. A
            case with no usable round for a pair contributes nothing for that pair.
            Default False keeps the historical behaviour; failures are counted in
            ``TournamentReport.data_quality`` either way.
    """

    def __init__(
        self,
        pairwise_judge: PairwiseJudge,
        generation_backend: LLMBackend | None = None,
        detect_positional_bias: bool = True,
        rating_method: Literal["btl", "elo"] = "btl",
        n_bootstrap: int = 500,
        bootstrap_seed: int | None = 0,
        ci_level: float = 0.95,
        exclude_parse_failures: bool = False,
        max_workers: int = 1,
    ) -> None:
        if rating_method not in ("btl", "elo"):
            raise ValueError("rating_method must be 'btl' or 'elo'")
        if n_bootstrap < 0 or n_bootstrap == 1:
            raise ValueError("n_bootstrap must be 0 (disabled) or at least 2")
        if not 0.0 < ci_level < 1.0:
            raise ValueError("ci_level must be in (0, 1)")
        self._judge = pairwise_judge
        self._gen_backend = generation_backend
        self._detect_positional_bias = detect_positional_bias
        self._rating_method = rating_method
        self._n_bootstrap = n_bootstrap
        self._bootstrap_seed = bootstrap_seed
        self._ci_level = ci_level
        self._exclude_parse_failures = exclude_parse_failures
        self._max_workers = validate_max_workers(max_workers)

    def run(
        self,
        cases: list[EvalCase],
        systems: Mapping[str, str | Callable[..., str]],
    ) -> TournamentReport:
        """
        Run a tournament.

        Args:
            cases: EvalCase list (from SkillAdapter, DatasetAdapter, or hand-built).
            systems: dict mapping system_name → either:
                     - a pre-recorded output string (same output for all cases), or
                     - a callable(inputs_dict) → str that generates per-case output.

        To compare live endpoints, use run_endpoints() instead.
        """
        system_names = list(systems.keys())
        n = len(system_names)
        if n < 2:
            raise ValueError("Tournament requires at least 2 systems.")
        run_state = begin_run(judges=[self._judge], backends=[("generation", self._gen_backend)])

        # Generate outputs per system per case
        outputs: dict[str, list[str]] = {name: [] for name in system_names}
        for case in cases:
            inputs = case.metadata.get("inputs", {"prompt": case.prompt})
            for name, fn_or_str in systems.items():
                if callable(fn_or_str):
                    outputs[name].append(fn_or_str(inputs))
                else:
                    outputs[name].append(str(fn_or_str))

        return self._run_tournament(cases, system_names, outputs, run_state)

    def run_endpoints(
        self,
        cases: list[EvalCase],
        variants: list[Variant],
    ) -> TournamentReport:
        """
        Compare N live endpoints. Each Variant generates its own outputs.

        Args:
            cases: EvalCase list.
            variants: List of Variant objects. Each must have either a
                      predict_fn or a generation_backend set.
        """
        if self._gen_backend is None and any(
            v.predict_fn is None and v.generation_backend is None for v in variants
        ):
            raise ValueError(
                "TournamentEvaluator requires a generation_backend when variants "
                "do not have their own predict_fn or generation_backend."
            )

        system_names = [v.name for v in variants]
        outputs: dict[str, list[str]] = {v.name: [] for v in variants}
        run_state = begin_run(
            judges=[self._judge],
            backends=[
                ("generation", self._gen_backend),
                *(("generation", v.generation_backend) for v in variants),
            ],
        )

        for case in cases:
            inputs = case.metadata.get("inputs", {"prompt": case.prompt})
            for variant in variants:
                if variant.predict_fn is not None:
                    outputs[variant.name].append(variant.predict_fn(inputs))
                else:
                    backend = variant.generation_backend or self._gen_backend
                    parts: list[str] = []
                    if variant.system_prompt:
                        parts.append(variant.system_prompt)
                    if variant.skill_context:
                        parts.append(f"## Skill Instructions\n{variant.skill_context}")
                    system_msg = "\n\n".join(parts) or "You are a helpful assistant."
                    outputs[variant.name].append(
                        call_backend(
                            backend,  # type: ignore[arg-type]
                            system_msg, case.prompt, variant.temperature,
                        ).text
                    )

        return self._run_tournament(cases, system_names, outputs, run_state)

    def run_dataset(
        self,
        data: list[dict[str, object]],
        systems: Mapping[str, Callable[..., str]],
    ) -> TournamentReport:
        """
        Convenience wrapper: loads cases from a list of dicts then runs tournament.
        """
        from judge_kappa.adapters.dataset import DatasetAdapter
        cases = DatasetAdapter().load(data)
        return self.run(cases, systems)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _play(
        self,
        a: str,
        b: str,
        k: int,
        case: EvalCase,
        out_a: str,
        out_b: str,
        index: dict[str, int],
    ) -> _PairCase:
        """Judge one (pair, case) in one or both display orders. Pure: no shared state."""
        games: list[ComparisonGame] = []
        rounds: list[tuple[float, float]] = []
        replies: list[ReplyFlag] = []
        skip_failed = self._exclude_parse_failures

        # Round 1: A first
        va, vb = self._judge.judge_pair(case, out_a, out_b, a, b)
        replies.append((self._judge.judge_id, va.parse_error, va.truncated))
        r1_ok = not (skip_failed and va.parse_failed)
        if r1_ok:
            games.append(ComparisonGame(k, index[a], index[b], 1, _outcome(va.score, vb.score)))
            rounds.append((va.score, vb.score))

        flip: bool | None = None
        if self._detect_positional_bias:
            # Round 2: B first (swapped)
            vb2, va2 = self._judge.judge_pair(case, out_b, out_a, b, a)
            replies.append((self._judge.judge_id, va2.parse_error, va2.truncated))
            r2_ok = not (skip_failed and va2.parse_failed)
            if r2_ok:
                games.append(ComparisonGame(k, index[a], index[b], -1, _outcome(va2.score, vb2.score)))
                rounds.append((va2.score, vb2.score))
            if r1_ok and r2_ok:
                # Positional bias = the first-positioned output wins in both rounds.
                flip = va.score > vb.score and vb2.score > va2.score

        scores: tuple[float, float] | None = None
        if rounds:
            # Balanced Position Calibration: average each system's score over both display
            # orders so a first-position preference cancels out.
            scores = (
                sum(r[0] for r in rounds) / len(rounds),
                sum(r[1] for r in rounds) / len(rounds),
            )
        return _PairCase(a=a, b=b, games=games, scores=scores, flip=flip, replies=replies)

    def _run_tournament(
        self,
        cases: list[EvalCase],
        system_names: list[str],
        outputs: dict[str, list[str]],
        run_state: RunState,
    ) -> TournamentReport:
        """Core: run all-pairs over all cases, compute win matrix and Elo."""

        # win_counts[a][b] = number of cases where A beat B
        wins:  dict[str, dict[str, int]] = {n: {m: 0 for m in system_names} for n in system_names}
        losses:dict[str, dict[str, int]] = {n: {m: 0 for m in system_names} for n in system_names}
        ties:  dict[str, dict[str, int]] = {n: {m: 0 for m in system_names} for n in system_names}
        score_sums: dict[str, dict[str, float]] = {
            n: {m: 0.0 for m in system_names} for n in system_names
        }
        pos_flips: dict[tuple[str, str], int] = {}
        pair_totals: dict[tuple[str, str], int] = {}
        games: list[ComparisonGame] = []
        replies: list[ReplyFlag] = []
        index = {name: k for k, name in enumerate(system_names)}

        tasks: list[tuple[str, str, int]] = []
        for i, a in enumerate(system_names):
            for b in system_names[i + 1:]:
                pos_flips[(a, b)] = 0
                pair_totals[(a, b)] = 0
                tasks.extend((a, b, k) for k in range(len(cases)))

        results = ordered_map(
            lambda t: self._play(t[0], t[1], t[2], cases[t[2]], outputs[t[0]][t[2]], outputs[t[1]][t[2]], index),
            tasks,
            self._max_workers,
        )

        # Aggregate in task order so the report does not depend on how calls were scheduled.
        for res in results:
            a, b = res.a, res.b
            games.extend(res.games)
            replies.extend(res.replies)
            if res.scores is None:
                continue                      # no usable round for this (pair, case)
            score_a, score_b = res.scores
            if res.flip is not None:
                pair_totals[(a, b)] += 1
                if res.flip:
                    pos_flips[(a, b)] += 1

            score_sums[a][b] += score_a
            score_sums[b][a] += score_b
            if score_a > score_b:
                wins[a][b] += 1
                losses[b][a] += 1
            elif score_b > score_a:
                wins[b][a] += 1
                losses[a][b] += 1
            else:
                ties[a][b] += 1
                ties[b][a] += 1

        n_cases = len(cases)

        # Build win-rate matrix
        win_matrix: dict[str, dict[str, float]] = {n: {} for n in system_names}
        pair_results: list[PairResult] = []

        for i, a in enumerate(system_names):
            for b in system_names[i + 1:]:
                total = wins[a][b] + wins[b][a] + ties[a][b]
                usable = total > 0
                if total == 0:
                    total = 1
                rate_a = (wins[a][b] + 0.5 * ties[a][b]) / total
                rate_b = (wins[b][a] + 0.5 * ties[b][a]) / total

                win_matrix[a][b] = rate_a
                win_matrix[b][a] = rate_b

                pkey = (a, b)
                pos_rate = (
                    pos_flips[pkey] / pair_totals[pkey]
                    if pair_totals.get(pkey, 0) > 0 else 0.0
                )
                pair_results.append(PairResult(
                    system_a=a, system_b=b,
                    preference_rate_a=round(rate_a, 4),
                    preference_rate_b=round(rate_b, 4),
                    tie_rate=round(ties[a][b] / total, 4),
                    # Mean over the cases that produced a usable round for this pair
                    # (all cases unless exclude_parse_failures dropped some).
                    mean_score_a=round(score_sums[a][b] / total, 4) if usable else 0.0,
                    mean_score_b=round(score_sums[b][a] / total, 4) if usable else 0.0,
                    positional_bias_rate=round(pos_rate, 4),
                ))

        # The order effect is only reported when every pair was judged in both display
        # orders. Without swaps it is identified solely through intransitive cycles, which
        # is model-dependent and weak, so it is fixed at zero and flagged as not identifiable.
        swap_based = self._detect_positional_bias
        try:
            btl = fit_btl(len(system_names), games, order_effect=swap_based)
        except ValueError as exc:
            seen = {g.i for g in games} | {g.j for g in games}
            unrated = [name for name in system_names if index[name] not in seen]
            raise ValueError(
                f"cannot rate the systems: {exc}. "
                f"{sum(1 for r in replies if r[1] is not None)} judge replies were unusable"
                f"{' and excluded' if self._exclude_parse_failures else ''}"
                + (f"; no usable game for: {', '.join(unrated)}." if unrated else ".")
            ) from exc
        boot = None
        # Resample over the cases that actually produced a game, not the input case count:
        # exclusion can leave fewer than two.
        usable_cases = len({g.case for g in games})
        if self._n_bootstrap >= 2 and usable_cases >= 2:
            boot = bootstrap_btl(
                len(system_names), games,
                n_boot=self._n_bootstrap, ci_level=self._ci_level, seed=self._bootstrap_seed,
                order_effect=swap_based,
            )
        if self._rating_method == "btl":
            elo_ratings = {name: 1000.0 + ELO_SCALE * btl.scores[index[name]] for name in system_names}
        else:
            elo_ratings = _compute_elo(win_matrix, system_names)

        standings: list[TournamentStanding] = []
        for name in system_names:
            w = sum(wins[name][o] for o in system_names if o != name)
            ls = sum(losses[name][o] for o in system_names if o != name)
            t = sum(ties[name][o] for o in system_names if o != name)
            total_games = w + ls + t
            standings.append(TournamentStanding(
                system=name,
                elo=round(elo_ratings[name], 1),
                wins=w, losses=ls, ties=t,
                win_rate=round(w / total_games, 4) if total_games else 0.0,
                btl_score=round(btl.scores[index[name]], 4),
                btl_ci_low=round(boot.scores_low[index[name]], 4) if boot else None,
                btl_ci_high=round(boot.scores_high[index[name]], 4) if boot else None,
            ))

        standings.sort(key=lambda s: s.elo, reverse=True)

        quality = compute_data_quality(
            replies, extra_truncated=generation_truncations(run_state)
        )
        if self._n_bootstrap >= 2 and boot is None:
            quality.warnings.append(
                f"Confidence intervals omitted: only {usable_cases} case(s) produced a usable "
                "game, and the bootstrap needs at least 2."
            )

        return TournamentReport(
            standings=standings,
            pair_results=pair_results,
            win_matrix=win_matrix,
            n_cases=n_cases,
            judge_id=self._judge.judge_id,
            rating_method=self._rating_method,
            order_effect=round(btl.order_effect, 4) if btl.order_effect is not None else None,
            order_effect_ci=(
                (round(boot.order_effect_low, 4), round(boot.order_effect_high, 4))
                if boot and boot.order_effect_low is not None and boot.order_effect_high is not None
                else None
            ),
            order_identifiable=swap_based and btl.order_identifiable,
            possibly_separated=btl.possibly_separated,
            ci_level=self._ci_level if boot else None,
            data_quality=quality,
            run=end_run(run_state, settings={
                "mode": "tournament",
                "n_systems": len(system_names),
                "n_cases": n_cases,
                "rating_method": self._rating_method,
                "detect_positional_bias": self._detect_positional_bias,
                "n_bootstrap": self._n_bootstrap,
                "ci_level": self._ci_level,
                "exclude_parse_failures": self._exclude_parse_failures,
                "max_workers": self._max_workers,
            }),
        )
