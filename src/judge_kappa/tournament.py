"""
TournamentEvaluator — compare N evaluation targets using round-robin pairwise scoring.

Design:
  - Runs every combination of (system_i, system_j) through a PairwiseJudge.
  - Produces a win matrix [N × N] and Elo ratings derived from it.
  - Returns TournamentReport: ranked leaderboard, Elo scores, win matrix,
    per-pair preference rates, and positional bias rate per pair.

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
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from judge_kappa.judges.pairwise import PairwiseJudge
from judge_kappa.llm.base import LLMBackend
from judge_kappa.models import EvalCase, Variant


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


@dataclass
class TournamentReport:
    standings: list[TournamentStanding]   # sorted by Elo descending
    pair_results: list[PairResult]
    win_matrix: dict[str, dict[str, float]]   # win_matrix[a][b] = rate A beat B
    n_cases: int
    judge_id: str

    def print_leaderboard(self) -> None:
        print(f"{'Rank':<5} {'System':<30} {'Elo':>7} {'Win%':>6} {'W':>4} {'L':>4} {'T':>4}")
        print("─" * 65)
        for i, s in enumerate(self.standings, 1):
            print(f"{i:<5} {s.system:<30} {s.elo:>7.1f} {s.win_rate:>6.1%}"
                  f" {s.wins:>4} {s.losses:>4} {s.ties:>4}")


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


class TournamentEvaluator:
    """
    Compare N evaluation targets (systems) using round-robin pairwise scoring.

    Args:
        pairwise_judge: A configured PairwiseJudge instance.
        generation_backend: Used to generate outputs from Variant objects.
            Not required if outputs are pre-recorded (run() with pre-set outputs).
        detect_positional_bias: Run each pair in both orders (A,B) and (B,A).
            Doubles judge calls but reports per-pair positional bias rate.
    """

    def __init__(
        self,
        pairwise_judge: PairwiseJudge,
        generation_backend: Optional[LLMBackend] = None,
        detect_positional_bias: bool = True,
    ) -> None:
        self._judge = pairwise_judge
        self._gen_backend = generation_backend
        self._detect_positional_bias = detect_positional_bias

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

        # Generate outputs per system per case
        outputs: dict[str, list[str]] = {name: [] for name in system_names}
        for case in cases:
            inputs = case.metadata.get("inputs", {"prompt": case.prompt})
            for name, fn_or_str in systems.items():
                if callable(fn_or_str):
                    outputs[name].append(fn_or_str(inputs))
                else:
                    outputs[name].append(str(fn_or_str))

        return self._run_tournament(cases, system_names, outputs)

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
                        backend.complete(system=system_msg, user=case.prompt,  # type: ignore[union-attr]
                                         temperature=variant.temperature)
                    )

        return self._run_tournament(cases, system_names, outputs)

    def run_dataset(
        self,
        data: list[dict],
        systems: Mapping[str, Callable[..., str]],
    ) -> TournamentReport:
        """
        Convenience wrapper: loads cases from a list of dicts then runs tournament.
        """
        from judge_kappa.adapters.dataset import DatasetAdapter
        cases = DatasetAdapter().load(data)
        return self.run(cases, systems)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_tournament(
        self,
        cases: list[EvalCase],
        system_names: list[str],
        outputs: dict[str, list[str]],
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

        for i, a in enumerate(system_names):
            for b in system_names[i + 1:]:
                pair_key = (a, b)
                pos_flips[pair_key] = 0
                pair_totals[pair_key] = 0

                for k, case in enumerate(cases):
                    out_a = outputs[a][k]
                    out_b = outputs[b][k]

                    # Round 1: A first
                    va, vb = self._judge.judge_pair(case, out_a, out_b, a, b)
                    score_sums[a][b] += va.score
                    score_sums[b][a] += vb.score

                    if va.score > vb.score:
                        wins[a][b] += 1; losses[b][a] += 1
                    elif vb.score > va.score:
                        wins[b][a] += 1; losses[a][b] += 1
                    else:
                        ties[a][b] += 1; ties[b][a] += 1

                    if self._detect_positional_bias:
                        # Round 2: B first (swapped)
                        vb2, va2 = self._judge.judge_pair(case, out_b, out_a, b, a)
                        r1_a_preferred = va.score > vb.score
                        r2_a_preferred = va2.score > vb2.score
                        if r1_a_preferred and r2_a_preferred:
                            pos_flips[pair_key] += 1
                        pair_totals[pair_key] += 1

        n_cases = len(cases)

        # Build win-rate matrix
        win_matrix: dict[str, dict[str, float]] = {n: {} for n in system_names}
        pair_results: list[PairResult] = []

        for i, a in enumerate(system_names):
            for b in system_names[i + 1:]:
                total = wins[a][b] + wins[b][a] + ties[a][b]
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
                    mean_score_a=round(score_sums[a][b] / n_cases, 4),
                    mean_score_b=round(score_sums[b][a] / n_cases, 4),
                    positional_bias_rate=round(pos_rate, 4),
                ))

        elo_ratings = _compute_elo(win_matrix, system_names)

        standings: list[TournamentStanding] = []
        for name in system_names:
            w = sum(wins[name][o] for o in system_names if o != name)
            l = sum(losses[name][o] for o in system_names if o != name)
            t = sum(ties[name][o] for o in system_names if o != name)
            total_games = w + l + t
            standings.append(TournamentStanding(
                system=name,
                elo=round(elo_ratings[name], 1),
                wins=w, losses=l, ties=t,
                win_rate=round(w / total_games, 4) if total_games else 0.0,
            ))

        standings.sort(key=lambda s: s.elo, reverse=True)

        return TournamentReport(
            standings=standings,
            pair_results=pair_results,
            win_matrix=win_matrix,
            n_cases=n_cases,
            judge_id=self._judge.judge_id,
        )
