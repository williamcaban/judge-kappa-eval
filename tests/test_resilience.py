"""
Resilience and throughput: response cache (checkpoint/resume) and opt-in concurrency.

The properties that matter for a measurement library:
  * concurrency never changes results (assembly is in input order),
  * an interrupted run resumes and ends with exactly the uninterrupted result,
  * a cache hit is free and never double counts tokens,
  * nothing is cached on failure, and the file survives a crash mid-write.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from judge_kappa.concurrency import ordered_map, validate_max_workers
from judge_kappa.evaluator import JuryEvaluator
from judge_kappa.judges.assertion import AssertionJudge
from judge_kappa.judges.pairwise import PairwiseJudge
from judge_kappa.llm.cache import CachedBackend
from judge_kappa.llm.types import LLMResponse
from judge_kappa.models import Assertion, EvalCase, Variant
from judge_kappa.panel.panel import JudgePanel
from judge_kappa.tournament import TournamentEvaluator

# ── ordered_map ───────────────────────────────────────────────────────────────


class TestOrderedMap:
    def test_results_are_in_input_order_even_when_completion_order_differs(self):
        def slow_first(i: int) -> int:
            time.sleep(0.03 * (5 - i))     # item 0 finishes last
            return i * 10

        assert ordered_map(slow_first, range(5), max_workers=5) == [0, 10, 20, 30, 40]

    def test_default_is_inline_in_the_calling_thread(self):
        main = threading.current_thread().name
        assert ordered_map(lambda _: threading.current_thread().name, range(3)) == [main] * 3

    def test_threads_are_used_above_one(self):
        names = set(ordered_map(lambda _: threading.current_thread().name, range(8), max_workers=4))
        assert any(n.startswith("judge-kappa") for n in names)

    def test_first_exception_propagates(self):
        def boom(i: int) -> int:
            if i == 3:
                raise RuntimeError("item 3 failed")
            return i

        with pytest.raises(RuntimeError, match="item 3 failed"):
            ordered_map(boom, range(10), max_workers=3)

    def test_empty_and_single(self):
        assert ordered_map(str, [], max_workers=4) == []
        assert ordered_map(str, [7], max_workers=4) == ["7"]

    @pytest.mark.parametrize("bad", [0, -1, 1.5, True])
    def test_validate_max_workers_rejects(self, bad):
        with pytest.raises(ValueError, match="max_workers"):
            validate_max_workers(bad)


# ── CachedBackend ─────────────────────────────────────────────────────────────


class _Counting:
    """Backend that records calls; replies are a function of the request."""

    def __init__(self, model: str = "m", fail_on_call: int | None = None, timeout_s: float | None = None) -> None:
        self.calls = 0
        self._model = model
        self._fail_on_call = fail_on_call
        self._timeout_s = timeout_s
        self._lock = threading.Lock()

    def describe(self) -> dict[str, Any]:
        return {"provider": "fake", "model": self._model, "timeout_s": self._timeout_s, "max_retries": 2}

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        return self.complete_with_usage(system, user, temperature).text

    def complete_with_usage(self, system: str, user: str, temperature: float = 0.0) -> LLMResponse:
        with self._lock:
            self.calls += 1
            n = self.calls
        if self._fail_on_call is not None and n == self._fail_on_call:
            raise RuntimeError(f"provider outage on call {n}")
        return LLMResponse(
            text=f"reply|{system}|{user}|{temperature}",
            input_tokens=5, output_tokens=2, finish_reason="stop", model=self._model, latency_s=0.01,
        )


class TestCachedBackend:
    def test_miss_then_hit_calls_the_model_once(self, tmp_path: Path):
        inner = _Counting()
        c = CachedBackend(inner, tmp_path / "c.jsonl")
        first = c.complete_with_usage("s", "u", 0.0)
        second = c.complete_with_usage("s", "u", 0.0)
        assert inner.calls == 1
        assert first.text == second.text
        assert not first.cached and second.cached

    def test_hits_are_free_in_the_usage_totals(self, tmp_path: Path):
        c = CachedBackend(_Counting(), tmp_path / "c.jsonl")
        for _ in range(3):
            c.complete("s", "u")
        snap = c.stats.snapshot()
        assert snap.calls == 3 and snap.cache_hits == 2
        assert snap.input_tokens == 5 and snap.output_tokens == 2   # billed once, not three times

    def test_persists_across_instances(self, tmp_path: Path):
        path = tmp_path / "c.jsonl"
        CachedBackend(_Counting(), path).complete("s", "u")
        inner = _Counting()
        again = CachedBackend(inner, path)
        assert len(again) == 1
        assert again.complete("s", "u").startswith("reply|")
        assert inner.calls == 0

    def test_file_is_private(self, tmp_path: Path):
        path = tmp_path / "c.jsonl"
        CachedBackend(_Counting(), path).complete("s", "u")
        if os.name == "posix":
            assert (path.stat().st_mode & 0o777) == 0o600

    @pytest.mark.parametrize(
        ("kwargs", "expect_call"),
        [
            ({"temperature": 0.7}, True),
            ({"system": "other"}, True),
            ({"user": "other"}, True),
            ({}, False),
        ],
    )
    def test_key_covers_request_fields(self, tmp_path: Path, kwargs, expect_call):
        inner = _Counting()
        c = CachedBackend(inner, tmp_path / "c.jsonl")
        c.complete("s", "u", 0.0)
        before = inner.calls
        c.complete(kwargs.get("system", "s"), kwargs.get("user", "u"), kwargs.get("temperature", 0.0))
        assert (inner.calls > before) is expect_call

    def test_key_covers_backend_identity_but_not_operational_settings(self, tmp_path: Path):
        path = tmp_path / "c.jsonl"
        CachedBackend(_Counting(model="a"), path).complete("s", "u")
        other_model = _Counting(model="b")
        CachedBackend(other_model, path).complete("s", "u")
        assert other_model.calls == 1                        # different model → miss
        same_model_other_timeout = _Counting(model="a", timeout_s=99.0)
        CachedBackend(same_model_other_timeout, path).complete("s", "u")
        assert same_model_other_timeout.calls == 0           # timeout is not part of the key

    def test_namespace_separates_entries(self, tmp_path: Path):
        path = tmp_path / "c.jsonl"
        CachedBackend(_Counting(), path, namespace="run-1").complete("s", "u")
        inner = _Counting()
        CachedBackend(inner, path, namespace="run-2").complete("s", "u")
        assert inner.calls == 1

    def test_failures_are_not_cached(self, tmp_path: Path):
        inner = _Counting(fail_on_call=1)
        c = CachedBackend(inner, tmp_path / "c.jsonl")
        with pytest.raises(RuntimeError):
            c.complete("s", "u")
        assert len(c) == 0 and c.stats.snapshot().errors == 1
        assert c.complete("s", "u").startswith("reply|")     # second attempt reaches the model
        assert inner.calls == 2

    def test_a_line_cut_short_by_a_crash_is_skipped(self, tmp_path: Path):
        path = tmp_path / "c.jsonl"
        CachedBackend(_Counting(), path).complete("s", "u")
        with path.open("a", encoding="utf-8") as fh:
            fh.write('{"v": 1, "key": "abc", "text": "half a rec')   # no closing, no newline
        reloaded = CachedBackend(_Counting(), path)
        assert len(reloaded) == 1 and reloaded.skipped_lines == 1

    def test_thread_safe_appends_produce_whole_lines(self, tmp_path: Path):
        path = tmp_path / "c.jsonl"
        c = CachedBackend(_Counting(), path)
        ordered_map(lambda i: c.complete("s", f"user-{i % 40}"), range(200), max_workers=8)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert all(json.loads(line)["key"] for line in lines)    # every line is a whole record
        assert len(CachedBackend(_Counting(), path)) == 40

    def test_describe_exposes_file_name_not_directory(self, tmp_path: Path):
        d = CachedBackend(_Counting(), tmp_path / "secret-dir" / "c.jsonl").describe()
        assert d["cached"] is True and d["cache_file"] == "c.jsonl"
        assert "secret-dir" not in json.dumps(d)

    def test_wraps_a_plain_backend(self, tmp_path: Path):
        class _Plain:
            def __init__(self) -> None:
                self.n = 0

            def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
                self.n += 1
                return "plain"

        inner = _Plain()
        c = CachedBackend(inner, tmp_path / "c.jsonl")
        assert c.complete("s", "u") == "plain" and c.complete("s", "u") == "plain"
        assert inner.n == 1


# ── Content-aware, deterministic mocks ──────────────────────────────────────────


class _PairQuality:
    """Pairwise judge mock: score = quality table lookup on the displayed output text."""

    def __init__(self, quality: dict[str, float], fail_on_call: int | None = None) -> None:
        self._q = quality
        self._fail_on_call = fail_on_call
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        with self._lock:
            self.calls += 1
            n = self.calls
        if self._fail_on_call is not None and n == self._fail_on_call:
            raise RuntimeError("provider outage")
        a = user.split("## Output A\n", 1)[1].split("\n\n## Output B\n", 1)[0]
        b = user.split("\n\n## Output B\n", 1)[1]
        # first-position bonus makes the order effect non-trivial
        return json.dumps({
            "score_a": min(1.0, self._q[a] + 0.1), "score_b": self._q[b],
            "preferred": "A", "rationale": "r",
        })


_Q = {"s1": 0.9, "s2": 0.6, "s3": 0.3}


def _systems() -> dict[str, Any]:
    return {n: (lambda _x, n=n: n) for n in _Q}


def _cases(n: int) -> list[EvalCase]:
    return [EvalCase(id=f"c{i}", prompt=f"Prompt {i}") for i in range(n)]


def _comparable(report: Any) -> dict[str, Any]:
    return {
        "standings": report.standings,
        "pairs": report.pair_results,
        "matrix": report.win_matrix,
        "order_effect": report.order_effect,
        "ci": report.order_effect_ci,
        "dq": report.data_quality,
    }


class TestTournamentConcurrencyAndResume:
    def test_concurrent_results_equal_sequential_results(self):
        def run(workers: int) -> Any:
            judge = PairwiseJudge("j", _PairQuality(_Q))
            return TournamentEvaluator(judge, n_bootstrap=50, max_workers=workers).run(_cases(6), _systems())

        assert _comparable(run(1)) == _comparable(run(6))

    def test_calls_really_overlap(self):
        barrier = threading.Barrier(2, timeout=5)

        class _Rendezvous(_PairQuality):
            def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
                barrier.wait()          # only releases when two calls are in flight together
                return super().complete(system, user, temperature)

        judge = PairwiseJudge("j", _Rendezvous(_Q))
        TournamentEvaluator(judge, n_bootstrap=0, max_workers=2).run(
            _cases(4), {"s1": lambda _: "s1", "s2": lambda _: "s2"}
        )

    def test_interrupted_run_resumes_and_matches_the_uninterrupted_result(self, tmp_path: Path):
        cases = _cases(4)                               # 3 pairs x 4 cases x 2 orders = 24 calls
        cache = tmp_path / "ckpt.jsonl"

        # Reference: uninterrupted, uncached.
        ref_backend = _PairQuality(_Q)
        reference = TournamentEvaluator(PairwiseJudge("j", ref_backend), n_bootstrap=0).run(cases, _systems())
        assert ref_backend.calls == 24

        # Run 1 dies on the 10th call.
        dying = _PairQuality(_Q, fail_on_call=10)
        with pytest.raises(RuntimeError, match="provider outage"):
            TournamentEvaluator(
                PairwiseJudge("j", CachedBackend(dying, cache)), n_bootstrap=0
            ).run(cases, _systems())
        assert len(CachedBackend(_PairQuality(_Q), cache)) == 9     # 9 replies were banked

        # Run 2 resumes: only the 15 outstanding calls reach the model.
        healthy = _PairQuality(_Q)
        resumed = TournamentEvaluator(
            PairwiseJudge("j", CachedBackend(healthy, cache)), n_bootstrap=0
        ).run(cases, _systems())
        assert healthy.calls == 15
        assert _comparable(resumed) == _comparable(reference)

        # Run 3 is free.
        free = _PairQuality(_Q)
        again = TournamentEvaluator(PairwiseJudge("j", CachedBackend(free, cache)), n_bootstrap=0).run(
            cases, _systems()
        )
        assert free.calls == 0
        assert _comparable(again) == _comparable(reference)
        usage = [b for b in again.run.backends if b.role == "judge"][0].usage   # type: ignore[union-attr]
        assert usage.calls == 24 and usage.cache_hits == 24 and usage.input_tokens == 0   # type: ignore[union-attr]

    def test_max_workers_validation(self):
        judge = PairwiseJudge("j", _PairQuality(_Q))
        with pytest.raises(ValueError, match="max_workers"):
            TournamentEvaluator(judge, max_workers=0)


class _Det:
    """Deterministic judge/generation mock for the A/B evaluator."""

    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        with self._lock:
            self.calls += 1
        if "evaluation judge" in system:
            out = user.split("## AI Output\n", 1)[1].split("\n\n## Assertions", 1)[0]
            texts = user.split("## Assertions to Evaluate\n", 1)[1].splitlines()
            scores = {t[2:]: (1.0 if "good" in out else 0.0) for t in texts if t.startswith("- ")}
            return json.dumps({"assertion_scores": scores, "rationale": f"len={len(out)}"})
        return "generated"


def _ab_cases(n: int) -> list[EvalCase]:
    return [
        EvalCase(id=f"c{i}", prompt=f"Q{i}", assertions=[Assertion(text="is correct"), Assertion(text="is clear")])
        for i in range(n)
    ]


def _ab_variants() -> tuple[Variant, Variant]:
    return (
        Variant(name="control", predict_fn=lambda inputs: f"plain {inputs['prompt']}"),
        Variant(name="treatment", predict_fn=lambda inputs: f"good {inputs['prompt']}"),
    )


def _evaluator(backend: Any, workers: int = 1) -> JuryEvaluator:
    panel = JudgePanel([AssertionJudge("j1", backend), AssertionJudge("j2", backend)])
    return JuryEvaluator(
        panel=panel, generation_backend=backend, compute_judge_fit=False,
        n_bootstrap=100, max_workers=workers,
    )


# Reports are compared through their JSON form: kappa is NaN when judges give constant scores,
# and NaN != NaN would make two identical dicts compare unequal.
class TestEvaluatorConcurrency:
    def test_concurrent_report_equals_sequential_report(self):
        ctrl, trt = _ab_variants()
        seq = _evaluator(_Det(), 1).evaluate(_ab_cases(8), ctrl, trt)
        con = _evaluator(_Det(), 4).evaluate(_ab_cases(8), ctrl, trt)
        assert seq.model_dump_json(exclude={"run"}) == con.model_dump_json(exclude={"run"})
        assert [c.case_id for c in con.cases] == [f"c{i}" for i in range(8)]

    def test_calls_really_overlap(self):
        barrier = threading.Barrier(2, timeout=5)

        class _Rendezvous(_Det):
            def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
                if "evaluation judge" in system:
                    barrier.wait()
                return super().complete(system, user, temperature)

        panel = JudgePanel([AssertionJudge("only", _Rendezvous())])
        ev = JuryEvaluator(panel=panel, generation_backend=_Det(), compute_significance=False,
                           bootstrap_ci=False, compute_judge_fit=False, max_workers=2)
        ctrl, trt = _ab_variants()
        ev.evaluate(_ab_cases(4), ctrl, trt)

    def test_pairwise_dataset_concurrent_equals_sequential(self):
        rows = [{"id": f"r{i}", "prompt": "Q", "output_a": "s1", "output_b": "s2"} for i in range(10)]

        def run(workers: int) -> Any:
            ev = _evaluator(_Det(), workers)
            return ev.evaluate_pairwise_dataset(rows, PairwiseJudge("p", _PairQuality(_Q)))

        a, b = run(1), run(5)
        assert a.model_dump_json(exclude={"run"}) == b.model_dump_json(exclude={"run"})

    def test_settings_record_max_workers(self):
        ctrl, trt = _ab_variants()
        report = _evaluator(_Det(), 3).evaluate(_ab_cases(2), ctrl, trt)
        assert report.run is not None and report.run.settings["max_workers"] == 3

    def test_cache_makes_a_rerun_free_and_identical(self, tmp_path: Path):
        ctrl, trt = _ab_variants()
        inner = _Det()
        cached = CachedBackend(inner, tmp_path / "ab.jsonl")
        first = _evaluator(cached).evaluate(_ab_cases(3), ctrl, trt)
        calls_after_first = inner.calls
        second = _evaluator(cached).evaluate(_ab_cases(3), ctrl, trt)
        assert inner.calls == calls_after_first
        assert first.model_dump_json(exclude={"run"}) == second.model_dump_json(exclude={"run"})
        usage = [b for b in second.run.backends if b.role == "judge"][0].usage   # type: ignore[union-attr]
        assert usage.cache_hits == usage.calls > 0               # type: ignore[union-attr]


# ── Config / builder ────────────────────────────────────────────────────────────


class TestConfigAndBuilder:
    def test_max_workers_must_be_positive(self):
        from pydantic import ValidationError

        from judge_kappa.cli.config_schema import (
            BackendConfig,
            GenerationConfig,
            PairwiseModeConfig,
            PositionalJudgeConfig,
        )

        backend = BackendConfig(provider="openai", model="m")
        base = {
            "mode": "pairwise", "dataset_file": "d.jsonl",
            "generation": GenerationConfig(backend=backend),
            "pairwise_judge": PositionalJudgeConfig(id="p", backend=backend),
        }
        assert PairwiseModeConfig(**base).max_workers == 1
        assert PairwiseModeConfig(**base, max_workers=8).max_workers == 8
        with pytest.raises(ValidationError):
            PairwiseModeConfig(**base, max_workers=0)

    def test_builder_wraps_the_backend_in_a_cache_when_configured(self, monkeypatch, tmp_path: Path):
        from judge_kappa.cli import builder
        from judge_kappa.cli.config_schema import BackendConfig

        class _Fake:
            def __init__(self, **kw: Any) -> None:
                self.kw = kw

            def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
                return "x"

        monkeypatch.setattr(builder, "OpenAIBackend", _Fake)
        plain = builder._build_backend(BackendConfig(provider="openai", model="m", api_key_env=""))
        cached = builder._build_backend(
            BackendConfig(provider="openai", model="m", api_key_env="", cache_path=str(tmp_path / "c.jsonl"))
        )
        assert isinstance(plain, _Fake)
        assert isinstance(cached, CachedBackend) and isinstance(cached.inner, _Fake)
