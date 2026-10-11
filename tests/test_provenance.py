"""
Run provenance and usage accounting on reports.

Reports must say which model, with which request parameters and SDK versions, produced
them, with usage scoped to the run, and must never carry credentials.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from judge_kappa.evaluator import JuryEvaluator
from judge_kappa.judges.assertion import AssertionJudge
from judge_kappa.judges.pairwise import PairwiseJudge
from judge_kappa.llm import AnthropicBackend, OpenAIBackend
from judge_kappa.llm.types import LLMResponse
from judge_kappa.models import Assertion, EvalCase, EvalReport, PairwiseReport, Variant
from judge_kappa.panel.base import EvaluationPanel
from judge_kappa.panel.panel import JudgePanel
from judge_kappa.tournament import TournamentEvaluator

_JUDGE_JSON = json.dumps({"assertion_scores": {"is correct": 1.0}, "rationale": "ok"})


class _FakeOpenAI:
    """openai.OpenAI stand-in: judge-shaped JSON for judge prompts, prose otherwise."""

    def __init__(self, **kw: Any) -> None:
        self.kw = kw
        outer = self

        class _Completions:
            def create(self, **req: Any) -> Any:
                system = req["messages"][0]["content"]
                is_judge = "evaluation judge" in system
                content = _JUDGE_JSON if is_judge else "a generated answer"
                return SimpleNamespace(
                    model=req["model"],
                    choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
                    usage=SimpleNamespace(prompt_tokens=10, completion_tokens=4),
                )

        self.chat = SimpleNamespace(completions=_Completions())
        outer.chat = self.chat


@pytest.fixture(autouse=True)
def _stub_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)


def _cases(n: int = 2) -> list[EvalCase]:
    return [EvalCase(id=f"c{i}", prompt="Q", assertions=[Assertion(text="is correct")]) for i in range(n)]


def _evaluator(judge_backend: Any, gen_backend: Any, **judge_kw: Any) -> JuryEvaluator:
    panel = JudgePanel([AssertionJudge("j1", judge_backend, temperature=0.0, **judge_kw)])
    return JuryEvaluator(
        panel=panel, generation_backend=gen_backend,
        compute_significance=False, bootstrap_ci=False, compute_judge_fit=False,
    )


class TestEvalReportProvenance:
    def _report(self) -> tuple[EvalReport, OpenAIBackend, OpenAIBackend]:
        judge_b = OpenAIBackend("judge-model", api_key="sk-SECRET", base_url="https://u:p@h.example/v1?k=v",
                                max_tokens=300, timeout_s=20)
        gen_b = OpenAIBackend("gen-model", api_key="sk-SECRET")
        report = _evaluator(judge_b, gen_b).evaluate(_cases(2), Variant(name="control"), Variant(name="treatment"))
        return report, judge_b, gen_b

    def test_judges_and_backends_are_recorded(self):
        report, _, _ = self._report()
        run = report.run
        assert run is not None
        assert run.judge_kappa_version and run.python_version
        assert "numpy" in run.packages and "openai" in run.packages
        assert [(j.judge_id, j.judge_class, j.temperature, j.on_parse_failure) for j in run.judges] == [
            ("j1", "AssertionJudge", 0.0, "flag")
        ]
        by_role = {b.role: b for b in run.backends}
        assert by_role["judge"].description["model"] == "judge-model"
        assert by_role["judge"].description["max_tokens"] == 300
        assert by_role["judge"].description["timeout_s"] == 20
        assert by_role["judge"].judge_ids == ["j1"]
        assert by_role["generation"].description["model"] == "gen-model"
        assert run.judges[0].backend_label == by_role["judge"].label

    def test_usage_is_counted_per_backend(self):
        report, _, _ = self._report()
        by_role = {b.role: b for b in report.run.backends}   # type: ignore[union-attr]
        # 2 cases x 2 variants = 4 generation calls and 4 judge calls
        assert by_role["generation"].usage.calls == 4          # type: ignore[union-attr]
        assert by_role["judge"].usage.calls == 4               # type: ignore[union-attr]
        assert by_role["judge"].usage.input_tokens == 40       # type: ignore[union-attr]
        assert by_role["judge"].usage.output_tokens == 16      # type: ignore[union-attr]

    def test_no_secrets_anywhere_in_the_serialised_report(self):
        report, _, _ = self._report()
        blob = report.model_dump_json()
        for secret in ("sk-SECRET", "u:p@", "k=v"):
            assert secret not in blob
        assert "https://h.example/v1" in blob

    def test_usage_is_scoped_to_each_run(self):
        judge_b = OpenAIBackend("judge-model", api_key="k")
        gen_b = OpenAIBackend("gen-model", api_key="k")
        ev = _evaluator(judge_b, gen_b)
        ev.evaluate(_cases(3), Variant(name="control"), Variant(name="treatment"))
        second = ev.evaluate(_cases(1), Variant(name="control"), Variant(name="treatment"))
        by_role = {b.role: b for b in second.run.backends}     # type: ignore[union-attr]
        assert by_role["judge"].usage.calls == 2               # type: ignore[union-attr]
        assert judge_b.stats.snapshot().calls == 8             # cumulative on the backend

    def test_report_round_trips_through_json(self):
        report, _, _ = self._report()
        again = EvalReport.model_validate_json(report.model_dump_json())
        assert again.run is not None and again.run.judges[0].judge_id == "j1"

    def test_settings_are_recorded(self):
        report, _, _ = self._report()
        s = report.run.settings                                # type: ignore[union-attr]
        assert s["mode"] == "ab_evaluation" and s["n_cases"] == 2
        assert s["control"] == "control" and s["treatment"] == "treatment"

    def test_a_shared_backend_is_recorded_once_with_all_its_judges(self):
        shared = OpenAIBackend("same-model", api_key="k")
        panel = JudgePanel([AssertionJudge("a", shared), AssertionJudge("b", shared)])
        ev = JuryEvaluator(panel=panel, generation_backend=shared, compute_significance=False,
                           bootstrap_ci=False, compute_judge_fit=False)
        run = ev.evaluate(_cases(1), Variant(name="control"), Variant(name="treatment")).run
        assert run is not None and len(run.backends) == 1
        assert run.backends[0].judge_ids == ["a", "b"] and run.backends[0].role == "judge"

    def test_two_backends_with_the_same_model_get_distinct_labels(self):
        a = OpenAIBackend("m", api_key="k", max_tokens=100)
        b = OpenAIBackend("m", api_key="k", max_tokens=200)
        panel = JudgePanel([AssertionJudge("a", a), AssertionJudge("b", b)])
        ev = JuryEvaluator(panel=panel, generation_backend=a, compute_significance=False,
                           bootstrap_ci=False, compute_judge_fit=False)
        run = ev.evaluate(_cases(1), Variant(name="control"), Variant(name="treatment")).run
        labels = [bk.label for bk in run.backends]             # type: ignore[union-attr]
        assert len(set(labels)) == len(labels) == 2

    def test_custom_backend_and_custom_panel_do_not_break_provenance(self):
        class _Plain:
            def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
                return _JUDGE_JSON

        class _OpaquePanel(EvaluationPanel):
            def __init__(self) -> None:
                self._inner = JudgePanel([AssertionJudge("x", _Plain())])

            def evaluate(self, case, output, variant_name):
                return self._inner.evaluate(case, output, variant_name)

            def aggregate_score(self, verdicts):
                return self._inner.aggregate_score(verdicts)

            @property
            def judge_ids(self):
                return self._inner.judge_ids

        ev = JuryEvaluator(panel=_OpaquePanel(), generation_backend=_Plain(),
                           compute_significance=False, bootstrap_ci=False, compute_judge_fit=False)
        run = ev.evaluate(_cases(1), Variant(name="control"), Variant(name="treatment")).run
        assert run is not None
        assert run.judges == []                                 # panel did not expose judges
        gen = [b for b in run.backends if b.role == "generation"][0]
        assert gen.description["provider"] == "_Plain" and gen.usage is None


class TestTruncationAndAnthropicProvenance:
    def test_generation_truncation_reaches_data_quality(self):
        class _Truncating:
            def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
                return "cut off mid-sen"

            def complete_with_usage(self, system: str, user: str, temperature: float = 0.0) -> LLMResponse:
                return LLMResponse(text="cut off mid-sen", finish_reason="length", output_tokens=2048)

        # Give the backend the stats object the evaluator reads for usage deltas.
        from judge_kappa.llm.types import UsageStats

        class _Tracked(_Truncating):
            def __init__(self) -> None:
                self.stats = UsageStats()

            def complete_with_usage(self, system, user, temperature=0.0):
                r = super().complete_with_usage(system, user, temperature)
                self.stats.record(r)
                return r

        class _Judge:
            def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
                return _JUDGE_JSON

        ev = _evaluator(_Judge(), _Tracked())
        report = ev.evaluate(_cases(2), Variant(name="control"), Variant(name="treatment"))
        assert report.data_quality.n_truncated_replies == 4
        assert any("output-token limit" in w for w in report.data_quality.warnings)

    def test_anthropic_backend_description_in_report(self, monkeypatch: pytest.MonkeyPatch):
        import anthropic

        class _FakeAnthropic:
            def __init__(self, **kw: Any) -> None:
                class _M:
                    def create(_s, **req: Any) -> Any:  # noqa: N805
                        return SimpleNamespace(
                            model=req["model"], content=[SimpleNamespace(type="text", text=_JUDGE_JSON)],
                            usage=SimpleNamespace(input_tokens=3, output_tokens=2), stop_reason="end_turn",
                        )

                self.messages = _M()

        monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
        backend = AnthropicBackend("claude-x", api_key="sk-ant-SECRET", max_tokens=1234)
        run = _evaluator(backend, backend).evaluate(_cases(1), Variant(name="c"), Variant(name="t")).run
        assert run is not None
        desc = run.backends[0].description
        assert desc["provider"] == "anthropic" and desc["max_tokens"] == 1234
        assert desc["omit_temperature"] is False
        assert "sk-ant-SECRET" not in json.dumps(run.model_dump())


class TestOtherReports:
    def test_pairwise_dataset_report_has_provenance(self):
        judge = PairwiseJudge("pj", OpenAIBackend("pm", api_key="k"))
        ev = _evaluator(OpenAIBackend("jm", api_key="k"), OpenAIBackend("gm", api_key="k"))
        report = ev.evaluate_pairwise_dataset(
            [{"id": "r0", "prompt": "Q", "output_a": "a", "output_b": "b"}], judge
        )
        assert isinstance(report, PairwiseReport)
        assert report.run is not None and report.run.settings["mode"] == "pairwise_dataset"
        assert report.run.judges[0].judge_class == "PairwiseJudge"
        assert report.run.backends[0].usage.calls == 1         # type: ignore[union-attr]

    def test_tournament_report_has_provenance_and_usage(self):
        judge = PairwiseJudge("tj", OpenAIBackend("tm", api_key="k"))
        report = TournamentEvaluator(judge, n_bootstrap=0).run(
            _cases(2), {"a": lambda _: "a", "b": lambda _: "b"}
        )
        run = report.run
        assert run is not None and run.settings["mode"] == "tournament"
        assert run.settings["n_systems"] == 2 and run.settings["rating_method"] == "btl"
        judge_backend = [b for b in run.backends if b.role == "judge"][0]
        assert judge_backend.usage.calls == 4                  # 1 pair x 2 cases x 2 orders  # type: ignore[union-attr]
