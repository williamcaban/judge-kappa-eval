"""
JuryEvaluator — main orchestrator.

Five entry points covering all evaluation target types:

  evaluate_skill(skill_dir, ...)
    → agent-skills-eval compatible; injects SKILL.md into treatment variant

  evaluate_dataset(data, predict_fn, ...)
    → MLflow LLMaJ compatible; calls predict_fn(inputs) per case

  evaluate_endpoints(cases, control, treatment)
    → two live OpenAI-compatible endpoints; each Variant carries its own
      generation_backend; no shared generation_backend required

  evaluate_prerecorded(data, ...)
    → outputs already exist in the dataset; no generation model called at all;
      supports {"output": "..."} (single) and
      {"output_control": "...", "output_treatment": "..."} (A/B)

  evaluate_pairwise_dataset(data, pairwise_judge, ...)
    → pre-recorded A/B pairs scored by a PairwiseJudge in one prompt;
      returns PairwiseReport (preference rates, not uplift)

  evaluate(cases, control, treatment)
    → low-level: bring-your-own EvalCase list and Variant objects

All A/B modes (skill, dataset, endpoints, prerecorded) produce EvalReport.
Pairwise mode produces PairwiseReport.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from jury_eval.adapters.dataset import DatasetAdapter
from jury_eval.adapters.skill import SkillAdapter
from jury_eval.agreement.alpha import KrippendorffAlpha
from jury_eval.agreement.kappa import CohenKappa
from jury_eval.bias.positional import PositionalBiasDetector, PositionalBiasReport
from jury_eval.bias.verbosity import VerbosityBiasDetector
from jury_eval.judges.pairwise import PairwiseJudge
from jury_eval.llm.base import LLMBackend
from jury_eval.models import (
    AgreementResult,
    BiasResult,
    CaseResult,
    EvalCase,
    EvalReport,
    JudgeVerdict,
    PairwiseCaseResult,
    PairwiseReport,
    RubricDimension,
    ScaleType,
    Variant,
    VariantResult,
)
from jury_eval.panel.base import EvaluationPanel


def _run_variant(variant: Variant, case: EvalCase, default_backend: LLMBackend) -> str:
    """
    Generate an output for a variant. Priority order:
      1. predict_fn                  — call this Python callable
      2. variant.generation_backend  — this variant's own LLM endpoint
      3. default_backend             — shared fallback (JuryEvaluator.generation_backend)
    """
    if variant.predict_fn is not None:
        inputs = case.metadata.get("inputs", {"prompt": case.prompt})
        return variant.predict_fn(inputs)

    backend: LLMBackend = variant.generation_backend or default_backend  # type: ignore[assignment]
    parts: list[str] = []
    if variant.system_prompt:
        parts.append(variant.system_prompt)
    if variant.skill_context:
        parts.append(f"## Skill Instructions\n{variant.skill_context}")
    system = "\n\n".join(parts) or "You are a helpful assistant."
    return backend.complete(system=system, user=case.prompt, temperature=variant.temperature)


class JuryEvaluator:
    def __init__(
        self,
        panel: EvaluationPanel,
        generation_backend: LLMBackend,
        scale_type: ScaleType = ScaleType.ORDINAL,
        positional_judge: Optional[PairwiseJudge] = None,
        verbosity_bias_threshold: float = 0.30,
    ) -> None:
        self._panel = panel
        self._gen_backend = generation_backend
        self._scale_type = scale_type
        self._positional_judge = positional_judge
        self._verbosity_threshold = verbosity_bias_threshold

        self._alpha_metric = KrippendorffAlpha()
        self._kappa_metric = CohenKappa()
        self._verbosity_detector = VerbosityBiasDetector(verbosity_bias_threshold)
        self._positional_detector = (
            PositionalBiasDetector(positional_judge) if positional_judge else None
        )

    # ── Public entry points ──────────────────────────────────────────────────

    def evaluate_skill(
        self,
        skill_dir: str | Path,
        control_variant: Optional[Variant] = None,
        treatment_variant: Optional[Variant] = None,
    ) -> EvalReport:
        adapter = SkillAdapter()
        cases, skill_md = adapter.load_with_context(skill_dir)
        ctrl = control_variant or Variant(name="control")
        trt  = treatment_variant or Variant(name="treatment", skill_context=skill_md)
        return self.evaluate(cases, ctrl, trt)

    def evaluate_dataset(
        self,
        data: list[dict],
        predict_fn: Callable[..., str],
        rubric: Optional[list[RubricDimension]] = None,
        control_variant: Optional[Variant] = None,
        treatment_variant: Optional[Variant] = None,
    ) -> EvalReport:
        """
        MLflow LLMaJ compatible. Calls predict_fn(inputs) to generate each output.
        predict_fn receives the 'inputs' dict from each dataset row.
        Both control and treatment call the same predict_fn by default.
        Pass different control/treatment variants with distinct predict_fns to
        compare two different callables (e.g., two RAG pipeline versions).
        """
        cases = DatasetAdapter().load(data)
        ctrl = control_variant or Variant(name="control", predict_fn=predict_fn)
        trt  = treatment_variant or Variant(name="treatment", predict_fn=predict_fn)
        return self.evaluate(cases, ctrl, trt)

    def evaluate_endpoints(
        self,
        cases: list[EvalCase],
        control: Variant,
        treatment: Variant,
    ) -> EvalReport:
        """
        Compare two live OpenAI-compatible endpoints as evaluation targets.

        Each Variant must carry its own generation_backend — the JuryEvaluator's
        shared generation_backend is only used as a fallback if a variant has none.

        Example:
            from jury_eval import OpenAIBackend, Variant

            ctrl = Variant(
                name="gpt-4o-mini",
                generation_backend=OpenAIBackend("gpt-4o-mini"),
            )
            trt = Variant(
                name="gpt-4o",
                generation_backend=OpenAIBackend("gpt-4o"),
            )
            report = evaluator.evaluate_endpoints(cases, ctrl, trt)
        """
        return self.evaluate(cases, control, treatment)

    def evaluate_prerecorded(
        self,
        data: list[dict],
        control_output_field: str = "output_control",
        treatment_output_field: str = "output_treatment",
        control_label: str = "control",
        treatment_label: str = "treatment",
    ) -> EvalReport:
        """
        Evaluate pre-recorded outputs — no generation model called.

        Dataset row schemas supported:

          Single output (same output scored for both variants):
            {"id": "...", "inputs": {"question": "..."}, "output": "..."}

          A/B outputs (each variant has its own pre-recorded output):
            {"id": "...", "inputs": {...},
             "output_control": "ctrl system output",
             "output_treatment": "new system output"}

        The field names are configurable via control_output_field and
        treatment_output_field. Rows missing the expected field fall back
        to calling the shared generation_backend.
        """
        cases = DatasetAdapter().load(data)

        def _make_prerecorded_fn(field: str) -> Callable[..., str]:
            def _fn(inputs: dict) -> str:
                return inputs.get(field) or inputs.get("output") or ""
            return _fn

        # Stash the pre-recorded outputs inside each case's metadata so
        # _run_variant() can retrieve them via predict_fn.
        for case, row in zip(cases, data):
            inputs_with_outputs = {**case.metadata.get("inputs", {})}
            if control_output_field in row:
                inputs_with_outputs[control_output_field] = row[control_output_field]
            if treatment_output_field in row:
                inputs_with_outputs[treatment_output_field] = row[treatment_output_field]
            if "output" in row:
                inputs_with_outputs["output"] = row["output"]
            case.metadata["inputs"] = inputs_with_outputs

        ctrl = Variant(
            name=control_label,
            predict_fn=_make_prerecorded_fn(control_output_field),
        )
        trt = Variant(
            name=treatment_label,
            predict_fn=_make_prerecorded_fn(treatment_output_field),
        )
        return self.evaluate(cases, ctrl, trt)

    def evaluate_pairwise_dataset(
        self,
        data: list[dict],
        pairwise_judge: "PairwiseJudge",
        output_a_field: str = "output_a",
        output_b_field: str = "output_b",
        label_a: str = "A",
        label_b: str = "B",
    ) -> PairwiseReport:
        """
        Score pre-recorded A/B output pairs using a PairwiseJudge.
        Returns PairwiseReport (preference rates) rather than EvalReport (uplift).

        Dataset row schema:
          {"id": "...", "prompt": "...",
           "output_a": "system A output",
           "output_b": "system B output"}

        The field names are configurable via output_a_field and output_b_field.

        Pairs are presented to the judge in both orders to detect positional bias
        at the corpus level (positional_bias_rate in the report).
        """
        results: list[PairwiseCaseResult] = []

        for i, row in enumerate(data):
            case = EvalCase(
                id=row.get("id", f"pair-{i:04d}"),
                prompt=row.get("prompt", row.get("inputs", {}).get("question", "")),
                metadata=row,
            )
            out_a = row.get(output_a_field, "")
            out_b = row.get(output_b_field, "")

            va, vb = pairwise_judge.judge_pair(case, out_a, out_b, label_a, label_b)
            preferred = label_a if va.score > vb.score else (
                label_b if vb.score > va.score else "tie"
            )
            results.append(PairwiseCaseResult(
                case_id=case.id,
                prompt=case.prompt,
                output_a=out_a,
                output_b=out_b,
                label_a=label_a,
                label_b=label_b,
                score_a=va.score,
                score_b=vb.score,
                preferred=preferred,
                rationale=va.rationale,
            ))

        n = len(results)
        pref_a = sum(1 for r in results if r.preferred == label_a) / n if n else 0.0
        pref_b = sum(1 for r in results if r.preferred == label_b) / n if n else 0.0
        ties   = sum(1 for r in results if r.preferred == "tie")   / n if n else 0.0

        return PairwiseReport(
            cases=results,
            mean_score_a=sum(r.score_a for r in results) / n if n else 0.0,
            mean_score_b=sum(r.score_b for r in results) / n if n else 0.0,
            preference_rate_a=round(pref_a, 4),
            preference_rate_b=round(pref_b, 4),
            tie_rate=round(ties, 4),
            label_a=label_a,
            label_b=label_b,
            judge_id=pairwise_judge.judge_id,
        )

    def evaluate(
        self,
        cases: list[EvalCase],
        control: Variant,
        treatment: Variant,
    ) -> EvalReport:
        all_verdicts: list[JudgeVerdict] = []
        pos_reports:  list[PositionalBiasReport] = []
        case_results: list[CaseResult] = []

        for case in cases:
            ctrl_output = _run_variant(control,   case, self._gen_backend)
            trt_output  = _run_variant(treatment, case, self._gen_backend)

            ctrl_verdicts = self._panel.evaluate(case, ctrl_output, control.name)
            trt_verdicts  = self._panel.evaluate(case, trt_output,  treatment.name)
            case_verdicts = ctrl_verdicts + trt_verdicts
            all_verdicts.extend(case_verdicts)

            ctrl_score = self._panel.aggregate_score(ctrl_verdicts)
            trt_score  = self._panel.aggregate_score(trt_verdicts)

            # Per-case agreement (meaningful only with ≥ 2 judges)
            case_agreement = AgreementResult()
            if len(ctrl_verdicts) >= 2:
                case_agreement = self._alpha_metric.compute(
                    ctrl_verdicts, self._scale_type
                )

            # Positional bias test
            if self._positional_detector:
                report = self._positional_detector.test_case(
                    case, ctrl_output, trt_output
                )
                pos_reports.append(report)
                pos_flag = report.positional_flip
            else:
                pos_flag = False

            case_results.append(CaseResult(
                case_id=case.id,
                control=VariantResult(
                    variant_name=control.name,
                    output=ctrl_output,
                    score=ctrl_score,
                    token_count=len(ctrl_output.split()),
                ),
                treatment=VariantResult(
                    variant_name=treatment.name,
                    output=trt_output,
                    score=trt_score,
                    token_count=len(trt_output.split()),
                ),
                uplift=round(trt_score - ctrl_score, 4),
                verdicts=case_verdicts,
                agreement=case_agreement,
                positional_flip=pos_flag,
            ))

        # Corpus-level metrics
        corpus_alpha  = self._alpha_metric.compute(all_verdicts, self._scale_type)
        corpus_kappa  = self._kappa_metric.compute(all_verdicts, self._scale_type)
        corpus_agreement = AgreementResult(
            kappa=corpus_kappa.kappa,
            alpha=corpus_alpha.alpha,
            expected_chance_agreement=corpus_kappa.expected_chance_agreement,
            alpha_interpretation=corpus_alpha.alpha_interpretation,
            n_judges=corpus_alpha.n_judges,
            n_cases=corpus_alpha.n_cases,
        )

        verb_bias = self._verbosity_detector.detect(verdicts=all_verdicts)
        pos_bias_result = (
            self._positional_detector.detect(reports=pos_reports)
            if self._positional_detector and pos_reports
            else BiasResult()
        )
        corpus_bias = BiasResult(
            positional_bias_rate=pos_bias_result.positional_bias_rate,
            verbosity_bias_rho=verb_bias.verbosity_bias_rho,
            verbosity_bias_p=verb_bias.verbosity_bias_p,
            verbosity_biased=verb_bias.verbosity_biased,
        )

        uplift_vals = [r.uplift for r in case_results]
        return EvalReport(
            cases=case_results,
            mean_uplift=round(float(np.mean(uplift_vals)), 4),
            mean_control_score=round(
                float(np.mean([r.control.score for r in case_results])), 4
            ),
            mean_treatment_score=round(
                float(np.mean([r.treatment.score for r in case_results])), 4
            ),
            agreement=corpus_agreement,
            bias=corpus_bias,
            judge_ids=self._panel.judge_ids,
            scale_type=self._scale_type,
        )
