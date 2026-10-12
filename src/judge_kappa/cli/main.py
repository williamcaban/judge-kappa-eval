"""
judge-kappa CLI entry point.

Usage:
  judge-kappa run config.yaml
  judge-kappa run config.yaml --output report.json --format json
  judge-kappa run config.yaml --format text
  judge-kappa schema                    # print JSON Schema for config file
  judge-kappa validate config.yaml      # validate config without running evaluation

The CLI is intentionally thin: all logic lives in builder.py and evaluator.py.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import yaml

from judge_kappa.bias.verbosity import NOT_ESTIMATED_TEXT
from judge_kappa.cli.builder import build_evaluator, load_dataset
from judge_kappa.cli.config_schema import DatasetModeConfig, PairwiseModeConfig, SkillModeConfig
from judge_kappa.judges.base import JudgeParseError
from judge_kappa.models import DataQuality, EvalReport, RunMetadata


def _load_config(path: str) -> SkillModeConfig | DatasetModeConfig | PairwiseModeConfig:
    """Parse a YAML file into the config model for its ``mode``.

    Exits with an error message for an unknown mode.
    """
    raw = yaml.safe_load(Path(path).read_text())
    mode = raw.get("mode")
    if mode == "skill":
        return SkillModeConfig.model_validate(raw)
    elif mode == "dataset":
        return DatasetModeConfig.model_validate(raw)
    elif mode == "pairwise":
        return PairwiseModeConfig.model_validate(raw)
    else:
        _die(f"Config 'mode' must be 'skill', 'dataset', or 'pairwise', got: {mode!r}")
    raise RuntimeError("unreachable")


_NEEDS_PAIR = "n/a (needs at least 2 judges and 2 shared cases)"
_NEEDS_TWO_JUDGES = "n/a (needs at least 2 judges)"


def _fmt_metric(value: float | None, missing: str) -> str:
    """Format an optional metric to 4 places; say why when it is absent or undefined."""
    if value is None:
        return missing
    if not math.isfinite(value):
        return "undefined (scores do not vary)"
    return f"{value:.4f}"


def _render_alpha(report: EvalReport) -> str:
    """Krippendorff's alpha line. Alpha is not defined for a single judge, so say so."""
    agreement = report.agreement
    if agreement.n_judges < 2:
        # The model carries 1.0 here (its single-value-domain fallback). Do not present that
        # as "perfect agreement" between judges when there is only one.
        return f"  Krippendorff α:   {_NEEDS_TWO_JUDGES}"
    line = f"  Krippendorff α:   {_fmt_metric(agreement.alpha, 'n/a')}"
    if agreement.alpha is not None and agreement.alpha_interpretation:
        line += f"  → {agreement.alpha_interpretation}"
    return line


def _render_verbosity_rho(report: EvalReport) -> str:
    """One line for the verbosity correlation; says so when it could not be estimated."""
    rho = report.bias.verbosity_bias_rho
    if rho is None:
        reason = report.bias.verbosity_not_estimated_reason
        why = f" ({NOT_ESTIMATED_TEXT[reason]})" if reason else ""
        return f"  Verbosity ρ:      not estimated{why}"
    p = report.bias.verbosity_bias_p
    return f"  Verbosity ρ:      {rho:.4f}" + (f"  (p={p:.4f})" if p is not None else "")


def _render_text(report: EvalReport) -> str:
    """Render the human-readable text report, including data-quality warnings and a usage line."""
    lines = [
        "=== JuryEval Report ===",
        f"Cases:              {len(report.cases)}",
        f"Mean uplift:        {report.mean_uplift:+.4f}  (treatment − control)",
        f"Control score:      {report.mean_control_score:.4f}",
        f"Treatment score:    {report.mean_treatment_score:.4f}",
        "",
        "Inter-rater agreement:",
        _render_alpha(report),
        f"  Cohen κ (mean):   {_fmt_metric(report.agreement.kappa, _NEEDS_PAIR)}",
        f"  Expected P(e):    {_fmt_metric(report.agreement.expected_chance_agreement, _NEEDS_PAIR)}",
        f"  Judges:           {report.agreement.n_judges}",
        "",
        "Bias diagnostics:",
        f"  Positional bias:  {report.bias.positional_bias_rate:.1%} of cases flipped",
        _render_verbosity_rho(report),
        f"  Verbosity flagged:{'n/a' if report.bias.verbosity_bias_rho is None else report.bias.verbosity_biased}",
        "",
        "Per-case results:",
    ]
    for c in report.cases:
        flag = "  ⚠ positional flip" if c.positional_flip else ""
        alpha_str = f"  α={c.agreement.alpha:.3f}" if c.agreement.alpha is not None else ""
        lines.append(
            f"  {c.case_id:<24} uplift={c.uplift:+.4f}"
            f"  ctrl={c.control.score:.3f}  trt={c.treatment.score:.3f}"
            f"{alpha_str}{flag}"
        )
    lines += _render_quality(report.data_quality, report.run)
    return "\n".join(lines)


def _render_quality(dq: DataQuality, run: RunMetadata | None) -> list[str]:
    """Data-quality warnings and a usage line; shown only when there is something to say."""
    lines: list[str] = []
    if dq.warnings:
        lines += ["", "Data quality warnings:"]
        lines += [f"  ⚠ {w}" for w in dq.warnings]
    if run is not None:
        calls = sum(b.usage.calls for b in run.backends if b.usage)
        hits = sum(b.usage.cache_hits for b in run.backends if b.usage)
        tin = sum(b.usage.input_tokens for b in run.backends if b.usage)
        tout = sum(b.usage.output_tokens for b in run.backends if b.usage)
        if calls:
            lines += [
                "",
                f"LLM calls: {calls} ({hits} from cache)  tokens in/out: {tin}/{tout}"
                f"  duration: {run.duration_s:.1f}s",
            ]
    return lines


def _write_output(content: str, file: str | None) -> None:
    """Write ``content`` to ``file`` (noted on stderr), or to stdout when None."""
    if file:
        Path(file).write_text(content)
        print(f"Report written to {file}", file=sys.stderr)
    else:
        print(content)


def _die(msg: str) -> None:
    """Print an error to stderr and exit with status 1."""
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


# ── Commands ─────────────────────────────────────────────────────────────────


def cmd_run(config_path: str, output_file: str | None, fmt: str, include_verdicts: bool) -> None:
    """Run an evaluation from a YAML config and emit the report.

    Args:
        config_path: Path to the YAML config.
        output_file: Overrides the config's output file when set.
        fmt: ``text``, ``json`` or ``jsonl``.
        include_verdicts: Keep per-judge verdicts in the JSON report.
    """
    config = _load_config(config_path)

    if isinstance(config, PairwiseModeConfig):
        from judge_kappa.cli.builder import _build_backend, _build_calibration
        from judge_kappa.evaluator import JuryEvaluator
        from judge_kappa.judges.pairwise import PairwiseJudge
        from judge_kappa.panel.panel import JudgePanel
        backend = _build_backend(config.pairwise_judge.backend)
        cal = _build_calibration(config.pairwise_judge.calibration)
        pj = PairwiseJudge(config.pairwise_judge.id, backend, calibration_examples=cal,
                           temperature=config.pairwise_judge.temperature,
                           on_parse_failure=config.pairwise_judge.on_parse_failure)
        gen_backend = _build_backend(config.generation.backend)
        # JudgePanel with a single assertion judge as a placeholder — pairwise mode
        # never calls panel.evaluate(); it calls evaluate_pairwise_dataset() directly.
        from judge_kappa.judges.assertion import AssertionJudge
        panel = JudgePanel(judges=[AssertionJudge("_placeholder", gen_backend)])
        evaluator = JuryEvaluator(
            panel=panel, generation_backend=gen_backend, max_workers=config.max_workers
        )
        data = load_dataset(config.dataset_file)
        pairwise_report = evaluator.evaluate_pairwise_dataset(
            data=data,
            pairwise_judge=pj,
            output_a_field=config.output_a_field,
            output_b_field=config.output_b_field,
            label_a=config.label_a,
            label_b=config.label_b,
        )
        if fmt == "json":
            content = pairwise_report.model_dump_json(indent=2)
        elif fmt == "jsonl":
            content = "\n".join(c.model_dump_json() for c in pairwise_report.cases)
        else:
            content = (
                f"=== judge-kappa Pairwise Report ===\n"
                f"System A: {pairwise_report.label_a:<20} preferred: {pairwise_report.preference_rate_a:.1%}"
                f"  mean score: {pairwise_report.mean_score_a:.3f}\n"
                f"System B: {pairwise_report.label_b:<20} preferred: {pairwise_report.preference_rate_b:.1%}"
                f"  mean score: {pairwise_report.mean_score_b:.3f}\n"
                f"Ties:                               {pairwise_report.tie_rate:.1%}\n"
                + "\n".join(
                    _render_quality(pairwise_report.data_quality, pairwise_report.run)
                ).lstrip("\n")
            )
        out = output_file or config.output.file
        _write_output(content, out)
        return

    evaluator = build_evaluator(config)

    if isinstance(config, SkillModeConfig):
        from judge_kappa.adapters.skill import SkillAdapter
        cases, skill_md = SkillAdapter().load_with_context(config.skill_dir)
        from judge_kappa.models import Variant
        ctrl = Variant(
            name=config.control.name,
            system_prompt=config.control.system_prompt,
            model=config.control.model or config.generation.backend.model,
            temperature=config.control.temperature,
        )
        trt = Variant(
            name=config.treatment.name,
            system_prompt=config.treatment.system_prompt,
            skill_context=config.treatment.skill_context or skill_md,
            model=config.treatment.model or config.generation.backend.model,
            temperature=config.treatment.temperature,
        )
        report = evaluator.evaluate(cases, ctrl, trt)

    else:  # dataset mode
        data = load_dataset(config.dataset_file)
        def _output_fn(x: dict[str, object]) -> str:
            """Return the row's recorded ``output`` as the variant output."""
            return str(x.get("output", ""))
        report = evaluator.evaluate_dataset(data=data, predict_fn=_output_fn)

    if fmt == "json":
        exclude = None if include_verdicts else {"cases": {"__all__": {"verdicts"}}}
        content = report.model_dump_json(indent=2, exclude=exclude)
    elif fmt == "jsonl":
        lines = [c.model_dump_json() for c in report.cases]
        content = "\n".join(lines)
    else:
        content = _render_text(report)

    out = output_file or config.output.file
    _write_output(content, out)


def cmd_validate(config_path: str) -> None:
    """Validate a config file; print a confirmation, or exit non-zero with the error."""
    try:
        _load_config(config_path)
        print(f"✓ Config valid: {config_path}")
    except Exception as exc:
        _die(str(exc))


def cmd_schema() -> None:
    """Print merged JSON Schema for all config modes."""
    import json

    from judge_kappa.cli.config_schema import DatasetModeConfig, PairwiseModeConfig, SkillModeConfig

    skill_schema    = SkillModeConfig.model_json_schema()
    dataset_schema  = DatasetModeConfig.model_json_schema()
    pairwise_schema = PairwiseModeConfig.model_json_schema()
    print(json.dumps(
        {"skill_mode": skill_schema, "dataset_mode": dataset_schema, "pairwise_mode": pairwise_schema},
        indent=2,
    ))


def app() -> None:
    """Minimal argument parser — no external deps (no click/typer)."""
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(
            "Usage:\n"
            "  judge-kappa run <config.yaml> [--output FILE] [--format json|text|jsonl] [--verdicts]\n"
            "  judge-kappa validate <config.yaml>\n"
            "  judge-kappa schema\n"
        )
        return

    cmd = args[0]

    if cmd == "schema":
        cmd_schema()

    elif cmd == "validate":
        if len(args) < 2:
            _die("validate requires a config file path")
        cmd_validate(args[1])

    elif cmd == "run":
        if len(args) < 2:
            _die("run requires a config file path")
        config_path = args[1]
        rest = args[2:]
        output_file = None
        fmt = "text"
        include_verdicts = False

        i = 0
        while i < len(rest):
            if rest[i] == "--output" and i + 1 < len(rest):
                output_file = rest[i + 1]
                i += 2
            elif rest[i] == "--format" and i + 1 < len(rest):
                fmt = rest[i + 1]
                i += 2
            elif rest[i] == "--verdicts":
                include_verdicts = True
                i += 1
            else:
                _die(f"Unknown argument: {rest[i]}")

        if fmt not in ("json", "text", "jsonl"):
            _die(f"--format must be json, text, or jsonl. Got: {fmt!r}")

        try:
            cmd_run(config_path, output_file, fmt, include_verdicts)
        except JudgeParseError as exc:
            _die(
                f"{exc}\n"
                "Set `on_parse_failure: flag` on the judge to keep going and count "
                "unusable replies in the report's data_quality instead."
            )

    else:
        _die(f"Unknown command: {cmd!r}. Run 'judge-kappa --help'.")
