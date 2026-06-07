"""
jury-eval CLI entry point.

Usage:
  jury-eval run config.yaml
  jury-eval run config.yaml --output report.json --format json
  jury-eval run config.yaml --format text
  jury-eval schema                    # print JSON Schema for config file
  jury-eval validate config.yaml      # validate config without running evaluation

The CLI is intentionally thin: all logic lives in builder.py and evaluator.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from jury_eval.cli.config_schema import DatasetModeConfig, SkillModeConfig
from jury_eval.cli.builder import build_evaluator, load_dataset
from jury_eval.models import EvalReport


def _load_config(path: str) -> SkillModeConfig | DatasetModeConfig:
    raw = yaml.safe_load(Path(path).read_text())
    mode = raw.get("mode")
    if mode == "skill":
        return SkillModeConfig.model_validate(raw)
    elif mode == "dataset":
        return DatasetModeConfig.model_validate(raw)
    else:
        _die(f"Config 'mode' must be 'skill' or 'dataset', got: {mode!r}")


def _render_text(report: EvalReport) -> str:
    lines = [
        "=== JuryEval Report ===",
        f"Cases:              {len(report.cases)}",
        f"Mean uplift:        {report.mean_uplift:+.4f}  (treatment − control)",
        f"Control score:      {report.mean_control_score:.4f}",
        f"Treatment score:    {report.mean_treatment_score:.4f}",
        "",
        "Inter-rater agreement:",
        f"  Krippendorff α:   {report.agreement.alpha:.4f}"
        + (f"  → {report.agreement.alpha_interpretation}" if report.agreement.alpha_interpretation else ""),
        f"  Cohen κ (mean):   {report.agreement.kappa:.4f}",
        f"  Expected P(e):    {report.agreement.expected_chance_agreement:.4f}",
        f"  Judges:           {report.agreement.n_judges}",
        "",
        "Bias diagnostics:",
        f"  Positional bias:  {report.bias.positional_bias_rate:.1%} of cases flipped",
        f"  Verbosity ρ:      {report.bias.verbosity_bias_rho:.4f}"
        + (f"  (p={report.bias.verbosity_bias_p:.4f})" if report.bias.verbosity_bias_p is not None else ""),
        f"  Verbosity flagged:{report.bias.verbosity_biased}",
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
    return "\n".join(lines)


def _write_output(content: str, file: str | None) -> None:
    if file:
        Path(file).write_text(content)
        print(f"Report written to {file}", file=sys.stderr)
    else:
        print(content)


def _die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


# ── Commands ─────────────────────────────────────────────────────────────────


def cmd_run(config_path: str, output_file: str | None, fmt: str, include_verdicts: bool) -> None:
    config = _load_config(config_path)
    evaluator = build_evaluator(config)

    if isinstance(config, SkillModeConfig):
        from jury_eval.adapters.skill import SkillAdapter
        cases, skill_md = SkillAdapter().load_with_context(config.skill_dir)
        from jury_eval.models import Variant
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
        report = evaluator.evaluate_dataset(data=data, predict_fn=lambda x: x.get("output", ""))

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
    try:
        _load_config(config_path)
        print(f"✓ Config valid: {config_path}")
    except Exception as exc:
        _die(str(exc))


def cmd_schema() -> None:
    """Print merged JSON Schema for both config modes."""
    import json
    from jury_eval.cli.config_schema import SkillModeConfig, DatasetModeConfig

    skill_schema   = SkillModeConfig.model_json_schema()
    dataset_schema = DatasetModeConfig.model_json_schema()
    print(json.dumps({"skill_mode": skill_schema, "dataset_mode": dataset_schema}, indent=2))


def app() -> None:
    """Minimal argument parser — no external deps (no click/typer)."""
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(
            "Usage:\n"
            "  jury-eval run <config.yaml> [--output FILE] [--format json|text|jsonl] [--verdicts]\n"
            "  jury-eval validate <config.yaml>\n"
            "  jury-eval schema\n"
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
                output_file = rest[i + 1]; i += 2
            elif rest[i] == "--format" and i + 1 < len(rest):
                fmt = rest[i + 1]; i += 2
            elif rest[i] == "--verdicts":
                include_verdicts = True; i += 1
            else:
                _die(f"Unknown argument: {rest[i]}")

        if fmt not in ("json", "text", "jsonl"):
            _die(f"--format must be json, text, or jsonl. Got: {fmt!r}")

        cmd_run(config_path, output_file, fmt, include_verdicts)

    else:
        _die(f"Unknown command: {cmd!r}. Run 'jury-eval --help'.")
