# judge-kappa

Unified LLM evaluation that bridges **agent-skills-eval** (A/B uplift, assertion lists) and **MLflow LLMaJ** (datasets, named rubric dimensions) while adding statistical rigor missing from both.

| Capability | agent-skills-eval | MLflow LLMaJ | judge-kappa |
|---|---|---|---|
| Native A/B uplift | ✅ | ❌ | ✅ |
| Assertion-based scoring | ✅ | ❌ | ✅ |
| Named rubric dimensions | ❌ | ✅ | ✅ |
| Multi-judge Panel / Jury | ❌ | ❌ | ✅ |
| Krippendorff's α | ❌ | ❌ | ✅ |
| Cohen's κ + P(chance) | ❌ | ❌ | ✅ |
| ICL judge alignment | ❌ | ❌ | ✅ |
| Positional bias detection | ❌ | ❌ | ✅ |
| Verbosity bias detection | ❌ | ❌ | ✅ |
| Pairwise preference rates | ❌ | ❌ | ✅ |
| N-system Elo tournament | ❌ | ❌ | ✅ |
| CLI-first | ✅ | ❌ | ✅ |

---

## Installation

```bash
git clone https://github.com/williamcaban/judge-kappa-eval
cd judge-kappa

# Python 3.12+ required
uv sync --all-extras          # recommended
# or: pip install -e '.[all]'
```

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
# For local vLLM/Ollama, set base_url in the config instead of an API key
```

---

## Quick start

```bash
# Validate a config
judge-kappa validate examples/config_skill_minimal.yaml

# Run evaluation
judge-kappa run examples/config_skill_minimal.yaml

# JSON output → file
judge-kappa run examples/config_skill.yaml --format json --output report.json

# Print config JSON Schema (for editor autocomplete)
judge-kappa schema
```

---

## Documentation

| Doc | Contents |
|---|---|
| [Decision guide](docs/01-decision-guide.md) | Which mode to use — decision tree covering input format, judge strategy, bias detection, N systems, and providers |
| [Input types](docs/02-input-types.md) | All six entry points: `evaluate_skill`, `evaluate_dataset`, `evaluate_endpoints`, `evaluate_prerecorded`, `evaluate_pairwise_dataset`, `TournamentEvaluator` |
| [Pairwise and Tournament](docs/03-pairwise-tournament.md) | When and how to use pairwise preference scoring and N-system Elo tournaments; cost tables; champion-challenger pattern |
| [Examples](docs/04-examples.md) | Nine worked examples with configs, expected output, and interpretation guidance |
| [Key management](docs/05-key-management.md) | `api_key_env` three-case rule; provider quick reference; mixed-provider panel; Python API key passing |
| [CLI reference](docs/06-cli-reference.md) | `run`, `validate`, `schema` commands; output formats (`text`, `json`, `jsonl`); `--verdicts` flag; CI validation |
| [Python API](docs/07-python-api.md) | Eight usage patterns with full code; accessing report fields; serialization |
| [Config reference](docs/08-config-reference.md) | Every YAML field documented; skill/dataset/pairwise mode schemas; annotated full example |
| [Architecture](docs/09-architecture.md) | Package structure; ABCs and extension points; adding providers, judges, metrics, and bias detectors |

---

## Example configs

| Config | What it demonstrates |
|---|---|
| `examples/config_skill_minimal.yaml` | Single judge, no bias detection — fastest start |
| `examples/config_skill.yaml` | 3-judge panel + ICL calibration + positional bias |
| `examples/config_dataset.yaml` | Diverse JudgeJury with weighted rubric dimensions |
| `examples/config_dataset_panel.yaml` | Homogeneous panel to verify rubric consistency |
| `examples/config_endpoints.yaml` | Per-variant generation backend (endpoint A vs B) |
| `examples/config_prerecorded.yaml` | Score pre-recorded outputs (no generation calls) |
| `examples/config_pairwise.yaml` | Pairwise preference rates (PairwiseReport) |
| `examples/config_regulatory.yaml` | Full bias evidence for compliance submissions |
| `examples/config_mixed_panel.yaml` | Anthropic + OpenRouter + vLLM + Ollama in one panel |

---

## Development

```bash
uv sync --group dev
uv run pytest                                  # tests
uv run pytest --cov --cov-report=term-missing  # with coverage
uv run ruff check src tests                    # lint
uv run mypy src                                # type check
```

---

## License

Apache 2.0 — [github.com/williamcaban/judge-kappa-eval](https://github.com/williamcaban/judge-kappa-eval)
