# judge-kappa

Unified LLM evaluation that bridges **agent-skills-eval** (A/B uplift, assertion lists) and **MLflow LLMaJ** (datasets, named rubric dimensions) while adding statistical rigor missing from both.

## Feature matrix

| Capability | agent-skills-eval | MLflow LLMaJ | judge-kappa |
|---|---|---|---|
| Native A/B uplift | ✅ | ❌ | ✅ |
| Assertion-based scoring | ✅ | ❌ | ✅ |
| Named rubric dimensions | ❌ | ✅ | ✅ |
| Multi-judge Panel / Jury | ❌ | ❌ | ✅ |
| Krippendorff's α + 95% CI | ❌ | ❌ | ✅ |
| ICC(2,k) variance decomposition | ❌ | ❌ | ✅ |
| Cohen's κ + P(chance) | ❌ | ❌ | ✅ |
| McNemar significance test for uplift | ❌ | ❌ | ✅ |
| Bootstrap CI for mean uplift | ❌ | ❌ | ✅ |
| ICL judge alignment (calibration examples) | ❌ | ❌ | ✅ |
| IRT-based judge weighting (2PL) | ❌ | ❌ | ✅ |
| Per-judge person-fit (outfit MNSQ) | ❌ | ❌ | ✅ |
| Positional bias detection | ❌ | ❌ | ✅ |
| Verbosity bias detection | ❌ | ❌ | ✅ |
| Differential Item Functioning (DIF) | ❌ | ❌ | ✅ |
| Behavioral alignment metric (DISC-style) | ❌ | ❌ | ✅ |
| Pairwise preference rates | ❌ | ❌ | ✅ |
| N-system Elo tournament (N ≤ 6) | ❌ | ❌ | ✅ |
| Listwise ranking for N ≥ 7 (RankJudge) | ❌ | ❌ | ✅ |
| CLI-first | ✅ | ❌ | ✅ |

---

## Installation

### From PyPI (recommended)

```bash
# Core — no LLM providers
pip install judge-kappa
uv add judge-kappa

# With Anthropic support
pip install "judge-kappa[anthropic]"
uv add "judge-kappa[anthropic]"

# With OpenAI-compatible support (OpenAI, vLLM, Ollama, OpenRouter, Groq, Together)
pip install "judge-kappa[openai]"
uv add "judge-kappa[openai]"

# Everything including tiktoken for accurate verbosity-bias token counts
pip install "judge-kappa[all]"
uv add "judge-kappa[all]"
```

### From source

```bash
git clone https://github.com/williamcaban/judge-kappa-eval
cd judge-kappa-eval

# Python 3.12+ required
uv sync --all-extras          # recommended
# or: pip install -e '.[all]'
```

### API keys

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

### Runnable demos (no API key required)

```bash
# 6 psychometric methods: bootstrap CI, ICC, person-fit, DIF, behavioral alignment, McNemar
python examples/demo_psychometric.py

# IRT-based judge weighting from calibration data
python examples/demo_irt_weighting.py

# Listwise ranking for 8 systems (98% fewer judge calls than tournament)
python examples/demo_rank_judge.py
```

---

## Documentation

| Doc | Contents |
|---|---|
| [Decision guide](docs/01-decision-guide.md) | Which mode to use — decision tree covering input format, judge strategy, bias detection, N systems, psychometric rigor, and providers |
| [Input types](docs/02-input-types.md) | All entry points: `evaluate_skill`, `evaluate_dataset`, `evaluate_endpoints`, `evaluate_prerecorded`, `evaluate_pairwise_dataset`, `TournamentEvaluator`, `RankJudge` |
| [Pairwise, Tournament, and Listwise](docs/03-pairwise-tournament.md) | Pairwise preference, N-system Elo tournament, RankJudge (N ≥ 7), champion-challenger; cost tables; mode selection guide |
| [Examples](docs/04-examples.md) | 12 worked examples: 9 YAML configs + 3 Python demos with expected output |
| [Key management](docs/05-key-management.md) | `api_key_env` three-case rule; provider quick reference; mixed-provider panel |
| [CLI reference](docs/06-cli-reference.md) | `run`, `validate`, `schema` commands; output formats; `--verdicts` flag; CI/CD integration |
| [Python API](docs/07-python-api.md) | 13 usage patterns; all new v0.2 patterns (McNemar, IRT weighting, RankJudge, DIF, behavioral alignment); report field reference |
| [Config reference](docs/08-config-reference.md) | Every YAML field documented; skill/dataset/pairwise mode schemas |
| [Architecture](docs/09-architecture.md) | Package structure; ABCs and extension points; data flow diagram |
| [Psychometric methods](docs/10-psychometric-methods.md) | In-depth guide to all 9 psychometric capabilities: when to use, API, interpretation tables |
| [Scholarly references](docs/11-scholarly-references.md) | 18 techniques × (description + in-library use + primary citations); consolidated bibliography |

---

## Example configs and scripts

| File | API key? | What it demonstrates |
|---|---|---|
| `examples/config_skill_minimal.yaml` | yes | Single judge, fastest start |
| `examples/config_skill.yaml` | yes | 3-judge panel + ICL calibration + positional bias |
| `examples/config_dataset.yaml` | yes | Diverse JudgeJury with weighted rubric dimensions |
| `examples/config_dataset_panel.yaml` | yes | Homogeneous panel — rubric consistency check |
| `examples/config_endpoints.yaml` | yes | Per-variant generation backend (endpoint A vs B) |
| `examples/config_prerecorded.yaml` | yes | Score pre-recorded outputs (no generation calls) |
| `examples/config_pairwise.yaml` | yes | Pairwise preference rates (PairwiseReport) |
| `examples/config_regulatory.yaml` | yes | Full bias evidence for compliance submissions |
| `examples/config_mixed_panel.yaml` | yes | Anthropic + OpenRouter + vLLM + Ollama in one panel |
| `examples/demo_psychometric.py` | **no** | Bootstrap CI, ICC, person-fit, DIF, behavioral alignment, McNemar |
| `examples/demo_irt_weighting.py` | **no** | IRT 2PL judge weighting from calibration data |
| `examples/demo_rank_judge.py` | **no** | Listwise ranking for 8 systems (98% call reduction) |

---

## Development

```bash
uv sync --group dev
uv run pytest                                  # tests (151 pass, 0 fail)
uv run pytest --cov --cov-report=term-missing  # with coverage
uv run ruff check src tests                    # lint
uv run mypy src                                # type check
```

---

## License

Apache 2.0 — [github.com/williamcaban/judge-kappa-eval](https://github.com/williamcaban/judge-kappa-eval)
