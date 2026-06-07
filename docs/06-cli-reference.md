# CLI Reference

The `judge-kappa` command is installed as a script entry point when you `pip install` or `uv sync` the package.

---

## Commands

### `judge-kappa run`

Run an evaluation from a YAML config file.

```bash
judge-kappa run <config.yaml> [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--format text\|json\|jsonl` | `text` | Output format |
| `--output <file>` | stdout | Write report to file instead of stdout |
| `--verdicts` | off | Include per-judge per-case verdict detail (JSON only) |

**Examples:**

```bash
# Human-readable output to stdout
judge-kappa run examples/config_skill.yaml

# JSON report to file
judge-kappa run examples/config_skill.yaml --format json --output report.json

# JSON with full per-judge per-case verdict detail
judge-kappa run examples/config_skill.yaml --format json --verdicts --output report.json

# JSONL — one case per line; good for large corpora or streaming
judge-kappa run examples/config_skill.yaml --format jsonl | jq '.uplift'

# Write to file (overrides output.file in config)
judge-kappa run examples/config_skill.yaml --output report.txt
```

---

### `judge-kappa validate`

Validate a config file without running evaluation. Checks schema, required fields, and `api_key_env` variable names (but does not verify the keys are set).

```bash
judge-kappa validate <config.yaml>
```

```bash
judge-kappa validate examples/config_skill.yaml
# ✓ Config valid: examples/config_skill.yaml
```

Exit code `0` on success, non-zero on failure.

---

### `judge-kappa schema`

Print the full JSON Schema for both config modes (`skill` and `dataset`) to stdout. Useful for:
- Editor autocomplete (paste into your `.vscode/settings.json` YAML schema mapping)
- CI-side config validation via `ajv` or `check-jsonschema`
- Understanding every field and its allowed values

```bash
judge-kappa schema
judge-kappa schema | jq '.skill_mode.properties.panel'
```

---

### `judge-kappa --help`

Print usage summary.

```bash
judge-kappa --help
```

---

## Output formats

### `--format text` (default)

Human-readable report to stdout.

```
=== judge-kappa Report ===
Cases:              3
Mean uplift:        +0.234   (treatment − control)
Control score:      0.512
Treatment score:    0.746

Inter-rater agreement:
  Krippendorff α:   0.812  → strong agreement (α ≥ 0.80)
  Cohen κ (mean):   0.791
  Expected P(e):    0.334
  Judges:           2

Bias diagnostics:
  Positional bias:  8.3% of cases flipped
  Verbosity ρ:      0.091  (p=0.412)
  Verbosity flagged:False

Per-case results:
  case-001      uplift=+0.310  ctrl=0.450  trt=0.760  α=0.834
  case-002      uplift=+0.190  ctrl=0.580  trt=0.770  α=0.801
  case-003      uplift=+0.203  ctrl=0.506  trt=0.709  α=0.799
```

### `--format json`

Full `EvalReport` serialized as pretty-printed JSON. Compatible with MLflow artifact logging, EvalHub, and any JSON consumer.

Per-judge per-case verdicts are excluded by default (add `--verdicts` to include them).

```bash
judge-kappa run config.yaml --format json | python3 -c "
import json, sys
r = json.load(sys.stdin)
print(f'uplift: {r[\"mean_uplift\"]:+.3f}')
print(f'alpha:  {r[\"agreement\"][\"alpha\"]:.3f}')
"
```

### `--format jsonl`

One `CaseResult` JSON object per line. Useful for streaming large corpora or piping individual cases into other tools.

```bash
judge-kappa run config.yaml --format jsonl \
  | jq 'select(.uplift > 0.3) | .case_id'
```

---

## Using `--verdicts` with JSON

When `--verdicts` is set, each `CaseResult` in the JSON output includes the full `verdicts` array with per-judge scores and rationales. Use this for:
- Audit trails (regulatory submissions)
- Debugging why a specific case scored unexpectedly
- Computing custom aggregations over judge verdicts

```bash
judge-kappa run examples/config_regulatory.yaml \
  --format json --verdicts --output audit-report.json
```

The `verdicts` field on each case:
```json
"verdicts": [
  {
    "judge_id": "judge-claude",
    "eval_case_id": "case-001",
    "variant_name": "treatment",
    "score": 0.833,
    "rationale": "Output identifies all three risks with specific details.",
    "assertion_scores": {
      "Output identifies the root cause.": 1.0,
      "Output provides a concrete recommendation.": 1.0,
      "Output is concise — under 3 sentences.": 0.5
    },
    "output_token_count": 47
  }
]
```

---

## Config file validation in CI

```bash
# Fail CI if config is invalid
judge-kappa validate examples/config_skill.yaml || exit 1

# Validate all configs
for f in examples/config_*.yaml; do
    judge-kappa validate "$f" || exit 1
done

# JSON Schema validation with check-jsonschema
judge-kappa schema > judge-kappa-schema.json
check-jsonschema --schemafile judge-kappa-schema.json examples/config_skill.yaml
```
