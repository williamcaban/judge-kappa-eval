# Config File Reference

All CLI runs are driven by a YAML config file. Print the full JSON Schema with:

```bash
jury-eval schema
jury-eval schema | jq '.skill_mode.properties'
```

---

## Top-level structure

Every config starts with `mode` and shares the `generation`, `control`, `treatment`, `panel`, and `output` blocks.

```yaml
mode: skill | dataset | pairwise    # required
```

---

## `generation` block

Backend used to run the control and treatment variants. Separate from judge backends — judge and generation models can differ.

```yaml
generation:
  backend:
    provider: anthropic | openai       # required
    model: "claude-sonnet-4-6"         # required
    base_url: "http://..."             # optional — vLLM, Ollama, OpenRouter, etc.
    api_key_env: "MY_VAR"             # optional — see docs/05-key-management.md
```

For per-variant generation (endpoint comparison), add `generation` inside the `control` or `treatment` block:

```yaml
control:
  name: gpt-4o-mini
  generation:
    backend:
      provider: openai
      model: gpt-4o-mini
```

---

## `control` and `treatment` blocks

```yaml
control:
  name: "control"              # required — used as label in output report
  system_prompt: "..."         # optional — prepended to user prompt
  skill_context: "..."         # optional — auto-loaded from SKILL.md in skill mode
  model: "..."                 # optional — overrides generation.backend.model
  temperature: 0.0             # optional — default 0.0
  generation:                  # optional — per-variant backend (overrides top-level)
    backend:
      provider: openai
      model: gpt-4o-mini

treatment:
  name: "treatment"
  ...                          # same fields as control
```

In **skill mode**, `skill_context` is auto-loaded from `SKILL.md` for the treatment variant. You do not need to specify it manually.

---

## `panel` block

```yaml
panel:
  type: panel | jury          # required
  strategy: mean | weighted_mean | majority_vote | median | trimmed_mean
  judges:
    - id: "my-judge"          # required — unique identifier
      type: assertion | rubric # required
      backend:                 # required
        provider: anthropic | openai
        model: "..."
        base_url: "..."        # optional
        api_key_env: "..."     # optional — see docs/05-key-management.md
      weight: 1.0              # optional — used by jury; panel uses equal weights
      temperature: 0.0         # optional
      calibration:             # optional — ICL anchors for score grounding
        - prompt: "..."
          output: "..."
          score: 0.0–1.0
          rationale: "..."
      rubric:                  # required when type: rubric
        - name: "..."
          description: "..."
          weight: 1.0
          anchors:             # optional — low/mid/high example strings
            low:  "..."
            mid:  "..."
            high: "..."
```

### `type: panel` vs `type: jury`

| | `panel` | `jury` |
|---|---|---|
| Rubric | Same for all judges | Different per judge |
| Krippendorff α | Agreement diagnostic — should be high | Diversity diagnostic — low α is expected |
| Aggregation | `mean` (default) or `median` for outlier resistance | `weighted_mean` (default) for trust-weighted aggregation |
| Use when | You want to verify rubric clarity | You want diverse quality perspectives |

### Aggregation strategies

| Strategy | Behaviour | Use when |
|---|---|---|
| `mean` | Equal weight average | Judges are equally trusted |
| `weighted_mean` | Weight by `judge.weight` | Jury with trusted vs exploratory judges |
| `majority_vote` | Binary pass/fail — majority wins | Categorical scoring only |
| `median` | Robust to single outlier | One judge is suspected unreliable |
| `trimmed_mean` | Drops top and bottom 10% | Panel with many judges, remove extremes |

---

## `positional_judge` block (optional)

Enables A/B swap bias detection. Adds 2× judge calls per case for the positional judge only.

```yaml
positional_judge:
  id: "pairwise-auditor"
  backend:
    provider: anthropic
    model: claude-sonnet-4-6
    api_key_env: "..."         # optional
  temperature: 0.0
  calibration:                 # optional
    - ...
```

Remove this section entirely to skip positional bias detection.

---

## Statistical settings

```yaml
scale_type: ordinal | interval | ratio | nominal
# Governs Krippendorff's α distance function:
#   ordinal  — use for LLM scores bucketed into ranks (default)
#   interval — use when score differences are meaningful (0.8 and 0.9 genuinely closer than 0.1 and 0.9)
#   ratio    — use when a true zero exists and ratios are meaningful
#   nominal  — use for categorical labels (PASS/FAIL classes)

verbosity_bias_threshold: 0.30
# Spearman ρ > threshold AND p < 0.05 → flag as verbosity-biased
# Default: 0.30; use 0.25 for regulatory submissions
```

---

## `output` block

```yaml
output:
  format: text | json | jsonl  # default: text
  file: "report.json"          # optional — defaults to stdout
  include_verdicts: false       # include per-judge per-case verdicts in JSON output
```

The `--output`, `--format`, and `--verdicts` CLI flags override these values.

---

## Skill mode fields

```yaml
mode: skill
skill_dir: "./my-skill"     # required — directory containing SKILL.md and evals/evals.json
```

**`evals/evals.json` format:**
```json
{
  "evals": [
    {
      "id": "case-001",
      "name": "Human-readable name",
      "prompt": "User prompt text",
      "assertions": [
        "Output identifies the root cause.",
        "Output provides a concrete recommendation."
      ],
      "expected_output": "Optional expected answer for reference"
    }
  ]
}
```

---

## Dataset mode fields

```yaml
mode: dataset
dataset_file: "./evals.jsonl"    # required — JSONL or JSON array
```

**JSONL row format:**
```json
{
  "id": "case-001",
  "inputs": {
    "question": "What is RAG?",
    "context": "RAG stands for..."
  },
  "expectations": {
    "answer": "Retrieval-Augmented Generation..."
  }
}
```

**Pre-recorded outputs format (for `evaluate_prerecorded`):**
```json
{
  "id": "case-001",
  "inputs": {"question": "..."},
  "output_control":   "System A output...",
  "output_treatment": "System B output..."
}
```

---

## Pairwise mode fields

```yaml
mode: pairwise
dataset_file: "./pairwise_evals.jsonl"

output_a_field: output_a        # default — field name in dataset rows
output_b_field: output_b
label_a: "System A"
label_b: "System B"

pairwise_judge:
  id: "pairwise-claude"
  backend:
    provider: anthropic
    model: claude-sonnet-4-6
  temperature: 0.0
```

**JSONL row format for pairwise:**
```json
{
  "id": "pair-001",
  "prompt": "User prompt used to generate both outputs",
  "output_a": "System A output...",
  "output_b": "System B output..."
}
```

---

## Full annotated example (skill mode)

```yaml
mode: skill
skill_dir: ./my-skill

generation:
  backend:
    provider: anthropic
    model: claude-sonnet-4-6
    # api_key_env absent → reads ANTHROPIC_API_KEY

control:
  name: control
  temperature: 0.0

treatment:
  name: treatment
  temperature: 0.0
  # skill_context auto-loaded from SKILL.md

panel:
  type: panel                  # homogeneous — all judges use same assertions
  strategy: mean
  judges:
    - id: judge-claude
      type: assertion
      backend:
        provider: anthropic
        model: claude-sonnet-4-6
      weight: 1.0
      calibration:
        - prompt: "Summarize the risks."
          output: "There are risks."
          score: 0.1
          rationale: "Too vague — doesn't name any risks."
        - prompt: "Summarize the risks."
          output: "Three risks: CVE-2024-1234 (critical), open S3 bucket, no MFA."
          score: 0.95
          rationale: "Names all risks with specifics."

    - id: judge-gpt4
      type: assertion
      backend:
        provider: openai
        model: gpt-4o

positional_judge:
  id: positional-auditor
  backend:
    provider: anthropic
    model: claude-sonnet-4-6

scale_type: ordinal
verbosity_bias_threshold: 0.30

output:
  format: text
  include_verdicts: false
```
