# Examples

Each example has a ready-to-run config file in `examples/`. Run any of them with:

```bash
jury-eval validate examples/<config>.yaml   # check config before running
jury-eval run examples/<config>.yaml        # run with text output (default)
jury-eval run examples/<config>.yaml --format json --output report.json
```

---

## 1 — Minimum viable evaluation

**When to use:** getting started, only one API key, want quick results.  
**Config:** `examples/config_skill_minimal.yaml`  
**Produces:** `EvalReport` with uplift and verbosity bias. No inter-rater agreement (requires ≥ 2 judges). No positional bias (no `positional_judge`).

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
jury-eval run examples/config_skill_minimal.yaml
```

```
=== JuryEval Report ===
Cases:              3
Mean uplift:        +0.310   (treatment − control)
Control score:      0.450
Treatment score:    0.760

Inter-rater agreement:
  Krippendorff α:   n/a  (requires ≥ 2 judges)
  Cohen κ:          n/a

Bias diagnostics:
  Positional bias:  n/a  (no positional_judge configured)
  Verbosity ρ:      0.091  (p=0.412)
  Verbosity flagged:False
```

---

## 2 — Skill evaluation with multi-judge panel

**When to use:** you want to know (a) whether the skill helps AND (b) whether the assertions are unambiguous.  
**Config:** `examples/config_skill.yaml`  
**Produces:** `EvalReport` with uplift + Krippendorff α measuring rubric clarity.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
jury-eval run examples/config_skill.yaml
```

```
Mean uplift:       +0.234
Krippendorff α:    0.812  → strong agreement (α ≥ 0.80)
Cohen κ:           0.791
Expected P(e):     0.334
```

**Interpreting α:**
- `α ≥ 0.80` — strong agreement; assertions are clear; scores are trustworthy
- `0.67–0.80` — tentative; add ICL calibration examples to the judges
- `α < 0.67` — unreliable; rewrite the ambiguous assertions before drawing conclusions

---

## 3 — Dataset evaluation with diverse jury

**When to use:** RAG pipeline or LLM quality scoring with multiple quality dimensions (safety, faithfulness, relevance) weighted differently.  
**Config:** `examples/config_dataset.yaml`  
**Produces:** `EvalReport`. Note: α will be low — that is expected when using a JudgeJury (diverse rubrics).

```bash
jury-eval run examples/config_dataset.yaml
```

---

## 4 — Panel with rubric dimensions (dataset mode)

**When to use:** dataset-style scoring but you need to verify rubric consistency before trusting the scores.  
**Config:** `examples/config_dataset_panel.yaml`  
**Produces:** `EvalReport`. Per-case α identifies which specific cases divided the judges.

```bash
jury-eval run examples/config_dataset_panel.yaml
```

If α < 0.67, look at the `per-case α` column in the output — low-α cases reveal where the rubric wording is ambiguous.

---

## 5 — Endpoint comparison (two live model APIs)

**When to use:** comparing two model endpoints head-to-head — different sizes, fine-tuned vs base, or two providers.  
**Config:** `examples/config_endpoints.yaml`  
**Key setting:** each variant has its own `generation.backend`.

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
jury-eval run examples/config_endpoints.yaml
```

Python API equivalent:
```python
ctrl = Variant(name="gpt-4o-mini", generation_backend=OpenAIBackend("gpt-4o-mini"))
trt  = Variant(name="gpt-4o",      generation_backend=OpenAIBackend("gpt-4o"))
report = evaluator.evaluate_endpoints(cases, ctrl, trt)
```

---

## 6 — Pre-recorded outputs (no generation model)

**When to use:** you already have both outputs captured from production or a prior run. No generation calls — only judge calls. Fast and cheap.  
**Config:** `examples/config_prerecorded.yaml`  
**Dataset format:** `{"output_control": "...", "output_treatment": "..."}`

```bash
jury-eval run examples/config_prerecorded.yaml
```

Python API equivalent:
```python
report = evaluator.evaluate_prerecorded(
    data,
    control_output_field="output_control",
    treatment_output_field="output_treatment",
)
```

---

## 7 — Pairwise pre-recorded (preference rates)

**When to use:** you want preference rates ("System B was preferred 68% of the time") rather than a score delta.  
**Config:** `examples/config_pairwise.yaml`  
**Dataset format:** `{"prompt": "...", "output_a": "...", "output_b": "..."}`  
**Produces:** `PairwiseReport` with preference rates, tie rate, positional bias rate.

```bash
jury-eval run examples/config_pairwise.yaml
```

```
=== JuryEval Pairwise Report ===
System A: gpt-4o-mini    preferred: 27.0%  mean score: 0.612
System B: gpt-4o         preferred: 68.0%  mean score: 0.821
Ties:                               5.0%
```

Python API equivalent:
```python
report = evaluator.evaluate_pairwise_dataset(
    data,
    pairwise_judge=PairwiseJudge("j", AnthropicBackend("claude-sonnet-4-6")),
    output_a_field="output_a",
    output_b_field="output_b",
    label_a="System A",
    label_b="System B",
)
print(f"System B preferred: {report.preference_rate_b:.1%}")
```

---

## 8 — Regulatory / compliance evaluation

**When to use:** producing evidence for a compliance review, audit, or regulatory submission.  
**Config:** `examples/config_regulatory.yaml`  
**Key settings:** `positional_judge` enabled, `verbosity_bias_threshold: 0.25` (stricter), `include_verdicts: true` (full audit trail).

```bash
jury-eval run examples/config_regulatory.yaml --format json --verdicts --output audit-report.json
```

Key output fields for regulatory evidence:
```json
{
  "bias": {
    "positional_bias_rate": 0.083,
    "verbosity_bias_rho": 0.091,
    "verbosity_bias_p": 0.412,
    "verbosity_biased": false
  },
  "agreement": {
    "alpha": 0.812,
    "kappa": 0.791,
    "expected_chance_agreement": 0.334,
    "alpha_interpretation": "strong agreement (α ≥ 0.80)"
  }
}
```

**Acceptance thresholds for regulatory use:**
- `positional_bias_rate < 0.15` — acceptable
- `verbosity_biased: false` — required
- `expected_chance_agreement < 0.50` — judges aren't guessing the modal category
- `α ≥ 0.67` — minimum for tentative conclusions
- `α ≥ 0.80` — required for formal quantitative claims

---

## 9 — Mixed-provider panel (Anthropic + OpenRouter + vLLM + Ollama)

**When to use:** you want to reduce single-provider bias by using judges from different model families simultaneously.  
**Config:** `examples/config_mixed_panel.yaml`

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY="sk-or-..."
# No keys needed for vLLM and Ollama (api_key_env: "")
jury-eval run examples/config_mixed_panel.yaml
```

See [docs/05-key-management.md](05-key-management.md) for the `api_key_env` field documentation.

---

## Example files index

| Config file | Mode | Panel type | Judge type | Special |
|---|---|---|---|---|
| `config_skill_minimal.yaml` | skill | panel | assertion | single judge, no bias detection |
| `config_skill.yaml` | skill | panel | assertion | 3 judges + ICL calibration |
| `config_dataset.yaml` | dataset | jury | rubric | diverse rubrics, weighted_mean |
| `config_dataset_panel.yaml` | dataset | panel | rubric | same rubric on 3 judges |
| `config_endpoints.yaml` | skill | panel | assertion | per-variant generation backend |
| `config_prerecorded.yaml` | dataset | panel | rubric | no generation model |
| `config_pairwise.yaml` | pairwise | — | pairwise | PairwiseReport output |
| `config_regulatory.yaml` | skill | panel | assertion | positional + verbosity bias |
| `config_mixed_panel.yaml` | skill | panel | assertion | 4 providers in one panel |
