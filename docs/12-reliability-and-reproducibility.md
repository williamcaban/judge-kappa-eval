# Reliability and Reproducibility of the LLM Layer

judge-kappa is a measurement library: a number is only as good as the model call behind
it. This page documents what the LLM layer guarantees about the requests it sends, the
replies it accepts, and what it records, and how to make a long run resumable and fast.

| Concern | Mechanism | Default |
|---|---|---|
| Request parameters are explicit | `temperature` always sent, configurable `max_tokens`, `timeout_s`, `max_retries` | on |
| Unusable judge replies are visible | `parse_failed` on verdicts, `data_quality` on reports, `on_parse_failure` policy | flag + count |
| Reports say what produced them | `run` block: versions, per-judge and per-backend config, usage | on |
| Interrupted runs resume | `CachedBackend` / `backend.cache_path` | off |
| Faster runs | `max_workers` | 1 (sequential) |

## 1. Request parameters

Every request parameter is sent explicitly and recorded in the report.

| Parameter | OpenAI-compatible | Anthropic |
|---|---|---|
| `temperature` | always sent, including `0.0` | always sent, including `0.0` |
| `omit_temperature` | `true` never sends it (models that reject it) | same |
| `max_tokens` | sent only when set | required by the API; default **2048** |
| `max_tokens_param` | `max_tokens` (default) or `max_completion_tokens` (OpenAI reasoning models) | n/a |
| `timeout_s`, `max_retries` | forwarded to the SDK client; unset keeps SDK defaults | same |

```yaml
generation:
  backend:
    provider: anthropic
    model: claude-sonnet-4-6
    max_tokens: 4096
    timeout_s: 120
    max_retries: 4
    cache_path: runs/gen-cache.jsonl
```

### Migration note: Anthropic judges at temperature 0

Earlier releases did **not** send `temperature` to Anthropic when it was `0.0`, so the API
default (documented as 1.0) applied and a judge configured as deterministic was not. Judges
on the Anthropic backend now actually run at the configured temperature. Scores for
Anthropic judges can therefore differ from earlier releases, and agreement statistics
usually improve. Re-baseline any stored comparison. If a model rejects the parameter, set
`omit_temperature: true`; the error message tells you so.

### Truncation

A reply that stops because of the output-token limit is detected from the provider's finish
reason (`length` / `max_tokens`). Truncated judge JSON is reported as `truncated_output`;
truncated generations are counted in `data_quality.n_truncated_replies`. Raise `max_tokens`
if you see either: a truncated generation silently biases an evaluation.

## 2. Parse failures and data quality

A judge reply that is not usable JSON, or does not match the expected schema, used to become
a score silently. Now every judge validates its reply:

| Judge | Fallback score on failure (unchanged) | Codes |
|---|---|---|
| `AssertionJudge` | 0.0 per missing assertion | `invalid_json`, `not_an_object`, `missing_key:…`, `missing_assertions:k/n`, `non_numeric_assertion_score`, `out_of_range_assertion_score` |
| `RubricJudge` | 0.0 per missing dimension | same, with `dimension` |
| `PairwiseJudge` | 0.5 / 0.5 (a tie) | `invalid_json`, `missing_key:score_a`, `non_numeric_score`, `out_of_range_score` |
| `RankJudge` | worst rank | `missing_rankings:k/n`, `invalid_ranks`, `non_integer_rank` |

`truncated_output` replaces the code when the reply was cut off by the token limit.

The verdict carries `parse_failed`, `parse_error` and `truncated`. The report carries:

```python
report.data_quality
# DataQuality(n_judge_calls=40, n_parse_failures=3, parse_failure_rate=0.075,
#             parse_failures_by_judge={"flaky": 3}, parse_errors={"invalid_json": 3},
#             n_truncated_replies=0, warnings=["3 of 40 judge replies (7.5%) were unusable ..."])
```

**Policy.** `on_parse_failure="flag"` (default) keeps the historical fallback score so
existing numbers do not move, logs a warning and counts the failure. Fallback scores *are*
included in agreement and uplift, which is why the count is reported. `on_parse_failure="raise"`
raises `JudgeParseError` (with `judge_id`, `case_id`, `error`) so a pipeline can fail fast.

```python
judge = AssertionJudge("j1", backend, on_parse_failure="raise")
```

**Tournaments.** `TournamentEvaluator(exclude_parse_failures=True)` drops unusable rounds
instead of rating them as 0.5/0.5 ties. A system left with no usable game cannot be rated and
is named in the error. `TournamentReport.data_quality` is always populated and printed by
`print_leaderboard()`.

## 3. Provenance and usage

`EvalReport.run`, `PairwiseReport.run` and `TournamentReport.run` hold a `RunMetadata`:

- library, Python and installed SDK / numeric-library versions;
- start, finish and duration, and the evaluator settings (`mode`, `n_cases`, `max_workers`, ...);
- one record per judge: class, temperature, `on_parse_failure`, calibration-example count;
- one record per backend: role, provider, model, base URL, `max_tokens`, `omit_temperature`,
  timeout, retries, and usage (calls, input/output tokens, truncations, cache hits, errors,
  summed latency).

Usage is the **delta for that run**, so a backend reused across runs is not double counted.
A backend shared by several judges is recorded once. Nothing secret is recorded: backends
describe themselves without keys and base URLs are stripped of credentials, query and fragment.

## 4. Response cache: checkpoint, resume, reproduce

```python
from judge_kappa import CachedBackend, OpenAIBackend

backend = CachedBackend(OpenAIBackend("gpt-4o-mini"), "runs/judge-cache.jsonl")
```

or in YAML: `backend.cache_path: runs/judge-cache.jsonl`.

- Every reply is appended to the file as soon as it completes. If a run dies (outage, rate
  limit, Ctrl-C), re-run it with the same file: only the calls that never finished reach the
  model, and the final report equals the uninterrupted one (tested).
- A re-run is free and exactly reproducible. Tuning bootstrap size or thresholds costs nothing.
- The key covers the backend description (provider, model, base URL, `max_tokens`,
  `omit_temperature`), the system and user prompts, and the temperature. Timeout and retries
  are not part of it. Use `namespace=` to separate runs that must not share replies.
- Failures are never cached. A line cut short by a crash is skipped on load.
- Cache hits count as calls but not as tokens, so usage totals reflect what was billed.
- **The file contains your prompts and the model replies in plain text** (created `0600`).
  Keep it out of version control.

## 5. Concurrency

```python
TournamentEvaluator(judge, max_workers=8)
JuryEvaluator(panel=panel, generation_backend=backend, max_workers=8)
```

or `max_workers: 8` at the top level of a YAML config. Judge calls are I/O-bound, so a thread
pool is enough. The default is `1`, which runs inline with no threads.

**Contract: results do not depend on `max_workers`.** Work is assembled in input order, so
reports are identical for any worker count. The test suite compares sequential and concurrent
reports and uses a barrier that only releases when two calls genuinely overlap.

Requirements when `max_workers > 1`: `Variant.predict_fn` callables and custom backends must
be thread-safe. The built-in backends are. Provider rate limits still apply; the SDK retries
429s according to `max_retries`.

## 6. Writing a custom backend

The original contract is unchanged: any object with
`complete(system, user, temperature=0.0) -> str` works. Optional methods unlock more:

| Method / attribute | Effect |
|---|---|
| `complete_with_usage(system, user, temperature) -> LLMResponse` | token usage, finish reason, latency, truncation detection |
| `describe() -> dict` | secret-free description recorded in `run.backends` and used in cache keys |
| `stats` (a `UsageStats`) | per-run usage in the report |

`call_backend(backend, system, user, temperature)` always returns an `LLMResponse`, falling
back to `complete()` when the richer method is absent.

## 7. What this does not do

- No client-side rate limiter or token bucket; rely on `max_retries` and a lower `max_workers`.
- No automatic re-ask after a parse failure; failures are flagged, counted and optionally fatal.
- No provider-native structured output / JSON mode yet; replies are parsed from text.
- No async API; concurrency is thread-based.
