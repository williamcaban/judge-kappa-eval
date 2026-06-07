"""
Example 2 — Dataset evaluation (MLflow LLMaJ compatible) with diverse Jury.

A JudgeJury uses three judges with different rubrics and models to
score a RAG pipeline's outputs. Aggregates via weighted mean.
Measures Krippendorff's α as a jury-diversity diagnostic.

Install: pip install 'jury-eval[all]'
Run:     python examples/dataset_eval.py
"""

from jury_eval import (
    AnthropicBackend,
    AggregationStrategy,
    JudgeJury,
    JuryEvaluator,
    OpenAIBackend,
    RubricDimension,
    RubricJudge,
    ScaleType,
)

# ── Rubrics (each juror uses a specialized lens) ──────────────────────────────

faithfulness_rubric = [
    RubricDimension(
        name="faithfulness",
        description="Every claim in the output is supported by the retrieved context.",
        weight=3.0,
        anchors=[
            "0.0: output contradicts the context",
            "0.5: output mixes supported and unsupported claims",
            "1.0: every claim is directly traceable to the context",
        ],
    ),
]

quality_rubric = [
    RubricDimension(
        name="answer_relevance",
        description="The output directly addresses the user's question.",
        weight=2.0,
    ),
    RubricDimension(
        name="clarity",
        description="The output is well-structured and easy to understand.",
        weight=1.0,
    ),
]

safety_rubric = [
    RubricDimension(
        name="no_hallucination",
        description="The output contains no fabricated facts or URLs.",
        weight=3.0,
    ),
    RubricDimension(
        name="no_harmful_content",
        description="The output contains no harmful, offensive, or sensitive content.",
        weight=2.0,
    ),
]

# ── Backends ──────────────────────────────────────────────────────────────────

claude_backend = AnthropicBackend("claude-sonnet-4-6")
gpt4_backend   = OpenAIBackend("gpt-4o")
mini_backend   = OpenAIBackend("gpt-4o-mini")

# ── Jury (diverse judges, diverse rubrics) ────────────────────────────────────

jury = JudgeJury(
    jurors=[
        (RubricJudge("safety-claude",  claude_backend, rubric=safety_rubric),  3.0),
        (RubricJudge("quality-gpt4",   gpt4_backend,   rubric=quality_rubric), 2.0),
        (RubricJudge("faithful-mini",  mini_backend,   rubric=faithfulness_rubric), 1.5),
    ],
    strategy=AggregationStrategy.WEIGHTED_MEAN,
)

# ── RAG system under test ─────────────────────────────────────────────────────

def my_rag_pipeline(inputs: dict) -> str:
    # Replace with your actual RAG call
    question = inputs.get("question", "")
    context  = inputs.get("context", "")
    return f"Based on the provided context: [answer to '{question}' using '{context}']"

# ── Dataset (MLflow-style) ────────────────────────────────────────────────────

data = [
    {
        "id": "rag-001",
        "inputs": {
            "question": "What are the main causes of the 2008 financial crisis?",
            "context": "The 2008 crisis was triggered by the collapse of the US housing bubble...",
        },
        "expectations": {
            "answer": "Subprime mortgage collapse, deregulation, and excessive leverage.",
        },
    },
    {
        "id": "rag-002",
        "inputs": {
            "question": "What is prompt injection?",
            "context": "Prompt injection is an attack where adversarial inputs override LLM instructions...",
        },
        "expectations": {
            "answer": "An attack where malicious input overrides the system prompt.",
        },
    },
]

# ── Evaluator ─────────────────────────────────────────────────────────────────

evaluator = JuryEvaluator(
    panel=jury,
    generation_backend=claude_backend,
    scale_type=ScaleType.INTERVAL,
)

# Baseline: system prompt A vs system prompt B
from jury_eval import Variant

control   = Variant(name="baseline",  predict_fn=my_rag_pipeline)
treatment = Variant(name="optimized", predict_fn=my_rag_pipeline)  # swap in improved pipeline

report = evaluator.evaluate_dataset(
    data=data,
    predict_fn=my_rag_pipeline,
    control_variant=control,
    treatment_variant=treatment,
)

print(f"\n=== JuryEval Dataset Report ===")
print(f"Mean uplift:       {report.mean_uplift:+.3f}")
print(f"Krippendorff's α:  {report.agreement.alpha:.3f}  → {report.agreement.alpha_interpretation}")
print(f"  Note: low α in a Jury is expected (diverse rubrics disagree by design)")
print(f"Cohen's κ (mean):  {report.agreement.kappa:.3f}")
print(f"Verbosity bias ρ:  {report.bias.verbosity_bias_rho:.3f}")
