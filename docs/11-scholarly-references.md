# Scholarly References

This document provides grounded references for every statistical method, psychometric technique, and evaluation framework implemented in judge-kappa. Each entry includes a succinct description of the method, how it is used in this library, and the primary scholarly sources.

A practitioner can use this document to understand the theoretical basis of each technique. A scholar or AI agent can use it as a curated bibliography for AI evaluation methodology.

---

## 1 — Krippendorff's Alpha (α)

**What it is.** A reliability coefficient that quantifies agreement among any number of raters coding the same items. Handles missing data (NaN entries), supports all measurement scales (nominal, ordinal, interval, ratio), and generalises beyond the pairwise constraints of Cohen's κ. The formula compares observed disagreement to expected disagreement under statistical independence.

**In judge-kappa.** Default agreement metric in `KrippendorffAlpha`. Computed corpus-wide (all judges × all cases) and per-case. Interpretation thresholds: α ≥ 0.80 = strong; 0.67–0.80 = tentative; α < 0.67 = unreliable. v0.2 adds bootstrap 95% CI via `bootstrap_ci=True`.

**Primary sources.**
- Krippendorff, K. (2004). Reliability in content analysis: Some common misconceptions and recommendations. *Human Communication Research*, 30(3), 411–433. https://doi.org/10.1111/j.1468-2958.2004.tb00738.x
- Krippendorff, K. (2013). *Content Analysis: An Introduction to Its Methodology* (3rd ed.). SAGE Publications.
- Hayes, A. F., & Krippendorff, K. (2007). Answering the call for a standard reliability measure for coding data. *Communication Methods and Measures*, 1(1), 77–89. https://doi.org/10.1080/19312450709336664

---

## 2 — Cohen's Kappa (κ)

**What it is.** A pairwise inter-rater agreement coefficient that corrects for chance agreement. Compares the proportion of observed agreement to the proportion expected by chance (P_e). Weighted κ (linear weights) is appropriate for ordinal scales — partial disagreement receives partial credit rather than full penalty.

**In judge-kappa.** `CohenKappa` computes mean pairwise κ across all judge pairs. For panels with more than two judges, all C(n,2) pairs are averaged. Discretises continuous 0–1 scores into 5 bins before computing κ. Reports `expected_chance_agreement` P_e as a rubric-quality diagnostic: P_e > 0.70 may indicate all judges cluster at the modal score.

**Primary sources.**
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46. https://doi.org/10.1177/001316446002000104
- Cohen, J. (1968). Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit. *Psychological Bulletin*, 70(4), 213–220. https://doi.org/10.1037/h0026256
- Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159–174. https://doi.org/10.2307/2529310

---

## 3 — Intraclass Correlation Coefficient — ICC(2,k)

**What it is.** A reliability coefficient based on ANOVA variance decomposition. ICC(2,k) (two-way random effects, absolute agreement, average of k raters) decomposes total score variance into between-cases variance (signal), between-raters variance (systematic leniency/harshness), and residual error. Unlike Krippendorff α, this decomposition reveals *why* judges disagree — systematic offset (fixable with ICL calibration) vs. random noise (fixable with more cases or clearer rubric).

**In judge-kappa.** `compute_icc(verdicts)` and `enrich_agreement_with_icc(result, verdicts)` in `agreement/icc.py`. Populates `AgreementResult.icc` and `AgreementResult.icc_interpretation`. Computed automatically by `JuryEvaluator` when `compute_icc=True`.

**Thresholds** (Cicchetti 1994): ICC ≥ 0.75 = excellent; 0.60–0.74 = good; 0.40–0.59 = fair; < 0.40 = poor.

**Primary sources.**
- Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: Uses in assessing rater reliability. *Psychological Bulletin*, 86(2), 420–428. https://doi.org/10.1037/0033-2909.86.2.420
- Cicchetti, D. V. (1994). Guidelines, criteria, and rules of thumb for evaluating normed and standardized assessment instruments in psychology. *Psychological Assessment*, 6(4), 284–290. https://doi.org/10.1037/1040-3590.6.4.284
- Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine*, 15(2), 155–163. https://doi.org/10.1016/j.jcm.2016.02.012

---

## 4 — Person-fit Statistics (Outfit MNSQ)

**What it is.** Person-fit statistics measure how well a person's (here: a judge's) observed response pattern conforms to what an IRT or response model predicts. The outfit mean square (Outfit MNSQ) is the average of standardised squared residuals across items (eval cases). Values near 1.0 indicate expected fit; MNSQ > 2 flags unexpectedly erratic scoring; MNSQ < 0.5 flags suspiciously uniform scoring. The Wright–Masters t-transformation standardises MNSQ to a z-score for hypothesis testing.

**In judge-kappa.** `PersonFitAnalyzer` in `agreement/personfit.py` uses a model-free 1PL-like approximation (no IRT model fitting required). Expected scores are derived from judge and case means via a logistic function. The t-statistic is compared to 1.96 (p = 0.05) to flag inconsistent judges. Populated in `EvalReport.judge_fit`.

**Primary sources.**
- Wright, B. D., & Masters, G. N. (1982). *Rating Scale Analysis: Rasch Measurement*. MESA Press.
- Meijer, R. R., & Sijtsma, K. (2001). Methodology review: Evaluating person fit. *Applied Psychological Measurement*, 25(2), 107–135. https://doi.org/10.1177/01466210122031957
- Linacre, J. M. (2002). What do infit and outfit, mean-square and standardized mean? *Rasch Measurement Transactions*, 16(2), 878. https://www.rasch.org/rmt/rmt162f.htm

---

## 5 — Bootstrap Confidence Intervals

**What it is.** A non-parametric resampling method for estimating the sampling distribution of a statistic. Cases are resampled with replacement B times (B = 2000 by default); the statistic is computed on each resample; the 2.5th and 97.5th percentiles of the distribution define the 95% CI. Requires no distributional assumptions and handles the complex dependency structure of Krippendorff α correctly (resamples cases, not individual observations).

**In judge-kappa.** `KrippendorffAlpha(bootstrap_ci=True)` produces `AgreementResult.alpha_ci_low` and `alpha_ci_high`. `_mcnemar_and_ci()` in `evaluator.py` produces `UpliftSignificance.uplift_ci_low` and `uplift_ci_high` by resampling `CaseResult.uplift` values.

**Primary sources.**
- Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *The Annals of Statistics*, 7(1), 1–26. https://doi.org/10.1214/aos/1176344552
- Efron, B., & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. Chapman and Hall. https://doi.org/10.1007/978-1-4899-4541-9
- Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods and Their Application*. Cambridge University Press. https://doi.org/10.1017/CBO9780511802843

---

## 6 — McNemar Test

**What it is.** A non-parametric test for the difference between paired binary proportions, analogous to a paired t-test for binary data. Applied to a 2×2 contingency table of treatment wins vs. control wins — only *discordant* pairs (cases where one variant beats the other) carry information. The continuity-corrected statistic is χ² = (|b − c| − 1)² / (b + c), where b = treatment wins and c = control wins.

**In judge-kappa.** `_mcnemar_and_ci()` in `evaluator.py`. Called automatically when `compute_significance=True`. Results in `EvalReport.significance.mcnemar_statistic`, `.p_value`, `.significant`. The two-step validation protocol (Krippendorff α ≥ 0.67 first, then McNemar) prevents false positives from noisy judges.

**Primary sources.**
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157. https://doi.org/10.1007/BF02295996
- Edwards, A. L. (1948). Note on the "correction for continuity" in testing the significance of the difference between correlated proportions. *Psychometrika*, 13(3), 185–187. https://doi.org/10.1007/BF02289261
- Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895–1923. https://doi.org/10.1162/089976698300017197

---

## 7 — Differential Item Functioning (DIF)

**What it is.** DIF occurs when an item (eval case) elicits different responses from members of different groups (here: judge model families) after controlling for the underlying ability or quality level. A DIF-flagged item gives one group systematically higher or lower scores than would be expected given overall quality — evidence of judge family preference rather than genuine quality differences. The Mantel-Haenszel procedure and logistic regression are the two dominant DIF detection approaches.

**In judge-kappa.** `DifferentialItemFunctioningDetector` in `bias/dif.py` uses logistic regression (Swaminathan & Rogers 1990). For each eval case: P(positive verdict) is modelled as a function of overall score level (covariate) and judge model family (binary group indicator). A significant group coefficient (Wald p < α) flags DIF. Model family is extracted from judge_id prefix.

**Primary sources.**
- Holland, P. W., & Thayer, D. T. (1988). Differential item performance and the Mantel-Haenszel procedure. In H. Wainer & H. I. Braun (Eds.), *Test Validity* (pp. 129–145). Lawrence Erlbaum Associates.
- Swaminathan, H., & Rogers, H. J. (1990). Detecting differential item functioning using logistic regression procedures. *Journal of Educational Measurement*, 27(4), 361–370. https://doi.org/10.1111/j.1745-3984.1990.tb00754.x
- Zieky, M. J. (1993). Practical questions in the use of DIF statistics in test development. In P. W. Holland & H. Wainer (Eds.), *Differential Item Functioning* (pp. 337–347). Lawrence Erlbaum Associates.

---

## 8 — Item Response Theory — 2PL Model

**What it is.** IRT is a family of latent trait models relating observable responses to an unobservable latent variable. The 2-Parameter Logistic (2PL) model characterises each item (calibration example) by a discrimination parameter a (steepness of the response curve) and a difficulty parameter b (location of the inflection point). A person's (judge's) latent ability θ determines the probability of a correct response: P(X=1|θ) = 1 / (1 + exp(−a(θ − b))).

**In judge-kappa.** `IRTJudgeWeighter` in `calibration/irt.py` fits a 2PL model via maximum likelihood (scipy L-BFGS-B) on a judge × calibration-item response matrix. Log-normal prior on discrimination and normal prior on difficulty prevent overfitting (especially important when n_items < 30). Per-judge θ is converted to panel weights via softmax. Motivated by Fonseca Rivera et al. (2026), who demonstrate 97–99% cost reduction in safety evaluation using IRT.

**Primary sources.**
- Birnbaum, A. (1968). Some latent trait models and their use in inferring an examinee's ability. In F. M. Lord & M. R. Novick (Eds.), *Statistical Theories of Mental Test Scores* (pp. 395–479). Addison-Wesley.
- Lord, F. M. (1980). *Applications of Item Response Theory to Practical Testing Problems*. Lawrence Erlbaum Associates.
- Baker, F. B., & Kim, S.-H. (2004). *Item Response Theory: Parameter Estimation Techniques* (2nd ed.). Marcel Dekker. https://doi.org/10.1201/9781482276725
- Fonseca Rivera, J., Shah, N., Africa, D. D., & Voudouris, K. (2026). Item Response Theory for AI Safety Evaluation. *arXiv:2608.05086*. https://arxiv.org/abs/2608.05086

---

## 9 — Elo Rating System

**What it is.** A method for calculating the relative skill levels of players in two-player games, originally designed for chess. After each match, ratings are updated by K × (actual_score − expected_score), where expected_score is derived from the logistic function of the rating difference. The 400-point scale anchors the interpretation: a 100-point gap implies ≈ 64% win probability for the higher-rated player.

**In judge-kappa.** `_compute_elo()` in `tournament.py` estimates ratings from a win-rate matrix using batch iterative updates (one virtual match per pair per iteration until convergence). Ratings are anchored at 1000. A 100-point gap implies ≈ 64% win rate; 200 points implies ≈ 76%.

**Primary sources.**
- Elo, A. E. (1978). *The Rating of Chessplayers, Past and Present*. Arco Publishing.
- Glickman, M. E. (1999). Parameter estimation in large dynamic paired comparison experiments. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 48(3), 377–394. https://doi.org/10.1111/1467-9876.00159
- Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems*, 36. https://arxiv.org/abs/2306.05685

---

## 10 — In-Context Learning (ICL) Calibration

**What it is.** Few-shot prompting technique where human-validated (input, output, label) examples are prepended to the model's context to anchor its interpretation of a task. For LLM judges, ICL calibration examples ground the judge's scoring scale — shared examples across multiple judges reduce systematic score drift and improve inter-rater agreement.

**In judge-kappa.** `LLMJudge._icl_block()` in `judges/base.py` renders `CalibrationExample` objects as a formatted "Calibration Anchors" section prepended to every judge prompt. Calibration examples carry a `rationale` field that explains *why* a score is assigned — the most important component for reducing judge drift (see Kim et al. 2023 on rationale quality).

**Primary sources.**
- Brown, T., Mann, B., Ryder, N., et al. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877–1901. https://arxiv.org/abs/2005.14165
- Min, S., Lyu, X., Holtzman, A., et al. (2022). Rethinking the role of demonstrations: What makes in-context learning work? In *Proceedings of EMNLP 2022* (pp. 11048–11064). https://arxiv.org/abs/2202.12837
- Kim, S., Shin, J., Cho, Y., et al. (2024). Prometheus: Inducing fine-grained evaluation capability in language models. In *Proceedings of ICLR 2024*. https://arxiv.org/abs/2310.08491

---

## 11 — LLM-as-Judge

**What it is.** A paradigm for automated evaluation of language model outputs using a language model as the evaluator. The judge receives a rubric or assertion list and returns scores/preferences, enabling scalable, reproducible evaluation at lower cost than human annotation. Key challenges include positional bias, verbosity bias, self-enhancement bias (a model preferring its own outputs), and calibration drift across temperature and context.

**In judge-kappa.** All `LLMJudge` subclasses implement this paradigm. `AssertionJudge` (binary pass/fail per assertion), `RubricJudge` (continuous 0–1 per named dimension), `PairwiseJudge` (A vs. B preference), and `RankJudge` (listwise ranking) are the four provided implementations.

**Primary sources.**
- Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems*, 36. https://arxiv.org/abs/2306.05685
- Shankar, S., Zamfirescu-Pereira, J., Hartmann, B., et al. (2024). Who validates the validators? Aligning LLM-assisted evaluation of LLM outputs with human preferences. In *Proceedings of UIST 2024*. https://arxiv.org/abs/2404.12272
- Wang, P., Li, L., Chen, L., et al. (2023). Large language models are not robust multiple choice selectors. *arXiv:2309.03882*. https://arxiv.org/abs/2309.03882
- Bavaresco, A., Bernardi, R., Bertolazzi, L., et al. (2024). LLMs instead of human judges? A large scale empirical study across 20 NLP evaluation tasks. *arXiv:2406.18403*. https://arxiv.org/abs/2406.18403

---

## 12 — Positional Bias Detection

**What it is.** In pairwise evaluation, a judge may prefer whichever output appears in the first position ("A"), regardless of quality. This is detected by running each case in two orderings — (A, B) and (B, A) — and checking whether the first-positioned output wins in both rounds. A positional flip is flagged when the first-positioned output wins in **both** orderings, indicating position preference rather than content preference.

**In judge-kappa.** `PositionalBiasDetector` in `bias/positional.py`. `TournamentEvaluator` in `tournament.py` also detects per-pair positional bias when `detect_positional_bias=True`. v0.2 corrected a bug where the direction of the positional flip was inverted.

**Primary sources.**
- Ko, J., Kim, S., Kim, H., et al. (2020). Look at the first sentence: Position bias in question answering. In *Proceedings of EMNLP 2020* (pp. 6791–6800). https://doi.org/10.18653/v1/2020.emnlp-main.549
- Wang, P., Li, L., Chen, L., et al. (2023). Large language models are not robust multiple choice selectors. *arXiv:2309.03882*. https://arxiv.org/abs/2309.03882
- Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems*, 36. https://arxiv.org/abs/2306.05685 (Section 4.1: positional bias analysis)

---

## 13 — Verbosity Bias Detection

**What it is.** An LLM judge may systematically reward longer outputs regardless of their content quality ("length bias" or "verbosity bias"). Detected via Spearman rank correlation between output token count and judge score. A significant positive ρ indicates the judge is rewarding length, not quality.

**In judge-kappa.** `VerbosityBiasDetector` in `bias/verbosity.py`. Flags when `|ρ| > threshold` AND `p < 0.05`. Default threshold 0.30; stricter 0.25 recommended for regulatory use. v0.2 documents tiktoken integration for accurate token counts (`pip install judge-kappa[tiktoken]`).

**Primary sources.**
- Saito, I., Oseki, Y., & Noji, H. (2023). Verbosity bias in preference labeling by large language models. *arXiv:2310.10076*. https://arxiv.org/abs/2310.10076
- Singhal, K., Tu, T., Gottweis, J., et al. (2023). Towards expert-level medical question answering with large language models. *arXiv:2305.09617*. https://arxiv.org/abs/2305.09617
- Dubois, Y., Galambosi, B., Liang, P., & Hashimoto, T. B. (2024). Length-controlled AlpacaEval: A simple way to debias automatic evaluators. *arXiv:2404.04475*. https://arxiv.org/abs/2404.04475

---

## 14 — Spearman Rank Correlation

**What it is.** A non-parametric measure of monotonic association between two ranked variables. The Spearman coefficient ρ is the Pearson correlation of the rank vectors. Unlike Pearson r, Spearman ρ is robust to outliers and non-linear monotonic relationships. The significance test uses a t-distribution approximation with n − 2 degrees of freedom.

**In judge-kappa.** Used by `VerbosityBiasDetector` via `scipy.stats.spearmanr(token_counts, scores)`. Also used in `examples/demo_rank_judge.py` to validate leaderboard accuracy against ground truth.

**Primary sources.**
- Spearman, C. (1904). The proof and measurement of association between two things. *The American Journal of Psychology*, 15(1), 72–101. https://doi.org/10.2307/1412159
- Zar, J. H. (2010). *Biostatistical Analysis* (5th ed.). Pearson Prentice-Hall. (Chapter 19: Spearman rank correlation)

---

## 15 — Behavioral Alignment Metric

**What it is.** A repurposing of Krippendorff α from inter-rater reliability to cross-condition behavioral consistency. Instead of measuring whether multiple judges agree on the same output, it measures whether the same model produces consistent outputs across different evaluation conditions (e.g., zero-shot vs. system-prompted vs. chain-of-thought). Low α signals high condition sensitivity — a deployment risk when users prompt in varied ways. This reframing is methodologically equivalent to Krippendorff α but with conditions as "raters" and model outputs as "ratings."

**In judge-kappa.** `BehavioralAlignmentMetric` in `agreement/behavioral.py`. `from_condition_scores()` convenience constructor builds verdicts from a `{condition: {case_id: score}}` dict. Motivated by the DISC (Discriminative System Conditioning) methodology for LLM behavioral analysis.

**Primary sources.**
- Krippendorff, K. (2004). Reliability in content analysis (cited above — foundational α method)
- Perez, E., Huang, S., Song, F., et al. (2022). Red teaming language models with language models. *arXiv:2202.03286*. https://arxiv.org/abs/2202.03286 (motivation for cross-condition behavioral testing)
- Mizrahi, M., Kaplan, G., Malkin, D., et al. (2024). State of what art? A call for multi-prompt LLM evaluation. *Transactions on Machine Learning Research*. https://arxiv.org/abs/2401.00595

---

## 16 — Listwise Learning to Rank

**What it is.** A ranking paradigm where all candidate documents (here: system outputs) are ranked in a single pass by a scoring function (here: the judge LLM), rather than through pairwise comparisons. Rank-to-score normalisation maps the discrete rank 1..N to a continuous score in [0, 1]: score_i = (N − rank_i) / (N − 1). This preserves ordinal information while producing values compatible with standard evaluation pipelines.

**In judge-kappa.** `RankJudge` in `judges/rank.py`. The judge receives all N outputs labelled A, B, C, … in a single prompt and returns a ranking JSON. Outputs are presented in the order provided by the caller (no shuffle is applied; users should shuffle `system_outputs.keys()` if they want randomised presentation order). Recommended for N ≥ 7; `TournamentEvaluator` remains preferred for N ≤ 6.

**Primary sources.**
- Cao, Z., Qin, T., Liu, T.-Y., et al. (2007). Learning to rank: From pairwise approach to listwise approach. In *Proceedings of ICML 2007* (pp. 129–136). https://doi.org/10.1145/1273496.1273513
- Liu, T.-Y. (2009). Learning to rank for information retrieval. *Foundations and Trends in Information Retrieval*, 3(3), 225–331. https://doi.org/10.1561/1500000016
- Liu, Y., Iter, D., Xu, Y., et al. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment. In *Proceedings of EMNLP 2023*. https://arxiv.org/abs/2303.16634

---

## 17 — Softmax Weighting

**What it is.** A function that converts a vector of real-valued scores (here: IRT θ estimates) into a probability distribution — non-negative weights summing to 1.0. The softmax of θ_i is exp(θ_i) / Σ_j exp(θ_j). Ensures that even low-reliability judges receive a small positive weight rather than zero, which prevents over-reliance on any single judge when IRT estimates have high uncertainty.

**In judge-kappa.** `IRTJudgeWeighter.weights()` applies softmax to per-judge θ values. Passed directly as `weights` to `JudgePanel(strategy=AggregationStrategy.WEIGHTED_MEAN, weights=[...])`.

**Primary sources.**
- Bridle, J. S. (1990). Probabilistic interpretation of feedforward classification network outputs, with relationships to statistical pattern recognition. In F. Fogelman-Soulié & J. Hérault (Eds.), *Neurocomputing: Algorithms, Architectures and Applications* (pp. 227–236). Springer. https://doi.org/10.1007/978-3-642-76153-9_28
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning* (Chapter 6.2.2.3: Softmax). MIT Press. https://www.deeplearningbook.org

---

## 18 — Classical Test Theory (CTT) — Background

**What it is.** The foundational psychometric framework that underpins Krippendorff α and Cohen's κ. CTT models an observed score as the sum of a true score plus measurement error: X = T + E. Reliability is the ratio of true score variance to observed score variance. Cronbach's α (not the same as Krippendorff's α) measures internal consistency of a test with multiple items.

**In judge-kappa.** CTT concepts motivate the use of multi-judge panels to reduce measurement error (more judges = lower error variance). The guard `if len(ctrl_verdicts) >= 2` before per-case α computation reflects the CTT insight that reliability cannot be estimated from a single rater.

**Primary sources.**
- Lord, F. M., & Novick, M. R. (1968). *Statistical Theories of Mental Test Scores*. Addison-Wesley.
- Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests. *Psychometrika*, 16(3), 297–334. https://doi.org/10.1007/BF02310555

---

## Consolidated bibliography

| # | Method | Key reference |
|---|---|---|
| 1 | Krippendorff's α | Krippendorff (2004) *Human Communication Research* |
| 2 | Cohen's κ | Cohen (1960) *Educational and Psychological Measurement* |
| 3 | ICC(2,k) | Shrout & Fleiss (1979) *Psychological Bulletin* |
| 4 | Outfit MNSQ / person-fit | Wright & Masters (1982) *Rating Scale Analysis* |
| 5 | Bootstrap CI | Efron (1979) *Annals of Statistics* |
| 6 | McNemar test | McNemar (1947) *Psychometrika* |
| 7 | DIF / logistic regression | Swaminathan & Rogers (1990) *Journal of Educational Measurement* |
| 8 | IRT 2PL model | Birnbaum (1968); Fonseca Rivera et al. (2026) *arXiv:2608.05086* |
| 9 | Elo rating | Elo (1978); Zheng et al. (2023) *arXiv:2306.05685* |
| 10 | ICL calibration | Brown et al. (2020) *NeurIPS*; Kim et al. (2024) *ICLR* |
| 11 | LLM-as-judge | Zheng et al. (2023) *NeurIPS*; Shankar et al. (2024) *UIST* |
| 12 | Positional bias | Ko et al. (2020) *EMNLP*; Wang et al. (2023) *arXiv:2309.03882* |
| 13 | Verbosity bias | Saito et al. (2023) *arXiv:2310.10076*; Dubois et al. (2024) *arXiv:2404.04475* |
| 14 | Spearman ρ | Spearman (1904) *American Journal of Psychology* |
| 15 | Behavioral alignment | Krippendorff (2004); Mizrahi et al. (2024) *TMLR* |
| 16 | Listwise ranking | Cao et al. (2007) *ICML*; Liu et al. (2023) *EMNLP* |
| 17 | Softmax weighting | Bridle (1990); Goodfellow et al. (2016) *Deep Learning* |
| 18 | Classical Test Theory | Lord & Novick (1968); Cronbach (1951) *Psychometrika* |
