# Unbundling LLM Recommenders

## Project Structure

Four simulation studies, ordered chronologically. Each section lists its
scripts, input data, and output locations.

---

### Shared inputs

```
data/
  catalogs/                 Product catalogs (3 categories: headphones, laptop, phone_charger)
    {category}.json
  consumers/                Consumer profiles (1000 per category)
    {category}.json
  fit_scores/               Consumer-product fit scores (Q, Q_std)
    {category}.csv

src/
  00_generate_catalogs.py   Generate product catalogs
  01_generate_consumers.py  Generate consumer profiles and fit scores
  core/
    data_io.py              Shared data-loading utilities for simulations 1-3
```

---

### Simulation 1 — Synthetic DGP

Pure-Python structural simulation. No LLM calls. Generates ground-truth
data from a known DGP to validate the factorized estimator against naive
and oracle benchmarks.

```
Scripts:
  src/02_simulate_core_mvp.py       Core MVP: modular + one-shot pipelines
  src/03_estimate_core_mvp.py       Estimate decomposition from simulated data
  src/04_make_core_figures.py        Naive-bias and lambda-sweep figures
  src/05_robustness_dgp.py          Robustness checks across DGP parameters

Data:
  data/simulated/
    modular_mvp.csv                 Modular pipeline outputs
    modular_robustness.csv          Robustness sweep outputs
    one_shot_mvp.csv                One-shot pipeline outputs
    one_shot_robustness.csv         Robustness sweep one-shot outputs

Results:
  results/tables/
    decomposition_mvp.csv           Core decomposition estimates
    one_shot_total_effect.csv       One-shot total effect
    naive_vs_oracle_persuasion.csv  Naive vs oracle estimator comparison
    robustness_*.csv                Robustness sweep tables
    lambda_sweep_*.csv              Lambda sensitivity analysis
  results/figures/
    naive_bias_vs_lambda.png        Bias as function of retrieval-expression coupling
```

---

### Simulation 2 — LLM Recommender (semi-synthetic)

Local Ollama LLM generates recommendations; LLM evaluator scores expression
quality (PI, TD); structural DGP produces demand outcomes. Validates that
LLM-generated text creates measurable retrieval and expression variation.

```
Scripts:
  src/06_llm_simulation.py          Generate recommendations via Ollama (qwen2.5:14b)
  src/07_estimate_llm.py            Estimate decomposition from LLM outputs
  src/08_report_llm.py              Generate summary report
  src/09_llm_evaluator.py           Score PI/TD/fit_specificity via Ollama
  src/10_manual_coding_sample.py    Draw sample for manual validation
  src/11_manual_vs_evaluator.py     Compare manual vs LLM evaluator scores
  src/12_evaluator_diagnostics.py   Evaluator reliability diagnostics
  src/13_mechanism_audit_robust.py  Mechanism audit with clustered inference
  src/14_semisynthetic_robust.py    Semi-synthetic demand with validated scales

Data:
  data/llm_sim/
    modular_llm.csv                 Modular pipeline LLM outputs
    one_shot_llm.csv                One-shot pipeline LLM outputs
    retrieval_llm.csv               Retrieval-only LLM outputs
  data/llm_eval/
    raw/{category}.jsonl            Raw evaluator responses
  data/semisynthetic/
    validated_scale_outcomes.csv    Semi-synthetic demand outcomes
    validated_scale_outcomes_mc.csv Monte Carlo replications

Results:
  results/tables/
    llm_decomposition.csv           LLM-based decomposition
    llm_naive_vs_oracle.csv         Naive vs oracle with LLM data
    llm_one_shot_total_effect.csv   One-shot total effect (LLM)
    expression_fit_correlations.csv PI/TD correlations with fit
    validated_*.csv                 Semi-synthetic validated estimates
  results/reports/
    validated_mechanism_audit.md            Mechanism audit report
    validated_mechanism_audit_robust.md     Robust mechanism audit
    validated_semisynthetic_results.md      Semi-synthetic results
    validated_semisynthetic_results_robust.md
    fit_specificity_redesign_plan.md        Evaluator redesign notes
  results/diagnostics/
    evaluator_scores.csv            Raw evaluator scores
    evaluator_diagnostic_*.csv      Reliability metrics
    evaluator_*.md                  Validation reports
    manual_coding_*.csv             Manual coding data
    manual_expression_audit.*       Expression audit
    step1_diagnosis.md              Initial diagnostic
  results/report_llm.md            LLM simulation summary
  results/report_tonight.md        Overnight run summary
```

---

### Simulation 3 — Architecture Diagnostic (unified vs two-stage)

Compares unified single-prompt and two-stage (selector + writer) architectures.
Local Ollama version and GPU/vLLM version for larger models.

```
Scripts (local Ollama):
  src/15_diagnostic_supply.py       Supply: unified + two-stage recommendations
  src/16_diagnostic_evaluate.py     Evaluator: PI/TD/fit_specificity scoring
  src/17_diagnostic_demand.py       Demand: simulated consumer response

Scripts (GPU/vLLM — self-contained, uploadable):
  src/gpu_vllm/
    supply.py                       vLLM supply (Qwen2.5-32B-Instruct-AWQ)
    evaluate.py                     vLLM evaluator (same model)
    demand.py                       vLLM demand (gemma-2-9b-it)
    pyproject.toml                  uv dependency spec
    requirements.txt                pip fallback
    README.md                       GPU setup and run instructions
    data/                           Bundled input data (catalogs, consumers, fit_scores)

Data:
  data/diagnostic/
    supply_outputs.csv              Product selections + recommendation text
    evaluator_outputs.csv           PI, TD, fit_specificity scores
    demand_outputs.csv              Purchase likelihood, fit, trust, risk
    manifest.csv                    Call tracking with timestamps
    raw/                            Per-call JSON responses (supply)
    eval_raw/                       Evaluator JSONL cache
    demand_raw/                     Demand JSONL cache
```

---

### Simulation 4 — History Shock (policy decomposition)

Four-cell modular audit: the policy shock is access to historical purchase
data. Two-stage architecture with 2x2 factorial (generic/history selector x
generic/history writer). Pairwise demand-side LLM comparisons.

```
Scripts:
  src/history_shock/
    utils.py                        Shared utilities, Ollama wrapper, seed computation
    prompts.py                      All prompt templates (selector, writer, evaluator, pairwise)
    01_generate_purchase_history.py  Transparent DGP: purchase logs + aggregation
    02_generate_audit_cells.py      Four-cell audit (selector + writer, Ollama)
    03_score_expression.py          PI/TD evaluator scoring
    04_pairwise_demand.py           6 pairwise comparisons per cluster (gemma2:9b)
    05_analyze_decomposition.py     Basic decomposition + cell invariant check
    06_decomposition_audit.py       Rigorous audit: Bradley-Terry, bootstrap CIs

Data:
  data/history_shock/
    {category}_purchase_logs.csv    Individual-level purchase logs (DGP)
    {category}_product_history.csv  Product-level aggregate history
    {category}_segment_history.csv  Segment x product aggregate history
    history_summary.md              DGP parameter summary
    audit_supply.csv                All audit cell outputs (120 rows)
    evaluator_scores.csv            Evaluator scores (120 rows)
    pairwise_demand.csv             Pairwise demand results (180 rows)
    raw_supply/                     Supply + evaluator JSONL caches
    pairwise_demand_raw/            Demand JSONL cache

Results:
  results/history_shock/
    decomposition_audit_report.md   Full audit report
    tables/
      pairwise_all_comparisons.csv  All 6 pairwise win/tie/loss rates
      cell_utilities.csv            Per-cluster Bradley-Terry + pairwise-score utilities
      decomposition_effects.csv     Retrieval / expression / interaction estimates
      retrieval_audit.csv           Q_std, price, quality, hist CR by cell
      expression_audit.csv          PI/TD merged with cell metadata
      decomposition_summary.csv     Basic decomposition
      evaluator_means.csv           Evaluator means by cell
      pairwise_win_rates.csv        Win rates by pair
      retrieval_agreement.csv       Generic vs history selector agreement
    figures/                        (reserved)
```

---

### Paper

```
paper/
  unbundling_llm_recommenders.tex   Main manuscript
  references.bib                    Bibliography
```

---

### Other

```
plans/                              Planning documents (simulation, writeup)
literature/                         Reference papers (PDFs)
temp/
  prompt_history.md                 Session-level prompt/response log
  change_log.md                     File change log
  old-versions/                     Superseded drafts and proposals
CLAUDE.md                           Project instructions and guidelines
pyproject.toml                      Root-level Python project config
```
