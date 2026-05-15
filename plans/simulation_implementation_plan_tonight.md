# Tonight MVP Implementation Plan

**Created:** 2026-05-13
**Target:** QME conference working paper, May 15 deadline
**Status:** Phase 1B – Structural Simulation MVP (no LLM calls)

This plan replaces `simulation_implementation_plan.md` as the active plan for tonight. The full LLM pipeline described in that document is deferred to Phase 2 (post-deadline).

---

## 1. Tonight's goal

Produce a minimal, defensible structural simulation that demonstrates the central empirical claims of *Unbundling LLM Recommenders*. The simulation uses the already-generated catalogs, consumer profiles, and Q_ij fit scores as ground truth, and simulates recommendation, expression, and outcome behavior with hand-specified kernels. No LLM calls are made.

**Deliverable:** one headline figure plus three tables that map cleanly onto the v2 theory propositions.

## 2. What is implemented tonight

1. `src/core/data_io.py` — loaders and validation for catalogs, consumers, fit scores
2. `src/02_simulate_core_mvp.py` — one-shot bundled policy and modular 2×2 policy simulation, plus logistic outcome DGP
3. `src/03_estimate_core_mvp.py` — three estimator tables (total-effect A/B, modular decomposition, naive vs. oracle vs. modular for the persuasion effect)
4. `src/04_make_core_figures.py` — λ_fit sweep figure with Monte Carlo replication and SE bands
5. Result tables in `results/tables/`, figures in `results/figures/`
6. A simulation report in `results/report_tonight.md` and `results/report_tonight.html`

## 3. What is deliberately NOT implemented tonight

- Full LLM generation pipeline (`Qwen2.5-32B`, vLLM, JSON-mode, evaluator)
- Query generation, real prompts, real recommendations
- Expression-feature extraction from generated text
- Five DGP "worlds" — only the `naive_regression_failure` DGP is needed for the headline result
- Bridge diagnostics between one-shot and modular kernels (deferred until LLM audit)
- Category-by-category heterogeneity beyond pooled tables (defer)
- Bootstrap SEs (analytical cell-contrast SEs are sufficient)
- Power-analysis sensitivity grids over (σ_R, λ_fit) jointly
- Human experiments
- Runner / CLI wrapper script
- Stochastic Opportunity reframing — this paper remains *Unbundling*

## 4. Theory mapping

Each output corresponds to a specific proposition in `temp/unbundling_llm_recommenders_theory_v2.tex`:

| Output | v2 proposition |
|---|---|
| `one_shot_total_effect.csv` | Prop 1 — prompt randomization identifies the total one-shot policy effect |
| `decomposition_mvp.csv` | Prop 3 — modular 2×2 randomization identifies retrieval, persuasion, interaction |
| `naive_vs_oracle_persuasion.csv` | Prop 4 — naive realized-expression regression misattributes product fit to persuasion |
| `naive_bias_vs_lambda.png` | Prop 4 — bias of the naive estimator grows with λ_fit (endogenous-expression strength) |
| The fact that one-shot has only one knob (z) but modular has two (q, r) | Prop 2 — one-shot prompt randomization does not generally identify retrieval and persuasion components |

## 5. DGP design

### 5.1 Inputs from already-generated data

For each of three categories `c ∈ {phone_charger, headphones, laptop}`:

- 1000 consumer profiles (`data/consumers/<c>.json`)
- Product catalog with `incumbent` and `focal_brand` indicators (`data/catalogs/<c>.json`)
- Latent fit scores Q_ij ∈ [0, 1] (`data/fit_scores/<c>.csv`)

### 5.2 Within-category Q standardization

For each (consumer i, product j) pair, define `Q_std_ij = (Q_ij - mean_c(Q)) / sd_c(Q)`. This puts Q on the same scale as 0/1 brand indicators and prevents the brand dummy from swamping fit.

### 5.3 Retrieval kernel (the source of selection)

Baseline retrieval (modular q=0 and one-shot z=0):

```
score_baseline_ij = a_Q · Q_std_ij + a_incumbent · incumbent_j + σ_R · ε_ij
```

Brand-forward retrieval (modular q=1 and one-shot z=1):

```
score_focal_ij = a_Q · Q_std_ij + a_incumbent · incumbent_j + a_focal · focal_brand_j + σ_R · ε_ij
```

Selected product J_i = argmax_j score_ij over products in the category. **No `credible_ij` gate** — let the brand-forward policy genuinely substitute brand for fit and let the outcome equation pay the cost.

Defaults: `a_Q = 1.0, a_incumbent = 0.20, a_focal = 0.50, σ_R = 0.35`.

### 5.4 Expression kernel (where the bundle asymmetry lives)

One-shot expression (researcher observes only z, not separate retrieval/expression):

```
E_i(z) = e0 + τ_backend · z + λ_fit · Q_std_iJ + λ_inc · incumbent_J + σ_E · η_i
```

Modular expression (researcher randomizes r independently of q):

```
E_i(q, r) = e0 + τ_R · r + λ_fit · Q_std_iJ + λ_inc · incumbent_J + σ_E · η_i
```

**Bundle asymmetry:** In one-shot, the backend prompt z=1 couples brand-forward retrieval (via `a_focal`) AND stronger expression (via `τ_backend`) — the researcher cannot tell which channel produced the lift. In modular, q and r are independent. This operationalizes Prop 2.

Defaults: `e0 = 0, τ_backend = 0.5, τ_R = 0.5, λ_fit = 1.0 (main run), λ_inc = 0.2, σ_E = 0.5`.

### 5.5 Outcome DGP (naive_regression_failure)

```
U_i = β_0 + β_Q · Q_std_iJ + β_E · E_i + β_QE · Q_std_iJ · E_i
      + β_inc · incumbent_J + β_incE · incumbent_J · E_i
Y_prob_i = sigmoid(U_i)
Y_i ~ Bernoulli(Y_prob_i)
```

Defaults: `β_0 = -0.8, β_Q = 0.7, β_E = 0.5, β_QE = 0.0, β_inc = 0.25, β_incE = -0.15`.

**Power note:** I bumped β_E from GPT's 0.35 to 0.5. With base rate ≈ 0.3 and standardized E (sd ≈ 0.5), the marginal persuasion effect in probability units is roughly β_E × σ_E × p(1−p) ≈ 0.5 × 0.5 × 0.21 ≈ 0.05 (5 percentage points). At N=3000 pooled with cell-pair SE ≈ 0.024, this gives ≈ 2 SE of signal per replication — comfortably detectable.

For **true** effects we compute potential outcomes Y_prob under each modular cell; for **estimator** outputs we use realized Y.

### 5.6 Seed handling

Master seed = 20260513. Spawn four independent streams:

```python
master = np.random.SeedSequence(20260513)
rng_retrieval, rng_expression, rng_outcome, rng_oneshot = [
    np.random.default_rng(s) for s in master.spawn(4)
]
```

This guarantees independent noise across the four layers and across λ values in the sweep.

## 6. Estimation

### 6.1 Table 1 — one-shot total effect

```
Δ_total_hat = mean(Y | z=1) − mean(Y | z=0)
Δ_total_true = mean(Y_prob(z=1) − Y_prob(z=0))
```

By category and pooled. HC1 SEs from a regression of Y on z with category fixed effects.

**Interpretation:** Prop 1. A/B identifies total effect but says nothing about retrieval/expression decomposition.

### 6.2 Table 2 — modular decomposition

Cell-mean contrasts on the modular sample:

```
Δ_R_hat    = mean(Y | q=1, r=0) − mean(Y | q=0, r=0)
Δ_P_hat    = mean(Y | q=0, r=1) − mean(Y | q=0, r=0)
Δ_R×P_hat  = mean(Y | q=1, r=1) − mean(Y | q=1, r=0) − mean(Y | q=0, r=1) + mean(Y | q=0, r=0)
```

True values computed identically using Y_prob. Analytical SEs `sqrt(s²_{q,r}/n_{q,r} + s²_{q',r'}/n_{q',r'})` for the difference, with proper combination for the four-cell interaction.

**Interpretation:** Prop 3. Modular randomization identifies the three components.

### 6.3 Table 3 — naive vs. oracle vs. modular for the persuasion effect

Fit two linear-probability regressions on the modular data:

```
Naive:  Y ~ E + incumbent + focal + category FE        (HC1 SEs)
Oracle: Y ~ E + Q_std_J + incumbent + focal + category FE  (HC1 SEs)
```

The coefficient on E is the estimator of interest. Plus the modular cell-mean Δ_P from Table 2.

True persuasion effect for comparison purposes is computed from potential outcomes.

**Interpretation:** Prop 4. The naive coefficient absorbs the latent-fit channel and is biased upward when λ_fit > 0; the oracle controls for Q_std_J and is approximately unbiased; the modular cell-mean contrast is correct by design.

## 7. Headline figure — `naive_bias_vs_lambda.png`

For `λ_fit ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}` and `M = 25` Monte Carlo replications per λ value:

1. Hold retrieval realizations fixed across reps (selected products do not depend on λ_fit).
2. Resample expression noise η and outcome Bernoulli draws within each rep.
3. Compute the naive coefficient, oracle coefficient, and modular cell-mean Δ_P.
4. Compute bias = estimator − true Δ_P.
5. Plot mean bias vs. λ_fit with ±1 SE shaded bands.

Three lines on one panel: naive, oracle, modular. Expected pattern: naive bias monotonically increases in λ_fit; oracle and modular stay near zero.

## 8. Validation / stopping rules

Fail loudly during the run if:

- any selected `product_id` is not in the category catalog
- any modular cell has unequal observation count (off by more than 1 across cells in the pooled sample)
- pooled outcome rate ∉ [0.10, 0.90]
- expression intensity has near-zero variance (sd < 0.05)
- `q=1` shifts no products (selected-product TVD between q=0 and q=1 < 0.02)
- `r=1` does not raise mean E by at least 0.05 relative to `r=0`
- `corr(E, Q_std_J)` < 0.10 at `λ_fit ≥ 0.5`
- naive bias does not rise monotonically with λ_fit over the seven points

If any check fails, abort, print the diagnostic, and require manual review before producing the figure.

## 9. Cuts and priorities

Priority order for what to keep if time runs short:

1. **Must have:** `decomposition_mvp.csv` and `naive_bias_vs_lambda.png`
2. **Must have:** `naive_vs_oracle_persuasion.csv`
3. **Should have:** `one_shot_total_effect.csv`
4. **Defer:** category heterogeneity tables
5. **Defer:** secondary `modular_decomposition_by_category.png`
6. **Defer:** runner CLI script and bootstrap SEs

## 10. Differences from GPT's revised plan

I made five edits before implementing:

1. **MC reps with SE bands on the headline figure** (M=25). Single-realization curves would have visible wiggles that reviewers will misread.
2. **Bumped β_E from 0.35 to 0.5.** Power calculation showed the original value was at the edge of cell-contrast detectability.
3. **Cut the secondary category-decomposition figure and the runner script** for tonight.
4. **Use `np.random.SeedSequence.spawn(4)`** for four independent noise streams (retrieval / expression / outcome / one-shot).
5. **Annotated each estimator with its v2 proposition** so the report cross-references theory directly.

## 11. After tonight

If the simulation produces the expected pattern, the path to May 15 is:

1. May 14: write the simulation section of the working paper using these tables and figures
2. May 14: assemble the QME conference submission packet
3. May 15: proofread and submit

If the simulation fails (e.g., the naive-bias monotonicity check fires), the priority is to debug parameter calibration, not to add features.
