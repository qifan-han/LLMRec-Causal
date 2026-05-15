# Semi-Synthetic LLM Recommender Plan

## Core decision

Phase 1 should no longer be the main evidence. It is useful as a clean validation/theory appendix, but it is not enough for a QME-style conference submission because it is essentially a stylized econometric simulation: a randomized 2x2 design decomposes retrieval and expression by construction, and omitted latent fit creates bias by construction. That is too close to a teaching example unless it is connected to real LLM-generated recommendation behavior.

The main paper should become a semi-synthetic empirical study of LLM recommender mechanisms.

The empirical object should be real generated recommendation outputs:
- which product the LLM recommends;
- how concentrated recommendations are across products/brands;
- how prompts shift retrieval;
- how prompts shift expression;
- whether expression is calibrated to consumer-product fit;
- whether brand or incumbent status changes tone, hedging, or tradeoff disclosure.

The synthetic component should be downstream consumer choice or welfare. That is acceptable if it is built on real generated recommendation corpora and used to compare estimators under known ground truth.

## What to learn from Feldman et al.

Feldman et al. are not merely running abstract textbook simulations. They build a semi-synthetic empirical pipeline:

1. Start from real text datasets.
2. Use LLM/SAE steering to generate quasi-counterfactual texts.
3. Measure ex-post treatment intensity.
4. Filter or score generated text quality/coherence.
5. Embed the generated texts.
6. Simulate outcomes with known ground truth.
7. Compare causal estimators under realistic generated-text variation.

The key lesson is: simulations become publishable when the hard part of the data-generating process is empirical. In Feldman, the text and treatment variation are real generated artifacts; only the final outcome is simulated. For this project, the analog is: retrieval choices and recommendation text should come from real LLMs; only consumer choices need to be simulated.

## Reframed contribution

Do not sell the paper as: “A 2x2 experiment identifies retrieval and persuasion effects.”

That is too trivial.

Sell the paper as:

“LLM shopping assistants bundle product retrieval and persuasive expression. Using a semi-synthetic empirical pipeline based on real LLM-generated recommendations, we show how prompt policies change product exposure, expression, tradeoff disclosure, and downstream choice estimates. Standard one-shot prompt A/B tests mask these mechanisms, and naive realized-expression regressions can misattribute product fit or brand favoritism to persuasion.”

## Target empirical findings

The next evidence should aim to establish concrete facts about real LLM recommender behavior.

Potential findings:

1. Retrieval concentration  
   A small number of products or brands receive a disproportionate share of recommendations.

2. Prompt-induced retrieval shift  
   Brand-forward or policy prompts change recommended product distributions, focal-brand share, incumbent share, average price, average quality, and average consumer-product fit.

3. Expression shift  
   Persuasive prompts increase endorsement strength, confidence, and positive framing; they may reduce tradeoff disclosure.

4. Fit calibration or miscalibration  
   The LLM may or may not write more confident, specific, and fit-justified text for better-fitting products. If it does not, that is itself an important finding: persuasion can be applied uniformly even to weak-fit recommendations.

5. Brand/expression favoritism  
   Conditional on fit, certain brands or incumbents may receive stronger endorsements or fewer caveats.

6. Bundling spillovers  
   A retrieval-oriented prompt may also change expression, even when expression style is nominally fixed. This supports the claim that one-shot backend prompts bundle mechanisms.

## Required external information and resources

Before implementing the next stage, collect or confirm:

1. API access
   - OpenAI API key or other frontier-model API key.
   - Store as environment variable, not hard-coded.
   - Example: OPENAI_API_KEY.

2. Model choice
   - Prefer a dated/snapshot model if available.
   - Also run one open-weight/local model for robustness if feasible.
   - Record exact model names.

3. Reproducibility metadata
   - Fixed seed if the API supports it.
   - Temperature, top_p, max tokens.
   - System prompt, user prompt, catalog order, consumer order.
   - Model name, timestamp, response id if available.
   - System fingerprint or equivalent backend version metadata if available.
   - Raw request and raw response JSON.

4. Existing data assets
   - Product catalogs.
   - Consumer profiles.
   - Existing latent fit scores Q_ij.
   - Product metadata including brand, price, quality, incumbent/focal indicators, attributes, and review summaries.

5. Python packages
   - pandas, numpy, statsmodels, scikit-learn.
   - API SDK if used.
   - sentence-transformers or API embeddings if doing text-embedding controls.
   - Optional: econml for R-learner style estimator comparisons.

6. Output storage
   - A raw generation directory.
   - A cleaned generation table.
   - An evaluator-score table.
   - A semi-synthetic outcome table.
   - Final tables and figures.

## Next implementation steps

### Step 0: Audit current code and data

Read the current scripts and outputs:
- `06_llm_simulation.py`
- `07_estimate_llm.py`
- current `manual_expression_audit.md`
- current `step1_diagnosis.md`
- existing catalogs, consumers, and fit-score files

Deliverable:
- A short note listing what can be reused and what must be replaced.

Expected conclusion:
- Reuse catalogs, consumers, fit scores, and some estimation code.
- Replace or heavily revise expression prompts, expression scoring, and main evidence framing.

### Step 1: Build a real generated recommendation corpus

For each category, consumer, model, retrieval policy, expression policy, and seed, generate:

- selected product id;
- top-k shortlist;
- recommendation text;
- rationale if available;
- raw model response;
- metadata.

Minimum design:

- Categories: phone charger, headphones, laptop.
- Consumers: start with 50 per category for pilot, then scale only if diagnostics look good.
- Models: one frontier API model plus one local open-weight model if feasible.
- Retrieval policy q:
  - q=0 neutral/baseline retrieval.
  - q=1 brand-forward or focal-brand retrieval.
- Expression policy r:
  - r=0 neutral/factual recommendation.
  - r=1 persuasive/confident recommendation.
- Seeds: at least 3 per consumer-policy cell if cost allows.

Important prompt fix:
- Expression generation must receive the full consumer profile and full selected-product information, including attributes and reviews, not only product id/brand/price/quality.
- Do not feed numeric Q_ij into the LLM.
- Let the model infer fit from consumer and product information.

Deliverables:
- `data/llm_corpus/raw/*.jsonl`
- `data/llm_corpus/recommendations_clean.csv`

### Step 2: Implement semantic expression evaluation

Do not rely on the current keyword heuristic as the main measure.

Use an LLM evaluator at temperature 0, or a hybrid evaluator, to score each recommendation text on 1–7 scales:

- endorsement strength;
- confidence;
- fit-specificity;
- consumer-need alignment;
- comparative strength;
- tradeoff disclosure;
- hedging/caution;
- brand emphasis;
- price/value emphasis;
- overall persuasive intensity.

The evaluator must see:
- consumer profile;
- selected product metadata;
- recommendation text.

It should return strict JSON.

Also keep the old keyword score only as a robustness/diagnostic measure.

Deliverables:
- `data/llm_corpus/expression_scores.csv`
- evaluator prompt file
- small validation audit of 20–40 rows comparing evaluator scores with manual judgment

### Step 3: Produce empirical mechanism tables

Before simulating outcomes, report facts about the LLM recommender corpus.

Tables/figures:

1. Retrieval concentration
   - Herfindahl index over products/brands.
   - Top product share.
   - Number of products accounting for 50/80/90 percent of recommendations.

2. Retrieval shift by q
   - Total variation distance between q=0 and q=1 product distributions.
   - Focal-brand share.
   - Incumbent share.
   - Mean price.
   - Mean product quality.
   - Mean selected Q_std.

3. Expression shift by r
   - Mean endorsement, confidence, fit-specificity, tradeoff disclosure, overall persuasion by r.
   - Within-cell standard deviations.

4. Fit-expression calibration
   - Regress expression scores on Q_std within q,r cells.
   - Correlations by category, model, and policy cell.
   - Test whether persuasive text is applied uniformly or calibrated to fit.

5. Brand-expression favoritism
   - Regress expression scores on brand/focal/incumbent indicators controlling for Q_std and category.

6. Bundling spillovers
   - Does changing q alter expression scores even holding r fixed?
   - Does changing r alter selected products in one-shot settings?

Deliverables:
- `results/tables/mechanism_*.csv`
- `results/figures/mechanism_*.png`
- `results/reports/mechanism_audit.md`

### Step 4: Construct semi-synthetic choice outcomes

Use real generated recommendations as inputs, but simulate consumer choice using known DGPs.

Outcome inputs:
- selected product fit Q_std;
- product price;
- product quality;
- incumbent/focal indicator;
- endorsement strength;
- confidence;
- fit-specificity;
- tradeoff disclosure;
- interaction between Q_std and expression;
- consumer persuasion susceptibility.

Construct multiple DGPs:

1. Persuasion-only DGP
   - Expression affects choice, independent of fit.

2. Fit-calibrated persuasion DGP
   - Expression matters more when product fit is high.

3. Misleading-persuasion DGP
   - Expression can increase choice even when fit is low, creating welfare loss.

4. Brand-amplification DGP
   - Brand-forward prompts raise exposure and endorsement for focal/incumbent brands.

5. Tradeoff-transparency DGP
   - More caveats reduce conversion but improve welfare/match quality.

Save both:
- purchase probability;
- realized Bernoulli outcome.

For small samples, use purchase probability as the main diagnostic and Bernoulli Y as secondary.

Deliverables:
- `data/semisynthetic/outcomes.csv`
- `results/tables/dgp_summary.csv`

### Step 5: Compare estimators

Compare:

1. One-shot prompt A/B
   - total effect only.

2. Modular 2x2 design
   - retrieval effect;
   - expression effect;
   - interaction.

3. Naive realized-expression regression
   - outcome on measured expression, without controlling for latent fit.

4. Oracle regression
   - controls for true Q_std; infeasible in real data but useful benchmark.

5. Text-embedding controls
   - control for full recommendation text embeddings.

6. Residualized text controls, Feldman-style
   - remove treatment/expression information from embeddings before using them as controls.

Main comparison:
- bias relative to known DGP truth;
- RMSE;
- coverage if applicable;
- stability across categories, models, and DGPs.

Deliverables:
- `results/tables/estimator_comparison.csv`
- `results/figures/bias_rmse_comparison.png`
- `results/reports/semisynthetic_results.md`

### Step 6: Decide the paper’s final evidence hierarchy

Expected hierarchy:

1. Main evidence:
   - real LLM recommender mechanism audit;
   - semi-synthetic outcome simulations using real generated recommendation corpus;
   - estimator comparison.

2. Secondary evidence:
   - current Phase 1 structural simulation as a simple validation appendix.

3. Not main evidence:
   - current Qwen-only pilot with keyword expression scoring.

## Red lines

Do not:
- present Phase 1 as the main evidence;
- scale the current weak Qwen/keyword pipeline;
- train a local model on a small set of GPT outputs;
- hard-code API keys;
- feed true Q_ij into the generator;
- rely only on binary realized Y when probabilities are available;
- claim empirical consumer effects without real consumer outcomes.

## Final expected paper claim

The strongest version of the paper should claim:

“LLM recommender policies affect consumer choice through multiple bundled mechanisms: product retrieval, persuasive expression, tradeoff disclosure, and brand amplification. A semi-synthetic empirical pipeline using real LLM-generated recommendations shows when conventional prompt A/B tests and realized-text regressions mislead, and why modular experiments or residualized text controls are needed to recover mechanism-specific effects.”

