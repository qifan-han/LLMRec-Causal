# Diagnostic Experiment Plan: Unified vs. Two-Stage LLM Recommender

**Date:** 2026-05-14 (revised)
**Revision:** v2 — corrected policy definitions, three-layer design, revised prompts and analysis.
**Status:** PLAN ONLY — do not implement until approved.
**Question:** Does a unified (one-stage) LLM recommender differ from a two-stage (selector + writer) LLM recommender in product selection, persuasive expression, or synthetic consumer response? And do those differences vary when the recommender is given explicit policy objectives?
**Why it matters:** If unified generation couples product selection and expression — for example, the model selects products it can write more persuasively about — that coupling motivates the modular audit design as a correction. If unified and two-stage produce indistinguishable outputs, the paper should frame the modular design as a practical decomposition/audit tool for policy shocks rather than as a correction for architecture-induced confounding.

---

## 1. Experimental Design

### 1.1 Architectures

**Architecture A: Unified (one-stage)**
- One LLM call per (consumer, policy) pair.
- The model sees the full catalog and consumer profile.
- It jointly selects a product and writes the recommendation in a single response.
- Output: `selected_product_id`, `shortlist`, `recommendation_text`, `selection_rationale`.

**Architecture B: Two-stage (selector + writer)**
- Two LLM calls per (consumer, policy) pair.
- **Stage 1 (Selector):** The model sees the full catalog and consumer profile. It selects a product and provides a brief selection rationale. It does NOT write recommendation text.
- **Stage 2 (Writer):** A separate call sees the consumer profile and the locked selected product metadata only (no catalog, no alternative products). It writes the recommendation text. It cannot change the product.
- Output: `selected_product_id` and `selection_rationale` from Stage 1; `recommendation_text` from Stage 2.

### 1.2 Policies

**Policy 0: Generic baseline**
- The LLM receives a consumer profile and product catalog.
- It is asked to recommend the single best product and explain the recommendation.
- It is NOT explicitly instructed to be consumer-first, business-objective, transparent, persuasive, brand-forward, or conversion-oriented.
- Represents the default behavior of a generic LLM shopping assistant.
- Minimal instruction: "Given the consumer profile and product catalog, recommend the single best product and explain your recommendation."
- Rationale: There is no truly intention-free recommender, because "recommend" already implies some implicit objective learned by the LLM during training. But the baseline should be as generic as possible so that we do not bake any specific normative objective into the control condition.

**Policy 1A: Consumer-centric shock**
- Ask the recommender to prioritize the consumer's needs, budget, use case, and preferences.
- Ask it to be balanced and transparent about tradeoffs.
- Corresponds to a firm asking: "What happens if we make the LLM recommender more consumer-centric?"

**Policy 1B: Business-objective / inventory-aware shock**
- Ask the recommender to recommend a product that reasonably fits the consumer while accounting for platform priorities such as focal brand, inventory, margin, sponsored products, or conversion.
- Ask it to remain plausible and non-deceptive.
- Corresponds to a firm asking: "What happens if we adapt the LLM recommender using inventory/business objectives?"

### 1.3 Data Design

| Dimension | Value |
|-----------|-------|
| Categories | 3 (phone_charger, headphones, laptop) |
| Consumers per category | 20 (consumer_id 0–19 from existing profiles) |
| Architectures | 2 (unified, two-stage) |
| Policies | 3 (generic baseline, consumer-centric, business-objective) |
| Total recommendation units | 3 × 20 × 2 × 3 = **360** |
| Unified supply calls | 3 × 20 × 3 = 180 |
| Two-stage selector calls | 3 × 20 × 3 = 180 |
| Two-stage writer calls | 3 × 20 × 3 = 180 |
| Total supply-side LLM calls | **540** |
| Evaluator calls (PI, TD) | **360** |
| Demand-side calls | **360** |
| Grand total LLM calls | **1,260** |

**Layer 1 only (Policy 0):**

| Dimension | Value |
|-----------|-------|
| Recommendation units | 3 × 20 × 2 = **120** |
| Unified supply calls | 60 |
| Two-stage selector calls | 60 |
| Two-stage writer calls | 60 |
| Total supply calls | **180** |
| Evaluator calls | **120** |
| Demand-side calls | **120** |
| Grand total | **420** |

**10 consumers per category (halved):**

| Scope | Full (3 policies) | Layer 1 only |
|-------|-------------------|-------------|
| Recommendation units | 180 | 60 |
| Total supply calls | 270 | 90 |
| Evaluator calls | 180 | 60 |
| Demand calls | 180 | 60 |
| Grand total | 630 | 210 |

### 1.4 Cell Structure

Each consumer appears in 6 cells: {unified, two-stage} × {policy_0, policy_1A, policy_1B}.
Cluster_id = (category, consumer_id), giving 60 clusters of size 6.

| Cell | Architecture | Policy | Supply calls | Eval calls | Demand calls |
|------|-------------|--------|-------------|-----------|-------------|
| 0 | Unified | 0 (generic) | 1 | 1 | 1 |
| 1 | Unified | 1A (consumer-centric) | 1 | 1 | 1 |
| 2 | Unified | 1B (business-obj) | 1 | 1 | 1 |
| 3 | Two-stage | 0 (generic) | 2 | 1 | 1 |
| 4 | Two-stage | 1A (consumer-centric) | 2 | 1 | 1 |
| 5 | Two-stage | 1B (business-obj) | 2 | 1 | 1 |

---

## 2. Three-Layer Design

### Layer 1: Generic Baseline Architecture Diagnostic

**Question:** Holding the objective fixed as generic recommendation, does unified one-stage generation differ from two-stage selector-then-writer generation?

**Run:** Unified and two-stage, both under Policy 0 (generic baseline).

**Analyze:**
- Same-product rate
- Product distribution TVD
- Selected Q_std
- Focal/incumbent share
- Selected price/quality
- PI (persuasive_intensity)
- TD (tradeoff_disclosure)
- Synthetic purchase_likelihood, perceived_fit, trust, perceived_tradeoff_risk
- Whether expression scores are more correlated with selected product fit in unified than in two-stage (the coupling test)

**Why this is the cleanest test:** Policy 0 uses minimal instructions, so any difference between architectures is attributable to the architecture itself — not to how the LLM interprets a specific objective. If coupling exists even under a generic prompt, it is an inherent property of unified generation.

### Layer 2: Consumer-Centric Shock Extension

**Question:** Relative to the generic baseline, what changes when the LLM recommender is explicitly instructed to be more consumer-centric?

**Run:** For both architectures: Policy 0 (generic) and Policy 1A (consumer-centric).

**Analyze:**
- Does the consumer-centric shock improve selected Q_std?
- Does it increase tradeoff disclosure (TD)?
- Does it reduce persuasive intensity (PI)?
- Does it change synthetic purchase_likelihood, perceived_fit, or trust?
- Is the effect of the consumer-centric shock different in unified vs. two-stage architectures?

### Layer 3: Business-Objective / Inventory-Aware Shock Extension

**Question:** Relative to the generic baseline, what changes when the LLM recommender is given business/inventory/conversion priorities?

**Run:** For both architectures: Policy 0 (generic) and Policy 1B (business-objective).

**Analyze:**
- Does the business-objective shock increase focal-brand share?
- Does it reduce selected Q_std?
- Does it increase persuasive intensity (PI)?
- Does it reduce tradeoff disclosure (TD)?
- Does it increase synthetic purchase_likelihood?
- Does it increase expected mismatch or lower perceived trust?
- Is the effect of the business-objective shock different in unified vs. two-stage architectures?

---

## 3. Model Assignment

### 3.1 Supply-side

**Model:** qwen2.5:14b via Ollama (already installed, 8.99 GB)
**Rationale:** Same model for both architectures ensures any differences are attributable to architecture, not model capability.
**Temperature:** 0.0 (deterministic; isolates architecture effects from stochastic decoding variation)
**Robustness check:** Re-run a subset at temperature 0.7 to verify findings are not an artifact of greedy decoding. This is optional and should only be done after the main analysis is complete.
**Seed:** Deterministic per-call: `MASTER_SEED + consumer_id * 1000 + cell_index`

Cell index encoding:

| Architecture | Policy | Stage | cell_index |
|-------------|--------|-------|------------|
| Unified | 0 (generic) | — | 0 |
| Unified | 1A (consumer-centric) | — | 1 |
| Unified | 1B (business-obj) | — | 2 |
| Two-stage | 0 (generic) | selector | 10 |
| Two-stage | 0 (generic) | writer | 11 |
| Two-stage | 1A (consumer-centric) | selector | 12 |
| Two-stage | 1A (consumer-centric) | writer | 13 |
| Two-stage | 1B (business-obj) | selector | 14 |
| Two-stage | 1B (business-obj) | writer | 15 |

### 3.2 Evaluator

**Model:** qwen2.5:14b via Ollama
**Temperature:** 0.0
**Seed:** 99999 (matches existing evaluator)
**Scales:** PI and TD only (validated). fit_specificity is computed but labeled exploratory.
**Prompt:** Identical to existing `src/09_llm_evaluator.py` evaluator prompt.
**Rationale:** Reusing the validated evaluator ensures comparability with existing results. Cross-validation against 3 frontier models (rho = 0.84 for PI, 0.86 for TD) supports this choice.

### 3.3 Demand-side

**Primary recommendation:** gemma2:9b via Ollama (Google, different model family from Qwen)
**Fallback:** llama3.1:8b via Ollama (Meta, different family)
**API fallback:** GPT-4o-mini via OpenAI API (if local alternatives produce low-quality JSON)

**Rationale for a different model family:** The supply-side model (Qwen) generates recommendations. If the same model family simulates consumer response, it may implicitly "agree" with its own outputs — generating higher purchase likelihood for texts that follow its own generation patterns. A different model family (Gemma from Google) reduces this self-confirming bias.

**Temperature:** 0.0 (deterministic demand response)
**Seed:** 77777 (fixed, different from evaluator seed)

### 3.4 Model separation summary

| Role | Model | Temperature | Seed scheme | Family |
|------|-------|------------|-------------|--------|
| Supply (both architectures) | qwen2.5:14b | 0.0 | per-call deterministic | Qwen (Alibaba) |
| Evaluator (PI, TD) | qwen2.5:14b | 0.0 | 99999 fixed | Qwen (Alibaba) |
| Demand simulator | gemma2:9b | 0.0 | 77777 fixed | Gemma (Google) |

The evaluator shares a model family with the supply side, but this is acceptable because (a) the evaluator is externally validated, (b) the evaluator sees no architecture or policy label, and (c) the evaluator scores text properties (PI, TD), not consumer response. The demand simulator is a different family to avoid self-confirmation on the consumer-response dimension.

---

## 4. Prompt Templates

### 4.1 Unified Recommender (Architecture A)

```
SYSTEM:
You are a product recommendation engine. You have access to the following catalog:

{catalog_text}

{policy_instruction}

You MUST respond with valid JSON in this exact format:
{
  "selected_product_id": "<exact product_id from catalog>",
  "shortlist": ["<id1>", "<id2>", "<id3>"],
  "recommendation_text": "<60-80 words>",
  "selection_rationale": "<25 words max>"
}

IMPORTANT: All product IDs must be exact values from the catalog above.
Do not invent product IDs.

USER:
{consumer_text}
```

### 4.2 Two-Stage Selector (Architecture B, Stage 1)

```
SYSTEM:
You are a product selection engine. You have access to the following catalog:

{catalog_text}

{selection_instruction}

You MUST respond with valid JSON in this exact format:
{
  "selected_product_id": "<exact product_id from catalog>",
  "shortlist": ["<id1>", "<id2>", "<id3>"],
  "selection_rationale": "<25 words max>"
}

IMPORTANT: Do NOT write a recommendation. Do NOT describe the product to the
consumer. Only select the product and explain why briefly.

All product IDs must be exact values from the catalog above.

USER:
{consumer_text}
```

### 4.3 Two-Stage Writer (Architecture B, Stage 2)

```
SYSTEM:
You are a product recommendation writer. A product has already been selected
for this consumer. You cannot change the selection.

Selected product:
- Product ID: {product_id}
- Brand: {brand_name}
- Price: ${price}
- Quality score: {quality_score}/100
- Key attributes: {attributes_text}
- Reviews: {review_summary}
- Known weakness: {weakness}

{writing_instruction}

Write ONLY the recommendation text. Do not wrap it in JSON or add any metadata.
Output the recommendation text directly (60-80 words).

USER:
{consumer_text}
```

### 4.4 Policy-Specific Instructions

#### Policy 0: Generic baseline

**Unified `policy_instruction`:**
```
Given the consumer profile and product catalog, recommend the single best
product and explain your recommendation in 60-80 words.
```

**Two-stage selector `selection_instruction`:**
```
Given the consumer profile and product catalog, select the single best product.
```

**Two-stage writer `writing_instruction`:**
```
Write a 60-80 word recommendation explaining why this product was selected for
this consumer.
```

Note: Policy 0 instructions are deliberately minimal. They do not specify tone, balance, transparency, persuasion, brand preference, or any other normative objective. The LLM's default training behavior determines how it writes.

#### Policy 1A: Consumer-centric shock

**Unified `policy_instruction`:**
```
Recommend the product that best serves the consumer's needs, budget, use case,
and preferences. Be balanced and transparent. Mention relevant limitations or
tradeoffs when they matter.

Write the recommendation in 60-80 words.
```

**Two-stage selector `selection_instruction`:**
```
Select the product that best serves the consumer's needs, budget, use case,
and preferences. Prioritize product-consumer fit.
```

**Two-stage writer `writing_instruction`:**
```
Write a balanced, transparent 60-80 word recommendation. Explain why this
product fits the consumer. Mention relevant limitations or tradeoffs when
they matter.
```

#### Policy 1B: Business-objective / inventory-aware shock

**Unified `policy_instruction`:**
```
Recommend a product that reasonably fits the consumer while accounting for
platform business priorities. When two products serve the consumer comparably
well, prefer products from {focal_brand}. The recommendation should be
plausible and non-deceptive but can emphasize product strengths and
conversion potential.

Write the recommendation in 60-80 words.
```

**Two-stage selector `selection_instruction`:**
```
Select a product that reasonably fits the consumer while accounting for platform
business priorities. When two products serve the consumer comparably well,
prefer products from {focal_brand}.
```

**Two-stage writer `writing_instruction`:**
```
Write a 60-80 word recommendation. The recommendation should be plausible and
non-deceptive but can emphasize product strengths and conversion potential.
```

### 4.5 Key Prompt Design Decisions

| Design choice | Rationale |
|--------------|-----------|
| Policy 0 is minimal — no normative instruction | Ensures baseline reflects default LLM behavior, not a baked-in objective |
| Policy 1A and 1B are symmetric shocks relative to Policy 0 | Enables clean Policy 1X - Policy 0 contrasts |
| Same selection instruction across architectures (per policy) | Isolates architecture effect from instruction effect |
| Same writing instruction across architectures (per policy) | Same |
| Unified outputs JSON with all fields | Captures selection + text in one call |
| Two-stage selector outputs JSON, no text | Prevents selector from being influenced by text-writing objectives |
| Two-stage writer outputs raw text, no JSON | Avoids JSON formatting artifacts in text quality |
| Two-stage writer sees product metadata but NOT the full catalog | Cannot comparison-shop or change selection |
| No Q_std, no Y, no architecture label in any prompt | Prevents leakage |
| Consumer text identical across all architectures and policies | Controls for consumer presentation effects |
| Policy 1B focal_brand instruction goes to selector in two-stage | The selection nudge belongs at the selection stage, not the writing stage |

### 4.6 Evaluator Prompt (reused)

Identical to `src/09_llm_evaluator.py`. The evaluator input includes:
- Consumer profile (use_case, budget, sensitivities, brand familiarity)
- Selected product metadata (id, brand, price, quality, attributes, reviews, weakness)
- Recommendation text

The evaluator does NOT receive:
- Architecture label
- Policy label
- Q_std
- Any experimental variable

Output: `{"fit_specificity": int, "persuasive_intensity": int, "tradeoff_disclosure": int, "rationale": "..."}`

### 4.7 Demand-Side Simulator Prompt

```
SYSTEM:
You are simulating a realistic consumer response to a product recommendation.
You will receive a consumer profile and a product recommendation.

Based ONLY on the information provided, assess how this consumer would likely
respond to this recommendation.

Respond with valid JSON in this exact format:
{
  "purchase_likelihood": <integer 0-100>,
  "perceived_fit": <integer 1-7>,
  "trust": <integer 1-7>,
  "perceived_tradeoff_risk": <integer 1-7>,
  "rationale": "<25 words max>"
}

Scoring guidelines:
- purchase_likelihood: 0 = definitely would not buy, 100 = definitely would buy.
  Consider price vs budget, use-case match, product quality, and how convincing
  the recommendation is.
- perceived_fit: 1 = product clearly does not match needs, 7 = excellent match.
- trust: 1 = recommendation feels manipulative or misleading, 7 = trustworthy
  and balanced.
- perceived_tradeoff_risk: 1 = no concerns mentioned or apparent,
  7 = significant unaddressed concerns about product limitations.

Be realistic. Not all recommendations deserve high scores. Consider whether
the product actually fits the consumer's budget, use case, and preferences.

USER:
=== CONSUMER PROFILE ===
{consumer_text}

=== RECOMMENDED PRODUCT ===
Product: {product_id} by {brand_name}
Price: ${price}
Quality: {quality_score}/100
Key features: {attributes_summary}

=== RECOMMENDATION ===
{recommendation_text}
```

**What the demand model does NOT receive:**
- Architecture label (unified vs. two-stage)
- Policy label (generic, consumer-centric, business-objective)
- Q_std or any latent fit score
- Generator model name
- Evaluator scores (PI, TD)
- Any experimental hypothesis or framing

The demand-side consumer text includes only observable consumer characteristics: use_case, budget, price_sensitivity, quality_sensitivity, brand_familiarity.

**Note:** `would_choose` is not requested from the demand model. It is computed in post-processing if needed (e.g., `would_choose = 1 if purchase_likelihood >= 50 and perceived_fit >= 4`).

---

## 5. Output Schemas

### 5.1 Supply-Side Raw Output

**File:** `data/diagnostic/raw/{architecture}_p{policy}_{category}_{consumer_id}[_stage{1|2}].json`

```json
{
  "row_id": "unified_p0_phone_charger_003",
  "architecture": "unified",
  "policy": "p0",
  "category": "phone_charger",
  "consumer_id": 3,
  "timestamp": "2026-05-15T14:23:01Z",
  "model": "qwen2.5:14b",
  "temperature": 0.0,
  "seed": 20260514003000,
  "stage": "unified",
  "raw_response": "<full model response text>",
  "parsed": {
    "selected_product_id": "charger_anker_nano_67w",
    "shortlist": ["charger_anker_nano_67w", "charger_samsung_45w", "charger_baseus_65w"],
    "recommendation_text": "The Anker Nano 67W is ...",
    "selection_rationale": "Best balance of power and portability for this user's travel needs."
  },
  "parse_success": true,
  "latency_ms": 28450
}
```

For two-stage, two files per unit: `*_stage1.json` (selector) and `*_stage2.json` (writer).

### 5.2 Supply-Side Consolidated CSV

**File:** `data/diagnostic/supply_outputs.csv`

| Column | Type | Description |
|--------|------|-------------|
| row_id | str | `{architecture}_p{policy}_{category}_{consumer_id}` |
| architecture | str | "unified" or "two_stage" |
| policy | str | "p0", "p1a", or "p1b" |
| category | str | Category name |
| consumer_id | int | Consumer ID |
| selected_product_id | str | Product selected |
| shortlist | str | JSON array of top 3 |
| recommendation_text | str | 60–80 word text |
| selection_rationale | str | 25-word rationale |
| Q_std | float | Merged from fit_scores (NOT shown to any LLM) |
| price | float | Selected product price |
| quality_score | int | Selected product quality |
| incumbent | int | 1 if brand_status == "incumbent" |
| focal | int | 1 if focal_brand |
| text_length | int | Character count of recommendation |
| word_count | int | Word count |
| parse_success | bool | Whether JSON/text parsed correctly |
| seed | int | Seed used for this call |

### 5.3 Evaluator Output CSV

**File:** `data/diagnostic/evaluator_outputs.csv`

| Column | Type | Description |
|--------|------|-------------|
| row_id | str | Matches supply row_id |
| persuasive_intensity | int | 1–7 (validated) |
| tradeoff_disclosure | int | 1–7 (validated) |
| fit_specificity | int | 1–7 (exploratory only) |
| eval_rationale | str | Evaluator's brief explanation |

### 5.4 Demand-Side Output CSV

**File:** `data/diagnostic/demand_outputs.csv`

| Column | Type | Description |
|--------|------|-------------|
| row_id | str | Matches supply row_id |
| purchase_likelihood | int | 0–100 |
| perceived_fit | int | 1–7 |
| trust | int | 1–7 |
| perceived_tradeoff_risk | int | 1–7 |
| demand_rationale | str | Demand model's explanation |
| demand_model | str | "gemma2:9b" or fallback |
| demand_parse_success | bool | |

### 5.5 Merged Analysis Dataset

**File:** `data/diagnostic/analysis_dataset.csv`

All supply, evaluator, and demand columns merged by row_id. Additional computed columns:
- `cluster_id`: `{category}_{consumer_id}`
- `same_product_p0`: 1 if unified and two-stage selected the same product under Policy 0
- `same_product_p1a`: same for Policy 1A
- `same_product_p1b`: same for Policy 1B
- `PI_std`, `TD_std`: standardized within the diagnostic sample
- `Q_std_centered`: standardized Q_std
- `would_choose`: computed as `1 if purchase_likelihood >= 50 and perceived_fit >= 4 else 0`

---

## 6. Run Plan

### 6.0 Prerequisites

1. Confirm Ollama is running with qwen2.5:14b loaded.
2. Pull demand-side model: `ollama pull gemma2:9b` (~5.4 GB download).
3. Verify gemma2:9b produces valid JSON on a test prompt.
4. Create output directories.

### Step 1. Smoke Test — Policy 0 Only (~20 minutes)

**Scope:** 2 consumers × 1 category (phone_charger) × 2 architectures × Policy 0 = 4 recommendation units.
- Unified supply calls: 2
- Two-stage selector calls: 2
- Two-stage writer calls: 2
- Total supply calls: 6
- Evaluator calls: 4
- Demand calls: 4
- **Total calls: 14**

**Checks:**
1. Unified produces valid JSON with all required fields.
2. Two-stage selector produces valid JSON without recommendation text.
3. Two-stage writer produces coherent raw text (not JSON-wrapped).
4. All selected_product_ids exist in the phone_charger catalog.
5. Evaluator produces valid PI/TD scores for all 4 texts.
6. Demand-side model (gemma2:9b) produces valid JSON with all 4 numeric fields.
7. All fields are in expected ranges.
8. Recommendation texts are approximately 60–80 words (tolerance ± 20).
9. No architecture label, Q_std, or policy label leaks into any downstream prompt.
10. Raw responses are logged correctly.

**If smoke test fails:** Diagnose and fix before proceeding. Common failure modes: JSON parse errors (add retry), wrong product IDs (add validation), gemma2 format issues (try llama3.1:8b or API fallback).

### Step 2. Run Layer 1 Full Diagnostic — Policy 0

**Scope:** 20 consumers × 3 categories × 2 architectures × Policy 0 = 120 recommendation units.

**Supply calls:** 60 unified + 60 two-stage selector + 60 two-stage writer = 180.
**Evaluator calls:** 120.
**Demand calls:** 120.
**Total:** 420 calls.

**Run order:**
1. All unified Policy 0 calls (60), logging raw responses.
2. All two-stage selector Policy 0 calls (60), logging raw responses.
3. All two-stage writer Policy 0 calls (60), using Stage 1 product selections.
4. Merge supply outputs, attach product metadata and Q_std from fit_scores.
5. Run evaluator on all 120 recommendation texts.
6. Switch Ollama model to gemma2:9b.
7. Run demand-side simulation on all 120 texts.
8. Merge all outputs into analysis dataset.

**Estimated timing:** 180 supply × ~30s + 120 eval × ~15s + 120 demand × ~15s ≈ **3.5 hours**.

### Step 3. Analyze Layer 1

Produce Tables A1–A4 (see Section 7). Apply Layer 1 decision rule.

**Decision point:** If unified and two-stage differ materially under Policy 0, this supports the architecture-coupling story. Proceed to Layer 2. If they do not differ, proceed to Layer 2 anyway (policy shocks may reveal latent differences), but do not force the coupling story.

### Step 4. Run Layer 2 — Consumer-Centric Shock (Policy 1A)

**Scope:** Same 20 consumers × 3 categories × 2 architectures × Policy 1A = 120 additional units.

**Additional calls:** 180 supply + 120 eval + 120 demand = 420.
**Estimated timing:** ~3.5 hours.

### Step 5. If Time Allows, Run Layer 3 — Business-Objective Shock (Policy 1B)

**Scope:** Same consumers × Policy 1B = 120 additional units.
**Additional calls:** 420.
**Estimated timing:** ~3.5 hours.

### Step 6. Cross-Layer Analysis

For each architecture, compute policy shock effects:
- Policy 1A minus Policy 0 (consumer-centric shock)
- Policy 1B minus Policy 0 (business-objective shock)

Then test architecture × policy interactions:
- Does the consumer-centric shock have different effects in unified vs. two-stage?
- Does the business-objective shock have different effects in unified vs. two-stage?

Produce Tables B1–B3 and C1–C3 (see Section 7).

**Cache/resume behavior (all steps):**
- Each raw response is saved as an individual JSON file immediately after the call.
- Before each call, check if the output file exists. If yes, skip (resume from interruption).
- A manifest file `data/diagnostic/manifest.csv` tracks completed calls with timestamps.

### Validation Checks (run after each layer)

| Check | Criterion | Action if failed |
|-------|-----------|-----------------|
| Parse rate | ≥ 95% for each cell | Re-run failures with retry; if persistent, adjust prompt |
| Product validity | 100% product_ids in catalog | Re-run invalid rows |
| Text length | Mean 60–80 words, no row > 150 | Truncate or re-prompt |
| PI/TD range | All 1–7 | Re-run evaluator on failures |
| Demand JSON | All fields present, in range | Re-run demand on failures |
| Seed reproducibility | Re-run 5 random rows, compare output | If not identical, document and flag |
| No Q_std leakage | Grep all prompts for "Q_std", "fit_score", numeric Q values | Fatal if found |
| No label leakage | Grep demand prompts for "unified", "two-stage", "architecture", "stage", "generic", "consumer-centric", "business" | Fatal if found |
| Balance | Same 20 consumers appear in all cells per category | Fatal if missing cells |

---

## 7. Analysis Tables

### Layer 1 Tables (Policy 0 only, architecture comparison)

**Table A1: Product Selection Comparison (Policy 0)**

| Metric | Unified | Two-stage | Diff | Test |
|--------|---------|-----------|------|------|
| Same-product rate | — | — | (descriptive) | — |
| TVD (product distribution) | — | — | — | Bootstrap CI |
| Focal share | | | | McNemar paired |
| Incumbent share | | | | McNemar paired |
| Mean selected price | | | | Paired t |
| Mean selected quality | | | | Paired t |
| Mean selected Q_std | | | | Paired t |
| Product concentration (HHI) | | | | Bootstrap |

Broken down by category.

**Table A2: Expression Comparison (Policy 0)**

| Metric | Unified | Two-stage | Paired diff | SE (paired) | t | p |
|--------|---------|-----------|-------------|-------------|---|---|
| PI | | | | | | |
| TD | | | | | | |
| Word count | | | | | | |

Within-consumer paired differences. Cluster-robust SEs by cluster_id.

**Table A3: Expression Conditional on Same Product (Policy 0)**

Restrict to consumers where unified and two-stage selected the same product under Policy 0. If the same-product rate is < 30%, flag as low-power.

| Metric | Unified | Two-stage | Diff | p |
|--------|---------|-----------|------|---|
| PI | | | | |
| TD | | | | |

**Table A4: Expression-Fit Coupling (Policy 0)**

Key test: does the unified architecture show stronger correlation between selected Q_std and expression than the two-stage architecture?

By-architecture correlations:

| Metric | Unified | Two-stage | Diff test |
|--------|---------|-----------|-----------|
| corr(PI, Q_std) | | | Fisher z |
| corr(TD, Q_std) | | | Fisher z |

Full regression (cluster-robust by cluster_id):
```
PI ~ architecture * Q_std_centered + C(category)
TD ~ architecture * Q_std_centered + C(category)
```

If the `architecture × Q_std` interaction is significant, the unified architecture couples selection and expression.

**Table A5: Demand-Side Comparison (Policy 0)**

| Metric | Unified | Two-stage | Paired diff | SE (paired) | t | p |
|--------|---------|-----------|-------------|-------------|---|---|
| purchase_likelihood | | | | | | |
| perceived_fit | | | | | | |
| trust | | | | | | |
| perceived_tradeoff_risk | | | | | | |

**Table A6: Demand Conditional on Product and Expression (Policy 0)**

Regression (cluster-robust):
```
purchase_likelihood ~ architecture + PI_std + TD_std + Q_std_centered + C(category)
```

If the architecture coefficient is significant after controlling for product quality and expression scores, there may be residual style/format effects not captured by PI and TD.

### Layer 2 Tables (Policy 1A shock effects)

**Table B1: Policy 1A Shock — Retrieval Effects**

For each architecture separately:

| Metric | Policy 0 | Policy 1A | Shock (1A - 0) | SE (paired) | t | p |
|--------|----------|----------|---------------|-------------|---|---|
| Mean Q_std | | | | | | |
| Focal share | | | | | | |
| Incumbent share | | | | | | |
| Mean price | | | | | | |
| Mean quality | | | | | | |

**Table B2: Policy 1A Shock — Expression Effects**

| Metric | Policy 0 | Policy 1A | Shock (1A - 0) | SE (paired) | t | p |
|--------|----------|----------|---------------|-------------|---|---|
| PI (unified) | | | | | | |
| PI (two-stage) | | | | | | |
| TD (unified) | | | | | | |
| TD (two-stage) | | | | | | |

**Table B3: Architecture × Consumer-Centric Shock Interaction**

Regression (cluster-robust, pooling Policy 0 and Policy 1A data):
```
PI ~ architecture * policy_1a + Q_std_centered + C(category)
TD ~ architecture * policy_1a + Q_std_centered + C(category)
purchase_likelihood ~ architecture * policy_1a + Q_std_centered + PI_std + TD_std + C(category)
```

If `architecture × policy_1a` is significant, the consumer-centric shock has different effects under unified vs. two-stage generation.

### Layer 3 Tables (Policy 1B shock effects)

**Table C1: Policy 1B Shock — Retrieval Effects**

Same structure as B1 but for Policy 1B.

**Table C2: Policy 1B Shock — Expression Effects**

Same structure as B2 but for Policy 1B.

**Table C3: Architecture × Business-Objective Shock Interaction**

Regression (cluster-robust, pooling Policy 0 and Policy 1B data):
```
PI ~ architecture * policy_1b + Q_std_centered + C(category)
TD ~ architecture * policy_1b + Q_std_centered + C(category)
purchase_likelihood ~ architecture * policy_1b + Q_std_centered + PI_std + TD_std + C(category)
```

### Cross-Layer Table (if all three policies are run)

**Table D1: Full Policy Comparison**

| Metric | P0 (generic) | P1A (consumer) | P1B (business) | 1A-0 shock | 1B-0 shock |
|--------|-------------|---------------|---------------|-----------|-----------|
| PI (unified) | | | | | |
| PI (two-stage) | | | | | |
| TD (unified) | | | | | |
| TD (two-stage) | | | | | |
| Q_std (unified) | | | | | |
| Q_std (two-stage) | | | | | |
| Focal share (unified) | | | | | |
| Focal share (two-stage) | | | | | |
| purchase_likelihood (unified) | | | | | |
| purchase_likelihood (two-stage) | | | | | |

### Figures

1. **Product distribution bar charts** by architecture × policy × category.
2. **PI and TD box plots** by architecture and policy.
3. **Scatter: PI vs. Q_std** by architecture, faceted by policy. Visual coupling check.
4. **Policy shock bar chart:** (1A - 0) and (1B - 0) effects on PI, TD, Q_std by architecture.
5. **purchase_likelihood distributions** by architecture and policy.

---

## 8. Failure Modes and Mitigations

| # | Failure mode | Likelihood | Detection | Mitigation |
|---|-------------|-----------|-----------|-----------|
| 1 | **Unified and two-stage outputs are identical** (same products, same text, same scores) | Medium-high | Same-product rate > 90%, PI/TD diffs < 0.2 | This is a finding, not a failure. Report: "the architecture does not induce coupling under a generic prompt." The modular design remains useful as a policy-shock decomposition tool. |
| 2 | **Demand-side model overreacts to persuasive wording** (purchase_likelihood ceiling at 90+ for all persuasive texts) | Medium | SD(purchase_likelihood) < 5 or mean > 85 | Reduce demand model temperature; add calibration instruction ("use the full 0–100 range"); try alternative demand model. |
| 3 | **Supply and demand from same family create self-confirmation** | Low (using different families) | Compare demand-score variance when supply = demand model vs. different model | Primary plan uses gemma2:9b (Google) vs. qwen2.5:14b (Alibaba). Fallback: llama3.1 (Meta) or API. |
| 4 | **Selected products differ too little across architectures** | Medium | TVD < 0.05 for all categories | May indicate one product dominates. Check HHI. If HHI > 0.5, differences emerge only at margins. |
| 5 | **JSON parse failures** | Low-medium | Parse rate < 95% | Retry with seed + 1. If systematic, simplify schema. Two-stage writer uses raw text to avoid JSON issues. |
| 6 | **Q_std leakage** | Very low | Automated grep of all prompts | Q_std is computed only in post-processing. Prompts use consumer profiles and product metadata only. |
| 7 | **Prompt leakage (labels reach demand model)** | Very low | Automated grep of demand prompts | Demand prompt template is hardcoded with no architecture/policy-dependent fields. |
| 8 | **Two-stage writer style differs from unified because it lacks catalog context** | Medium | Compare text length, vocabulary diversity, evaluator rationales | This is an expected architecture effect, not a failure. Document as a finding. The two-stage writer has no alternatives to compare against, which may produce less comparison-oriented text. |
| 9 | **Evaluator implicitly detects architecture from text style** | Low | Check if evaluator rationales mention architecture-related features | Unlikely because evaluator is externally validated for PI and TD. If one architecture scores systematically higher, investigate whether it reflects genuine text quality difference. |
| 10 | **gemma2:9b produces low-quality or repetitive demand responses** | Medium | Score distributions, unique rationale count, SD(purchase_likelihood) | Run 10-row pilot with gemma2 before full run. If degenerate, switch to llama3.1:8b or API. |
| 11 | **Policy 0 generic prompt produces chaotic/unstructured text** | Low-medium | Check if Policy 0 texts are coherent recommendations | If generic prompt is too unconstrained, add minimal structure: "Write a clear, well-organized recommendation." But do not add normative content. |
| 12 | **Policy 1A and 1B produce identical outputs to Policy 0** | Medium | Shock effects < 0.2 on PI/TD, same product selections | The policy instructions may be too subtle for the model. Strengthen the wording or increase the contrast (e.g., add "strongly prioritize" or a more explicit focal-brand naming). |

---

## 9. Decision Rules

### 9.1 Definitions

**Material retrieval difference:** TVD(product distribution | unified vs. two-stage) ≥ 0.10 in at least 2 of 3 categories, OR mean |ΔQ_std| ≥ 0.15 SD units (paired within consumer).

**Material expression difference:** |ΔPI| ≥ 0.5 scale points OR |ΔTD| ≥ 0.5 scale points (paired within consumer, p < 0.10 with clustered SEs).

**Material coupling:** The `architecture × Q_std` interaction in the PI or TD regression is significant at p < 0.10 with cluster-robust SEs, AND the interaction coefficient is ≥ 0.10 in magnitude.

**Material demand difference:** |Δpurchase_likelihood| ≥ 5 points (paired within consumer, p < 0.10).

**Material policy shock:** |shock effect| on PI ≥ 0.5 or on TD ≥ 0.5 or on Q_std ≥ 0.15 SD (paired, p < 0.10).

### 9.2 Layer 1 Decision (Policy 0, architecture comparison)

| Retrieval diff? | Expression diff? | Coupling? | Demand diff? | Conclusion |
|-----------------|-----------------|-----------|-------------|-----------|
| Yes | Yes | Yes | Yes | **Strong architecture effect.** Unified generation couples selection and expression. The modular design is motivated as a correction. |
| Yes | Yes | No | Any | **Architecture affects both channels independently.** Modular design decomposes but does not correct for coupling. |
| No | Yes | No | Any | **Architecture affects expression only.** The models select the same products but write differently. Check if the difference comes from the writer lacking catalog context. |
| No | No | No | No | **No architecture effect.** Frame the modular design as a practical decomposition tool for policy shocks, not as a correction for coupling. |
| Any | Any | Yes | Any | **Coupling detected.** This is the strongest evidence. The unified model adjusts expression based on product fit in a way the two-stage model does not (by construction). |

### 9.3 Layer 2 Decision (Consumer-centric shock)

1. If the consumer-centric shock increases selected Q_std and TD while reducing PI: This supports the interpretation that explicit consumer-welfare instructions change both product allocation and information disclosure.
2. If the shock effect differs by architecture: The architecture modulates how the LLM responds to policy instructions — a form of coupling that emerges under intentional objectives.
3. If the shock has no effect: The generic baseline may already be consumer-centric by default (a finding about LLM training priors).

### 9.4 Layer 3 Decision (Business-objective shock)

1. If the business-objective shock increases focal-brand share and PI while reducing TD: This is the main marketing-relevant mechanism — a realistic platform policy changes both what is recommended and how it is described.
2. If the shock effect differs by architecture: Same interpretation as Layer 2, but now for the business-relevant direction.
3. If the shock has no effect: The focal-brand nudge may be too weak. Strengthen wording or use a more explicit manipulation (e.g., name a specific sponsored product).

### 9.5 Cross-Layer Decision

1. **Policy shocks change retrieval and expression in both architectures:** The paper should focus on policy-shock decomposition. The modular audit is a tool for measuring how different policies shift the retrieval and expression channels.
2. **Policy shocks have different effects by architecture:** Architecture matters, but as a moderator of policy effects rather than as an inherent source of coupling. This is a nuanced but publishable finding.
3. **None of these differences appear:** The current synthetic catalog/prompt setup may be too weak. Revise policies, catalog contrast, or model before scaling.

### 9.6 Thresholds are conservative

The thresholds (TVD ≥ 0.10, |ΔPI| ≥ 0.5, |interaction| ≥ 0.10, |Δpurchase_likelihood| ≥ 5) are deliberately conservative. With N = 60 clusters and a paired/within-consumer design, we have reasonable power to detect effects of this magnitude. If effects are smaller, they are unlikely to matter practically.

---

## 10. File Structure

```
src/
  15_diagnostic_supply.py          # Supply-side generation (unified + two-stage, all policies)
  16_diagnostic_evaluate.py        # PI/TD evaluation (reuses evaluator logic)
  17_diagnostic_demand.py          # Demand-side simulation (gemma2:9b)
  18_diagnostic_analysis.py        # Merge, compare, tables, figures

data/diagnostic/
  raw/                             # Individual JSON responses
    unified_p0_{cat}_{cid}.json
    unified_p1a_{cat}_{cid}.json
    unified_p1b_{cat}_{cid}.json
    twostage_p0_{cat}_{cid}_stage1.json
    twostage_p0_{cat}_{cid}_stage2.json
    twostage_p1a_{cat}_{cid}_stage1.json
    twostage_p1a_{cat}_{cid}_stage2.json
    twostage_p1b_{cat}_{cid}_stage1.json
    twostage_p1b_{cat}_{cid}_stage2.json
  supply_outputs.csv               # Consolidated supply-side data
  evaluator_outputs.csv            # PI/TD scores
  demand_outputs.csv               # Synthetic demand responses
  analysis_dataset.csv             # Merged analysis-ready dataset
  manifest.csv                     # Call tracking for resume

results/diagnostic/
  tables/
    a1_product_selection_p0.csv
    a2_expression_comparison_p0.csv
    a3_expression_conditional_p0.csv
    a4_expression_fit_coupling_p0.csv
    a5_demand_comparison_p0.csv
    a6_demand_conditional_p0.csv
    b1_shock_1a_retrieval.csv
    b2_shock_1a_expression.csv
    b3_shock_1a_interaction.csv
    c1_shock_1b_retrieval.csv
    c2_shock_1b_expression.csv
    c3_shock_1b_interaction.csv
    d1_full_policy_comparison.csv
    expression_fit_correlations_diagnostic.csv
  figures/
    product_distribution_by_arch_policy.png
    pi_td_boxplots.png
    pi_vs_qstd_by_architecture.png
    policy_shock_effects.png
    demand_distribution.png
  diagnostic_report.md             # Summary report

plans/
  diagnostic_unified_vs_twostage.md  # This plan
```

---

## 11. Replicability Specification

| Parameter | Value | Notes |
|-----------|-------|-------|
| MASTER_SEED | 20260514 | Same as existing experiments |
| Supply model | qwen2.5:14b | Ollama, Q4_K_M quantization |
| Supply temperature | 0.0 | Greedy decoding; optional robustness at 0.7 |
| Supply seed | MASTER_SEED + consumer_id × 1000 + cell_index | See Section 3.1 for cell_index table |
| Evaluator model | qwen2.5:14b | Temperature 0.0, seed 99999 |
| Demand model | gemma2:9b | Temperature 0.0, seed 77777 |
| Consumer ordering | consumer_id 0–19, sorted ascending | |
| Category ordering | headphones, laptop, phone_charger (alphabetical) | |
| Catalog product ordering | As stored in JSON (creation order) | |
| N_CONSUMERS | 20 | First 20 from existing profiles |
| Prompt versioning | v2 (this document); template text stored in script header as constants | |
| Raw logging | Full request/response JSON per call | |
| Timestamps | ISO 8601 UTC in each raw file | |
| Output schemas | Strict JSON schemas validated before saving | |

---

## 12. How This Diagnostic Supports or Weakens the Paper

### If the diagnostic finds architecture coupling (Layer 1: coupling detected)

**Supports the paper.** The finding demonstrates that when an LLM jointly selects and describes a product, it couples product selection with persuasive expression — for example, selecting products it can write more enthusiastically about, or adjusting tradeoff disclosure based on product fit. This directly motivates the modular audit design: a firm deploying a unified recommender cannot separate "what was recommended" from "how it was described" without an experimental decomposition.

This would appear in the paper as empirical evidence (Section 5) that architecture choice has causal consequences for the text consumers receive, which the firm cannot observe without the proposed factorized experiment.

### If the diagnostic finds no architecture effect but policy shocks work (Layers 2–3)

**Still useful.** The paper would frame the modular audit as a tool for decomposing realistic policy shocks (consumer-centric vs. business-objective) rather than as a correction for architecture-induced confounding. The contribution shifts from "unified LLMs create coupling that naive A/B tests cannot detect" to "when a firm changes its recommender policy (e.g., introducing focal-brand nudges), the modular audit separates the retrieval shift from the expression shift."

This framing aligns with the treatment in the final paper being a realistic policy shock to a generic LLM recommender, not an artificial on/off toggle.

### If the diagnostic finds no architecture effect and no policy effects

**Weakens the current setup but does not kill the paper.** It may indicate that the synthetic catalog, consumer profiles, or prompt wording are too weak to generate meaningful variation. Options:
- Revise prompts with stronger policy contrast.
- Use a different or larger LLM.
- Add more products or consumers.
- Refocus the paper on the theoretical framework and human experiment design.

### What the diagnostic cannot tell us

- Whether real consumers respond differently to unified vs. two-stage recommendations (requires human experiment).
- Whether the architecture or policy effects would persist with different LLMs, catalogs, or consumer populations.
- Whether the coupling (if found) is economically meaningful — the magnitude matters and depends on the firm's objective function.

### Important framing notes

- The treatment in the final paper should be a realistic policy shock to a generic LLM recommender, not "turn retrieval on/off" or "turn persuasion on/off."
- The modular audit should be framed as a decomposition/audit tool, not as the primary treatment itself.
- The generic baseline (Policy 0) should not be over-engineered.
- Demand-side outcomes are **synthetic consumer-response proxies**, not real consumer behavior. The report must state this clearly.

---

## 13. Feasibility Assessment

### Hardware
- qwen2.5:14b is already installed and working via Ollama.
- gemma2:9b requires ~5.4 GB download; Ollama swaps models in/out of memory, so both can run sequentially.
- No GPU rental or API costs required for the primary plan.

### Time Estimates

| Scope | Supply calls | Eval calls | Demand calls | Total calls | Est. time |
|-------|-------------|-----------|-------------|-------------|-----------|
| Smoke test (2 cons × 1 cat × P0) | 6 | 4 | 4 | 14 | ~20 min |
| Layer 1 (20 cons × 3 cat × P0) | 180 | 120 | 120 | 420 | ~3.5 hrs |
| Layer 2 (add P1A) | +180 | +120 | +120 | +420 | ~3.5 hrs |
| Layer 3 (add P1B) | +180 | +120 | +120 | +420 | ~3.5 hrs |
| Full experiment | 540 | 360 | 360 | 1,260 | ~10.5 hrs |
| Analysis and report | — | — | — | — | ~2 hrs |

With 10 consumers per category: halve all call counts and times.

### Data
- All required data (catalogs, consumer profiles, fit scores) already exist.
- Using consumers 0–19 from existing profiles (no regeneration needed).
- Q_std can be computed from existing fit_scores CSVs.

### Risk
- Main risk: gemma2:9b producing degenerate demand responses. Mitigated by smoke test and fallback options.
- Secondary risk: low product-selection variation across architectures (if one product dominates per category). Mitigated by using 3 categories with different product distributions.
- Tertiary risk: Policy 0 generic prompt producing incoherent or inconsistent text. Mitigated by smoke test.

### Verdict

Feasible with current local resources. No API costs required.

**Recommended execution:** Start with Layer 1 only (Policy 0, ~3.5 hours). If Layer 1 produces meaningful variation (architecture effects or at least sufficient text diversity), proceed to Layer 2 (Policy 1A). Layer 3 (Policy 1B) is the most marketing-relevant but can wait if time is tight.

---

## 14. Recommended Execution Sequence

1. **Pull gemma2:9b** and smoke-test its JSON output (~15 min).
2. **Run smoke test** (2 consumers, 1 category, Policy 0; 14 calls): ~20 min.
3. **Run Layer 1 full** (20 consumers × 3 categories, Policy 0, both architectures; 420 calls including supply, evaluator, and demand): ~3.5 hours total.
4. **Analyze Layer 1.** Produce Tables A1–A6 and apply Layer 1 decision rule.
5. **Run Layer 2** (Policy 1A, both architectures; 420 calls including supply, evaluator, and demand): ~3.5 hours total.
6. **Analyze Layers 1+2.** Produce Tables B1–B3. Test architecture × consumer-centric shock interaction.
7. **(If time allows) Run Layer 3** (Policy 1B, both architectures; 420 calls including supply, evaluator, and demand): ~3.5 hours total.
8. **Analyze all layers.** Produce Tables C1–C3, D1. Apply cross-layer decision rules.
9. **Write diagnostic report** with all tables, figures, and decision-rule outcomes.
