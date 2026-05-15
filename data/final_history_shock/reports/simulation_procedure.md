# LLM Recommender Simulation: Complete Procedure

**Project**: Unbundling LLM Recommender Effects — Retrieval vs. Expression
**Date**: 2026-05-16
**Supply model**: Qwen 2.5 14B (local via Ollama, temperature 0.7)
**Demand model**: GPT-5.3-chat-latest (OpenAI API)
**Category**: Headphones (30 real Amazon products, 60 consumer personas)

---

## Overview

This document records the complete procedure for the LLM recommender simulation. The pipeline has 14 numbered steps (scripts `00` through `13`, plus `14`/`14b`/`14c` for the unified black-box experiment). Each step is described with its purpose, inputs, outputs, exact prompts where applicable, and key implementation details.

The simulation has two parts:
1. **Supply side**: A local LLM (Qwen 2.5 14B) generates product recommendations under different information conditions.
2. **Demand side**: GPT-5.3 evaluates those recommendations as a blinded synthetic consumer judge.

All scripts are in `src/final_history_shock/`. All data are in `data/final_history_shock/`.

---

## Step 0: API Test (`00_api_test.py`)

**Purpose**: Verify the OpenAI API key and model work before running the pipeline.

**What it does**: Sends 5 test prompts to GPT-5.3-chat-latest, checks for valid responses and JSON parsing.

**Test prompts**:
```
1. Return exactly this JSON and nothing else: {"ok": true}
2. In one sentence, explain why over-ear headphones differ from earbuds.
3. A student on a tight budget is choosing between Product A ($29, decent sound, wired) and Product B ($89, noise cancelling, wireless). Recommend one. Return JSON: {"product_id": "A" or "B", "reason": "..."}
4. You are comparing two product recommendations for the same consumer. [comparison prompt] Return JSON: {"winner": "A", "B", or "tie", "reason": "..."}
5. Return JSON: {"test": "usage", "model_check": true}
```

**Output**: `data/final_history_shock/api_test_results.json`

---

## Step 1: Build Product Catalogs (`01_build_or_collect_catalogs.py`)

**Purpose**: Load real Amazon product metadata and map it to the pipeline's schema.

**Data source**: Amazon Reviews 2023 dataset (McAuley Lab, NeurIPS 2024). Pre-curated product files at `data/real_metadata/products_headphones.csv` containing 30 headphones with real ASINs, prices, ratings, review counts, and feature text.

**Schema mapping**: Each product gets:
- `product_id`: `headphones_001` through `headphones_030`
- `brand`, `model_name`, `title`, `price`, `price_tier`
- `average_rating`, `rating_count`, `popularity_rank`, `popularity_tier`
- `key_features`: Parsed from raw feature string (up to 6 items, max 120 chars each)
- `best_for`: Automatically inferred from keywords in features + title (limited to 3 per product):
  - "noise cancel" / "anc" → "commuting and travel"
  - "workout" / "sweat" / "sport" / "water resistant" / "ipx" → "exercise and gym"
  - "gaming" / "game" / "low latency" → "gaming"
  - "studio" / "monitor" / "hi-fi" / "audiophile" → "music production"
  - "wireless" / "bluetooth" → "everyday wireless use"
  - "over-ear" / "over ear" → "home and office listening"
  - "microphone" / "mic" / "call" → "calls and remote work"
  - "kid" / "child" → "children"
  - Budget price tier → "budget-conscious buyers"
  - Default if no keywords match: "general use"
- `drawbacks`: Automatically inferred (limited to 2 per product):
  - No wireless/bluetooth detected → "wired only — no Bluetooth"
  - Budget price tier → "build quality may not match premium alternatives"
  - Premium price tier → "high price point"
  - Rating < 4.2 → "mixed user reviews on durability or comfort"
  - Default: "no major drawbacks reported"
- `review_summary`: Formatted as `"★{rating:.1f} from {rating_count:,} reviews. {popularity_tier}. {product}"`

**Validation constraints**: Requires ≥20 products, ≥8 brands, ≥3 price tiers (budget/midrange/premium), no missing price data.

**Output**: `data/final_history_shock/catalogs/headphones_catalog.csv` (30 rows)

---

## Step 2: Generate GPT Exemplars (`02_generate_gpt_exemplars.py`)

**Purpose**: Generate high-quality recommendation examples from GPT to calibrate the local LLM via few-shot prompting.

**Scale**: 180 GPT calls (6 categories × 10 personas × 3 regimes).

**Three regimes with exact instruction prompts**:

### Generic regime
```
You are a helpful product recommender. Recommend the single best product for this consumer from the catalog. Be realistic, concise, and helpful. Focus on fit with the consumer's needs.
```

### Consumer-centric regime
```
You are a balanced, consumer-first product recommender. Prioritize fit, budget, tradeoffs, and post-purchase satisfaction. Be transparent about limitations. Do not oversell.
```

### History-aware regime
```
You are a product recommender with access to internal historical purchase summaries. Treat them as background evidence only. Do not reveal, quote, or approximate any conversion rates, satisfaction rates, percentages, rankings, sample sizes, scores, or raw historical numbers. You may only refer qualitatively to historical patterns using phrases like "popular among similar buyers," "historically reliable," "often chosen for this use case," or "mixed feedback among heavy users." Write a natural, consumer-facing recommendation.
```

**Mini-catalog for exemplar generation**: Hardcoded 8-product catalog per category (not the full 30-product real catalog). For headphones: Sony WH-1000XM5, JBL Tune 510BT, Beyerdynamic DT 770 Pro, Apple AirPods Max, Audio-Technica ATH-M50x, Sennheiser HD 560S, Anker Soundcore Q45, Bose QuietComfort 45.

**10 fixed persona types** (same across all 6 categories): Budget Student, Commuter Professional, Tech Enthusiast, Parent Buying Gift, Remote Worker, Casual User, Fitness Enthusiast, Small Business Owner, Retiree, Content Creator.

**History signal for history-aware regime**:
```
Historical background (internal only, do not cite numbers): This product has been frequently chosen by budget-conscious buyers in this category. Post-purchase feedback is generally positive, with occasional notes about build quality. Among similar buyers, this product is often considered a reliable choice.
```

**Output schema requested from GPT**:
```json
{
  "selected_product_id": "...",
  "recommendation_text": "A 2-4 sentence consumer-facing recommendation",
  "why_it_fits": "1 sentence on why this product fits the consumer",
  "tradeoff_note": "1 sentence on the main tradeoff or limitation",
  "history_used_qualitatively": true/false,
  "forbidden_numeric_history_leakage": false
}
```

**GPT model**: `gpt-5.3-chat-latest` (configurable via `OPENAI_MODEL` env var). Default temperature (API default). 3 retries on JSON parse failure.

**Outputs**:
- `data/final_history_shock/gpt_exemplars/gpt_recommendation_exemplars.jsonl` (all 180 exemplars with prompts and raw responses)
- `data/final_history_shock/gpt_exemplars/selected_few_shot_examples.json` (2 best exemplars per regime for few-shot use)

---

## Step 3: Generate Consumer Personas (`03_generate_personas.py`)

**Purpose**: Generate 60 diverse, realistic consumer personas via GPT.

**Scale**: 6 GPT calls (6 batches of 10 personas each).

**Exact prompt template** (filled per batch):
```
Generate exactly {n} diverse, realistic consumer personas who might buy {category}.

Batch {batch_num} of {total_batches}. These personas should be DIFFERENT from typical tech-savvy reviewer profiles.

Requirements:
- Vary across: budget level, technical knowledge, age range, purchase urgency, use case, brand awareness, risk tolerance
- Include non-obvious consumer types: parents buying for kids, gift buyers, people replacing broken devices, first-time buyers, elderly, office managers, students
- Each persona must have genuine needs that make product choice nontrivial
- No two personas in this batch should have the same primary use case + budget combination
- Budget values should be realistic dollar ranges for {category}

Return a JSON array of {n} personas, each matching this schema:
{
  "persona_id": "{category}_{start_id:03d}",
  "category": "{category}",
  "age_range": "18-24|25-34|35-44|45-54|55-64|65+",
  "purchase_context": "Why they're buying now, in one sentence",
  "budget": "$X-$Y realistic range",
  "technical_knowledge": "low|medium|high",
  "primary_use_case": "Main intended use",
  "secondary_use_case": "Secondary use or none",
  "brand_preference": "Specific brand or 'no preference' or 'avoids X'",
  "price_sensitivity": "low|medium|high",
  "quality_sensitivity": "low|medium|high",
  "risk_aversion": "low|medium|high",
  "must_have_features": ["feature1", "feature2"],
  "features_to_avoid": ["feature1"],
  "prior_experience": "Brief note on past purchases or experience level",
  "one_paragraph_description": "2-3 sentence realistic consumer description"
}

Return ONLY a valid JSON array. No markdown, no extra text.
```

**Output**: `data/final_history_shock/personas/headphones_personas.json` (60 personas)

---

## Step 4: Validate Personas (`04_validate_personas.py`)

**Purpose**: Check diversity, completeness, and plausibility of generated personas. No LLM calls.

**Checks**:
- At least 55 personas
- At least 5 personas per technical knowledge level (low/medium/high)
- At least 5 personas per price sensitivity level
- At least 3 distinct risk aversion levels
- All required fields present: `persona_id`, `category`, `budget`, `primary_use_case`, `one_paragraph_description`, `technical_knowledge`
- No duplicate descriptions

**Output**: Console validation report (pass/fail).

---

## Step 5: Generate Historical DGP (`05_generate_historical_dgp.py`)

**Purpose**: Build qualitative historical buyer-feedback summaries from real Amazon metadata. These summaries serve as the "history shock" — the information advantage given to the history-aware recommender.

**No LLM calls.** All summaries are constructed algorithmically from real data fields (rating count, average rating, popularity rank).

**8 consumer segments**: `budget_student`, `commuter`, `remote_worker`, `audiophile`, `gym_user`, `gamer`, `frequent_traveler`, `casual_listener`

**Segment-product affinity score** (0–1): Deterministic, computed as a weighted sum of three components:

1. **Keyword match** (50% weight): Count of keyword hits in product features + title, normalized to [0, 0.5]. Segment-specific keyword lists:
   - `budget_student`: ["wireless", "bluetooth", "affordable"]
   - `commuter`: ["noise cancel", "anc", "wireless", "bluetooth"]
   - `remote_worker`: ["microphone", "mic", "comfort", "call"]
   - `audiophile`: ["studio", "hi-fi", "over-ear", "wired"]
   - `gym_user`: ["sweat", "sport", "water", "secure fit", "wireless"]
   - `gamer`: ["gaming", "game", "latency", "microphone"]
   - `frequent_traveler`: ["noise cancel", "anc", "foldable", "portable"]
   - `casual_listener`: ["wireless", "bluetooth", "lightweight"]

2. **Price-tier match** (30% weight): Exact tier match → +0.30; adjacent tier (e.g., midrange for budget segment) → +0.15; mismatch → +0.

3. **Rating bonus** (20% weight): `(average_rating − 3.5) / 1.5 × 0.20`, capped so total score ≤ 1.0.

**Summary construction** (per product × segment, 30 × 8 = 240 entries): Three components combined into natural language:

1. **Popularity language** (based on popularity rank and rating count):
   - Top 10%: `"one of the top sellers in its category with over X customer ratings"`
   - Top 30%: `"a popular choice with X customer ratings"`
   - Top 60%: `"a moderately popular option with X ratings"`
   - Bottom 40%: `"a niche product with a smaller but dedicated following (X ratings)"`

2. **Satisfaction language** (based on average rating):
   - ≥4.6: `"Buyers overwhelmingly rate it highly, with very few complaints"`
   - ≥4.3: `"Most buyers are satisfied, though some report minor issues"`
   - ≥4.0: `"Reviews are generally positive but with notable criticisms from some buyers"`
   - ≥3.5: `"Reviews are mixed — roughly split between satisfied and dissatisfied buyers"`
   - <3.5: `"A significant share of buyers report disappointment or quality issues"`

3. **Segment fit language** (based on affinity score):
   - ≥0.7: `"This is a strong match for {segment} buyers based on features and price point"`
   - ≥0.45: `"A reasonable option for {segment} buyers, though not an ideal match on all dimensions"`
   - <0.45: `"Not the typical choice for {segment} buyers — may lack key features they prioritize"`

**Additional output**: `data/final_history_shock/history_dgp/headphones_history_aggregates.csv` (product-level stats: product_id, brand, price, price_tier, average_rating, rating_count, popularity_rank, popularity_tier)

**Output**: `data/final_history_shock/history_dgp/headphones_history_qualitative.json` (30 products × 8 segments = 240 entries, each with fields: `segment`, `product_id`, `affinity_score`, `popularity_qualitative`, `satisfaction_qualitative`, `segment_fit_qualitative`, `review_snippet`, `summary`)

---

## Step 6: Build Few-Shot Prompts (`06_build_local_prompts.py`)

**Purpose**: Select the best GPT exemplars from Step 2 and format them as few-shot blocks for the local LLM.

**Selection logic**: 2 exemplars per regime (generic, consumer-centric, history-aware). Filters out parse failures. Prefers headphones category if at least 2 available per regime; otherwise falls back to any category.

**Few-shot block format**:
```
--- Example recommendations (for reference) ---

Example 1 (headphones):
Recommendation: [GPT exemplar text]
Tradeoff note: [GPT exemplar tradeoff]

Example 2 (headphones):
Recommendation: [GPT exemplar text]
Tradeoff note: [GPT exemplar tradeoff]

Note: These examples show how to reference historical patterns qualitatively without citing any numbers, rates, or statistics.
```

**Output**: `data/final_history_shock/gpt_exemplars/final_few_shot_prompts.json`

---

## Step 7: Smoke Run (`07_smoke_run_local_supply.py`)

**Purpose**: Run 3 personas × 4 cells = 12 recommendation packages as a quality check before the full run.

**Persona-to-segment mapping** (heuristic keyword matching in persona description/use_case/purchase_context, used in Steps 7, 8, and 14 to select segment-specific history summaries):

| Keyword(s) in persona text | Assigned segment |
|---------------------------|-----------------|
| "student", "budget" | `budget_student` |
| "commut" | `commuter` |
| "travel" | `frequent_traveler` |
| "remote", "work from home", "office", "conference", "call" | `remote_worker` |
| "audiophile", "music quality", "studio", "hi-fi" | `audiophile` |
| "gym", "workout", "fitness", "exercise" | `gym_user` |
| "gaming", "gamer", "esport" | `gamer` |
| "casual", "general", "everyday", "kid", "child" | `casual_listener` |
| No match | `casual_listener` (default) |

Keywords are matched in order; first match wins. This mapping is transparent and deterministic but imperfect — it is a heuristic, not a ground-truth segment assignment.

**Checks**:
- Parse success rate ≥95%
- Leakage rate ≤5%
- Cell invariant: cells (0,0) and (0,1) must have the same product; cells (1,0) and (1,1) must have the same product

**Output**: `data/final_history_shock/local_supply/smoke_supply_rows.csv`

---

## Step 8: Full Supply Generation (`08_run_local_supply_full.py`)

**Purpose**: Generate all 240 recommendation packages (60 personas × 4 cells).

**Architecture**: Two-stage modular design.

**Per cluster (one persona), 6 LLM calls**:

### Call 1: Generic retrieval (seed_base + 1)

**Full prompt** (`build_generic_retrieval_prompt`):
```
You are a product recommender. You only have access to the product catalog and the consumer's stated needs. You do NOT have access to sales data, review summaries, popularity rankings, or historical purchase feedback. Recommend based solely on product specifications and how they match the consumer's requirements.

Select the single best product for this consumer based on their stated needs and the product specifications.

Write a straightforward product recommendation based on the product specs and the consumer's stated needs. Be factual and concise. Stick to what is listed in the product features. Do not speculate about popularity, user experience, or track record — you only have the spec sheet.

Consumer profile:
[persona one_paragraph_description]
Budget: [budget]
Primary use: [primary_use_case]
Must-have: [must_have_features]
Avoids: [features_to_avoid]

Product catalog:
- headphones_001: Sony WH-1000XM5 ($348 ★4.6) — noise cancelling, wireless, over-ear
- headphones_002: JBL Tune 510BT ($35 ★4.5) — wireless, on-ear, budget
[... all 30 products ...]

[few-shot block from Step 6]

Return ONLY valid JSON:
{
  "selected_product_id": "...",
  "retrieval_rationale_internal": "1-2 sentences on why this product fits best"
}
```

### Call 2: History-aware retrieval (seed_base + 2)

**Full prompt** (`build_history_retrieval_prompt`):
```
You are an experienced shopping recommender. You have access to product attributes, public review summaries, and internal historical purchase-feedback patterns. Your goal is not to recommend the most popular product. Your goal is to recommend the product most suitable for this specific consumer. Use historical evidence as background only. Do not cite exact sales numbers, conversion rates, satisfaction rates, ranks, sample sizes, or percentages. You may refer qualitatively to evidence such as "often chosen by similar buyers," "strong post-purchase feedback among travelers," "mixed feedback from budget buyers," or "frequent complaints about comfort."

Select the single best product for this consumer, informed by both specs and historical purchase patterns.

Consumer profile:
[persona details]

Internal historical purchase-feedback data (use as background evidence only):
- headphones_001: This product is one of the top sellers in its category with over 50,000 customer ratings. Buyers overwhelmingly rate it highly, with very few complaints. This is a strong match for commuter buyers based on features and price point.
[... segment-relevant history for all products ...]

Product catalog (with review counts and popularity):
- headphones_001: Sony WH-1000XM5 ($348, ★4.6, 50,234 reviews [top_seller]) — noise cancelling, wireless
[... all 30 products with popularity data ...]

[few-shot block]

Consider both the consumer's stated needs AND which products have historically satisfied similar buyers. If a popular, well-reviewed product fits the consumer's needs, prefer it over an obscure alternative — unless the consumer has specific requirements the popular option cannot meet.

Return ONLY valid JSON:
{
  "selected_product_id": "...",
  "retrieval_rationale_internal": "1-2 sentences on why this product fits best",
  "history_signal_used": "none|weak|moderate|strong"
}
```

### Calls 3–6: Expression (4 cells)

The product is **locked** from the retrieval stage. The expression prompt receives the selected product and writes a consumer-facing recommendation.

**Cell (0,0) and (1,0): Generic expression** (seed_base + 10 or +20)

Full prompt (`build_generic_expression_prompt`):
```
You are writing a consumer-facing product recommendation.

Write a straightforward product recommendation based on the product specs and the consumer's stated needs. Be factual and concise. Stick to what is listed in the product features. Do not speculate about popularity, user experience, or track record — you only have the spec sheet.

Consumer profile:
[persona details]

Selected product:
Product: Sony WH-1000XM5 ($348, premium tier)
Features: noise cancelling, wireless, Bluetooth 5.2, 30-hour battery
Best for: commuting and travel, everyday wireless use
Drawbacks: high price point

[few-shot block]

Write a recommendation based ONLY on the product specifications and the consumer's needs. Do not mention popularity, reviews, or how other buyers feel — you only have the spec sheet.

Return ONLY valid JSON:
{
  "recommendation_text": "3-5 sentence consumer-facing recommendation",
  "tradeoff_text": "1-2 sentences on the main limitation or tradeoff",
  "persuasion_text": "1 sentence on the strongest reason to buy"
}
```

**Cell (0,1) and (1,1): History-aware expression** (seed_base + 30 or +40)

Full prompt (`build_history_expression_prompt`):
```
You are an experienced product advisor writing a recommendation backed by real-world buyer data.

Do not cite exact sales numbers, conversion rates, satisfaction rates, ranks, sample sizes, or percentages. You may refer qualitatively to evidence such as "often chosen by similar buyers," "strong post-purchase feedback among travelers," "mixed feedback from budget buyers," or "frequent complaints about comfort."

Write like an experienced shopping advisor who has seen thousands of customers buy and return products. You have deep knowledge of how products perform in the real world — which ones customers love, which ones get returned, and which ones surprise people. Be opinionated and specific. Use your historical insight to give advice that goes beyond the spec sheet: mention real-world reliability, common buyer satisfaction patterns, and how this product compares to alternatives that similar buyers have tried. Your goal is to help the consumer avoid regret.

Consumer profile:
[persona details]

Selected product and historical intelligence:
Product: Sony WH-1000XM5 ($348, premium tier)
Features: noise cancelling, wireless, Bluetooth 5.2, 30-hour battery
Best for: commuting and travel, everyday wireless use
Drawbacks: high price point
Review: ★4.6 from 50,234 reviews. Top seller product.

--- Internal Historical Intelligence (DO NOT cite numbers) ---
This product is one of the top sellers in its category with over 50,000 customer ratings. Buyers overwhelmingly rate it highly, with very few complaints. This is a strong match for commuter buyers based on features and price point.

[few-shot block]

Write a recommendation that naturally weaves in your knowledge of how this product performs in the real world. Reference patterns like "buyers in your situation tend to..." or "this is a proven choice for..." or "the main complaint from similar users is..." — but NEVER cite specific numbers, percentages, or rankings.

Your recommendation should feel noticeably different from a generic spec-based recommendation. Show the consumer you know this product's track record.

Return ONLY valid JSON:
{
  "recommendation_text": "3-5 sentence consumer-facing recommendation that reflects real-world buyer experience",
  "tradeoff_text": "1-2 sentences on limitations informed by actual user complaints",
  "persuasion_text": "1 sentence on the strongest reason to buy, grounded in buyer track record",
  "history_language_used": "none|weak|moderate|strong"
}
```

**Seed structure**: `MASTER_SEED (20260515) + persona_index * 100 + offset`
- Generic retrieval: +1
- History retrieval: +2
- Cell 00 expression: +10
- Cell 10 expression: +20
- Cell 01 expression: +30
- Cell 11 expression: +40

**Cell invariant enforced**: Cells sharing the same retrieval condition share the same product. Cell (0,0) and (0,1) both use the generic-retrieved product. Cell (1,0) and (1,1) both use the history-retrieved product.

**Output fields per row**: `cluster_id`, `category`, `persona_id`, `cell`, `retrieval_condition`, `expression_condition`, `selected_product_id`, `recommendation_text`, `tradeoff_text`, `persuasion_text`, `full_recommendation_package`, `history_language_used`, `local_model`, `word_count`, `parse_failed`, `retrieval_changed`, `leakage_flag`

**Output**: `data/final_history_shock/local_supply/final_supply_rows.csv` (240 rows)

---

## Step 9: Leakage Audit and Regeneration (`09_leakage_audit_and_regen.py`)

**Purpose**: Scan all 240 supply rows for forbidden statistical leakage patterns. Regenerate flagged rows with stricter prompts.

**18 forbidden regex patterns** (`utils_parse.py`):
```python
FORBIDDEN_PATTERNS = [
    r"\d+\s*%",                    # any percentage
    r"conversion rate",
    r"satisfaction rate",
    r"CTR",
    r"CVR",
    r"click-through",
    r"sample size",
    r"n\s*=",
    r"historical data shows",
    r"\b0\.\d+\b",                 # decimal proportions
    r"\b1\.0\b",
    r"ranked #?1 by conversion",
    r"purchase rate",
    r"return rate of",
    r"\d+\s*out of\s*\d+",        # "X out of Y"
    r"rating of \d",
    r"scored? \d+(\.\d+)?",
    r"\d+ stars?",
]
```

**Strict anti-leakage prompt** (prepended to regeneration):
```
CRITICAL RULE: Your response MUST NOT contain any of these:
- Any percentage (e.g., 42%, 0.73)
- Any conversion rate, satisfaction rate, or purchase rate
- Any sample size (n=, out of)
- Any star rating or numerical score
- CTR, CVR, click-through
- Any phrase like 'historical data shows' followed by a number
If you feel tempted to cite a number, instead use qualitative language like 'popular among similar buyers' or 'generally well-received.'

Do not cite exact sales numbers, conversion rates, satisfaction rates, ranks, sample sizes, or percentages. You may refer qualitatively to evidence such as "often chosen by similar buyers," "strong post-purchase feedback among travelers," "mixed feedback from budget buyers," or "frequent complaints about comfort."
```

**Target**: <2% leakage after regeneration.

**Result**: 1/240 flagged (0.4%), confirmed false positive from product spec "90% noise reduction".

**Output**: `data/final_history_shock/local_supply/final_supply_rows_clean.csv`

---

## Step 10: GPT Absolute Evaluation (`10_gpt_absolute_eval.py`)

**Purpose**: Rate each of the 240 recommendation packages independently on 8 scales using GPT-5.3.

**Scale**: 240 GPT calls (60 clusters × 4 cells).

**System prompt**:
```
You are an expert consumer behavior analyst evaluating product recommendations. Rate each recommendation on how well it serves the given consumer. You do not know how the recommendation was generated. Be calibrated and use the full scale.
```

**User prompt** (`build_gpt_absolute_eval_prompt`):
```
Evaluate this product recommendation for the given consumer.

Consumer profile:
[persona details]

Recommended product: Sony WH-1000XM5 ($348)

Full recommendation:
[full_recommendation_package text]

Rate on these scales. Return ONLY valid JSON:
{
  "fit_score_1_7": <1-7, how well the product fits the consumer's needs>,
  "purchase_probability_0_100": <0-100, likelihood consumer would purchase>,
  "expected_satisfaction_0_100": <0-100, predicted post-purchase satisfaction>,
  "trust_score_1_7": <1-7, how trustworthy the recommendation feels>,
  "clarity_score_1_7": <1-7, how clear and understandable>,
  "persuasive_intensity_1_7": <1-7, how persuasive the language is>,
  "tradeoff_disclosure_1_7": <1-7, how honestly it discusses limitations>,
  "regret_risk_1_7": <1-7, risk of post-purchase regret>,
  "brief_reason": "1-2 sentences explaining your ratings"
}
```

**GPT evaluator is blinded**: It does not know which cell the recommendation came from.

**Output**: `data/final_history_shock/gpt_eval/absolute_eval_rows.csv` (240 rows)

---

## Step 11: GPT Pairwise Evaluation (`11_gpt_pairwise_eval.py`)

**Purpose**: Compare all 6 pairs of cells within each cluster, producing the primary outcome variable for Bradley-Terry decomposition.

**Scale**: 360 GPT calls (60 clusters × 6 pairs).

**6 pairs per cluster**: (00,10), (00,01), (00,11), (10,01), (10,11), (01,11)

**Randomized A/B ordering**: For each pair, a fair coin (seeded at `MASTER_SEED + 777`) determines whether cell_i is shown as Package A or Package B. This prevents position bias.

**System prompt**:
```
You are evaluating two recommendation packages for the same consumer. You do not know how the recommendations were generated. Choose which package is more likely to lead to a good consumer outcome. Consider purchase likelihood, expected post-purchase satisfaction, trust, and whether the recommendation honestly communicates tradeoffs. Use tie only if the two packages are genuinely indistinguishable.
```

**User prompt** (`build_gpt_pairwise_eval_prompt`):
```
Compare these two product recommendations for the same consumer.

Consumer profile:
[persona details]

=== Package A ===
Product: Sony WH-1000XM5 ($348)
[full recommendation package A]

=== Package B ===
Product: JBL Tune 510BT ($35)
[full recommendation package B]

Which package would lead to a better outcome for this consumer? Return ONLY valid JSON:
{
  "overall_winner": "A|B|tie",
  "purchase_winner": "A|B|tie",
  "satisfaction_winner": "A|B|tie",
  "trust_winner": "A|B|tie",
  "confidence_1_5": <1-5>,
  "reason": "1-2 sentences explaining your choice"
}
```

**Winner remapping**: The A/B winner label is remapped back to cell labels (e.g., "A" → "01" if cell 01 was shown as Package A).

**Output**: `data/final_history_shock/gpt_eval/pairwise_eval_rows.csv` (360 rows)

---

## Step 12: Analysis and Decomposition (`12_analyze_decomposition.py`)

**Purpose**: Compute the Bradley-Terry decomposition and all summary statistics. No LLM calls.

**6 outputs**:

### Table 1: Design summary
Sample sizes for each component.

### Table 2: Retrieval variation
What fraction of clusters have different products under generic vs. history retrieval.

### Table 3: Pairwise win rates
Raw win/loss/tie counts for all 6 cell pairs.

### Table 4: Bradley-Terry decomposition

**Bradley-Terry model** (`utils_stats.py`):

Win matrix `W[i,j]` = number of times cell i beats cell j. Ties split as half-wins. Log-likelihood:

```
LL = sum over all (i,j) pairs: W[i,j] * log(p_ij) + W[j,i] * log(1 - p_ij)
where p_ij = 1 / (1 + exp(theta_j - theta_i))
```

Optimized via Nelder-Mead with `theta[0] = 0` (cell 00 is the reference).

**Decomposition from utilities** `theta = [theta_00, theta_10, theta_01, theta_11]`:
```
Delta_J (retrieval)   = theta[10] - theta[00]
Delta_T (expression)  = theta[01] - theta[00]
tau^MOD (total)       = theta[11] - theta[00]
Delta_JT (interaction) = theta[11] - theta[10] - theta[01] + theta[00]
```

**Cluster bootstrap** (B=1000, seed=42): Resample clusters with replacement, refit BT model, report mean, SE, 2.5th/97.5th percentiles, and P(>0).

### Table 5: Multi-outcome decomposition
Same BT decomposition applied separately to `overall_winner_cell`, `purchase_winner_cell`, `satisfaction_winner_cell`, `trust_winner_cell` (B=500 each).

### Table 6: Text mechanisms
Cell means of `persuasive_intensity_1_7`, `tradeoff_disclosure_1_7`, `regret_risk_1_7`, `trust_score_1_7` from absolute evaluation.

### Figure 1: Decomposition bar chart
Bar chart with error bars for retrieval, expression, interaction, total effects.

### Figure 2: Multi-outcome decomposition
Four-panel chart comparing decomposition across overall, purchase, satisfaction, trust.

**Output**: `data/final_history_shock/analysis/` (6 CSVs + 2 PNGs)

---

## Step 13: Summary Report (`13_write_summary_report.py`)

**Purpose**: Read all analysis tables and produce a markdown report. No LLM calls.

**Output**: `data/final_history_shock/reports/final_simulation_report.md`

---

## Step 14: Unified Black-Box Supply (`14_unified_bb_supply.py`)

**Purpose**: Generate unified (single-call) LLM recommendations for the architecture gap (Gamma) estimation.

**Scale**: 120 Ollama calls (60 personas × 2 conditions).

**Key difference from modular**: The LLM selects a product AND writes the recommendation in a single call. There is no researcher-controlled separation between retrieval and expression.

### Z=0: Unified feature-only prompt (`build_unified_generic_prompt`)
```
You are a product recommender. You only have access to the product catalog and the consumer's stated needs. You do NOT have access to sales data, review summaries, popularity rankings, or historical purchase feedback. Recommend based solely on product specifications and how they match the consumer's requirements.

Recommend the single best product from the catalog for this consumer. Select the product AND write a consumer-facing recommendation in one response.

Write a straightforward product recommendation based on the product specs and the consumer's stated needs. Be factual and concise. Stick to what is listed in the product features. Do not speculate about popularity, user experience, or track record — you only have the spec sheet.

Consumer profile:
[persona details]

Product catalog:
[30 products, specs only — no ratings, no review counts, no popularity]

[few-shot block]

Return ONLY valid JSON:
{
  "selected_product_id": "...",
  "recommendation_text": "3-5 sentence consumer-facing recommendation based on specs",
  "tradeoff_text": "1-2 sentences on the main limitation or tradeoff",
  "persuasion_text": "1 sentence on the strongest reason to buy",
  "selection_rationale": "1-2 sentences on why this product fits best"
}
```

### Z=1: Unified history-aware prompt (`build_unified_history_prompt`)
```
You are an experienced shopping recommender. You have access to product attributes, public review summaries, and internal historical purchase-feedback patterns. Your goal is not to recommend the most popular product. Your goal is to recommend the product most suitable for this specific consumer. Use historical evidence as background only. Do not cite exact sales numbers, conversion rates, satisfaction rates, ranks, sample sizes, or percentages. You may refer qualitatively to evidence such as "often chosen by similar buyers," "strong post-purchase feedback among travelers," "mixed feedback from budget buyers," or "frequent complaints about comfort."

Recommend the single best product from the catalog for this consumer. Select the product AND write a consumer-facing recommendation in one response. Use historical evidence to inform both your selection and your writing.

Write like an experienced shopping advisor who has seen thousands of customers buy and return products. You have deep knowledge of how products perform in the real world — which ones customers love, which ones get returned, and which ones surprise people. Be opinionated and specific. Use your historical insight to give advice that goes beyond the spec sheet: mention real-world reliability, common buyer satisfaction patterns, and how this product compares to alternatives that similar buyers have tried. Your goal is to help the consumer avoid regret.

Consumer profile:
[persona details]

Internal historical purchase-feedback data (use as background evidence only):
[segment-relevant history summaries for all products]

Product catalog (with review counts and popularity):
[30 products with ratings, review counts, popularity tiers]

[few-shot block]

Consider both the consumer's stated needs AND which products have historically satisfied similar buyers. Write like an experienced advisor who has seen real outcomes.

Do not cite exact sales numbers, conversion rates, satisfaction rates, ranks, sample sizes, or percentages. You may refer qualitatively to evidence such as "often chosen by similar buyers," "strong post-purchase feedback among travelers," "mixed feedback from budget buyers," or "frequent complaints about comfort."

Return ONLY valid JSON:
{
  "selected_product_id": "...",
  "recommendation_text": "3-5 sentence recommendation reflecting real-world buyer experience",
  "tradeoff_text": "1-2 sentences on limitations informed by actual user complaints",
  "persuasion_text": "1 sentence on the strongest reason to buy, grounded in buyer track record",
  "selection_rationale": "1-2 sentences on why this product fits best",
  "history_language_used": "none|weak|moderate|strong"
}
```

**Seed offsets**: +50 (Z=0), +60 (Z=1) — distinct from all modular seeds.

**Output**: `data/final_history_shock/unified_bb/unified_bb_supply.csv` (120 rows)

---

## Step 14b: GPT Evaluation of Unified BB (`14b_gpt_eval_unified_bb.py`)

**Purpose**: Apply the same GPT absolute evaluation protocol (Step 10) to the 120 unified BB recommendation packages.

**Scale**: 120 GPT calls. Same system prompt and user prompt as Step 10.

**Output**: `data/final_history_shock/unified_bb/unified_bb_eval.csv` (120 rows)

---

## Step 14c: Compute Architecture Gap Gamma (`14c_compute_gamma.py`)

**Purpose**: Compute the architecture gap Gamma = tau^BB - tau^MOD with bootstrap confidence intervals.

**Quantities computed per outcome variable**:
```
mu^BB_0  = mean outcome under unified Z=0
mu^BB_1  = mean outcome under unified Z=1
tau^BB   = mu^BB_1 - mu^BB_0

mu^MOD_00 = mean outcome under modular cell (0,0)
mu^MOD_11 = mean outcome under modular cell (1,1)
tau^MOD   = mu^MOD_11 - mu^MOD_00

Gamma     = tau^BB - tau^MOD
Approximation ratio = tau^MOD / tau^BB
```

**Bootstrap**: B=2000, cluster-level, seed=42. 95% CIs from 2.5th/97.5th percentiles.

**Also reports**:
- Product agreement between unified and modular architectures (BB Z=0 vs MOD cell 0; BB Z=1 vs MOD cell 11)
- Modular decomposition (Delta_J, Delta_T, Delta_JT)

**Output**:
- `data/final_history_shock/unified_bb/gamma_estimates.csv`
- `data/final_history_shock/unified_bb/gamma_report.md`

---

## Utility Modules

### `utils_local_llm.py`
- `ollama_generate()`: Calls Ollama REST API at `http://localhost:11434/api/generate`. Model: `qwen2.5:14b`, temperature: 0.7, keep_alive: 30 min.
- `ollama_json_call()`: Wraps `ollama_generate` with JSON parsing and up to 3 retries on parse failure.
- `append_jsonl()`, `load_jsonl()`: JSONL file I/O for caching.

### `utils_openai.py`
- `gpt_call()`: Calls OpenAI API via `client.responses.create()`. Model: `gpt-5.3-chat-latest`. Retry: 3 attempts with exponential backoff (tenacity).
- `gpt_json_call()`: Wraps `gpt_call` with JSON parsing, markdown fence stripping, and retry on parse failure.

### `utils_parse.py`
- `detect_leakage()`: Scans text against 18 forbidden regex patterns.
- `has_leakage()`: Boolean wrapper.
- `clean_json_text()`: Strip markdown fences.

### `utils_stats.py`
- `fit_bradley_terry()`: Maximum likelihood BT model via Nelder-Mead.
- `decompose_from_utilities()`: Extract retrieval/expression/total/interaction from BT theta vector.
- `cluster_bootstrap_bt()`: Cluster-level bootstrap of BT decomposition.

---

## Step 15: Mechanism Diagnostics (`15_diagnostics.py`)

**Purpose**: Diagnose the mechanism behind the negative retrieval effect and the purchase interaction. No new LLM calls — reads existing CSV/JSON outputs only.

**Outputs**:

### 15.1 Cluster-level diagnostic dataset
Merges supply, absolute eval, pairwise eval, catalog metadata, persona metadata, and history affinity into one row per cluster.
- **File**: `data/final_history_shock/analysis/diagnostics/cluster_level_diagnostics.csv` (60 rows × 102 columns)

### 15.2 Retrieval switch anatomy
For each cluster where retrieval changed the product (49/60 = 82%), computes:
- Price diff (history − generic product)
- Log rating-count diff
- Average rating diff
- Popularity rank improvement (generic_rank − history_rank; positive = more popular)
- Budget violation switch (generic within budget, history exceeds budget)
- Feature-fit heuristic overlap (persona must-have features vs. product keywords)
- Avoid-violation overlap
- Segment affinity diff (using persona→segment heuristic mapping)

Reported for 4 subgroups: all clusters, retrieval-changed only, history-retrieval-loses-pairwise, history-retrieval-wins-pairwise.

**Key finding**: History retrieval shifts toward **more expensive** (+$25 mean), **less popular** (rank worsens by 3.8), and **fewer-reviewed** products. This is a premium-aspiration bias, not a popularity bias. Budget violation switch rate: 27%.

- **File**: `data/final_history_shock/analysis/diagnostics/table_retrieval_switch_anatomy.csv`

### 15.3 Absolute DID by outcome
Cluster-level difference-in-differences for 8 outcome variables × 5 components:
- Retrieval effect under generic expression: cell10 − cell00
- Retrieval effect under history expression: cell11 − cell01
- Expression effect under generic retrieval: cell01 − cell00
- Expression effect under history retrieval: cell11 − cell10
- Interaction DID: (cell11 − cell10) − (cell01 − cell00)

All with cluster bootstrap (B=2000, seed=42).

**Key findings** (CI excludes zero marked with *):
- Retrieval hurts fit* (−0.70), purchase* (−13.8 pp), satisfaction* (−7.5 pp), persuasive intensity* (−0.32)
- Retrieval increases regret risk* (+0.60) and tradeoff disclosure* (+0.60)
- Expression increases trust* (+0.58) and tradeoff disclosure* (+0.97)
- Trust has significant negative interaction* (−0.37): the trust gain from history expression is smaller when retrieval is also history-aware
- Tradeoff disclosure has significant negative interaction* (−0.60): similar diminishing-returns pattern
- Purchase interaction is near zero in absolute scale (−0.4 pp, CI [−3.1, +2.2])

- **File**: `data/final_history_shock/analysis/diagnostics/table_absolute_did_by_outcome.csv`

### 15.4 Purchase interaction diagnostics
Tests whether the BT pairwise positive purchase interaction (+0.426, P>0=95.4%) replicates in absolute evaluation.

**Result**: It does not. Absolute purchase interaction: −0.4 pp [−3.1, +2.2]. Only 11/60 clusters (18%) show the compensatory pattern (history expression helps purchase under history retrieval but not under generic retrieval).

Includes qualitative examples (6 positive interaction + 2 counterexamples).

- **Files**:
  - `data/final_history_shock/analysis/diagnostics/table_purchase_interaction_drivers.csv`
  - `data/final_history_shock/reports/purchase_interaction_examples.md`

### 15.5 Pairwise reason coding
Heuristic keyword classification of GPT evaluator reason text for 3 key comparisons:

| Comparison | Question | Top winner | Top reason codes |
|------------|----------|------------|-----------------|
| 00 vs 10 | Why does generic retrieval win? | Cell 00 (73%) | feature_match (53%), tradeoff_disclosure (40%), budget_fit (37%) |
| 10 vs 11 | Why does history expression win? | Cell 11 (63%) | tradeoff_disclosure (83%), trust (62%), budget_fit (33%) |
| 01 vs 11 | Why does (0,1) beat (1,1)? | Cell 01 (72%) | feature_match (48%), budget_fit (40%), tradeoff_disclosure (40%) |

- **Files**:
  - `data/final_history_shock/analysis/diagnostics/table_pairwise_reason_codes.csv`
  - `data/final_history_shock/reports/pairwise_reason_summary.md`

### 15.6 Paper-ready figures

| Figure | Description | File prefix |
|--------|-------------|-------------|
| A | BT decomposition bar chart with 95% CIs | `fig_main_decomposition` |
| B | 2×2 cell means for 5 outcome dimensions | `fig_cell_means_outcomes` |
| C | Retrieval switch anatomy (product attribute diffs) | `fig_retrieval_switch_anatomy` |
| D | Fit loss vs. price/review-count/popularity shift (scatter) | `fig_retrieval_harm_scatter` |
| E | Expression lift on purchase by retrieval condition | `fig_purchase_interaction` |
| F | 4×4 pairwise win-rate heatmap | `fig_pairwise_win_matrix` |

All saved as PNG+PDF in `data/final_history_shock/analysis/figures_diagnostics/` and `paper/figures/`.

### 15.7 Mechanism diagnostics report
Comprehensive narrative combining all diagnostic findings.
- **File**: `data/final_history_shock/reports/mechanism_diagnostics_report.md`

---

## Data Flow Summary

```
Real Amazon metadata ──→ [Step 1] Catalog (30 products)
                         [Step 5] Historical DGP (240 summaries, 30 prod × 8 seg)

GPT-5.3 ──→ [Step 2] Exemplars (180 calls, 6 cat × 10 pers × 3 regimes)
             [Step 3] Personas (60 per category, 6 batches of 10)
             [Step 6] Few-shot prompt blocks (2 exemplars per regime)

Qwen 14B (local) ──→ [Step 7] Smoke run (12 packages, 3 personas × 4 cells)
                      [Step 8] Full supply (240 packages, 60 pers × 4 cells, 6 calls each)
                      [Step 14] Unified BB supply (120 packages, 60 pers × 2 conditions)

Regex audit ──→ [Step 9] Leakage check + regeneration (18 forbidden patterns)

GPT-5.3 ──→ [Step 10] Absolute eval (240 ratings, 8 outcome scales)
             [Step 11] Pairwise eval (360 comparisons, 6 pairs × 60 clusters)
             [Step 14b] Unified BB eval (120 ratings)

Analysis ──→ [Step 12] BT decomposition + tables + figures
             [Step 13] Summary report
             [Step 14c] Gamma computation + report

Diagnostics ──→ [Step 15] Mechanism diagnostics (5 tables, 6 figures, 3 reports)
                           Reads all existing outputs, no new LLM calls
```
