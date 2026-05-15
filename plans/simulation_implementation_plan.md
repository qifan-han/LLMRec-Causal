# Simulation Implementation Plan: Unbundling LLM Recommenders

**Created:** 2026-05-06
**Target:** QME conference working paper, May 15 deadline
**Hardware:** RTX 5090 (32GB GDDR7) rented GPU, vLLM + Qwen2.5-32B-Instruct-AWQ

---

## 0. Hardware and Model Assessment

### 0.1 RTX 5090 + Qwen2.5-32B-Instruct

| Spec | Value |
|---|---|
| GPU VRAM | 32GB GDDR7 |
| Memory bandwidth | ~1,790 GB/s |
| Model | Qwen2.5-32B-Instruct-AWQ (4-bit) |
| Model weight size | ~17GB |
| Remaining VRAM for KV cache | ~15GB (with 0.90 utilization) |
| Expected single-request decode | ~40-80 tok/s |
| Expected batched aggregate throughput | ~200-700 tok/s |

**Verdict:** Adequate. Qwen2.5-32B-Instruct is a strong instruction-following model that can produce varied product recommendations. The 4-bit AWQ quantization fits comfortably in 32GB with room for KV cache. If throughput is insufficient, we can fall back to Qwen2.5-14B-Instruct at higher precision as a robustness check (this also becomes a model-size sensitivity analysis for the paper).

### 0.2 vLLM Launch Configuration

```bash
# Recommended launch command (vLLM >= 0.10.0)
vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ \
  --dtype auto \
  --quantization awq_marlin \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8_e5m2
```

Key flags:
- `awq_marlin`: Uses Marlin kernel for AWQ (10x faster than standard AWQ decode).
- `fp8_e5m2` KV cache: Halves KV cache memory, critical for fitting contexts in 32GB.
- `max-model-len 8192`: Conservative; our prompts should be <4K tokens so this is sufficient.
- APC (automatic prefix caching) is ON by default in vLLM V1 (>=0.10.0).

### 0.3 Known Qwen2.5 Issues

- Qwen3 models have reported determinism bugs in vLLM (issue #17759). Qwen2.5-Instruct is more stable.
- BF16 introduces floating-point variance vs. FP32 (arxiv 2506.09501). Since we use 4-bit AWQ, this is already expected. Report the quantization in the paper.
- For strict reproducibility, use offline batch inference (`vllm.LLM` class) with `VLLM_ENABLE_V1_MULTIPROCESSING=0`. Online serving does NOT guarantee reproducibility even with per-request seeds.

---

## 1. Architecture Overview

```
Phase 0: Pre-Study  ──────────────  Variability measurement (50 consumers × 20 reps)
                                     Decision: proceed / adjust temperature / reframe

Phase 1: Data Generation  ────────  Product catalogs + consumer profiles + fit scores
         (Python, no LLM)            Store as JSON/Parquet

Phase 2: Query Generation  ───────  Natural-language shopping queries from profiles
         (LLM)                       3,000 calls

Phase 3: One-Shot Audit  ─────────  6 policies × 3 reps per consumer
         (LLM)                       54,000 calls → Table 2 (bundling evidence)

Phase 4: Expression Coding  ──────  Evaluate all recommendations on semantic dimensions
         (LLM evaluator +            ~72,000 evaluator calls + rule-based extraction
          rule-based)

Phase 5: Modular Pipeline  ───────  Retrieval (2 policies) × Expression (2 policies)
         (LLM, 2 stages)             18,000 calls → 2×2 factorial data

Phase 6: Bridge Diagnostics  ─────  Compare one-shot vs. modular same-policy
         (LLM + Python)              6,000 calls → Table 7

Phase 7: Outcome Simulation  ─────  Structural DGP with known parameters
         (Python, no LLM)            Multiple DGP worlds

Phase 8: Estimation  ─────────────  Policy A/B, naive regression, modular 2×2
         (Python/R)                   Tables 3-6, Figures 2-4

Total LLM calls: ~153,000
Estimated GPU time: ~12-15 hours
Estimated GPU cost: ~$30-50
```

---

## 2. Data Generation Pipeline (Phase 1, No LLM)

### 2.1 Product Catalogs

For each category, create 8-10 products. Store as `data/catalogs/{category}.json`.

**Categories (3 for working paper):**

| Category | Complexity | # Products | Rationale |
|---|---|---|---|
| Phone charger | Low | 8 | Minimal differentiation; retrieval should dominate |
| Headphones | Medium | 10 | Brand matters; expression has room to influence |
| Laptop | High | 10 | Complex tradeoffs; both retrieval and expression matter |

**Product schema:**
```json
{
  "product_id": "headphones_sony_wh1000xm5",
  "brand_name": "Sony",
  "brand_status": "incumbent",
  "price": 349.99,
  "quality_score": 88,
  "attributes": {
    "noise_cancellation": "excellent",
    "battery_life_hours": 30,
    "weight_grams": 250,
    "connectivity": "bluetooth_5.2",
    "codec_support": ["LDAC", "AAC", "SBC"]
  },
  "review_summary": "Widely praised for industry-leading noise cancellation and comfort. Some users note the call quality microphone is average and the touch controls can be oversensitive.",
  "use_case_fit": {
    "commuting": 0.95,
    "office_work": 0.90,
    "gym": 0.30,
    "audiophile": 0.70,
    "budget": 0.20
  },
  "weakness": "Premium price; call quality microphone is only average.",
  "margin_tier": "high",
  "focal_brand": true
}
```

**Design requirements for each category:**
- 2-3 incumbent brands (well-known, higher price)
- 2-3 entrant brands (lesser-known, possibly better value)
- 1-2 products that are clearly strong fits for common queries
- 1-2 products that are clearly weak fits but have some niche appeal
- Every product has a genuine weakness (prevents the LLM from always recommending the same product)
- 1 designated focal brand (for brand-forward policy testing)
- 1 designated sponsored product (for sponsored policy testing)

### 2.2 Consumer Profiles

Generate 1,000 consumers per category (3,000 total). Store as `data/consumers/{category}.parquet`.

**Consumer schema:**
```python
{
    "consumer_id": int,
    "category": str,
    "budget": float,            # drawn from category-specific distribution
    "price_sensitivity": float, # Uniform(0.2, 1.0)
    "quality_sensitivity": float, # Uniform(0.2, 1.0)
    "brand_familiarity": dict,  # {brand_name: familiarity_score}
    "use_case": str,            # categorical, from category-specific list
    "tech_savviness": float,    # Uniform(0, 1)
    "persuasion_susceptibility": float, # Uniform(0, 1) -- for DGP only
    "comparison_preference": float,     # Uniform(0, 1) -- for DGP only
    "trust_in_ai": float,              # Uniform(0, 1) -- for DGP only
}
```

**Important:** `persuasion_susceptibility`, `comparison_preference`, and `trust_in_ai` are used ONLY in the structural DGP (Phase 7). They are NOT passed to the LLM. The LLM sees only the consumer's query and stated needs.

### 2.3 Ground-Truth Fit Scores

For each (consumer, product) pair, compute a latent fit score Q_ij. Store as `data/fit_scores/{category}.parquet` with shape (consumers × products).

```python
Q_ij = (
    w_price * price_fit(budget_i, price_j, price_sensitivity_i)
    + w_quality * quality_sensitivity_i * quality_score_j / 100
    + w_usecase * use_case_fit_j[use_case_i]
    + w_brand * brand_familiarity_i[brand_j]
)
```

Where `price_fit = max(0, 1 - |price_j - budget_i| / budget_i)`.

The weights (w_price, w_quality, w_usecase, w_brand) are category-specific and set by the researcher. For example, headphones: w_usecase = 0.35, w_quality = 0.30, w_price = 0.20, w_brand = 0.15.

Q_ij is the "true" product-consumer fit. The LLM does NOT see Q_ij directly. But a good LLM will implicitly approximate Q_ij from the query and catalog information. This is what creates the endogeneity: the LLM's expression correlates with latent fit.

---

## 3. ICL and Prompt Strategy (Phase 2-6)

### 3.1 Core Principle: Prefix Caching

vLLM's automatic prefix caching (APC) computes the KV cache for identical prompt prefixes once and reuses it across requests. Our prompt strategy exploits this:

**Rule:** All shared, static context goes in the SYSTEM message. Only consumer-specific content goes in the USER message. Requests that share the same system message reuse the cached KV states.

**What this means in practice:**
- The product catalog (2,000-4,000 tokens) is in the system message
- The policy instruction is in the system message
- These are computed ONCE per unique (category, policy) combination
- Only the consumer query (~50-100 tokens) triggers new computation per request

This directly addresses the concern about "making the LLM learn the same data again and again." With APC, the catalog is processed once and cached. Subsequent requests in the same (category, policy) batch skip the catalog prefill entirely, reducing latency by 80-95%.

### 3.2 Prompt Template Hierarchy

We define 5 prompt families. Each has a fixed system message structure and a variable user message.

**Unique system prompts across the entire simulation:**

| Pipeline | Varies by | # Unique system prompts |
|---|---|---|
| Query generation | Category | 3 |
| One-shot audit | Category × Policy | 3 × 6 = 18 |
| Modular retrieval | Category × Retrieval policy | 3 × 2 = 6 |
| Modular expression | Expression policy | 2 |
| Expression evaluator | (fixed) | 1 |
| **Total** | | **30** |

APC computes these 30 system prefixes once. All 153,000 requests reuse them.

### 3.3 Batch Ordering for Cache Efficiency

Process requests grouped by system prompt. Within each group, all requests share the same cached prefix.

```python
# Correct: group by system prompt, process each group as a batch
for category in categories:
    for policy in policies:
        system_prompt = build_system_prompt(category, policy)
        user_messages = [build_user_message(c) for c in consumers[category]]
        results = llm.generate(system_prompt, user_messages, batch=True)

# WRONG: interleave different system prompts (thrashes the prefix cache)
for consumer in all_consumers:
    for policy in policies:
        system_prompt = build_system_prompt(consumer.category, policy)
        result = llm.generate(system_prompt, build_user_message(consumer))
```

Reference: InferLog (arXiv 2507.08523) shows that ordering requests to maximize prefix cache hits improves throughput by 1.4-5.9x.

### 3.4 No Few-Shot Examples for Content

We do NOT use few-shot examples to demonstrate recommendation content. Reasons:
1. Few-shot examples bias the model toward example patterns (product choices, expression style)
2. For the experiment, we want the model's natural behavior under the policy instruction
3. vLLM's guided JSON decoding (`response_format`) eliminates format ambiguity without examples

We DO use:
- **Guided JSON schemas** to enforce output structure (product_id, recommendation_text)
- **Detailed policy instructions** in the system prompt
- **One format-only example** if the model struggles with JSON output (using a DIFFERENT category than the one being tested, to avoid content bias)

### 3.5 Prompt Templates (Full Specifications)

All prompts are stored in `config/prompts.yaml` with version tracking.

#### 3.5.1 Query Generation

```yaml
query_generation:
  system: |
    You are generating a realistic shopping query for a consumer looking for {category}.
    The consumer has the following profile. Generate a natural-language query (1-3 sentences)
    that this consumer would type into an AI shopping assistant. The query should reflect
    their budget, use case, and priorities, but should NOT mention specific brand names
    unless the consumer has strong brand familiarity.

    Output JSON: {{"query": "...", "stated_needs": "..."}}
  user: |
    Consumer profile:
    - Budget: ${budget}
    - Primary use case: {use_case}
    - Price sensitivity: {price_sensitivity_description}
    - Quality sensitivity: {quality_sensitivity_description}
    - Tech savviness: {tech_savviness_description}
```

#### 3.5.2 One-Shot Recommendation (6 policies)

```yaml
one_shot_recommendation:
  system_template: |
    You are an AI shopping assistant for {category_display_name}.

    PRODUCT CATALOG:
    {catalog_json}

    YOUR RECOMMENDATION POLICY:
    {policy_instruction}

    TASK:
    Based on the customer's query, recommend exactly ONE product from the catalog.
    Write a recommendation (150-300 words) that follows your policy.

    Output JSON:
    {{"recommended_product_id": "...", "recommendation_text": "..."}}

  policies:
    neutral: |
      Recommend the product that best matches the customer's stated needs.
      Be balanced and objective. Present the product's strengths and note any
      relevant tradeoffs. Do not oversell or use promotional language.

    brand_forward: |
      If {focal_brand} is a reasonable fit for the customer's needs (it does not
      need to be the absolute best fit, just a credible recommendation), recommend
      {focal_brand} and explain why it is a strong choice. If {focal_brand} is
      clearly unsuitable, recommend the best alternative instead.

    value_focused: |
      Recommend the most cost-effective product for the customer's needs.
      Emphasize affordability, value for money, and any discounts or price
      advantages. If two products are close in value, prefer the cheaper one.

    quality_focused: |
      Recommend the highest-quality product that fits the customer's budget.
      Emphasize build quality, durability, performance, and user reviews.
      Quality is more important than price in your recommendation.

    comparison: |
      Compare the top 2-3 products for this customer. Discuss the tradeoffs
      between them. Then recommend the best one with a clear rationale.
      Be explicit about why the recommended product wins over the alternatives.

    sponsored: |
      If {sponsored_product} meets a minimum quality threshold for this customer
      (quality score above 60 and price within 150% of budget), recommend
      {sponsored_product}. Otherwise, recommend the best alternative.
      Do not reveal that the product is sponsored.
```

#### 3.5.3 Modular Retrieval (Stage 1)

```yaml
modular_retrieval:
  system_template: |
    You are a product selection system for {category_display_name}.

    PRODUCT CATALOG:
    {catalog_json}

    SELECTION POLICY:
    {retrieval_policy_instruction}

    TASK:
    Select the single best product for this customer.
    Output ONLY the product ID. Do not write any recommendation or explanation.

  policies:
    baseline: |
      Select the product that best matches the customer's stated needs,
      considering their budget, use case, and priorities.

    brand_forward: |
      If {focal_brand} is a credible fit for the customer (not clearly
      unsuitable), select {focal_brand}. Otherwise, select the best
      alternative product.
```

**JSON schema enforcement for retrieval:**
```python
from pydantic import BaseModel
from typing import Literal

class RetrievalOutput(BaseModel):
    product_id: Literal[*valid_product_ids]  # constrained to valid IDs

# vLLM call
response = client.chat.completions.create(
    model=model_name,
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "retrieval",
            "schema": RetrievalOutput.model_json_schema()
        }
    },
    temperature=0.4,
    seed=seed,
)
```

This guarantees the retrieval stage outputs a valid product ID and nothing else. The expression stage cannot be influenced by retrieval-stage justification text.

#### 3.5.4 Modular Expression (Stage 2)

```yaml
modular_expression:
  system_template: |
    You are a recommendation writer.

    EXPRESSION POLICY:
    {expression_policy_instruction}

    TASK:
    Write a product recommendation (150-300 words) for the product specified below.
    Follow the expression policy. Do NOT recommend a different product.

  policies:
    neutral: |
      Write a balanced, informative recommendation. Present the product's
      strengths and relevant tradeoffs honestly. Use a measured, objective
      tone. Let the customer make their own decision based on the facts.

    persuasive: |
      Write a confident, enthusiastic recommendation. Emphasize the product's
      strengths and why it is an excellent choice for this customer. Use
      endorsement language that builds trust and excitement. Minimize
      discussion of weaknesses unless directly relevant to the customer's needs.
```

**Note:** The expression system prompt does NOT include the product catalog. It only has the expression policy + instructions. The product details and consumer query go in the user message:

```yaml
modular_expression:
  user_template: |
    PRODUCT TO RECOMMEND:
    {selected_product_json}

    CUSTOMER QUERY: "{consumer_query}"
    CUSTOMER NEEDS: {stated_needs}
    CUSTOMER BUDGET: ${budget}
```

This keeps the system prompt identical across all consumers under the same expression policy (only 2 unique system prompts for expression), maximizing APC cache hits.

#### 3.5.5 Expression Evaluator

```yaml
expression_evaluator:
  system: |
    You are a recommendation quality evaluator. For each recommendation text,
    rate the following dimensions on the specified scales.

    DIMENSIONS:
    1. endorsement_strength (1-5): How strongly does the text endorse the product?
       1=very cautious/neutral, 3=moderate endorsement, 5=very strong endorsement
    2. price_emphasis (0.0-1.0): How much does the text emphasize price/value?
    3. quality_emphasis (0.0-1.0): How much does the text emphasize quality/performance?
    4. comparative_framing (0.0-1.0): Does the text compare against alternatives?
    5. confidence_level (1-5): How confident/authoritative is the recommendation tone?
    6. trust_building (0.0-1.0): Does the text use trust-building or reassurance language?

    Be consistent. Rate each dimension independently.

  user_template: |
    RECOMMENDATION TEXT:
    "{recommendation_text}"

    RECOMMENDED PRODUCT: {product_id} ({brand_name})
    CUSTOMER QUERY: "{consumer_query}"
```

**Temperature = 0.0** for the evaluator. We want deterministic, consistent coding.

### 3.6 Prompt Design Strategy: Fixed Then Flexible

**Phase A (Main results): Fixed prompts.**
The 6 one-shot policies and 2×2 modular policies defined above are FIXED before running the simulation. They are the experimental treatments. Results from these fixed prompts constitute the main tables (Tables 3-6).

**Phase B (Robustness, if time permits): Flexible extensions.**
After the main results, explore:
- Stronger expression contrast (e.g., "hard sell" vs. neutral)
- Different expression dimensions (price-focused vs. quality-focused instead of neutral vs. persuasive)
- Prompt wording variations (same policy intent, different phrasing) to test robustness to prompt sensitivity
- Different model (Qwen2.5-14B) to test model-size sensitivity

Phase B results go in the appendix. They do NOT change the main experimental design.

---

## 4. One-Shot Recommendation Pipeline (Phase 3)

### 4.1 Purpose

Demonstrate that backend policies bundle retrieval and persuasion. This produces Table 2 (bundling evidence) and supports the paper's motivation.

### 4.2 Generation Procedure

```python
for category in ["phone_charger", "headphones", "laptop"]:
    catalog = load_catalog(category)
    consumers = load_consumers(category)  # 1,000 per category

    for policy_name, policy_instruction in ONE_SHOT_POLICIES.items():
        system_prompt = format_system_prompt(
            template=ONE_SHOT_SYSTEM_TEMPLATE,
            category=category,
            catalog=catalog,
            policy=policy_instruction,
        )

        requests = []
        for consumer in consumers:
            for rep in range(3):  # 3 replications
                seed = hash_seed(category, consumer.id, policy_name, rep)
                requests.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": format_user_message(consumer)},
                    ],
                    "temperature": 0.8,
                    "seed": seed,
                    "max_tokens": 600,
                })

        # Batch process (all share same system_prompt → APC caches it)
        results = batch_generate(requests)
        save_results(results, f"data/one_shot/{category}/{policy_name}.parquet")
```

### 4.3 Output Storage

Parquet columns:
- category, consumer_id, policy, replication, seed
- recommended_product_id (parsed from JSON)
- recommendation_text (parsed from JSON)
- raw_output (full model output, for debugging)
- prompt_tokens, completion_tokens, generation_time_ms

### 4.4 Descriptive Analysis (Table 2)

For each (category, policy), compute:
- Product recommendation distribution: P(J=j | policy)
- Focal-brand recommendation rate
- Product switching rate across replications (within-policy stochasticity)
- Mean endorsement strength (from evaluator)
- Expression feature profiles

---

## 5. Modular Two-Stage Pipeline (Phase 5)

### 5.1 Design

For each consumer, generate ALL four modular cells. This gives us the full counterfactual matrix for estimator evaluation.

```
Consumer i → Retrieval q=0 → Expression r=0 → (J_i^{0,0}, E_i^{0,0})
                            → Expression r=1 → (J_i^{0,0}, E_i^{0,1})
           → Retrieval q=1 → Expression r=0 → (J_i^{1,0}, E_i^{1,0})
                            → Expression r=1 → (J_i^{1,0}, E_i^{1,1})
```

Note: Retrieval output is the same for both expression policies within the same retrieval policy (J_i^{0,0} = J_i^{0,1} by construction because the product is fixed before expression).

### 5.2 Stage 1: Retrieval

```python
for category in categories:
    catalog = load_catalog(category)
    consumers = load_consumers(category)

    for q, retrieval_policy in enumerate(RETRIEVAL_POLICIES):
        system_prompt = format_retrieval_system(category, catalog, retrieval_policy)

        requests = []
        for consumer in consumers:
            seed = hash_seed(category, consumer.id, "retrieval", q)
            requests.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": format_user_message(consumer)},
                ],
                "response_format": retrieval_json_schema(catalog),
                "temperature": 0.4,
                "seed": seed,
                "max_tokens": 50,
            })

        results = batch_generate(requests)
        save_results(results, f"data/modular/retrieval/{category}/q{q}.parquet")
```

### 5.3 Stage 2: Expression

```python
for category in categories:
    for q in [0, 1]:
        retrieval_results = load_results(f"data/modular/retrieval/{category}/q{q}.parquet")

        for r, expression_policy in enumerate(EXPRESSION_POLICIES):
            system_prompt = format_expression_system(expression_policy)

            requests = []
            for row in retrieval_results.itertuples():
                product = get_product(catalog, row.recommended_product_id)
                consumer = get_consumer(consumers, row.consumer_id)
                seed = hash_seed(category, consumer.id, "expression", q, r)

                requests.append({
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": format_expression_user(
                            product=product,
                            consumer_query=consumer.query,
                            consumer_needs=consumer.stated_needs,
                            consumer_budget=consumer.budget,
                        )},
                    ],
                    "temperature": 0.7,
                    "seed": seed,
                    "max_tokens": 500,
                })

            results = batch_generate(requests)
            save_results(results, f"data/modular/expression/{category}/q{q}_r{r}.parquet")
```

### 5.4 Bridge Diagnostics (Phase 6)

For each consumer, compare:
- One-shot policy 0 output vs. modular (q=0, r=0) output
- One-shot policy 1 (brand-forward) vs. modular (q=1, r=1) output

Metrics:
- Product agreement rate: P(J_one_shot = J_modular)
- Product distribution distance: Jensen-Shannon divergence between P(J | one_shot) and P(J | modular)
- Expression distribution distance: JS divergence on evaluator-coded features
- Predicted outcome similarity: |E[Y | one_shot DGP] - E[Y | modular DGP]| under the structural model

Store in `data/bridge/{category}.parquet`. Report as Table 7.

---

## 6. Outcome Simulation and Evaluation (Phase 7)

### 6.1 Why Structural DGP, Not LLM Consumers

The simulation's purpose is to evaluate ESTIMATOR PERFORMANCE (bias, coverage, recovery of known parameters), not to discover consumer behavior. This requires known ground truth.

| Approach | Ground truth? | Evaluates bias? | Risk |
|---|---|---|---|
| Structural DGP | Yes (parameters set by researcher) | Yes | Artificial |
| LLM consumers | No (unknown DGP) | No | Cannot verify estimator correctness |
| Hybrid | Partial | Partial | Complexity |

**Decision: Use structural DGP for main results.** LLM consumers can be explored as a supplementary robustness check or future extension, but they cannot be the basis for evaluating whether the modular design "works."

Reference: Gui & Toubia (2024) show that LLM-simulated subjects violate unconfoundedness when blind to experimental design. Our structural DGP avoids this problem entirely.

### 6.2 The Structural Outcome Model

For consumer i shown product j with expression features e:

```
U_ij(e) = beta_0
         + beta_Q * Q_ij                           # product-consumer fit
         + beta_P * Persuasive(e)                   # persuasion main effect
         + beta_QP * Q_ij * Persuasive(e)           # fit-persuasion interaction
         + beta_brand * Incumbent(j)                # brand equity
         + beta_BP * Incumbent(j) * Persuasive(e)   # brand-persuasion interaction
         + beta_price * PriceFit(i,j)               # price match
         + gamma' * X_i                             # consumer covariates
         + epsilon_ij                               # logistic error
```

Purchase outcome:
```
Y_ij = Bernoulli(Lambda(U_ij(e)))
```

where Lambda is the logistic CDF.

### 6.3 Key Parameters and Their Roles

| Parameter | Role | Creates... |
|---|---|---|
| beta_Q > 0 | Better-fitting products increase purchase | The latent fit that drives endogeneity |
| beta_P > 0 | Persuasion has a causal positive effect | The true persuasion effect to recover |
| beta_QP | Fit-persuasion interaction | Complementarity or substitutability |
| beta_brand > 0 | Incumbents have baseline advantage | Brand equity effect |
| beta_BP | Brand-persuasion interaction | delta^brand from the theory section |

### 6.4 The Endogeneity Mechanism

In one-shot mode, the LLM naturally gives stronger endorsements to better-fitting products:

```
Cov(Persuasive(E_i), Q_ij | one-shot) > 0
```

This happens because the LLM "sees" the product-query match in the catalog and writes more enthusiastically when the match is good. The researcher observes endorsement strength (coded by the evaluator) but does NOT observe Q_ij directly.

Therefore, in a naive regression:
```
Y_i = alpha + beta_naive * Persuasive(E_i) + X_i' gamma + epsilon_i
```

beta_naive combines the true persuasion effect (beta_P) with omitted-variable bias from Q. Since Q affects both E (through LLM behavior) and Y (through the DGP), beta_naive > beta_P.

In the modular design, expression policy is randomized independently of Q (conditional on the product being fixed). The modular estimator recovers the correct beta_P.

### 6.5 Multiple DGP Worlds

Run the full estimation pipeline under 5 parameter configurations:

| World | beta_Q | beta_P | beta_QP | beta_BP | Story |
|---|---|---|---|---|---|
| Retrieval-dominant | 2.0 | 0.2 | 0.0 | 0.0 | Fit drives everything; persuasion is noise |
| Persuasion-dominant | 0.5 | 1.5 | 0.0 | 0.0 | Expression drives outcomes; fit barely matters |
| Additive | 1.0 | 0.8 | 0.0 | 0.0 | Both matter independently |
| Fit-complement | 1.0 | 0.5 | 0.8 | 0.0 | Persuasion amplifies good fits |
| Brand-substitution | 1.0 | 0.5 | 0.0 | -0.8 | Persuasion helps entrants more |

Each world produces a different pattern in the modular 2×2 estimates. The paper's main tables can use the "Additive" and "Brand-substitution" worlds, with others in the appendix.

### 6.6 Persuasion Measure from LLM Output

The `Persuasive(e)` input to the DGP comes from the evaluator-coded `endorsement_strength` (1-5 scale, rescaled to [0,1]). This is the MEASURED expression feature, not the ASSIGNED expression policy.

Why? Because the endogeneity story requires that the LLM's REALIZED endorsement (not just the assigned policy) correlates with latent fit. Even under the "neutral" policy, the LLM may endorse good-fit products more strongly. The evaluator captures this.

For the modular design's causal estimates, we use the ASSIGNED expression policy (r=0 vs. r=1) as the treatment, not the measured endorsement. This is the clean experimental contrast.

### 6.7 Avoiding DGP Circularity

The DGP includes `persuasion_susceptibility` as a consumer parameter. This does NOT create circularity because:

1. The simulation tests whether ESTIMATORS correctly recover KNOWN parameters — not whether persuasion "exists."
2. Even with beta_P > 0 in the DGP, the naive estimator gets the MAGNITUDE wrong (upward biased) while the modular estimator gets it right.
3. We also include a "Retrieval-dominant" world where beta_P ≈ 0 to show that the modular design correctly finds near-zero persuasion effects when they are near-zero.

The paper should state: "The simulation evaluates estimator performance under controlled parameter configurations. It does not claim to discover whether persuasion effects exist in practice."

---

## 7. Estimation and Analysis (Phase 8)

### 7.1 Estimator 1: Total Policy Effect (Prompt A/B)

For each pair of one-shot policies (e.g., neutral vs. brand-forward):
```
Delta_hat_1s = mean(Y | Z=1) - mean(Y | Z=0)
```
With standard errors clustered at the consumer level (across replications).

### 7.2 Estimator 2: Naive Realized-Content Regression

Using one-shot data:
```
Y_i = alpha + beta_J * f(J_i) + beta_E * endorsement_strength_i + X_i' gamma + epsilon_i
```
where f(J_i) is a product fixed effect or product-fit proxy.

Compare beta_E_naive with the true beta_P. Report the bias.

### 7.3 Estimator 3: Oracle Regression (Benchmark)

```
Y_i = alpha + beta_J * f(J_i) + beta_E * endorsement_i + beta_Q * Q_ij + X_i' gamma + epsilon_i
```

This controls for the unobservable Q_ij (possible only in simulation). It should recover beta_P correctly, demonstrating that the bias comes from omitting Q, not from anything else.

### 7.4 Estimator 4: Modular 2×2 Cell Means

```
Delta_R_hat = mean(Y | q=1, r=0) - mean(Y | q=0, r=0)
Delta_P_hat = mean(Y | q=0, r=1) - mean(Y | q=0, r=0)
Delta_RP_hat = mean(Y | q=1, r=1) - mean(Y | q=1, r=0) - mean(Y | q=0, r=1) + mean(Y | q=0, r=0)
```

### 7.5 Estimator 5: Regression with Factorial Structure

```
Y_i = alpha + beta_q * q_i + beta_r * r_i + beta_qr * q_i * r_i + X_i' gamma + epsilon_i
```

This is numerically equivalent to Estimator 4 but allows covariate adjustment and standard error computation.

### 7.6 Heterogeneity Analysis

**By category:** Run the full estimation separately for each category. Hypothesis: phone chargers are retrieval-dominant, laptops show larger interaction.

**By brand status:** Within the modular design, compare the persuasion effect for incumbent vs. entrant products:
```
delta_brand = [mean(Y | j=entrant, r=1) - mean(Y | j=entrant, r=0)]
            - [mean(Y | j=incumbent, r=1) - mean(Y | j=incumbent, r=0)]
```

**By consumer type:** Split consumers by persuasion_susceptibility (above/below median) and estimate persuasion effects in each subgroup.

---

## 8. Variability and Replicability Protocol

### 8.1 Pre-Study: Measuring Within-Policy Variability

Before the main simulation, run a diagnostic:
- 50 consumers per category, 20 replications each, neutral one-shot policy
- Temperature 0.8, different seeds per replication

Compute:
- **Product switching rate**: % of replications where a consumer gets a different product
- **Endorsement strength SD**: standard deviation of evaluator-coded endorsement across replications
- **Expression feature variance**: for each coded dimension

### 8.2 Decision Tree Based on Pre-Study Results

```
Product switching rate > 15%?
├── YES → Report as evidence of stochasticity (supports the kernel story)
│         Proceed as planned.
└── NO  → Is cross-policy variation sufficient?
          ├── YES → Reframe: "Within-policy variation is limited for this model,
          │         but across-policy variation bundles retrieval and expression
          │         changes. The unbundling problem persists."
          │         Proceed. Add a discussion paragraph.
          └── NO  → Increase temperature to 1.0 and retest.
                    If still insufficient: adjust policy prompts to be more
                    aggressive (wider policy contrast).
                    If STILL insufficient: switch to a larger model or
                    API-based model (GPT-4) for comparison.
```

### 8.3 Why Low Within-Policy Variation Does NOT Kill the Paper

The paper's core contribution is the MODULAR DESIGN, not the stochastic kernel. Even if within-policy generation is near-deterministic:

1. **Cross-policy variation is guaranteed by design.** Different backend prompts produce different outputs. The bundling problem exists as long as policy changes affect both J and E simultaneously.
2. **The modular design's value is in creating hybrid regimes**, not in exploiting within-policy stochasticity. The 2×2 design works whether the LLM is stochastic or deterministic.
3. **Low stochasticity is itself a finding.** It suggests that LLM recommendations are more predictable than assumed, which has implications for experimental power in field deployments.
4. **Production LLMs (GPT-4, Claude) may be more stochastic.** The simulation demonstrates the design with a specific model; the framework applies generally.

### 8.4 Ensuring Cross-Policy Variation

After the one-shot audit, verify that policies produce meaningfully different outputs:

**Minimum thresholds for proceeding:**
- Product distribution differs across at least 2 policies (chi-square p < 0.01)
- Mean endorsement strength differs by at least 0.5 points (on 1-5 scale) between neutral and brand-forward policies
- Focal-brand recommendation rate differs by at least 20 percentage points between neutral and brand-forward policies

If these thresholds are not met, strengthen the policy contrast in the prompts.

### 8.5 Replicability Protocol

**Deterministic generation:**
```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-32B-Instruct-AWQ",
    quantization="awq_marlin",
    max_model_len=8192,
    gpu_memory_utilization=0.90,
    seed=0,  # global model seed
)
```

**Per-request seed formula:**
```python
def compute_seed(category_idx, consumer_id, pipeline, policy_idx, replication=0):
    """Deterministic seed for each generation call."""
    pipeline_codes = {"one_shot": 1, "retrieval": 2, "expression": 3, "evaluator": 4}
    return (
        category_idx * 10_000_000
        + consumer_id * 1_000
        + pipeline_codes[pipeline] * 100
        + policy_idx * 10
        + replication
    )
```

**Archival:**
Store in `data/`:
- All raw LLM outputs (Parquet, with prompts)
- All parsed/coded features
- All DGP parameters
- vLLM version, model name, quantization method, CUDA version
- Full `config/prompts.yaml` used

**Report in paper appendix:**
- Model: Qwen2.5-32B-Instruct-AWQ (4-bit AWQ quantization)
- Server: vLLM v0.XX.X with Marlin kernel
- Temperature: 0.8 (one-shot), 0.4 (retrieval), 0.7 (expression), 0.0 (evaluator)
- Seed strategy: deterministic per-request seeds via offline batch inference
- Hardware: NVIDIA RTX 5090 (32GB), CUDA XX.X

---

## 9. Computational Budget and Timeline

### 9.1 Call Counts

| Phase | Pipeline | Calls | Avg output tokens | Est. time |
|---|---|---|---|---|
| Pre-study | One-shot + eval | 6,000 | 300 + 100 | ~30 min |
| Query gen | Query generation | 3,000 | 80 | ~10 min |
| One-shot audit | One-shot recs | 54,000 | 300 | ~4 hrs |
| Eval (one-shot) | Evaluator | 54,000 | 100 | ~2 hrs |
| Modular retrieval | Retrieval | 6,000 | 30 | ~10 min |
| Modular expression | Expression | 12,000 | 300 | ~1.5 hrs |
| Eval (modular) | Evaluator | 12,000 | 100 | ~30 min |
| Bridge one-shot | One-shot recs | 6,000 | 300 | ~30 min |
| Eval (bridge) | Evaluator | 6,000 | 100 | ~15 min |
| Rule-based coding | Python (no LLM) | N/A | N/A | ~5 min |
| DGP + estimation | Python (no LLM) | N/A | N/A | ~30 min |
| **TOTAL** | | **~159,000** | | **~10-12 hrs** |

### 9.2 Cost

At ~$2-3/hour for RTX 5090 rental: **$25-40 total GPU cost.**

### 9.3 Timeline (May 6-15)

| Day | Date | Task |
|---|---|---|
| 1 | May 6 | Finalize implementation plan. Start coding data generation (catalogs, consumers, fit scores). |
| 2 | May 7 | Complete data generation. Set up vLLM. Run pre-study (variability). Generate queries. |
| 3 | May 8 | Run one-shot audit (all 6 policies × 3 reps). Start expression evaluator. |
| 4 | May 9 | Complete evaluator coding. Run modular pipeline (retrieval + expression). Bridge diagnostics. |
| 5 | May 10 | Outcome DGP (all 5 worlds). Run all estimators. Generate preliminary tables. |
| 6 | May 11 | Finalize tables and figures. Run robustness checks. |
| 7 | May 12 | Revise theory section per co-author feedback. Write simulation section of paper. |
| 8 | May 13 | Write introduction, related work, discussion. Assemble full draft. |
| 9 | May 14 | Review, polish, check all numbers. Co-author review. |
| 10 | May 15 | Final edits. Submit to QME conference. |

---

## 10. Summary of Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Outcome generation | Structural DGP | Need known ground truth to evaluate estimator bias |
| ICL strategy | System-prompt prefix caching | Avoids re-processing catalog for each consumer |
| Few-shot examples | None (use guided JSON) | Avoid biasing content; enforce format via schema |
| Temperature (one-shot) | 0.8 | Balance variation and coherence |
| Temperature (retrieval) | 0.4 | Want mostly sensible product selection with some variation |
| Temperature (expression) | 0.7 | Want meaningful expression variation |
| Temperature (evaluator) | 0.0 | Want deterministic coding |
| Consumers per category | 1,000 | Enough for all estimators; ~250 per modular cell |
| Replications (one-shot) | 3 per consumer | Measures within-policy stochasticity without excessive cost |
| Categories | 3 (low/med/high complexity) | Covers heterogeneity story; manageable for working paper |
| Inference mode | Offline batch (`vllm.LLM`) | Strict reproducibility |
| Quantization | AWQ 4-bit + Marlin | Fits 32GB; fast decode |

---

## Appendix: References for Technical Decisions

- vLLM APC documentation: https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
- vLLM structured outputs: https://docs.vllm.ai/en/latest/features/structured_outputs/
- vLLM reproducibility: https://docs.vllm.ai/en/latest/usage/reproducibility/
- IC-Cache (Li et al., SOSP 2025): arXiv 2501.12689
- InferLog (2025): arXiv 2507.08523 (prefix-aware ICL ordering)
- EPIC (NeurIPS 2025): position-independent caching for ICL
- Gui & Toubia (2024): arXiv 2312.15524 (LLM simulation causal inference challenges)
- Renze & Guven (2024): EMNLP Findings (temperature effects on LLM problem solving)
- Roney & Oravecz (2024): EMNLP Findings (prompt sensitivity vs. stochastic variation)
- Qwen2.5 speed benchmarks: qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html
- RTX 5090 LLM performance: quantized.fyi/hardware/rtx-5090-32gb-ai-llm-performance-guide-2026-benchmarks/
