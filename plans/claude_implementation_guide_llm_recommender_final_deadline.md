# Claude Implementation Guide: Final LLM Recommender History-Shock Simulation

**Project owner:** Qifan Han  
**Deadline context:** ~36 hours until deadline; target ~20 hours of simulation work.  
**Primary objective:** Produce a final simulation that is credible enough to support the paper, with no further major redesign.

---

## 0. Read This First: Ask Only If Blocked

Claude: before implementing, ask Qifan **only** if one of the following is genuinely unclear or unavailable:

1. You do not know which local LLM command/model to call, e.g. Ollama model name.
2. You cannot access the local repository or existing simulation scripts.
3. You cannot run Python scripts or install required packages.
4. You cannot access product information through web search or a user-provided product list.
5. The OpenAI API test fails after the key is set correctly.

Otherwise, proceed directly. Do **not** spend time debating the conceptual setup. The conceptual setup is fixed below.

---

## 1. Critical Security / API Key Instruction

Do **not** hard-code the API key inside any `.py`, `.md`, `.json`, `.csv`, or Git-tracked file.

Qifan will set the key locally using an environment variable. Use this pattern:

```bash
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY_HERE
```

Recommended: put it in a local untracked `.env` file:

```bash
OPENAI_API_KEY=PASTE_THE_ROTATED_API_KEY_HERE
OPENAI_MODEL=gpt-5.3-chat-latest
OPENAI_EVAL_MODEL=gpt-5.3-chat-latest
```

Then make sure `.env` is ignored:

```bash
echo ".env" >> .gitignore
```

Because the key has already been exposed in chat, Qifan should rotate/revoke it after the final run.

---

## 2. High-Level Plan

We are studying a realistic policy shock:

> A firm gives an LLM recommender access to historical purchase information.

A standard A/B test would compare the full history-informed recommender against the generic recommender. Our paper instead decomposes the effect into:

1. **Retrieval channel:** the product selected.
2. **Expression channel:** how the selected product is recommended.
3. **Interaction:** whether history-informed retrieval and history-informed expression reinforce each other.

The final simulation should use:

- **Local LLM** as the main supply-side recommender.
- **GPT API** for prompt-calibration exemplars and final demand-side evaluation.
- **Known historical DGP** only to create historical purchase summaries shown to the shocked/history-informed local recommender.
- **Blinded GPT evaluations** as synthetic outcomes.

Important: this project does **not** claim real consumer purchase effects. It estimates effects on synthetic LLM-judged demand outcomes.

---

## 3. What We Are Fixing From the Previous Pilot

Previous problem:

> The history-informed / shocked local LLM frequently cited exact historical conversion and satisfaction rates.

This is unrealistic. Real consumer-facing LLM recommenders usually would not disclose internal conversion-rate tables.

The new simulation must enforce:

- No exact historical rates in recommendations.
- No percentages.
- No raw satisfaction rates.
- No sample sizes.
- No “conversion rate = ...” style language.
- Qualitative use of history is allowed.

Allowed phrases:

- “popular among similar buyers”
- “often chosen by budget-conscious customers”
- “historically reliable”
- “a frequent choice for commuters”
- “mixed feedback among heavy users”
- “strong track record for everyday use”

Forbidden phrases:

- “42% conversion rate”
- “satisfaction rate of 0.73”
- “historical data shows 51.2%”
- “n = 1,204”
- “ranked #1 by conversion”
- “CTR”
- “CVR”

---

## 4. Exact GPT Call Budget

Use GPT for three roles only:

1. Generate recommendation exemplars for local-prompt calibration.
2. Generate/repair consumer personas.
3. Evaluate final recommendation packages.

### 4.1 Exact GPT recommendation exemplar calls

Run exactly:

> **180 GPT recommendation exemplar calls**

Construction:

- 6 categories
- 10 personas per category
- 3 recommendation regimes per persona

`6 × 10 × 3 = 180`

The 6 exemplar categories should include, but not be limited to, the final simulation categories:

1. headphones
2. phone chargers
3. wireless routers
4. coffee makers
5. office chairs
6. running shoes

The 3 regimes:

1. **Generic recommender:** catalog + persona, no history.
2. **Consumer-centric recommender:** catalog + persona, explicitly balanced and transparent.
3. **History-aware recommender:** catalog + persona + qualitative historical summary, with anti-leakage rule.

Save every prompt-answer pair in:

```text
data/gpt_exemplars/gpt_recommendation_exemplars.jsonl
```

Each JSONL row should include:

```json
{
  "example_id": "...",
  "category": "...",
  "regime": "generic|consumer_centric|history_aware",
  "prompt": "...",
  "response_text": "...",
  "parsed_product_id": "...",
  "model": "...",
  "usage": {...},
  "timestamp": "..."
}
```

These 180 GPT responses are **not** for formal fine-tuning unless a reliable local fine-tuning pipeline already exists. By default, use them for:

- prompt calibration,
- few-shot examples,
- style rules,
- anti-leakage examples,
- local recommender debugging.

---

## 5. Final Simulation Size

Use two final categories:

1. headphones
2. phone chargers

Use:

> **60 consumer personas per category**

Total final clusters:

```text
2 categories × 60 personas = 120 clusters
```

Each cluster has four audit cells:

| Cell | Retrieval | Expression |
|---|---|---|
| 00 | generic retrieval | generic expression |
| 10 | history-informed retrieval | generic expression |
| 01 | generic retrieval | history-informed expression |
| 11 | history-informed retrieval | history-informed expression |

Supply-side local LLM outputs:

```text
120 clusters × 4 cells = 480 recommendation packages
```

Pairwise GPT demand judgments:

```text
120 clusters × 6 pairwise comparisons = 720 pairwise GPT calls
```

Absolute GPT score calls:

```text
480 recommendation packages × 1 absolute scoring call = 480 GPT calls
```

Persona-generation GPT calls:

```text
12 calls, each generating 10 personas = 120 personas
```

Setup/smoke API calls:

```text
5 simple API test calls
```

Expected base GPT call count:

```text
180 exemplar calls
+ 12 persona calls
+ 480 absolute evaluation calls
+ 720 pairwise evaluation calls
+ 5 API test calls
= 1,397 GPT calls
```

Allow at most 20% retry overhead:

```text
maximum planned GPT calls = 1,700
```

If time or budget becomes binding, keep the 720 pairwise calls and drop/reduce the 480 absolute scoring calls.

---

## 6. Directory Structure

Create a clean subfolder:

```text
src/final_history_shock/
```

Recommended structure:

```text
src/final_history_shock/
  00_api_test.py
  01_build_or_collect_catalogs.py
  02_generate_gpt_exemplars.py
  03_generate_personas.py
  04_validate_personas.py
  05_generate_historical_dgp.py
  06_build_local_prompts.py
  07_smoke_run_local_supply.py
  08_run_local_supply_full.py
  09_leakage_audit_and_regen.py
  10_gpt_absolute_eval.py
  11_gpt_pairwise_eval.py
  12_analyze_decomposition.py
  13_write_summary_report.py
  prompts.py
  utils_openai.py
  utils_local_llm.py
  utils_parse.py
  utils_stats.py
```

Data folders:

```text
data/final_history_shock/
  catalogs/
  personas/
  history_dgp/
  gpt_exemplars/
  local_supply/
  gpt_eval/
  analysis/
  reports/
```

---

## 7. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install openai pandas numpy scipy statsmodels tqdm tenacity python-dotenv scikit-learn
```

Optional if using embeddings:

```bash
pip install sentence-transformers
```

Use `python-dotenv` to load `.env` locally. Never print the key.

---

## 8. API Test Script

Create `00_api_test.py`.

Goal: verify the API works before any expensive run.

Run 5 simple calls:

1. Return exact JSON: `{"ok": true}`
2. Summarize one sentence.
3. Return a product recommendation JSON for a toy catalog.
4. Return a pairwise A/B judgment JSON.
5. Return token usage.

Required output file:

```text
data/final_history_shock/api_test_results.json
```

Example code skeleton:

```python
from openai import OpenAI
from dotenv import load_dotenv
import os, json, time

load_dotenv()
client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.3-chat-latest")

TESTS = [
    "Return exactly this JSON: {\"ok\": true}",
    "In one sentence, explain why headphones differ from earbuds.",
    "Given products A and B, recommend one for a student on a budget. Return JSON with product_id and reason.",
    "Compare recommendation A and B. Return JSON with winner equal to A, B, or tie.",
    "Return JSON: {\"test\": \"usage\"}."
]

rows = []
for i, prompt in enumerate(TESTS):
    resp = client.responses.create(model=MODEL, input=prompt)
    rows.append({
        "i": i,
        "prompt": prompt,
        "output_text": resp.output_text,
        "usage": resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else str(resp.usage),
        "model": MODEL,
        "timestamp": time.time()
    })

os.makedirs("data/final_history_shock", exist_ok=True)
with open("data/final_history_shock/api_test_results.json", "w") as f:
    json.dump(rows, f, indent=2)

print("API test complete")
```

Stop if this fails.

---

## 9. Product Catalog Collection

### 9.1 Final categories

Use exactly two final categories:

1. headphones
2. phone chargers

For each final category, collect or construct:

> **25 products per category**

Total:

```text
50 final products
```

Each product row must include:

```json
{
  "category": "headphones",
  "product_id": "headphones_001",
  "brand": "...",
  "model_name": "...",
  "price": 99.99,
  "price_tier": "budget|midrange|premium",
  "key_features": ["...", "..."],
  "best_for": ["...", "..."],
  "drawbacks": ["...", "..."],
  "review_summary": "...",
  "quality_score": 0.0,
  "brand_familiarity": 0.0,
  "margin_proxy": 0.0,
  "inventory_pressure": 0.0
}
```

`quality_score`, `brand_familiarity`, `margin_proxy`, and `inventory_pressure` can be normalized to `[0,1]`. They are simulation attributes, not necessarily real public facts.

### 9.2 Real-life product list requirement

Preferred method:

- Use real product names and features from public retail/manufacturer pages.
- Record source URLs in a separate metadata file if web access is available.

If web access is unavailable:

- Use realistic synthetic product names/features.
- Clearly label the catalog as `realistic_synthetic_catalog`.
- Do not claim these are scraped real products.

Output files:

```text
data/final_history_shock/catalogs/headphones_catalog.csv
data/final_history_shock/catalogs/phone_chargers_catalog.csv
data/final_history_shock/catalogs/catalog_sources.json
```

### 9.3 Variation checks for catalog

Before simulation, compute:

- price range,
- number of brands,
- number of price tiers,
- feature overlap,
- no single brand exceeds 30–40% of products,
- at least 8–10 products per category are genuinely plausible choices.

If catalog is too homogeneous, fix before proceeding.

---

## 10. GPT Recommendation Exemplars

Create `02_generate_gpt_exemplars.py`.

Use 6 categories:

1. headphones
2. phone chargers
3. wireless routers
4. coffee makers
5. office chairs
6. running shoes

For each category:

- create or collect a small catalog of 8–12 products,
- create 10 personas,
- run 3 regimes per persona.

Total calls:

```text
6 × 10 × 3 = 180
```

### 10.1 Regime A: generic recommender

Prompt objective:

> Recommend the single best product for the consumer from the catalog. Be realistic, concise, and helpful.

### 10.2 Regime B: consumer-centric recommender

Prompt objective:

> Prioritize fit, budget, tradeoffs, and post-purchase satisfaction. Be balanced and transparent.

### 10.3 Regime C: history-aware recommender

Prompt objective:

> Use qualitative historical purchase summaries as internal background. Do not cite exact numerical historical data.

### 10.4 Exemplar output format

Force JSON:

```json
{
  "selected_product_id": "...",
  "recommendation_text": "...",
  "why_it_fits": "...",
  "tradeoff_note": "...",
  "history_used_qualitatively": true,
  "forbidden_numeric_history_leakage": false
}
```

After collection, automatically select 3–6 high-quality examples for few-shot local prompts:

- 2 generic examples,
- 2 history-aware examples,
- 1–2 examples showing tradeoff disclosure.

Save selected few-shot examples in:

```text
data/final_history_shock/gpt_exemplars/selected_few_shot_examples.json
```

---

## 11. Local Model “Training” / Prompt Calibration

Important: default method is **prompt calibration**, not true fine-tuning.

Do not waste time trying to fine-tune unless a working fine-tuning script already exists and can finish quickly.

Use GPT exemplars to create:

1. `GENERIC_RETRIEVAL_PROMPT`
2. `HISTORY_RETRIEVAL_PROMPT`
3. `GENERIC_EXPRESSION_PROMPT`
4. `HISTORY_EXPRESSION_PROMPT`

Each prompt should include:

- task definition,
- realistic recommendation style rules,
- anti-leakage rules,
- output JSON schema,
- 1–3 few-shot examples.

### 11.1 Generic retrieval prompt

Input:

- persona,
- catalog.

Output:

```json
{
  "selected_product_id": "...",
  "retrieval_rationale_internal": "..."
}
```

Do not include final recommendation text at retrieval stage.

### 11.2 History retrieval prompt

Input:

- persona,
- catalog,
- qualitative historical summary.

Output:

```json
{
  "selected_product_id": "...",
  "retrieval_rationale_internal": "...",
  "history_signal_used": "none|weak|moderate|strong"
}
```

Anti-leakage rule still applies.

### 11.3 Generic expression prompt

Input:

- persona,
- locked selected product,
- catalog attributes.

Output:

```json
{
  "recommendation_text": "...",
  "tradeoff_text": "...",
  "persuasion_text": "..."
}
```

### 11.4 History expression prompt

Input:

- persona,
- locked selected product,
- catalog attributes,
- qualitative historical summary.

Output:

```json
{
  "recommendation_text": "...",
  "tradeoff_text": "...",
  "persuasion_text": "...",
  "history_language_used": "none|weak|moderate|strong"
}
```

Rules:

- May say: “popular among similar customers.”
- May say: “historically reliable for this use case.”
- Must not cite exact historical data.
- Must not claim the product is best only because of history.
- Must remain consumer-facing and realistic.

---

## 12. Historical Purchase DGP

Create `05_generate_historical_dgp.py`.

Purpose:

- Generate historical purchase and satisfaction records from a known DGP.
- Aggregate them into qualitative history summaries shown to the history-informed recommender.
- Keep the raw DGP hidden from final GPT demand judges.

### 12.1 Historical consumer segments

Create segments such as:

For headphones:

- budget student,
- commuter,
- remote worker,
- audiophile,
- gym user,
- gamer,
- frequent traveler,
- casual listener.

For phone chargers:

- budget student,
- frequent traveler,
- iPhone user,
- Android fast-charge user,
- multi-device household,
- office worker,
- parent buying for family,
- safety-conscious buyer.

### 12.2 DGP structure

For historical user `i` and product `p`:

```text
latent_purchase_utility_ip =
    segment_product_match_ip
  + quality_weight_i * quality_p
  - price_sensitivity_i * normalized_price_p
  + brand_weight_i * brand_familiarity_p
  + convenience_weight_i * convenience_features_p
  + history_noise_ip
```

Purchase probability:

```text
Pr(Y_hist_purchase = 1) = sigmoid(latent_purchase_utility_ip)
```

Satisfaction probability conditional on purchase:

```text
Pr(Y_hist_satisfaction = 1 | purchase) = sigmoid(
    product_fit_ip
  + quality_p
  - drawback_penalty_ip
  - price_regret_ip
  + noise
)
```

Simulate:

```text
20,000 historical sessions per category
```

Output raw DGP files:

```text
data/final_history_shock/history_dgp/headphones_history_raw.csv
data/final_history_shock/history_dgp/phone_chargers_history_raw.csv
```

Output product-segment aggregates:

```text
data/final_history_shock/history_dgp/headphones_history_aggregates.csv
data/final_history_shock/history_dgp/phone_chargers_history_aggregates.csv
```

### 12.3 Convert numerical history to qualitative summaries

Do not pass raw rates into the local model prompt.

Convert quantiles into language:

| Quantile | Popularity language |
|---|---|
| top 20% | “frequently chosen by similar buyers” |
| 60–80% | “often considered by similar buyers” |
| 40–60% | “moderately common among similar buyers” |
| bottom 40% | “less commonly selected by similar buyers” |

Satisfaction language:

| Quantile | Satisfaction language |
|---|---|
| top 20% | “strong post-purchase feedback” |
| 60–80% | “generally positive post-purchase feedback” |
| 40–60% | “mixed post-purchase feedback” |
| bottom 40% | “weaker post-purchase feedback for this segment” |

Create prompt-facing summary files:

```text
data/final_history_shock/history_dgp/headphones_history_qualitative.json
data/final_history_shock/history_dgp/phone_chargers_history_qualitative.json
```

These qualitative summaries are what the shocked model sees.

---

## 13. Consumer Persona Generation and Validation

Create `03_generate_personas.py`.

Use GPT to generate exactly:

```text
60 personas for headphones
60 personas for phone chargers
```

Use batching:

```text
12 GPT calls total, 10 personas per call
```

Each persona must be a realistic consumer who might buy the product. They do not need to be technically knowledgeable.

Persona schema:

```json
{
  "persona_id": "headphones_001",
  "category": "headphones",
  "age_range": "...",
  "purchase_context": "...",
  "budget": "...",
  "technical_knowledge": "low|medium|high",
  "primary_use_case": "...",
  "secondary_use_case": "...",
  "brand_preference": "...",
  "price_sensitivity": "low|medium|high",
  "quality_sensitivity": "low|medium|high",
  "risk_aversion": "low|medium|high",
  "must_have_features": ["..."],
  "features_to_avoid": ["..."],
  "prior_experience": "...",
  "one_paragraph_description": "..."
}
```

### 13.1 Persona quality rules

Personas should vary along:

- budget,
- technical knowledge,
- urgency,
- risk tolerance,
- brand familiarity,
- quality sensitivity,
- intended use,
- constraints,
- prior bad/good experiences.

Do not make all personas tech-savvy. Real consumers include:

- parents buying for children,
- students on budget,
- office workers,
- travelers,
- people replacing a broken device,
- gift buyers,
- people who only know a few brands,
- people who care about convenience more than specs.

### 13.2 Persona validation

Create `04_validate_personas.py`.

Check:

1. At least 4 price-sensitivity groups/categories represented.
2. At least 3 technical-knowledge levels represented.
3. No duplicate or near-duplicate personas.
4. No impossible persona/category mismatch.
5. Persona has enough information to make product choice nontrivial.

If validation fails, repair using GPT or rule-based edits.

Output:

```text
data/final_history_shock/personas/final_personas.csv
```

---

## 14. Local Supply-Side Audit

Create `08_run_local_supply_full.py`.

For each cluster `(category, persona)`:

1. Run generic retrieval.
2. Run history-informed retrieval.
3. Run generic expression for the generic-selected product.
4. Run history-informed expression for the generic-selected product.
5. Run generic expression for the history-selected product.
6. Run history-informed expression for the history-selected product.

Construct four cells:

| Cell | Selected product | Text |
|---|---|---|
| 00 | generic retrieval product | generic expression |
| 10 | history retrieval product | generic expression |
| 01 | generic retrieval product | history expression |
| 11 | history retrieval product | history expression |

Important cell invariant:

- 00 and 01 must have the **same selected product**.
- 10 and 11 must have the **same selected product**.
- 00 and 10 differ only by retrieval condition.
- 00 and 01 differ only by expression condition.
- 11 is the full history condition.

Output dataframe:

```text
data/final_history_shock/local_supply/final_supply_rows.csv
```

Each row:

```json
{
  "cluster_id": "...",
  "category": "...",
  "persona_id": "...",
  "persona_json": {...},
  "cell": "00|10|01|11",
  "retrieval_condition": "generic|history",
  "expression_condition": "generic|history",
  "selected_product_id": "...",
  "selected_product_json": {...},
  "recommendation_text": "...",
  "tradeoff_text": "...",
  "persuasion_text": "...",
  "full_recommendation_package": "...",
  "history_language_used": "none|weak|moderate|strong",
  "local_model": "...",
  "raw_retrieval_output": "...",
  "raw_expression_output": "..."
}
```

---

## 15. Leakage Audit and Regeneration

Create `09_leakage_audit_and_regen.py`.

Flag recommendation packages if they include:

```python
FORBIDDEN_PATTERNS = [
    r"\d+\s*%",
    r"conversion rate",
    r"satisfaction rate",
    r"CTR",
    r"CVR",
    r"click-through",
    r"sample size",
    r"n\s*=",
    r"historical data shows",
    r"\b0\.\d+\b",
    r"\b1\.0\b",
    r"ranked #?1 by conversion",
]
```

For flagged rows:

1. Regenerate once with stricter anti-leakage instruction.
2. Re-audit.
3. If still leaking, keep the row but mark:

```json
{"leakage_flag": true, "excluded_from_main": true}
```

Target:

```text
Leakage rate after regeneration should be below 2%.
```

If leakage is above 5%, stop and revise prompts.

---

## 16. Smoke Run Before Full Run

Create `07_smoke_run_local_supply.py`.

Smoke design:

```text
2 categories × 3 personas = 6 clusters
6 clusters × 4 cells = 24 recommendation packages
6 clusters × 6 pairwise comparisons = 36 pairwise comparisons
24 absolute scoring calls
```

Smoke success criteria:

1. API test passed.
2. Local model returns parseable JSON at least 95% of the time after retries.
3. Cell invariant passes for all 6 clusters.
4. History changes retrieval in at least 1 or 2 of 6 clusters.
5. Leakage rate after regeneration is 0 or close to 0.
6. GPT absolute evaluation returns valid JSON for all 24 packages.
7. GPT pairwise evaluation returns valid JSON for all 36 pairs.
8. Pairwise tie rate below 70% in the smoke run.

If smoke fails, fix before full run.

---

## 17. GPT Absolute Evaluation

Create `10_gpt_absolute_eval.py`.

Evaluate each recommendation package separately.

Number of calls:

```text
480 calls
```

Input to GPT:

- category,
- persona,
- selected product attributes,
- full recommendation package.

Do **not** reveal:

- cell label,
- retrieval condition,
- expression condition,
- DGP parameters,
- historical numerical data,
- whether the text was generated by local or GPT.

Output JSON schema:

```json
{
  "fit_score_1_7": 0,
  "purchase_probability_0_100": 0,
  "expected_satisfaction_0_100": 0,
  "trust_score_1_7": 0,
  "clarity_score_1_7": 0,
  "persuasive_intensity_1_7": 0,
  "tradeoff_disclosure_1_7": 0,
  "regret_risk_1_7": 0,
  "brief_reason": "..."
}
```

Important: absolute scores are **secondary** because they may be compressed. The main outcome is pairwise evaluation.

Output:

```text
data/final_history_shock/gpt_eval/absolute_eval_rows.csv
```

---

## 18. GPT Pairwise Demand Evaluation: Main Y

Create `11_gpt_pairwise_eval.py`.

For each cluster, compare all 6 pairs among cells:

```text
00 vs 10
00 vs 01
00 vs 11
10 vs 01
10 vs 11
01 vs 11
```

Randomize A/B order.

Number of calls:

```text
120 clusters × 6 pairs = 720 calls
```

Input to GPT:

- same persona,
- same category,
- package A,
- package B.

Do **not** reveal:

- cell labels,
- which package used history,
- DGP details,
- local model metadata.

Output JSON schema:

```json
{
  "overall_winner": "A|B|tie",
  "purchase_winner": "A|B|tie",
  "satisfaction_winner": "A|B|tie",
  "trust_winner": "A|B|tie",
  "confidence_1_5": 0,
  "reason": "..."
}
```

Convert A/B winner back to cell winner after the call.

Output:

```text
data/final_history_shock/gpt_eval/pairwise_eval_rows.csv
```

Main outcome variables:

1. `overall_winner`
2. `purchase_winner`
3. `satisfaction_winner`
4. `trust_winner`

Primary paper outcome:

```text
synthetic pairwise consumer preference, based on overall_winner
```

Secondary outcomes:

- purchase preference,
- satisfaction preference,
- trust preference.

---

## 19. Parsing, Retries, and Reliability

All GPT and local LLM outputs must be parsed as JSON.

Use:

- explicit JSON prompt,
- retry up to 3 times for parse failures,
- deterministic-ish temperature if available,
- log raw output always.

Recommended temperature:

```text
GPT exemplars: 0.7
GPT evaluations: 0.2
Local retrieval: 0.3–0.6
Local expression: 0.5–0.7
```

For GPT evaluation, prefer stable outputs over creative outputs.

Track:

- parse failure rate,
- retry count,
- invalid value rate,
- missing field rate,
- leakage rate,
- tie rate.

If pairwise tie rate exceeds 70%, prompt is too conservative or packages are too similar.

---

## 20. Analysis

Create `12_analyze_decomposition.py`.

### 20.1 Supply-side checks

Report:

1. Retrieval changed product share:

```text
Pr(product_10 != product_00)
```

2. Product concentration by category:

- top product share,
- entropy,
- HHI.

3. Leakage rate.
4. Word counts.
5. Cell invariant pass/fail.

### 20.2 Pairwise win-rate tables

For each outcome `overall`, `purchase`, `satisfaction`, `trust`, compute win/tie/loss for:

- 10 vs 00: retrieval-only effect
- 01 vs 00: expression-only effect
- 11 vs 00: total effect
- 11 vs 10: expression effect holding history retrieval fixed
- 11 vs 01: retrieval effect holding history expression fixed
- 10 vs 01: retrieval-only vs expression-only

### 20.3 Bradley–Terry utility decomposition

Estimate latent utilities for cells 00, 10, 01, 11.

Normalize:

```text
U_00 = 0
```

Then compute:

```text
Retrieval effect = U_10 - U_00
Expression effect = U_01 - U_00
Total effect = U_11 - U_00
Interaction = U_11 - U_10 - U_01 + U_00
```

Cluster bootstrap by `cluster_id`.

Recommended bootstrap:

```text
B = 1000 if time permits; B = 300 minimum
```

Report:

- estimate,
- standard error,
- 95% CI,
- probability estimate > 0 from bootstrap draws.

### 20.4 Absolute score analysis

For absolute GPT scores, report cell means and cluster-robust differences for:

- fit score,
- purchase probability,
- expected satisfaction,
- trust,
- persuasive intensity,
- tradeoff disclosure,
- regret risk.

But do not make absolute scores the main causal evidence.

### 20.5 Text mechanism audit

Use two validated text mechanisms as main:

1. persuasive intensity,
2. tradeoff disclosure.

Fit score can be reported as descriptive only, unless external validation is performed.

Main text mechanism claims should be about:

- Does history-informed expression increase persuasive intensity?
- Does it reduce tradeoff disclosure?
- Does the demand-side lift align more with expression than retrieval?

---

## 21. Required Tables and Figures

Create paper-ready CSV/Markdown/LaTeX tables.

### Table 1: Simulation design

Columns:

- categories,
- products per category,
- personas per category,
- clusters,
- local supply packages,
- GPT pairwise judgments,
- GPT absolute evaluations.

### Table 2: Supply-side retrieval variation

Rows by category and overall:

- history changes selected product share,
- top product share under generic retrieval,
- top product share under history retrieval,
- entropy,
- HHI.

### Table 3: Pairwise win/tie/loss decomposition

Rows:

- 10 vs 00,
- 01 vs 00,
- 11 vs 00,
- 11 vs 10,
- 11 vs 01,
- 10 vs 01.

Columns:

- A wins,
- B wins,
- ties,
- net win rate.

### Table 4: Bradley–Terry decomposition

Rows:

- retrieval,
- expression,
- total,
- interaction.

Columns:

- estimate,
- SE,
- 95% CI,
- P(effect > 0).

### Table 5: Purchase vs satisfaction vs trust

Same decomposition for:

- overall,
- purchase,
- satisfaction,
- trust.

### Table 6: Text mechanism audit

Rows:

- generic expression,
- history expression.

Columns:

- persuasive intensity,
- tradeoff disclosure,
- regret risk,
- trust.

### Figure 1: Decomposition bar chart

Bar chart:

- retrieval contribution,
- expression contribution,
- interaction,
- total.

### Figure 2: Purchase vs satisfaction decomposition

Show whether history-informed expression increases purchase more than satisfaction.

---

## 22. Summary Report

Create `13_write_summary_report.py`.

Output:

```text
data/final_history_shock/reports/final_simulation_report.md
```

Report structure:

1. Executive summary.
2. What changed relative to previous pilot.
3. Design and sample size.
4. Prompt-calibration procedure using GPT exemplars.
5. Historical DGP and qualitative history construction.
6. Supply-side results.
7. Demand-side GPT evaluation results.
8. Decomposition results.
9. Text mechanism results.
10. Robustness and failure checks.
11. What can and cannot be claimed.
12. Paper-ready interpretation.

### 22.1 Exact wording for cautious interpretation

Use this kind of language:

> The outcome is a blinded GPT-based synthetic pairwise preference judgment, not observed market demand.

> The simulation is designed to test whether a modular audit can reveal the channels through which historical purchase information changes LLM recommendation packages.

> The decomposition separates product-selection changes from expression changes while holding the consumer and category fixed.

Do not claim:

- real purchase effects,
- real welfare effects,
- actual market demand,
- human consumer validation,
- that local LLM behavior represents all frontier models.

---

## 23. Decision Rules During Implementation

After smoke run, proceed only if:

1. Cell invariant pass rate = 100%.
2. Local parse success after retry ≥ 95%.
3. Leakage after regeneration ≤ 2%.
4. History changes product selection in at least 25–35% of smoke/final clusters.
5. Pairwise tie rate < 70%.
6. GPT evaluation JSON parse success after retry ≥ 98%.

If retrieval change is too low:

- enrich catalog,
- strengthen persona heterogeneity,
- strengthen qualitative history summaries,
- but do not make the prompt unrealistically business-aggressive.

If leakage is too high:

- strengthen anti-leakage prompt,
- remove numerical history from prompt entirely,
- pass only qualitative summaries.

If pairwise tie rate is too high:

- make recommendation packages fuller and more differentiated,
- ask judge to choose tie only when truly indistinguishable,
- keep tie option but discourage lazy ties.

---

## 24. Recommended Prompt Wording

### 24.1 Anti-leakage history instruction

Use this exact instruction in history-aware prompts:

> You have access to internal historical purchase summaries. Treat them as background evidence only. Do not reveal, quote, or approximate any conversion rates, satisfaction rates, percentages, rankings, sample sizes, scores, or raw historical numbers. In a consumer-facing recommendation, you may only refer qualitatively to historical patterns when useful, using phrases like “popular among similar buyers,” “historically reliable,” “often chosen for this use case,” or “mixed feedback among heavy users.” If historical information conflicts with the consumer’s needs, prioritize the consumer’s needs.

### 24.2 Realistic recommender style instruction

> Write like a modern shopping assistant. Be specific about the consumer’s needs, compare relevant tradeoffs, avoid generic hype, and do not overstate certainty. The recommendation should sound natural and consumer-facing, not like a data report.

### 24.3 Pairwise GPT judge instruction

> You are evaluating two recommendation packages for the same consumer. You do not know how the recommendations were generated. Choose which package is more likely to lead to a good consumer outcome. Consider purchase likelihood, expected post-purchase satisfaction, trust, and whether the recommendation honestly communicates tradeoffs. Use tie only if the two packages are genuinely indistinguishable.

---

## 25. Final Deliverables

At the end, produce:

```text
data/final_history_shock/reports/final_simulation_report.md
```

and the following tables:

```text
data/final_history_shock/analysis/table1_design.csv
data/final_history_shock/analysis/table2_retrieval_variation.csv
data/final_history_shock/analysis/table3_pairwise_win_rates.csv
data/final_history_shock/analysis/table4_bt_decomposition.csv
data/final_history_shock/analysis/table5_outcome_channels.csv
data/final_history_shock/analysis/table6_text_mechanisms.csv
```

and figures:

```text
data/final_history_shock/analysis/figure1_decomposition.png
data/final_history_shock/analysis/figure2_purchase_vs_satisfaction.png
```

and raw data:

```text
data/final_history_shock/local_supply/final_supply_rows.csv
data/final_history_shock/gpt_eval/absolute_eval_rows.csv
data/final_history_shock/gpt_eval/pairwise_eval_rows.csv
```

---

## 26. Final Implementation Priority

If time becomes tight, prioritize in this order:

1. Smoke run correctness.
2. Leakage-free local supply generation.
3. Pairwise GPT demand evaluation.
4. Bradley–Terry decomposition.
5. Text mechanism audit.
6. Absolute GPT scores.
7. Extra robustness.

The final paper can survive without absolute purchase probability scores. It cannot survive without a clean pairwise decomposition and leakage control.

---

## 27. One-Sentence Target Result

The final simulation should be able to support or reject this statement:

> Giving an LLM recommender access to historical purchase information changes the recommendation package; a modular audit decomposes the resulting synthetic preference lift into product-selection and expression channels, revealing whether the apparent improvement comes from better retrieval, more persuasive language, or both.

