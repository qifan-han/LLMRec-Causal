# Modular Two-Stage Decomposition — Full Report

**Date**: 2026-05-16
**Status**: Complete (all steps finished)

---

## 1. Executive Summary

The modular two-stage history-routing experiment is complete. Using the H^J/H^T notation:

- **History-aware retrieval (H^J=1) significantly hurts** overall recommendation quality: BT estimate = -0.828, 95% CI [-1.379, -0.326], P(>0) = 0.0%.
- **History-aware expression (H^T=1) significantly helps**: BT estimate = +0.704, 95% CI [0.388, 1.033], P(>0) = 100.0%.
- **Net total effect is slightly negative but insignificant**: -0.369, 95% CI [-0.898, 0.130], P(>0) = 7.8%.
- **Interaction is small and insignificant**: -0.245, 95% CI [-0.699, 0.201], P(>0) = 14.1%.

A naive Z=0 vs Z=1 comparison would show a near-zero total effect and miss the fact that two large, opposing channels nearly cancel.

---

## 2. Experimental Design

| Parameter | Value |
|-----------|-------|
| Category | Headphones |
| Products in catalog | 25 |
| Consumer personas | 60 |
| Clusters (persona x category) | 60 |
| Cells per cluster | 4 (2x2 factorial) |
| Total supply packages | 240 |
| Supply model | Qwen 2.5 14B (local, Ollama, temperature 0.7) |
| Demand model (absolute) | GPT-5.3-chat-latest (OpenAI API) |
| Demand model (pairwise) | GPT-5.3-chat-latest (OpenAI API) |
| GPT absolute evaluations | 240 |
| GPT pairwise comparisons | 360 (6 pairs per cluster) |
| Master seed | 20260515 |

### Cell encoding

| Cell | H^J (retrieval) | H^T (expression) | Interpretation |
|------|-----------------|-------------------|----------------|
| 0 | Generic | Generic | Baseline: specs only in both stages |
| 1 | Generic | History | Product from specs, text informed by buyer history |
| 10 | History | History retrieves product, text from specs only |
| 11 | History | History | Full history access in both stages |

### Two-stage architecture

1. **Stage 1 (Retrieval)**: LLM selects a product from the catalog. Under generic condition, only product specs shown. Under history condition, review summaries, popularity tiers, and buyer feedback patterns are also provided.
2. **Stage 2 (Expression)**: LLM writes a consumer-facing recommendation for the selected product. Under generic condition, only specs shown. Under history condition, product-specific buyer history is provided plus an experienced-advisor persona.

The product selected in Stage 1 is **locked** before Stage 2 runs. This architectural constraint is what enables clean decomposition.

---

## 3. Supply-Side Results

### Retrieval variation

| Metric | Value |
|--------|-------|
| Retrieval changed (H^J=0 vs H^J=1) | 49/60 clusters (81.7%) |
| Generic top product share | 25.0% |
| History top product share | 28.3% |
| Generic selection entropy | 3.00 |
| History selection entropy | 3.33 |
| Parse failures | 0/240 |
| Leakage flags | 1/240 (0.4%), confirmed false positive from product spec |

History-aware retrieval changes the selected product in 82% of clusters — confirming that the H^J treatment is not inert.

### Expression variation (text mechanisms from GPT absolute eval)

| Cell | Persuasive Intensity (1-7) | Tradeoff Disclosure (1-7) | Regret Risk (1-7) | Trust (1-7) |
|------|---------------------------|--------------------------|-------------------|-------------|
| (0,0) | 4.30 | 3.30 | 4.47 | 4.18 |
| (1,0) | 3.98 | 3.90 | 5.07 | 4.07 |
| (0,1) | 4.37 | 4.27 | 4.50 | 4.77 |
| (1,1) | 4.07 | 4.27 | 5.02 | 4.28 |

Key patterns:
- History expression (H^T=1) increases **tradeoff disclosure** by ~1 point and **trust** by ~0.5 points.
- History retrieval (H^J=1) increases **regret risk** by ~0.5 points — the LLM picks products that the GPT evaluator judges as riskier.

### Absolute evaluation cell means

| Outcome | Cell 0 | Cell 1 | Cell 10 | Cell 11 |
|---------|--------|--------|---------|---------|
| Fit score (1-7) | 4.15 | 4.12 | 3.45 | 3.37 |
| Purchase probability (0-100) | 55.55 | 55.53 | 41.75 | 41.33 |
| Trust score (1-7) | 4.18 | 4.77 | 4.07 | 4.28 |
| Tradeoff disclosure (1-7) | 3.30 | 4.27 | 3.90 | 4.27 |
| Persuasive intensity (1-7) | 4.30 | 4.37 | 3.98 | 4.07 |

---

## 4. Demand-Side Results: Pairwise Comparisons

### Win rates (GPT blinded evaluation, overall winner)

| A vs B | n | A wins | B wins | ties | A win% | B win% |
|--------|---|--------|--------|------|--------|--------|
| (0,0) vs (0,1) | 60 | 13 | 47 | 0 | 21.7% | **78.3%** |
| (0,0) vs (1,0) | 60 | 44 | 14 | 2 | **73.3%** | 23.3% |
| (0,0) vs (1,1) | 60 | 39 | 21 | 0 | **65.0%** | 35.0% |
| (0,1) vs (1,1) | 60 | 43 | 17 | 0 | **71.7%** | 28.3% |
| (1,0) vs (0,1) | 60 | 16 | 44 | 0 | 26.7% | **73.3%** |
| (1,0) vs (1,1) | 60 | 21 | 38 | 1 | 35.0% | **63.3%** |

Summary:
- Cell (0,1) — generic retrieval + history expression — dominates all other cells.
- Cell (1,0) — history retrieval + generic expression — is the worst cell.
- Overall tie rate: 0.8% (very low — GPT evaluator is decisive).

---

## 5. Bradley-Terry Decomposition

### Main decomposition (overall winner)

| Component | Estimate | SE | 95% CI | P(>0) |
|-----------|----------|----|--------|-------|
| Delta_J (retrieval) | -0.828 | 0.273 | [-1.379, -0.326] | 0.0% |
| Delta_T (expression) | +0.704 | 0.162 | [0.388, 1.033] | 100.0% |
| tau^MOD (total) | -0.369 | 0.269 | [-0.898, 0.130] | 7.8% |
| Delta_JT (interaction) | -0.245 | 0.228 | [-0.699, 0.201] | 14.1% |

Bootstrap: B=1000, cluster-level, seed=42.

### By outcome dimension

| Outcome | Retrieval | Expression | Total | Interaction |
|---------|-----------|------------|-------|-------------|
| **Overall** | -0.848 [0.0%] | +0.699 [100%] | -0.387 [8.0%] | -0.237 [14.8%] |
| **Purchase** | **-1.663** [0.0%] | -0.096 [32%] | -1.333 [0.0%] | +0.426 [95.4%] |
| **Satisfaction** | -0.913 [0.0%] | +0.649 [100%] | -0.228 [19.4%] | +0.035 [58.2%] |
| **Trust** | -0.578 [1.0%] | **+1.117** [100%] | -0.177 [25.6%] | **-0.717** [0.4%] |

Brackets show P(>0).

### Interpretation

1. **Retrieval always hurts.** Across all outcome dimensions, history-aware product selection reduces recommendation quality. The effect is largest for purchase (-1.663) and smallest for trust (-0.578). The LLM, when given buyer history, shifts away from products that best match the consumer's specs toward products that are popular or well-reviewed in general.

2. **Expression always helps (except purchase).** History-aware text generation improves overall (+0.699), satisfaction (+0.649), and especially trust (+1.117). For purchase specifically, expression is negligible (-0.096) — suggesting that text style alone doesn't drive purchase intent without a well-matched product.

3. **Trust has a strong negative interaction (-0.717).** When BOTH stages receive history, the trust gain from expression is partially offset. This may reflect evaluator skepticism when the text sounds experienced but the product fit is poor.

4. **Purchase has a positive interaction (+0.426, P>0=95.4%).** This is interesting — the combined history treatment performs better for purchase than the sum of parts would predict, possibly because history-aware text can partially compensate for a mismatched product by emphasizing its real-world track record.

---

## 6. Implications for the Paper

The modular decomposition delivers the paper's core contribution: the history shock has **opposite** effects through the retrieval and expression channels. This is invisible in a standard A/B test, which would show a near-zero total effect and conclude (incorrectly) that history access doesn't matter.

The decomposition reveals that history access matters a great deal — it just matters in opposite directions depending on where it enters the recommender architecture. This has direct managerial implications: a firm should route historical information to the text-generation stage (for trust and satisfaction gains) while being cautious about routing it to product selection (where it can degrade fit).

---

## 7. Mechanism Diagnostics (Step 15)

A companion diagnostic analysis (`15_diagnostics.py`) examines the mechanisms behind the retrieval and expression effects. Key findings:

### Why retrieval is negative: premium-aspiration bias

History-aware retrieval changes the selected product in 49/60 clusters (82%). Among changed clusters:
- Mean price increase: +$25 (history products are more expensive)
- History products are **less popular** (mean rank worsens by 3.8) and have **fewer reviews** (log-count diff: −0.18)
- Budget violation switch rate: 27% (generic product within budget, history product exceeds budget)
- Absolute fit score drops −0.84 (1-7 scale); purchase probability drops −17.2 pp

This is **not** popularity bias — the LLM does not shift toward bestsellers. Instead, it gravitates toward more expensive, niche products when exposed to segment satisfaction feedback (premium-aspiration bias).

GPT pairwise reason codes for 00 vs 10 (generic wins 73%): feature match (53%), tradeoff disclosure (40%), budget fit (37%).

### Why expression helps: trust and disclosure

History-aware expression increases trust (+0.58, CI excludes zero) and tradeoff disclosure (+0.97, CI excludes zero) consistently across retrieval conditions. These are the largest and most robust expression effects in absolute evaluation.

### Purchase interaction: BT vs. absolute discrepancy

The BT pairwise purchase interaction is positive (+0.426, P>0=95.4%), but the absolute evaluation purchase interaction is near zero (−0.4 pp, CI [−3.1, +2.2]). Only 11/60 clusters show the compensatory pattern. The paper should report the BT finding honestly but not build a mechanism story around it.

### Trust interaction: diminishing returns

Trust has a significant negative interaction (−0.37, CI excludes zero): the trust gain from history expression is smaller when retrieval is also history-aware. This is consistent with a ceiling/credibility effect — trust-building text is less effective when the product itself is a poor fit.

Full details in `data/final_history_shock/reports/mechanism_diagnostics_report.md`.

---

## 8. Files and Locations

| File | Description |
|------|-------------|
| `data/final_history_shock/local_supply/final_supply_rows.csv` | 240 supply rows (60 clusters x 4 cells) |
| `data/final_history_shock/gpt_eval/absolute_eval_rows.csv` | 240 GPT absolute evaluations |
| `data/final_history_shock/gpt_eval/pairwise_eval_rows.csv` | 360 GPT pairwise evaluations |
| `data/final_history_shock/analysis/table1_design.csv` | Design summary |
| `data/final_history_shock/analysis/table2_retrieval_variation.csv` | Retrieval variation stats |
| `data/final_history_shock/analysis/table3_pairwise_win_rates.csv` | Pairwise win rates |
| `data/final_history_shock/analysis/table4_bt_decomposition.csv` | BT decomposition (main) |
| `data/final_history_shock/analysis/table5_outcome_channels.csv` | BT decomposition by outcome |
| `data/final_history_shock/analysis/table6_text_mechanisms.csv` | Text mechanism audit |
| `data/final_history_shock/analysis/figure1_decomposition.png` | Decomposition bar chart |
| `data/final_history_shock/analysis/figure2_purchase_vs_satisfaction.png` | Purchase vs satisfaction scatter |
| `data/final_history_shock/reports/final_simulation_report.md` | Auto-generated summary (Step 13) |
| `src/final_history_shock/08_run_local_supply_full.py` | Supply generation script |
| `src/final_history_shock/10_gpt_absolute_eval.py` | Absolute evaluation script |
| `src/final_history_shock/11_gpt_pairwise_eval.py` | Pairwise evaluation script |
| `src/final_history_shock/12_analyze_decomposition.py` | Analysis script |
| `src/final_history_shock/13_write_summary_report.py` | Report generation script |
| `src/final_history_shock/15_diagnostics.py` | Mechanism diagnostics script |
| `data/final_history_shock/analysis/diagnostics/cluster_level_diagnostics.csv` | 60-row diagnostic dataset |
| `data/final_history_shock/analysis/diagnostics/table_retrieval_switch_anatomy.csv` | Product-switch attribute analysis |
| `data/final_history_shock/analysis/diagnostics/table_absolute_did_by_outcome.csv` | Absolute DID by outcome (B=2000) |
| `data/final_history_shock/analysis/diagnostics/table_purchase_interaction_drivers.csv` | Purchase interaction cluster analysis |
| `data/final_history_shock/analysis/diagnostics/table_pairwise_reason_codes.csv` | GPT reason keyword coding |
| `data/final_history_shock/reports/mechanism_diagnostics_report.md` | Full mechanism narrative + draft paragraphs |
| `data/final_history_shock/reports/purchase_interaction_examples.md` | Qualitative examples (6 positive + 2 counter) |
| `data/final_history_shock/reports/pairwise_reason_summary.md` | Reason code distributions |
| `paper/figures/fig_main_decomposition.pdf` | BT decomposition bar chart |
| `paper/figures/fig_cell_means_outcomes.pdf` | 2×2 cell means, 5 outcomes |
| `paper/figures/fig_pairwise_win_matrix.pdf` | 4×4 win-rate heatmap |
| `paper/figures/fig_retrieval_switch_anatomy.pdf` | Product attribute diffs |
| `paper/figures/fig_retrieval_harm_scatter.pdf` | Fit loss vs. product attributes |
| `paper/figures/fig_purchase_interaction.pdf` | Expression lift by retrieval condition |
