# Mechanism Diagnostics Report — Modular Two-Stage History Shock

**Date**: 2026-05-16
**Status**: Complete

---

## 1. Executive Summary

- History-aware retrieval changes the selected product in 49/60 clusters (82%), shifting toward more expensive products (mean price diff: $+25.2) that are less popular (mean rank worsening: -3.8) and have fewer reviews (mean log-count diff: -0.18). This comes at the cost of persona fit: mean fit score drops by 0.84 points (1-7 scale) and purchase probability drops by 17.2 pp (0-100).
- In pairwise evaluation, generic retrieval beats history retrieval in 44/60 clusters (73%); history retrieval wins in only 14/60 (23%).
- The purchase-specific positive interaction in the Bradley-Terry pairwise decomposition (+0.426, P>0=95.4%) does **not** replicate in absolute evaluation (-0.4 pp, 95% CI [-3.1, +2.2]). The absolute evidence is too imprecise to support a compensatory story.
- History-aware expression increases trust (+0.58, 95% CI [+0.35, +0.83]) and tradeoff disclosure (+0.97) consistently across retrieval conditions.
- Budget violations partially explain retrieval harm: history retrieval moves products outside the persona's stated budget in 22% of clusters where generic products were within budget.

---

## 2. Negative Retrieval Effect: Mechanism

### 2.1 What history retrieval changes

When the LLM receives buyer history (popularity tiers, rating counts, segment-level feedback), it shifts product selection as follows. Among retrieval-changed clusters (n=49):

| Attribute | Mean diff (history − generic) | Direction | Count |
|-----------|-------------------------------|-----------|-------|
| Price | $+25.2 | More expensive | 20/49 more expensive |
| Log review count | -0.18 | Lower | 34/49 fewer reviews |
| Average rating | -0.06 | Lower | — |
| Popularity rank | -3.8 | Less popular | 34/49 less popular |
| Segment affinity | +0.04 | Higher | — |
| Feature fit (heuristic) | +0.102 | Better | — |

The pattern is **not** a simple bestseller/popularity bias. History retrieval shifts toward products that are more expensive, less popular, and have fewer reviews. The segment-affinity improvement is negligible (+0.04), suggesting the LLM is not selecting products that are empirically better for the persona's segment.

### 2.2 Why this hurts

The evidence is consistent with a **premium-aspiration bias**: the LLM, when given qualitative buyer feedback describing product satisfaction, gravitates toward higher-priced products — potentially interpreting satisfaction signals as evidence that spending more is worthwhile — even when the premium product is a poor fit for the persona's budget and feature requirements.

1. **Budget violations**: 22% of clusters experience a budget violation switch (generic product within budget, history product exceeds budget). Over-budget rate: generic products 3%, history products 22%.

2. **Fit score deterioration**: Absolute fit score drops by 0.84 points on average (1-7 scale) when retrieval switches to the history-aware product, even holding expression constant. The CI excludes zero.

3. **GPT pairwise reasons**: In the 00 vs 10 comparison, the most frequent reason codes for cell 0 winning are: feature_match (32/60), tradeoff_disclosure (24/60), budget_fit (22/60). Feature match and budget fit are the top codes, consistent with the interpretation that history-retrieved products violate persona-specific constraints.

4. **Regret risk increases**: History-retrieved products are judged as higher regret risk (+0.60 on 1-7 scale, CI excludes zero), suggesting the GPT evaluator recognizes that these products carry higher downside for the persona.

### 2.3 Among clusters where history retrieval loses pairwise

When generic retrieval wins the overall pairwise comparison (n=44):
- Mean price diff: $+11.7
- Mean log review count diff: -0.07
- Mean fit delta: -1.27
- Mean segment affinity diff: +0.03

The retrieval harm is concentrated in clusters where history information causes the largest price increase and fit deterioration.

### 2.4 Interpretation (with discipline)

In this controlled simulation, history-aware retrieval appears to shift product selection toward more expensive, niche products at the expense of persona-specific fit and budget alignment. The LLM interprets segment-level satisfaction signals as reasons to recommend premium alternatives, without adequately weighing the individual persona's stated budget constraints and feature requirements.

This is consistent with the interpretation that the LLM treats qualitative buyer history as a **population-level quality signal** that overrides persona-specific constraints — a form of aspiration bias rather than popularity bias.

We do not claim this pattern generalizes to all LLM architectures, all product categories, or real consumer populations. The pattern is observed in a single product category (headphones) with a single LLM (Qwen 2.5 14B) evaluated by a single judge (GPT-5.3).

---

## 3. Purchase Interaction: BT vs. Absolute Discrepancy

### 3.1 The finding

In the Bradley-Terry pairwise decomposition, the purchase-specific interaction is positive and marginally significant (+0.426, P>0=95.4%). However, in absolute evaluation, the purchase interaction is -0.4 pp (95% CI [-3.1, +2.2]) — essentially zero and imprecise.

### 3.2 Expression lift comparison (absolute scale)

| Condition | Expression lift on purchase | 95% CI |
|-----------|-----------------------------|--------|
| Under generic retrieval | -0.0 pp | [-1.8, +1.9] |
| Under history retrieval | -0.4 pp | [-2.6, +1.8] |
| **Difference (interaction)** | **-0.4 pp** | **[-3.1, +2.2]** |

Expression has near-zero effect on absolute purchase probability under **both** retrieval conditions. This contrasts with the BT pairwise decomposition, where the purchase interaction is the largest positive signal. The discrepancy may reflect that pairwise comparisons are more sensitive to within-cluster relative differences that wash out in absolute scoring.

### 3.3 Compensatory cluster pattern

11/60 clusters (18%) show the compensatory pattern (history expression lifts purchase under history retrieval but not under generic retrieval). This is a minority of clusters — the pattern is not dominant.

### 3.4 Assessment

The positive purchase interaction is:
- **Present in pairwise BT decomposition** (+0.426, P>0=95.4%)
- **Absent in absolute evaluation** (-0.4 pp, CI includes zero)
- **Observed in only 11/60 clusters** fitting the compensatory pattern
- **Not robust** across evaluation methods

We characterize this as a **pairwise-scale artifact** or, at best, a pattern that is too imprecise to support a compensatory mechanism claim. The paper should report the BT purchase interaction honestly but should not build a mechanism story around it. The key finding for purchase is the large negative retrieval main effect (-1.663 BT, and -17.2 pp absolute), not the interaction.

---

## 4. Implications for Paper Framing

### 4.1 What we can claim

1. **The modular decomposition reveals opposing channels that are invisible in an aggregate A/B test.** The near-zero total effect masks a large negative retrieval effect and a large positive expression effect. This is the paper's core empirical contribution from the simulation.

2. **History-aware retrieval shifts product selection toward more expensive, less popular products at the expense of individual persona fit and budget compliance.** The pattern is consistent with a premium-aspiration bias — the LLM interprets satisfaction signals as reasons to up-sell — and is supported by product-level attribute comparisons, absolute evaluation scores, and pairwise evaluator reasons citing feature match and budget fit.

3. **History-aware expression improves trust and tradeoff disclosure consistently.** The trust gain (+0.58) and tradeoff disclosure gain (+0.97) are robust across absolute evaluation, with CIs excluding zero.

4. **The purchase-specific positive interaction in BT pairwise decomposition does not replicate in absolute evaluation.** It should be reported but not interpreted as a confirmed compensatory mechanism.

### 4.2 What we cannot claim

1. We cannot claim these patterns reflect real consumer behavior — the evaluator is a GPT model, not a human.
2. We cannot claim premium-aspiration bias is the only explanation for negative retrieval — alternative explanations include the LLM optimizing for a different objective (e.g., segment-typical choices rather than persona-specific fit).
3. We cannot claim the positive purchase interaction is robust — it appears in BT pairwise decomposition but not in absolute evaluation.
4. We cannot generalize to other product categories, LLM architectures, or prompt designs without additional experiments.
5. We should not describe the pattern as "popularity bias" — history retrieval actually shifts *away* from bestsellers, toward more expensive niche products.

### 4.3 Draft paragraphs for the paper

**Paragraph 1 (Results section):**
The modular decomposition reveals that the near-zero total history shock conceals two large, opposing channel effects. History-aware retrieval reduces overall recommendation quality (BT estimate: -0.828, 95% CI [-1.379, -0.326]), while history-aware expression improves it (+0.704, [0.388, 1.033]). Product-level diagnostics suggest that the retrieval harm arises from a premium-aspiration pattern: when exposed to segment-level buyer feedback, the LLM shifts product selection toward more expensive products (mean price increase: +$25) that are less popular and have fewer reviews, while violating the persona's stated budget in 22% of cases where the generic retrieval respected it. The GPT evaluator's pairwise reasons most frequently cite better feature match and budget fit as reasons for preferring the generic retrieval product.

**Paragraph 2 (Discussion section):**
The opposing channel effects carry direct implications for firms deploying LLM-based recommender systems. Our simulation suggests that routing historical purchase information to the text-generation stage yields trust and satisfaction gains without the fit penalties associated with history-aware product selection. A modular architecture that uses specification-based retrieval with history-informed expression may outperform both the feature-only baseline and the fully history-aware system — a finding that would be invisible under a standard A/B comparison. The retrieval harm arises not from the LLM defaulting to bestsellers, but from its tendency to interpret segment satisfaction feedback as a signal to recommend premium alternatives that exceed individual budget constraints.

**Paragraph 3 (Limitations):**
The purchase-specific interaction merits caution. While the BT pairwise decomposition shows a positive purchase interaction (+0.426, P>0=95.4%), the absolute evaluation finds near-zero interaction (-0.4 pp, CI [-3.1, +2.2]). This discrepancy may reflect greater sensitivity of pairwise comparisons to within-cluster relative differences, or it may indicate that the BT interaction is partly a scale artifact. We do not build a compensatory-mechanism story on this finding. Human-subject validation is needed to determine whether these patterns hold under real consumer evaluation.

---

## 5. Recommended Figures

| Figure | File | Placement | Importance |
|--------|------|-----------|------------|
| **Main decomposition** | `fig_main_decomposition.pdf` | **Main text** | **Primary** — the paper's core result visualization |
| **Cell means by outcome** | `fig_cell_means_outcomes.pdf` | **Main text** | **Primary** — shows the full 2x2 pattern readers need |
| **Retrieval switch anatomy** | `fig_retrieval_switch_anatomy.pdf` | Main text or appendix | Supports retrieval mechanism story |
| **Retrieval harm scatter** | `fig_retrieval_harm_scatter.pdf` | Appendix | Robustness — shows cluster-level correlation |
| **Purchase interaction** | `fig_purchase_interaction.pdf` | Main text or appendix | Visualizes the compensatory pattern |
| **Pairwise win matrix** | `fig_pairwise_win_matrix.pdf` | Main text | Intuitive summary of all head-to-head comparisons |

The two most important figures for the main paper body are:
1. **fig_main_decomposition** — the decomposition bar chart is the paper's signature visualization
2. **fig_cell_means_outcomes** — shows readers the raw pattern before the BT decomposition

---

*Report generated from existing simulation outputs. No new LLM calls were made. All confidence intervals use cluster-level bootstrap (B=2000, seed=42). Reason coding uses heuristic keyword matching, not LLM classification.*
