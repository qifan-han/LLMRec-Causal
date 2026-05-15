# Unified Black-Box & Architecture Gap (Gamma) — Report

**Date**: 2026-05-16
**Status**: In progress — supply generation running, GPT eval and Gamma computation pending

---

## 1. Purpose

This experiment estimates the **architecture gap** Gamma, the key bridge estimand proposed in the new paper framing (`temp/new_framing_H_notation_gamma.md`):

```
Gamma = tau^BB - tau^MOD
```

where:
- `tau^BB = mu^BB_1 - mu^BB_0` is the total effect of the history shock under a **unified black-box** LLM recommender (single call, model jointly decides what to recommend and how to describe it).
- `tau^MOD = mu^MOD_11 - mu^MOD_00` is the diagonal effect under the **modular two-stage** architecture (separate retrieval and expression calls).

If Gamma is approximately zero, the modular decomposition can be interpreted as explaining where the black-box gain comes from. If Gamma is large, the modular decomposition is still useful as a policy comparison but should not be described as decomposing the black-box shock.

---

## 2. Why This Experiment Was Needed

### What existed before
The modular experiment produced all four cell means `mu^MOD_00, mu^MOD_01, mu^MOD_10, mu^MOD_11`, enabling the full modular decomposition (Exercise 2). An earlier diagnostic experiment (`data/diagnostic/supply_outputs.csv`) ran both unified and two-stage architectures, but **only under the feature-only baseline (p0)**. No unified history-aware outputs (Z=1) were ever generated.

### What was missing
- `mu^BB_1` (unified black-box, history-aware mean outcome) — completely absent
- Therefore `tau^BB` — not computable
- Therefore `Gamma` — not computable

### What the diagnostic baseline showed
Under the feature-only baseline (p0), unified and two-stage architectures agreed on product selection only 27.3% of the time, yet their demand-side outcomes were similar (purchase likelihood: 77.1 vs 77.8). This suggested the architectures produce different products with comparable consumer value — but we couldn't test whether this holds under the history shock.

---

## 3. Experimental Design

| Parameter | Value |
|-----------|-------|
| Category | Headphones (same as modular experiment) |
| Personas | 60 (same as modular experiment) |
| Catalog | 25 products (same as modular experiment) |
| Conditions per persona | 2 (Z=0 feature-only, Z=1 history-aware) |
| Total supply rows | 120 (60 x 2) |
| Supply model | Qwen 2.5 14B (local, Ollama, temperature 0.7) |
| Demand model | GPT-5.3-chat-latest (OpenAI API) |
| GPT evaluations | 120 (60 x 2) |
| Bootstrap for Gamma | B=2000, cluster-level, seed=42 |
| Master seed | 20260515 (same base as modular) |
| Seed offsets | +50 (Z=0), +60 (Z=1) — distinct from modular offsets |

### Unified prompt design

**Z=0 (feature-only)**: The LLM sees the full product catalog (specs only, no reviews/ratings/popularity) and the consumer persona. It must select a product AND write a consumer-facing recommendation in a single JSON response. Uses the `RECOMMENDER_PERSONA_GENERIC` and `STYLE_INSTRUCTION_GENERIC` from the modular experiment's prompts.

**Z=1 (history-aware)**: The LLM sees the catalog with popularity tiers and rating counts, plus segment-level historical buyer feedback summaries. It must select a product AND write a recommendation in a single call. Uses `RECOMMENDER_PERSONA_HISTORY` and `STYLE_INSTRUCTION_HISTORY`. Anti-leakage instruction is included.

### Key architectural difference from modular

| Aspect | Modular two-stage | Unified black-box |
|--------|-------------------|-------------------|
| Calls per condition | 2 (retrieval + expression) | 1 (joint) |
| Information routing | Researcher-controlled | Model-internal |
| Product lock | Enforced between stages | N/A (single call) |
| Factorial manipulation | 4 cells (H^J x H^T) | 2 conditions (Z=0 vs Z=1) |

In the unified design, the LLM can use history information to jointly influence both product selection and text generation in ways the researcher cannot separately observe — this is precisely the "black box" that Gamma measures against.

---

## 4. Current Progress

### Supply generation
- **Started**: 2026-05-16 00:00
- **Progress**: Running (estimated ~2 hours for 120 Ollama calls)
- **Early quality checks** (first 3 clusters):
  - Parse failures: 0/6
  - Leakage flags: 0/6
  - Product differentiation: 3/3 clusters show different products under Z=0 vs Z=1
  - Word counts: 55-67 (within target range)

### Pending steps
1. GPT absolute evaluation (120 calls) — script ready: `14b_gpt_eval_unified_bb.py`
2. Gamma computation with bootstrap CIs — script ready: `14c_compute_gamma.py`

---

## 5. What Gamma Will Tell Us

### Scenario A: Gamma approximately zero
The modular diagonal approximates the black-box shock. The paper can claim:

> For the local Qwen model and simulation environment studied here, the two-stage modular diagonal closely approximates the unified black-box history shock. We therefore use the modular design as a local decomposition of the black-box effect.

This validates the modular decomposition as an explanation of the black-box effect, not just as a standalone policy comparison.

### Scenario B: Gamma is large
The modular design is still useful, but it should not be described as decomposing the black-box shock:

> The modular design identifies policy-relevant component effects for a two-stage implementation, but these effects should not be interpreted as decomposing the unified black-box LLM shock.

The modular findings (retrieval hurts, expression helps) remain valid as implementable policy benchmarks — they just can't claim to explain the monolithic model's behavior.

### What the modular results suggest about Gamma
From the modular experiment, `tau^MOD = -0.369` (overall BT). If the unified black-box shows a similar total effect, Gamma will be close to zero. The key uncertainty is whether the unified model's internal information routing produces meaningfully different outcomes than the researcher-controlled modular routing.

---

## 6. Quantities to Be Computed

| Quantity | Symbol | Source |
|----------|--------|--------|
| Unified feature-only mean | `mu^BB_0` | GPT eval of unified Z=0 |
| Unified history-aware mean | `mu^BB_1` | GPT eval of unified Z=1 |
| Black-box effect | `tau^BB = mu^BB_1 - mu^BB_0` | Computed |
| Modular diagonal baseline | `mu^MOD_00` | Already available (cell 0) |
| Modular diagonal treatment | `mu^MOD_11` | Already available (cell 11) |
| Modular diagonal effect | `tau^MOD = mu^MOD_11 - mu^MOD_00` | Already available |
| Architecture gap | `Gamma = tau^BB - tau^MOD` | To be computed |
| Approximation ratio | `tau^MOD / tau^BB` | To be computed |

All quantities will be reported with cluster-level bootstrap 95% CIs (B=2000).

---

## 7. Modular Estimates Already Available (for reference)

From the modular experiment absolute eval:

| Outcome | mu^MOD_00 | mu^MOD_11 | tau^MOD |
|---------|-----------|-----------|---------|
| Fit score (1-7) | 4.15 | 3.37 | -0.78 |
| Purchase probability (0-100) | 55.55 | 41.33 | -14.22 |
| Trust score (1-7) | 4.18 | 4.28 | +0.10 |
| Persuasive intensity (1-7) | 4.30 | 4.07 | -0.23 |
| Tradeoff disclosure (1-7) | 3.30 | 4.27 | +0.97 |

---

## 8. Files and Locations

| File | Description | Status |
|------|-------------|--------|
| `src/final_history_shock/14_unified_bb_supply.py` | Unified BB supply generation | Created |
| `src/final_history_shock/14b_gpt_eval_unified_bb.py` | GPT evaluation of unified BB | Created |
| `src/final_history_shock/14c_compute_gamma.py` | Gamma computation + report | Created |
| `data/final_history_shock/unified_bb/unified_bb_cache.jsonl` | Supply generation cache | In progress |
| `data/final_history_shock/unified_bb/unified_bb_supply.csv` | Final supply CSV | Pending |
| `data/final_history_shock/unified_bb/unified_bb_eval_cache.jsonl` | GPT eval cache | Pending |
| `data/final_history_shock/unified_bb/unified_bb_eval.csv` | GPT eval results | Pending |
| `data/final_history_shock/unified_bb/gamma_estimates.csv` | Gamma estimates table | Pending |
| `data/final_history_shock/unified_bb/gamma_report.md` | Auto-generated Gamma report | Pending |
| `temp/new_framing_H_notation_gamma.md` | Conceptual framing document | Reference |
| `temp/architecture_gap_experiment_log.md` | Design log and rationale | Created |

---

## 9. Relationship to Paper Exercises

### Exercise 1 (reframed as architecture bridge)
This experiment IS Exercise 1 in the new framing. It answers: does the modular two-stage diagonal approximate the unified black-box history shock?

### Exercise 2 (modular decomposition, conditional on Exercise 1)
If Gamma is approximately zero, the modular decomposition results from the companion report (`modular_decomposition_report.md`) can be interpreted as decomposing the black-box effect:
- Delta_J (retrieval): -0.828
- Delta_T (expression): +0.704
- Delta_JT (interaction): -0.245
- tau^MOD (total): -0.369

---

*This report will be updated with final Gamma estimates once the supply generation, GPT evaluation, and computation steps complete.*
