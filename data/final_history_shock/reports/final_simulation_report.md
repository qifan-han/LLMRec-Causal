# Final History-Shock Simulation Report

## 1. Executive Summary

The full history-shock treatment has a total effect of -0.369 (95% CI [-0.898, 0.130], P>0 = 7.8%).
Expression effect: 0.704 [0.388, 1.033], P>0 = 100.0%.
Retrieval effect: -0.828 [-1.379, -0.326], P>0 = 0.0%.
Interaction: -0.245 [-0.699, 0.201], P>0 = 14.1%.

## 2. What Changed Relative to Previous Pilot

- Qualitative-only history: no numerical rates shown to the LLM recommender
- Anti-leakage audit with regex detection and regeneration
- GPT-5.3 as demand-side judge (replacing local gemma2:9b)
- 120 clusters (vs 30 in pilot), 25 products per category (vs 10)
- GPT-generated diverse consumer personas (120 total)
- Both absolute and pairwise GPT evaluation
- Multiple outcome dimensions (overall, purchase, satisfaction, trust)

## 3. Design and Sample Size

| Dimension | Value |
|-----------|-------|
| categories | 1.0 |
| products per category | 25.0 |
| personas per category | 60.0 |
| clusters | 60.0 |
| local supply packages | 240.0 |
| gpt pairwise judgments | 360.0 |
| gpt absolute evaluations | 240.0 |

## 4. Supply-Side Results

### Retrieval Variation

| Category | Change Rate | Generic Top Share | History Top Share |
|----------|-------------|-------------------|-------------------|
| headphones | 81.7% | 25.0% | 28.3% |
| overall | 81.7% | 25.0% | 28.3% |

Leakage rate: 1/240 (0.4%)

## 5. Demand-Side Results

### All Pairwise Comparisons

| A vs B | n | A wins | B wins | ties | A win% | B win% | tie% |
|--------|---|--------|--------|------|--------|--------|------|
| 0.0 vs 1.0 | 60 | 13 | 47 | 0 | 21.7% | 78.3% | 0.0% |
| 0.0 vs 10.0 | 60 | 44 | 14 | 2 | 73.3% | 23.3% | 3.3% |
| 0.0 vs 11.0 | 60 | 39 | 21 | 0 | 65.0% | 35.0% | 0.0% |
| 1.0 vs 11.0 | 60 | 43 | 17 | 0 | 71.7% | 28.3% | 0.0% |
| 10.0 vs 1.0 | 60 | 16 | 44 | 0 | 26.7% | 73.3% | 0.0% |
| 10.0 vs 11.0 | 60 | 21 | 38 | 1 | 35.0% | 63.3% | 1.7% |

## 6. Bradley-Terry Decomposition

| Component | Estimate | SE | 95% CI | P(>0) |
|-----------|----------|----|--------|-------|
| retrieval | -0.828 | 0.273 | [-1.379, -0.326] | 0.0% |
| expression | 0.704 | 0.162 | [0.388, 1.033] | 100.0% |
| total | -0.369 | 0.269 | [-0.898, 0.130] | 7.8% |
| interaction | -0.245 | 0.228 | [-0.699, 0.201] | 14.1% |

## 7. Decomposition by Outcome Dimension

### Overall

| Component | Estimate | 95% CI | P(>0) |
|-----------|----------|--------|-------|
| retrieval | -0.848 | [-1.443, -0.336] | 0.0% |
| expression | 0.699 | [0.397, 1.028] | 100.0% |
| total | -0.387 | [-0.977, 0.134] | 8.0% |
| interaction | -0.237 | [-0.702, 0.203] | 14.8% |

### Purchase

| Component | Estimate | 95% CI | P(>0) |
|-----------|----------|--------|-------|
| retrieval | -1.663 | [-2.220, -1.165] | 0.0% |
| expression | -0.096 | [-0.486, 0.368] | 32.0% |
| total | -1.333 | [-1.913, -0.867] | 0.0% |
| interaction | 0.426 | [-0.079, 0.906] | 95.4% |

### Satisfaction

| Component | Estimate | 95% CI | P(>0) |
|-----------|----------|--------|-------|
| retrieval | -0.913 | [-1.527, -0.375] | 0.0% |
| expression | 0.649 | [0.352, 0.963] | 100.0% |
| total | -0.228 | [-0.787, 0.284] | 19.4% |
| interaction | 0.035 | [-0.351, 0.449] | 58.2% |

### Trust

| Component | Estimate | 95% CI | P(>0) |
|-----------|----------|--------|-------|
| retrieval | -0.578 | [-1.102, -0.076] | 1.0% |
| expression | 1.117 | [0.806, 1.502] | 100.0% |
| total | -0.177 | [-0.714, 0.310] | 25.6% |
| interaction | -0.717 | [-1.269, -0.226] | 0.4% |

## 8. Text Mechanism Audit

| Cell | PI | TD | Regret Risk | Trust |
|------|----|----|----|-------|
| 0.0 | 4.3 | 3.3 | 4.466666666666667 | 4.183333333333334 |
| 10.0 | 3.9833333333333334 | 3.9 | 5.066666666666666 | 4.066666666666666 |
| 1.0 | 4.366666666666666 | 4.266666666666667 | 4.5 | 4.766666666666667 |
| 11.0 | 4.066666666666666 | 4.266666666666667 | 5.016666666666667 | 4.283333333333333 |

## 9. What Can and Cannot Be Claimed

The outcome is a blinded GPT-based synthetic pairwise preference judgment, not observed market demand.

The simulation is designed to test whether a modular audit can reveal the channels through which historical purchase information changes LLM recommendation packages.

The decomposition separates product-selection changes from expression changes while holding the consumer and category fixed.

This simulation does **not** claim:
- Real purchase effects
- Real welfare effects
- Actual market demand
- Human consumer validation
- That local LLM behavior represents all frontier models
