# Architecture Diagnostic: Unified BB vs Modular Diagonal

Blinded pairwise GPT evaluation comparing:
- Unified Z=0 vs Modular cell (0,0) — baseline condition
- Unified Z=1 vs Modular cell (1,1) — treated condition

Clusters: 60
Bootstrap: B=2000, cluster-level, seed=42

## Baseline condition: Z=0 vs cell (0,0)

Product agreement: 50/60 (83.3%)

| Outcome | Unified wins | Modular wins | Tie | Unified 95% CI |
|---|---|---|---|---|
| overall | 46.7% | 51.7% | 1.7% | [35.0%, 60.0%] |
| purchase | 40.0% | 48.3% | 11.7% | [28.3%, 53.3%] |
| satisfaction | 35.0% | 43.3% | 21.7% | [23.3%, 46.7%] |
| trust | 45.0% | 48.3% | 6.7% | [31.7%, 58.3%] |

## Treated condition: Z=1 vs cell (1,1)

Product agreement: 37/60 (61.7%)

| Outcome | Unified wins | Modular wins | Tie | Unified 95% CI |
|---|---|---|---|---|
| overall | 28.3% | 71.7% | 0.0% | [18.3%, 40.0%] |
| purchase | 40.0% | 56.7% | 3.3% | [26.7%, 51.7%] |
| satisfaction | 25.0% | 68.3% | 6.7% | [15.0%, 36.7%] |
| trust | 23.3% | 71.7% | 5.0% | [13.3%, 35.0%] |

## Interpretation

If win rates are roughly balanced (each architecture winning ~30-50% 
with substantial ties), the modular diagonal approximates the unified 
black-box recommender. The main decomposition results can then be 
interpreted as informative about the original system.

If one architecture dominates, the modular design remains a valid 
policy experiment, but its decomposition should be read as effects 
within an engineered two-stage recommender rather than as a full 
explanation of the unified LLM.
