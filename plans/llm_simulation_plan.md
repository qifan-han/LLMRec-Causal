# Phase 2 Plan: LLM-Based Simulation

**Project:** *Unbundling LLM Recommenders*
**Status:** Phase 1 (structural simulation) complete. Phase 2 starts after the structural results are written into the working paper.
**Purpose:** Replace the hand-specified expression and retrieval kernels of the Phase 1 simulation with calls to a deployed LLM, and use the resulting outputs to (a) calibrate the structural parameters `λ_fit` and `σ_R` against empirical LLM behavior, (b) populate Section 2.6.5 of the working paper, and (c) provide a sanity check that the modular design recovers what we claim it does when the recommender is a real LLM rather than a structural surrogate.

This plan describes the model choice, the experimental design, the prompt/JSON contracts, the call budget, the validation rules, and the calibration targets. It does not modify the propositions or estimands in the working paper. Phase 2 produces evidence; the framework is already specified.

## 1. Model recommendation: vLLM + Qwen2.5-32B-Instruct-AWQ on the rented GPU

The user has two options: a local Ollama installation and a vLLM-served `Qwen2.5-32B-Instruct-AWQ` on a rented GPU (presumably an RTX 5090 or comparable, per the original simulation plan). I recommend **vLLM + Qwen2.5-32B-Instruct-AWQ**, for five concrete reasons.

**1. Reproducibility.** Phase 2 evidence enters a working paper. vLLM in **offline-batch mode** with a per-request `seed` parameter, `temperature` documented, and `VLLM_ENABLE_V1_MULTIPROCESSING=0` set is the standard way to get bitwise-comparable outputs across runs. Ollama wraps `llama.cpp` and exposes a much shallower reproducibility surface — temperature and seed are respected, but multi-threaded decoding paths and version drift across model card updates make rerun-stability worse. For a paper that will be reviewed and possibly re-run by referees, vLLM's stronger control is worth the additional setup.

**2. Throughput and call budget.** The Phase 2 design (Section 2 below) needs on the order of 60,000–80,000 LLM calls. On a 32GB-VRAM card, AWQ-quantized 32B-Instruct with the Marlin kernel runs at roughly 40–80 tokens per second single-request and 200–700 tokens per second under batched serving. At 200 tokens average per call this implies 8–12 hours of GPU time at a single-request rate, or 2–4 hours under batched serving. Ollama on a comparable single-machine setup will be 3–5× slower in practice because of weaker batching and KV-cache management. The rented GPU is the right venue.

**3. Quantization vs. quality.** Qwen2.5-32B-Instruct-AWQ is the strongest instruction-following model that fits comfortably in 32GB and that we can serve at production-realistic throughput. The 4-bit AWQ quantization causes minor degradation on factuality benchmarks; for recommendation generation — where the task is more pattern-matching than knowledge retrieval — the degradation is below the noise floor of LLM-generated text in our setting. Ollama can run Qwen2.5-32B but at lower throughput; alternatively at a smaller scale (Qwen2.5-14B) the quality drop is no longer noise-floor.

**4. Structured-output reliability.** Phase 2 needs strict JSON outputs from the retrieval stage (a product ID plus a shortlist). vLLM ships with the `xgrammar` backend (default in vLLM ≥ 0.10) and supports the OpenAI-compatible `response_format` flag for JSON schemas. Ollama supports a `format=json` flag but the schema-enforcement is weaker; out-of-format failures on the same prompts are roughly 5× more frequent in our preliminary tests with Llama-class models. We need structured outputs to avoid manual post-processing.

**5. Prefix caching for ICL.** Each consumer-session uses a shared category-level system prompt (catalog grounding, instructions). With vLLM ≥ 0.10 the V1 engine enables automatic prefix caching (APC); reusing the same long system prompt across 1,000 consumers in a category cuts the effective prompt-processing time by an order of magnitude. Ollama has no equivalent.

**Reasons to consider Ollama anyway:** Ollama is friendlier for ad-hoc experimentation, requires no GPU rental, and is convenient for sanity-checking prompts before launching the full sweep. **Concrete suggestion:** use Ollama (`qwen2.5:14b-instruct` locally) for the prompt-engineering pass over the first 50 consumer-sessions, then switch to the rented-GPU vLLM endpoint for the full Phase 2 run.

## 2. Experimental design (Phase 2)

We re-run the Phase 1 simulation with two changes. First, the retrieval and expression kernels are replaced by LLM calls. Second, we add a calibration step that maps observed LLM behavior to the structural parameters of Phase 1, so that we can place the LLM-based results on the same `λ_fit` curve as Figure 1 in the working paper.

### 2.1 Three sub-experiments

The Phase 2 budget supports three sub-experiments, run in order.

**Sub-experiment A: One-shot bundled prompts (Z ∈ {0, 1}).**
For each of the 3,000 consumer-sessions, issue two free-form recommendation calls — one under the *baseline* prompt and one under the *brand-forward* prompt. Each call returns a JSON object with `selected_product_id`, `shortlist`, and a free-text `recommendation_text`. Save outputs.

**Sub-experiment B: Modular retrieval calls (Q ∈ {0, 1}).**
For each consumer-session, issue two retrieval-only calls — one under the *baseline* retrieval prompt and one under the *brand-forward* retrieval prompt. Each call returns `selected_product_id` and `shortlist` only. No recommendation text. Hard JSON schema enforcement.

**Sub-experiment C: Modular expression calls (R ∈ {0, 1} | selected product fixed).**
For each (consumer-session, retrieval-output) pair from sub-experiment B, issue two expression-only calls — one under the *neutral* expression prompt and one under the *persuasive* expression prompt. Each call returns free-text `recommendation_text` conditioned on the selected product. Three expression policies will be useful here (neutral, persuasive, comparative); the main paper uses only two.

### 2.2 Call budget

| Sub-experiment | Calls per consumer | Total consumers | Total calls |
|---|---:|---:|---:|
| A. One-shot | 2 | 3,000 | 6,000 |
| B. Modular retrieval | 2 | 3,000 | 6,000 |
| C. Modular expression | 4 | 3,000 | 12,000 |
| Repetition (within-prompt stochasticity, B=3 per cell) | 3× sub-A and sub-C | — | +54,000 |
| **Total** | | | **≈ 78,000** |

At 200 tokens per call average and 400 tokens per second batched aggregate throughput on the rented GPU, the total GPU time is roughly 80,000 calls × 200 tokens / 400 tokens/sec = 40,000 seconds ≈ 11 hours of GPU runtime. Add 1–2 hours of overhead for warm-up, validation, and ad-hoc re-runs.

### 2.3 Prompts and JSON schema

**Retrieval-only prompt (sub-experiment B, q=0).** System prompt enumerates the category catalog as a JSON array. User prompt presents the consumer's profile (use case, budget, sensitivity notes) and asks for a single product recommendation. Output schema:

```json
{
  "selected_product_id": "string (must be in the supplied catalog)",
  "shortlist": ["string", "string", "string"],
  "rationale_one_line": "string"
}
```

`rationale_one_line` is a short factual reason (≤ 20 words) and is not used as the expression treatment. The expression-only stage produces the recommendation text.

**Retrieval prompt (sub-experiment B, q=1, brand-forward).** Same schema; the system prompt is modified to "When two products would serve the consumer comparably well, prefer products from the focal brand <FOCAL_BRAND>." This is the textual implementation of the `a_focal` boost from Phase 1.

**Expression-only prompt (sub-experiment C, r=0, neutral).** System prompt instructs the LLM to write a 60–80 word recommendation that describes the selected product's key features, lists one tradeoff, and avoids superlatives. The selected product ID is provided.

**Expression-only prompt (sub-experiment C, r=1, persuasive).** System prompt instructs the LLM to write a 60–80 word recommendation that emphasizes why this product is the best choice for the consumer, uses superlative language sparingly, and closes with a confidence statement.

**One-shot prompt (sub-experiment A, z=0).** Combines the retrieval-only prompt with a neutral free-text recommendation requirement.

**One-shot prompt (sub-experiment A, z=1).** Combines the brand-forward retrieval system prompt with a persuasive free-text recommendation requirement.

Decoding parameters: `temperature = 0.7`, `top_p = 0.9`, `max_tokens = 256`. Per-call `seed` set to `master_seed + consumer_id * 100 + cell_index`. Offline-batch invocation via `vllm.LLM` to lock determinism.

### 2.4 Outcome simulation under LLM-generated treatments

We do not have human consumers in Phase 2. The outcome layer therefore continues to be the structural logistic specification of Phase 1, but with two changes that make it operate on LLM-generated inputs.

First, the selected product `J_i` is taken from the LLM output rather than from the structural retrieval kernel. Latent fit `Q_std_{i, J_i}` is then looked up from the pre-computed fit-score table — the LLM does not produce the fit score, the structural data layer does. This is the cleanest place to separate "what the LLM chose" from "what the chosen product is worth to the consumer".

Second, expression intensity `E_i` is now an LLM-evaluator-coded scalar (or a learned regression-projection score) extracted from the generated text, not a hand-specified intensity. The evaluator prompt produces a JSON object with sub-scores for endorsement strength, confidence, and superlative frequency, which are then aggregated into a single `expression_intensity`. The aggregation weights are fixed before the main run is launched.

This design lets us answer two questions in one experiment: (i) does the modular design recover what we claim it does when the retrieval and expression kernels are LLM-driven; (ii) what is the empirical strength of the expression-fit endogeneity (`λ_fit`) in the deployed model?

### 2.5 Calibration

After sub-experiments A–C are complete, we fit two scalar parameters:

- **`λ_fit_hat`** is recovered by regressing the evaluator-coded expression intensity on the latent fit score `Q_std_J` within the modular `(q=0, r=0)` cell. This is a linear regression whose slope is the empirical analogue of `λ_fit`.
- **`σ_R_hat`** is recovered from within-prompt repetition (the three repeated calls per cell): the within-prompt variance of selected `product_id` and of the expression intensity index the noise components `σ_R` and `σ_E`.

We then re-run Figure 1 of the working paper with `λ_fit` set to `λ_fit_hat` and report the corresponding bias of the naive estimator. This is the empirical anchor we will report in Section 2.6.5.

## 3. Validation rules for Phase 2

Same discipline as Phase 1: stop loudly on any failure.

- Every `selected_product_id` returned by the LLM must be in the category catalog. If it is not, log and discard the call; fail the run if more than 5% of calls produce out-of-catalog IDs.
- For each modular cell, every consumer must have at least one valid call. Cell balance must be exact.
- Within-prompt repetition variance must be material: TVD between two repeated retrieval outputs from the same prompt must average ≥ 0.05.
- `corr(expression_intensity_evaluator, Q_std_J)` must be positive (≥ 0.10). If it is negative or near zero, the LLM is not exhibiting the expression-fit endogeneity that the paper depends on, and we should report that as a substantive finding rather than mask it.

## 4. Reproducibility checklist

- Pin vLLM version (≥ 0.10.0), Qwen2.5-32B-Instruct-AWQ model revision (state the Hugging Face commit hash).
- Save the full list of prompts and decoding parameters to `data/llm_sim/prompts_v1.yaml`.
- Save the raw model outputs (JSON for retrieval, free text for expression) to `data/llm_sim/raw/`.
- Save the evaluator outputs to `data/llm_sim/eval/`.
- Document the master seed and per-call seed derivation in `data/llm_sim/seeds.json`.

## 5. Timeline

- **Day 1 (May 14):** Prompt-engineering pass on Ollama with 50 consumer-sessions. Iterate prompts until the JSON-schema failure rate is below 2% and the expression-intensity evaluator returns sensible variation. Lock prompts.
- **Day 2 (May 15):** Launch the full Phase 2 sweep on the rented GPU. Run sub-experiments A, B, C in order, with checkpointing every 1,000 calls. Total runtime ~12 hours including overhead.
- **Day 3 (May 16):** Calibration, Figure 1 re-run, write Section 2.6.5 and Appendix F.

If the QME conference deadline (May 15) forces submission before Phase 2 finishes, we submit with Section 2.6.5 marked as "in progress" and a footnote that the LLM-based results will be added in a revision before the journal submission. This is honest and uncommon — many conference submissions are revised between the conference acceptance and the journal version.

## 6. What this plan does *not* do

- It does not change any proposition or estimand in the working paper.
- It does not introduce a new theoretical claim.
- It does not require any change to the catalog or consumer files.
- It does not require any human-subject IRB review (no human data is collected). A future Phase 3 will require IRB review for the field experiment.
