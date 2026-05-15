# Working Paper Writeup Plan

**Working title:** *Unbundling LLM Recommenders: Retrieval, Persuasion, and Consumer Choice*
**Target venue:** QME conference (May 15 submission); journal target: Quantitative Marketing and Economics.
**Target main-paper length:** approximately 20 pages of body text, plus references and appendix.
**Style anchors:**

- Overall structure and streamlined exposition: Gui & Kim (2025), *Leveraging LLMs to Improve Experimental Design*. Specifically: short Introduction (~2 pp), focused Literature Review (~1.5 pp), method section that defers proofs to appendices, empirical/simulation section organized around two or three named comparisons, single Conclusion that doubles as Managerial Implications, no separate Discussion.
- Econometric expression: Feldman, Venugopal, Spiess and Feder (2026), *Causal Effect Estimation with Latent Textual Treatments*. Specifically: italic capitals for random variables, parenthesized treatment levels inside potential outcomes, tilde for residualized / second-best objects, named Propositions with one-line preconditions and one-line conclusions, all proofs deferred to a dedicated appendix.
- Writing voice: The author's prior work (Twitch gambling-ban paper). Specifically: first-person plural, active voice, sober tone, recurring "First / Second / Third" enumerations, "We assemble / We study / We find" topic sentences, "These estimates suggest" → managerial interpretation transitions, "Taken together" wrap-sentences, sparse content-rich footnotes, italicized terms on first use with `_term_` form, bold inline subheadings inside dense passages.

## 1. Goals and constraints

The main paper must be readable in one sitting by a quantitative-marketing referee. The reader should leave with three things: (i) a clear marketing question — why care about decomposing an LLM recommendation into retrieval and persuasion; (ii) a precise formal framework — what is identified and what is not under a one-shot prompt A/B test versus a modular 2 × 2 design; (iii) credible structural evidence — a simulation that demonstrates the design recovers the truth in two different DGP worlds and shows that the obvious-looking naive estimator is materially biased.

We will not overclaim. We will not say that LLMs break causal inference. We will not claim to be the first paper on causal inference with text. We will state that prompt A/B identifies the total policy effect on average, and that the modular design identifies the components.

## 2. Main-body structure (~20 pages)

### 2.1 Introduction (~2.5 pages)

Following the gambling-livestreams author's eight-paragraph arc:

1. **Hook.** LLM shopping assistants increasingly mediate product discovery. One sentence on scale and pace.
2. **Setup move: "In this paper, we investigate..."** Frame the central question: when an LLM recommender changes consumer choice, is it acting as a retrieval engine, a persuasion engine, or both at once?
3. **Twin-stakes "On the one hand / on the other hand."** A firm running a prompt A/B test sees only the total effect of a backend policy change. On one side, the standard randomization argument identifies total effects. On the other side, the same total effect could come from retrieval, expression, or their interaction — and these correspond to different investment decisions.
4. **What we do.** A modular 2 × 2 design crossing a retrieval policy `q ∈ {0,1}` with an expression policy `r ∈ {0,1}`. Italicize *retrieval policy* and *expression policy* on first use.
5. **Why it identifies.** One sentence each on Proposition 1 (A/B identifies the total), Proposition 2 (one-shot does not identify components), Proposition 3 (modular identifies all three), Proposition 4 (naive realized-expression regression is biased).
6. **Simulation evidence — first finding.** "First, we find that the modular design recovers retrieval, persuasion, and retrieval-persuasion interaction in two distinct DGPs..."
7. **Simulation evidence — second finding.** "Second, we find that a naive regression of outcomes on realized expression intensity overstates the persuasion effect by 80–130% of the truth across the policy-relevant range of expression endogeneity..."
8. **Contribution paragraph: "To the best of our knowledge, this paper is the first to..."** Then "Our findings provide at least threefold insights" — formal framework, design that is deployable on platforms, empirical demonstration that the naive estimator fails. Close with "The remainder of the paper is structured as follows."

Footnotes (2–3): (a) define "LLM shopping assistant" with examples; (b) note that the design assumes the firm can manipulate retrieval and expression separately; (c) cite the ARF/MSI generative-AI motivation.

### 2.2 Literature Review (~2 pages)

Three paragraphs, following the gambling-livestreams paper's "several strands" pattern. Each opens with "First/Second/Third, we contribute to the literature on X," cites 3–4 papers within the paragraph, briefly states the closest paper's main takeaway, and closes with an explicit differentiation sentence.

**Paragraph 1: Causal inference with text and generative content.** Cite Egami et al. (2022) and Feder et al. (2022) as foundational; Feldman et al. (2026) on latent textual treatments via SAE steering + residualization (closest paper — state their bias finding); Nakamura and Imai (2025) on the GPI framework. Positioning: "We extend this literature by studying a decomposition specific to product recommendation: LLM-mediated recommendations bundle a discrete product-selection decision with a continuous expression decision, and a modular experimental design can identify each as a separate policy effect."

**Paragraph 2: Persuasion and information in marketing communications.** Cite Shin and Wang (2024) on information vs. persuasion in advertising from a Bayesian persuasion perspective (closest — mention their messenger/signal-structure separation); Reisenbichler et al. (2022) on NLG for content marketing; Angelopoulos, Lee, and Misra (2024) on causal alignment; Costello et al. (2024) on LLM persuasion in political settings. Positioning: "Our paper differs in that we do not study whether LLM text is persuasive per se, but propose a design that separately identifies retrieval and persuasion channels within a single recommendation event."

**Paragraph 3: Platform experimentation and AI-mediated recommendation.** Cite Johnson, Lewis, and Nubbemeyer (2017) on ghost ads; Deldjoo (2024) on ChatGPT prompt sensitivity; Gui and Toubia (2024) on causal concerns with LLM-simulated subjects (closest — mention their unblinding insight); Muralidharan, Romero, and Wuthrich (2025) on factorial design misspecification in top-5 economics journals. Positioning: "We contribute by treating retrieval and persuasion as two separately manipulable policy components and providing a deployable factorial design that estimates each as a local policy effect."

No subsections. Roughly 900–1000 words.

### 2.3 Setup and Notation (~2 pages)

Following Feldman et al.: introduce the substantive object before the notation, then the notation in one compact subsection.

**2.3.1 Setting.** A consumer-session `i` arrives at an LLM-mediated shopping interface, submits a query, and receives a natural-language recommendation. The interface returns a tuple `(J, E)` — a selected product `J ∈ J` and an expression state `E ∈ E` — and the consumer takes an action `Y ∈ {0,1}` (click, purchase, search-continuation). A backend policy controls both layers. Concrete example: a backend prompt change at a retailer LLM raises the click rate from 0.30 to 0.35; the firm needs to attribute that lift.

**2.3.2 Random variables.** Italic capitals: `X` (consumer context), `J` (selected product), `E` (expression state, modeled as a real-valued intensity for tractability), `Y` (binary outcome). Treatment levels in parentheses inside potential outcomes: `Y(j, e)` is the outcome if shown product `j` with expression `e`. Policy assignments: `Z ∈ {0,1}` for the one-shot bundled policy; `(Q, R) ∈ {0,1}²` for the modular policies. We adopt the potential-outcomes framework.

**2.3.3 Policy kernels.** Write the one-shot stochastic kernel `K_z^{1s}(j, e | x)` and the modular kernel `K_{q,r}^{2s}(j, e | x) = K_q^J(j | x) · K_r^E(e | x, j)`. Make the bundle asymmetry explicit: in the one-shot kernel the policy index `z` enters both factors; in the modular kernel `q` enters only the retrieval factor and `r` only the expression factor.

### 2.4 Identification: One-shot is not modular (~3 pages)

This section carries the four propositions. Following Feldman: one-line precondition, one-line conclusion, brief discussion paragraph, proofs in Appendix A.

**Remark 1 (Total one-shot effect — benchmark).** Under SUTVA, consistency, and randomized assignment of `Z`, the total effect is identified by `E[Y | Z=1] − E[Y | Z=0]`. This is a standard result; we state it as a benchmark, not a contribution. *Proof in Appendix A.1 for completeness.*

**Proposition 1 (One-shot non-identification of components).** Under the same conditions, the one-shot difference does *not* identify the retrieval component, the persuasion component, or their interaction separately. (Use Feldman-style *generally* flag over `≠`.) *Proof via counterexample in Appendix A.2.*

**Proposition 2 (Modular identification).** Under SUTVA, consistency, randomized `(Q, R)` over the feasible support `A_i`, and overlap, the four modular cell means identify retrieval, persuasion, and retrieval-persuasion interaction effects. *Proof in Appendix A.3.*

**Proposition 3 (Naive realized-expression regression is biased).** If realized expression intensity is endogenously correlated with latent product fit, the OLS coefficient has asymptotic bias `λ_fit Var(Q̃) / (λ_fit² Var(Q̃) + σ_E²) × β_Q`. The bias is non-monotonic in `λ_fit`. *Proof in Appendix A.4.*

Discussion paragraph after each result, ~120 words, with bold inline subheading `**Interpretation.**` connecting the result to a deployable experiment. The Remark 1 interpretation explicitly states it is not a contribution.

### 2.5 Proposed Framework: A Modular 2×2 Experiment (~2 pages)

A short, plainspoken section describing the experimental architecture. Two subsections:

**2.5.1 Two-stage architecture.** Stage 1 takes consumer context and feasible catalog as input and returns a product under retrieval policy `q`. Stage 2 takes the selected product and consumer context as input and returns recommendation text under expression policy `r`. Italicize *retrieval stage* and *expression stage* on first use. State three deployment requirements: (i) the platform must be able to issue separate retrieval and expression calls; (ii) the expression stage cannot change the selected product; (iii) cell-assignment randomization must be uncorrelated with consumer covariates.

**2.5.2 Estimands.** Restate `Δ_R`, `Δ_P`, `Δ_R×P` as cell-mean contrasts, and connect each to a managerial decision: retrieval effect ↔ invest in catalog grounding; persuasion effect ↔ invest in expression prompts; interaction ↔ invest in matched product-expression strategies. One paragraph each, two sentences. Close with the affordance statement Gui & Kim use: *the design affects experimental allocation, not identification*.

### 2.6 Simulation Evidence (~6 pages, the central empirical section)

Two passes following Gui & Kim's two-pass empirical structure.

**2.6.1 Simulation setup.** One paragraph each on the catalog data (three categories, 8–10 products each, hand-curated for fit/price/use-case variation), the consumer profiles (1,000 per category with budgets, sensitivities, brand familiarity), the latent fit scores `Q_ij`, and the structural DGPs. The two DGPs:

- *Persuasion-leaning DGP.* Brand-forward retrieval substitutes brand affinity for fit, expression depends endogenously on latent fit through `λ_fit · Q_std_J`, and the outcome equation places moderate weight on both `Q` and `E`. The retrieval effect washes out; the persuasion effect is the leading channel.
- *Retrieval-dominant DGP.* Fit-aware retrieval raises Q-weighting under `q=1`, the brand-forward channel is turned off, and the persuasion coefficient `β_E` is reduced. Both retrieval and persuasion are positive, with retrieval as the leading channel.

We hold the master seed and the consumer and catalog data fixed across both DGPs.

**2.6.2 Pass 1 — Total-effect prompt A/B (Table 1).** Restate the one-shot total-effect estimates by category and pooled. State that the estimates are unbiased in expectation but underpowered at this sample size, and use this to motivate why a researcher cannot rely on the total effect alone.

**2.6.3 Pass 2 — Modular decomposition (Table 2 + Figure 1).** Restate the three components by DGP. Use bold inline subheading `**Persuasion-leaning DGP.**` and `**Retrieval-dominant DGP.**` to anchor the two sub-passes. State the estimates, then a "this pattern suggests" line that the modular design tracks the truth across both worlds.

**2.6.4 Naive bias and the λ_fit sweep (Figure 1).** Restate Table 3 and the λ-sweep figure. State that the naive bias peaks at moderate endogeneity (~130% of the true effect) and attenuates at the extremes — explain in one sentence the classical-OVB intuition (`Cov(E, Q) ∝ λ`, `Var(E) ∝ λ²`). Close with the managerial sentence: *a firm running a regression of conversions on realized expression strength systematically overstates the persuasion channel*.

**2.6.5 LLM-based simulation (placeholder, ~1 page).** This subsection will be written after the Phase 2 LLM simulations. Footnote: "We present LLM-based simulation results in Section 2.6.5; the structural simulations of 2.6.2–2.6.4 establish that the design recovers the truth under known DGPs, while the LLM-based simulations calibrate the structural parameters against the empirically observed expression-endogeneity strength of a deployed LLM recommender."

### 2.7 Conclusion (~2 pages)

Following Gui & Kim's single-section closing.

Paragraph 1: recap framework and findings. "We propose a modular 2 × 2 experimental design..."
Paragraph 2: managerial implications, with "First / Second / Third" enumeration. First: prompt A/B tests answer the wrong question for decomposition. Second: firms that interpret realized-expression regressions as persuasion effects will overstate the channel. Third: the modular design is deployable on any platform that can issue separate retrieval and expression calls.
Paragraph 3: limitations and future work. "Our study is subject to several limitations that point to potential directions for future work. First, ... Second, ... Third, ..."

## 3. Appendix structure

Indexed in order of appearance in the main paper. Cross-referenced from the main paper as "Appendix A.1", "Appendix B.2", etc.

- **Appendix A: Proofs.** A.1 Remark 1 (standard, included for completeness), A.2 Prop 1 (with worked counterexample showing two different `(Δ_R, Δ_P, Δ_R×P)` triples produce the same total effect), A.3 Prop 2, A.4 Prop 3 (with the OVB derivation that explains the non-monotonic bias path observed in the figure).
- **Appendix B: Structural DGP specification.** B.1 Retrieval kernel, B.2 Expression kernel, B.3 Outcome equation, B.4 Parameter table, B.5 Seed handling.
- **Appendix C: Catalog and consumer-profile generation.** C.1 Catalog construction and validation, C.2 Consumer-profile generator and `Q_ij` definition, C.3 Within-category standardization.
- **Appendix D: Estimator details.** D.1 Cell-mean contrast SEs, D.2 OLS LPM with HC1, D.3 Monte Carlo replication.
- **Appendix E: Per-category panels and additional figures.** E.1 Decomposition by category, E.2 lambda sweep raw values, E.3 robustness DGP detail tables.
- **Appendix F: LLM simulation protocol** (filled in after Phase 2). F.1 Model choice and quantization, F.2 prompt templates and JSON schema, F.3 calibration of `λ_fit` and `σ_R` to empirical LLM behavior, F.4 reproducibility notes.

## 4. What to write tonight versus tomorrow

**Tonight (May 13).** Sections 1 (Introduction), 2 (Literature Review), 3 (Setup and Notation), 4 (Identification), 5 (Framework), 6.1 (Simulation setup), 6.2–6.4 (Pass 1, Pass 2, and the λ sweep — these already exist in the report), and Section 7 (Conclusion as a stub). Appendices A.1–A.4, B, C, D, E.

**Tomorrow (May 14).** Section 6.5 (LLM-based simulation) and Appendix F. Polish pass on the introduction and conclusion. Final figure formatting.

**Constraint:** main-paper text must be tight. Anything that is method-detail, parameter-tuning rationale, robustness commentary, or proof body lives in the appendix.

## 5. Writing conventions

- **Voice.** First-person plural. Active. No contractions.
- **Tense.** Present tense for the framework ("the modular design identifies..."). Past tense for the simulation results ("we found that...").
- **Hedging.** "Suggest", "is consistent with", "we read this as evidence that". Hedges live on mechanism claims, not headline numbers.
- **Notation.** Italic capitals (`Y`, `J`, `E`, `Q`); parenthesized treatment levels (`Y(j, e)`); tilde for residualized objects (`Q̃`); `Δ_R`, `Δ_P`, `Δ_R×P` for the three estimands; `Δ_total` for the total effect.
- **Signposting inside long paragraphs.** "First, ...", "Second, ...", "Third, ...", "Finally, ...".
- **Term introduction.** Italicize on first use: *retrieval policy*, *expression policy*, *modular 2 × 2 design*, *naive realized-expression regression*, *bridge diagnostic*.
- **Bold inline subheadings.** Reserve for cases when a dense passage covers several distinct subcases (e.g., the two DGPs in 2.6.3). Avoid in narrative prose.
- **Footnotes.** Source URLs, definitional clarifications, scope caveats. Never digressions or jokes. Target ≤ 15 footnotes in the main paper.
- **Numbers in prose.** Lead with the headline number ("the naive estimator overstates persuasion by 130% of the truth"); detailed tables in tables, not in sentences.

## 6. References to draft into BibTeX

A short list of must-cite papers, to ensure the LaTeX `.bib` is ready before drafting begins.

- Angelopoulos, P., Lee, K., and Misra, S. (2024). *Causal Alignment*. Working paper.
- Berman, R. (2018). Beyond the Last Touch: Attribution in Online Advertising. *Marketing Science*.
- Deldjoo, Y. (2024/2025). Auditing ChatGPT-based recommendation. Working paper.
- Egami, N., Fong, C. J., Grimmer, J., Roberts, M. E., and Stewart, B. M. (2018). How to Make Causal Inferences Using Texts.
- Feldman, O., Venugopal, A., Spiess, J., and Feder, A. (2026). *Causal Effect Estimation with Latent Textual Treatments*.
- Gui, G., and Kim, K. (2025). *Leveraging LLMs to Improve Experimental Design: A Generative Stratification Approach*.
- Jannach, D. (2023). Evaluating Conversational Recommender Systems. *Artificial Intelligence Review*.
- Johnson, G., Lewis, R., and Nubbemeyer, E. (2017). Ghost Ads. *Journal of Marketing Research*.
- Nakamura, K., and Imai, K. (2025). *Causal Representation Learning with Generative Artificial Intelligence: Application to Texts as Treatments.*
- Reisenbichler, M., Reutterer, T., Schweidel, D. A., and Dan, D. (2022). Frontiers: Supporting Content Marketing with Natural Language Generation. *Marketing Science*.
- Feder, A., Keith, K. A., Manzoor, E., et al. (2022). Causal Inference in Natural Language Processing. *Transactions of the ACL*.
- Shin, J., and Wang, J. (2024). The Role of Messenger in Advertising Content: Bayesian Persuasion Perspective. *Marketing Science*.
- Costello, T. H., Pennycook, G., and Rand, D. G. (2024). Durably Reducing Conspiracy Beliefs Through Dialogues with AI. *Science*.
- Gui, G., and Toubia, O. (2024). The Causal Effects of Generative AI on Online Behavioral Experiments. Working paper.
- Muralidharan, K., Romero, M., and Wuthrich, K. (2025). Factorial Designs, Model Selection, and (Incorrect) Inference in Randomized Experiments. *Review of Economics and Statistics*.
- Wager, S. (2024). *Causal Inference: A Statistical Learning Approach.* Stanford University.

## 7. A 20-page budget

| Section | Target pages |
|---|---:|
| 1. Introduction | 2.5 |
| 2. Literature Review | 2.0 |
| 3. Setup and Notation | 2.0 |
| 4. Identification | 3.0 |
| 5. Proposed Framework | 2.0 |
| 6. Simulation Evidence | 6.5 |
| 7. Conclusion | 2.0 |
| **Main body total** | **20.0** |
| References | 1.5 |
| Appendix A: Proofs | 3.5 |
| Appendix B: DGP specification | 2.0 |
| Appendix C: Catalogs and consumers | 2.0 |
| Appendix D: Estimators | 2.0 |
| Appendix E: Additional tables and figures | 3.0 |
| Appendix F: LLM simulation protocol | 4.0 |
| **Appendix total** | **16.5** |

The 20-page main-body target is firm. The appendix is allowed to grow. Anything that does not justify a sentence in the main body but reviewers will want to see goes in the appendix.
