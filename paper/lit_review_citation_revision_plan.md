# Literature Review, Citation, and Claim-Strength Revision Plan

This note is intended for Claude to revise `unbundling_llm_recommenders.tex` and `references.bib`. Do **not** treat the unfinished main-simulation placeholders as problems; the main simulation is still running. Focus only on the related literature, citation correctness, and claim-strength/structure issues.

## 0. Bottom-line diagnosis

The paper idea is clean: LLM recommenders bundle (i) product selection/retrieval and (ii) generated persuasive/explanatory expression; a modular `2 x 2` design can separately estimate the effects of experimentally manipulated retrieval and expression policies.

The current literature review is directionally right but needs cleanup. The biggest issues are:

1. Several paper summaries are too broad or slightly wrong.
2. Some papers are placed in branches where they are only indirectly relevant.
3. The second and third literature-review paragraphs overlap: both discuss recommender architecture, LLM recommendation, language/explanation effects, and causal text methods. Split these branches more cleanly.
4. The bibliography contains incorrect names, titles, years, venues, and author order. This must be fixed before circulation.
5. The main claims should be sharpened around the actual estimand: effects of *experimentally manipulated retrieval and expression policies over feasible recommendation support*, not a universal decomposition of all possible LLM recommendation effects.

---

## 1. Literature-review problems to fix

### 1.1 Current branch: causal recommendation and counterfactual policy learning

This branch is relevant and should stay. The cited papers are mostly appropriate:

- `swaminathan2015batch`: relevant for logged bandit feedback and counterfactual risk minimization.
- `schnabel2016recommendations`: relevant because it explicitly treats recommendations as treatments and uses causal/propensity ideas for debiased learning/evaluation.
- `sharma2015estimating`: relevant for estimating the causal impact/incrementality of recommendation systems from observational data.
- `sato2020unbiased` and `sato2021online`: relevant for optimizing/evaluating the causal effect of recommendations.

Revision needed: do not mix this branch too early with mediation/direct-indirect-effects literature. The causal-recommendation branch is about whether recommendation exposure/items cause outcomes; the mediation/factorial literature is about effect decomposition. Keep them adjacent but conceptually separate.

Suggested framing:

> Existing causal-recommendation papers ask whether recommended items or exposures causally increase user actions. This paper asks a different question: when the recommender is an LLM system, which experimentally manipulable component of the recommendation pipeline drives the effect?

### 1.2 Current branch: LLM-based recommendation, retrieval architectures, and persuasive expression

This branch currently combines three things that should be separated:

1. LLM-based recommenders / ranking / popularity bias.
2. Two-stage recommender architecture and component interactions.
3. Recommendation explanations and persuasive language.

Specific issues:

- `hou2023bridging`: current wording says it “propose[s] a unified framework that bridges language representations and item embeddings for both retrieval and recommendation stages.” This is mostly fine, but the paper is more specifically about **BLaIR**, pretrained sentence embedding models specialized for recommendation scenarios, and it introduces/uses Amazon Reviews 2023. Do not overclaim that it establishes the entire retrieval--recommendation distinction.

- `hou2024large`: summary is broadly correct, but it should say the paper focuses on **LLMs as zero-shot rankers** and documents limits such as sensitivity to item order/history and popularity/item-position bias. This is directly useful because your paper asks what drives LLM recommendation effects beyond ranker performance.

- `lichtenberg2024llm`: current text calls this a “survey,” which is wrong. It is an empirical study of **LLM-based recommender systems and popularity bias**, including prompt-tuning considerations. Correct this.

- `deldjoo2024`: current text says “position, popularity, and verbosity.” The title and abstract emphasize **provider fairness, temporal stability, and recency**, with prompt design and system-role effects. Correct the summary.

- `xrec2024`: current text says XRec shows explanation quality varies with prompting strategy. That is not the clean summary. XRec introduces a model-agnostic LLM framework for **explainable recommendation** by connecting collaborative signals to generated explanations. Do not frame it as mainly a prompt-variation paper.

- `rahman2026persuasive`: current text says it shows natural-language explanations increase click-through and adoption rates relative to unexplained recommendations. That is too specific and likely wrong. Safer/correct summary: the paper studies persuasive natural-language explanations in recommender systems and shows persuasive framing can influence users' choices/preferences among recommended options.

- `costello2024persuasion`: relevant only as broader evidence that AI-generated dialogue can persuade. It is not a recommender-systems paper. Keep it only in a broader “AI-generated persuasive content” sentence, not as direct evidence about recommendation explanations.

- `shin2024messenger`: relevant as broader advertising/persuasion theory. But do not say it “parallels our retrieval--expression decomposition” too strongly. Messenger identity is not retrieval. Use it only to support the idea that message source/content can shape information structures and attention.

### 1.3 Current branch: two-stage architectures, component interactions, and causal text analysis

The current paragraph has the right papers but too many distinct branches in one paragraph.

- `ma2020off`: relevant. It develops off-policy learning for two-stage recommender systems and shows that ignoring stage interaction can lead to suboptimal policies.

- `hron2021component`: relevant. It explicitly studies component interactions in two-stage recommenders and shows independent optimization can fail because stage interactions matter.

These two papers should be moved into the architecture paragraph, not bundled with causal text analysis.

- `egami2018causal`: current summary says it lays out identification challenges when text is a treatment variable. Better: the published paper introduces a split-sample workflow for making causal inferences using discovered measures from text as treatments or outcomes.

- `feder2022causal`: summary is fine.

- `feldman2026latent`: very relevant. Summary should mention latent textual treatments, SAE-based hypothesis generation/steering, and robust causal estimation/residualization; the key connection is that naive text-as-treatment estimation can be biased because text contains both treatment and covariate information.

- `nakamura2025gpi`: author order and year need correction. The paper is by Kosuke Imai and Kentaro Nakamura. Summary should say it uses generative AI/internal representations to improve causal representation learning for text-as-treatment problems.

- `gui2024causal`: current title is wrong in the bibliography. The paper is about the challenge of using LLMs to simulate human behavior from a causal-inference perspective. It is relevant because your paper uses LLM-based demand-side evaluation; the review should say this paper cautions that LLM-simulated subjects can violate unconfoundedness when treatment prompts change unspecified covariates.

- `angelopoulos2024causal`: relevant but the current summary is a little generic. Use it as a marketing paper that aligns language models to A/B-test causal signals for downstream marketing communication decisions.

### 1.4 Irrelevant or currently unused papers

The following entries are in `references.bib` but not cited in the `.tex` file:

- `jannach2023crs`
- `johnson2017ghost`
- `berman2018attribution`
- `li2014attribution`

Recommendation:

- `jannach2023crs` can be added to the LLM/conversational recommender paragraph if you want a citation on holistic evaluation of conversational recommender systems.
- `johnson2017ghost`, `berman2018attribution`, and `li2014attribution` should be removed unless the paper adds a short paragraph about ghost ads/advertising attribution. Right now they are not necessary for the unbundling story.

---

## 2. Replacement Related Literature section

Use the following as a direct replacement for the current `Related Literature` section. This version separates the branches more cleanly and avoids the inaccurate summaries above.

```tex
\section{Related Literature}\label{sec:lit}
%-------------------------------------------------------------------------

This paper connects four strands of existing literature: causal recommendation and counterfactual policy learning, two-stage and LLM-based recommender systems, persuasive/explanatory recommendation language, and causal inference with text and generative AI.

\paragraph{Causal effects of recommendations and counterfactual policy learning.}
A growing literature treats recommendation as a causal action rather than merely a prediction problem. \citet{swaminathan2015batch} study learning from logged bandit feedback and develop counterfactual risk minimization for stochastic policy learning in online systems such as recommendation, advertising, and search. \citet{schnabel2016recommendations} explicitly frame recommendations as treatments and use propensity-based methods to debias learning and evaluation under self-selection and recommender-induced selection bias. \citet{sharma2015estimating} estimate the causal impact of recommender systems from observational data, emphasizing the distinction between observed recommendation-driven activity and incremental behavior. \citet{sato2020unbiased} propose learning to rank items by the causal effect of recommendation rather than by predicted purchase probability, and \citet{sato2021online} develop online interleaving methods for evaluating recommendation models by causal effects. This work shares the premise that recommendations are interventions, but differs in the object of decomposition: existing work typically studies the causal effect of item exposure or recommendation policies, whereas we decompose an LLM recommendation into experimentally manipulable retrieval and expression components.

\paragraph{Two-stage and LLM-based recommender systems.}
Modern recommender systems often separate candidate generation, retrieval, ranking, and presentation. \citet{ma2020off} develop off-policy learning methods for two-stage recommender systems and show that ignoring interactions between stages can lead to suboptimal policies. \citet{hron2021component} further show that two-stage recommenders cannot generally be understood as the sum of independently optimized components because interactions between nominators and rankers affect overall performance. Recent work brings large language models into this pipeline. \citet{hou2023bridging} introduce BLaIR, a family of language-item representation models for retrieval and recommendation, together with the Amazon Reviews 2023 dataset used in our empirical exercise. \citet{hou2024large} show that LLMs can serve as zero-shot rankers for recommender systems, while also documenting limitations related to history order, item position, and popularity. \citet{lichtenberg2024llm} study popularity bias in LLM-based recommender systems and the role of prompt tuning, and \citet{deldjoo2024} examine provider fairness, temporal stability, recency, and prompt-design effects in ChatGPT-based recommender systems. This literature establishes that LLM recommenders can select, rank, and bias product exposure, but it does not provide a causal experiment that separates product selection from generated expression.

\paragraph{Recommendation explanations and persuasive expression.}
A separate literature studies the language that accompanies recommendations. \citet{xrec2024} introduce an LLM-based framework for generating comprehensive natural-language explanations of recommendations by connecting collaborative signals to language generation. \citet{rahman2026persuasive} show that persuasive natural-language explanations can influence users' choices among recommended options. More broadly, \citet{reisenbichler2022nlg} show that natural-language generation can support content marketing, while \citet{costello2024persuasion} provide evidence that personalized AI dialogue can durably shift beliefs in a noncommercial setting. In marketing theory, \citet{shin2024messenger} model advertising content and messenger choice as a Bayesian-persuasion problem in which communication changes consumers' information environment. These papers motivate the expression side of our framework: generated recommendation text is not a passive explanation of a fixed product choice, but an economically meaningful component that may affect consumer outcomes.

\paragraph{Causal inference with text and generative AI.}
Our design also relates to work on text as treatment, text as outcome, and generative AI as an experimental tool. \citet{egami2018causal} introduce a split-sample workflow for making causal inferences using discovered text measures as treatments or outcomes, and \citet{feder2022causal} survey the broader role of causal inference in NLP. Most closely related methodologically, \citet{feldman2026latent} develop a pipeline for estimating effects of latent textual treatments, combining LLM-based generation and steering with robust causal estimation, and show why naive text-as-treatment estimates can be biased when text mixes treatment and covariate information. \citet{nakamura2025gpi} use generative AI representations to improve causal representation learning for text-as-treatment problems. \citet{gui2024causal} caution that LLM-simulated subjects can violate experimental unconfoundedness when treatment prompts unintentionally change unspecified covariates, and \citet{angelopoulos2024causal} show how A/B-test evidence can be used to causally align language-model outputs with downstream marketing outcomes. Relative to this literature, our contribution is not a general method for arbitrary textual treatments. We define a specific marketing decomposition --- retrieval versus expression in LLM-mediated recommendation --- and show how a modular factorial design identifies the corresponding component effects without relying on post hoc mediation through realized text features.

Finally, our estimands draw on the language of mediation and factorial experiments. Classical mediation work formalizes direct, indirect, and interaction-related decompositions \citep{robins1992mediation, pearl2001direct, imai2010identification, vanderweele2014three}. Factorial-experiment methods characterize main and interaction effects when multiple randomized factors are crossed \citep{egami2019causal, muralidharan2025factorial}. We use these tools in a policy-component experiment: the retrieval policy and expression policy are directly randomized, so the main empirical objects are cell-mean contrasts rather than text-as-mediator estimands.
```

---

## 3. Corrected `references.bib` entries

The following entries retain the current citation keys where possible so that the `.tex` file does not need widespread citation-key changes. The key names may contain old years or imperfect labels, but the bibliographic metadata inside each entry is corrected. Optional key-renaming can be done later.

```bibtex
@article{egami2018causal,
  title={How to Make Causal Inferences Using Texts},
  author={Egami, Naoki and Fong, Christian J. and Grimmer, Justin and Roberts, Margaret E. and Stewart, Brandon M.},
  journal={Science Advances},
  volume={8},
  number={42},
  pages={eabg2652},
  year={2022},
  doi={10.1126/sciadv.abg2652}
}

@misc{feldman2026latent,
  title={Causal Effect Estimation with Latent Textual Treatments},
  author={Feldman, Omri and Venugopal, Amar and Spiess, Jann and Feder, Amir},
  year={2026},
  eprint={2602.15730},
  archivePrefix={arXiv}
}

@misc{nakamura2025gpi,
  title={Causal Representation Learning with Generative Artificial Intelligence: Application to Texts as Treatments},
  author={Imai, Kosuke and Nakamura, Kentaro},
  year={2025},
  eprint={2410.00903},
  archivePrefix={arXiv}
}

@article{reisenbichler2022nlg,
  title={Frontiers: Supporting Content Marketing with Natural Language Generation},
  author={Reisenbichler, Martin and Reutterer, Thomas and Schweidel, David A. and Dan, Daniel},
  journal={Marketing Science},
  volume={41},
  number={3},
  pages={441--452},
  year={2022},
  doi={10.1287/mksc.2022.1354}
}

@article{angelopoulos2024causal,
  title={Causal Alignment: Augmenting Language Models with A/B Tests},
  author={Angelopoulos, Panagiotis and Lee, Kevin and Misra, Sanjog},
  journal={SSRN Electronic Journal},
  year={2024},
  doi={10.2139/ssrn.4781850}
}

@article{deldjoo2024,
  title={Understanding Biases in {ChatGPT}-Based Recommender Systems: Provider Fairness, Temporal Stability, and Recency},
  author={Deldjoo, Yashar},
  journal={ACM Transactions on Recommender Systems},
  year={2025},
  doi={10.1145/3690655}
}

@article{jannach2023crs,
  title={Evaluating Conversational Recommender Systems: A Landscape of Research},
  author={Jannach, Dietmar},
  journal={Artificial Intelligence Review},
  volume={56},
  number={3},
  pages={2365--2400},
  year={2023},
  doi={10.1007/s10462-022-10229-x}
}

@article{johnson2017ghost,
  title={Ghost Ads: Improving the Economics of Measuring Online Ad Effectiveness},
  author={Johnson, Garrett A. and Lewis, Randall A. and Nubbemeyer, Elmar I.},
  journal={Journal of Marketing Research},
  volume={54},
  number={6},
  pages={867--884},
  year={2017},
  doi={10.1509/jmr.15.0297}
}

@article{berman2018attribution,
  title={Beyond the Last Touch: Attribution in Online Advertising},
  author={Berman, Ron},
  journal={Marketing Science},
  volume={37},
  number={5},
  pages={771--792},
  year={2018},
  doi={10.1287/mksc.2018.1104}
}

@article{li2014attribution,
  title={Attributing Conversions in a Multichannel Online Marketing Environment: An Empirical Model and a Field Experiment},
  author={Li, Hongshuang and Kannan, P. K.},
  journal={Journal of Marketing Research},
  volume={51},
  number={1},
  pages={40--56},
  year={2014},
  doi={10.1509/jmr.13.0050}
}

@article{feder2022causal,
  title={Causal Inference in Natural Language Processing: Estimation, Prediction, Interpretation and Beyond},
  author={Feder, Amir and Keith, Katherine A. and Manzoor, Emaad and Pryzant, Reid and Sridhar, Dhanya and Wood-Doughty, Zach and Eisenstein, Jacob and Grimmer, Justin and Reichart, Roi and Roberts, Margaret E. and Stewart, Brandon M. and Veitch, Victor and Yang, Diyi},
  journal={Transactions of the Association for Computational Linguistics},
  volume={10},
  pages={1138--1158},
  year={2022},
  doi={10.1162/tacl_a_00511}
}

@article{shin2024messenger,
  title={The Role of Messenger in Advertising Content: Bayesian Persuasion Perspective},
  author={Shin, Jiwoong and Wang, Chi-Ying},
  journal={Marketing Science},
  volume={43},
  number={4},
  pages={840--862},
  year={2024},
  doi={10.1287/mksc.2022.0405}
}

@article{costello2024persuasion,
  title={Durably Reducing Conspiracy Beliefs Through Dialogues with {AI}},
  author={Costello, Thomas H. and Pennycook, Gordon and Rand, David G.},
  journal={Science},
  volume={385},
  number={6714},
  pages={eadq1814},
  year={2024},
  doi={10.1126/science.adq1814}
}

@misc{gui2024causal,
  title={The Challenge of Using {LLMs} to Simulate Human Behavior: A Causal Inference Perspective},
  author={Gui, George and Toubia, Olivier},
  year={2025},
  eprint={2312.15524},
  archivePrefix={arXiv}
}

@article{muralidharan2025factorial,
  title={Factorial Designs, Model Selection, and (Incorrect) Inference in Randomized Experiments},
  author={Muralidharan, Karthik and Romero, Mauricio and W{"u}thrich, Kaspar},
  journal={Review of Economics and Statistics},
  volume={107},
  number={3},
  pages={589--604},
  year={2025}
}

@book{wager2024causal,
  title={Causal Inference: A Statistical Learning Approach},
  author={Wager, Stefan},
  year={2024},
  publisher={Manuscript, Stanford University},
  note={Available at \url{https://web.stanford.edu/~swager/causal_inf_book.pdf}}
}

@article{swaminathan2015batch,
  title={Batch Learning from Logged Bandit Feedback through Counterfactual Risk Minimization},
  author={Swaminathan, Adith and Joachims, Thorsten},
  journal={Journal of Machine Learning Research},
  volume={16},
  number={52},
  pages={1731--1755},
  year={2015}
}

@inproceedings{schnabel2016recommendations,
  title={Recommendations as Treatments: Debiasing Learning and Evaluation},
  author={Schnabel, Tobias and Swaminathan, Adith and Singh, Ashudeep and Chandak, Navin and Joachims, Thorsten},
  booktitle={Proceedings of the 33rd International Conference on Machine Learning},
  series={Proceedings of Machine Learning Research},
  volume={48},
  pages={1670--1679},
  year={2016},
  publisher={PMLR}
}

@inproceedings{sharma2015estimating,
  title={Estimating the Causal Impact of Recommendation Systems from Observational Data},
  author={Sharma, Amit and Hofman, Jake M. and Watts, Duncan J.},
  booktitle={Proceedings of the Sixteenth ACM Conference on Economics and Computation},
  pages={453--470},
  year={2015},
  doi={10.1145/2764468.2764488}
}

@inproceedings{sato2020unbiased,
  title={Unbiased Learning for the Causal Effect of Recommendation},
  author={Sato, Masahiro and Takemori, Sho and Singh, Janmajay and Ohkuma, Tomoko},
  booktitle={Proceedings of the 14th ACM Conference on Recommender Systems},
  pages={378--387},
  year={2020},
  doi={10.1145/3383313.3412261}
}

@inproceedings{sato2021online,
  title={Online Evaluation Methods for the Causal Effect of Recommendations},
  author={Sato, Masahiro},
  booktitle={Proceedings of the 15th ACM Conference on Recommender Systems},
  pages={96--101},
  year={2021},
  doi={10.1145/3460231.3474235}
}

@inproceedings{ma2020off,
  title={Off-Policy Learning in Two-Stage Recommender Systems},
  author={Ma, Jiaqi and Zhao, Zhe and Yi, Xinyang and Yang, Ji and Chen, Minmin and Tang, Jiaxi and Hong, Lichan and Chi, Ed H.},
  booktitle={Proceedings of The Web Conference 2020},
  pages={463--473},
  year={2020},
  doi={10.1145/3366423.3380130}
}

@inproceedings{hron2021component,
  title={On Component Interactions in Two-Stage Recommender Systems},
  author={Hron, Jiri and Krauth, Karl and Jordan, Michael I. and Kilbertus, Niki},
  booktitle={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}

@inproceedings{hou2024large,
  title={Large Language Models Are Zero-Shot Rankers for Recommender Systems},
  author={Hou, Yupeng and Zhang, Junjie and Lin, Zihan and Lu, Hongyu and Xie, Ruobing and McAuley, Julian and Zhao, Wayne Xin},
  booktitle={Advances in Information Retrieval: 46th European Conference on Information Retrieval},
  year={2024},
  note={ECIR 2024; arXiv:2305.08845}
}

@misc{lichtenberg2024llm,
  title={Large Language Models as Recommender Systems: A Study of Popularity Bias},
  author={Lichtenberg, Jan Malte and Buchholz, Alexander and Schw{"o}bel, Pola},
  year={2024},
  eprint={2406.01285},
  archivePrefix={arXiv}
}

@inproceedings{xrec2024,
  title={{XR}ec: Large Language Models for Explainable Recommendation},
  author={Ma, Qiyao and Ren, Xubin and Huang, Chao},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2024},
  pages={391--402},
  year={2024},
  publisher={Association for Computational Linguistics},
  doi={10.18653/v1/2024.findings-emnlp.22}
}

@article{rahman2026persuasive,
  title={Persuasive Explanations for Recommender Systems: How Explanations Can Influence Users' Choices?},
  author={Rahman, S. M. Tahsinur and Siemon, Dominik and Ruotsalo, Tuukka},
  journal={International Journal of Human-Computer Studies},
  year={2025},
  doi={10.1016/j.ijhcs.2025.103654}
}

@article{robins1992mediation,
  title={Identifiability and Exchangeability for Direct and Indirect Effects},
  author={Robins, James M. and Greenland, Sander},
  journal={Epidemiology},
  volume={3},
  number={2},
  pages={143--155},
  year={1992},
  doi={10.1097/00001648-199203000-00013}
}

@inproceedings{pearl2001direct,
  title={Direct and Indirect Effects},
  author={Pearl, Judea},
  booktitle={Proceedings of the Seventeenth Conference on Uncertainty in Artificial Intelligence},
  pages={411--420},
  year={2001}
}

@article{imai2010identification,
  title={Identification, Inference and Sensitivity Analysis for Causal Mediation Effects},
  author={Imai, Kosuke and Keele, Luke and Yamamoto, Teppei},
  journal={Statistical Science},
  volume={25},
  number={1},
  pages={51--71},
  year={2010},
  doi={10.1214/10-STS321}
}

@article{vanderweele2014three,
  title={A Unification of Mediation and Interaction: A Four-Way Decomposition},
  author={VanderWeele, Tyler J.},
  journal={Epidemiology},
  volume={25},
  number={5},
  pages={749--761},
  year={2014},
  doi={10.1097/EDE.0000000000000121}
}

@article{hou2023bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}

@misc{hou2024amazon,
  title={Amazon Reviews 2023 Dataset},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  year={2024},
  note={Available at \url{https://amazon-reviews-2023.github.io}}
}

@article{egami2019causal,
  title={Causal Interaction in Factorial Experiments: Application to Conjoint Analysis},
  author={Egami, Naoki and Imai, Kosuke},
  journal={Journal of the American Statistical Association},
  volume={114},
  number={526},
  pages={529--540},
  year={2019},
  doi={10.1080/01621459.2018.1476246}
}
```

Notes:

- The key `nakamura2025gpi` is misleading because the first author is Kosuke Imai. Keep it only to avoid changing `.tex`; otherwise rename to `imai2025causal`.
- The key `egami2018causal` is misleading if citing the published Science Advances version. Keep it only to avoid changing `.tex`; otherwise rename to `egami2022text`.
- The key `gui2024causal` is misleading because the paper first appeared on arXiv in 2023 and has later revisions. Keep it only to avoid changing `.tex`; otherwise rename to `gui2025challenge` or `gui2023challenge` depending on which version you cite.
- `rahman2026persuasive` should probably be renamed to `rahman2025persuasive`, but the old key can be retained for now.
- `johnson2017ghost`, `berman2018attribution`, and `li2014attribution` are corrected above but should be removed if they remain uncited.

---

## 4. Revision plan for overclaiming and structural issues

The following are not “placeholder-result” problems. They are claim-strength and structure issues that should be revised even after the main simulation finishes.

### 4.1 Define the estimand more precisely

Current recurring claim:

> The design identifies the retrieval effect, persuasion effect, and interaction effect.

Problem:

This is true only for the experimentally manipulated policies and feasible recommendation support. It is not a universal decomposition of all possible retrieval and expression mechanisms.

Revision direction:

> The design identifies the effects of the assigned retrieval and expression policies over the feasible support induced by the experimental system.

Apply this wherever the draft says “separately identify these two channels” or “determine which layer drives response.”

### 4.2 Moderate “negligible interaction” language

Current claim:

> The pooled interaction is statistically and economically zero.

Problem:

This is too strong, especially with 30 clusters and category-specific interactions with different signs. It is fine as a diagnostic, but not as a universal architecture result.

Revision direction:

> In the architecture diagnostic, we find no pooled evidence of a systematic retrieval--expression interaction; the estimated pooled interaction is close to zero, although category-specific estimates remain noisy.

Use “supports the feasibility of the factored design” rather than “proves/additively justifies the factored design.”

### 4.3 Replace “wrong question” with “incomplete question”

Current claim:

> Standard prompt A/B tests answer the wrong question when the decomposition matters.

Problem:

This is rhetorically strong and useful, but it may annoy reviewers because prompt A/B tests correctly identify the total policy effect. The problem is not wrong identification; the problem is that the total effect is insufficient for resource-allocation decisions.

Revision direction:

> Standard prompt A/B tests answer the total-effect question, but not the component-allocation question.

Or:

> A prompt A/B test is sufficient for deciding whether the bundled policy works, but insufficient for deciding whether the lift came from retrieval, expression, or their interaction.

### 4.4 Clarify that this is not mediation through realized text

Current risk:

The paper sometimes sounds like a mediation paper because it uses words like “channels,” “components,” “direct,” “indirect,” and “decomposition.”

Problem:

Reviewers may ask why standard mediation assumptions are not discussed in detail.

Revision direction:

Add or repeat one sentence in the framework and related literature:

> The design is not a post hoc mediation analysis of realized text. It is a factorial experiment over policy components: retrieval policy and expression policy are assigned by the experimenter before recommendation generation.

### 4.5 Be careful with “deployable on any platform”

Current claim:

> The modular design is deployable on any platform that already issues separable retrieval and expression calls.

Problem:

“Any platform” is too broad. Some platforms may not be able to lock the selected item, prevent expression-stage product substitution, or randomize policies independently.

Revision direction:

> The design is deployable on platforms that can separately assign retrieval and expression policies, lock the selected product before text generation, and verify that the expression stage does not change the nominated item.

### 4.6 Do not overstate the architecture diagnostic as external validity

Current risk:

The diagnostic uses a local LLM and a limited set of products/categories. It should support feasibility, not general validity.

Revision direction:

> The diagnostic is an internal architecture check for the modular design under the local model and product categories studied. It does not establish that retrieval--expression interactions are generally small for all frontier LLMs or product domains.

### 4.7 Strengthen contribution language where it is too apologetic

Current tone issue:

The draft repeatedly says the contribution is “not novel econometric theory.” This is true but over-repeated. Once is enough.

Revision direction:

Keep one statement such as:

> The econometric identification follows standard factorial-design logic; the contribution is defining the marketing-relevant policy components and implementing the corresponding experiment for LLM-mediated recommendation.

Then stop apologizing. The paper should sound confident that the contribution is the causal object/design, not a new theorem.

### 4.8 Structural plan for the introduction

Recommended introduction structure:

1. LLM recommenders bundle item selection and generated language.
2. A bundled prompt A/B test identifies total effect but not the component that caused the lift.
3. This matters managerially because retrieval and expression imply different investments.
4. Introduce modular `2 x 2` design.
5. State estimands carefully: retrieval-policy effect, expression-policy effect, interaction effect, over feasible support.
6. State architecture diagnostic modestly.
7. State main history-shock simulation as the central empirical exercise once results arrive.
8. State naive realized-expression regression bias as an additional warning against post hoc text regressions.

### 4.9 Structural plan for related literature

Recommended related-literature structure:

1. Causal recommendation and policy learning.
2. Two-stage / LLM recommender architectures.
3. Recommendation explanations and AI persuasion.
4. Causal inference with text and generative AI.
5. Final bridge to mediation/factorial methods.

Avoid mixing architecture and causal-text papers in the same paragraph.

---

## 5. Immediate Claude task list

Claude should do the following in order:

1. Replace the current `Related Literature` section with the replacement section above.
2. Replace the current `references.bib` contents with the corrected entries above, unless the user wants to keep unused references out.
3. Remove uncited bibliography entries if they remain uncited after the literature-review replacement.
4. Apply claim-strength edits from Section 4, but do not revise the placeholder main-simulation result sections yet.
5. Recompile LaTeX and check for missing citation keys or bibliography warnings.
6. Report any remaining citation warnings, overfull boxes, or unresolved references.
