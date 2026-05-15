
# Project workflow

This project uses persistent local project memory.

Key files:
- `temp/prompt_history.md`
- `temp/change_log.md`

## Setup

If either file does not exist, create it (including the `temp/` directory) before proceeding.

## Required recording — do this every session

### 1. Prompt + response log (`temp/prompt_history.md`)

After EVERY exchange with the user, append one line in this format:

```
[YYYY-MM-DD] Prompt: <verbatim or very close paraphrase of user prompt> | Response summary: <1–2 sentence summary of what Claude did or answered>
```

Write this at the END of your response, after all other work is done.

### 2. Change log (`temp/change_log.md`)

After ANY file is created or modified, append one line per changed file in this format:

```
[YYYY-MM-DD] File: <relative path> | Change: <brief description of what was changed and why>
```

Write this immediately after making the file changes, before finishing your response.

## General workflow

- Review recent entries in both files before planning or making major edits.
- Keep changes aligned with the recent history of the project.
- Prefer concise summaries of prior work rather than repeating long logs.
- When revising files, keep changes minimal and well-targeted.
- When resuming work after a break, recover recent goals, modified files, and unfinished tasks from the log files.

--------------------------------------------------------------------------

# Target journal and project goals

## Primary target: Quantitative Marketing and Economics (QME)

QME publishes research at the intersection of Marketing, Economics, and Statistics, focused on important applied problems using quantitative approaches. It covers consumer preferences, demand, decision-making, firm strategy, pricing, promotion, targeting, product design/positioning, and channels. Methods include applied economic theory, econometrics, statistical methods, and empirical research with primary, secondary, or experimental data.

**What QME reviewers expect:**
- Precise, well-stated assumptions — overclaimed or trivially obvious propositions will be rejected.
- Formal identification results that are non-obvious or at least clearly motivated by a substantive problem.
- Computational or simulation evidence is acceptable for a working paper, but must demonstrate that the identification problem is quantitatively meaningful (not just algebraically possible).
- Clear connection between the formal framework and a real marketing decision a firm would face.
- Honest scope: state what the paper does and does not do. QME respects modesty over overselling.

## Secondary target: Marketing Science (INFORMS)

Marketing Science is the premier journal for empirical and theoretical quantitative marketing research. It accepts diverse approaches: mathematical modeling, experiments, aggregate data, deductive analyses, and behaviorally oriented papers. Manuscripts must make significant contributions — substantive findings, modeling improvements, theoretical developments, methodological advances, or tests of existing theories. The best papers provide knowledge that allows a target audience (managers, policymakers, researchers) to make superior decisions.

**What Marketing Science reviewers additionally expect beyond QME:**
- Stronger empirical evidence — a human-subject or field experiment is typically needed, not just simulations.
- Clear managerial implications with teeth (e.g., "for complex products, 70% of the lift comes from retrieval; for simple products, expression dominates").
- Sharp positioning against the closest competitors (Nakamura & Imai 2024, Feldman et al. 2026).
- Broader audience appeal — the paper should be readable by non-specialists in causal inference.

## Project contribution and scope

This project develops a causal framework for decomposing LLM product recommendation effects into **retrieval** (what is recommended) and **persuasion/expression** (how it is recommended). The intended contributions are:

1. **A novel causal framework** for LLM-mediated product recommendation that defines retrieval, expression, and interaction effects as interventional policy components a firm can deploy.
2. **A practical computational study** using online LLMs that reveals empirical findings about how LLM recommenders jointly shift product selection and expression, and demonstrates that naive analyses can be biased while the proposed factorized design recovers correct causal components.
3. **A human-subject experiment** (later phase) to validate the framework with real consumer outcomes.

## Current milestone

**Immediate goal:** Write a working paper (theory + simulation, no experiment section) for submission to the **QME conference by May 15, 2026**. The paper should include:
- Formal theory with precise assumptions, estimands, and identification results
- Computational evidence from LLM-generated recommendations showing stochasticity, naive-estimator bias, and factorized-design recovery
- A roadmap for human-subject validation (discussed but not executed)

## Writing and framing guidelines for this project

- Frame estimands as **deployable firm interventions** ("use the new item selector but keep the old expression style"), not abstract kernel manipulations.
- Use **interventional decomposition** language, not natural-mediation language.
- Do NOT claim that LLMs break causal inference or that standard A/B tests are confounded.
- Do NOT claim to be the first causal framework for text treatments — the contribution is narrower: retrieval-vs-persuasion decomposition in product recommendation.
- Do NOT rely on digital twins as final consumer evidence — use them only for design validation and estimator stress-testing.
- Proposition 2 (non-identification) should state precise sufficient conditions, not overclaim impossibility. The factorized design satisfies those conditions by construction.
- Keep theory modest and honest — the algebra is simple; the value is in defining the right marketing objects and an implementable experiment.

---