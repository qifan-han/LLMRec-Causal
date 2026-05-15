"""Generate summary report for Phase 2 LLM simulation results.

Reads outputs from 06_llm_simulation.py and 07_estimate_llm.py,
produces results/report_llm.md and results/report_llm.html.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.data_io import ROOT

OUTPUT_DIR = ROOT / "data" / "llm_sim"
RESULTS_DIR = ROOT / "results"
TABLE_DIR = RESULTS_DIR / "tables"


def _load_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / name)


def _load_json(name: str) -> dict:
    with open(TABLE_DIR / name) as f:
        return json.load(f)


def _load_seeds() -> dict:
    with open(OUTPUT_DIR / "seeds.json") as f:
        return json.load(f)


def generate_report() -> str:
    seeds = _load_seeds()
    t1 = _load_table("llm_one_shot_total_effect.csv")
    t2 = _load_table("llm_decomposition.csv")
    t3 = _load_table("llm_naive_vs_oracle.csv")
    cal = _load_json("llm_calibration.json")

    one_shot = pd.read_csv(OUTPUT_DIR / "one_shot_llm.csv")
    modular = pd.read_csv(OUTPUT_DIR / "modular_llm.csv")
    retrieval = pd.read_csv(OUTPUT_DIR / "retrieval_llm.csv")

    today = date.today().isoformat()

    # Retrieval shift summary
    ret_summary_lines = []
    for cat in sorted(retrieval["category"].unique()):
        sub = retrieval[retrieval["category"] == cat]
        d0 = sub[sub["q"] == 0]["product_id"].value_counts(normalize=True)
        d1 = sub[sub["q"] == 1]["product_id"].value_counts(normalize=True)
        all_pids = set(d0.index) | set(d1.index)
        tvd = 0.5 * sum(abs(d0.get(p, 0) - d1.get(p, 0)) for p in all_pids)
        ret_summary_lines.append(f"| {cat} | {tvd:.3f} | {len(sub)//2} |")

    # Expression intensity summary
    e_r0 = modular[modular["r"] == 0]["expression_intensity"]
    e_r1 = modular[modular["r"] == 1]["expression_intensity"]

    # Pooled results from tables
    pooled_os = t1[t1["category"] == "POOLED"].iloc[0] if "POOLED" in t1["category"].values else None
    pooled_ret = t2[(t2["category"] == "POOLED") & (t2["estimand"] == "retrieval")]
    pooled_per = t2[(t2["category"] == "POOLED") & (t2["estimand"] == "persuasion")]
    pooled_int = t2[(t2["category"] == "POOLED") & (t2["estimand"] == "interaction")]

    md = f"""# Phase 2 LLM Simulation Report

**Date:** {today}
**Model:** {seeds.get('model', 'qwen2.5:14b')}
**N consumers per category:** {seeds.get('n_consumers', 50)}
**Master seed:** {seeds.get('master_seed', 'N/A')}
**Temperature:** {seeds.get('temperature', 0.7)}

## 1. Overview

This report summarizes the results of the Phase 2 prompt-engineering pass, which replaces
the hand-specified retrieval and expression kernels from Phase 1 with actual LLM calls via
Ollama. The structural DGP outcome layer (logistic specification) remains unchanged.

**Data produced:**
- One-shot: {len(one_shot)} rows ({len(one_shot)//2} consumers x 2 conditions)
- Retrieval: {len(retrieval)} rows ({len(retrieval)//2} consumers x 2 conditions)
- Modular: {len(modular)} rows ({len(modular)//4} consumers x 4 cells)

## 2. Sub-experiment A: One-shot Total Effect

{t1.to_markdown(index=False)}

**Interpretation:** The one-shot total effect captures the combined impact of switching
from baseline to brand-forward+persuasive prompting. This parallels Remark 1 in the paper.

## 3. Sub-experiment B: Retrieval Shift

| Category | TVD (q=0 vs q=1) | N per condition |
|----------|-------------------|-----------------|
{chr(10).join(ret_summary_lines)}

**Interpretation:** TVD measures how much the brand-forward instruction shifts the product
distribution. Higher TVD = stronger retrieval effect.

## 4. Sub-experiment C: Modular Decomposition

{t2.to_markdown(index=False)}

**Interpretation:** The 2x2 factorial design identifies retrieval (Prop 1), persuasion (Prop 1),
and interaction (Prop 3) effects separately.

## 5. Expression Intensity

| Condition | Mean E | Std E | N |
|-----------|--------|-------|---|
| r=0 (neutral) | {e_r0.mean():.4f} | {e_r0.std():.4f} | {len(e_r0)} |
| r=1 (persuasive) | {e_r1.mean():.4f} | {e_r1.std():.4f} | {len(e_r1)} |

**Gap (r=1 - r=0):** {e_r1.mean() - e_r0.mean():.4f}

## 6. Naive vs Oracle vs Modular

{t3.to_markdown(index=False)}

**Interpretation:** The naive estimator regresses Y on E without controlling for Q_std,
which is the omitted-variable bias channel identified in Proposition 2. The oracle adds
Q_std as a control (infeasible in practice). The modular cell contrast identifies the
persuasion effect by design (Proposition 3).

## 7. Calibration

| Parameter | Value |
|-----------|-------|
| corr(E, Q_std) | {cal.get('corr_E_Qstd', 'N/A')} |
| lambda_fit_hat | {cal.get('lambda_fit_hat', 'N/A')} |
| lambda_fit_se | {cal.get('lambda_fit_se', 'N/A')} |
| lambda_fit_t | {cal.get('lambda_fit_t', 'N/A')} |
| sigma_R_hat | {cal.get('sigma_R_hat', 'N/A')} |

**lambda_fit_hat** is the slope of expression intensity on fit score in the (q=0, r=0)
baseline cell. This is the empirical analogue of the structural parameter lambda_fit
from Phase 1. A positive value means the LLM generates more enthusiastic text for
better-fitting products — the endogeneity channel the paper studies.

**sigma_R_hat** is the residual standard deviation of expression intensity after removing
cell means. It indexes the stochasticity of LLM expression.

## 8. Comparison with Phase 1

| Metric | Phase 1 (structural) | Phase 2 (LLM) |
|--------|---------------------|---------------|
| Naive bias (pooled) | +0.050 | See Table 6 |
| Modular persuasion | +0.032 | {pooled_per.iloc[0]['estimate'] if len(pooled_per) > 0 else 'N/A'} |
| corr(E, Q_std) | 0.806 | {cal.get('corr_E_Qstd', 'N/A')} |

## 9. Issues and Observations

*To be filled in based on results.*

## 10. Next Steps

1. If expression-intensity gap (r=1 vs r=0) is too small, revise prompts.
2. If corr(E, Q_std) is near zero, the endogeneity channel may not operate for this model.
3. Scale to full 1000 consumers per category on vLLM for the paper.
4. Add within-prompt repetition (B=3) for sigma_R_hat calibration.
"""
    return md


def main():
    md = generate_report()

    md_path = RESULTS_DIR / "report_llm.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved -> {md_path}")

    # HTML version
    try:
        import markdown
        html_body = markdown.markdown(md, extensions=["tables"])
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Phase 2 LLM Simulation Report</title>
<style>
body {{ font-family: Georgia, serif; max-width: 900px; margin: 2em auto; line-height: 1.6; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
th {{ background: #f5f5f5; }}
code {{ background: #f0f0f0; padding: 2px 4px; }}
h1 {{ border-bottom: 2px solid #333; }}
h2 {{ color: #333; margin-top: 2em; }}
</style></head><body>
{html_body}
</body></html>"""
        html_path = RESULTS_DIR / "report_llm.html"
        with open(html_path, "w") as f:
            f.write(html)
        print(f"Saved -> {html_path}")
    except ImportError:
        print("markdown not installed, skipping HTML report")


if __name__ == "__main__":
    main()
