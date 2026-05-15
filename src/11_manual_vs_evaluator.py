"""Compare manual coding labels with LLM evaluator scores.

Reads:
  - results/diagnostics/manual_coding_sample.csv (with manual_* columns filled in)
  - results/diagnostics/manual_coding_key.csv (row_id -> row_key mapping)
  - results/diagnostics/evaluator_scores.csv

Produces:
  - results/diagnostics/manual_vs_evaluator_validation.csv
  - results/diagnostics/manual_vs_evaluator_validation.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.data_io import ROOT

DIAG_DIR = ROOT / "results" / "diagnostics"

SCALES = ["fit_specificity", "persuasive_intensity", "tradeoff_disclosure"]


def main():
    manual = pd.read_csv(DIAG_DIR / "manual_coding_sample.csv")
    key = pd.read_csv(DIAG_DIR / "manual_coding_key.csv")
    eval_scores = pd.read_csv(DIAG_DIR / "evaluator_scores.csv")

    merged = manual[["row_id"] + [f"manual_{s}" for s in SCALES]].merge(key[["row_id", "row_key"]], on="row_id")
    merged = merged.merge(eval_scores[["row_key"] + SCALES], on="row_key", suffixes=("_manual", "_eval"))

    results = []
    for scale in SCALES:
        m_col = f"manual_{scale}"
        e_col = scale

        valid = merged[[m_col, e_col]].dropna()
        valid = valid[(valid[m_col] >= 1) & (valid[m_col] <= 7)]
        valid[m_col] = valid[m_col].astype(int)
        n = len(valid)

        if n < 5:
            results.append({
                "scale": scale,
                "n": n,
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "mean_manual": np.nan,
                "mean_eval": np.nan,
                "mean_diff": np.nan,
                "pass_threshold": False,
            })
            continue

        sp = stats.spearmanr(valid[m_col], valid[e_col])
        pr = stats.pearsonr(valid[m_col], valid[e_col])

        results.append({
            "scale": scale,
            "n": n,
            "spearman_rho": round(sp.statistic, 4),
            "spearman_p": round(sp.pvalue, 4),
            "pearson_r": round(pr.statistic, 4),
            "pearson_p": round(pr.pvalue, 4),
            "mean_manual": round(valid[m_col].mean(), 2),
            "mean_eval": round(valid[e_col].mean(), 2),
            "mean_diff": round(valid[e_col].mean() - valid[m_col].mean(), 2),
            "pass_threshold": bool(sp.statistic >= 0.5),
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(DIAG_DIR / "manual_vs_evaluator_validation.csv", index=False)

    detail = merged.copy()
    detail.to_csv(DIAG_DIR / "manual_vs_evaluator_detail.csv", index=False)

    all_pass = all(r["pass_threshold"] for r in results if not np.isnan(r.get("spearman_rho", np.nan)))
    n_pass = sum(1 for r in results if r["pass_threshold"])

    md = f"""# Manual vs Evaluator Validation

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
**N coded rows:** {len(merged)}

## Summary

| Scale | n | Spearman rho | p | Pearson r | p | Mean(manual) | Mean(eval) | Diff | Pass? |
|-------|---|-------------|---|-----------|---|-------------|-----------|------|-------|
"""
    for r in results:
        status = "YES" if r["pass_threshold"] else "NO"
        md += (f"| {r['scale']} | {r['n']} | {r['spearman_rho']} | {r['spearman_p']} | "
               f"{r['pearson_r']} | {r['pearson_p']} | {r['mean_manual']} | {r['mean_eval']} | "
               f"{r['mean_diff']} | {status} |\n")

    md += f"""
## Interpretation

Acceptance threshold: Spearman rho >= 0.5 per scale.

Scales passing: {n_pass}/{len(SCALES)}
All pass: {'YES' if all_pass else 'NO'}

## Row-level Comparison

"""
    for scale in SCALES:
        m_col = f"manual_{scale}"
        e_col = scale
        md += f"### {scale}\n\n"
        md += "| row_id | manual | eval | diff |\n"
        md += "|--------|--------|------|------|\n"
        for _, row in merged.iterrows():
            m = row.get(m_col)
            e = row.get(e_col)
            if pd.notna(m) and pd.notna(e):
                md += f"| {row['row_id']} | {int(m)} | {int(e)} | {int(e) - int(m):+d} |\n"
        md += "\n"

    with open(DIAG_DIR / "manual_vs_evaluator_validation.md", "w") as f:
        f.write(md)

    print(res_df.to_string(index=False))
    print(f"\nAll pass: {'YES' if all_pass else 'NO'}")
    print(f"Saved to {DIAG_DIR}")


if __name__ == "__main__":
    main()
