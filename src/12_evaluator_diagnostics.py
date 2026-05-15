"""Diagnostic correlations and regressions using LLM evaluator scores.

Reads evaluator_scores.csv, computes correlations with Q_std by cell/category,
and runs category-FE regressions.

Produces:
  - results/diagnostics/evaluator_diagnostic_correlations.csv
  - results/diagnostics/evaluator_score_by_cell.csv
  - results/diagnostics/evaluator_regressions.csv
  - results/diagnostics/evaluator_step1_report.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    import statsmodels.api as sm
except ImportError:
    sm = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.data_io import ROOT

DIAG_DIR = ROOT / "results" / "diagnostics"
SCALES = ["fit_specificity", "persuasive_intensity", "tradeoff_disclosure"]


def _corr_row(label: str, x: pd.Series, y: pd.Series) -> dict:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(valid)
    if n < 5:
        return {"slice": label, "n": n, "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_rho": np.nan, "spearman_p": np.nan}
    pr = stats.pearsonr(valid["x"], valid["y"])
    sp = stats.spearmanr(valid["x"], valid["y"])
    return {
        "slice": label, "n": n,
        "pearson_r": round(pr.statistic, 4), "pearson_p": round(pr.pvalue, 4),
        "spearman_rho": round(sp.statistic, 4), "spearman_p": round(sp.pvalue, 4),
    }


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scale in SCALES:
        rows.append({"scale": scale, **_corr_row("overall", df[scale], df["Q_std"])})
        for cat in sorted(df["category"].unique()):
            sub = df[df["category"] == cat]
            rows.append({"scale": scale, **_corr_row(f"cat={cat}", sub[scale], sub["Q_std"])})
        for q in [0, 1]:
            for r in [0, 1]:
                cell = df[(df["q"] == q) & (df["r"] == r)]
                rows.append({"scale": scale, **_corr_row(f"q={q},r={r}", cell[scale], cell["Q_std"])})
        for r in [0, 1]:
            sub = df[df["r"] == r]
            rows.append({"scale": scale, **_corr_row(f"r={r}", sub[scale], sub["Q_std"])})
    return pd.DataFrame(rows)


def compute_cell_means(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q in [0, 1]:
        for r in [0, 1]:
            cell = df[(df["q"] == q) & (df["r"] == r)]
            row = {"q": q, "r": r, "n": len(cell)}
            for scale in SCALES:
                row[f"{scale}_mean"] = round(cell[scale].mean(), 3)
                row[f"{scale}_sd"] = round(cell[scale].std(), 3)
            rows.append(row)
    total_row = {"q": "ALL", "r": "ALL", "n": len(df)}
    for scale in SCALES:
        total_row[f"{scale}_mean"] = round(df[scale].mean(), 3)
        total_row[f"{scale}_sd"] = round(df[scale].std(), 3)
    rows.append(total_row)
    return pd.DataFrame(rows)


def run_regressions(df: pd.DataFrame) -> pd.DataFrame:
    if sm is None:
        print("  statsmodels not available, skipping regressions")
        return pd.DataFrame()

    cat_dummies = pd.get_dummies(df["category"], drop_first=True, dtype=float)
    results = []

    for scale in SCALES:
        # Model 1: score ~ Q_std + C(category)
        X1 = pd.concat([df[["Q_std"]].reset_index(drop=True),
                         cat_dummies.reset_index(drop=True)], axis=1)
        X1 = sm.add_constant(X1)
        y = df[scale].reset_index(drop=True)
        try:
            mod1 = sm.OLS(y, X1).fit(cov_type="HC1")
            results.append({
                "scale": scale,
                "model": "score ~ Q_std + C(cat)",
                "coef_Q_std": round(mod1.params["Q_std"], 4),
                "se_Q_std": round(mod1.bse["Q_std"], 4),
                "t_Q_std": round(mod1.tvalues["Q_std"], 4),
                "p_Q_std": round(mod1.pvalues["Q_std"], 4),
                "R2": round(mod1.rsquared, 4),
                "n": int(mod1.nobs),
            })
        except Exception as e:
            print(f"  Regression failed for {scale} model 1: {e}")

        # Model 2: score ~ Q_std + q + r + q*r + C(category)
        X2_data = df[["Q_std"]].copy().reset_index(drop=True)
        X2_data["q"] = df["q"].values.astype(float)
        X2_data["r"] = df["r"].values.astype(float)
        X2_data["q_x_r"] = X2_data["q"] * X2_data["r"]
        X2 = pd.concat([X2_data, cat_dummies.reset_index(drop=True)], axis=1)
        X2 = sm.add_constant(X2)
        try:
            mod2 = sm.OLS(y, X2).fit(cov_type="HC1")
            results.append({
                "scale": scale,
                "model": "score ~ Q_std + q + r + q*r + C(cat)",
                "coef_Q_std": round(mod2.params["Q_std"], 4),
                "se_Q_std": round(mod2.bse["Q_std"], 4),
                "t_Q_std": round(mod2.tvalues["Q_std"], 4),
                "p_Q_std": round(mod2.pvalues["Q_std"], 4),
                "R2": round(mod2.rsquared, 4),
                "n": int(mod2.nobs),
                "coef_r": round(mod2.params.get("r", np.nan), 4),
                "t_r": round(mod2.tvalues.get("r", np.nan), 4),
                "coef_q": round(mod2.params.get("q", np.nan), 4),
                "t_q": round(mod2.tvalues.get("q", np.nan), 4),
            })
        except Exception as e:
            print(f"  Regression failed for {scale} model 2: {e}")

    return pd.DataFrame(results)


def write_report(corr_df, cell_df, reg_df, df):
    # Determine recommendation
    corr_q0r0 = corr_df[(corr_df["slice"] == "q=0,r=0")]

    fit_corr_baseline = corr_q0r0[corr_q0r0["scale"] == "fit_specificity"]
    fit_r = float(fit_corr_baseline["pearson_r"].iloc[0]) if len(fit_corr_baseline) > 0 else np.nan

    # Check r effect on persuasive_intensity
    pi_r0 = cell_df[cell_df["r"] == 0]["persuasive_intensity_mean"].mean() if "persuasive_intensity_mean" in cell_df.columns else np.nan
    pi_r1 = cell_df[cell_df["r"] == 1]["persuasive_intensity_mean"].mean() if "persuasive_intensity_mean" in cell_df.columns else np.nan

    td_r0 = cell_df[cell_df["r"] == 0]["tradeoff_disclosure_mean"].mean() if "tradeoff_disclosure_mean" in cell_df.columns else np.nan
    td_r1 = cell_df[cell_df["r"] == 1]["tradeoff_disclosure_mean"].mean() if "tradeoff_disclosure_mean" in cell_df.columns else np.nan

    # Load validation results if available
    val_path = DIAG_DIR / "manual_vs_evaluator_validation.csv"
    val_df = pd.read_csv(val_path) if val_path.exists() else None
    val_pass = False
    if val_df is not None:
        val_pass = val_df["pass_threshold"].all()

    md = f"""# Evaluator Step 1 Diagnostic Report

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}
**N rows:** {len(df)}
**Evaluator model:** qwen2.5:14b, temperature=0

---

## 1. Did the evaluator produce valid JSON reliably?

Parse failures are logged in data/llm_eval/raw/. Check evaluator_scores.csv row count
vs input (120 rows expected).

Rows scored: {len(df)}

## 2. Does evaluator scoring agree with manual coding?

{"Validation file found." if val_df is not None else "Validation not yet run (manual coding pending)."}

{val_df.to_markdown(index=False) if val_df is not None else "*Run src/11_manual_vs_evaluator.py after manual coding.*"}

Overall pass: {"YES" if val_pass else "NO / PENDING"}

## 3. Which scales are usable?

{_scale_usability(val_df) if val_df is not None else "*Pending manual validation.*"}

## 4-6. Correlations with Q_std

{corr_df.to_markdown(index=False)}

## 7. Does r=1 increase persuasive_intensity?

| Cell | persuasive_intensity mean |
|------|--------------------------|
| r=0 | {pi_r0:.3f} |
| r=1 | {pi_r1:.3f} |
| diff | {(pi_r1 - pi_r0):.3f} |

{"YES — r=1 increases persuasive_intensity." if pd.notna(pi_r1) and pd.notna(pi_r0) and pi_r1 > pi_r0 + 0.3 else "Weak or no effect."}

## 8. Does r=1 reduce tradeoff_disclosure?

| Cell | tradeoff_disclosure mean |
|------|--------------------------|
| r=0 | {td_r0:.3f} |
| r=1 | {td_r1:.3f} |
| diff | {(td_r1 - td_r0):.3f} |

{"YES — r=1 reduces tradeoff_disclosure." if pd.notna(td_r1) and pd.notna(td_r0) and td_r1 < td_r0 - 0.3 else "Weak or no effect."}

## 9. Does q change expression scores holding r fixed?

"""
    for scale in SCALES:
        q0r0 = cell_df[(cell_df["q"] == 0) & (cell_df["r"] == 0)]
        q1r0 = cell_df[(cell_df["q"] == 1) & (cell_df["r"] == 0)]
        col = f"{scale}_mean"
        v00 = float(q0r0[col].iloc[0]) if len(q0r0) > 0 and col in q0r0.columns else np.nan
        v10 = float(q1r0[col].iloc[0]) if len(q1r0) > 0 and col in q1r0.columns else np.nan
        md += f"- {scale}: q=0,r=0 mean={v00:.3f}, q=1,r=0 mean={v10:.3f}, diff={v10-v00:+.3f}\n"

    md += f"""
## Cell means

{cell_df.to_markdown(index=False)}

## Regressions

{reg_df.to_markdown(index=False) if len(reg_df) > 0 else "*Statsmodels not available.*"}

## 10. Recommendation

"""
    md += _recommendation(corr_df, cell_df, reg_df, val_df, fit_r)

    report_path = DIAG_DIR / "evaluator_step1_report.md"
    with open(report_path, "w") as f:
        f.write(md)
    print(f"Report: {report_path}")


def _scale_usability(val_df):
    if val_df is None:
        return "*Pending.*"
    lines = []
    for _, row in val_df.iterrows():
        status = "USABLE" if row["pass_threshold"] else "NOT VALIDATED"
        lines.append(f"- {row['scale']}: Spearman rho={row['spearman_rho']:.3f} -> {status}")
    return "\n".join(lines)


def _recommendation(corr_df, cell_df, reg_df, val_df, fit_corr_q0r0):
    if val_df is not None and not val_df["pass_threshold"].all():
        n_fail = (~val_df["pass_threshold"]).sum()
        return (
            f"**Recommendation C:** Evaluator does not fully validate against manual coding "
            f"({n_fail} scale(s) below rho=0.5 threshold). Fix the evaluator/rubric before "
            f"interpreting diagnostic correlations.\n"
        )

    has_fit_signal = pd.notna(fit_corr_q0r0) and abs(fit_corr_q0r0) > 0.2
    has_r_effect = False
    r0_rows = cell_df[cell_df["r"] == 0]
    r1_rows = cell_df[cell_df["r"] == 1]
    if len(r0_rows) > 0 and len(r1_rows) > 0:
        pi_diff = r1_rows["persuasive_intensity_mean"].mean() - r0_rows["persuasive_intensity_mean"].mean()
        has_r_effect = pi_diff > 0.5

    if has_fit_signal and has_r_effect:
        return (
            f"**Recommendation A:** Evaluator validates and current texts show meaningful "
            f"mechanism signal. Scale the corpus cautiously to 200-300 consumers per category.\n\n"
            f"Evidence: fit_specificity corr with Q_std in baseline cell = {fit_corr_q0r0:.3f}; "
            f"r=1 shifts persuasive_intensity.\n"
        )
    elif not has_fit_signal and has_r_effect:
        return (
            f"**Recommendation B:** Evaluator validates but current texts show weak fit-expression "
            f"signal (corr={fit_corr_q0r0:.3f} in baseline cell). The problem is generation, not "
            f"measurement. Revise generation prompts before scaling.\n"
        )
    elif has_fit_signal and not has_r_effect:
        return (
            f"**Recommendation D:** Mixed result. fit_specificity correlates with Q_std "
            f"(corr={fit_corr_q0r0:.3f}) but r=1 does not shift persuasive_intensity meaningfully. "
            f"Use fit_specificity as main expression measure; revise persuasive prompt.\n"
        )
    else:
        return (
            f"**Recommendation B:** Both fit-expression signal and prompt manipulation are weak. "
            f"Revise generation prompts before scaling. Current texts lack meaningful variation "
            f"on the dimensions the paper needs.\n"
        )


def main():
    df = pd.read_csv(DIAG_DIR / "evaluator_scores.csv")
    print(f"Loaded {len(df)} evaluator scores")

    print("\n=== Correlations ===")
    corr_df = compute_correlations(df)
    corr_df.to_csv(DIAG_DIR / "evaluator_diagnostic_correlations.csv", index=False)
    print(corr_df.to_string(index=False))

    print("\n=== Cell means ===")
    cell_df = compute_cell_means(df)
    cell_df.to_csv(DIAG_DIR / "evaluator_score_by_cell.csv", index=False)
    print(cell_df.to_string(index=False))

    print("\n=== Regressions ===")
    reg_df = run_regressions(df)
    if len(reg_df) > 0:
        reg_df.to_csv(DIAG_DIR / "evaluator_regressions.csv", index=False)
        print(reg_df.to_string(index=False))

    print("\n=== Report ===")
    write_report(corr_df, cell_df, reg_df, df)


if __name__ == "__main__":
    main()
