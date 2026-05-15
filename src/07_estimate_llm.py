"""Phase 2 estimation: analyze LLM simulation outputs.

Produces three tables paralleling the Phase 1 estimation:
  1. One-shot total effect (Prop 1 / Remark 1)
  2. Modular decomposition (Prop 1-3)
  3. Naive vs oracle vs modular persuasion estimate (Prop 2-3)

Also produces calibration estimates:
  - lambda_fit_hat: slope of E on Q_std in (q=0, r=0) cell
  - sigma_R_hat: within-cell variance of expression intensity
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except ImportError:
    sm = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.data_io import ROOT

OUTPUT_DIR = ROOT / "data" / "llm_sim"
RESULTS_DIR = ROOT / "results" / "tables"


def _load():
    one_shot = pd.read_csv(OUTPUT_DIR / "one_shot_llm.csv")
    modular = pd.read_csv(OUTPUT_DIR / "modular_llm.csv")
    retrieval = pd.read_csv(OUTPUT_DIR / "retrieval_llm.csv")
    print(f"Loaded one_shot ({len(one_shot)} rows), modular ({len(modular)} rows), "
          f"retrieval ({len(retrieval)} rows)")
    return one_shot, modular, retrieval


def table1_one_shot_total(one_shot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat in sorted(one_shot["category"].unique()):
        sub = one_shot[one_shot["category"] == cat]
        y0 = sub[sub["z"] == 0]["Y"]
        y1 = sub[sub["z"] == 1]["Y"]
        diff = y1.mean() - y0.mean()
        se = np.sqrt(y0.var(ddof=1) / len(y0) + y1.var(ddof=1) / len(y1))
        t = diff / se if se > 0 else 0
        rows.append({
            "category": cat,
            "estimand": "total_one_shot_effect",
            "estimate": round(diff, 4),
            "se": round(se, 4),
            "t_stat": round(t, 4),
            "n_z0": len(y0),
            "n_z1": len(y1),
            "Y_z0": round(y0.mean(), 4),
            "Y_z1": round(y1.mean(), 4),
        })

    y0 = one_shot[one_shot["z"] == 0]["Y"]
    y1 = one_shot[one_shot["z"] == 1]["Y"]
    diff = y1.mean() - y0.mean()
    se = np.sqrt(y0.var(ddof=1) / len(y0) + y1.var(ddof=1) / len(y1))
    t = diff / se if se > 0 else 0
    rows.append({
        "category": "POOLED",
        "estimand": "total_one_shot_effect",
        "estimate": round(diff, 4),
        "se": round(se, 4),
        "t_stat": round(t, 4),
        "n_z0": len(y0),
        "n_z1": len(y1),
        "Y_z0": round(y0.mean(), 4),
        "Y_z1": round(y1.mean(), 4),
    })
    return pd.DataFrame(rows)


def table2_decomposition(modular: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat in list(sorted(modular["category"].unique())) + ["POOLED"]:
        sub = modular if cat == "POOLED" else modular[modular["category"] == cat]

        cells = {}
        for q in [0, 1]:
            for r in [0, 1]:
                cell = sub[(sub["q"] == q) & (sub["r"] == r)]
                cells[(q, r)] = cell["Y"].mean() if len(cell) > 0 else np.nan

        retrieval = cells[(1, 0)] - cells[(0, 0)]
        persuasion = cells[(0, 1)] - cells[(0, 0)]
        interaction = (cells[(1, 1)] - cells[(1, 0)]) - (cells[(0, 1)] - cells[(0, 0)])

        n = len(sub) // 4

        for name, est in [("retrieval", retrieval),
                          ("persuasion", persuasion),
                          ("interaction", interaction)]:
            se = np.sqrt(2 * sub["Y"].var(ddof=1) / max(n, 1)) if n > 0 else np.nan
            t = est / se if se > 0 else 0
            rows.append({
                "category": cat,
                "estimand": name,
                "estimate": round(est, 4),
                "se": round(se, 4),
                "t_stat": round(t, 4),
            })
    return pd.DataFrame(rows)


def table3_naive_vs_modular(modular: pd.DataFrame) -> pd.DataFrame:
    rows = []

    cat_dummies = pd.get_dummies(modular["category"], drop_first=True, dtype=float)
    q0 = modular[modular["q"] == 0].copy()

    # Naive: regress Y on E with category FE (no Q_std control)
    if sm and len(q0) > 10:
        X_naive = pd.concat([q0[["expression_intensity"]].reset_index(drop=True),
                             cat_dummies.loc[q0.index].reset_index(drop=True)], axis=1)
        X_naive = sm.add_constant(X_naive)
        y = q0["Y"].reset_index(drop=True)
        try:
            mod = sm.OLS(y, X_naive).fit(cov_type="HC1")
            coef = mod.params["expression_intensity"]
            se = mod.bse["expression_intensity"]
            E_range = q0["expression_intensity"].quantile(0.75) - q0["expression_intensity"].quantile(0.25)
            rows.append({
                "estimator": "naive_regression",
                "raw_coef_on_E": round(coef, 6),
                "raw_se_on_E": round(se, 6),
                "E_iqr": round(E_range, 4),
                "scaled_estimate_iqr": round(coef * E_range, 4),
            })
        except Exception:
            pass

    # Oracle: add Q_std as control
    if sm and len(q0) > 10:
        X_oracle = pd.concat([q0[["expression_intensity", "Q_std"]].reset_index(drop=True),
                              cat_dummies.loc[q0.index].reset_index(drop=True)], axis=1)
        X_oracle = sm.add_constant(X_oracle)
        y = q0["Y"].reset_index(drop=True)
        try:
            mod = sm.OLS(y, X_oracle).fit(cov_type="HC1")
            coef = mod.params["expression_intensity"]
            se = mod.bse["expression_intensity"]
            E_range = q0["expression_intensity"].quantile(0.75) - q0["expression_intensity"].quantile(0.25)
            rows.append({
                "estimator": "oracle_regression",
                "raw_coef_on_E": round(coef, 6),
                "raw_se_on_E": round(se, 6),
                "E_iqr": round(E_range, 4),
                "scaled_estimate_iqr": round(coef * E_range, 4),
            })
        except Exception:
            pass

    # Modular cell contrast
    y_r0 = q0[q0["r"] == 0]["Y"].mean()
    y_r1 = q0[q0["r"] == 1]["Y"].mean()
    rows.append({
        "estimator": "modular_2x2_cell_contrast",
        "raw_coef_on_E": None,
        "raw_se_on_E": None,
        "E_iqr": None,
        "scaled_estimate_iqr": round(y_r1 - y_r0, 4),
    })

    return pd.DataFrame(rows)


def calibration(modular: pd.DataFrame) -> dict:
    """Recover lambda_fit_hat and sigma_R_hat from LLM outputs."""
    q0r0 = modular[(modular["q"] == 0) & (modular["r"] == 0)]

    results = {}

    # lambda_fit_hat: slope of E on Q_std
    if len(q0r0) > 5:
        corr = q0r0["expression_intensity"].corr(q0r0["Q_std"])
        results["corr_E_Qstd"] = round(corr, 4)

        if sm:
            X = sm.add_constant(q0r0["Q_std"])
            y = q0r0["expression_intensity"]
            mod = sm.OLS(y, X).fit()
            results["lambda_fit_hat"] = round(mod.params["Q_std"], 4)
            results["lambda_fit_se"] = round(mod.bse["Q_std"], 4)
            results["lambda_fit_t"] = round(mod.tvalues["Q_std"], 4)

    # sigma_R_hat: within-cell variance of E
    for label, mask in [("q0r0", (modular["q"] == 0) & (modular["r"] == 0)),
                        ("q0r1", (modular["q"] == 0) & (modular["r"] == 1)),
                        ("q1r0", (modular["q"] == 1) & (modular["r"] == 0)),
                        ("q1r1", (modular["q"] == 1) & (modular["r"] == 1))]:
        cell = modular[mask]
        if len(cell) > 1:
            results[f"E_mean_{label}"] = round(cell["expression_intensity"].mean(), 4)
            results[f"E_std_{label}"] = round(cell["expression_intensity"].std(), 4)

    # Overall sigma_R_hat: residual std of E after removing cell means
    modular_copy = modular.copy()
    modular_copy["cell"] = modular_copy["q"].astype(str) + "_" + modular_copy["r"].astype(str)
    cell_means = modular_copy.groupby("cell")["expression_intensity"].transform("mean")
    residuals = modular_copy["expression_intensity"] - cell_means
    results["sigma_R_hat"] = round(residuals.std(), 4)

    return results


def main():
    one_shot, modular, retrieval = _load()

    print("\n=== Table 1: One-shot total effect (LLM) ===")
    t1 = table1_one_shot_total(one_shot)
    print(t1.to_string(index=False))
    t1.to_csv(RESULTS_DIR / "llm_one_shot_total_effect.csv", index=False)

    print("\n=== Table 2: Modular decomposition (LLM) ===")
    t2 = table2_decomposition(modular)
    print(t2.to_string(index=False))
    t2.to_csv(RESULTS_DIR / "llm_decomposition.csv", index=False)

    print("\n=== Table 3: Naive vs oracle vs modular (LLM) ===")
    t3 = table3_naive_vs_modular(modular)
    print(t3.to_string(index=False))
    t3.to_csv(RESULTS_DIR / "llm_naive_vs_oracle.csv", index=False)

    print("\n=== Calibration ===")
    cal = calibration(modular)
    for k, v in cal.items():
        print(f"  {k}: {v}")
    with open(RESULTS_DIR / "llm_calibration.json", "w") as f:
        json.dump(cal, f, indent=2)

    print(f"\nAll tables saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
