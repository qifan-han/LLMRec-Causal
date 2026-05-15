"""Estimation pass for the Tonight-MVP simulation.

Produces three tables in results/tables/:
  1. one_shot_total_effect.csv     <- Prop 1: A/B identifies total effect
  2. decomposition_mvp.csv         <- Prop 3: modular 2x2 cell-mean contrasts
  3. naive_vs_oracle_persuasion.csv <- Prop 4: naive realized-E regression biased

All standard errors are analytical:
- Cell-mean contrasts: combined Bernoulli/normal SEs from the simple difference formulas.
- Regression coefficients: HC1 robust SEs from statsmodels.

True effects are computed from potential outcomes Y_prob.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "simulated"
OUT_DIR = ROOT / "results" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _mean_diff_se(y_a: np.ndarray, y_b: np.ndarray) -> tuple[float, float]:
    """Difference in means with two-sample SE (independent samples)."""
    mean_diff = y_a.mean() - y_b.mean()
    var_a = y_a.var(ddof=1) / len(y_a)
    var_b = y_b.var(ddof=1) / len(y_b)
    se = float(np.sqrt(var_a + var_b))
    return float(mean_diff), se


def _four_cell_interaction_se(modular: pd.DataFrame, col: str) -> tuple[float, float]:
    """Variance of the 2x2 interaction (q=1,r=1) - (q=1,r=0) - (q=0,r=1) + (q=0,r=0)."""
    cells = {}
    for q in (0, 1):
        for r in (0, 1):
            y = modular.loc[(modular["q"] == q) & (modular["r"] == r), col].values
            cells[(q, r)] = y
    val = cells[(1, 1)].mean() - cells[(1, 0)].mean() - cells[(0, 1)].mean() + cells[(0, 0)].mean()
    var = sum(cells[k].var(ddof=1) / len(cells[k]) for k in cells)
    return float(val), float(np.sqrt(var))


# -----------------------------------------------------------------------------
# Table 1: One-shot total effect
# -----------------------------------------------------------------------------

def table_one_shot_total_effect(one_shot: pd.DataFrame) -> pd.DataFrame:
    """A/B contrast of Y under z=1 vs z=0. Categories and pooled.

    True effect uses potential-outcome Y_prob; estimate uses realized Y.
    """
    rows = []
    for cat in [*sorted(one_shot["category"].unique()), "POOLED"]:
        if cat == "POOLED":
            sub = one_shot
        else:
            sub = one_shot[one_shot["category"] == cat]
        y_z1 = sub.loc[sub["z"] == 1, "Y"].values
        y_z0 = sub.loc[sub["z"] == 0, "Y"].values
        p_z1 = sub.loc[sub["z"] == 1, "Y_prob"].values
        p_z0 = sub.loc[sub["z"] == 0, "Y_prob"].values

        est, se = _mean_diff_se(y_z1, y_z0)
        true_val = float(p_z1.mean() - p_z0.mean())

        rows.append(
            {
                "category": cat,
                "estimand": "total_one_shot_effect",
                "true_value": true_val,
                "estimate": est,
                "se": se,
                "t_stat": est / se if se > 0 else np.nan,
                "n_z0": int(len(y_z0)),
                "n_z1": int(len(y_z1)),
                "interpretation": "Prop 1: prompt A/B identifies total effect",
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Table 2: Modular decomposition
# -----------------------------------------------------------------------------

def table_modular_decomposition(modular: pd.DataFrame) -> pd.DataFrame:
    """Cell-mean contrasts for the three modular estimands. Pooled + per-category."""
    rows = []
    for cat in [*sorted(modular["category"].unique()), "POOLED"]:
        if cat == "POOLED":
            sub = modular
        else:
            sub = modular[modular["category"] == cat]

        # Pull the four cell vectors of realized Y
        cell_Y = {(q, r): sub.loc[(sub["q"] == q) & (sub["r"] == r), "Y"].values for q in (0, 1) for r in (0, 1)}
        cell_P = {(q, r): sub.loc[(sub["q"] == q) & (sub["r"] == r), "Y_prob"].values for q in (0, 1) for r in (0, 1)}

        # Retrieval effect: (q=1, r=0) - (q=0, r=0)
        est_R, se_R = _mean_diff_se(cell_Y[(1, 0)], cell_Y[(0, 0)])
        true_R = float(cell_P[(1, 0)].mean() - cell_P[(0, 0)].mean())

        # Persuasion effect: (q=0, r=1) - (q=0, r=0)
        est_P, se_P = _mean_diff_se(cell_Y[(0, 1)], cell_Y[(0, 0)])
        true_P = float(cell_P[(0, 1)].mean() - cell_P[(0, 0)].mean())

        # Interaction: (q=1, r=1) - (q=1, r=0) - (q=0, r=1) + (q=0, r=0)
        est_I, se_I = _four_cell_interaction_se(sub, "Y")
        true_I = float(
            cell_P[(1, 1)].mean() - cell_P[(1, 0)].mean() - cell_P[(0, 1)].mean() + cell_P[(0, 0)].mean()
        )

        for estimand, est, se, true_val, prop in [
            ("retrieval", est_R, se_R, true_R, "Prop 3 (retrieval component)"),
            ("persuasion", est_P, se_P, true_P, "Prop 3 (persuasion component)"),
            ("interaction", est_I, se_I, true_I, "Prop 3 (interaction)"),
        ]:
            rows.append(
                {
                    "category": cat,
                    "estimand": estimand,
                    "true_value": true_val,
                    "estimate": est,
                    "se": se,
                    "t_stat": est / se if se > 0 else np.nan,
                    "bias": est - true_val,
                    "interpretation": prop,
                }
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Table 3: Naive vs oracle vs modular for the persuasion effect
# -----------------------------------------------------------------------------

def _ols_lpm(df: pd.DataFrame, formula: str) -> tuple[float, float]:
    """Fit OLS LPM with HC1 SEs, return (coef on E, robust SE)."""
    model = smf.ols(formula, data=df).fit(cov_type="HC1")
    coef = float(model.params["expression_intensity"])
    se = float(model.bse["expression_intensity"])
    return coef, se


def table_naive_vs_oracle(modular: pd.DataFrame) -> pd.DataFrame:
    """Compare the naive, oracle, and modular estimators of the persuasion effect.

    NOTE: the naive/oracle coefficients are *per-unit-E* effects; the modular
    cell-contrast is the *r=1 vs r=0 policy* effect, which equals
    (tau_R + lambda_fit*0) shifted on the latent index, with mean shift tau_R
    in expression intensity. To compare apples to apples, we scale the
    naive/oracle coefficients by mean(E | r=1) - mean(E | r=0) on the
    modular data.  This gives all three estimators the same target: the
    probability-scale persuasion effect of moving r=0 -> r=1.
    """
    # Compute the mean-E shift under r=1 vs r=0 on baseline-q rows
    df_q0 = modular[modular["q"] == 0].copy()
    delta_E = float(
        df_q0.loc[df_q0["r"] == 1, "expression_intensity"].mean()
        - df_q0.loc[df_q0["r"] == 0, "expression_intensity"].mean()
    )

    # True persuasion effect (probability-scale, at q=0)
    P00 = modular.loc[(modular["q"] == 0) & (modular["r"] == 0), "Y_prob"].mean()
    P01 = modular.loc[(modular["q"] == 0) & (modular["r"] == 1), "Y_prob"].mean()
    true_P = float(P01 - P00)

    # Naive regression: Y ~ E + incumbent + focal_brand + category FE
    naive_coef, naive_se = _ols_lpm(
        modular,
        "Y ~ expression_intensity + incumbent + focal_brand + C(category)",
    )

    # Oracle regression adds Q_selected_std as a control
    oracle_coef, oracle_se = _ols_lpm(
        modular,
        "Y ~ expression_intensity + Q_selected_std + incumbent + focal_brand + C(category)",
    )

    # Modular cell-mean persuasion contrast
    y00 = modular.loc[(modular["q"] == 0) & (modular["r"] == 0), "Y"].values
    y01 = modular.loc[(modular["q"] == 0) & (modular["r"] == 1), "Y"].values
    modular_est, modular_se = _mean_diff_se(y01, y00)

    rows = [
        {
            "estimator": "naive_regression",
            "target": "persuasion effect (probability-scale, r=1 vs r=0 at q=0)",
            "raw_coef_on_E": naive_coef,
            "raw_se_on_E": naive_se,
            "scaled_estimate": naive_coef * delta_E,
            "true_value": true_P,
            "bias_scaled": naive_coef * delta_E - true_P,
            "notes": "OLS LPM, HC1 SEs, FE by category; coef interpreted per unit of E and rescaled to r=1 policy",
        },
        {
            "estimator": "oracle_regression",
            "target": "persuasion effect (probability-scale, r=1 vs r=0 at q=0)",
            "raw_coef_on_E": oracle_coef,
            "raw_se_on_E": oracle_se,
            "scaled_estimate": oracle_coef * delta_E,
            "true_value": true_P,
            "bias_scaled": oracle_coef * delta_E - true_P,
            "notes": "Adds Q_selected_std as control (infeasible benchmark)",
        },
        {
            "estimator": "modular_2x2_cell_contrast",
            "target": "persuasion effect (probability-scale, r=1 vs r=0 at q=0)",
            "raw_coef_on_E": np.nan,
            "raw_se_on_E": np.nan,
            "scaled_estimate": modular_est,
            "true_value": true_P,
            "bias_scaled": modular_est - true_P,
            "notes": "Cell-mean contrast on modular data; design-based identification (Prop 3)",
        },
    ]
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    one_shot = pd.read_csv(DATA_DIR / "one_shot_mvp.csv")
    modular = pd.read_csv(DATA_DIR / "modular_mvp.csv")
    print(f"Loaded one_shot ({len(one_shot):,} rows) and modular ({len(modular):,} rows)")

    t1 = table_one_shot_total_effect(one_shot)
    t2 = table_modular_decomposition(modular)
    t3 = table_naive_vs_oracle(modular)

    p1 = OUT_DIR / "one_shot_total_effect.csv"
    p2 = OUT_DIR / "decomposition_mvp.csv"
    p3 = OUT_DIR / "naive_vs_oracle_persuasion.csv"
    t1.to_csv(p1, index=False)
    t2.to_csv(p2, index=False)
    t3.to_csv(p3, index=False)

    print("\n=== Table 1: one-shot total effect ===")
    print(t1.to_string(index=False))
    print(f"\nSaved -> {p1}")
    print("\n=== Table 2: modular decomposition ===")
    print(t2.to_string(index=False))
    print(f"\nSaved -> {p2}")
    print("\n=== Table 3: naive vs oracle vs modular (persuasion) ===")
    print(t3.to_string(index=False))
    print(f"\nSaved -> {p3}")


if __name__ == "__main__":
    main()
