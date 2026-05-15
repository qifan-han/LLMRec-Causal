"""Headline figure: naive bias vs lambda_fit (with Monte Carlo SE bands).

For lambda_fit in a sweep, rerun the simulation M times with the same
retrieval realizations but fresh expression noise eta and outcome noise.
At each lambda, compute:
  - naive regression coefficient on E (scaled to the r=1 policy effect)
  - oracle regression coefficient on E (scaled likewise)
  - modular cell-mean contrast for persuasion
  - true persuasion effect from potential outcomes
Plot mean bias vs lambda with +/- 1 SE shaded bands.

Also saves a CSV of the underlying numbers so the figure is reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from importlib import import_module  # noqa: E402

sim = import_module("02_simulate_core_mvp")
from core.data_io import load_all_categories  # noqa: E402

FIG_DIR = ROOT / "results" / "figures"
TAB_DIR = ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
M_REPS = 25
MASTER_SEED = 20260513


# -----------------------------------------------------------------------------
# Single replication
# -----------------------------------------------------------------------------

def _one_rep(
    sel_baseline_by_cat: dict[str, pd.DataFrame],
    sel_focal_by_cat: dict[str, pd.DataFrame],
    lambda_fit: float,
    rng_expr: np.random.Generator,
    rng_outcome: np.random.Generator,
    params: dict,
) -> dict:
    """Build the modular frame for all categories at a single lambda value."""
    parts = []
    for cat in sel_baseline_by_cat:
        sb = sel_baseline_by_cat[cat]
        sf = sel_focal_by_cat[cat]
        mod_df = sim._build_modular_frame(
            sb, sf, rng_expr, rng_outcome, params, lambda_fit=lambda_fit
        )
        parts.append(mod_df)
    modular = pd.concat(parts, ignore_index=True)

    # Mean-E shift on baseline-q rows
    df_q0 = modular[modular["q"] == 0]
    delta_E = float(
        df_q0.loc[df_q0["r"] == 1, "expression_intensity"].mean()
        - df_q0.loc[df_q0["r"] == 0, "expression_intensity"].mean()
    )

    # True persuasion effect (probability-scale, at q=0)
    P00 = modular.loc[(modular["q"] == 0) & (modular["r"] == 0), "Y_prob"].mean()
    P01 = modular.loc[(modular["q"] == 0) & (modular["r"] == 1), "Y_prob"].mean()
    true_P = float(P01 - P00)

    naive = smf.ols(
        "Y ~ expression_intensity + incumbent + focal_brand + C(category)",
        data=modular,
    ).fit(cov_type="HC1")
    naive_coef = float(naive.params["expression_intensity"])

    oracle = smf.ols(
        "Y ~ expression_intensity + Q_selected_std + incumbent + focal_brand + C(category)",
        data=modular,
    ).fit(cov_type="HC1")
    oracle_coef = float(oracle.params["expression_intensity"])

    # Modular cell contrast
    y00 = modular.loc[(modular["q"] == 0) & (modular["r"] == 0), "Y"].values
    y01 = modular.loc[(modular["q"] == 0) & (modular["r"] == 1), "Y"].values
    modular_est = float(y01.mean() - y00.mean())

    return {
        "lambda_fit": lambda_fit,
        "true_persuasion": true_P,
        "naive_scaled": naive_coef * delta_E,
        "oracle_scaled": oracle_coef * delta_E,
        "modular": modular_est,
        "delta_E": delta_E,
    }


# -----------------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------------

def run_sweep(
    master_seed: int = MASTER_SEED,
    lambda_values=LAMBDA_VALUES,
    m_reps: int = M_REPS,
    verbose: bool = True,
) -> pd.DataFrame:
    params = sim.DEFAULTS.copy()

    # Set up RNGs. Retrieval RNG is shared so selected products are fixed
    # across all replications; expression and outcome RNGs are unique per rep.
    seq = np.random.SeedSequence(master_seed)
    seed_retrieval, expr_seq, out_seq = seq.spawn(3)
    rng_R = np.random.default_rng(seed_retrieval)

    # Pre-compute selected-product frames per category, once
    cats = load_all_categories(verbose=verbose)
    sel_baseline_by_cat: dict[str, pd.DataFrame] = {}
    sel_focal_by_cat: dict[str, pd.DataFrame] = {}
    for cat, data in cats.items():
        sb, sf = sim.simulate_recommendation_for_category(data, rng_R, params)
        sel_baseline_by_cat[cat] = sb
        sel_focal_by_cat[cat] = sf

    # Per-rep expression and outcome seeds — independent across reps and lambdas
    expr_seeds = expr_seq.spawn(len(lambda_values) * m_reps)
    out_seeds = out_seq.spawn(len(lambda_values) * m_reps)

    rows = []
    k = 0
    for lf in lambda_values:
        for rep in range(m_reps):
            rng_e = np.random.default_rng(expr_seeds[k])
            rng_y = np.random.default_rng(out_seeds[k])
            row = _one_rep(
                sel_baseline_by_cat, sel_focal_by_cat,
                lambda_fit=lf, rng_expr=rng_e, rng_outcome=rng_y,
                params=params,
            )
            row["rep"] = rep
            rows.append(row)
            k += 1
        if verbose:
            sub = pd.DataFrame([r for r in rows if r["lambda_fit"] == lf])
            print(
                f"  lambda={lf:.2f}  M={len(sub):2d}  "
                f"true_P={sub['true_persuasion'].mean():+.4f}  "
                f"naive={sub['naive_scaled'].mean():+.4f}  "
                f"oracle={sub['oracle_scaled'].mean():+.4f}  "
                f"modular={sub['modular'].mean():+.4f}"
            )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Aggregate + figure
# -----------------------------------------------------------------------------

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and SE of bias (estimate - true) by lambda, by estimator."""
    df = df.copy()
    df["naive_bias"] = df["naive_scaled"] - df["true_persuasion"]
    df["oracle_bias"] = df["oracle_scaled"] - df["true_persuasion"]
    df["modular_bias"] = df["modular"] - df["true_persuasion"]

    rows = []
    for lf, sub in df.groupby("lambda_fit"):
        for est_name in ("naive", "oracle", "modular"):
            b = sub[f"{est_name}_bias"]
            rows.append(
                {
                    "lambda_fit": lf,
                    "estimator": est_name,
                    "mean_bias": float(b.mean()),
                    "se_bias": float(b.std(ddof=1) / np.sqrt(len(b))),
                    "mean_estimate": float(sub[f"{est_name}_scaled" if est_name != "modular" else "modular"].mean()),
                    "mean_true": float(sub["true_persuasion"].mean()),
                    "n_reps": int(len(b)),
                }
            )
    return pd.DataFrame(rows)


def make_figure(agg: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    colors = {"naive": "#d62728", "oracle": "#2ca02c", "modular": "#1f77b4"}
    labels = {
        "naive": "Naive realized-E regression",
        "oracle": "Oracle (adds Q$_{std}$)",
        "modular": "Modular 2×2 cell contrast",
    }

    for est in ("naive", "oracle", "modular"):
        sub = agg[agg["estimator"] == est].sort_values("lambda_fit")
        x = sub["lambda_fit"].values
        y = sub["mean_bias"].values
        se = sub["se_bias"].values
        ax.plot(x, y, marker="o", color=colors[est], label=labels[est], linewidth=2)
        ax.fill_between(x, y - se, y + se, color=colors[est], alpha=0.18)

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel(r"Endogenous-expression strength $\lambda_{fit}$")
    ax.set_ylabel("Bias = estimate − true persuasion effect (probability-scale)")
    ax.set_title("Naive expression regression is biased; modular 2×2 is not")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    print(f"Running lambda sweep: M={M_REPS} reps over {len(LAMBDA_VALUES)} lambda values")
    df = run_sweep(verbose=True)
    print(f"\nCompleted {len(df)} simulation reps.")

    raw_path = TAB_DIR / "lambda_sweep_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"Saved raw -> {raw_path}")

    agg = aggregate(df)
    agg_path = TAB_DIR / "lambda_sweep_aggregated.csv"
    agg.to_csv(agg_path, index=False)
    print(f"Saved aggregated -> {agg_path}")

    print("\n=== Aggregated (mean_bias by lambda x estimator) ===")
    pivot = agg.pivot(index="lambda_fit", columns="estimator", values="mean_bias")
    print(pivot.to_string(float_format=lambda x: f"{x:+.4f}"))

    # Validation: at moderate lambda (0.25 - 1.0), naive bias must be materially
    # larger than oracle and modular bias. This is the substantive claim — not
    # strict monotonicity. The classical OVB formula b ~ Cov(E,Q)/Var(E) is
    # non-monotonic in lambda because Var(E) grows quadratically while Cov
    # grows linearly, so the bias peaks at moderate endogeneity and attenuates
    # at extremes. Both ends of that pattern are theoretically correct.
    naive = agg[(agg["estimator"] == "naive") & (agg["lambda_fit"].between(0.25, 1.0))]
    modular = agg[(agg["estimator"] == "modular") & (agg["lambda_fit"].between(0.25, 1.0))]
    if naive["mean_bias"].mean() < 0.01:
        print("\nWARNING: naive bias is unexpectedly small at moderate lambda.")
    elif naive["mean_bias"].mean() <= modular["mean_bias"].abs().mean() * 3:
        print("\nWARNING: naive bias does not clearly dominate modular bias at moderate lambda.")
    else:
        print(
            "\nMain finding OK: at moderate lambda (0.25-1.0), naive bias "
            f"= {naive['mean_bias'].mean():+.4f}, modular bias = {modular['mean_bias'].mean():+.4f}."
        )
        print(
            "  (The non-monotonic shape is the classical OVB pattern: "
            "Cov(E, Q) is linear in lambda but Var(E) is quadratic, so bias peaks "
            "at moderate endogeneity and attenuates at the extremes.)"
        )

    fig_path = FIG_DIR / "naive_bias_vs_lambda.png"
    make_figure(agg, fig_path)
    print(f"\nSaved figure -> {fig_path}")


if __name__ == "__main__":
    main()
