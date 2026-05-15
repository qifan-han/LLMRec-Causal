"""
Semi-synthetic DGP analysis with Monte Carlo repetitions and parameter sweeps.

Steps 3-8 of the robustness/repair pass.
- Three DGPs using only validated scales (PI, TD) and Q_std
- 500 MC Bernoulli draws per DGP per parameter setting
- Deterministic probability-based estimands as primary
- Naive vs oracle vs modular estimator comparison
- Honest bias framing given near-zero PI/TD vs Q_std correlations
"""

import pathlib, warnings, itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
REPORTS = RESULTS / "reports"
SEMI = DATA / "semisynthetic"

for d in [TABLES, REPORTS, SEMI]:
    d.mkdir(parents=True, exist_ok=True)

N_MC = 500
MASTER_SEED = 20260514


def load_data():
    df = pd.read_csv(RESULTS / "diagnostics" / "evaluator_scores.csv")
    df["cluster_id"] = df["category"] + "_" + df["consumer_id"].astype(str)
    # standardize PI and TD within the full sample
    for col in ["persuasive_intensity", "tradeoff_disclosure"]:
        df[f"{col}_std"] = (df[col] - df[col].mean()) / df[col].std()
    df["Q_std_centered"] = (df["Q_std"] - df["Q_std"].mean()) / df["Q_std"].std()
    # category median Q_std for mismatch definition
    df["Q_median_cat"] = df.groupby("category")["Q_std"].transform("median")
    df["low_fit"] = (df["Q_std"] < df["Q_median_cat"]).astype(int)
    return df


# ── DGP Definitions ───────────────────────────────────────────────

def dgp1_conversion(df, beta_Q=0.8, beta_PI=0.5, beta_TD=-0.15, beta_0=-0.5):
    """
    Conversion-pressure DGP.
    Q_std increases purchase probability.
    PI increases purchase probability.
    TD weakly decreases purchase probability (caveats may reduce conversion).
    Target: conversion effect of r.
    """
    eta = (beta_0
           + beta_Q * df["Q_std_centered"].values
           + beta_PI * df["persuasive_intensity_std"].values
           + beta_TD * df["tradeoff_disclosure_std"].values)
    prob = expit(eta)
    return prob


def dgp2_welfare(df, beta_Q=0.8, beta_PI=0.4, beta_TD=0.1, beta_0=-0.3):
    """
    Transparency/welfare DGP.
    Purchase probability driven by Q, PI, and weakly by TD.
    Welfare is a separate quantity:
      welfare = P(purchase) * match_utility
      match_utility = Q_std + alpha * TD (higher transparency → better-informed choice)
    where alpha controls how much transparency improves match quality.
    """
    eta = (beta_0
           + beta_Q * df["Q_std_centered"].values
           + beta_PI * df["persuasive_intensity_std"].values
           + beta_TD * df["tradeoff_disclosure_std"].values)
    prob = expit(eta)
    return prob


def dgp2_welfare_score(df, prob, alpha_td=0.3):
    """
    Welfare = P(purchase) * match_utility.
    match_utility = max(0, Q_std + alpha * TD_std).
    Clamp match_utility at 0 to avoid negative welfare from very negative standardized values.
    """
    match_utility = np.maximum(0, df["Q_std_centered"].values + alpha_td * df["tradeoff_disclosure_std"].values)
    welfare = prob * match_utility
    return welfare


def dgp3_mismatch(df, beta_Q=0.5, beta_PI=0.8, beta_TD=0.0, beta_0=0.0):
    """
    Over-persuasion/mismatch DGP.
    PI increases purchase probability (strong effect).
    Low Q_std purchases are mismatch events.
    expected_mismatch = P(purchase) * 1{Q_std below category median}.
    """
    eta = (beta_0
           + beta_Q * df["Q_std_centered"].values
           + beta_PI * df["persuasive_intensity_std"].values
           + beta_TD * df["tradeoff_disclosure_std"].values)
    prob = expit(eta)
    return prob


# ── Estimators ─────────────────────────────────────────────────────

def modular_cell_contrasts(df, prob_col):
    """2x2 cell-mean contrasts on deterministic probabilities."""
    cell_means = df.groupby(["q", "r"])[prob_col].mean()
    # retrieval effect: mean(q=1) - mean(q=0), averaged over r
    retrieval = 0.5 * ((cell_means[(1, 0)] - cell_means[(0, 0)]) +
                       (cell_means[(1, 1)] - cell_means[(0, 1)]))
    # expression effect: mean(r=1) - mean(r=0), averaged over q
    expression = 0.5 * ((cell_means[(0, 1)] - cell_means[(0, 0)]) +
                        (cell_means[(1, 1)] - cell_means[(1, 0)]))
    # interaction
    interaction = ((cell_means[(1, 1)] - cell_means[(1, 0)]) -
                   (cell_means[(0, 1)] - cell_means[(0, 0)]))
    return {"retrieval": retrieval, "expression": expression, "interaction": interaction}


def naive_regression(df, y_col, x_col):
    """Y ~ PI (or TD), no Q_std control."""
    X = sm.add_constant(df[[x_col]])
    model = sm.OLS(df[y_col], X).fit(cov_type="cluster",
                                      cov_kwds={"groups": df["cluster_id"].values})
    return {"coef": model.params[x_col], "se": model.bse[x_col],
            "t": model.tvalues[x_col], "p": model.pvalues[x_col]}


def oracle_regression(df, y_col, x_col):
    """Y ~ PI (or TD) + Q_std + C(category)."""
    cat_dummies = pd.get_dummies(df["category"], drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([df[[x_col, "Q_std_centered"]], cat_dummies], axis=1))
    model = sm.OLS(df[y_col], X).fit(cov_type="cluster",
                                      cov_kwds={"groups": df["cluster_id"].values})
    return {"coef": model.params[x_col], "se": model.bse[x_col],
            "t": model.tvalues[x_col], "p": model.pvalues[x_col]}


# ── Monte Carlo ────────────────────────────────────────────────────

def run_mc_for_dgp(df, probs, dgp_name, n_mc=N_MC, seed=MASTER_SEED):
    """Run Monte Carlo Bernoulli draws and estimate effects on realized Y."""
    rng = np.random.default_rng(seed)
    n = len(df)

    # true effects on probabilities (deterministic)
    true_contrasts = modular_cell_contrasts(df.assign(**{"prob": probs}), "prob")

    mc_results = []
    for rep in range(n_mc):
        Y = rng.binomial(1, probs, size=n)
        df_mc = df.copy()
        df_mc["Y_mc"] = Y

        # modular contrasts on realized Y
        contrasts = modular_cell_contrasts(df_mc, "Y_mc")

        # naive regressions on realized Y
        naive_pi = naive_regression(df_mc, "Y_mc", "persuasive_intensity_std")
        oracle_pi = oracle_regression(df_mc, "Y_mc", "persuasive_intensity_std")

        mc_results.append({
            "rep": rep,
            "mod_retrieval": contrasts["retrieval"],
            "mod_expression": contrasts["expression"],
            "mod_interaction": contrasts["interaction"],
            "naive_PI_coef": naive_pi["coef"],
            "naive_PI_se": naive_pi["se"],
            "oracle_PI_coef": oracle_pi["coef"],
            "oracle_PI_se": oracle_pi["se"],
        })

    mc_df = pd.DataFrame(mc_results)
    return mc_df, true_contrasts


def summarize_mc(mc_df, true_contrasts, dgp_name):
    """Summarize MC results: mean, sd, bias, RMSE."""
    records = []
    for est_name, true_val in [
        ("mod_retrieval", true_contrasts["retrieval"]),
        ("mod_expression", true_contrasts["expression"]),
        ("mod_interaction", true_contrasts["interaction"]),
    ]:
        vals = mc_df[est_name].values
        records.append({
            "dgp": dgp_name, "estimator": est_name,
            "true_value": round(true_val, 5),
            "mc_mean": round(vals.mean(), 5),
            "mc_sd": round(vals.std(), 5),
            "bias": round(vals.mean() - true_val, 5),
            "rmse": round(np.sqrt(((vals - true_val) ** 2).mean()), 5),
        })

    # naive vs oracle PI
    true_expression = true_contrasts["expression"]
    for est_name, col in [("naive_PI", "naive_PI_coef"), ("oracle_PI", "oracle_PI_coef")]:
        vals = mc_df[col].values
        records.append({
            "dgp": dgp_name, "estimator": est_name,
            "true_value": round(true_expression, 5),
            "mc_mean": round(vals.mean(), 5),
            "mc_sd": round(vals.std(), 5),
            "bias": round(vals.mean() - true_expression, 5),
            "rmse": round(np.sqrt(((vals - true_expression) ** 2).mean()), 5),
        })

    return records


# ── Parameter Sweep ────────────────────────────────────────────────

def run_parameter_sweep(df):
    """Sweep over beta_PI and beta_TD for each DGP."""
    beta_PI_grid = [0.2, 0.5, 0.8]
    beta_TD_grid = [-0.3, -0.15, 0.0]
    beta_Q_vals = [0.5, 0.8]

    rng = np.random.default_rng(MASTER_SEED + 1000)
    records = []

    for beta_Q, beta_PI, beta_TD in itertools.product(beta_Q_vals, beta_PI_grid, beta_TD_grid):
        # DGP1: conversion
        probs = dgp1_conversion(df, beta_Q=beta_Q, beta_PI=beta_PI, beta_TD=beta_TD)
        contrasts = modular_cell_contrasts(df.assign(prob=probs), "prob")

        # deterministic naive vs oracle on probabilities
        naive = naive_regression(df.assign(prob=probs), "prob", "persuasive_intensity_std")
        oracle = oracle_regression(df.assign(prob=probs), "prob", "persuasive_intensity_std")

        records.append({
            "dgp": "DGP1_conversion",
            "beta_Q": beta_Q, "beta_PI": beta_PI, "beta_TD": beta_TD,
            "true_expression_effect": round(contrasts["expression"], 5),
            "true_retrieval_effect": round(contrasts["retrieval"], 5),
            "naive_PI_coef": round(naive["coef"], 5),
            "oracle_PI_coef": round(oracle["coef"], 5),
            "naive_bias": round(naive["coef"] - contrasts["expression"], 5),
            "mean_prob": round(probs.mean(), 4),
        })

    # DGP3: mismatch sweep over beta_PI
    for beta_Q, beta_PI in itertools.product([0.3, 0.5, 0.8], [0.3, 0.5, 0.8]):
        probs = dgp3_mismatch(df, beta_Q=beta_Q, beta_PI=beta_PI)
        df_tmp = df.assign(prob=probs)
        exp_mismatch = probs * df["low_fit"].values

        # mismatch rate by r
        mismatch_r0 = exp_mismatch[df["r"] == 0].mean()
        mismatch_r1 = exp_mismatch[df["r"] == 1].mean()

        records.append({
            "dgp": "DGP3_mismatch",
            "beta_Q": beta_Q, "beta_PI": beta_PI, "beta_TD": 0.0,
            "true_expression_effect": round(
                modular_cell_contrasts(df_tmp, "prob")["expression"], 5),
            "true_retrieval_effect": round(
                modular_cell_contrasts(df_tmp, "prob")["retrieval"], 5),
            "naive_PI_coef": np.nan,
            "oracle_PI_coef": np.nan,
            "naive_bias": np.nan,
            "mean_prob": round(probs.mean(), 4),
            "mismatch_r0": round(mismatch_r0, 4),
            "mismatch_r1": round(mismatch_r1, 4),
            "mismatch_diff": round(mismatch_r1 - mismatch_r0, 4),
        })

    return pd.DataFrame(records)


# ── DGP2 Welfare Analysis ─────────────────────────────────────────

def welfare_analysis(df):
    """Compute welfare under different r conditions for DGP2."""
    records = []
    for alpha_td in [0.1, 0.3, 0.5]:
        probs = dgp2_welfare(df)
        welfare = dgp2_welfare_score(df, probs, alpha_td=alpha_td)
        df_tmp = df.assign(prob=probs, welfare=welfare)

        for r_val in [0, 1]:
            sub = df_tmp[df_tmp["r"] == r_val]
            records.append({
                "alpha_td": alpha_td,
                "r": r_val,
                "mean_prob": round(sub["prob"].mean(), 4),
                "mean_welfare": round(sub["welfare"].mean(), 4),
                "mean_match_utility": round(
                    np.maximum(0, sub["Q_std_centered"] + alpha_td * sub["tradeoff_disclosure_std"]).mean(), 4),
            })

    return pd.DataFrame(records)


def main():
    print("Loading data...")
    df = load_data()

    # ── Baseline DGPs ──────────────────────────────────────────────
    print("\n=== DGP1: Conversion-pressure ===")
    probs1 = dgp1_conversion(df)
    print(f"  Mean prob: {probs1.mean():.4f}")
    for r_val in [0, 1]:
        mask = df["r"] == r_val
        print(f"  r={r_val}: mean prob = {probs1[mask].mean():.4f}")

    contrasts1 = modular_cell_contrasts(df.assign(prob=probs1), "prob")
    print(f"  True expression effect: {contrasts1['expression']:.5f}")
    print(f"  True retrieval effect:  {contrasts1['retrieval']:.5f}")

    print("\n=== DGP2: Transparency/welfare ===")
    probs2 = dgp2_welfare(df)
    welfare2 = dgp2_welfare_score(df, probs2, alpha_td=0.3)
    print(f"  Mean prob: {probs2.mean():.4f}")
    for r_val in [0, 1]:
        mask = df["r"] == r_val
        print(f"  r={r_val}: mean prob = {probs2[mask].mean():.4f}, "
              f"mean welfare = {welfare2[mask].mean():.4f}")

    print("\n=== DGP3: Over-persuasion/mismatch ===")
    probs3 = dgp3_mismatch(df)
    exp_mismatch3 = probs3 * df["low_fit"].values
    print(f"  Mean prob: {probs3.mean():.4f}")
    for r_val in [0, 1]:
        mask = df["r"] == r_val
        print(f"  r={r_val}: mean prob = {probs3[mask].mean():.4f}, "
              f"E[mismatch] = {exp_mismatch3[mask].mean():.4f}")

    # ── Monte Carlo ────────────────────────────────────────────────
    print("\n=== Monte Carlo (500 reps each) ===")
    all_mc_summaries = []

    for dgp_name, probs, seed_offset in [
        ("DGP1_conversion", probs1, 0),
        ("DGP2_welfare", probs2, 1000),
        ("DGP3_mismatch", probs3, 2000),
    ]:
        print(f"  Running {dgp_name}...")
        mc_df, true_c = run_mc_for_dgp(df, probs, dgp_name,
                                        seed=MASTER_SEED + seed_offset)
        summaries = summarize_mc(mc_df, true_c, dgp_name)
        all_mc_summaries.extend(summaries)

    mc_summary_df = pd.DataFrame(all_mc_summaries)
    mc_summary_df.to_csv(TABLES / "validated_estimator_comparison_mc.csv", index=False)
    print("\nMC Summary:")
    print(mc_summary_df.to_string(index=False))

    # ── Parameter Sweep ────────────────────────────────────────────
    print("\n=== Parameter sweep ===")
    sweep_df = run_parameter_sweep(df)
    sweep_df.to_csv(TABLES / "validated_dgp_parameter_grid.csv", index=False)
    print(sweep_df.to_string(index=False))

    # ── Welfare Analysis ───────────────────────────────────────────
    print("\n=== Welfare analysis ===")
    welfare_df = welfare_analysis(df)
    welfare_df.to_csv(TABLES / "validated_welfare_analysis.csv", index=False)
    print(welfare_df.to_string(index=False))

    # ── Save outcomes ──────────────────────────────────────────────
    outcomes = df[["row_key", "category", "consumer_id", "q", "r",
                   "product_id", "Q_std", "persuasive_intensity",
                   "tradeoff_disclosure", "cluster_id",
                   "persuasive_intensity_std", "tradeoff_disclosure_std",
                   "Q_std_centered", "low_fit"]].copy()
    outcomes["prob_dgp1"] = probs1
    outcomes["prob_dgp2"] = probs2
    outcomes["welfare_dgp2"] = welfare2
    outcomes["prob_dgp3"] = probs3
    outcomes["exp_mismatch_dgp3"] = exp_mismatch3
    outcomes.to_csv(SEMI / "validated_scale_outcomes_mc.csv", index=False)

    print("\nDone. Files saved.")


if __name__ == "__main__":
    main()
