"""
Robust mechanism audit with clustered inference and retrieval-side verification.

Steps 1-2 of the robustness/repair pass.
- Cluster-robust SEs by (category, consumer_id)
- Paired within-consumer differences for r and q effects
- Retrieval-side product metadata verification
"""

import json, pathlib, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
REPORTS = RESULTS / "reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(RESULTS / "diagnostics" / "evaluator_scores.csv")
    df["cluster_id"] = df["category"] + "_" + df["consumer_id"].astype(str)
    return df


def load_catalogs():
    cats = {}
    for cat in ["phone_charger", "headphones", "laptop"]:
        with open(DATA / "catalogs" / f"{cat}.json") as f:
            d = json.load(f)
        prod_list = d["products"]
        cats[cat] = {p["product_id"]: p for p in prod_list}
    return cats


# ── Step 1: Clustered mechanism audit ──────────────────────────────

def paired_differences(df, scale, by_var):
    """Compute within-consumer paired differences for a binary variable."""
    rows = []
    for cid, grp in df.groupby("cluster_id"):
        cat = grp["category"].iloc[0]
        cons = grp["consumer_id"].iloc[0]
        m1 = grp.loc[grp[by_var] == 1, scale].mean()
        m0 = grp.loc[grp[by_var] == 0, scale].mean()
        rows.append({"cluster_id": cid, "category": cat, "consumer_id": cons,
                      "diff": m1 - m0, "mean_1": m1, "mean_0": m0})
    return pd.DataFrame(rows)


def compute_clustered_effects(df):
    """Mean effects with paired inference and cluster-bootstrap CIs."""
    records = []
    rng = np.random.default_rng(20260514)

    for scale in ["persuasive_intensity", "tradeoff_disclosure"]:
        # r effect (paired)
        pd_r = paired_differences(df, scale, "r")
        diffs = pd_r["diff"].values
        n = len(diffs)
        mean_diff = diffs.mean()
        se_paired = diffs.std(ddof=1) / np.sqrt(n)
        t_stat = mean_diff / se_paired
        p_val = 2 * stats.t.sf(abs(t_stat), df=n - 1)

        # cluster bootstrap SE (2000 reps)
        boot_means = np.array([rng.choice(diffs, size=n, replace=True).mean()
                               for _ in range(2000)])
        se_boot = boot_means.std()

        records.append({
            "scale": scale, "effect": "r_effect",
            "mean_diff": round(mean_diff, 4),
            "mean_r0": round(pd_r["mean_0"].mean(), 4),
            "mean_r1": round(pd_r["mean_1"].mean(), 4),
            "se_paired": round(se_paired, 4),
            "se_bootstrap": round(se_boot, 4),
            "t_paired": round(t_stat, 3),
            "p_paired": round(p_val, 6),
            "n_clusters": n
        })

        # q effect holding r fixed (paired within r=0 and r=1)
        for r_val in [0, 1]:
            sub = df[df["r"] == r_val]
            pd_q = paired_differences(sub, scale, "q")
            diffs_q = pd_q["diff"].values
            mean_q = diffs_q.mean()
            se_q = diffs_q.std(ddof=1) / np.sqrt(len(diffs_q))
            t_q = mean_q / se_q if se_q > 0 else 0
            p_q = 2 * stats.t.sf(abs(t_q), df=len(diffs_q) - 1)

            records.append({
                "scale": scale, "effect": f"q_effect_r{r_val}",
                "mean_diff": round(mean_q, 4),
                "mean_r0": np.nan, "mean_r1": np.nan,
                "se_paired": round(se_q, 4),
                "se_bootstrap": np.nan,
                "t_paired": round(t_q, 3),
                "p_paired": round(p_q, 6),
                "n_clusters": len(diffs_q)
            })

    # category heterogeneity (paired within category)
    for cat in df["category"].unique():
        sub = df[df["category"] == cat]
        for scale in ["persuasive_intensity", "tradeoff_disclosure"]:
            pd_r = paired_differences(sub, scale, "r")
            diffs = pd_r["diff"].values
            n = len(diffs)
            mean_diff = diffs.mean()
            se = diffs.std(ddof=1) / np.sqrt(n)
            t_stat = mean_diff / se if se > 0 else 0
            p_val = 2 * stats.t.sf(abs(t_stat), df=n - 1)

            records.append({
                "scale": scale, "effect": f"r_effect_{cat}",
                "mean_diff": round(mean_diff, 4),
                "mean_r0": round(pd_r["mean_0"].mean(), 4),
                "mean_r1": round(pd_r["mean_1"].mean(), 4),
                "se_paired": round(se, 4),
                "se_bootstrap": np.nan,
                "t_paired": round(t_stat, 3),
                "p_paired": round(p_val, 6),
                "n_clusters": n
            })

    return pd.DataFrame(records)


def clustered_regressions(df):
    """OLS regressions with cluster-robust SEs."""
    records = []
    df = df.copy()
    df["qxr"] = df["q"] * df["r"]
    cat_dummies = pd.get_dummies(df["category"], drop_first=True, dtype=float)

    for scale in ["persuasive_intensity", "tradeoff_disclosure"]:
        y = df[scale].values
        X = sm.add_constant(pd.concat([df[["q", "r", "qxr"]], cat_dummies], axis=1))

        model = sm.OLS(y, X).fit(
            cov_type="cluster",
            cov_kwds={"groups": df["cluster_id"].values}
        )

        # also fit with HC1 for comparison
        model_hc1 = sm.OLS(y, X).fit(cov_type="HC1")

        rec = {"scale": scale, "n": int(model.nobs), "n_clusters": df["cluster_id"].nunique(),
               "R2": round(model.rsquared, 4)}

        for var in ["q", "r", "qxr"]:
            rec[f"coef_{var}"] = round(model.params[var], 4)
            rec[f"se_cluster_{var}"] = round(model.bse[var], 4)
            rec[f"t_cluster_{var}"] = round(model.tvalues[var], 3)
            rec[f"p_cluster_{var}"] = round(model.pvalues[var], 6)
            rec[f"se_hc1_{var}"] = round(model_hc1.bse[var], 4)
            rec[f"t_hc1_{var}"] = round(model_hc1.tvalues[var], 3)

        records.append(rec)

    return pd.DataFrame(records)


# ── Step 2: Retrieval-side verification ────────────────────────────

def verify_retrieval(df, catalogs):
    """Verify retrieval-side product metadata and compute statistics."""
    records = []
    issues = []

    for cat in df["category"].unique():
        cat_prods = catalogs[cat]
        sub = df[df["category"] == cat]

        for q_val in [0, 1]:
            cell = sub[sub["q"] == q_val]
            selected = cell["product_id"].values

            # verify each product_id belongs to the correct category catalog
            for pid in selected:
                if pid not in cat_prods:
                    issues.append(f"MISMATCH: {pid} not in {cat} catalog")

            # compute product-level stats using category-specific metadata
            prices = [cat_prods[pid]["price"] for pid in selected if pid in cat_prods]
            qualities = [cat_prods[pid]["quality_score"] for pid in selected if pid in cat_prods]
            incumbents = [1 if cat_prods[pid].get("brand_status") == "incumbent" else 0
                          for pid in selected if pid in cat_prods]
            focals = [1 if cat_prods[pid].get("focal_brand", False) else 0
                      for pid in selected if pid in cat_prods]

            # product distribution
            unique_prods = pd.Series(selected).value_counts()
            n_unique = len(unique_prods)

            # focal share
            focal_count = sum(focals)
            focal_share = focal_count / len(selected) if selected.size else 0

            # incumbent share
            inc_count = sum(incumbents)
            inc_share = inc_count / len(selected) if selected.size else 0

            records.append({
                "category": cat,
                "q": q_val,
                "n": len(selected),
                "n_unique_products": n_unique,
                "focal_share": round(focal_share, 3),
                "incumbent_share": round(inc_share, 3),
                "mean_price": round(np.mean(prices), 2) if prices else np.nan,
                "mean_quality": round(np.mean(qualities), 2) if qualities else np.nan,
                "mean_Q_std": round(cell["Q_std"].mean(), 4),
                "sd_Q_std": round(cell["Q_std"].std(), 4),
            })

    # compute TVD per category
    tvd_records = []
    for cat in df["category"].unique():
        sub = df[df["category"] == cat]
        q0_prods = sub[sub["q"] == 0]["product_id"]
        q1_prods = sub[sub["q"] == 1]["product_id"]
        all_prods = set(q0_prods) | set(q1_prods)

        q0_dist = q0_prods.value_counts(normalize=True)
        q1_dist = q1_prods.value_counts(normalize=True)

        tvd = 0
        for p in all_prods:
            tvd += abs(q0_dist.get(p, 0) - q1_dist.get(p, 0))
        tvd /= 2

        tvd_records.append({"category": cat, "TVD": round(tvd, 3)})

    retrieval_df = pd.DataFrame(records)

    # merge TVD
    tvd_df = pd.DataFrame(tvd_records)
    retrieval_df = retrieval_df.merge(tvd_df, on="category", how="left")

    if issues:
        print("RETRIEVAL VERIFICATION ISSUES:")
        for iss in issues:
            print(f"  {iss}")
    else:
        print("Retrieval verification: all product_ids match their category catalogs.")

    return retrieval_df, issues


# ── Step 6 (preview): Expression-fit correlations ──────────────────

def expression_fit_correlations(df):
    """Correlations between PI/TD and Q_std for bias interpretation."""
    records = []
    for scale in ["persuasive_intensity", "tradeoff_disclosure"]:
        r, p = stats.pearsonr(df[scale], df["Q_std"])
        rho, sp = stats.spearmanr(df[scale], df["Q_std"])
        records.append({"scale": scale, "slice": "overall", "n": len(df),
                         "pearson_r": round(r, 4), "pearson_p": round(p, 4),
                         "spearman_rho": round(rho, 4), "spearman_p": round(sp, 4)})

        for q_val in [0, 1]:
            for r_val in [0, 1]:
                sub = df[(df["q"] == q_val) & (df["r"] == r_val)]
                r2, p2 = stats.pearsonr(sub[scale], sub["Q_std"])
                rho2, sp2 = stats.spearmanr(sub[scale], sub["Q_std"])
                records.append({"scale": scale, "slice": f"q={q_val},r={r_val}",
                                 "n": len(sub),
                                 "pearson_r": round(r2, 4), "pearson_p": round(p2, 4),
                                 "spearman_rho": round(rho2, 4), "spearman_p": round(sp2, 4)})

        for cat in df["category"].unique():
            sub = df[df["category"] == cat]
            r2, p2 = stats.pearsonr(sub[scale], sub["Q_std"])
            rho2, sp2 = stats.spearmanr(sub[scale], sub["Q_std"])
            records.append({"scale": scale, "slice": f"cat={cat}", "n": len(sub),
                             "pearson_r": round(r2, 4), "pearson_p": round(p2, 4),
                             "spearman_rho": round(rho2, 4), "spearman_p": round(sp2, 4)})

    return pd.DataFrame(records)


def main():
    print("Loading data...")
    df = load_data()
    catalogs = load_catalogs()

    print("\n=== Step 1: Clustered mechanism audit ===")
    effects = compute_clustered_effects(df)
    effects.to_csv(TABLES / "validated_mechanism_clustered.csv", index=False)
    print(effects.to_string(index=False))

    print("\n=== Step 1b: Clustered regressions ===")
    regs = clustered_regressions(df)
    regs.to_csv(TABLES / "validated_modularity_clustered_regressions.csv", index=False)
    print(regs.to_string(index=False))

    print("\n=== Step 2: Retrieval-side verification ===")
    retrieval, issues = verify_retrieval(df, catalogs)
    retrieval.to_csv(TABLES / "retrieval_mechanism_verified.csv", index=False)
    print(retrieval.to_string(index=False))

    print("\n=== Step 6: Expression-fit correlations ===")
    corrs = expression_fit_correlations(df)
    corrs.to_csv(TABLES / "expression_fit_correlations.csv", index=False)
    print(corrs.to_string(index=False))


if __name__ == "__main__":
    main()
