"""06 — Rigorous decomposition audit of the full pilot.

Uses existing outputs only; no new LLM calls.  Produces:
  - decomposition_audit_report.md
  - tables/pairwise_all_comparisons.csv
  - tables/cell_utilities.csv
  - tables/decomposition_effects.csv
  - tables/retrieval_audit.csv
  - tables/expression_audit.csv
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from utils import (
    HIST_DATA, HIST_RESULTS, DATA_DIR,
    load_catalog, load_consumers, load_fit_scores, assign_segment,
)

TABLE_DIR = HIST_RESULTS / "tables"
REPORT_PATH = HIST_RESULTS / "decomposition_audit_report.md"

CELLS = ["00", "10", "01", "11"]
PAIRS_ORDERED = [
    ("10", "00"), ("01", "00"), ("11", "00"),
    ("11", "10"), ("11", "01"), ("10", "01"),
]
B_BOOT = 2000
RNG_SEED = 42


# ===================================================================
# Data loading
# ===================================================================

def load_all():
    supply = pd.read_csv(HIST_DATA / "audit_supply.csv")
    supply["cell"] = supply["cell"].astype(str).str.zfill(2)

    demand = pd.read_csv(HIST_DATA / "pairwise_demand.csv")
    for col in ("cell_i", "cell_j", "cell_as_A", "cell_as_B"):
        if col in demand.columns:
            demand[col] = demand[col].astype(str).str.zfill(2)
    demand["choice_winner"] = demand["choice_winner"].astype(str).str.zfill(2)
    demand.loc[demand["choice_winner"] == "ie", "choice_winner"] = "tie"

    evalu = pd.read_csv(HIST_DATA / "evaluator_scores.csv")

    supply["cluster_id"] = supply["category"] + "_" + supply["consumer_id"].astype(str)
    demand["cluster_id"] = demand["category"] + "_" + demand["consumer_id"].astype(str)
    evalu["cluster_id"] = evalu["row_id"].str.rsplit("_", n=1).str[0]

    return supply, demand, evalu


# ===================================================================
# 1. All six pairwise comparisons
# ===================================================================

def pairwise_table(demand: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell_a, cell_b in PAIRS_ORDERED:
        mask = (
            ((demand["cell_i"] == cell_a) & (demand["cell_j"] == cell_b))
            | ((demand["cell_i"] == cell_b) & (demand["cell_j"] == cell_a))
        )
        sub = demand[mask].copy()
        n = len(sub)
        if n == 0:
            continue
        wins_a = (sub["choice_winner"] == cell_a).sum()
        wins_b = (sub["choice_winner"] == cell_b).sum()
        ties = (sub["choice_winner"] == "tie").sum()
        rows.append({
            "cell_A": cell_a, "cell_B": cell_b, "n": n,
            "A_wins": wins_a, "B_wins": wins_b, "ties": ties,
            "A_win_pct": wins_a / n, "B_win_pct": wins_b / n,
            "tie_pct": ties / n,
            "avg_strength": sub["preference_strength"].mean(),
        })
    return pd.DataFrame(rows)


# ===================================================================
# 2a. Simple pairwise score utility
# ===================================================================

def _cell_score_from_pair(row, cell):
    """Score for `cell` in one pairwise row: 1=win, 0.5=tie, 0=loss."""
    winner = row["choice_winner"]
    ci, cj = row["cell_i"], row["cell_j"]
    if cell not in (ci, cj):
        return np.nan
    if winner == "tie":
        return 0.5
    if winner == cell:
        return 1.0
    return 0.0


def cluster_utilities_simple(demand: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster, per-cell average pairwise score."""
    rows = []
    for cid, grp in demand.groupby("cluster_id"):
        cat = grp["category"].iloc[0]
        consumer_id = int(grp["consumer_id"].iloc[0])
        utils = {}
        for cell in CELLS:
            scores = []
            for _, r in grp.iterrows():
                s = _cell_score_from_pair(r, cell)
                if not np.isnan(s):
                    scores.append(s)
            utils[cell] = np.mean(scores) if scores else np.nan
        u00 = utils.get("00", 0)
        rows.append({
            "cluster_id": cid, "category": cat, "consumer_id": consumer_id,
            "U00_raw": utils.get("00", np.nan),
            "U10_raw": utils.get("10", np.nan),
            "U01_raw": utils.get("01", np.nan),
            "U11_raw": utils.get("11", np.nan),
            "U00": 0.0,
            "U10": utils.get("10", np.nan) - u00,
            "U01": utils.get("01", np.nan) - u00,
            "U11": utils.get("11", np.nan) - u00,
        })
    return pd.DataFrame(rows)


# ===================================================================
# 2b. Bradley-Terry
# ===================================================================

def _bt_nll(params, win_matrix, n_cells):
    theta = np.zeros(n_cells)
    theta[1:] = params
    nll = 0.0
    for i in range(n_cells):
        for j in range(i + 1, n_cells):
            nij = win_matrix[i, j] + win_matrix[j, i]
            if nij == 0:
                continue
            log_denom = np.logaddexp(theta[i], theta[j])
            nll -= win_matrix[i, j] * (theta[i] - log_denom)
            nll -= win_matrix[j, i] * (theta[j] - log_denom)
    return nll


def fit_bradley_terry(demand: pd.DataFrame, cell_order=None):
    if cell_order is None:
        cell_order = CELLS
    cell_idx = {c: i for i, c in enumerate(cell_order)}
    n = len(cell_order)
    W = np.zeros((n, n))

    for _, row in demand.iterrows():
        ci, cj = row["cell_i"], row["cell_j"]
        i, j = cell_idx.get(ci), cell_idx.get(cj)
        if i is None or j is None:
            continue
        winner = row["choice_winner"]
        if winner == ci:
            W[i, j] += 1.0
        elif winner == cj:
            W[j, i] += 1.0
        else:
            W[i, j] += 0.5
            W[j, i] += 0.5

    res = minimize(_bt_nll, np.zeros(n - 1), args=(W, n), method="Nelder-Mead",
                   options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-10})
    theta = np.zeros(n)
    theta[1:] = res.x
    return {cell_order[k]: theta[k] for k in range(n)}


# ===================================================================
# 3. Decomposition with bootstrap
# ===================================================================

def decompose(U: dict) -> dict:
    return {
        "retrieval": U["10"] - U["00"],
        "expression": U["01"] - U["00"],
        "total": U["11"] - U["00"],
        "interaction": U["11"] - U["10"] - U["01"] + U["00"],
    }


def pooled_decomposition(cluster_utils: pd.DataFrame) -> dict:
    U = {
        "00": 0.0,
        "10": cluster_utils["U10"].mean(),
        "01": cluster_utils["U01"].mean(),
        "11": cluster_utils["U11"].mean(),
    }
    return decompose(U)


def bootstrap_decomposition(cluster_utils: pd.DataFrame, rng, B=B_BOOT,
                            approach="simple"):
    cluster_ids = cluster_utils["cluster_id"].unique()
    boot_results = []
    for _ in range(B):
        idx = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sample = cluster_utils[cluster_utils["cluster_id"].isin(idx)]
        counts = pd.Series(idx).value_counts()
        expanded = []
        for cid, cnt in counts.items():
            rows = sample[sample["cluster_id"] == cid]
            for _ in range(cnt):
                expanded.append(rows)
        boot_df = pd.concat(expanded, ignore_index=True)
        boot_results.append(pooled_decomposition(boot_df))
    return pd.DataFrame(boot_results)


def bootstrap_bt(demand: pd.DataFrame, rng, B=B_BOOT):
    cluster_ids = demand["cluster_id"].unique()
    boot_results = []
    for _ in range(B):
        idx = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        counts = pd.Series(idx).value_counts()
        parts = []
        for cid, cnt in counts.items():
            rows = demand[demand["cluster_id"] == cid]
            for _ in range(cnt):
                parts.append(rows)
        boot_demand = pd.concat(parts, ignore_index=True)
        theta = fit_bradley_terry(boot_demand)
        boot_results.append(decompose(theta))
    return pd.DataFrame(boot_results)


def summarize_boot(boot_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in boot_df.columns:
        vals = boot_df[col].dropna()
        rows.append({
            "component": col,
            "mean": vals.mean(),
            "se": vals.std(),
            "ci_lo": vals.quantile(0.025),
            "ci_hi": vals.quantile(0.975),
            "pct_positive": (vals > 0).mean(),
        })
    return pd.DataFrame(rows)


# ===================================================================
# 5. Retrieval audit
# ===================================================================

def retrieval_audit(supply: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat in supply["category"].unique():
        catalog = load_catalog(cat)
        consumers = load_consumers(cat)
        fit_df = load_fit_scores(cat)
        prod_hist = pd.read_csv(HIST_DATA / f"{cat}_product_history.csv")
        seg_hist = pd.read_csv(HIST_DATA / f"{cat}_segment_history.csv")

        prod_lookup = {p["product_id"]: p for p in catalog["products"]}
        fit_lookup = fit_df.set_index(["consumer_id", "product_id"])["Q_std"].to_dict()
        hist_lookup = prod_hist.set_index("product_id").to_dict("index")
        cons_lookup = {c["consumer_id"]: c for c in consumers}

        cat_supply = supply[supply["category"] == cat]
        for _, row in cat_supply.iterrows():
            pid = row["selected_product_id"]
            cid = int(row["consumer_id"])
            product = prod_lookup.get(pid, {})
            hist = hist_lookup.get(pid, {})
            consumer = cons_lookup.get(cid, {})
            segment = assign_segment(consumer, cat, consumers) if consumer else ""
            seg_rows = seg_hist[(seg_hist["segment_id"] == segment) & (seg_hist["product_id"] == pid)]
            seg_cr = seg_rows["conversion_rate"].values[0] if len(seg_rows) > 0 else np.nan

            rows.append({
                "row_id": row["row_id"],
                "category": cat,
                "consumer_id": cid,
                "cell": row["cell"],
                "selector_type": row["selector_type"],
                "selected_product_id": pid,
                "Q_std": fit_lookup.get((cid, pid), np.nan),
                "price": product.get("price", np.nan),
                "quality_score": product.get("quality_score", np.nan),
                "focal_brand": int(product.get("focal_brand", False)),
                "hist_conversion_rate": hist.get("conversion_rate", np.nan),
                "hist_avg_satisfaction": hist.get("avg_satisfaction", np.nan),
                "hist_return_rate": hist.get("return_rate", np.nan),
                "segment_conversion_rate": seg_cr,
            })
    return pd.DataFrame(rows)


# ===================================================================
# 6. Expression audit
# ===================================================================

def expression_audit(supply: pd.DataFrame, evalu: pd.DataFrame) -> pd.DataFrame:
    merged = supply[["row_id", "cluster_id", "category", "consumer_id", "cell"]].merge(
        evalu[["row_id", "persuasive_intensity", "tradeoff_disclosure", "fit_specificity"]],
        on="row_id",
    )
    return merged


def paired_expression_diffs(expr_df: pd.DataFrame) -> pd.DataFrame:
    pivot_pi = expr_df.pivot_table(
        index="cluster_id", columns="cell", values="persuasive_intensity"
    )
    pivot_td = expr_df.pivot_table(
        index="cluster_id", columns="cell", values="tradeoff_disclosure"
    )
    diffs = pd.DataFrame(index=pivot_pi.index)
    labels = [
        ("01_minus_00", "01", "00"),
        ("11_minus_10", "11", "10"),
        ("10_minus_00", "10", "00"),
        ("11_minus_01", "11", "01"),
    ]
    for name, a, b in labels:
        if a in pivot_pi.columns and b in pivot_pi.columns:
            diffs[f"PI_{name}"] = pivot_pi[a] - pivot_pi[b]
        if a in pivot_td.columns and b in pivot_td.columns:
            diffs[f"TD_{name}"] = pivot_td[a] - pivot_td[b]
    return diffs


def diff_summary(diffs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in diffs.columns:
        vals = diffs[col].dropna()
        n = len(vals)
        mean = vals.mean()
        se = vals.std() / np.sqrt(n) if n > 1 else np.nan
        t_stat = mean / se if se > 0 else np.nan
        rows.append({
            "contrast": col, "n": n, "mean": mean, "se": se,
            "t_stat": t_stat,
            "ci_lo": mean - 1.96 * se if not np.isnan(se) else np.nan,
            "ci_hi": mean + 1.96 * se if not np.isnan(se) else np.nan,
        })
    return pd.DataFrame(rows)


# ===================================================================
# Report generation
# ===================================================================

def build_report(pw_table, cluster_utils, bt_pooled, bt_by_cat,
                 decomp_simple, boot_simple, decomp_bt, boot_bt,
                 boot_bt_by_cat,
                 ret_audit, expr_audit_df, diff_summ,
                 ret_by_cell, supply) -> str:
    lines = ["# Decomposition Audit Report — History-Shock Full Pilot\n"]

    # --- Section 1 ---
    lines.append("## 1. All Six Pairwise Comparisons\n")
    lines.append("Directionality: 'A vs B' reports whether A beats B.\n")
    lines.append("| A vs B | n | A wins | B wins | ties | A win% | B win% | tie% | avg strength |")
    lines.append("|--------|---|--------|--------|------|--------|--------|------|-------------|")
    for _, r in pw_table.iterrows():
        lines.append(
            f"| {r['cell_A']} vs {r['cell_B']} | {r['n']} | "
            f"{r['A_wins']} | {r['B_wins']} | {r['ties']} | "
            f"{r['A_win_pct']:.1%} | {r['B_win_pct']:.1%} | "
            f"{r['tie_pct']:.1%} | {r['avg_strength']:.1f} |"
        )

    # --- Section 2 ---
    lines.append("\n## 2. Cell-Level Demand Utility\n")
    lines.append("### Approach A: Simple Pairwise Score\n")
    lines.append("win=1, tie=0.5, loss=0. Per-cluster average, normalized so U(00)=0.\n")
    means = cluster_utils[["U00", "U10", "U01", "U11"]].mean()
    sds = cluster_utils[["U00", "U10", "U01", "U11"]].std()
    lines.append("| Cell | Mean U | SD |")
    lines.append("|------|--------|----|")
    for c in CELLS:
        lines.append(f"| {c} | {means[f'U{c}']:.4f} | {sds[f'U{c}']:.4f} |")

    lines.append("\n### Approach B: Bradley–Terry (pooled)\n")
    lines.append("Ties split as half-win/half-loss. θ(00) = 0 (identified).\n")
    lines.append("| Cell | θ_BT |")
    lines.append("|------|------|")
    for c in CELLS:
        lines.append(f"| {c} | {bt_pooled.get(c, 0):.4f} |")

    lines.append("\n### Bradley–Terry by category\n")
    for cat, bt_cat in bt_by_cat.items():
        line = f"- **{cat}**: " + ", ".join(f"{c}={v:.3f}" for c, v in bt_cat.items())
        lines.append(line)

    # --- Section 3 ---
    lines.append("\n## 3. Decomposition\n")
    lines.append("### Approach A: Simple Pairwise Score (pooled)\n")
    lines.append("| Component | Estimate | Boot SE | 95% CI | P(>0) |")
    lines.append("|-----------|----------|---------|--------|-------|")
    bs_summ = summarize_boot(boot_simple)
    for _, r in bs_summ.iterrows():
        lines.append(
            f"| {r['component']} | {r['mean']:.4f} | {r['se']:.4f} | "
            f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {r['pct_positive']:.1%} |"
        )

    lines.append("\n### Approach B: Bradley–Terry (pooled)\n")
    lines.append("| Component | Estimate | Boot SE | 95% CI | P(>0) |")
    lines.append("|-----------|----------|---------|--------|-------|")
    bt_summ = summarize_boot(boot_bt)
    for _, r in bt_summ.iterrows():
        lines.append(
            f"| {r['component']} | {r['mean']:.4f} | {r['se']:.4f} | "
            f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {r['pct_positive']:.1%} |"
        )

    lines.append("\n### Bradley–Terry by category\n")
    for cat, bt_boot_cat in boot_bt_by_cat.items():
        lines.append(f"\n**{cat}**\n")
        cat_summ = summarize_boot(bt_boot_cat)
        lines.append("| Component | Mean | 95% CI | P(>0) |")
        lines.append("|-----------|------|--------|-------|")
        for _, r in cat_summ.iterrows():
            lines.append(
                f"| {r['component']} | {r['mean']:.4f} | "
                f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {r['pct_positive']:.1%} |"
            )

    # --- Section 4 ---
    lines.append("\n## 4. Interpretation Checks\n")

    bt_d = decompose(bt_pooled)
    bs_s = summarize_boot(boot_simple)
    bs_b = summarize_boot(boot_bt)

    def _check(label, component, threshold=0):
        row_s = bs_s[bs_s["component"] == component].iloc[0]
        row_b = bs_b[bs_b["component"] == component].iloc[0]
        simple_pos = row_s["pct_positive"]
        bt_pos = row_b["pct_positive"]
        s_ci = f"[{row_s['ci_lo']:.4f}, {row_s['ci_hi']:.4f}]"
        b_ci = f"[{row_b['ci_lo']:.4f}, {row_b['ci_hi']:.4f}]"
        stable = "Yes" if min(simple_pos, bt_pos) >= 0.90 else "No"
        return f"- **{label}**: Simple {row_s['mean']:.4f} {s_ci}, BT {row_b['mean']:.4f} {b_ci}. Stable: {stable} (P(>0): {simple_pos:.0%}/{bt_pos:.0%})"

    lines.append(_check("A. Cell 11 > Cell 00 (full bundle > baseline)", "total"))
    lines.append(_check("B. Cell 01 > Cell 00 (expression > baseline)", "expression"))
    lines.append(_check("C. Cell 10 > Cell 00 (retrieval > baseline)", "retrieval"))

    pw_11v01 = pw_table[(pw_table["cell_A"] == "11") & (pw_table["cell_B"] == "01")]
    pw_11v10 = pw_table[(pw_table["cell_A"] == "11") & (pw_table["cell_B"] == "10")]
    if len(pw_11v01) > 0:
        r = pw_11v01.iloc[0]
        lines.append(f"- **D. Cell 11 > Cell 01**: 11 wins {r['A_win_pct']:.0%}, 01 wins {r['B_win_pct']:.0%}, ties {r['tie_pct']:.0%}")
    if len(pw_11v10) > 0:
        r = pw_11v10.iloc[0]
        lines.append(f"- **E. Cell 11 > Cell 10**: 11 wins {r['A_win_pct']:.0%}, 10 wins {r['B_win_pct']:.0%}, ties {r['tie_pct']:.0%}")

    lines.append(_check("F. Interaction positive after cardinalization", "interaction"))

    int_row_s = bs_s[bs_s["component"] == "interaction"].iloc[0]
    int_row_b = bs_b[bs_b["component"] == "interaction"].iloc[0]
    both_stable = min(int_row_s["pct_positive"], int_row_b["pct_positive"]) >= 0.90
    lines.append(f"- **G. Interaction stable under cluster bootstrap**: "
                 f"{'Yes' if both_stable else 'No'} "
                 f"(Simple P(>0)={int_row_s['pct_positive']:.0%}, "
                 f"BT P(>0)={int_row_b['pct_positive']:.0%})")

    # --- Section 5 ---
    lines.append("\n## 5. Retrieval-Side Audit\n")
    ret_by = ret_audit.groupby("cell").agg({
        "Q_std": "mean",
        "price": "mean",
        "quality_score": "mean",
        "focal_brand": "mean",
        "hist_conversion_rate": "mean",
        "hist_avg_satisfaction": "mean",
        "hist_return_rate": "mean",
        "segment_conversion_rate": "mean",
    }).round(4)
    lines.append("### Mean selected-product attributes by cell\n")
    lines.append("| Cell | Q_std | Price | Quality | Focal% | Hist CR | Hist Sat | Hist RetR | Seg CR |")
    lines.append("|------|-------|-------|---------|--------|---------|----------|-----------|--------|")
    for cell in CELLS:
        if cell in ret_by.index:
            r = ret_by.loc[cell]
            lines.append(
                f"| {cell} | {r['Q_std']:.3f} | ${r['price']:.0f} | "
                f"{r['quality_score']:.0f} | {r['focal_brand']:.0%} | "
                f"{r['hist_conversion_rate']:.1%} | {r['hist_avg_satisfaction']:.2f} | "
                f"{r['hist_return_rate']:.1%} | {r['segment_conversion_rate']:.1%} |"
            )

    ret_generic = ret_audit[ret_audit["selector_type"] == "generic"]
    ret_history = ret_audit[ret_audit["selector_type"] == "history"]
    lines.append("\n### Generic vs history selector (pooled across writer type)\n")
    for label, sub in [("Generic selector", ret_generic), ("History selector", ret_history)]:
        lines.append(
            f"- **{label}**: Q_std={sub['Q_std'].mean():.3f}, "
            f"price=${sub['price'].mean():.0f}, "
            f"quality={sub['quality_score'].mean():.0f}, "
            f"hist CR={sub['hist_conversion_rate'].mean():.1%}, "
            f"focal={sub['focal_brand'].mean():.0%}"
        )

    n_clusters = supply["cluster_id"].nunique()
    n_changed = supply[supply["cell"] == "00"].merge(
        supply[supply["cell"] == "10"], on="cluster_id", suffixes=("_00", "_10")
    )
    change_rate = (n_changed["selected_product_id_00"] != n_changed["selected_product_id_10"]).mean()
    lines.append(f"\nProduct change rate (generic → history selector): {change_rate:.0%} of clusters")

    # --- Section 6 ---
    lines.append("\n## 6. Expression-Side Audit\n")
    lines.append("### Mean evaluator scores by cell\n")
    eval_means = expr_audit_df.groupby("cell")[
        ["persuasive_intensity", "tradeoff_disclosure", "fit_specificity"]
    ].mean().round(2)
    lines.append("| Cell | PI | TD | FS |")
    lines.append("|------|----|----|----|")
    for cell in CELLS:
        if cell in eval_means.index:
            r = eval_means.loc[cell]
            lines.append(f"| {cell} | {r['persuasive_intensity']:.2f} | "
                         f"{r['tradeoff_disclosure']:.2f} | {r['fit_specificity']:.2f} |")

    lines.append("\n### Paired differences (cluster-level)\n")
    lines.append("| Contrast | Mean | SE | t | 95% CI |")
    lines.append("|----------|------|----|---|--------|")
    for _, r in diff_summ.iterrows():
        ci = f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]" if not np.isnan(r['ci_lo']) else "N/A"
        t = f"{r['t_stat']:.2f}" if not np.isnan(r['t_stat']) else "N/A"
        lines.append(f"| {r['contrast']} | {r['mean']:.2f} | {r['se']:.2f} | {t} | {ci} |")

    lines.append("\nNote: fit_specificity is exploratory only; PI and TD are the validated measures.")

    # --- Section 7 ---
    lines.append("\n## 7. Interpretation\n")

    total_s = bs_s[bs_s["component"] == "total"].iloc[0]
    expr_s = bs_s[bs_s["component"] == "expression"].iloc[0]
    retr_s = bs_s[bs_s["component"] == "retrieval"].iloc[0]
    inter_s = bs_s[bs_s["component"] == "interaction"].iloc[0]

    total_sig = total_s["ci_lo"] > 0
    expr_sig = expr_s["ci_lo"] > 0
    retr_sig = retr_s["ci_lo"] > 0
    inter_pos_stable = inter_s["pct_positive"] >= 0.90

    if total_sig and inter_pos_stable:
        lines.append(
            "Historical purchase data creates **complementarity** between retrieval "
            "and expression: the full bundle (cell 11) outperforms either component "
            "alone, and the interaction term is positive and stable under bootstrap."
        )
    elif total_sig and expr_sig and not retr_sig:
        lines.append(
            "The treatment effect is **mostly expression-driven**. Cell 01 (history "
            "writer with generic selector) accounts for most of the lift. Retrieval "
            "alone (cell 10) does not reliably beat baseline."
        )
    elif not total_sig:
        lines.append(
            "The full-bundle effect is **not statistically stable**. Neither "
            "retrieval nor expression reliably beats the generic baseline at "
            "conventional confidence levels."
        )
    else:
        lines.append(
            "Results are mixed. See component-level estimates for details."
        )

    pi_01_00 = diff_summ[diff_summ["contrast"] == "PI_01_minus_00"]
    td_01_00 = diff_summ[diff_summ["contrast"] == "TD_01_minus_00"]
    if len(pi_01_00) > 0 and len(td_01_00) > 0:
        pi_val = pi_01_00.iloc[0]["mean"]
        td_val = td_01_00.iloc[0]["mean"]
        if pi_val > 0 and td_val < 0:
            lines.append(
                "\nHistory-informed expression makes recommendations **more persuasive "
                f"(PI +{pi_val:.2f}) and less transparent (TD {td_val:.2f})**."
            )
        elif pi_val > 0 and td_val >= 0:
            lines.append(
                f"\nHistory-informed expression increases persuasiveness (PI +{pi_val:.2f}) "
                f"while maintaining or improving transparency (TD +{td_val:.2f})."
            )

    retr_q = ret_audit.groupby("selector_type")["Q_std"].mean()
    if "generic" in retr_q.index and "history" in retr_q.index:
        q_diff = retr_q["history"] - retr_q["generic"]
        hist_cr_diff = (
            ret_audit[ret_audit["selector_type"] == "history"]["hist_conversion_rate"].mean()
            - ret_audit[ret_audit["selector_type"] == "generic"]["hist_conversion_rate"].mean()
        )
        lines.append(
            f"\nHistory-informed retrieval shifts Q_std by {q_diff:+.3f} and "
            f"historical conversion rate by {hist_cr_diff:+.1%} relative to generic retrieval."
        )
        if q_diff < -0.05:
            lines.append(
                "The history selector **sacrifices consumer-product fit** to chase "
                "historically popular products."
            )
        elif q_diff > 0.05:
            lines.append(
                "The history selector **improves consumer-product fit**, leveraging "
                "revealed-preference signal."
            )

    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================

def main():
    print("Loading data...")
    supply, demand, evalu = load_all()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Pairwise table
    print("1. Pairwise comparisons...")
    pw = pairwise_table(demand)
    pw.to_csv(TABLE_DIR / "pairwise_all_comparisons.csv", index=False)

    # 2a. Simple utilities
    print("2a. Simple pairwise utilities...")
    cu = cluster_utilities_simple(demand)
    cu.to_csv(TABLE_DIR / "cell_utilities.csv", index=False)

    # 2b. BT pooled
    print("2b. Bradley-Terry (pooled)...")
    bt_pooled = fit_bradley_terry(demand)
    bt_by_cat = {}
    for cat in demand["category"].unique():
        bt_by_cat[cat] = fit_bradley_terry(demand[demand["category"] == cat])

    # 3. Decomposition + bootstrap
    rng = np.random.default_rng(RNG_SEED)
    print("3. Bootstrap (simple)...")
    decomp_simple = pooled_decomposition(cu)
    boot_simple = bootstrap_decomposition(cu, rng)

    print("3. Bootstrap (BT pooled)...")
    boot_bt = bootstrap_bt(demand, rng)
    decomp_bt = decompose(bt_pooled)

    print("3. Bootstrap (BT by category)...")
    boot_bt_by_cat = {}
    for cat in demand["category"].unique():
        cat_demand = demand[demand["category"] == cat]
        boot_bt_by_cat[cat] = bootstrap_bt(cat_demand, rng, B=B_BOOT)

    effects = pd.DataFrame([
        {"approach": "simple", **decomp_simple},
        {"approach": "bradley_terry", **decomp_bt},
    ])
    effects.to_csv(TABLE_DIR / "decomposition_effects.csv", index=False)

    # 5. Retrieval audit
    print("5. Retrieval audit...")
    ret = retrieval_audit(supply)
    ret.to_csv(TABLE_DIR / "retrieval_audit.csv", index=False)

    # 6. Expression audit
    print("6. Expression audit...")
    expr_df = expression_audit(supply, evalu)
    diffs = paired_expression_diffs(expr_df)
    diff_summ = diff_summary(diffs)
    expr_df.to_csv(TABLE_DIR / "expression_audit.csv", index=False)

    # Report
    print("7. Generating report...")
    report = build_report(
        pw, cu, bt_pooled, bt_by_cat,
        decomp_simple, boot_simple, decomp_bt, boot_bt,
        boot_bt_by_cat,
        ret, expr_df, diff_summ,
        ret.groupby("cell"), supply,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(f"\nReport: {REPORT_PATH}")
    print(f"Tables: {TABLE_DIR}")

    print("\n" + "=" * 60)
    print("  QUICK SUMMARY")
    print("=" * 60)
    print(f"\nSimple decomposition: {decomp_simple}")
    print(f"BT decomposition:     {decomp_bt}")
    bs = summarize_boot(boot_bt)
    for _, r in bs.iterrows():
        print(f"  {r['component']:12s}: {r['mean']:+.4f}  "
              f"95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
              f"P(>0)={r['pct_positive']:.0%}")


if __name__ == "__main__":
    main()
