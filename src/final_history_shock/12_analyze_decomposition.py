"""12 — Decomposition analysis: Bradley-Terry + cluster bootstrap.

Reads pairwise and absolute evaluation data, computes decomposition
for all outcome variables.

Output:
  data/final_history_shock/analysis/table1_design.csv
  data/final_history_shock/analysis/table2_retrieval_variation.csv
  data/final_history_shock/analysis/table3_pairwise_win_rates.csv
  data/final_history_shock/analysis/table4_bt_decomposition.csv
  data/final_history_shock/analysis/table5_outcome_channels.csv
  data/final_history_shock/analysis/table6_text_mechanisms.csv
  data/final_history_shock/analysis/figure1_decomposition.png
  data/final_history_shock/analysis/figure2_purchase_vs_satisfaction.png

Usage:
  python 12_analyze_decomposition.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_stats import fit_bradley_terry, decompose_from_utilities, cluster_bootstrap_bt
from utils_openai import DATA_DIR

SUPPLY_DIR = DATA_DIR / "local_supply"
EVAL_DIR = DATA_DIR / "gpt_eval"
ANALYSIS_DIR = DATA_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

CELLS = ["00", "10", "01", "11"]
CELL_IDX = {c: i for i, c in enumerate(CELLS)}


def load_data():
    supply_path = SUPPLY_DIR / "final_supply_rows_clean.csv"
    if not supply_path.exists():
        supply_path = SUPPLY_DIR / "final_supply_rows.csv"
    supply = pd.read_csv(supply_path)
    supply["cell"] = supply["cell"].astype(str).str.zfill(2)

    pairwise = pd.read_csv(EVAL_DIR / "pairwise_eval_rows.csv")
    for col in ["cell_i", "cell_j", "cell_as_A", "cell_as_B"]:
        if col in pairwise.columns:
            pairwise[col] = pairwise[col].astype(str).str.zfill(2)

    absolute = None
    abs_path = EVAL_DIR / "absolute_eval_rows.csv"
    if abs_path.exists():
        absolute = pd.read_csv(abs_path)
        absolute["cell"] = absolute["cell"].astype(str).str.zfill(2)

    return supply, pairwise, absolute


# ── Table 1: Design summary ───────────────────────────────────────

def make_table1(supply: pd.DataFrame, pairwise: pd.DataFrame,
                absolute: pd.DataFrame | None) -> pd.DataFrame:
    return pd.DataFrame([{
        "categories": supply["category"].nunique(),
        "products_per_category": 25,
        "personas_per_category": supply.groupby("category")["persona_id"].nunique().mean(),
        "clusters": supply["cluster_id"].nunique(),
        "local_supply_packages": len(supply),
        "gpt_pairwise_judgments": len(pairwise),
        "gpt_absolute_evaluations": len(absolute) if absolute is not None else 0,
    }])


# ── Table 2: Retrieval variation ──────────────────────────────────

def make_table2(supply: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat in list(supply["category"].unique()) + ["overall"]:
        sub = supply if cat == "overall" else supply[supply["category"] == cat]
        ret_changed = sub.groupby("cluster_id")["retrieval_changed"].first()

        generic = sub[sub["retrieval_condition"] == "generic"]
        history = sub[sub["retrieval_condition"] == "history"]

        def _concentration(df):
            counts = df["selected_product_id"].value_counts(normalize=True)
            top_share = counts.iloc[0] if len(counts) > 0 else 0
            entropy = -(counts * np.log2(counts + 1e-10)).sum()
            hhi = (counts ** 2).sum()
            return top_share, entropy, hhi

        g_top, g_ent, g_hhi = _concentration(generic)
        h_top, h_ent, h_hhi = _concentration(history)

        rows.append({
            "category": cat,
            "retrieval_change_rate": ret_changed.mean(),
            "generic_top_product_share": g_top,
            "history_top_product_share": h_top,
            "generic_entropy": g_ent,
            "history_entropy": h_ent,
            "generic_hhi": g_hhi,
            "history_hhi": h_hhi,
        })

    return pd.DataFrame(rows)


# ── Table 3: Pairwise win rates ──────────────────────────────────

def make_table3(pairwise: pd.DataFrame, outcome: str = "overall_winner_cell") -> pd.DataFrame:
    rows = []
    for (ci, cj), grp in pairwise.groupby(["cell_i", "cell_j"]):
        n = len(grp)
        wins_i = (grp[outcome] == ci).sum()
        wins_j = (grp[outcome] == cj).sum()
        ties = (grp[outcome] == "tie").sum()
        rows.append({
            "cell_A": ci, "cell_B": cj, "n": n,
            "A_wins": int(wins_i), "B_wins": int(wins_j), "ties": int(ties),
            "A_win_pct": wins_i / n, "B_win_pct": wins_j / n, "tie_pct": ties / n,
            "net_B_win_rate": (wins_j - wins_i) / n,
        })
    return pd.DataFrame(rows)


# ── Table 4: BT decomposition ────────────────────────────────────

def make_table4(pairwise: pd.DataFrame, outcome: str = "overall_winner_cell",
                B: int = 1000) -> pd.DataFrame:
    K = len(CELLS)
    win_mat = np.zeros((K, K))
    tie_mat = np.zeros((K, K))

    for _, row in pairwise.iterrows():
        ci = CELL_IDX.get(row["cell_i"])
        cj = CELL_IDX.get(row["cell_j"])
        if ci is None or cj is None:
            continue
        winner = row[outcome]
        if winner == row["cell_i"]:
            win_mat[ci, cj] += 1
        elif winner == row["cell_j"]:
            win_mat[cj, ci] += 1
        else:
            tie_mat[ci, cj] += 1
            tie_mat[cj, ci] += 1

    theta = fit_bradley_terry(win_mat, tie_mat)
    point = decompose_from_utilities(theta)

    boot = cluster_bootstrap_bt(pairwise, B=B, winner_col=outcome)

    boot["point_estimate"] = boot["component"].map(point)
    return boot


# ── Table 5: Multi-outcome decomposition ─────────────────────────

def make_table5(pairwise: pd.DataFrame) -> pd.DataFrame:
    outcomes = {
        "overall": "overall_winner_cell",
        "purchase": "purchase_winner_cell",
        "satisfaction": "satisfaction_winner_cell",
        "trust": "trust_winner_cell",
    }
    all_rows = []
    for label, col in outcomes.items():
        if col not in pairwise.columns:
            continue
        bt = make_table4(pairwise, outcome=col, B=500)
        bt["outcome"] = label
        all_rows.append(bt)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


# ── Table 6: Text mechanisms ─────────────────────────────────────

def make_table6(absolute: pd.DataFrame | None) -> pd.DataFrame | None:
    if absolute is None:
        return None

    metrics = ["persuasive_intensity_1_7", "tradeoff_disclosure_1_7",
               "regret_risk_1_7", "trust_score_1_7"]
    available = [m for m in metrics if m in absolute.columns]

    rows = []
    for cell in CELLS:
        cell_data = absolute[absolute["cell"] == cell]
        row = {"cell": cell}
        for m in available:
            row[m] = cell_data[m].mean()
        rows.append(row)

    return pd.DataFrame(rows)


# ── Figures ───────────────────────────────────────────────────────

def make_figures(table4: pd.DataFrame, table5: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figures")
        return

    # Figure 1: Decomposition bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    comps = table4[table4["component"].isin(["retrieval", "expression", "interaction", "total"])]
    colors = {"retrieval": "#2196F3", "expression": "#FF9800",
              "interaction": "#9C27B0", "total": "#4CAF50"}

    x = range(len(comps))
    bars = ax.bar(x, comps["mean"], yerr=[comps["mean"] - comps["ci_lo"],
                                           comps["ci_hi"] - comps["mean"]],
                  color=[colors.get(c, "gray") for c in comps["component"]],
                  capsize=5, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(comps["component"], fontsize=11)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Bradley-Terry Effect Size", fontsize=12)
    ax.set_title("History-Shock Decomposition (Overall Winner)", fontsize=13)

    for i, (_, row) in enumerate(comps.iterrows()):
        ax.text(i, row["ci_hi"] + 0.02, f"P>0: {row['p_positive']:.0%}",
                ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "figure1_decomposition.png", dpi=150)
    plt.close()
    print("  Figure 1 saved")

    # Figure 2: Multi-outcome decomposition
    if len(table5) == 0:
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    outcomes = table5["outcome"].unique()

    for i, outcome in enumerate(outcomes[:4]):
        ax = axes[i] if i < len(axes) else axes[-1]
        sub = table5[(table5["outcome"] == outcome) &
                     table5["component"].isin(["retrieval", "expression", "total"])]
        if len(sub) == 0:
            continue

        x = range(len(sub))
        ax.bar(x, sub["mean"],
               yerr=[sub["mean"] - sub["ci_lo"], sub["ci_hi"] - sub["mean"]],
               color=[colors.get(c, "gray") for c in sub["component"]],
               capsize=4, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["component"], fontsize=9, rotation=30)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(outcome.capitalize(), fontsize=11)

    axes[0].set_ylabel("BT Effect Size", fontsize=11)
    fig.suptitle("Decomposition by Outcome Dimension", fontsize=13)
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "figure2_purchase_vs_satisfaction.png", dpi=150)
    plt.close()
    print("  Figure 2 saved")


# ── Main ──────────────────────────────────────────────────────────

def main():
    supply, pairwise, absolute = load_data()
    print(f"Supply: {len(supply)} rows, Pairwise: {len(pairwise)} rows, "
          f"Absolute: {len(absolute) if absolute is not None else 0} rows")

    # Cell invariant check
    n_violations = 0
    for cid, grp in supply.groupby("cluster_id"):
        cells = grp.set_index("cell")
        if "00" in cells.index and "01" in cells.index:
            if cells.loc["00", "selected_product_id"] != cells.loc["01", "selected_product_id"]:
                n_violations += 1
        if "10" in cells.index and "11" in cells.index:
            if cells.loc["10", "selected_product_id"] != cells.loc["11", "selected_product_id"]:
                n_violations += 1
    print(f"Cell invariant violations: {n_violations}")

    print("\nGenerating tables...")
    t1 = make_table1(supply, pairwise, absolute)
    t1.to_csv(ANALYSIS_DIR / "table1_design.csv", index=False)
    print("  Table 1: design summary")

    t2 = make_table2(supply)
    t2.to_csv(ANALYSIS_DIR / "table2_retrieval_variation.csv", index=False)
    print(f"  Table 2: retrieval variation")
    for _, r in t2.iterrows():
        print(f"    {r['category']}: change rate {r['retrieval_change_rate']:.1%}")

    t3 = make_table3(pairwise)
    t3.to_csv(ANALYSIS_DIR / "table3_pairwise_win_rates.csv", index=False)
    print(f"  Table 3: pairwise win rates")

    print("  Table 4: BT decomposition (bootstrapping B=1000)...")
    t4 = make_table4(pairwise, B=1000)
    t4.to_csv(ANALYSIS_DIR / "table4_bt_decomposition.csv", index=False)
    for _, r in t4.iterrows():
        print(f"    {r['component']}: {r['mean']:.3f} "
              f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] P>0={r['p_positive']:.1%}")

    print("  Table 5: multi-outcome decomposition...")
    t5 = make_table5(pairwise)
    if len(t5) > 0:
        t5.to_csv(ANALYSIS_DIR / "table5_outcome_channels.csv", index=False)

    t6 = make_table6(absolute)
    if t6 is not None:
        t6.to_csv(ANALYSIS_DIR / "table6_text_mechanisms.csv", index=False)
        print("  Table 6: text mechanisms")

    print("\nGenerating figures...")
    make_figures(t4, t5)

    print(f"\nAll analysis saved → {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
