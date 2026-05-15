"""BT decomposition analysis for history-shock 2x2 design.

Reads supply + evaluation CSVs, computes:
  - Pairwise win rates and BT parameters
  - Retrieval / expression / interaction decomposition
  - Cluster bootstrap CIs (B=2000)
  - Absolute evaluation summaries by cell

Usage:
  python analyze.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

OUT_DIR = Path(os.environ.get("DATA_DIR", os.path.expanduser("~/llmrec_results")))
SUPPLY_CSV = OUT_DIR / "final_supply_rows.csv"
ABS_CSV = OUT_DIR / "absolute_eval_rows.csv"
PAIR_CSV = OUT_DIR / "pairwise_eval_rows.csv"
ANALYSIS_DIR = OUT_DIR / "analysis"

B_BOOTSTRAP = 2000
SEED = 42


def bt_mle(wins: np.ndarray, n_items: int = 4) -> np.ndarray:
    if wins.shape != (n_items, n_items):
        raise ValueError(f"Expected {n_items}x{n_items} win matrix")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        def neg_ll(params):
            p = np.exp(params)
            ll = 0
            for i in range(n_items):
                for j in range(i + 1, n_items):
                    nij = wins[i, j] + wins[j, i]
                    if nij == 0:
                        continue
                    prob_i = p[i] / (p[i] + p[j])
                    ll += wins[i, j] * np.log(prob_i + 1e-12)
                    ll += wins[j, i] * np.log(1 - prob_i + 1e-12)
            return -ll

        x0 = np.zeros(n_items)
        x0[0] = 0
        result = minimize(neg_ll, x0, method="Nelder-Mead",
                          options={"maxiter": 5000, "xatol": 1e-8})
        params = result.x - result.x[0]
    return params


def build_win_matrix(pw: pd.DataFrame, winner_col: str = "overall_winner_cell") -> np.ndarray:
    cells = ["00", "10", "01", "11"]
    cell_idx = {c: i for i, c in enumerate(cells)}
    W = np.zeros((4, 4))
    for _, row in pw.iterrows():
        ci = str(row["cell_i"]).zfill(2)
        cj = str(row["cell_j"]).zfill(2)
        winner = str(row[winner_col]).zfill(2) if row[winner_col] != "tie" else "tie"
        if ci not in cell_idx or cj not in cell_idx:
            continue
        if winner == ci:
            W[cell_idx[ci], cell_idx[cj]] += 1
        elif winner == cj:
            W[cell_idx[cj], cell_idx[ci]] += 1
        else:
            W[cell_idx[ci], cell_idx[cj]] += 0.5
            W[cell_idx[cj], cell_idx[ci]] += 0.5
    return W


def decompose(theta: np.ndarray) -> dict:
    retrieval = 0.5 * ((theta[1] - theta[0]) + (theta[3] - theta[2]))
    expression = 0.5 * ((theta[2] - theta[0]) + (theta[3] - theta[1]))
    interaction = theta[3] - theta[2] - theta[1] + theta[0]
    return {"retrieval": retrieval, "expression": expression, "interaction": interaction}


def cluster_bootstrap(pw: pd.DataFrame, winner_col: str, B: int = B_BOOTSTRAP) -> dict:
    rng = np.random.default_rng(SEED)
    clusters = pw["cluster_id"].unique()
    boot_r, boot_e, boot_i = [], [], []

    for _ in range(B):
        idx = rng.choice(len(clusters), size=len(clusters), replace=True)
        boot_clusters = clusters[idx]
        boot_pw = pd.concat([pw[pw["cluster_id"] == c] for c in boot_clusters],
                            ignore_index=True)
        W = build_win_matrix(boot_pw, winner_col)
        try:
            theta = bt_mle(W)
            d = decompose(theta)
            boot_r.append(d["retrieval"])
            boot_e.append(d["expression"])
            boot_i.append(d["interaction"])
        except Exception:
            continue

    def summarize(vals):
        arr = np.array(vals)
        return {
            "mean": float(np.mean(arr)),
            "se": float(np.std(arr, ddof=1)),
            "ci_lo": float(np.percentile(arr, 2.5)),
            "ci_hi": float(np.percentile(arr, 97.5)),
            "p_positive": float(np.mean(arr > 0)),
        }

    return {
        "retrieval": summarize(boot_r),
        "expression": summarize(boot_e),
        "interaction": summarize(boot_i),
        "B_actual": len(boot_r),
    }


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    if not PAIR_CSV.exists():
        sys.exit(f"Pairwise data not found: {PAIR_CSV}")

    pw = pd.read_csv(PAIR_CSV)
    pw["cell_i"] = pw["cell_i"].astype(str).str.zfill(2)
    pw["cell_j"] = pw["cell_j"].astype(str).str.zfill(2)

    print(f"Loaded {len(pw)} pairwise rows, {pw['cluster_id'].nunique()} clusters")

    # Point estimates
    W = build_win_matrix(pw)
    theta = bt_mle(W)
    cells = ["00", "10", "01", "11"]
    print(f"\nBT parameters: {dict(zip(cells, theta.round(3)))}")

    d = decompose(theta)
    print(f"Decomposition: R={d['retrieval']:.3f}, E={d['expression']:.3f}, "
          f"I={d['interaction']:.3f}")

    # Bootstrap
    print(f"\nBootstrapping (B={B_BOOTSTRAP})...")
    outcomes = ["overall_winner_cell", "purchase_winner_cell",
                "satisfaction_winner_cell", "trust_winner_cell"]

    all_results = {}
    for oc in outcomes:
        if oc not in pw.columns:
            continue
        print(f"  {oc}...")
        boot = cluster_bootstrap(pw, oc)
        all_results[oc] = boot

    # Save decomposition table
    rows = []
    for oc, res in all_results.items():
        for component in ["retrieval", "expression", "interaction"]:
            s = res[component]
            rows.append({
                "outcome": oc.replace("_winner_cell", ""),
                "component": component,
                "estimate": round(s["mean"], 4),
                "se": round(s["se"], 4),
                "ci_lo": round(s["ci_lo"], 4),
                "ci_hi": round(s["ci_hi"], 4),
                "p_positive": round(s["p_positive"], 3),
            })
    decomp_df = pd.DataFrame(rows)
    decomp_df.to_csv(ANALYSIS_DIR / "decomposition_bootstrap.csv", index=False)
    print(f"\n{decomp_df.to_string(index=False)}")

    # Win rates
    print("\n--- Win Rates ---")
    for ci, cj in [("00", "10"), ("00", "01"), ("00", "11"),
                    ("10", "01"), ("10", "11"), ("01", "11")]:
        sub = pw[((pw["cell_i"] == ci) & (pw["cell_j"] == cj)) |
                 ((pw["cell_i"] == cj) & (pw["cell_j"] == ci))]
        if len(sub) == 0:
            continue
        wins_i = (sub["overall_winner_cell"] == ci).sum()
        wins_j = (sub["overall_winner_cell"] == cj).sum()
        ties = (sub["overall_winner_cell"] == "tie").sum()
        total = len(sub)
        print(f"  {ci} vs {cj}: {ci} wins {wins_i}/{total} ({wins_i/total:.0%}), "
              f"{cj} wins {wins_j}/{total} ({wins_j/total:.0%}), "
              f"ties {ties}/{total}")

    win_rates = []
    for ci, cj in [("00", "10"), ("00", "01"), ("00", "11"),
                    ("10", "01"), ("10", "11"), ("01", "11")]:
        sub = pw[((pw["cell_i"] == ci) & (pw["cell_j"] == cj)) |
                 ((pw["cell_i"] == cj) & (pw["cell_j"] == ci))]
        if len(sub) == 0:
            continue
        win_rates.append({
            "cell_i": ci, "cell_j": cj,
            "win_i": int((sub["overall_winner_cell"] == ci).sum()),
            "win_j": int((sub["overall_winner_cell"] == cj).sum()),
            "ties": int((sub["overall_winner_cell"] == "tie").sum()),
            "n": len(sub),
        })
    pd.DataFrame(win_rates).to_csv(ANALYSIS_DIR / "win_rates.csv", index=False)

    # Absolute evaluation summary
    if ABS_CSV.exists():
        abs_df = pd.read_csv(ABS_CSV)
        abs_df["cell"] = abs_df["cell"].astype(str).str.zfill(2)
        print("\n--- Absolute Evaluation by Cell ---")
        score_cols = [c for c in abs_df.columns if c.endswith(("_1_7", "_0_100"))]
        summary = abs_df.groupby("cell")[score_cols].mean().round(2)
        print(summary.to_string())
        summary.to_csv(ANALYSIS_DIR / "absolute_eval_by_cell.csv")

    # Save full results as JSON
    with open(ANALYSIS_DIR / "decomposition_full.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAnalysis saved → {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
