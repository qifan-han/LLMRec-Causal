"""05 — Retrieval / expression / interaction decomposition.

Combines audit supply, evaluator scores, and pairwise demand data to
compute causal components of the history-shock treatment.

Outputs:
  results/history_shock/tables/decomposition_summary.csv
  results/history_shock/tables/pairwise_win_rates.csv
  results/history_shock/tables/evaluator_means.csv
  results/history_shock/tables/retrieval_agreement.csv

Usage:
  python 05_analyze_decomposition.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import HIST_DATA, HIST_RESULTS

SUPPLY_CSV = HIST_DATA / "audit_supply.csv"
EVAL_CSV = HIST_DATA / "evaluator_scores.csv"
DEMAND_CSV = HIST_DATA / "pairwise_demand.csv"
TABLE_DIR = HIST_RESULTS / "tables"


def load_data():
    for path, name in [(SUPPLY_CSV, "audit_supply"), (DEMAND_CSV, "pairwise_demand")]:
        if not path.exists():
            sys.exit(f"{name} not found: {path}")
    supply = pd.read_csv(SUPPLY_CSV)
    supply["cell"] = supply["cell"].astype(str).str.zfill(2)
    demand = pd.read_csv(DEMAND_CSV)
    for col in ("cell_i", "cell_j", "cell_as_A", "cell_as_B"):
        if col in demand.columns:
            demand[col] = demand[col].astype(str).str.zfill(2)
    evaluator = pd.read_csv(EVAL_CSV) if EVAL_CSV.exists() else None
    return supply, evaluator, demand


def assert_cell_invariants(supply: pd.DataFrame):
    """Verify the factored design before any analysis."""
    n_violations = 0
    for (cat, cid), grp in supply.groupby(["category", "consumer_id"]):
        cells = grp.set_index("cell")
        present = set(cells.index)
        if present != {"00", "10", "01", "11"}:
            missing = {"00", "10", "01", "11"} - present
            print(f"  WARN: {cat} consumer {cid} missing cells: {missing}")
            n_violations += 1
            continue

        pid = cells["selected_product_id"]
        st = cells["selector_type"]
        wt = cells["writer_type"]

        if pid["00"] != pid["01"]:
            print(f"  FAIL: {cat} c{cid}: cells 00 and 01 differ in product "
                  f"({pid['00']} vs {pid['01']})")
            n_violations += 1
        if pid["10"] != pid["11"]:
            print(f"  FAIL: {cat} c{cid}: cells 10 and 11 differ in product "
                  f"({pid['10']} vs {pid['11']})")
            n_violations += 1

        if st["00"] != "generic" or st["01"] != "generic":
            print(f"  FAIL: {cat} c{cid}: cells 00/01 selector not generic")
            n_violations += 1
        if st["10"] != "history" or st["11"] != "history":
            print(f"  FAIL: {cat} c{cid}: cells 10/11 selector not history")
            n_violations += 1

        if wt["00"] != "generic" or wt["10"] != "generic":
            print(f"  FAIL: {cat} c{cid}: cells 00/10 writer not generic")
            n_violations += 1
        if wt["01"] != "history" or wt["11"] != "history":
            print(f"  FAIL: {cat} c{cid}: cells 01/11 writer not history")
            n_violations += 1

    if n_violations > 0:
        sys.exit(f"\nCell-construction invariant FAILED with {n_violations} violation(s). "
                 "Fix audit data before analysis.")
    print(f"  Cell-construction invariant: PASSED "
          f"({supply['category'].nunique()} categories, "
          f"{len(supply) // 4} clusters)")


# ---------------------------------------------------------------------------
# Retrieval agreement
# ---------------------------------------------------------------------------

def retrieval_agreement(supply: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cat, cid), grp in supply.groupby(["category", "consumer_id"]):
        pids = grp.set_index("cell")["selected_product_id"]
        if "00" not in pids.index or "10" not in pids.index:
            continue
        rows.append({
            "category": cat,
            "consumer_id": cid,
            "pid_generic": pids["00"],
            "pid_history": pids["10"],
            "same_product": int(pids["00"] == pids["10"]),
        })
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Pairwise win rates
# ---------------------------------------------------------------------------

def pairwise_win_rates(demand: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ci, cj), grp in demand.groupby(["cell_i", "cell_j"]):
        n = len(grp)
        wins_i = (grp["choice_winner"] == ci).sum()
        wins_j = (grp["choice_winner"] == cj).sum()
        ties = (grp["choice_winner"] == "tie").sum()
        avg_strength = grp["preference_strength"].mean()

        fit_i = (grp["better_fit"] == ci).sum()
        fit_j = (grp["better_fit"] == cj).sum()
        trust_i = (grp["more_trustworthy"] == ci).sum()
        trust_j = (grp["more_trustworthy"] == cj).sum()

        rows.append({
            "cell_i": ci, "cell_j": cj, "n_pairs": n,
            "win_i": wins_i, "win_j": wins_j, "ties": ties,
            "win_rate_i": wins_i / n if n > 0 else 0,
            "win_rate_j": wins_j / n if n > 0 else 0,
            "tie_rate": ties / n if n > 0 else 0,
            "avg_preference_strength": avg_strength,
            "fit_i": fit_i, "fit_j": fit_j,
            "trust_i": trust_i, "trust_j": trust_j,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def decompose_effects(demand: pd.DataFrame) -> pd.DataFrame:
    wr = pairwise_win_rates(demand)
    wr_lookup = {(r["cell_i"], r["cell_j"]): r for _, r in wr.iterrows()}

    def _get(ci, cj, key="win_rate_j"):
        row = wr_lookup.get((ci, cj))
        if row is None:
            return np.nan
        return row[key]

    retrieval_wr = _get("00", "10")
    expression_wr = _get("00", "01")
    full_bundle_wr = _get("00", "11")

    retrieval_strength = _get("00", "10", "avg_preference_strength")
    expression_strength = _get("00", "01", "avg_preference_strength")
    full_bundle_strength = _get("00", "11", "avg_preference_strength")

    additive = retrieval_wr + expression_wr - 1.0 if not np.isnan(retrieval_wr) and not np.isnan(expression_wr) else np.nan
    interaction = full_bundle_wr - additive if not np.isnan(additive) else np.nan

    return pd.DataFrame([{
        "component": "retrieval",
        "comparison": "10 vs 00",
        "win_rate_treatment": retrieval_wr,
        "avg_strength": retrieval_strength,
    }, {
        "component": "expression",
        "comparison": "01 vs 00",
        "win_rate_treatment": expression_wr,
        "avg_strength": expression_strength,
    }, {
        "component": "full_bundle",
        "comparison": "11 vs 00",
        "win_rate_treatment": full_bundle_wr,
        "avg_strength": full_bundle_strength,
    }, {
        "component": "interaction",
        "comparison": "11 vs 00 − (10 vs 00) − (01 vs 00) + baseline",
        "win_rate_treatment": interaction,
        "avg_strength": np.nan,
    }])


# ---------------------------------------------------------------------------
# Evaluator summary
# ---------------------------------------------------------------------------

def evaluator_summary(supply: pd.DataFrame, evaluator: pd.DataFrame | None) -> pd.DataFrame | None:
    if evaluator is None:
        return None
    merged = supply[["row_id", "cell", "category"]].merge(evaluator, on="row_id")
    return merged.groupby("cell")[
        ["persuasive_intensity", "tradeoff_disclosure", "fit_specificity"]
    ].agg(["mean", "std"]).round(2)


# ---------------------------------------------------------------------------
# Print report
# ---------------------------------------------------------------------------

def print_report(ret_df, wr_df, decomp_df, eval_df):
    print("\n" + "=" * 70)
    print("  HISTORY-SHOCK DECOMPOSITION REPORT")
    print("=" * 70)

    print("\n--- Retrieval Agreement ---")
    if len(ret_df) > 0:
        by_cat = ret_df.groupby("category")["same_product"].mean()
        for cat, rate in by_cat.items():
            print(f"  {cat}: {rate:.0%} same product")
        print(f"  Overall: {ret_df['same_product'].mean():.0%}")
    else:
        print("  No data")

    print("\n--- Pairwise Win Rates ---")
    for _, r in wr_df.iterrows():
        print(f"  {r['cell_i']} vs {r['cell_j']}: "
              f"{r['cell_i']} wins {r['win_rate_i']:.0%}, "
              f"{r['cell_j']} wins {r['win_rate_j']:.0%}, "
              f"ties {r['tie_rate']:.0%} "
              f"(strength {r['avg_preference_strength']:.1f})")

    print("\n--- Decomposition ---")
    for _, r in decomp_df.iterrows():
        wr = f"{r['win_rate_treatment']:.1%}" if not np.isnan(r["win_rate_treatment"]) else "N/A"
        print(f"  {r['component']:15s}: treatment win rate = {wr}")

    if eval_df is not None:
        print("\n--- Evaluator Scores by Cell ---")
        print(eval_df.to_string())

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    supply, evaluator, demand = load_data()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Pre-analysis invariant check ---")
    assert_cell_invariants(supply)

    ret_df = retrieval_agreement(supply)
    ret_df.to_csv(TABLE_DIR / "retrieval_agreement.csv", index=False)

    wr_df = pairwise_win_rates(demand)
    wr_df.to_csv(TABLE_DIR / "pairwise_win_rates.csv", index=False)

    decomp_df = decompose_effects(demand)
    decomp_df.to_csv(TABLE_DIR / "decomposition_summary.csv", index=False)

    eval_df = evaluator_summary(supply, evaluator)
    if eval_df is not None:
        eval_df.to_csv(TABLE_DIR / "evaluator_means.csv")

    print_report(ret_df, wr_df, decomp_df, eval_df)
    print(f"\nTables saved → {TABLE_DIR}")


if __name__ == "__main__":
    main()
