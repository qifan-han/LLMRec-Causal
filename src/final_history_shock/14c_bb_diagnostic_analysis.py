"""14c — Analyze pairwise architecture diagnostic results.

Reads pairwise comparisons (unified BB vs modular diagonal) and reports
win rates with cluster-level bootstrap CIs.

Reads:
  data/final_history_shock/unified_bb/bb_pairwise_diagnostic.csv

Output:
  data/final_history_shock/unified_bb/bb_diagnostic_report.md
  data/final_history_shock/unified_bb/bb_diagnostic_summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "final_history_shock"

DIAG_PATH = DATA_DIR / "unified_bb" / "bb_pairwise_diagnostic.csv"
REPORT_PATH = DATA_DIR / "unified_bb" / "bb_diagnostic_report.md"
SUMMARY_PATH = DATA_DIR / "unified_bb" / "bb_diagnostic_summary.csv"

OUTCOMES = ["overall_winner", "purchase_winner",
            "satisfaction_winner", "trust_winner"]

B_BOOT = 2000
RNG_SEED = 42


def bootstrap_win_rate(series: pd.Series, label: str,
                       B: int = B_BOOT) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    n = len(series)
    unified_wins = (series == "unified").mean()
    modular_wins = (series == "modular").mean()
    tie_rate = (series == "tie").mean()

    boot_uni = []
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        s = series.iloc[idx]
        boot_uni.append((s == "unified").mean())

    boot_uni = np.array(boot_uni)
    return {
        "outcome": label,
        "unified_win_pct": unified_wins,
        "modular_win_pct": modular_wins,
        "tie_pct": tie_rate,
        "unified_ci_lo": np.percentile(boot_uni, 2.5),
        "unified_ci_hi": np.percentile(boot_uni, 97.5),
    }


def main():
    if not DIAG_PATH.exists():
        sys.exit(f"Diagnostic data not found: {DIAG_PATH}\nRun 14b first.")

    df = pd.read_csv(DIAG_PATH)
    print(f"Loaded {len(df)} diagnostic comparisons")
    print(f"  Parse failures: {df['parse_failed'].sum()}")

    results = []
    report = [
        "# Architecture Diagnostic: Unified BB vs Modular Diagonal",
        "",
        "Blinded pairwise GPT evaluation comparing:",
        "- Unified Z=0 vs Modular cell (0,0) — baseline condition",
        "- Unified Z=1 vs Modular cell (1,1) — treated condition",
        "",
        f"Clusters: {df['cluster_id'].nunique()}",
        f"Bootstrap: B={B_BOOT}, cluster-level, seed={RNG_SEED}",
        "",
    ]

    df["mod_cell"] = df["mod_cell"].astype(str).str.zfill(2)

    for z_val, mod_cell, condition in [(0, "00", "baseline"), (1, "11", "treated")]:
        sub = df[(df["z"] == z_val) & (df["mod_cell"] == mod_cell)]
        n = len(sub)
        same_product = sub["same_product"].sum()

        report.append(f"## {condition.title()} condition: Z={z_val} vs cell ({mod_cell[0]},{mod_cell[1]})")
        report.append("")
        report.append(f"Product agreement: {same_product}/{n} ({same_product/n:.1%})")
        report.append("")
        report.append("| Outcome | Unified wins | Modular wins | Tie | Unified 95% CI |")
        report.append("|---|---|---|---|---|")

        for outcome in OUTCOMES:
            if outcome not in sub.columns:
                continue
            stats = bootstrap_win_rate(sub[outcome], outcome)
            stats["condition"] = condition
            stats["z"] = z_val
            stats["mod_cell"] = mod_cell
            stats["n"] = n
            stats["same_product_pct"] = same_product / n
            results.append(stats)

            report.append(
                f"| {outcome.replace('_winner', '')} "
                f"| {stats['unified_win_pct']:.1%} "
                f"| {stats['modular_win_pct']:.1%} "
                f"| {stats['tie_pct']:.1%} "
                f"| [{stats['unified_ci_lo']:.1%}, {stats['unified_ci_hi']:.1%}] |"
            )

        report.append("")

    report.extend([
        "## Interpretation",
        "",
        "If win rates are roughly balanced (each architecture winning ~30-50% ",
        "with substantial ties), the modular diagonal approximates the unified ",
        "black-box recommender. The main decomposition results can then be ",
        "interpreted as informative about the original system.",
        "",
        "If one architecture dominates, the modular design remains a valid ",
        "policy experiment, but its decomposition should be read as effects ",
        "within an engineered two-stage recommender rather than as a full ",
        "explanation of the unified LLM.",
    ])

    results_df = pd.DataFrame(results)
    results_df.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSaved summary -> {SUMMARY_PATH}")

    report_text = "\n".join(report) + "\n"
    REPORT_PATH.write_text(report_text)
    print(f"Saved report -> {REPORT_PATH}")

    print("\n=== Summary ===")
    for _, r in results_df.iterrows():
        print(f"  {r['condition']:8s} | {r['outcome']:25s} | "
              f"unified {r['unified_win_pct']:.0%} | modular {r['modular_win_pct']:.0%} | "
              f"tie {r['tie_pct']:.0%}")


if __name__ == "__main__":
    main()
