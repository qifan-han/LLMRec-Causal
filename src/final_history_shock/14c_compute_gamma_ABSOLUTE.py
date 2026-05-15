"""14c — Compute architecture gap Gamma and write summary report.

Gamma = tau^BB - tau^MOD
  tau^BB  = mu^BB_1  - mu^BB_0   (unified black-box effect)
  tau^MOD = mu^MOD_11 - mu^MOD_00 (modular diagonal effect)

Also computes cluster-level bootstrap CIs for Gamma.

Reads:
  data/final_history_shock/gpt_eval/absolute_eval_rows.csv   (modular 4-cell)
  data/final_history_shock/unified_bb/unified_bb_eval.csv     (unified BB)

Output:
  data/final_history_shock/unified_bb/gamma_report.md
  data/final_history_shock/unified_bb/gamma_estimates.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "final_history_shock"

MOD_EVAL = DATA_DIR / "gpt_eval" / "absolute_eval_rows.csv"
BB_EVAL = DATA_DIR / "unified_bb" / "unified_bb_eval.csv"
BB_SUPPLY = DATA_DIR / "unified_bb" / "unified_bb_supply.csv"
MOD_SUPPLY = DATA_DIR / "local_supply" / "final_supply_rows.csv"

OUT_DIR = DATA_DIR / "unified_bb"
GAMMA_CSV = OUT_DIR / "gamma_estimates.csv"
REPORT_PATH = OUT_DIR / "gamma_report.md"

OUTCOMES = [
    "fit_score_1_7",
    "purchase_probability_0_100",
    "trust_score_1_7",
    "persuasive_intensity_1_7",
    "tradeoff_disclosure_1_7",
]

B_BOOT = 2000
RNG_SEED = 42


def cluster_bootstrap_gamma(mod_df: pd.DataFrame, bb_df: pd.DataFrame,
                            outcome: str, B: int = B_BOOT) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    mod_clusters = mod_df["cluster_id"].unique()
    bb_clusters = bb_df["cluster_id"].unique()
    shared = np.intersect1d(mod_clusters, bb_clusters)

    mod_sub = mod_df[mod_df["cluster_id"].isin(shared)]
    bb_sub = bb_df[bb_df["cluster_id"].isin(shared)]

    def point_estimates(m, b):
        mu_mod_00 = m[m["cell"] == 0][outcome].mean()
        mu_mod_11 = m[m["cell"] == 11][outcome].mean()
        mu_bb_0 = b[b["z"] == 0][outcome].mean()
        mu_bb_1 = b[b["z"] == 1][outcome].mean()
        tau_mod = mu_mod_11 - mu_mod_00
        tau_bb = mu_bb_1 - mu_bb_0
        gamma = tau_bb - tau_mod
        return {
            "mu_mod_00": mu_mod_00, "mu_mod_11": mu_mod_11,
            "mu_bb_0": mu_bb_0, "mu_bb_1": mu_bb_1,
            "tau_mod": tau_mod, "tau_bb": tau_bb, "gamma": gamma,
        }

    est = point_estimates(mod_sub, bb_sub)

    boot_gammas = []
    boot_tau_bb = []
    boot_tau_mod = []
    for _ in range(B):
        idx = rng.choice(shared, size=len(shared), replace=True)
        m_b = mod_sub[mod_sub["cluster_id"].isin(idx)]
        b_b = bb_sub[bb_sub["cluster_id"].isin(idx)]
        try:
            be = point_estimates(m_b, b_b)
            boot_gammas.append(be["gamma"])
            boot_tau_bb.append(be["tau_bb"])
            boot_tau_mod.append(be["tau_mod"])
        except Exception:
            continue

    boot_gammas = np.array(boot_gammas)
    boot_tau_bb = np.array(boot_tau_bb)
    boot_tau_mod = np.array(boot_tau_mod)

    est["gamma_se"] = boot_gammas.std(ddof=1)
    est["gamma_ci_lo"] = np.percentile(boot_gammas, 2.5)
    est["gamma_ci_hi"] = np.percentile(boot_gammas, 97.5)
    est["tau_bb_se"] = boot_tau_bb.std(ddof=1)
    est["tau_bb_ci_lo"] = np.percentile(boot_tau_bb, 2.5)
    est["tau_bb_ci_hi"] = np.percentile(boot_tau_bb, 97.5)
    est["tau_mod_se"] = boot_tau_mod.std(ddof=1)
    est["tau_mod_ci_lo"] = np.percentile(boot_tau_mod, 2.5)
    est["tau_mod_ci_hi"] = np.percentile(boot_tau_mod, 97.5)

    if abs(est["tau_bb"]) > 1e-6:
        est["approx_ratio"] = est["tau_mod"] / est["tau_bb"]
    else:
        est["approx_ratio"] = np.nan

    return est


def compute_modular_decomposition(mod_df: pd.DataFrame, outcome: str) -> dict:
    mu = {}
    for cell in [0, 1, 10, 11]:
        mu[cell] = mod_df[mod_df["cell"] == cell][outcome].mean()

    delta_j = mu[10] - mu[0]
    delta_t = mu[1] - mu[0]
    delta_jt = mu[11] - mu[10] - mu[1] + mu[0]
    tau_mod = mu[11] - mu[0]

    return {
        "mu_00": mu[0], "mu_01": mu[1], "mu_10": mu[10], "mu_11": mu[11],
        "delta_j": delta_j, "delta_t": delta_t, "delta_jt": delta_jt,
        "tau_mod": tau_mod,
    }


def main():
    if not MOD_EVAL.exists():
        sys.exit(f"Modular eval not found: {MOD_EVAL}")
    if not BB_EVAL.exists():
        sys.exit(f"Unified BB eval not found: {BB_EVAL}")

    mod = pd.read_csv(MOD_EVAL)
    bb = pd.read_csv(BB_EVAL)
    bb_supply = pd.read_csv(BB_SUPPLY) if BB_SUPPLY.exists() else None
    mod_supply = pd.read_csv(MOD_SUPPLY) if MOD_SUPPLY.exists() else None

    print(f"Modular eval: {len(mod)} rows, cells {sorted(mod['cell'].unique())}")
    print(f"Unified BB eval: {len(bb)} rows, Z values {sorted(bb['z'].unique())}")

    # Product agreement: unified Z=0 vs modular cell 0
    if bb_supply is not None and mod_supply is not None:
        bb_z0 = bb_supply[bb_supply["z"] == 0][["cluster_id", "selected_product_id"]].rename(
            columns={"selected_product_id": "bb_pid"})
        mod_00 = mod_supply[mod_supply["cell"] == 0][["cluster_id", "selected_product_id"]].rename(
            columns={"selected_product_id": "mod_pid"})
        merged = bb_z0.merge(mod_00, on="cluster_id")
        agree_z0 = (merged["bb_pid"] == merged["mod_pid"]).mean()

        bb_z1 = bb_supply[bb_supply["z"] == 1][["cluster_id", "selected_product_id"]].rename(
            columns={"selected_product_id": "bb_pid"})
        mod_11 = mod_supply[mod_supply["cell"] == 11][["cluster_id", "selected_product_id"]].rename(
            columns={"selected_product_id": "mod_pid"})
        merged1 = bb_z1.merge(mod_11, on="cluster_id")
        agree_z1 = (merged1["bb_pid"] == merged1["mod_pid"]).mean()

        # BB product differentiation
        bb_merged = bb_supply[bb_supply["z"] == 0][["cluster_id", "selected_product_id"]].rename(
            columns={"selected_product_id": "z0_pid"}).merge(
            bb_supply[bb_supply["z"] == 1][["cluster_id", "selected_product_id"]].rename(
                columns={"selected_product_id": "z1_pid"}),
            on="cluster_id")
        bb_changed = (bb_merged["z0_pid"] != bb_merged["z1_pid"]).mean()

        print(f"\nProduct agreement:")
        print(f"  BB Z=0 vs MOD cell 0: {agree_z0:.1%}")
        print(f"  BB Z=1 vs MOD cell 11: {agree_z1:.1%}")
        print(f"  BB retrieval changed (Z=0 vs Z=1): {bb_changed:.1%}")

    # Compute Gamma for each outcome
    results = []
    for outcome in OUTCOMES:
        if outcome not in mod.columns or outcome not in bb.columns:
            print(f"  Skipping {outcome} — not in both datasets")
            continue
        est = cluster_bootstrap_gamma(mod, bb, outcome)
        decomp = compute_modular_decomposition(mod, outcome)
        row = {"outcome": outcome, **est, **{f"mod_{k}": v for k, v in decomp.items()}}
        results.append(row)

        print(f"\n{outcome}:")
        print(f"  mu^BB_0 = {est['mu_bb_0']:.2f}, mu^BB_1 = {est['mu_bb_1']:.2f}")
        print(f"  tau^BB = {est['tau_bb']:.2f} [{est['tau_bb_ci_lo']:.2f}, {est['tau_bb_ci_hi']:.2f}]")
        print(f"  mu^MOD_00 = {est['mu_mod_00']:.2f}, mu^MOD_11 = {est['mu_mod_11']:.2f}")
        print(f"  tau^MOD = {est['tau_mod']:.2f} [{est['tau_mod_ci_lo']:.2f}, {est['tau_mod_ci_hi']:.2f}]")
        print(f"  GAMMA = {est['gamma']:.2f} [{est['gamma_ci_lo']:.2f}, {est['gamma_ci_hi']:.2f}]")
        if not np.isnan(est["approx_ratio"]):
            print(f"  Approx ratio (tau^MOD / tau^BB) = {est['approx_ratio']:.2f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(GAMMA_CSV, index=False)
    print(f"\nSaved estimates -> {GAMMA_CSV}")

    # Write report
    report_lines = [
        "# Architecture Gap (Gamma) Report",
        "",
        f"Date: 2026-05-15",
        f"Supply model: Qwen 2.5 14B (local, temperature 0.7)",
        f"Demand model: GPT-5.3-chat-latest (OpenAI API)",
        f"Bootstrap: B={B_BOOT}, cluster-level, seed={RNG_SEED}",
        "",
        "## Design",
        "",
        "- **Unified black-box (BB)**: Single LLM call selects product AND writes text.",
        "  - Z=0: feature-only (no reviews, sales, or history data)",
        "  - Z=1: history-aware (reviews, sales, popularity, buyer feedback)",
        "- **Modular two-stage (MOD)**: Separate retrieval and expression calls.",
        "  - Cell (0,0): generic retrieval + generic expression",
        "  - Cell (1,1): history retrieval + history expression",
        "",
        "## Product Agreement",
        "",
    ]

    if bb_supply is not None and mod_supply is not None:
        report_lines.extend([
            f"| Comparison | Agreement |",
            f"|---|---|",
            f"| BB Z=0 vs MOD (0,0) | {agree_z0:.1%} |",
            f"| BB Z=1 vs MOD (1,1) | {agree_z1:.1%} |",
            f"| BB retrieval diff (Z=0 vs Z=1) | {bb_changed:.1%} |",
            "",
        ])

    report_lines.extend([
        "## Architecture Gap Estimates",
        "",
        "| Outcome | tau^BB | 95% CI | tau^MOD | 95% CI | Gamma | 95% CI | Ratio |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for _, r in results_df.iterrows():
        ratio_str = f"{r['approx_ratio']:.2f}" if not np.isnan(r["approx_ratio"]) else "—"
        report_lines.append(
            f"| {r['outcome']} | {r['tau_bb']:.2f} | [{r['tau_bb_ci_lo']:.2f}, {r['tau_bb_ci_hi']:.2f}] "
            f"| {r['tau_mod']:.2f} | [{r['tau_mod_ci_lo']:.2f}, {r['tau_mod_ci_hi']:.2f}] "
            f"| {r['gamma']:.2f} | [{r['gamma_ci_lo']:.2f}, {r['gamma_ci_hi']:.2f}] "
            f"| {ratio_str} |"
        )

    report_lines.extend([
        "",
        "## Modular Decomposition (Exercise 2)",
        "",
        "| Outcome | mu_00 | mu_01 | mu_10 | mu_11 | Delta_J | Delta_T | Delta_JT | tau_MOD |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for _, r in results_df.iterrows():
        report_lines.append(
            f"| {r['outcome']} | {r['mod_mu_00']:.2f} | {r['mod_mu_01']:.2f} "
            f"| {r['mod_mu_10']:.2f} | {r['mod_mu_11']:.2f} "
            f"| {r['mod_delta_j']:.2f} | {r['mod_delta_t']:.2f} "
            f"| {r['mod_delta_jt']:.2f} | {r['mod_tau_mod']:.2f} |"
        )

    report_lines.extend([
        "",
        "## Interpretation",
        "",
    ])

    # Auto-interpret
    gamma_purchase = results_df[results_df["outcome"] == "purchase_probability_0_100"]
    if len(gamma_purchase) > 0:
        g = gamma_purchase.iloc[0]
        if g["gamma_ci_lo"] <= 0 <= g["gamma_ci_hi"]:
            report_lines.append(
                "For purchase probability, the 95% CI for Gamma includes zero, "
                "supporting the claim that the modular diagonal approximates "
                "the unified black-box history shock."
            )
        else:
            report_lines.append(
                f"For purchase probability, Gamma = {g['gamma']:.2f} with 95% CI "
                f"[{g['gamma_ci_lo']:.2f}, {g['gamma_ci_hi']:.2f}], suggesting "
                f"a {'positive' if g['gamma'] > 0 else 'negative'} architecture gap."
            )

    report_lines.extend([
        "",
        "## Bottom Line",
        "",
        "If Gamma is approximately zero across key outcomes, the paper can claim:",
        "",
        "> For the local Qwen model and simulation environment studied here, the two-stage ",
        "> modular diagonal closely approximates the unified black-box history shock. We ",
        "> therefore use the modular design as a local decomposition of the black-box effect.",
        "",
        "If Gamma is large, the paper should instead say:",
        "",
        "> The modular design identifies policy-relevant component effects for a two-stage ",
        "> implementation, but these effects should not be interpreted as decomposing the ",
        "> unified black-box LLM shock.",
    ])

    report_text = "\n".join(report_lines) + "\n"
    REPORT_PATH.write_text(report_text)
    print(f"Report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
