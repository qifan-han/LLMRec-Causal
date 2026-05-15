"""13 — Write final simulation summary report.

Reads all analysis outputs and produces a markdown report.

Output:
  data/final_history_shock/reports/final_simulation_report.md

Usage:
  python 13_write_summary_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_openai import DATA_DIR

ANALYSIS_DIR = DATA_DIR / "analysis"
REPORTS_DIR = DATA_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
SUPPLY_DIR = DATA_DIR / "local_supply"


def load_csv(name: str) -> pd.DataFrame | None:
    path = ANALYSIS_DIR / name
    if path.exists():
        return pd.read_csv(path)
    return None


def build_report() -> str:
    t1 = load_csv("table1_design.csv")
    t2 = load_csv("table2_retrieval_variation.csv")
    t3 = load_csv("table3_pairwise_win_rates.csv")
    t4 = load_csv("table4_bt_decomposition.csv")
    t5 = load_csv("table5_outcome_channels.csv")
    t6 = load_csv("table6_text_mechanisms.csv")

    supply_path = SUPPLY_DIR / "final_supply_rows_clean.csv"
    if not supply_path.exists():
        supply_path = SUPPLY_DIR / "final_supply_rows.csv"
    supply = pd.read_csv(supply_path) if supply_path.exists() else None

    lines = []
    lines.append("# Final History-Shock Simulation Report")
    lines.append("")

    # 1. Executive summary
    lines.append("## 1. Executive Summary")
    lines.append("")
    if t4 is not None:
        total_row = t4[t4["component"] == "total"]
        expr_row = t4[t4["component"] == "expression"]
        retr_row = t4[t4["component"] == "retrieval"]
        inter_row = t4[t4["component"] == "interaction"]

        if len(total_row) > 0:
            tr = total_row.iloc[0]
            lines.append(f"The full history-shock treatment has a total effect of "
                         f"{tr['mean']:.3f} (95% CI [{tr['ci_lo']:.3f}, {tr['ci_hi']:.3f}], "
                         f"P>0 = {tr['p_positive']:.1%}).")
        if len(expr_row) > 0 and len(retr_row) > 0:
            er = expr_row.iloc[0]
            rr = retr_row.iloc[0]
            lines.append(f"Expression effect: {er['mean']:.3f} [{er['ci_lo']:.3f}, {er['ci_hi']:.3f}], "
                         f"P>0 = {er['p_positive']:.1%}.")
            lines.append(f"Retrieval effect: {rr['mean']:.3f} [{rr['ci_lo']:.3f}, {rr['ci_hi']:.3f}], "
                         f"P>0 = {rr['p_positive']:.1%}.")
        if len(inter_row) > 0:
            ir = inter_row.iloc[0]
            lines.append(f"Interaction: {ir['mean']:.3f} [{ir['ci_lo']:.3f}, {ir['ci_hi']:.3f}], "
                         f"P>0 = {ir['p_positive']:.1%}.")
    lines.append("")

    # 2. What changed
    lines.append("## 2. What Changed Relative to Previous Pilot")
    lines.append("")
    lines.append("- Qualitative-only history: no numerical rates shown to the LLM recommender")
    lines.append("- Anti-leakage audit with regex detection and regeneration")
    lines.append("- GPT-5.3 as demand-side judge (replacing local gemma2:9b)")
    lines.append("- 120 clusters (vs 30 in pilot), 25 products per category (vs 10)")
    lines.append("- GPT-generated diverse consumer personas (120 total)")
    lines.append("- Both absolute and pairwise GPT evaluation")
    lines.append("- Multiple outcome dimensions (overall, purchase, satisfaction, trust)")
    lines.append("")

    # 3. Design
    lines.append("## 3. Design and Sample Size")
    lines.append("")
    if t1 is not None:
        r = t1.iloc[0]
        lines.append(f"| Dimension | Value |")
        lines.append(f"|-----------|-------|")
        for col in t1.columns:
            lines.append(f"| {col.replace('_', ' ')} | {r[col]} |")
    lines.append("")

    # 4. Supply-side results
    lines.append("## 4. Supply-Side Results")
    lines.append("")
    if t2 is not None:
        lines.append("### Retrieval Variation")
        lines.append("")
        lines.append("| Category | Change Rate | Generic Top Share | History Top Share |")
        lines.append("|----------|-------------|-------------------|-------------------|")
        for _, r in t2.iterrows():
            lines.append(f"| {r['category']} | {r['retrieval_change_rate']:.1%} | "
                         f"{r['generic_top_product_share']:.1%} | {r['history_top_product_share']:.1%} |")
    lines.append("")

    if supply is not None and "leakage_flag" in supply.columns:
        n_leak = supply["leakage_flag"].sum()
        lines.append(f"Leakage rate: {n_leak}/{len(supply)} ({n_leak / len(supply):.1%})")
    lines.append("")

    # 5. Pairwise results
    lines.append("## 5. Demand-Side Results")
    lines.append("")
    if t3 is not None:
        lines.append("### All Pairwise Comparisons")
        lines.append("")
        lines.append("| A vs B | n | A wins | B wins | ties | A win% | B win% | tie% |")
        lines.append("|--------|---|--------|--------|------|--------|--------|------|")
        for _, r in t3.iterrows():
            lines.append(f"| {r['cell_A']} vs {r['cell_B']} | {r['n']:.0f} | "
                         f"{r['A_wins']:.0f} | {r['B_wins']:.0f} | {r['ties']:.0f} | "
                         f"{r['A_win_pct']:.1%} | {r['B_win_pct']:.1%} | {r['tie_pct']:.1%} |")
    lines.append("")

    # 6. Decomposition
    lines.append("## 6. Bradley-Terry Decomposition")
    lines.append("")
    if t4 is not None:
        lines.append("| Component | Estimate | SE | 95% CI | P(>0) |")
        lines.append("|-----------|----------|----|--------|-------|")
        for _, r in t4.iterrows():
            lines.append(f"| {r['component']} | {r['mean']:.3f} | {r['se']:.3f} | "
                         f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {r['p_positive']:.1%} |")
    lines.append("")

    # 7. Multi-outcome
    if t5 is not None and len(t5) > 0:
        lines.append("## 7. Decomposition by Outcome Dimension")
        lines.append("")
        for outcome in t5["outcome"].unique():
            sub = t5[t5["outcome"] == outcome]
            lines.append(f"### {outcome.capitalize()}")
            lines.append("")
            lines.append("| Component | Estimate | 95% CI | P(>0) |")
            lines.append("|-----------|----------|--------|-------|")
            for _, r in sub.iterrows():
                lines.append(f"| {r['component']} | {r['mean']:.3f} | "
                             f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {r['p_positive']:.1%} |")
            lines.append("")

    # 8. Text mechanisms
    if t6 is not None:
        lines.append("## 8. Text Mechanism Audit")
        lines.append("")
        lines.append("| Cell | PI | TD | Regret Risk | Trust |")
        lines.append("|------|----|----|----|-------|")
        for _, r in t6.iterrows():
            lines.append(f"| {r['cell']} | "
                         f"{r.get('persuasive_intensity_1_7', 'N/A')} | "
                         f"{r.get('tradeoff_disclosure_1_7', 'N/A')} | "
                         f"{r.get('regret_risk_1_7', 'N/A')} | "
                         f"{r.get('trust_score_1_7', 'N/A')} |")
        lines.append("")

    # 9. Caveats
    lines.append("## 9. What Can and Cannot Be Claimed")
    lines.append("")
    lines.append("The outcome is a blinded GPT-based synthetic pairwise preference judgment, "
                 "not observed market demand.")
    lines.append("")
    lines.append("The simulation is designed to test whether a modular audit can reveal the "
                 "channels through which historical purchase information changes LLM "
                 "recommendation packages.")
    lines.append("")
    lines.append("The decomposition separates product-selection changes from expression changes "
                 "while holding the consumer and category fixed.")
    lines.append("")
    lines.append("This simulation does **not** claim:")
    lines.append("- Real purchase effects")
    lines.append("- Real welfare effects")
    lines.append("- Actual market demand")
    lines.append("- Human consumer validation")
    lines.append("- That local LLM behavior represents all frontier models")
    lines.append("")

    return "\n".join(lines)


def main():
    report = build_report()
    out_path = REPORTS_DIR / "final_simulation_report.md"
    with open(out_path, "w") as f:
        f.write(report)
    print(f"Report saved → {out_path}")
    print(f"Report length: {len(report)} chars, {len(report.splitlines())} lines")


if __name__ == "__main__":
    main()
