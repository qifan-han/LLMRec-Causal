"""Create a 30-row blinded manual-coding sample from the modular LLM outputs.

Sampling: stratified by category (10 per), balanced across q/r cells and Q_std terciles.
Strips Q_std, Y, q, r from the human-facing file. Keeps a stable row_id for merging.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.data_io import ROOT

INPUT_CSV = ROOT / "data" / "llm_sim" / "modular_llm.csv"
DIAG_DIR = ROOT / "results" / "diagnostics"
OUTPUT_CSV = DIAG_DIR / "manual_coding_sample.csv"
OUTPUT_MD = DIAG_DIR / "manual_coding_sample.md"
KEY_FILE = DIAG_DIR / "manual_coding_key.csv"

SEED = 42
N_PER_CATEGORY = 10


def _load_lookups():
    catalogs = {}
    consumers = {}
    for cat_file in (ROOT / "data" / "catalogs").glob("*.json"):
        cat_name = cat_file.stem
        with open(cat_file) as f:
            catalogs[cat_name] = json.load(f)
    for con_file in (ROOT / "data" / "consumers").glob("*.json"):
        cat_name = con_file.stem
        with open(con_file) as f:
            con_list = json.load(f)
            consumers[cat_name] = {c["consumer_id"]: c for c in con_list}
    product_lookup = {}
    for cat_name, cat_data in catalogs.items():
        for p in cat_data["products"]:
            product_lookup[(cat_name, p["product_id"])] = p
    return catalogs, consumers, product_lookup


def main():
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(INPUT_CSV)
    df["row_key"] = (df["category"] + "_" + df["consumer_id"].astype(str)
                     + "_" + df["q"].astype(str) + "_" + df["r"].astype(str))

    catalogs, consumers, product_lookup = _load_lookups()

    sampled = []
    for cat in sorted(df["category"].unique()):
        sub = df[df["category"] == cat].copy()
        sub["Q_tercile"] = pd.qcut(sub["Q_std"], 3, labels=["low", "mid", "high"])

        # Try to get balanced across q/r cells and Q terciles
        # 4 cells x 3 terciles = 12 strata; pick ~1 per stratum, fill remaining randomly
        picked_indices = []
        for (q_val, r_val), cell_df in sub.groupby(["q", "r"]):
            for terc in ["low", "mid", "high"]:
                terc_rows = cell_df[cell_df["Q_tercile"] == terc]
                if len(terc_rows) > 0:
                    idx = rng.choice(terc_rows.index, size=1)[0]
                    if idx not in picked_indices:
                        picked_indices.append(idx)

        # If we have more than N_PER_CATEGORY, trim; if fewer, add random
        if len(picked_indices) > N_PER_CATEGORY:
            picked_indices = list(rng.choice(picked_indices, N_PER_CATEGORY, replace=False))
        elif len(picked_indices) < N_PER_CATEGORY:
            remaining = [i for i in sub.index if i not in picked_indices]
            n_need = N_PER_CATEGORY - len(picked_indices)
            extra = rng.choice(remaining, min(n_need, len(remaining)), replace=False)
            picked_indices.extend(extra)

        sampled.append(sub.loc[picked_indices])

    sample_df = pd.concat(sampled, ignore_index=True)
    sample_df = sample_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample_df["row_id"] = range(1, len(sample_df) + 1)

    # Save the key file (with Q_std, q, r, Y) for later merging
    key_df = sample_df[["row_id", "row_key", "category", "consumer_id",
                         "q", "r", "product_id", "Q_std", "expression_intensity", "Y"]].copy()
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    key_df.to_csv(KEY_FILE, index=False)
    print(f"Key file (with hidden labels): {KEY_FILE} ({len(key_df)} rows)")

    # Build blinded output
    blind_rows = []
    for _, row in sample_df.iterrows():
        cat = row["category"]
        cid = int(row["consumer_id"])
        pid = row["product_id"]

        consumer = consumers.get(cat, {}).get(cid, {})
        product = product_lookup.get((cat, pid), {})

        brand_fam = consumer.get("brand_familiarity", {})
        brand_name = product.get("brand_name", "unknown")
        attrs = product.get("attributes", {})
        attr_str = "; ".join(f"{k}: {v}" for k, v in attrs.items())

        consumer_text = (
            f"use_case={consumer.get('use_case', '?')}, "
            f"budget=${consumer.get('budget', 0):.2f}, "
            f"price_sens={consumer.get('price_sensitivity', 0):.2f}, "
            f"quality_sens={consumer.get('quality_sensitivity', 0):.2f}, "
            f"brand_fam_{brand_name}={brand_fam.get(brand_name, '?'):.2f}"
        )

        product_text = (
            f"{pid} by {brand_name}, "
            f"${product.get('price', 0):.2f}, "
            f"quality={product.get('quality_score', 0)}/100, "
            f"{attr_str}"
        )

        blind_rows.append({
            "row_id": row["row_id"],
            "category": cat,
            "consumer_profile": consumer_text,
            "product_info": product_text,
            "product_reviews": product.get("review_summary", ""),
            "recommendation_text": row["recommendation_text"],
            "manual_fit_specificity": "",
            "manual_persuasive_intensity": "",
            "manual_tradeoff_disclosure": "",
            "manual_notes": "",
        })

    blind_df = pd.DataFrame(blind_rows)
    blind_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Blinded sample: {OUTPUT_CSV} ({len(blind_df)} rows)")

    # Markdown version for easier reading
    md_lines = [
        "# Manual Coding Sample\n",
        "Score each recommendation on three 1-7 scales:\n",
        "- **fit_specificity**: 1=generic, 7=specifically links product to consumer needs",
        "- **persuasive_intensity**: 1=neutral/factual, 7=highly confident/endorsing",
        "- **tradeoff_disclosure**: 1=hides caveats, 7=clearly states limitations\n",
        "---\n",
    ]

    for _, row in blind_df.iterrows():
        md_lines.append(f"## Row {row['row_id']} ({row['category']})\n")
        md_lines.append(f"**Consumer:** {row['consumer_profile']}\n")
        md_lines.append(f"**Product:** {row['product_info']}\n")
        md_lines.append(f"**Reviews:** {row['product_reviews']}\n")
        md_lines.append(f"**Recommendation:**\n> {row['recommendation_text']}\n")
        md_lines.append("| Scale | Score (1-7) | Notes |")
        md_lines.append("|-------|-------------|-------|")
        md_lines.append("| fit_specificity | ___ | |")
        md_lines.append("| persuasive_intensity | ___ | |")
        md_lines.append("| tradeoff_disclosure | ___ | |\n")
        md_lines.append("---\n")

    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown sample: {OUTPUT_MD}")

    # Print sampling summary
    print(f"\nSampling summary:")
    print(f"  Total rows: {len(sample_df)}")
    for cat in sorted(sample_df["category"].unique()):
        sub = sample_df[sample_df["category"] == cat]
        print(f"  {cat}: {len(sub)} rows, "
              f"q=[{sub['q'].value_counts().to_dict()}], "
              f"r=[{sub['r'].value_counts().to_dict()}], "
              f"Q_std range=[{sub['Q_std'].min():.2f}, {sub['Q_std'].max():.2f}]")


if __name__ == "__main__":
    main()
