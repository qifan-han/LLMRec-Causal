"""01 — Generate synthetic historical purchase data (transparent DGP).

Produces product-level and segment-level purchase statistics from a
logistic model of consumer choice.  These tables serve as the information
shock in the modular audit: cells 10 and 11 see this data; cells 00 and 01
do not.

Usage:
  python 01_generate_purchase_history.py                  # all 3 categories
  python 01_generate_purchase_history.py --categories headphones
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from utils import (
    CATEGORIES, HIST_DATA, MASTER_SEED,
    load_catalog, load_consumers, load_fit_scores, assign_segment,
)

# ---------------------------------------------------------------------------
# DGP parameters (documented for transparency)
# ---------------------------------------------------------------------------

DGP = {
    "intercept": -2.0,
    "beta_Q": 1.0,
    "beta_afford": 1.5,
    "beta_quality": 0.3,
    "beta_brand": 0.5,
    "sigma_eps": 0.5,
    "sat_intercept": 2.5,
    "sat_Q": 1.0,
    "sat_quality": 0.3,
    "sat_brand": 0.2,
    "sigma_eta": 0.4,
    "return_intercept": -1.5,
    "return_sat": -2.0,
}


# ---------------------------------------------------------------------------
# Core DGP
# ---------------------------------------------------------------------------

def generate_category_history(category: str, rng: np.random.Generator) -> pd.DataFrame:
    consumers = load_consumers(category)
    catalog = load_catalog(category)
    fit_df = load_fit_scores(category)
    products = catalog["products"]

    pid_to_product = {p["product_id"]: p for p in products}
    fit_lookup = fit_df.set_index(["consumer_id", "product_id"])["Q_std"].to_dict()

    rows = []
    for c in consumers:
        cid = c["consumer_id"]
        segment = assign_segment(c, category, consumers)
        for p in products:
            pid = p["product_id"]
            rows.append({
                "consumer_id": cid,
                "product_id": pid,
                "segment_id": segment,
                "budget": c["budget"],
                "price": p["price"],
                "quality_score": p["quality_score"],
                "brand_familiarity": c.get("brand_familiarity", {}).get(p["brand_name"], 0.0),
                "Q_std": fit_lookup.get((cid, pid), 0.0),
            })

    df = pd.DataFrame(rows)

    df["afford"] = np.minimum(1.0, df["budget"] / df["price"])
    df["quality_norm"] = (df["quality_score"] - 50) / 50

    eps = rng.normal(0, DGP["sigma_eps"], len(df))
    logit_p = (DGP["intercept"]
               + DGP["beta_Q"] * df["Q_std"]
               + DGP["beta_afford"] * df["afford"]
               + DGP["beta_quality"] * df["quality_norm"]
               + DGP["beta_brand"] * df["brand_familiarity"]
               + eps)
    df["purchase_prob"] = 1 / (1 + np.exp(-logit_p))
    df["purchased"] = rng.binomial(1, df["purchase_prob"].values)

    eta = rng.normal(0, DGP["sigma_eta"], len(df))
    df["satisfaction"] = np.clip(
        DGP["sat_intercept"]
        + DGP["sat_Q"] * df["Q_std"]
        + DGP["sat_quality"] * df["quality_norm"]
        + DGP["sat_brand"] * df["brand_familiarity"]
        + eta,
        1.0, 5.0,
    )

    ret_logit = DGP["return_intercept"] + DGP["return_sat"] * (df["satisfaction"] - 3.0)
    df["return_prob"] = 1 / (1 + np.exp(-ret_logit))
    df["returned"] = rng.binomial(1, df["return_prob"].values) * df["purchased"]

    df.loc[df["purchased"] == 0, ["satisfaction", "returned"]] = np.nan
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_product_history(logs: pd.DataFrame) -> pd.DataFrame:
    purchased = logs[logs["purchased"] == 1]
    impr = logs.groupby("product_id").size().rename("impressions")
    purch = purchased.groupby("product_id").size().rename("purchases")
    sat = purchased.groupby("product_id")["satisfaction"].mean().rename("avg_satisfaction")
    ret = purchased.groupby("product_id")["returned"].sum().rename("returns")

    out = pd.DataFrame({"impressions": impr}).join([purch, sat, ret]).fillna(0)
    out["purchases"] = out["purchases"].astype(int)
    out["returns"] = out["returns"].astype(int)
    out["conversion_rate"] = out["purchases"] / out["impressions"]
    out["return_rate"] = out["returns"] / out["purchases"].clip(lower=1)
    return out.reset_index()


def aggregate_segment_history(logs: pd.DataFrame) -> pd.DataFrame:
    purchased = logs[logs["purchased"] == 1]
    impr = logs.groupby(["segment_id", "product_id"]).size().rename("impressions")
    purch = purchased.groupby(["segment_id", "product_id"]).size().rename("purchases")
    sat = purchased.groupby(["segment_id", "product_id"])["satisfaction"].mean().rename("avg_satisfaction")

    out = pd.DataFrame({"impressions": impr}).join([purch, sat]).fillna(0)
    out["purchases"] = out["purchases"].astype(int)
    out["conversion_rate"] = out["purchases"] / out["impressions"]
    return out.reset_index()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary(all_prod_hists: dict[str, pd.DataFrame], path):
    lines = ["# History-Shock DGP Summary\n"]
    lines.append("## DGP Parameters\n")
    for k, v in DGP.items():
        lines.append(f"- {k}: {v}")
    lines.append(f"\n- master_seed: {MASTER_SEED}\n")

    for cat, ph in all_prod_hists.items():
        lines.append(f"## {cat}\n")
        lines.append(f"Total impressions: {ph['impressions'].sum()}")
        lines.append(f"Total purchases: {ph['purchases'].sum()}")
        overall_cr = ph["purchases"].sum() / ph["impressions"].sum()
        lines.append(f"Overall conversion rate: {overall_cr:.1%}")
        lines.append(f"Mean satisfaction: {ph['avg_satisfaction'].mean():.2f}")
        lines.append(f"Mean return rate: {ph['return_rate'].mean():.1%}")
        cr_range = f"{ph['conversion_rate'].min():.1%} – {ph['conversion_rate'].max():.1%}"
        lines.append(f"Conversion rate range: {cr_range}\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    args = parser.parse_args()

    rng = np.random.default_rng(MASTER_SEED)
    HIST_DATA.mkdir(parents=True, exist_ok=True)
    all_prod = {}

    for cat in args.categories:
        print(f"\n=== {cat} ===")
        logs = generate_category_history(cat, rng)
        logs.to_csv(HIST_DATA / f"{cat}_purchase_logs.csv", index=False)
        print(f"  Purchase logs: {len(logs)} rows, {int(logs['purchased'].sum())} purchases")

        prod = aggregate_product_history(logs)
        prod.to_csv(HIST_DATA / f"{cat}_product_history.csv", index=False)
        print(f"  Product history: {len(prod)} products, "
              f"CR range {prod['conversion_rate'].min():.1%}–{prod['conversion_rate'].max():.1%}")

        seg = aggregate_segment_history(logs)
        seg.to_csv(HIST_DATA / f"{cat}_segment_history.csv", index=False)
        print(f"  Segment history: {len(seg)} rows ({seg['segment_id'].nunique()} segments)")

        all_prod[cat] = prod

    write_summary(all_prod, HIST_DATA / "history_summary.md")
    print(f"\nDone. Files saved to {HIST_DATA}")


if __name__ == "__main__":
    main()
