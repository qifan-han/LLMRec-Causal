"""Loaders and validation for catalogs, consumers, and fit scores.

This module is consumed by the MVP simulation scripts. It is deliberately
minimal: only the JSON/CSV outputs of 00_generate_catalogs.py and
01_generate_consumers.py are supported.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CATALOG_DIR = ROOT / "data" / "catalogs"
CONSUMER_DIR = ROOT / "data" / "consumers"
FIT_DIR = ROOT / "data" / "fit_scores"

CATEGORIES = ["phone_charger", "headphones", "laptop"]


def list_categories() -> List[str]:
    """Return the three categories the project ships with."""
    return list(CATEGORIES)


def load_catalog(category: str) -> dict:
    with open(CATALOG_DIR / f"{category}.json") as f:
        return json.load(f)


def load_consumers(category: str) -> List[dict]:
    with open(CONSUMER_DIR / f"{category}.json") as f:
        return json.load(f)


def load_fit_scores(category: str) -> pd.DataFrame:
    """Long-format fit-score frame: consumer_id, product_id, Q.

    Source file is wide (one row per consumer, one column per product) — we
    melt it on load because every downstream operation wants the long form.
    """
    wide = pd.read_csv(FIT_DIR / f"{category}.csv")
    long = wide.melt(
        id_vars="consumer_id",
        var_name="product_id",
        value_name="Q",
    )
    return long


@dataclass
class CategoryData:
    category: str
    catalog: dict
    consumers: List[dict]
    fit_long: pd.DataFrame  # consumer_id, product_id, Q, Q_std
    product_df: pd.DataFrame  # product_id, brand_name, brand_status, incumbent, focal_brand, price


def _build_product_df(catalog: dict) -> pd.DataFrame:
    focal_brand = catalog.get("focal_brand")
    rows = []
    for p in catalog["products"]:
        rows.append(
            {
                "product_id": p["product_id"],
                "brand_name": p["brand_name"],
                "brand_status": p["brand_status"],
                "incumbent": 1 if p["brand_status"] == "incumbent" else 0,
                "focal_brand": 1 if p["brand_name"] == focal_brand else 0,
                "price": p["price"],
                "quality_score": p["quality_score"],
            }
        )
    return pd.DataFrame(rows)


def load_category_data(category: str) -> CategoryData:
    catalog = load_catalog(category)
    consumers = load_consumers(category)
    fit_long = load_fit_scores(category)

    # Within-category standardization of Q
    mu = fit_long["Q"].mean()
    sd = fit_long["Q"].std(ddof=0)
    fit_long = fit_long.copy()
    fit_long["Q_std"] = (fit_long["Q"] - mu) / sd

    product_df = _build_product_df(catalog)

    # Attach product attributes to the long frame for downstream convenience
    fit_long = fit_long.merge(product_df, on="product_id", how="left")

    return CategoryData(
        category=category,
        catalog=catalog,
        consumers=consumers,
        fit_long=fit_long,
        product_df=product_df,
    )


def validate_category(data: CategoryData) -> Dict[str, object]:
    """Run consistency checks. Raise AssertionError on hard failures.

    Returns a small dict of summary statistics for the QC report.
    """
    cat = data.category
    catalog_pids = {p["product_id"] for p in data.catalog["products"]}
    fit_pids = set(data.fit_long["product_id"].unique())

    missing = fit_pids - catalog_pids
    extra = catalog_pids - fit_pids
    assert not missing, f"[{cat}] fit_scores has unknown product_ids: {missing}"
    assert not extra, f"[{cat}] catalog products missing from fit_scores: {extra}"

    consumer_ids_consumers = {c["consumer_id"] for c in data.consumers}
    consumer_ids_fit = set(data.fit_long["consumer_id"].unique())
    assert consumer_ids_fit == consumer_ids_consumers, (
        f"[{cat}] consumer IDs in fit scores do not match consumer file"
    )

    q = data.fit_long["Q"]
    assert q.std() > 1e-6, f"[{cat}] Q has no variation"

    return {
        "category": cat,
        "n_consumers": len(data.consumers),
        "n_products": len(data.product_df),
        "n_pairs": len(data.fit_long),
        "Q_mean": float(q.mean()),
        "Q_std": float(q.std(ddof=0)),
        "Q_min": float(q.min()),
        "Q_max": float(q.max()),
        "n_incumbent_products": int(data.product_df["incumbent"].sum()),
        "n_focal_products": int(data.product_df["focal_brand"].sum()),
    }


def load_all_categories(verbose: bool = True) -> Dict[str, CategoryData]:
    out = {}
    summaries = []
    for c in list_categories():
        d = load_category_data(c)
        s = validate_category(d)
        summaries.append(s)
        out[c] = d
    if verbose:
        print("Category data summary:")
        df = pd.DataFrame(summaries)
        print(df.to_string(index=False))
    return out


if __name__ == "__main__":
    # Quick self-test
    data = load_all_categories(verbose=True)
    print()
    for c, d in data.items():
        print(f"\n[{c}] sample of fit_long:")
        print(d.fit_long.head(3).to_string(index=False))
