"""01 — Load real product catalogs from curated Amazon metadata.

Reads products from data/real_metadata/products_*.csv and maps to the
schema expected by downstream pipeline scripts.

Output:
  data/final_history_shock/catalogs/{cat}_catalog.csv
  data/final_history_shock/catalogs/catalog_sources.json

Usage:
  python 01_build_or_collect_catalogs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_openai import DATA_DIR

CAT_DIR = DATA_DIR / "catalogs"
CAT_DIR.mkdir(parents=True, exist_ok=True)

REAL_META_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "real_metadata"
CATEGORIES = ["headphones"]


def load_and_map(category: str) -> pd.DataFrame:
    src = REAL_META_DIR / f"products_{category}.csv"
    if not src.exists():
        sys.exit(f"Product file not found: {src}")

    df = pd.read_csv(src)

    def parse_features(feat_str: str) -> list[str]:
        if not isinstance(feat_str, str) or not feat_str.strip():
            return []
        parts = [f.strip() for f in feat_str.split("|") if f.strip()]
        return [p[:120] for p in parts[:6]]

    def infer_best_for(row) -> list[str]:
        feats = (str(row.get("features", "")) + " " + str(row.get("title", ""))).lower()
        uses = []
        if any(k in feats for k in ["noise cancel", "anc"]):
            uses.append("commuting and travel")
        if any(k in feats for k in ["workout", "sweat", "sport", "water resistant", "ipx"]):
            uses.append("exercise and gym")
        if any(k in feats for k in ["gaming", "game", "low latency"]):
            uses.append("gaming")
        if any(k in feats for k in ["studio", "monitor", "hi-fi", "audiophile"]):
            uses.append("music production")
        if any(k in feats for k in ["wireless", "bluetooth"]):
            uses.append("everyday wireless use")
        if any(k in feats for k in ["over-ear", "over ear"]):
            uses.append("home and office listening")
        if any(k in feats for k in ["microphone", "mic", "call"]):
            uses.append("calls and remote work")
        if any(k in feats for k in ["kid", "child"]):
            uses.append("children")
        if row.get("price_tier") == "budget":
            uses.append("budget-conscious buyers")
        return uses[:3] if uses else ["general use"]

    def infer_drawbacks(row) -> list[str]:
        drawbacks = []
        feats = str(row.get("features", "")).lower()
        title = str(row.get("title", "")).lower()
        if "wireless" not in feats and "bluetooth" not in feats and "wireless" not in title:
            drawbacks.append("wired only — no Bluetooth")
        if row.get("price_tier") == "budget":
            drawbacks.append("build quality may not match premium alternatives")
        if row.get("price_tier") == "premium":
            drawbacks.append("high price point")
        if row.get("average_rating", 5) < 4.2:
            drawbacks.append("mixed user reviews on durability or comfort")
        return drawbacks[:2] if drawbacks else ["no major drawbacks reported"]

    mapped = pd.DataFrame({
        "product_id": [f"{category}_{i+1:03d}" for i in range(len(df))],
        "asin": df["asin"],
        "brand": df["brand"],
        "model_name": df["title"].str.replace(df["brand"].iloc[0], "", regex=False).str.strip(),
        "title": df["title"],
        "price": df["price"],
        "price_tier": df["price_tier"],
        "average_rating": df["average_rating"],
        "rating_count": df["rating_count"],
        "popularity_rank": df["popularity_rank"],
        "popularity_tier": df["popularity_tier"],
        "key_features": df.apply(lambda r: json.dumps(parse_features(r.get("features", ""))), axis=1),
        "best_for": df.apply(lambda r: json.dumps(infer_best_for(r)), axis=1),
        "drawbacks": df.apply(lambda r: json.dumps(infer_drawbacks(r)), axis=1),
        "review_summary": df.apply(
            lambda r: f"★{r['average_rating']:.1f} from {r['rating_count']:,} reviews. "
                      f"{r['popularity_tier'].replace('_', ' ').capitalize()} product.",
            axis=1,
        ),
        "features_raw": df["features"],
        "description": df["description"],
        "category": category,
    })

    # Fix model_name: strip brand prefix from title for each row individually
    for i, row in mapped.iterrows():
        brand = row["brand"]
        title = row["title"]
        model = title.replace(brand, "").strip().lstrip("-").strip()
        mapped.at[i, "model_name"] = model if model else title

    return mapped


def validate_catalog(df: pd.DataFrame, category: str) -> list[str]:
    issues = []
    if len(df) < 20:
        issues.append(f"Only {len(df)} products (need at least 20)")
    n_brands = df["brand"].nunique()
    if n_brands < 8:
        issues.append(f"Only {n_brands} brands (need at least 8)")
    if df["price"].isna().any():
        issues.append(f"{df['price'].isna().sum()} products missing price")
    n_tiers = df["price_tier"].nunique()
    if n_tiers < 3:
        issues.append(f"Only {n_tiers} price tiers")
    return issues


def main():
    sources = {}

    for category in CATEGORIES:
        print(f"\n--- {category} ---")
        df = load_and_map(category)

        issues = validate_catalog(df, category)
        if issues:
            print(f"  Validation issues:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"  PASSED validation ({len(df)} products, "
                  f"{df['brand'].nunique()} brands, "
                  f"price ${df['price'].min():.0f}-${df['price'].max():.0f})")

        out_path = CAT_DIR / f"{category}_catalog.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved → {out_path}")

        sources[category] = {
            "method": "real_amazon_metadata",
            "source_file": str(REAL_META_DIR / f"products_{category}.csv"),
            "dataset": "Amazon Reviews 2023 (McAuley Lab, NeurIPS 2024)",
            "n_products": len(df),
            "n_brands": int(df["brand"].nunique()),
            "price_range": [float(df["price"].min()), float(df["price"].max())],
            "rating_count_range": [int(df["rating_count"].min()), int(df["rating_count"].max())],
            "tier_distribution": df["price_tier"].value_counts().to_dict(),
        }

    with open(CAT_DIR / "catalog_sources.json", "w") as f:
        json.dump(sources, f, indent=2)

    print("\nCatalog loading complete.")


if __name__ == "__main__":
    main()
