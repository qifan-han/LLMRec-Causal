"""Extract real product metadata from Amazon Reviews 2023 dataset.

Filters Electronics metadata for headphones and smartwatches,
extracts key fields, and saves clean CSVs.

Usage:
  python src/collect_real_metadata.py
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "real_metadata"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_ELECTRONICS = Path("/tmp/meta_Electronics.jsonl.gz")
META_CELLPHONES = Path("/tmp/meta_Cell_Phones_and_Accessories.jsonl.gz")

HEADPHONE_KEYWORDS = [
    "headphone", "headset", "earphone", "earbud", "over-ear",
    "on-ear", "noise cancelling", "noise canceling", "wireless headphone",
    "bluetooth headphone", "anc headphone",
]

SMARTWATCH_KEYWORDS = [
    "smartwatch", "smart watch", "fitness tracker", "fitness watch",
    "gps watch", "sport watch", "health watch", "activity tracker",
]

EXCLUDE_KEYWORDS = [
    "case", "band", "strap", "charger", "cable", "adapter", "screen protector",
    "stand", "holder", "mount", "replacement", "cover", "sleeve", "pouch",
    "ear pad", "ear cushion", "ear tip", "ear gel", "ear hook",
    "headphone splitter", "headphone jack", "headphone amp",
]


def matches_category(item: dict, keywords: list[str], exclude: list[str]) -> bool:
    title = (item.get("title") or "").lower()
    cats = str(item.get("categories") or "").lower()
    features = " ".join(item.get("features") or []).lower()
    searchable = f"{title} {cats} {features}"

    if any(kw in searchable for kw in exclude):
        if not any(kw in title for kw in keywords[:3]):
            return False

    return any(kw in searchable for kw in keywords)


def extract_brand(item: dict) -> str:
    details = item.get("details")
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            details = {}
    if isinstance(details, dict):
        for key in ["Brand", "brand", "Manufacturer", "manufacturer"]:
            if key in details:
                return str(details[key]).strip()
    store = item.get("store") or ""
    return store.strip() if store else "Unknown"


def extract_price(item: dict) -> float | None:
    price = item.get("price")
    if price is None:
        return None
    if isinstance(price, (int, float)):
        return float(price)
    if isinstance(price, str):
        match = re.search(r"[\d]+\.?\d*", price.replace(",", ""))
        if match:
            return float(match.group())
    return None


def extract_details_field(item: dict, field: str) -> str:
    details = item.get("details")
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except json.JSONDecodeError:
            return ""
    if isinstance(details, dict):
        return str(details.get(field, ""))
    return ""


def process_item(item: dict) -> dict:
    return {
        "asin": item.get("parent_asin") or "",
        "title": (item.get("title") or "").strip(),
        "brand": extract_brand(item),
        "price": extract_price(item),
        "average_rating": item.get("average_rating"),
        "rating_count": item.get("rating_number", 0),
        "features": " | ".join(item.get("features") or []),
        "description": " ".join(item.get("description") or [])[:500],
        "store": (item.get("store") or "").strip(),
        "categories": str(item.get("categories") or ""),
    }


def scan_file(gz_path: Path, keywords: list[str], exclude: list[str],
              label: str, min_ratings: int = 5) -> list[dict]:
    """Scan a gzipped JSONL file and extract matching products."""
    matches = []
    scanned = 0

    print(f"Scanning {gz_path.name} for {label}...")
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            scanned += 1
            if scanned % 200000 == 0:
                print(f"  scanned {scanned:,} items, found {len(matches)} {label}...")

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            rating_count = item.get("rating_number", 0) or 0
            if rating_count < min_ratings:
                continue

            if matches_category(item, keywords, exclude):
                matches.append(process_item(item))

    print(f"  Done: scanned {scanned:,} items, found {len(matches)} {label}")
    return matches


def curate_products(df: pd.DataFrame, category: str,
                    target_n: int = 30) -> pd.DataFrame:
    """Select a diverse set of products from raw matches."""
    df = df.copy()
    df = df[df["rating_count"] >= 50]
    df = df.drop_duplicates(subset=["title"])
    df = df[df["title"].str.len() > 10]
    df = df[df["brand"] != "Unknown"]

    df = df.sort_values("rating_count", ascending=False)

    if len(df) <= target_n:
        return df.reset_index(drop=True)

    # Stratified selection: top sellers, mid-range, long-tail
    n_top = target_n // 3
    n_mid = target_n // 3
    n_tail = target_n - n_top - n_mid

    top = df.head(n_top)
    remaining = df.iloc[n_top:]

    mid_start = len(remaining) // 4
    mid_end = mid_start + len(remaining) // 2
    mid_pool = remaining.iloc[mid_start:mid_end]

    # Diversify by brand in mid selection
    mid_selected = []
    brands_used = set(top["brand"].values)
    for _, row in mid_pool.iterrows():
        if len(mid_selected) >= n_mid:
            break
        if row["brand"] not in brands_used or len(mid_selected) >= n_mid - 3:
            mid_selected.append(row)
            brands_used.add(row["brand"])
    mid = pd.DataFrame(mid_selected)

    tail_pool = remaining.iloc[mid_end:]
    if len(tail_pool) > 0:
        tail_brands = set(top["brand"].values) | set(mid["brand"].values) if len(mid) > 0 else set()
        tail_selected = []
        for _, row in tail_pool.iterrows():
            if len(tail_selected) >= n_tail:
                break
            if row["brand"] not in tail_brands or len(tail_selected) >= n_tail - 2:
                tail_selected.append(row)
                tail_brands.add(row["brand"])
        tail = pd.DataFrame(tail_selected)
    else:
        tail = pd.DataFrame()

    result = pd.concat([top, mid, tail], ignore_index=True)
    result = result.head(target_n)

    result["popularity_rank"] = range(1, len(result) + 1)
    result["category"] = category

    return result.reset_index(drop=True)


def assign_price_tier(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    prices = df["price"].dropna()
    if len(prices) == 0:
        df["price_tier"] = "unknown"
        return df

    q33 = prices.quantile(0.33)
    q67 = prices.quantile(0.67)
    df["price_tier"] = df["price"].apply(
        lambda p: "budget" if p <= q33 else ("midrange" if p <= q67 else "premium")
        if pd.notna(p) else "unknown"
    )
    return df


def main():
    all_results = {}

    # Headphones from Electronics
    if META_ELECTRONICS.exists():
        hp_raw = scan_file(META_ELECTRONICS, HEADPHONE_KEYWORDS, EXCLUDE_KEYWORDS,
                           "headphones", min_ratings=5)
        hp_df = pd.DataFrame(hp_raw)
        print(f"\nRaw headphone matches: {len(hp_df)}")
        if len(hp_df) > 0:
            print(f"  With 50+ ratings: {(hp_df['rating_count'] >= 50).sum()}")
            print(f"  With price: {hp_df['price'].notna().sum()}")
            print(f"  Unique brands: {hp_df['brand'].nunique()}")

            hp_curated = curate_products(hp_df, "headphones", target_n=30)
            hp_curated = assign_price_tier(hp_curated)
            hp_curated.to_csv(OUT_DIR / "products_headphones_raw.csv", index=False)
            all_results["headphones"] = hp_curated
            print(f"  Curated: {len(hp_curated)} headphones saved")
    else:
        print(f"Electronics metadata not found at {META_ELECTRONICS}")

    # Smartwatches from Cell Phones or Electronics
    sw_raw = []
    for src in [META_CELLPHONES, META_ELECTRONICS]:
        if src.exists():
            sw_raw.extend(scan_file(src, SMARTWATCH_KEYWORDS, EXCLUDE_KEYWORDS,
                                    "smartwatches", min_ratings=5))

    if sw_raw:
        sw_df = pd.DataFrame(sw_raw).drop_duplicates(subset=["asin"])
        print(f"\nRaw smartwatch matches: {len(sw_df)}")
        if len(sw_df) > 0:
            print(f"  With 50+ ratings: {(sw_df['rating_count'] >= 50).sum()}")
            print(f"  With price: {sw_df['price'].notna().sum()}")
            print(f"  Unique brands: {sw_df['brand'].nunique()}")

            sw_curated = curate_products(sw_df, "smartwatches", target_n=30)
            sw_curated = assign_price_tier(sw_curated)
            sw_curated.to_csv(OUT_DIR / "products_smartwatches_raw.csv", index=False)
            all_results["smartwatches"] = sw_curated
            print(f"  Curated: {len(sw_curated)} smartwatches saved")

    # Diagnostic summary
    print(f"\n{'=' * 60}")
    print(f"  DIAGNOSTIC SUMMARY")
    print(f"{'=' * 60}")
    for cat, df in all_results.items():
        print(f"\n{cat.upper()} ({len(df)} products):")
        print(f"  Rating count range: {df['rating_count'].min():,} - {df['rating_count'].max():,}")
        print(f"  Rating count median: {df['rating_count'].median():,.0f}")
        if df["price"].notna().any():
            print(f"  Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        print(f"  Avg rating range: {df['average_rating'].min():.1f} - {df['average_rating'].max():.1f}")
        print(f"  Unique brands: {df['brand'].nunique()}")
        print(f"  Price tiers: {dict(df['price_tier'].value_counts())}")
        print(f"  Top 5 by review count:")
        for _, r in df.head(5).iterrows():
            print(f"    {r['brand']:20s} {r['title'][:50]:50s} "
                  f"ratings={r['rating_count']:>6,} "
                  f"${r['price'] if pd.notna(r['price']) else '?':>8} "
                  f"★{r['average_rating']:.1f}")


if __name__ == "__main__":
    main()
