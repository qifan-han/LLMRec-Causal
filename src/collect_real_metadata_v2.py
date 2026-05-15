"""Extract real product metadata from Amazon Reviews 2023 dataset (v2).

Stricter filtering: excludes accessories, adapters, cases, cables.
Better curation: requires price, enforces product-type validation.

Usage:
  python src/collect_real_metadata_v2.py
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "real_metadata"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_ELECTRONICS = Path("/tmp/meta_Electronics.jsonl.gz")
META_CELLPHONES = Path("/tmp/meta_Cell_Phones_and_Accessories.jsonl.gz")

# ── Headphones: must be an actual headphone/earbud product ──────────

HP_MUST_MATCH = [
    "headphone", "headset", "earphone", "earbud", "earbuds",
    "over-ear", "on-ear", "in-ear",
]

HP_EXCLUDE_TITLE = [
    "adapter", "cable", "cord", "splitter", "extension",
    "replacement", "ear pad", "ear cushion", "ear tip", "ear gel",
    "ear hook", "headband", "case", "stand", "holder", "mount",
    "amplifier", "amp", "dac", "microphone", "mic stand",
    "speaker", "portable speaker", "bluetooth speaker",
    "laptop", "computer", "tablet", "phone", "tv", "monitor",
    "walkie talkie", "two way radio", "radio", "alarm clock",
    "screen protector", "charger", "power bank",
    "jack adapter", "dongle", "converter",
]

# ── Smartwatches: must be an actual watch, not an accessory ─────────

SW_MUST_MATCH = [
    "smartwatch", "smart watch", "fitness tracker", "fitness watch",
    "gps watch", "sport watch", "activity tracker",
    "apple watch", "galaxy watch", "garmin", "fitbit versa",
    "fitbit sense", "fitbit charge",
]

SW_EXCLUDE_TITLE = [
    "band", "strap", "bracelet", "case", "cover", "protector",
    "screen protector", "tempered glass", "film", "bumper",
    "charger", "charging", "cable", "cord", "dock", "cradle",
    "holder", "stand", "mount", "organizer", "storage",
    "replacement", "accessory pack",
]


def title_matches(title: str, must_match: list[str], exclude: list[str]) -> bool:
    t = title.lower()
    if not any(kw in t for kw in must_match):
        return False
    if any(kw in t for kw in exclude):
        return False
    return True


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
                val = str(details[key]).strip()
                if val and val.lower() not in ("unknown", "unbranded", "generic", "n/a"):
                    return val
    store = (item.get("store") or "").strip()
    return store if store else ""


def extract_price(item: dict) -> float | None:
    price = item.get("price")
    if price is None:
        return None
    if isinstance(price, (int, float)):
        return float(price) if price > 0 else None
    if isinstance(price, str):
        match = re.search(r"[\d]+\.?\d*", price.replace(",", ""))
        if match:
            val = float(match.group())
            return val if val > 0 else None
    return None


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


def scan_file(gz_path: Path, must_match: list[str], exclude: list[str],
              label: str, min_ratings: int = 50) -> list[dict]:
    matches = []
    scanned = 0

    print(f"Scanning {gz_path.name} for {label}...")
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            scanned += 1
            if scanned % 200000 == 0:
                print(f"  scanned {scanned:,}, found {len(matches)} {label}...")

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            rating_count = item.get("rating_number", 0) or 0
            if rating_count < min_ratings:
                continue

            title = (item.get("title") or "").strip()
            if not title or len(title) < 15:
                continue

            if title_matches(title, must_match, exclude):
                processed = process_item(item)
                if processed["brand"]:
                    matches.append(processed)

    print(f"  Done: scanned {scanned:,}, found {len(matches)} {label}")
    return matches


def curate_products(df: pd.DataFrame, category: str,
                    target_n: int = 30) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset=["asin"])
    df = df.drop_duplicates(subset=["title"])

    # Sort by rating count (popularity proxy)
    df = df.sort_values("rating_count", ascending=False).reset_index(drop=True)

    if len(df) <= target_n:
        selected = df
    else:
        # Stratified: top 10 bestsellers, mid 10, tail 10
        n_top = 10
        n_mid = 10
        n_tail = target_n - n_top - n_mid

        top = df.head(n_top)
        remaining = df.iloc[n_top:]

        # Mid: from top 25-75th percentile, diversify by brand
        mid_pool = remaining.iloc[:len(remaining)//2]
        mid_selected = _diverse_select(mid_pool, n_mid, set(top["brand"].values))

        # Tail: from bottom half, diversify by brand
        tail_pool = remaining.iloc[len(remaining)//2:]
        used_brands = set(top["brand"].values)
        if len(mid_selected) > 0:
            used_brands |= set(pd.DataFrame(mid_selected)["brand"].values)
        tail_selected = _diverse_select(tail_pool, n_tail, used_brands)

        selected = pd.concat([
            top,
            pd.DataFrame(mid_selected),
            pd.DataFrame(tail_selected),
        ], ignore_index=True)

    selected = selected.head(target_n)
    selected["popularity_rank"] = range(1, len(selected) + 1)
    selected["category"] = category
    selected = _assign_tiers(selected)
    return selected.reset_index(drop=True)


def _diverse_select(pool: pd.DataFrame, n: int, used_brands: set) -> list[dict]:
    selected = []
    for _, row in pool.iterrows():
        if len(selected) >= n:
            break
        if row["brand"] not in used_brands:
            selected.append(row.to_dict())
            used_brands.add(row["brand"])
    # Fill remaining from any brand
    for _, row in pool.iterrows():
        if len(selected) >= n:
            break
        if row.to_dict() not in selected:
            selected.append(row.to_dict())
    return selected[:n]


def _assign_tiers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    prices = df["price"].dropna()
    if len(prices) >= 3:
        q33 = prices.quantile(0.33)
        q67 = prices.quantile(0.67)
        df["price_tier"] = df["price"].apply(
            lambda p: "budget" if pd.notna(p) and p <= q33
            else ("midrange" if pd.notna(p) and p <= q67
                  else ("premium" if pd.notna(p) else "unknown"))
        )
    else:
        df["price_tier"] = "unknown"

    # Popularity tier based on position
    df["popularity_tier"] = df["popularity_rank"].apply(
        lambda r: "bestseller" if r <= 5
        else ("mainstream" if r <= 15
              else ("niche" if r <= 25 else "long_tail"))
    )
    return df


def print_diagnostic(df: pd.DataFrame, category: str):
    print(f"\n{'=' * 80}")
    print(f"  {category.upper()} — {len(df)} products")
    print(f"{'=' * 80}")

    print(f"\n  Rating count: min={df['rating_count'].min():,}, "
          f"max={df['rating_count'].max():,}, "
          f"median={df['rating_count'].median():,.0f}")

    has_price = df["price"].notna()
    print(f"  Price coverage: {has_price.sum()}/{len(df)}")
    if has_price.any():
        print(f"  Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")

    print(f"  Rating range: {df['average_rating'].min():.1f} - {df['average_rating'].max():.1f}")
    print(f"  Unique brands: {df['brand'].nunique()}")
    print(f"  Price tiers: {dict(df['price_tier'].value_counts())}")
    print(f"  Popularity tiers: {dict(df['popularity_tier'].value_counts())}")

    print(f"\n  Full product list:")
    print(f"  {'Rank':>4} {'Brand':20s} {'Title':55s} {'Price':>8} {'Ratings':>8} {'★':>4} {'Pop Tier':12} {'Price Tier'}")
    print(f"  {'─'*4} {'─'*20} {'─'*55} {'─'*8} {'─'*8} {'─'*4} {'─'*12} {'─'*10}")
    for _, r in df.iterrows():
        price_str = f"${r['price']:.2f}" if pd.notna(r['price']) else "N/A"
        print(f"  {r['popularity_rank']:>4} {r['brand'][:20]:20s} "
              f"{r['title'][:55]:55s} {price_str:>8} "
              f"{r['rating_count']:>8,} {r['average_rating']:>4.1f} "
              f"{r['popularity_tier']:12} {r['price_tier']}")

    # Correlation checks
    has_both = df[df["price"].notna() & (df["rating_count"] > 0)]
    if len(has_both) >= 5:
        corr_price = has_both["rating_count"].corr(has_both["price"])
        corr_rating = has_both["rating_count"].corr(has_both["average_rating"])
        print(f"\n  Corr(rating_count, price): {corr_price:.3f}")
        print(f"  Corr(rating_count, avg_rating): {corr_rating:.3f}")


def main():
    results = {}

    # ── Headphones ──
    if META_ELECTRONICS.exists():
        hp_raw = scan_file(META_ELECTRONICS, HP_MUST_MATCH, HP_EXCLUDE_TITLE,
                           "headphones", min_ratings=50)
        hp_df = pd.DataFrame(hp_raw)
        print(f"\nFiltered headphones: {len(hp_df)}")

        hp_curated = curate_products(hp_df, "headphones", target_n=30)
        hp_curated.to_csv(OUT_DIR / "products_headphones_raw.csv", index=False)
        results["headphones"] = hp_curated
        print_diagnostic(hp_curated, "headphones")

    # ── Smartwatches ──
    sw_raw = []
    for src in [META_CELLPHONES, META_ELECTRONICS]:
        if src.exists():
            sw_raw.extend(scan_file(src, SW_MUST_MATCH, SW_EXCLUDE_TITLE,
                                    "smartwatches", min_ratings=50))

    if sw_raw:
        sw_df = pd.DataFrame(sw_raw).drop_duplicates(subset=["asin"])
        print(f"\nFiltered smartwatches: {len(sw_df)}")

        sw_curated = curate_products(sw_df, "smartwatches", target_n=30)
        sw_curated.to_csv(OUT_DIR / "products_smartwatches_raw.csv", index=False)
        results["smartwatches"] = sw_curated
        print_diagnostic(sw_curated, "smartwatches")

    # ── Summary ──
    print(f"\n{'=' * 80}")
    print(f"  FILES SAVED")
    print(f"{'=' * 80}")
    for cat in results:
        path = OUT_DIR / f"products_{cat}_raw.csv"
        print(f"  {path}")

    # Quality flags
    print(f"\n{'=' * 80}")
    print(f"  QUALITY FLAGS")
    print(f"{'=' * 80}")
    for cat, df in results.items():
        missing_price = df["price"].isna().sum()
        if missing_price > 0:
            print(f"  [{cat}] {missing_price} products missing price — need manual lookup")
        low_ratings = (df["rating_count"] < 100).sum()
        if low_ratings > 0:
            print(f"  [{cat}] {low_ratings} products with < 100 ratings — check if real products")
        dup_brands = df["brand"].value_counts()
        repeats = dup_brands[dup_brands > 3]
        if len(repeats) > 0:
            print(f"  [{cat}] Brand concentration: {dict(repeats)}")


if __name__ == "__main__":
    main()
