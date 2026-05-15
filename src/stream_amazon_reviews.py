"""Stream Amazon reviews from UCSD dataset, extracting only our 60 ASINs.

Streams gzipped JSONL over HTTP without saving the full file to disk.
Keeps up to `quota` reviews per product (proportional to rating_count).

Usage:
  python src/stream_amazon_reviews.py [--max-per-product 100]
"""

from __future__ import annotations

import gzip
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "real_metadata"
CACHE_DIR = DATA_DIR / "review_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

REVIEW_URLS = {
    "Electronics": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz",
    "Cell_Phones": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Cell_Phones_and_Accessories.jsonl.gz",
}


def load_products_and_quotas(max_per_product: int = 100) -> tuple[set[str], dict[str, int], pd.DataFrame]:
    products = pd.concat([
        pd.read_csv(DATA_DIR / f"products_{cat}.csv")
        for cat in ["headphones", "smartwatches"]
    ], ignore_index=True)

    target_asins = set(products["asin"].tolist())

    total_ratings = products["rating_count"].sum()
    total_budget = max_per_product * len(products)
    quotas = {}
    for _, row in products.iterrows():
        raw = row["rating_count"] / total_ratings * total_budget
        quotas[row["asin"]] = int(np.clip(round(raw), 5, max_per_product))

    return target_asins, quotas, products


def stream_reviews_http(url: str, target_asins: set[str],
                        quotas: dict[str, int],
                        collected: dict[str, list[dict]]) -> None:
    """Stream gzipped JSONL over HTTP, keeping only matching ASINs."""
    print(f"\n  Streaming from: {url.split('/')[-1]}")

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (academic research)")

    start = time.time()
    resp = urllib.request.urlopen(req, timeout=60)
    total_size = int(resp.headers.get("Content-Length", 0))
    print(f"  File size: {total_size / 1e9:.2f} GB")

    decompressor = gzip.GzipFile(fileobj=resp)
    reader = io.TextIOWrapper(decompressor, encoding="utf-8", errors="replace")

    scanned = 0
    matched = 0
    saturated = set()

    for line in reader:
        scanned += 1
        if scanned % 2_000_000 == 0:
            elapsed = time.time() - start
            print(f"    {scanned:>10,} lines | {matched:>5} matched | "
                  f"{len(saturated)}/{len(target_asins)} saturated | "
                  f"{elapsed:.0f}s elapsed")

        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        asin = item.get("parent_asin") or item.get("asin", "")
        if asin not in target_asins or asin in saturated:
            continue

        text = (item.get("text") or "").strip()
        if len(text) < 30:
            continue

        review = {
            "asin": asin,
            "rating": item.get("rating", 0),
            "title": (item.get("title") or "").strip()[:200],
            "text": text[:1000],
            "helpful_vote": item.get("helpful_vote", 0),
            "verified_purchase": item.get("verified_purchase", False),
            "timestamp": item.get("timestamp", 0),
        }
        collected[asin].append(review)
        matched += 1

        quota = quotas.get(asin, 100)
        if len(collected[asin]) >= quota * 3:
            collected[asin].sort(key=lambda r: r["helpful_vote"], reverse=True)
            collected[asin] = collected[asin][:quota]
            saturated.add(asin)

            if len(saturated) == len(target_asins):
                print(f"    All {len(target_asins)} products saturated — stopping early.")
                break

    elapsed = time.time() - start
    print(f"  Done: {scanned:,} lines, {matched} matched, {elapsed:.0f}s")


def stream_reviews_local(gz_path: Path, target_asins: set[str],
                         quotas: dict[str, int],
                         collected: dict[str, list[dict]]) -> None:
    """Stream from a local gzipped file (if previously downloaded)."""
    print(f"\n  Streaming from local: {gz_path.name}")

    start = time.time()
    scanned = 0
    matched = 0
    saturated = set()

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            scanned += 1
            if scanned % 2_000_000 == 0:
                elapsed = time.time() - start
                print(f"    {scanned:>10,} lines | {matched:>5} matched | "
                      f"{len(saturated)}/{len(target_asins)} saturated | "
                      f"{elapsed:.0f}s elapsed")

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = item.get("parent_asin") or item.get("asin", "")
            if asin not in target_asins or asin in saturated:
                continue

            text = (item.get("text") or "").strip()
            if len(text) < 30:
                continue

            review = {
                "asin": asin,
                "rating": item.get("rating", 0),
                "title": (item.get("title") or "").strip()[:200],
                "text": text[:1000],
                "helpful_vote": item.get("helpful_vote", 0),
                "verified_purchase": item.get("verified_purchase", False),
                "timestamp": item.get("timestamp", 0),
            }
            collected[asin].append(review)
            matched += 1

            quota = quotas.get(asin, 100)
            if len(collected[asin]) >= quota * 3:
                collected[asin].sort(key=lambda r: r["helpful_vote"], reverse=True)
                collected[asin] = collected[asin][:quota]
                saturated.add(asin)

                if len(saturated) == len(target_asins):
                    print(f"    All {len(target_asins)} products saturated — stopping early.")
                    break

    elapsed = time.time() - start
    print(f"  Done: {scanned:,} lines, {matched} matched, {elapsed:.0f}s")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-product", type=int, default=100)
    args = parser.parse_args()

    target_asins, quotas, products = load_products_and_quotas(args.max_per_product)
    print(f"Target ASINs: {len(target_asins)}")
    print(f"Total review budget: {sum(quotas.values())}")

    collected: dict[str, list[dict]] = {a: [] for a in target_asins}

    # Check for local review files first (from prior download attempts)
    local_files = {
        "Electronics": Path("/tmp/reviews_Electronics.jsonl.gz"),
        "Cell_Phones": Path("/tmp/reviews_Cell_Phones_and_Accessories.jsonl.gz"),
    }

    for name, url in REVIEW_URLS.items():
        local = local_files.get(name)
        if local and local.exists() and local.stat().st_size > 1_000_000:
            stream_reviews_local(local, target_asins, quotas, collected)
        else:
            stream_reviews_http(url, target_asins, quotas, collected)

    # Final trim to quotas
    for asin in collected:
        collected[asin].sort(key=lambda r: r["helpful_vote"], reverse=True)
        collected[asin] = collected[asin][:quotas.get(asin, 100)]

    # Save per-product cache
    for asin, revs in collected.items():
        if revs:
            cache_path = CACHE_DIR / f"{asin}.json"
            with open(cache_path, "w") as f:
                json.dump(revs, f)

    # Flatten and save
    rows = []
    for asin, revs in collected.items():
        rows.extend(revs)

    review_df = pd.DataFrame(rows)
    out_path = DATA_DIR / "amazon_reviews_raw.csv"
    review_df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Total reviews: {len(review_df)}")
    products_with = review_df["asin"].nunique() if len(review_df) > 0 else 0
    print(f"  Products with reviews: {products_with}/{len(target_asins)}")

    if len(review_df) > 0:
        counts = review_df.groupby("asin").size()
        print(f"  Per-product: min={counts.min()}, max={counts.max()}, "
              f"median={counts.median():.0f}")

    missing = target_asins - (set(review_df["asin"].unique()) if len(review_df) > 0 else set())
    if missing:
        print(f"\n  Missing ({len(missing)}):")
        for asin in sorted(missing):
            prod = products[products["asin"] == asin]
            if len(prod) > 0:
                row = prod.iloc[0]
                print(f"    {asin}: {row['brand']} — {row['title'][:50]}")

    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
