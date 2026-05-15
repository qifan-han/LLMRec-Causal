"""Collect real Amazon reviews for curated products from Amazon Reviews 2023.

Streams through gzipped JSONL review files, extracting only reviews
matching our 60 curated product ASINs. Keeps top-K reviews per product
sorted by helpfulness (helpful_vote count).

The review files are from the same academic dataset (McAuley Lab, UCSD,
NeurIPS 2024) that we used for metadata.

Usage:
  python src/collect_real_reviews.py [--max-per-product 20] [--skip-download]
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "real_metadata"

REVIEW_FILES = {
    "Electronics": {
        "url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz",
        "local": Path("/tmp/reviews_Electronics.jsonl.gz"),
    },
    "Cell_Phones_and_Accessories": {
        "url": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Cell_Phones_and_Accessories.jsonl.gz",
        "local": Path("/tmp/reviews_Cell_Phones_and_Accessories.jsonl.gz"),
    },
}

REVIEW_SCHEMA = {
    "rating": float,
    "title": str,
    "text": str,
    "asin": str,
    "parent_asin": str,
    "user_id": str,
    "timestamp": int,
    "helpful_vote": int,
    "verified_purchase": bool,
}


def load_target_asins() -> set[str]:
    asins = set()
    for cat in ["headphones", "smartwatches"]:
        path = DATA_DIR / f"products_{cat}.csv"
        if path.exists():
            df = pd.read_csv(path)
            asins.update(df["asin"].tolist())
    return asins


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        size_gb = dest.stat().st_size / 1e9
        print(f"  Already exists: {dest.name} ({size_gb:.2f} GB)")
        return

    print(f"  Downloading {dest.name}...")
    start = time.time()

    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (academic research)")

    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(8 * 1024 * 1024)  # 8MB chunks
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    elapsed = time.time() - start
                    speed = downloaded / elapsed / 1e6
                    print(f"    {pct:.1f}% ({downloaded/1e9:.2f}/{total/1e9:.2f} GB) "
                          f"@ {speed:.1f} MB/s", end="\r")

    elapsed = time.time() - start
    print(f"\n  Downloaded in {elapsed:.0f}s")


def stream_extract_reviews(gz_path: Path, target_asins: set[str],
                           max_per_product: int) -> dict[str, list[dict]]:
    """Stream through gzipped JSONL, keep only matching reviews."""
    reviews: dict[str, list[dict]] = {asin: [] for asin in target_asins}
    scanned = 0
    matched = 0
    saturated = set()

    print(f"  Scanning {gz_path.name}...")
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            scanned += 1
            if scanned % 2_000_000 == 0:
                print(f"    scanned {scanned:,}, matched {matched}, "
                      f"saturated {len(saturated)}/{len(target_asins)}")
                if len(saturated) == len(target_asins):
                    print("    All products saturated, stopping early.")
                    break

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = item.get("parent_asin") or item.get("asin", "")
            if asin not in target_asins or asin in saturated:
                continue

            text = (item.get("text") or "").strip()
            if len(text) < 20:
                continue

            review = {
                "asin": asin,
                "rating": item.get("rating", 0),
                "title": (item.get("title") or "").strip(),
                "text": text[:1000],
                "helpful_vote": item.get("helpful_vote", 0),
                "verified_purchase": item.get("verified_purchase", False),
                "timestamp": item.get("timestamp", 0),
            }
            reviews[asin].append(review)
            matched += 1

            if len(reviews[asin]) >= max_per_product * 3:
                reviews[asin].sort(key=lambda r: r["helpful_vote"], reverse=True)
                reviews[asin] = reviews[asin][:max_per_product]
                saturated.add(asin)

    print(f"  Done: scanned {scanned:,}, matched {matched}")
    return reviews


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-product", type=int, default=20)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    target_asins = load_target_asins()
    print(f"Target ASINs: {len(target_asins)}")

    if not args.skip_download:
        print("\nDownloading review files...")
        for name, info in REVIEW_FILES.items():
            download_file(info["url"], info["local"])

    all_reviews: dict[str, list[dict]] = {a: [] for a in target_asins}

    for name, info in REVIEW_FILES.items():
        if not info["local"].exists():
            print(f"  Skipping {name} (not downloaded)")
            continue
        extracted = stream_extract_reviews(
            info["local"], target_asins, args.max_per_product
        )
        for asin, revs in extracted.items():
            all_reviews[asin].extend(revs)

    for asin in all_reviews:
        all_reviews[asin].sort(key=lambda r: r["helpful_vote"], reverse=True)
        all_reviews[asin] = all_reviews[asin][:args.max_per_product]

    rows = []
    for asin, revs in all_reviews.items():
        for r in revs:
            rows.append(r)

    review_df = pd.DataFrame(rows)
    out_path = DATA_DIR / "amazon_reviews_raw.csv"
    review_df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Total reviews collected: {len(review_df)}")
    products_with_reviews = review_df["asin"].nunique()
    print(f"  Products with reviews: {products_with_reviews}/{len(target_asins)}")

    counts = review_df.groupby("asin").size()
    print(f"  Reviews per product: min={counts.min()}, max={counts.max()}, "
          f"median={counts.median():.0f}")

    products = pd.concat([
        pd.read_csv(DATA_DIR / f"products_{cat}.csv")
        for cat in ["headphones", "smartwatches"]
    ])
    missing = set(target_asins) - set(review_df["asin"].unique())
    if missing:
        print(f"\n  Products with NO reviews ({len(missing)}):")
        for asin in missing:
            prod = products[products["asin"] == asin]
            if len(prod) > 0:
                print(f"    {asin}: {prod.iloc[0]['brand']} — "
                      f"{prod.iloc[0]['title'][:50]}")

    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
