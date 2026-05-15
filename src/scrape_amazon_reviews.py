"""Scrape Amazon product reviews for curated products.

Collects real review text from Amazon product pages using ASINs.
Review count per product is proportional to rating_count (sales proxy),
capped at 100.

Usage:
  python src/scrape_amazon_reviews.py [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path
from urllib.parse import urlencode

import requests
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "real_metadata"
OUT_RAW = DATA_DIR / "amazon_reviews_raw.csv"
CACHE_DIR = DATA_DIR / "review_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
}


def compute_quotas(products: pd.DataFrame, max_per_product: int = 100,
                   min_per_product: int = 5) -> dict[str, int]:
    total_ratings = products["rating_count"].sum()
    total_budget = max_per_product * len(products)
    quotas = {}
    for _, row in products.iterrows():
        raw = row["rating_count"] / total_ratings * total_budget
        quotas[row["asin"]] = int(np.clip(round(raw), min_per_product, max_per_product))
    return quotas


def scrape_reviews_for_asin(asin: str, quota: int,
                            session: requests.Session) -> list[dict]:
    """Scrape up to `quota` reviews for a single ASIN from Amazon."""
    cache_path = CACHE_DIR / f"{asin}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if len(cached) >= quota:
            return cached[:quota]
        existing = cached
    else:
        existing = []

    reviews = list(existing)
    seen_texts = {r["text"][:100] for r in reviews}
    page = 1
    max_pages = (quota // 10) + 2
    consecutive_empty = 0

    while len(reviews) < quota and page <= max_pages and consecutive_empty < 3:
        url = (
            f"https://www.amazon.com/product-reviews/{asin}"
            f"?pageNumber={page}&sortBy=helpful"
        )

        try:
            time.sleep(random.uniform(2.0, 5.0))
            resp = session.get(url, headers=HEADERS, timeout=15)

            if resp.status_code == 503:
                print(f"    Rate limited on page {page}, waiting 30s...")
                time.sleep(30)
                resp = session.get(url, headers=HEADERS, timeout=15)

            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code} on page {page}")
                consecutive_empty += 1
                page += 1
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            review_divs = soup.select("div[data-hook='review']")

            if not review_divs:
                consecutive_empty += 1
                page += 1
                continue

            consecutive_empty = 0
            new_on_page = 0

            for div in review_divs:
                if len(reviews) >= quota:
                    break

                title_el = div.select_one("a[data-hook='review-title'] span")
                if not title_el:
                    title_el = div.select_one("[data-hook='review-title']")
                title = title_el.get_text(strip=True) if title_el else ""

                body_el = div.select_one("span[data-hook='review-body']")
                text = body_el.get_text(strip=True) if body_el else ""

                if len(text) < 20:
                    continue

                text_key = text[:100]
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)

                rating_el = div.select_one("i[data-hook='review-star-rating'] span")
                rating = 0.0
                if rating_el:
                    m = re.search(r"([\d.]+)", rating_el.get_text())
                    if m:
                        rating = float(m.group(1))

                helpful_el = div.select_one("span[data-hook='helpful-vote-statement']")
                helpful = 0
                if helpful_el:
                    m = re.search(r"(\d[\d,]*)", helpful_el.get_text())
                    if m:
                        helpful = int(m.group(1).replace(",", ""))

                verified_el = div.select_one("span[data-hook='avp-badge']")
                verified = verified_el is not None

                date_el = div.select_one("span[data-hook='review-date']")
                date_str = date_el.get_text(strip=True) if date_el else ""

                reviews.append({
                    "asin": asin,
                    "rating": rating,
                    "title": title[:200],
                    "text": text[:1000],
                    "helpful_vote": helpful,
                    "verified_purchase": verified,
                    "date": date_str,
                })
                new_on_page += 1

            page += 1

        except requests.RequestException as e:
            print(f"    Request error on page {page}: {e}")
            consecutive_empty += 1
            time.sleep(10)
            page += 1

    with open(cache_path, "w") as f:
        json.dump(reviews, f)

    return reviews[:quota]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-per-product", type=int, default=100)
    args = parser.parse_args()

    products = pd.concat([
        pd.read_csv(DATA_DIR / f"products_{cat}.csv")
        for cat in ["headphones", "smartwatches"]
    ], ignore_index=True)

    quotas = compute_quotas(products, max_per_product=args.max_per_product)

    print(f"Products: {len(products)}")
    print(f"Total review budget: {sum(quotas.values())}")

    if args.dry_run:
        print("\n[DRY RUN] Would scrape:")
        for _, row in products.iterrows():
            print(f"  {row['asin']}  {row['brand']:15s}  "
                  f"quota={quotas[row['asin']]:>3}  {row['title'][:40]}")
        return

    session = requests.Session()
    all_reviews = []

    for i, (_, row) in enumerate(products.iterrows()):
        asin = row["asin"]
        quota = quotas[asin]
        cache_path = CACHE_DIR / f"{asin}.json"

        if cache_path.exists():
            with open(cache_path) as f:
                cached = json.load(f)
            if len(cached) >= quota:
                print(f"[{i+1:>2}/{len(products)}] {row['brand']:15s} "
                      f"{asin} — cached ({len(cached)} reviews)")
                all_reviews.extend(cached[:quota])
                continue

        print(f"[{i+1:>2}/{len(products)}] {row['brand']:15s} "
              f"{asin} — scraping {quota} reviews...")
        revs = scrape_reviews_for_asin(asin, quota, session)
        print(f"    Got {len(revs)} reviews")
        all_reviews.extend(revs)

    review_df = pd.DataFrame(all_reviews)
    review_df.to_csv(OUT_RAW, index=False)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Total reviews: {len(review_df)}")
    print(f"  Products with reviews: {review_df['asin'].nunique()}/{len(products)}")
    counts = review_df.groupby("asin").size()
    print(f"  Reviews per product: min={counts.min()}, max={counts.max()}, "
          f"median={counts.median():.0f}")

    missing = set(products["asin"]) - set(review_df["asin"].unique())
    if missing:
        print(f"\n  Products with NO reviews ({len(missing)}):")
        for asin in missing:
            prod = products[products["asin"] == asin]
            if len(prod) > 0:
                print(f"    {asin}: {prod.iloc[0]['brand']} — "
                      f"{prod.iloc[0]['title'][:50]}")

    print(f"\n  Saved → {OUT_RAW}")


if __name__ == "__main__":
    main()
