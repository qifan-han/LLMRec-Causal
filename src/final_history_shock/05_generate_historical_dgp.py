"""05 — Build qualitative historical summaries from real Amazon metadata.

Uses real rating_count (popularity proxy), average_rating, and review
summaries to construct the history signal for the recommender. No
synthetic purchase simulation — all signals are derived from real data.

Output:
  data/final_history_shock/history_dgp/{cat}_history_qualitative.json
  data/final_history_shock/history_dgp/{cat}_history_aggregates.csv

Usage:
  python 05_generate_historical_dgp.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_openai import DATA_DIR

HIST_DIR = DATA_DIR / "history_dgp"
HIST_DIR.mkdir(parents=True, exist_ok=True)
CAT_DIR = DATA_DIR / "catalogs"
REAL_META_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "real_metadata"

CATEGORIES = ["headphones"]

SEGMENTS = {
    "headphones": [
        "budget_student", "commuter", "remote_worker", "audiophile",
        "gym_user", "gamer", "frequent_traveler", "casual_listener",
    ],
}

SEGMENT_FEATURE_AFFINITY = {
    "headphones": {
        "budget_student":    {"price_pref": "budget", "keywords": ["wireless", "bluetooth", "affordable"]},
        "commuter":          {"price_pref": "midrange", "keywords": ["noise cancel", "anc", "wireless", "bluetooth"]},
        "remote_worker":     {"price_pref": "midrange", "keywords": ["microphone", "mic", "comfort", "call"]},
        "audiophile":        {"price_pref": "premium", "keywords": ["studio", "hi-fi", "over-ear", "wired"]},
        "gym_user":          {"price_pref": "budget", "keywords": ["sweat", "sport", "water", "secure fit", "wireless"]},
        "gamer":             {"price_pref": "midrange", "keywords": ["gaming", "game", "latency", "microphone"]},
        "frequent_traveler": {"price_pref": "premium", "keywords": ["noise cancel", "anc", "foldable", "portable"]},
        "casual_listener":   {"price_pref": "budget", "keywords": ["wireless", "bluetooth", "lightweight"]},
    },
}


def compute_segment_affinity(product: pd.Series, segment: str, category: str) -> float:
    """How well a product matches a segment's needs (0-1)."""
    affinity = SEGMENT_FEATURE_AFFINITY.get(category, {}).get(segment, {})
    if not affinity:
        return 0.5

    score = 0.0
    feats_text = (str(product.get("features_raw", "")) + " " +
                  str(product.get("title", ""))).lower()

    kw_matches = sum(1 for kw in affinity["keywords"] if kw in feats_text)
    kw_score = kw_matches / max(len(affinity["keywords"]), 1)
    score += kw_score * 0.5

    price_tier = product.get("price_tier", "midrange")
    pref = affinity.get("price_pref", "midrange")
    if price_tier == pref:
        score += 0.3
    elif (price_tier == "midrange" and pref in ("budget", "premium")) or \
         (pref == "midrange" and price_tier in ("budget", "premium")):
        score += 0.15

    rating = product.get("average_rating", 4.0)
    score += (rating - 3.5) / 1.5 * 0.2

    return min(score, 1.0)


def popularity_language(rank: int, n_products: int, rating_count: int) -> str:
    pct = rank / n_products
    if pct <= 0.1:
        return f"one of the top sellers in its category with over {rating_count:,} customer ratings"
    elif pct <= 0.3:
        return f"a popular choice with {rating_count:,} customer ratings"
    elif pct <= 0.6:
        return f"a moderately popular option with {rating_count:,} ratings"
    else:
        return f"a niche product with a smaller but dedicated following ({rating_count:,} ratings)"


def satisfaction_language(avg_rating: float) -> str:
    if avg_rating >= 4.6:
        return "Buyers overwhelmingly rate it highly, with very few complaints"
    elif avg_rating >= 4.3:
        return "Most buyers are satisfied, though some report minor issues"
    elif avg_rating >= 4.0:
        return "Reviews are generally positive but with notable criticisms from some buyers"
    elif avg_rating >= 3.5:
        return "Reviews are mixed — roughly split between satisfied and dissatisfied buyers"
    else:
        return "A significant share of buyers report disappointment or quality issues"


def segment_fit_language(affinity: float, segment: str) -> str:
    seg_readable = segment.replace("_", " ")
    if affinity >= 0.7:
        return f"This is a strong match for {seg_readable} buyers based on features and price point"
    elif affinity >= 0.45:
        return f"A reasonable option for {seg_readable} buyers, though not an ideal match on all dimensions"
    else:
        return f"Not the typical choice for {seg_readable} buyers — may lack key features they prioritize"


def build_qualitative_summaries(catalog_df: pd.DataFrame, category: str,
                                reviews_df: pd.DataFrame | None = None) -> list[dict]:
    segments = SEGMENTS[category]
    n_products = len(catalog_df)
    summaries = []

    for seg in segments:
        for _, product in catalog_df.iterrows():
            pid = product["product_id"]
            rank = product["popularity_rank"]
            rating_count = product["rating_count"]
            avg_rating = product["average_rating"]

            pop_lang = popularity_language(rank, n_products, rating_count)
            sat_lang = satisfaction_language(avg_rating)
            affinity = compute_segment_affinity(product, seg, category)
            fit_lang = segment_fit_language(affinity, seg)

            review_snippet = ""
            if reviews_df is not None:
                prod_reviews = reviews_df[reviews_df["asin"] == product.get("asin", "")]
                if len(prod_reviews) > 0:
                    top = prod_reviews.nlargest(3, "helpful_vote")
                    snippets = []
                    for _, rev in top.iterrows():
                        text = str(rev.get("text", ""))[:150]
                        rating = rev.get("rating", 0)
                        snippets.append(f'"{text}..." (★{rating:.0f})')
                    review_snippet = "\nSample buyer feedback:\n" + "\n".join(f"  - {s}" for s in snippets)

            summary = (
                f"This product is {pop_lang}. "
                f"{sat_lang}. "
                f"{fit_lang}."
                f"{review_snippet}"
            )

            summaries.append({
                "segment": seg,
                "product_id": pid,
                "popularity_qualitative": pop_lang,
                "satisfaction_qualitative": sat_lang,
                "segment_fit_qualitative": fit_lang,
                "affinity_score": round(affinity, 3),
                "review_snippet": review_snippet.strip() if review_snippet else "",
                "summary": summary,
            })

    return summaries


def build_aggregates(catalog_df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Build product-level aggregate stats for reference."""
    rows = []
    for _, p in catalog_df.iterrows():
        rows.append({
            "product_id": p["product_id"],
            "brand": p["brand"],
            "price": p["price"],
            "price_tier": p["price_tier"],
            "average_rating": p["average_rating"],
            "rating_count": p["rating_count"],
            "popularity_rank": p["popularity_rank"],
            "popularity_tier": p["popularity_tier"],
            "category": category,
        })
    return pd.DataFrame(rows)


def main():
    for category in CATEGORIES:
        cat_path = CAT_DIR / f"{category}_catalog.csv"
        if not cat_path.exists():
            sys.exit(f"Catalog not found: {cat_path}. Run 01 first.")
        catalog = pd.read_csv(cat_path)

        reviews_df = None
        review_path = REAL_META_DIR / "amazon_reviews_raw.csv"
        if review_path.exists():
            reviews_df = pd.read_csv(review_path)
            n_matching = reviews_df[reviews_df["asin"].isin(catalog["asin"])].shape[0]
            print(f"  Loaded {n_matching} real reviews for {category}")
        else:
            print(f"  No real reviews found at {review_path} — using metadata only")

        print(f"\n--- {category} ---")
        print(f"  Catalog: {len(catalog)} products")

        agg = build_aggregates(catalog, category)
        agg_path = HIST_DIR / f"{category}_history_aggregates.csv"
        agg.to_csv(agg_path, index=False)
        print(f"  Aggregates: {len(agg)} products → {agg_path}")

        qual = build_qualitative_summaries(catalog, category, reviews_df)
        qual_path = HIST_DIR / f"{category}_history_qualitative.json"
        with open(qual_path, "w") as f:
            json.dump(qual, f, indent=2)
        print(f"  Qualitative: {len(qual)} entries → {qual_path}")

        seg_counts = {}
        for q in qual:
            seg_counts[q["segment"]] = seg_counts.get(q["segment"], 0) + 1
        print(f"  Segments: {len(seg_counts)} × {list(seg_counts.values())[0]} products")

        affinities = [q["affinity_score"] for q in qual]
        print(f"  Affinity range: {min(affinities):.3f} – {max(affinities):.3f}, "
              f"mean={np.mean(affinities):.3f}")

        with_snippets = sum(1 for q in qual if q["review_snippet"])
        print(f"  Entries with real review snippets: {with_snippets}/{len(qual)}")

    print("\nHistorical DGP complete.")


if __name__ == "__main__":
    main()
