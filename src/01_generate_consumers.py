"""
Generate consumer profiles and ground-truth fit scores.
Phase 1B-1C of the simulation implementation plan.

Creates 1,000 consumer profiles per category with:
- Budget (lognormal, category-specific)
- Price/quality sensitivity (Uniform 0.2-1.0)
- Brand familiarity (Beta, incumbent > entrant, tech-savvy boosts entrant knowledge)
- Use case (categorical with category-specific weights)
- DGP-only parameters: persuasion_susceptibility, comparison_preference, trust_in_ai

Computes Q_ij fit scores for all (consumer, product) pairs using:
  Q_ij = w_price * price_sensitivity * price_fit
       + w_quality * quality_sensitivity * quality_score/100
       + w_usecase * use_case_fit[use_case]
       + w_brand * brand_familiarity[brand]
"""

import json
import csv
import pathlib
import numpy as np
from collections import Counter

SEED = 42
N_CONSUMERS = 1000

ROOT = pathlib.Path(__file__).resolve().parent.parent
CATALOG_DIR = ROOT / "data" / "catalogs"
CONSUMER_DIR = ROOT / "data" / "consumers"
FIT_DIR = ROOT / "data" / "fit_scores"

CONSUMER_DIR.mkdir(parents=True, exist_ok=True)
FIT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_CONFIG = {
    "phone_charger": {
        "budget_log_mu": 3.6,
        "budget_log_sigma": 0.45,
        "budget_clip": (15, 120),
        "use_case_weights": {
            "travel": 0.25,
            "office_desk": 0.20,
            "bedside": 0.15,
            "fast_phone_only": 0.25,
            "multi_device": 0.15,
        },
        "fit_weights": {
            "price": 0.25,
            "quality": 0.25,
            "usecase": 0.40,
            "brand": 0.10,
        },
    },
    "headphones": {
        "budget_log_mu": 5.25,
        "budget_log_sigma": 0.55,
        "budget_clip": (30, 600),
        "use_case_weights": {
            "commuting": 0.25,
            "office_work": 0.20,
            "gym_running": 0.10,
            "audiophile_home": 0.10,
            "gaming": 0.15,
            "budget_casual": 0.20,
        },
        "fit_weights": {
            "price": 0.20,
            "quality": 0.30,
            "usecase": 0.35,
            "brand": 0.15,
        },
    },
    "laptop": {
        "budget_log_mu": 7.1,
        "budget_log_sigma": 0.45,
        "budget_clip": (400, 3500),
        "use_case_weights": {
            "data_analysis": 0.15,
            "software_development": 0.15,
            "creative_media": 0.10,
            "business_travel": 0.20,
            "student_general": 0.25,
            "budget_office": 0.15,
        },
        "fit_weights": {
            "price": 0.25,
            "quality": 0.25,
            "usecase": 0.30,
            "brand": 0.20,
        },
    },
}

INCUMBENT_FAMILIARITY_PARAMS = (6, 3)  # Beta, mean ~0.67
ENTRANT_FAMILIARITY_PARAMS = (2, 6)    # Beta, mean ~0.25
ENTRANT_TECH_BOOST = 0.15


def load_catalog(category):
    with open(CATALOG_DIR / f"{category}.json") as f:
        return json.load(f)


def get_brands(catalog):
    brands = {}
    for p in catalog["products"]:
        if p["brand_name"] not in brands:
            brands[p["brand_name"]] = p["brand_status"]
    return brands


def generate_consumers(category, config, catalog, rng):
    brands = get_brands(catalog)
    use_cases = list(config["use_case_weights"].keys())
    use_case_probs = np.array(list(config["use_case_weights"].values()))
    use_case_probs = use_case_probs / use_case_probs.sum()

    budgets = np.clip(
        rng.lognormal(config["budget_log_mu"], config["budget_log_sigma"], N_CONSUMERS),
        *config["budget_clip"],
    )
    price_sens = rng.uniform(0.2, 1.0, N_CONSUMERS)
    quality_sens = rng.uniform(0.2, 1.0, N_CONSUMERS)
    tech_savvy = rng.uniform(0.0, 1.0, N_CONSUMERS)
    persuasion_susc = rng.uniform(0.0, 1.0, N_CONSUMERS)
    comparison_pref = rng.uniform(0.0, 1.0, N_CONSUMERS)
    trust_ai = rng.uniform(0.0, 1.0, N_CONSUMERS)
    assigned_use_cases = rng.choice(use_cases, size=N_CONSUMERS, p=use_case_probs)

    consumers = []
    for i in range(N_CONSUMERS):
        brand_fam = {}
        for brand_name, status in brands.items():
            if status == "incumbent":
                base = rng.beta(*INCUMBENT_FAMILIARITY_PARAMS)
            else:
                base = rng.beta(*ENTRANT_FAMILIARITY_PARAMS)
                base = min(1.0, base + ENTRANT_TECH_BOOST * tech_savvy[i])
            brand_fam[brand_name] = round(float(base), 3)

        consumers.append({
            "consumer_id": i,
            "category": category,
            "budget": round(float(budgets[i]), 2),
            "price_sensitivity": round(float(price_sens[i]), 3),
            "quality_sensitivity": round(float(quality_sens[i]), 3),
            "brand_familiarity": brand_fam,
            "use_case": str(assigned_use_cases[i]),
            "tech_savviness": round(float(tech_savvy[i]), 3),
            "persuasion_susceptibility": round(float(persuasion_susc[i]), 3),
            "comparison_preference": round(float(comparison_pref[i]), 3),
            "trust_in_ai": round(float(trust_ai[i]), 3),
        })

    return consumers


def compute_fit_scores(consumers, catalog, config):
    products = catalog["products"]
    w = config["fit_weights"]

    scores = []
    for consumer in consumers:
        row = {"consumer_id": consumer["consumer_id"]}
        for product in products:
            price_fit = max(
                0.0,
                1.0 - abs(product["price"] - consumer["budget"]) / consumer["budget"],
            )
            quality_term = consumer["quality_sensitivity"] * product["quality_score"] / 100.0
            usecase_term = product["use_case_fit"].get(consumer["use_case"], 0.0)
            brand_term = consumer["brand_familiarity"].get(product["brand_name"], 0.0)

            q_ij = (
                w["price"] * consumer["price_sensitivity"] * price_fit
                + w["quality"] * quality_term
                + w["usecase"] * usecase_term
                + w["brand"] * brand_term
            )
            row[product["product_id"]] = round(q_ij, 4)
        scores.append(row)

    return scores


def validate_and_report(category, consumers, fit_scores, catalog):
    products = catalog["products"]
    product_ids = [p["product_id"] for p in products]

    print(f"\n{'=' * 60}")
    print(f"  {category.upper()} — Consumer & Fit Score Quality Report")
    print(f"{'=' * 60}")

    # --- Budget ---
    budgets = [c["budget"] for c in consumers]
    prices = [p["price"] for p in products]
    print(f"\n  Budget distribution:")
    print(f"    Mean: ${np.mean(budgets):.2f}  Median: ${np.median(budgets):.2f}")
    print(f"    P10: ${np.percentile(budgets, 10):.2f}  P90: ${np.percentile(budgets, 90):.2f}")
    print(f"    Min: ${np.min(budgets):.2f}  Max: ${np.max(budgets):.2f}")
    print(f"    Product price range: ${min(prices):.2f} – ${max(prices):.2f}")

    can_afford_cheapest = sum(1 for b in budgets if b >= min(prices) * 0.5)
    budget_above_median_price = sum(1 for b in budgets if b >= np.median(prices))
    print(f"    Budget >= 50% of cheapest product: {can_afford_cheapest}/{len(budgets)}")
    print(f"    Budget >= median product price: {budget_above_median_price}/{len(budgets)}")

    # --- Use case ---
    uc_counts = Counter(c["use_case"] for c in consumers)
    print(f"\n  Use case distribution:")
    for uc, count in sorted(uc_counts.items(), key=lambda x: -x[1]):
        print(f"    {uc}: {count} ({count / len(consumers) * 100:.1f}%)")

    # --- Brand familiarity ---
    brands = get_brands(catalog)
    inc_fams, ent_fams = [], []
    for c in consumers:
        for brand, status in brands.items():
            fam = c["brand_familiarity"].get(brand, 0)
            if status == "incumbent":
                inc_fams.append(fam)
            else:
                ent_fams.append(fam)

    print(f"\n  Brand familiarity:")
    print(f"    Incumbent brands: mean={np.mean(inc_fams):.3f}, std={np.std(inc_fams):.3f}")
    print(f"    Entrant brands:   mean={np.mean(ent_fams):.3f}, std={np.std(ent_fams):.3f}")
    fam_gap = np.mean(inc_fams) - np.mean(ent_fams)
    print(f"    Gap (inc - ent): {fam_gap:.3f}")

    # --- Sensitivity params ---
    print(f"\n  Sensitivity parameters (mean +/- std):")
    for param in [
        "price_sensitivity", "quality_sensitivity", "tech_savviness",
        "persuasion_susceptibility", "comparison_preference", "trust_in_ai",
    ]:
        vals = [c[param] for c in consumers]
        print(f"    {param}: {np.mean(vals):.3f} +/- {np.std(vals):.3f}")

    # --- Fit scores ---
    all_scores = []
    for row in fit_scores:
        for pid in product_ids:
            all_scores.append(row[pid])
    all_scores = np.array(all_scores)

    print(f"\n  Fit scores (Q_ij):")
    print(f"    Mean: {all_scores.mean():.4f}  Std: {all_scores.std():.4f}")
    print(f"    Min: {all_scores.min():.4f}  Max: {all_scores.max():.4f}")
    print(f"    P10: {np.percentile(all_scores, 10):.4f}  P90: {np.percentile(all_scores, 90):.4f}")

    # --- Best product distribution ---
    best_product_counts = Counter()
    for row in fit_scores:
        best_pid = max(product_ids, key=lambda pid: row[pid])
        best_product_counts[best_pid] += 1

    print(f"\n  Best-fit product distribution:")
    for pid, count in sorted(best_product_counts.items(), key=lambda x: -x[1]):
        brand = next(p["brand_name"] for p in products if p["product_id"] == pid)
        print(f"    {pid} ({brand}): {count} ({count / len(consumers) * 100:.1f}%)")

    n_distinct = len(best_product_counts)
    max_share = max(best_product_counts.values()) / len(consumers)
    print(f"    Distinct top products: {n_distinct}")

    # --- Per-use-case best product ---
    print(f"\n  Best product by use case:")
    for uc in sorted(uc_counts.keys()):
        uc_consumers = [row for row, c in zip(fit_scores, consumers) if c["use_case"] == uc]
        if not uc_consumers:
            continue
        uc_best = Counter()
        for row in uc_consumers:
            best_pid = max(product_ids, key=lambda pid: row[pid])
            uc_best[best_pid] += 1
        top_pid, top_count = uc_best.most_common(1)[0]
        top_brand = next(p["brand_name"] for p in products if p["product_id"] == top_pid)
        runner_up = uc_best.most_common(2)
        runner_str = ""
        if len(runner_up) > 1:
            r_pid, r_count = runner_up[1]
            r_brand = next(p["brand_name"] for p in products if p["product_id"] == r_pid)
            runner_str = f", runner-up: {r_brand} ({r_count})"
        print(f"    {uc}: {top_brand} ({top_count}/{len(uc_consumers)}){runner_str}")

    # --- Validation checks ---
    issues = []
    if max_share > 0.50:
        issues.append(f"Single product dominates {max_share * 100:.1f}% of consumers")
    if n_distinct < 3:
        issues.append(f"Only {n_distinct} distinct best products")
    if all_scores.std() < 0.05:
        issues.append(f"Fit score variation too low (std={all_scores.std():.4f})")
    if fam_gap < 0.15:
        issues.append(f"Incumbent/entrant familiarity gap too small ({fam_gap:.3f})")
    if budget_above_median_price < len(consumers) * 0.3:
        issues.append(f"Too few consumers can afford median product ({budget_above_median_price})")

    if issues:
        for issue in issues:
            print(f"\n  WARNING: {issue}")
    else:
        print(f"\n  ALL CHECKS PASSED")

    return {"n_distinct": n_distinct, "max_share": max_share, "score_std": all_scores.std(), "issues": issues}


def save_fit_scores(fit_scores, product_ids, filepath):
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["consumer_id"] + product_ids)
        writer.writeheader()
        writer.writerows(fit_scores)


def main():
    rng = np.random.default_rng(SEED)
    all_issues = []

    for category, config in CATEGORY_CONFIG.items():
        print(f"\nGenerating {category} consumers...")
        catalog = load_catalog(category)

        consumers = generate_consumers(category, config, catalog, rng)
        fit_scores = compute_fit_scores(consumers, catalog, config)
        product_ids = [p["product_id"] for p in catalog["products"]]

        consumer_path = CONSUMER_DIR / f"{category}.json"
        with open(consumer_path, "w") as f:
            json.dump(consumers, f, indent=2)
        print(f"  Saved {len(consumers)} consumers -> {consumer_path}")

        fit_path = FIT_DIR / f"{category}.csv"
        save_fit_scores(fit_scores, product_ids, fit_path)
        print(f"  Saved fit scores -> {fit_path}")

        stats = validate_and_report(category, consumers, fit_scores, catalog)
        all_issues.extend([(category, i) for i in stats["issues"]])

    print(f"\n{'=' * 60}")
    if all_issues:
        print("  ISSUES FOUND:")
        for cat, issue in all_issues:
            print(f"    [{cat}] {issue}")
    else:
        print("  ALL CATEGORIES PASSED VALIDATION")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
