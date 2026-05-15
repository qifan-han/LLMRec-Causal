"""Prompt templates for the history-shock modular audit.

Four cells: 00 (generic/generic), 10 (history/generic),
01 (generic/history), 11 (history/history).

Cells 00 and 01 share the generic selector.
Cells 10 and 11 share the history-informed selector.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Text formatters
# ---------------------------------------------------------------------------

def format_catalog(catalog: dict) -> str:
    lines = []
    for p in catalog["products"]:
        attrs = p.get("attributes", {})
        attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items())
        lines.append(
            f"- {p['product_id']} | {p['brand_name']} | ${p['price']:.2f} | "
            f"Quality: {p['quality_score']}/100 | {attr_str}\n"
            f"  Reviews: {p['review_summary']}\n"
            f"  Weakness: {p.get('weakness', 'N/A')}"
        )
    return "\n\n".join(lines)


def format_consumer(consumer: dict) -> str:
    brand_fam = consumer.get("brand_familiarity", {})
    fam_str = ", ".join(f"{b}: {v:.2f}" for b, v in brand_fam.items())
    return (
        f"Use case: {consumer['use_case']}\n"
        f"Budget: ${consumer['budget']:.2f}\n"
        f"Price sensitivity: {consumer['price_sensitivity']:.2f} (0=insensitive, 1=very sensitive)\n"
        f"Quality sensitivity: {consumer['quality_sensitivity']:.2f}\n"
        f"Brand familiarity: {fam_str}"
    )


def format_product(product: dict) -> str:
    attrs = product.get("attributes", {})
    attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items())
    return (
        f"Product ID: {product['product_id']}\n"
        f"Brand: {product['brand_name']}\n"
        f"Price: ${product['price']:.2f}\n"
        f"Quality score: {product['quality_score']}/100\n"
        f"Key attributes: {attr_str}\n"
        f"Reviews: {product['review_summary']}\n"
        f"Known weakness: {product.get('weakness', 'N/A')}"
    )


def format_product_history(product_hist: dict) -> str:
    return (
        f"Historical data for {product_hist['product_id']}:\n"
        f"  Impressions: {product_hist['impressions']}\n"
        f"  Purchases: {product_hist['purchases']}\n"
        f"  Conversion rate: {product_hist['conversion_rate']:.1%}\n"
        f"  Average satisfaction: {product_hist['avg_satisfaction']:.2f}/5\n"
        f"  Return rate: {product_hist['return_rate']:.1%}"
    )


def format_product_history_table(product_hist_df) -> str:
    lines = ["Historical purchase data for all products:"]
    lines.append(f"{'Product':<40s} {'Impr':>5s} {'Purch':>5s} {'Conv%':>6s} {'Sat':>5s} {'Ret%':>5s}")
    lines.append("-" * 70)
    for _, row in product_hist_df.iterrows():
        lines.append(
            f"{row['product_id']:<40s} {int(row['impressions']):>5d} "
            f"{int(row['purchases']):>5d} {row['conversion_rate']:>5.1%} "
            f"{row['avg_satisfaction']:>5.2f} {row['return_rate']:>4.1%}"
        )
    return "\n".join(lines)


def format_segment_history(seg_hist_df, segment_id: str) -> str:
    sub = seg_hist_df[seg_hist_df["segment_id"] == segment_id]
    if len(sub) == 0:
        return ""
    lines = [f"Purchase history for buyers in the '{segment_id}' segment:"]
    lines.append(f"{'Product':<40s} {'Impr':>5s} {'Purch':>5s} {'Conv%':>6s} {'Sat':>5s}")
    lines.append("-" * 60)
    for _, row in sub.iterrows():
        lines.append(
            f"{row['product_id']:<40s} {int(row['impressions']):>5d} "
            f"{int(row['purchases']):>5d} {row['conversion_rate']:>5.1%} "
            f"{row['avg_satisfaction']:>5.2f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Selector prompts
# ---------------------------------------------------------------------------

def build_generic_selector_prompt(catalog: dict, consumer: dict) -> tuple[str, str]:
    catalog_text = format_catalog(catalog)
    system = (
        "You are a product selection engine. You have access to the following catalog:\n\n"
        f"{catalog_text}\n\n"
        "Given the consumer profile, select the single best product.\n\n"
        "You MUST respond with valid JSON in this exact format:\n"
        '{\n'
        '  "selected_product_id": "<exact product_id from catalog>",\n'
        '  "shortlist": ["<id1>", "<id2>", "<id3>"],\n'
        '  "selection_rationale": "<25 words max>"\n'
        '}\n\n'
        "IMPORTANT: Do NOT write a recommendation. Do NOT describe the product to the\n"
        "consumer. Only select the product and explain why briefly.\n\n"
        "All product IDs must be exact values from the catalog above."
    )
    user = format_consumer(consumer)
    return system, user


def build_history_selector_prompt(catalog: dict, consumer: dict,
                                  product_hist_df, segment_hist_df,
                                  segment_id: str) -> tuple[str, str]:
    catalog_text = format_catalog(catalog)
    hist_table = format_product_history_table(product_hist_df)
    seg_table = format_segment_history(segment_hist_df, segment_id)

    system = (
        "You are a product selection engine. You have access to the following catalog:\n\n"
        f"{catalog_text}\n\n"
        "You also have access to historical purchase data from past buyers:\n\n"
        f"{hist_table}\n\n"
        f"{seg_table}\n\n"
        "Given the consumer profile and the historical data, select the single best product.\n"
        "You may use historical conversion rates, satisfaction scores, and return rates\n"
        "to inform your selection, in addition to product attributes and consumer fit.\n\n"
        "You MUST respond with valid JSON in this exact format:\n"
        '{\n'
        '  "selected_product_id": "<exact product_id from catalog>",\n'
        '  "shortlist": ["<id1>", "<id2>", "<id3>"],\n'
        '  "selection_rationale": "<25 words max>"\n'
        '}\n\n'
        "IMPORTANT: Do NOT write a recommendation. Do NOT describe the product to the\n"
        "consumer. Only select the product and explain why briefly.\n\n"
        "All product IDs must be exact values from the catalog above."
    )
    user = format_consumer(consumer)
    return system, user


# ---------------------------------------------------------------------------
# Writer prompts
# ---------------------------------------------------------------------------

def build_generic_writer_prompt(product: dict, consumer: dict) -> tuple[str, str]:
    prod_text = format_product(product)
    system = (
        "You are a product recommendation writer. A product has already been selected\n"
        "for this consumer. You cannot change the selection.\n\n"
        f"Selected product:\n{prod_text}\n\n"
        "Write a recommendation explaining why this product was selected for\n"
        "this consumer.\n"
        "Write at least 60 words and at most 90 words.\n\n"
        "Write ONLY the recommendation text. Do not wrap it in JSON or add any metadata.\n"
        "Output the recommendation text directly.\n"
        "Write at least 60 words and at most 90 words."
    )
    user = format_consumer(consumer)
    return system, user


def build_history_writer_prompt(product: dict, consumer: dict,
                                product_hist: dict | None,
                                segment_hist_row: dict | None) -> tuple[str, str]:
    prod_text = format_product(product)

    hist_context = ""
    if product_hist:
        hist_context += (
            f"\nHistorical purchase data for this product:\n"
            f"  Purchases: {product_hist['purchases']} out of {product_hist['impressions']} impressions "
            f"({product_hist['conversion_rate']:.1%} conversion rate)\n"
            f"  Average buyer satisfaction: {product_hist['avg_satisfaction']:.2f}/5\n"
            f"  Return rate: {product_hist['return_rate']:.1%}\n"
        )
    if segment_hist_row:
        hist_context += (
            f"\nAmong buyers in a similar budget segment:\n"
            f"  Purchases: {segment_hist_row['purchases']} out of {segment_hist_row['impressions']} impressions "
            f"({segment_hist_row['conversion_rate']:.1%} conversion rate)\n"
            f"  Average satisfaction: {segment_hist_row['avg_satisfaction']:.2f}/5\n"
        )

    system = (
        "You are a product recommendation writer. A product has already been selected\n"
        "for this consumer. You cannot change the selection.\n\n"
        f"Selected product:\n{prod_text}\n"
        f"{hist_context}\n"
        "Write a recommendation explaining why this product was selected for\n"
        "this consumer. You may reference the historical purchase data as evidence\n"
        "(e.g., popularity among similar buyers, satisfaction rates) but do not\n"
        "invent numbers that were not provided.\n"
        "Write at least 60 words and at most 90 words.\n\n"
        "Write ONLY the recommendation text. Do not wrap it in JSON or add any metadata.\n"
        "Output the recommendation text directly.\n"
        "Write at least 60 words and at most 90 words."
    )
    user = format_consumer(consumer)
    return system, user


# ---------------------------------------------------------------------------
# Evaluator prompt
# ---------------------------------------------------------------------------

EVALUATOR_SYSTEM = """\
You are an expert evaluator of product recommendation quality.

You will receive:
- A consumer profile (use case, budget, preferences)
- A product that was recommended (name, brand, price, attributes, reviews)
- The recommendation text that was written for this consumer

Score the recommendation text on three scales (integers 1-7):

1. fit_specificity
   1 = generic product description with no consumer-specific fit reasoning.
   4 = mentions some consumer needs but superficially.
   7 = strongly and specifically links product features to the consumer's stated needs, budget, use case, and preferences.

2. persuasive_intensity
   1 = neutral, factual, weak/no endorsement.
   4 = moderately positive with some endorsement language.
   7 = highly confident, strongly endorsing, conversion-oriented language.

3. tradeoff_disclosure
   1 = hides or minimizes caveats, limitations, or mismatch risks.
   4 = mentions one tradeoff briefly.
   7 = clearly states relevant limitations, caveats, price/performance tradeoffs, or mismatch risks.

You MUST respond with valid JSON in this exact format:
{"fit_specificity": <integer 1-7>, "persuasive_intensity": <integer 1-7>, "tradeoff_disclosure": <integer 1-7>, "rationale": "<short explanation, max 40 words>"}

Score based ONLY on what the recommendation text says relative to the consumer profile and product information. Do not infer anything beyond what is written."""


def build_evaluator_prompt(consumer: dict, product: dict, rec_text: str) -> str:
    attrs = product.get("attributes", {})
    attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items())
    brand_fam = consumer.get("brand_familiarity", {})
    brand_of_product = product.get("brand_name", "unknown")
    fam_score = brand_fam.get(brand_of_product, "unknown")

    return (
        f"=== CONSUMER PROFILE ===\n"
        f"Primary use case: {consumer['use_case']}\n"
        f"Budget: ${consumer['budget']:.2f}\n"
        f"Price sensitivity: {consumer['price_sensitivity']:.2f}\n"
        f"Quality sensitivity: {consumer['quality_sensitivity']:.2f}\n"
        f"Brand familiarity with {brand_of_product}: {fam_score}\n\n"
        f"=== RECOMMENDED PRODUCT ===\n"
        f"Product: {product['product_id']} by {product['brand_name']}\n"
        f"Price: ${product['price']:.2f}\n"
        f"Quality score: {product['quality_score']}/100\n"
        f"Attributes: {attr_str}\n"
        f"Reviews: {product['review_summary']}\n"
        f"Known weakness: {product.get('weakness', 'N/A')}\n\n"
        f"=== RECOMMENDATION TEXT TO EVALUATE ===\n"
        f"{rec_text}"
    )


# ---------------------------------------------------------------------------
# Pairwise demand prompt
# ---------------------------------------------------------------------------

PAIRWISE_DEMAND_SYSTEM = """\
You are simulating a realistic consumer choosing between two product recommendations.

You will receive a consumer profile and two recommendation packages (A and B).
Each package contains a recommended product and the recommendation text.

Based ONLY on the information provided, decide which recommendation the consumer
would prefer.

Respond with valid JSON in this exact format:
{
  "choice": "A" or "B" or "tie",
  "preference_strength": <integer 1-5>,
  "which_has_better_fit": "A" or "B" or "tie",
  "which_is_more_trustworthy": "A" or "B" or "tie",
  "which_raises_more_tradeoff_concern": "A" or "B" or "tie",
  "rationale": "<short explanation, max 30 words>"
}

Scoring guidelines:
- choice: which package the consumer would choose to act on.
- preference_strength: 1 = almost indifferent, 5 = strong preference.
- which_has_better_fit: which product better matches the consumer's needs and budget.
- which_is_more_trustworthy: which recommendation text feels more balanced and honest.
- which_raises_more_tradeoff_concern: which package has more visible risks or unaddressed limitations.

Be realistic. Consider price vs budget, use-case match, product quality, and
how convincing each recommendation is. If the packages are very similar,
a tie is acceptable."""


def build_pairwise_demand_prompt(consumer: dict,
                                 product_a: dict, text_a: str,
                                 product_b: dict, text_b: str) -> str:
    attrs_a = product_a.get("attributes", {})
    attr_str_a = ", ".join(f"{k}: {v}" for k, v in list(attrs_a.items())[:6])
    attrs_b = product_b.get("attributes", {})
    attr_str_b = ", ".join(f"{k}: {v}" for k, v in list(attrs_b.items())[:6])

    brand_fam = consumer.get("brand_familiarity", {})
    fam_str = ", ".join(f"{b}: {v:.2f}" for b, v in brand_fam.items())

    return (
        f"=== CONSUMER PROFILE ===\n"
        f"Use case: {consumer['use_case']}\n"
        f"Budget: ${consumer['budget']:.2f}\n"
        f"Price sensitivity: {consumer['price_sensitivity']:.2f}\n"
        f"Quality sensitivity: {consumer['quality_sensitivity']:.2f}\n"
        f"Brand familiarity: {fam_str}\n\n"
        f"=== PACKAGE A ===\n"
        f"Product: {product_a['product_id']} by {product_a['brand_name']}\n"
        f"Price: ${product_a['price']:.2f}\n"
        f"Quality: {product_a['quality_score']}/100\n"
        f"Key features: {attr_str_a}\n"
        f"Recommendation:\n{text_a}\n\n"
        f"=== PACKAGE B ===\n"
        f"Product: {product_b['product_id']} by {product_b['brand_name']}\n"
        f"Price: ${product_b['price']:.2f}\n"
        f"Quality: {product_b['quality_score']}/100\n"
        f"Key features: {attr_str_b}\n"
        f"Recommendation:\n{text_b}"
    )
