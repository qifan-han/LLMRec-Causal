"""Prompt templates for final history-shock simulation (v2 — real metadata).

Four supply prompts (generic/history retrieval, generic/history expression)
plus GPT evaluation prompts (absolute and pairwise).

v2 changes:
- Stronger differentiation between generic and history expression
- Real product metadata fields (rating_count, average_rating, popularity_tier)
- Richer history signal with review snippets and segment-level fit
- More opinionated recommender persona for history condition
"""

from __future__ import annotations

import json

ANTI_LEAKAGE_INSTRUCTION = (
    "Do not cite exact sales numbers, conversion rates, satisfaction rates, ranks, "
    "sample sizes, or percentages. You may refer qualitatively to evidence such as "
    '"often chosen by similar buyers," "strong post-purchase feedback among travelers," '
    '"mixed feedback from budget buyers," or "frequent complaints about comfort."'
)

RECOMMENDER_PERSONA_GENERIC = (
    "You are a product recommender. You only have access to the product catalog "
    "and the consumer's stated needs. You do NOT have access to sales data, review "
    "summaries, popularity rankings, or historical purchase feedback. Recommend "
    "based solely on product specifications and how they match the consumer's "
    "requirements."
)

RECOMMENDER_PERSONA_HISTORY = (
    "You are an experienced shopping recommender. You have access to product attributes, "
    "public review summaries, and internal historical purchase-feedback patterns. "
    "Your goal is not to recommend the most popular product. Your goal is to recommend "
    "the product most suitable for this specific consumer. Use historical evidence as "
    "background only. " + ANTI_LEAKAGE_INSTRUCTION
)

STYLE_INSTRUCTION_GENERIC = (
    "Write a straightforward product recommendation based on the product specs and "
    "the consumer's stated needs. Be factual and concise. Stick to what is listed in "
    "the product features. Do not speculate about popularity, user experience, or "
    "track record — you only have the spec sheet."
)

STYLE_INSTRUCTION_HISTORY = (
    "Write like an experienced shopping advisor who has seen thousands of customers "
    "buy and return products. You have deep knowledge of how products perform in the "
    "real world — which ones customers love, which ones get returned, and which ones "
    "surprise people. Be opinionated and specific. Use your historical insight to give "
    "advice that goes beyond the spec sheet: mention real-world reliability, common "
    "buyer satisfaction patterns, and how this product compares to alternatives that "
    "similar buyers have tried. Your goal is to help the consumer avoid regret."
)


def _format_catalog(catalog: list[dict]) -> str:
    lines = []
    for p in catalog:
        features = p.get("key_features", [])
        if isinstance(features, str):
            try:
                features = json.loads(features.replace("'", '"'))
            except (json.JSONDecodeError, ValueError):
                features = [features]
        feat_str = ", ".join(features[:4]) if features else "general use"
        rating = p.get("average_rating", "")
        rating_str = f" ★{rating}" if rating else ""
        lines.append(
            f"- {p['product_id']}: {p['brand']} {p.get('model_name', p.get('title', ''))} "
            f"(${p['price']}{rating_str}) — {feat_str}"
        )
    return "\n".join(lines)


def _format_catalog_with_popularity(catalog: list[dict]) -> str:
    lines = []
    for p in catalog:
        features = p.get("key_features", [])
        if isinstance(features, str):
            try:
                features = json.loads(features.replace("'", '"'))
            except (json.JSONDecodeError, ValueError):
                features = [features]
        feat_str = ", ".join(features[:4]) if features else "general use"
        rating = p.get("average_rating", "")
        pop_tier = p.get("popularity_tier", "")
        rating_count = p.get("rating_count", 0)
        pop_str = ""
        if pop_tier:
            pop_str = f" [{pop_tier}]"
        lines.append(
            f"- {p['product_id']}: {p['brand']} {p.get('model_name', p.get('title', ''))} "
            f"(${p['price']}, ★{rating}, {rating_count:,} reviews{pop_str}) — {feat_str}"
        )
    return "\n".join(lines)


def _format_persona(persona: dict) -> str:
    parts = [persona.get("one_paragraph_description", "")]
    if persona.get("budget"):
        parts.append(f"Budget: {persona['budget']}")
    if persona.get("primary_use_case"):
        parts.append(f"Primary use: {persona['primary_use_case']}")
    if persona.get("must_have_features"):
        feats = persona["must_have_features"]
        if isinstance(feats, str):
            try:
                feats = json.loads(feats.replace("'", '"'))
            except (json.JSONDecodeError, ValueError):
                feats = [feats]
        parts.append(f"Must-have: {', '.join(feats)}")
    if persona.get("features_to_avoid"):
        avoid = persona["features_to_avoid"]
        if isinstance(avoid, str):
            try:
                avoid = json.loads(avoid.replace("'", '"'))
            except (json.JSONDecodeError, ValueError):
                avoid = [avoid]
        if avoid:
            parts.append(f"Avoids: {', '.join(avoid)}")
    return "\n".join(parts)


def _format_product_detail_generic(product: dict) -> str:
    """Product detail for the GENERIC condition — specs only, no review data."""
    features = product.get("key_features", [])
    if isinstance(features, str):
        try:
            features = json.loads(features.replace("'", '"'))
        except (json.JSONDecodeError, ValueError):
            features = [features]
    best_for = product.get("best_for", [])
    if isinstance(best_for, str):
        try:
            best_for = json.loads(best_for.replace("'", '"'))
        except (json.JSONDecodeError, ValueError):
            best_for = [best_for]
    drawbacks = product.get("drawbacks", [])
    if isinstance(drawbacks, str):
        try:
            drawbacks = json.loads(drawbacks.replace("'", '"'))
        except (json.JSONDecodeError, ValueError):
            drawbacks = [drawbacks]

    return (
        f"Product: {product['brand']} {product.get('model_name', product.get('title', ''))} "
        f"(${product['price']}, {product.get('price_tier', 'unknown')} tier)\n"
        f"Features: {', '.join(features)}\n"
        f"Best for: {', '.join(best_for)}\n"
        f"Drawbacks: {', '.join(drawbacks)}"
    )


def _format_product_detail(product: dict) -> str:
    """Product detail for the HISTORY condition — includes review summary."""
    base = _format_product_detail_generic(product)
    return f"{base}\nReview: {product.get('review_summary', 'N/A')}"


def _format_product_detail_with_history(product: dict, history_summary: str) -> str:
    base = _format_product_detail(product)
    return (
        f"{base}\n\n"
        f"--- Internal Historical Intelligence (DO NOT cite numbers) ---\n"
        f"{history_summary}"
    )


# ── Retrieval prompts ──────────────────────────────────────────────

def build_generic_retrieval_prompt(persona: dict, catalog: list[dict],
                                   few_shot: str = "") -> str:
    return f"""{RECOMMENDER_PERSONA_GENERIC}

Select the single best product for this consumer based on their stated needs and the product specifications.

{STYLE_INSTRUCTION_GENERIC}

Consumer profile:
{_format_persona(persona)}

Product catalog:
{_format_catalog(catalog)}
{few_shot}
Return ONLY valid JSON:
{{
  "selected_product_id": "...",
  "retrieval_rationale_internal": "1-2 sentences on why this product fits best"
}}"""


def build_history_retrieval_prompt(persona: dict, catalog: list[dict],
                                   history_summary: str,
                                   few_shot: str = "") -> str:
    return f"""{RECOMMENDER_PERSONA_HISTORY}

Select the single best product for this consumer, informed by both specs and historical purchase patterns.

Consumer profile:
{_format_persona(persona)}

Internal historical purchase-feedback data (use as background evidence only):
{history_summary}

Product catalog (with review counts and popularity):
{_format_catalog_with_popularity(catalog)}
{few_shot}
Consider both the consumer's stated needs AND which products have historically satisfied similar buyers. If a popular, well-reviewed product fits the consumer's needs, prefer it over an obscure alternative — unless the consumer has specific requirements the popular option cannot meet.

Return ONLY valid JSON:
{{
  "selected_product_id": "...",
  "retrieval_rationale_internal": "1-2 sentences on why this product fits best",
  "history_signal_used": "none|weak|moderate|strong"
}}"""


# ── Expression prompts ─────────────────────────────────────────────

def build_generic_expression_prompt(persona: dict, product: dict,
                                    few_shot: str = "") -> str:
    return f"""You are writing a consumer-facing product recommendation.

{STYLE_INSTRUCTION_GENERIC}

Consumer profile:
{_format_persona(persona)}

Selected product:
{_format_product_detail_generic(product)}
{few_shot}
Write a recommendation based ONLY on the product specifications and the consumer's needs. Do not mention popularity, reviews, or how other buyers feel — you only have the spec sheet.

Return ONLY valid JSON:
{{
  "recommendation_text": "3-5 sentence consumer-facing recommendation",
  "tradeoff_text": "1-2 sentences on the main limitation or tradeoff",
  "persuasion_text": "1 sentence on the strongest reason to buy"
}}"""


def build_history_expression_prompt(persona: dict, product: dict,
                                    history_summary: str,
                                    few_shot: str = "") -> str:
    return f"""You are an experienced product advisor writing a recommendation backed by real-world buyer data.

{ANTI_LEAKAGE_INSTRUCTION}

{STYLE_INSTRUCTION_HISTORY}

Consumer profile:
{_format_persona(persona)}

Selected product and historical intelligence:
{_format_product_detail_with_history(product, history_summary)}
{few_shot}
Write a recommendation that naturally weaves in your knowledge of how this product performs in the real world. Reference patterns like "buyers in your situation tend to..." or "this is a proven choice for..." or "the main complaint from similar users is..." — but NEVER cite specific numbers, percentages, or rankings.

Your recommendation should feel noticeably different from a generic spec-based recommendation. Show the consumer you know this product's track record.

Return ONLY valid JSON:
{{
  "recommendation_text": "3-5 sentence consumer-facing recommendation that reflects real-world buyer experience",
  "tradeoff_text": "1-2 sentences on limitations informed by actual user complaints",
  "persuasion_text": "1 sentence on the strongest reason to buy, grounded in buyer track record",
  "history_language_used": "none|weak|moderate|strong"
}}"""


# ── GPT evaluation prompts ────────────────────────────────────────

GPT_ABSOLUTE_EVAL_SYSTEM = (
    "You are an expert consumer behavior analyst evaluating product recommendations. "
    "Rate each recommendation on how well it serves the given consumer. "
    "You do not know how the recommendation was generated. Be calibrated and use the full scale."
)


def build_gpt_absolute_eval_prompt(persona: dict, product: dict,
                                   recommendation_package: str) -> str:
    return f"""Evaluate this product recommendation for the given consumer.

Consumer profile:
{_format_persona(persona)}

Recommended product: {product['brand']} {product.get('model_name', product.get('title', ''))} (${product['price']})

Full recommendation:
{recommendation_package}

Rate on these scales. Return ONLY valid JSON:
{{
  "fit_score_1_7": <1-7, how well the product fits the consumer's needs>,
  "purchase_probability_0_100": <0-100, likelihood consumer would purchase>,
  "expected_satisfaction_0_100": <0-100, predicted post-purchase satisfaction>,
  "trust_score_1_7": <1-7, how trustworthy the recommendation feels>,
  "clarity_score_1_7": <1-7, how clear and understandable>,
  "persuasive_intensity_1_7": <1-7, how persuasive the language is>,
  "tradeoff_disclosure_1_7": <1-7, how honestly it discusses limitations>,
  "regret_risk_1_7": <1-7, risk of post-purchase regret>,
  "brief_reason": "1-2 sentences explaining your ratings"
}}"""


GPT_PAIRWISE_EVAL_SYSTEM = (
    "You are evaluating two recommendation packages for the same consumer. "
    "You do not know how the recommendations were generated. Choose which package "
    "is more likely to lead to a good consumer outcome. Consider purchase likelihood, "
    "expected post-purchase satisfaction, trust, and whether the recommendation "
    "honestly communicates tradeoffs. Use tie only if the two packages are genuinely "
    "indistinguishable."
)


def build_gpt_pairwise_eval_prompt(persona: dict,
                                   product_a: dict, package_a: str,
                                   product_b: dict, package_b: str) -> str:
    return f"""Compare these two product recommendations for the same consumer.

Consumer profile:
{_format_persona(persona)}

=== Package A ===
Product: {product_a['brand']} {product_a.get('model_name', product_a.get('title', ''))} (${product_a['price']})
{package_a}

=== Package B ===
Product: {product_b['brand']} {product_b.get('model_name', product_b.get('title', ''))} (${product_b['price']})
{package_b}

Which package would lead to a better outcome for this consumer? Return ONLY valid JSON:
{{
  "overall_winner": "A|B|tie",
  "purchase_winner": "A|B|tie",
  "satisfaction_winner": "A|B|tie",
  "trust_winner": "A|B|tie",
  "confidence_1_5": <1-5>,
  "reason": "1-2 sentences explaining your choice"
}}"""
