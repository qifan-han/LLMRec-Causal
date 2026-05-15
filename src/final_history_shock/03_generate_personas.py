"""03 — Generate consumer personas via GPT.

12 GPT calls total (6 per category, 10 personas per call).
120 personas total: 60 headphones + 60 phone chargers.

Output:
  data/final_history_shock/personas/headphones_personas.json
  data/final_history_shock/personas/phone_chargers_personas.json
  data/final_history_shock/personas/all_personas.csv

Usage:
  python 03_generate_personas.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_openai import gpt_json_call, DATA_DIR

PERSONA_DIR = DATA_DIR / "personas"
PERSONA_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["headphones"]
PERSONAS_PER_CATEGORY = 60
BATCH_SIZE = 10

PERSONA_PROMPT_TEMPLATE = """Generate exactly {n} diverse, realistic consumer personas who might buy {category}.

Batch {batch_num} of {total_batches}. These personas should be DIFFERENT from typical tech-savvy reviewer profiles.

Requirements:
- Vary across: budget level, technical knowledge, age range, purchase urgency, use case, brand awareness, risk tolerance
- Include non-obvious consumer types: parents buying for kids, gift buyers, people replacing broken devices, first-time buyers, elderly, office managers, students
- Each persona must have genuine needs that make product choice nontrivial
- No two personas in this batch should have the same primary use case + budget combination
- Budget values should be realistic dollar ranges for {category}

Return a JSON array of {n} personas, each matching this schema:
{{
  "persona_id": "{category}_{start_id:03d}",
  "category": "{category}",
  "age_range": "18-24|25-34|35-44|45-54|55-64|65+",
  "purchase_context": "Why they're buying now, in one sentence",
  "budget": "$X-$Y realistic range",
  "technical_knowledge": "low|medium|high",
  "primary_use_case": "Main intended use",
  "secondary_use_case": "Secondary use or none",
  "brand_preference": "Specific brand or 'no preference' or 'avoids X'",
  "price_sensitivity": "low|medium|high",
  "quality_sensitivity": "low|medium|high",
  "risk_aversion": "low|medium|high",
  "must_have_features": ["feature1", "feature2"],
  "features_to_avoid": ["feature1"],
  "prior_experience": "Brief note on past purchases or experience level",
  "one_paragraph_description": "2-3 sentence realistic consumer description"
}}

Return ONLY a valid JSON array. No markdown, no extra text."""


def generate_personas(category: str) -> list[dict]:
    n_batches = PERSONAS_PER_CATEGORY // BATCH_SIZE
    all_personas = []

    for batch in range(n_batches):
        start_id = batch * BATCH_SIZE + 1
        prompt = PERSONA_PROMPT_TEMPLATE.format(
            n=BATCH_SIZE,
            category=category.replace("_", " "),
            batch_num=batch + 1,
            total_batches=n_batches,
            start_id=start_id,
        )

        print(f"  {category} batch {batch + 1}/{n_batches} "
              f"(personas {start_id}-{start_id + BATCH_SIZE - 1})...", end=" ", flush=True)

        parsed, raw = gpt_json_call(prompt)

        if isinstance(parsed, dict) and "_parse_failed" in parsed:
            print("PARSE FAIL — retrying with simpler prompt")
            parsed, raw = gpt_json_call(f"Return ONLY valid JSON array.\n\n{prompt}")

        personas = parsed if isinstance(parsed, list) else []

        for i, p in enumerate(personas):
            p["persona_id"] = f"{category}_{start_id + i:03d}"
            p["category"] = category

        all_personas.extend(personas)
        print(f"got {len(personas)} personas")

    return all_personas


def main():
    all_rows = []

    for category in CATEGORIES:
        print(f"\nGenerating {PERSONAS_PER_CATEGORY} personas for {category}...")
        personas = generate_personas(category)

        out_path = PERSONA_DIR / f"{category}_personas.json"
        with open(out_path, "w") as f:
            json.dump(personas, f, indent=2)
        print(f"  Saved {len(personas)} personas → {out_path}")

        all_rows.extend(personas)

    df = pd.json_normalize(all_rows)
    df.to_csv(PERSONA_DIR / "all_personas.csv", index=False)
    print(f"\nTotal: {len(all_rows)} personas saved")

    for cat in CATEGORIES:
        cat_personas = [p for p in all_rows if p["category"] == cat]
        tech_levels = set(p.get("technical_knowledge", "?") for p in cat_personas)
        price_levels = set(p.get("price_sensitivity", "?") for p in cat_personas)
        print(f"  {cat}: {len(cat_personas)} personas, "
              f"tech levels: {tech_levels}, price sensitivity: {price_levels}")


if __name__ == "__main__":
    main()
