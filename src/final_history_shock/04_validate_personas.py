"""04 — Validate consumer personas.

Checks diversity, completeness, and plausibility of generated personas.

Usage:
  python 04_validate_personas.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_openai import DATA_DIR

PERSONA_DIR = DATA_DIR / "personas"
CATEGORIES = ["headphones"]


def validate_category(category: str) -> list[str]:
    path = PERSONA_DIR / f"{category}_personas.json"
    if not path.exists():
        return [f"File not found: {path}"]

    with open(path) as f:
        personas = json.load(f)

    issues = []
    n = len(personas)

    if n < 55:
        issues.append(f"Only {n} personas (need at least 55)")

    tech_counts = Counter(p.get("technical_knowledge", "?") for p in personas)
    for level in ["low", "medium", "high"]:
        if tech_counts.get(level, 0) < 5:
            issues.append(f"Only {tech_counts.get(level, 0)} personas with tech_knowledge={level} (need >=5)")

    price_counts = Counter(p.get("price_sensitivity", "?") for p in personas)
    for level in ["low", "medium", "high"]:
        if price_counts.get(level, 0) < 5:
            issues.append(f"Only {price_counts.get(level, 0)} personas with price_sensitivity={level} (need >=5)")

    risk_counts = Counter(p.get("risk_aversion", "?") for p in personas)
    if len(risk_counts) < 3:
        issues.append(f"Only {len(risk_counts)} distinct risk_aversion levels")

    required = ["persona_id", "category", "budget", "primary_use_case",
                 "one_paragraph_description", "technical_knowledge"]
    for i, p in enumerate(personas):
        missing = [f for f in required if f not in p or not p[f]]
        if missing:
            issues.append(f"Persona {i} missing fields: {missing}")

    descs = [p.get("one_paragraph_description", "") for p in personas]
    for i in range(len(descs)):
        for j in range(i + 1, len(descs)):
            if descs[i] and descs[j] and descs[i] == descs[j]:
                issues.append(f"Duplicate descriptions: persona {i} and {j}")

    return issues


def main():
    all_ok = True
    for cat in CATEGORIES:
        print(f"\n--- Validating {cat} ---")
        issues = validate_category(cat)
        if issues:
            all_ok = False
            for issue in issues:
                print(f"  WARN: {issue}")
        else:
            print(f"  PASSED all checks")

        path = PERSONA_DIR / f"{cat}_personas.json"
        if path.exists():
            with open(path) as f:
                personas = json.load(f)
            tech = Counter(p.get("technical_knowledge", "?") for p in personas)
            price = Counter(p.get("price_sensitivity", "?") for p in personas)
            risk = Counter(p.get("risk_aversion", "?") for p in personas)
            print(f"  n={len(personas)}")
            print(f"  tech_knowledge: {dict(tech)}")
            print(f"  price_sensitivity: {dict(price)}")
            print(f"  risk_aversion: {dict(risk)}")

    if all_ok:
        print("\nAll persona validations PASSED.")
    else:
        print("\nSome validations have warnings. Review before proceeding.")


if __name__ == "__main__":
    main()
