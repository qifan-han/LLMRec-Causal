"""06 — Build local prompt templates with few-shot examples from GPT exemplars.

Reads selected GPT exemplars and constructs the final few-shot prompt strings
used by the local LLM (Ollama) for supply-side generation.

Output:
  data/final_history_shock/gpt_exemplars/final_few_shot_prompts.json

Usage:
  python 06_build_local_prompts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_openai import DATA_DIR, load_jsonl

EXEMPLAR_DIR = DATA_DIR / "gpt_exemplars"
CACHE_PATH = EXEMPLAR_DIR / "gpt_recommendation_exemplars.jsonl"
OUT_PATH = EXEMPLAR_DIR / "final_few_shot_prompts.json"


def select_best_exemplars(exemplars: list[dict]) -> dict:
    """Select 2 generic + 2 history-aware + 1 consumer-centric examples."""
    by_regime = {}
    for ex in exemplars:
        regime = ex.get("regime", "")
        parsed = ex.get("parsed", {})
        if parsed.get("_parse_failed"):
            continue
        by_regime.setdefault(regime, []).append(ex)

    final_cats = ["headphones"]
    selected = {}

    for regime, pool in by_regime.items():
        preferred = [e for e in pool if e.get("category") in final_cats]
        if len(preferred) < 2:
            preferred = pool
        selected[regime] = preferred[:2]

    return selected


def format_few_shot_block(exemplars: list[dict], regime: str) -> str:
    """Format exemplars into a few-shot block for local prompts."""
    if not exemplars:
        return ""

    parts = ["\n--- Example recommendations (for reference) ---"]
    for i, ex in enumerate(exemplars):
        parsed = ex.get("parsed", {})
        rec_text = parsed.get("recommendation_text", "")
        tradeoff = parsed.get("tradeoff_note", "")
        if not rec_text:
            continue

        parts.append(f"\nExample {i + 1} ({ex.get('category', 'unknown')}):")
        parts.append(f"Recommendation: {rec_text}")
        if tradeoff:
            parts.append(f"Tradeoff note: {tradeoff}")

    if regime == "history_aware":
        parts.append(
            "\nNote: These examples show how to reference historical patterns "
            "qualitatively without citing any numbers, rates, or statistics."
        )

    return "\n".join(parts)


def main():
    exemplars = load_jsonl(CACHE_PATH)
    if not exemplars:
        sys.exit(f"No exemplars found at {CACHE_PATH}. Run 02 first.")

    print(f"Loaded {len(exemplars)} exemplars")
    selected = select_best_exemplars(exemplars)

    few_shot_blocks = {}
    for regime, exs in selected.items():
        block = format_few_shot_block(exs, regime)
        few_shot_blocks[regime] = block
        print(f"  {regime}: {len(exs)} exemplars, {len(block)} chars")

    with open(OUT_PATH, "w") as f:
        json.dump(few_shot_blocks, f, indent=2)

    print(f"\nFew-shot prompt blocks saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
