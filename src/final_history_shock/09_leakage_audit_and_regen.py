"""09 — Leakage audit and regeneration.

Scans all supply rows for forbidden statistical leakage patterns.
Regenerates flagged rows with stricter anti-leakage prompts.
Target: <2% leakage after regeneration.

Usage:
  python 09_leakage_audit_and_regen.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_local_llm import ollama_json_call, DATA_DIR, SUPPLY_MODEL
from utils_parse import detect_leakage, has_leakage
from prompts import (
    build_generic_expression_prompt, build_history_expression_prompt,
    ANTI_LEAKAGE_INSTRUCTION,
)

SUPPLY_DIR = DATA_DIR / "local_supply"
SUPPLY_PATH = SUPPLY_DIR / "final_supply_rows.csv"
OUT_PATH = SUPPLY_DIR / "final_supply_rows_clean.csv"
CAT_DIR = DATA_DIR / "catalogs"
PERSONA_DIR = DATA_DIR / "personas"
HIST_DIR = DATA_DIR / "history_dgp"

MASTER_SEED = 20260515

STRICT_ANTI_LEAKAGE = (
    "CRITICAL RULE: Your response MUST NOT contain any of these:\n"
    "- Any percentage (e.g., 42%, 0.73)\n"
    "- Any conversion rate, satisfaction rate, or purchase rate\n"
    "- Any sample size (n=, out of)\n"
    "- Any star rating or numerical score\n"
    "- CTR, CVR, click-through\n"
    "- Any phrase like 'historical data shows' followed by a number\n"
    "If you feel tempted to cite a number, instead use qualitative language like "
    "'popular among similar buyers' or 'generally well-received.'\n\n"
    + ANTI_LEAKAGE_INSTRUCTION
)


def _guess_segment(persona: dict) -> str:
    desc = (persona.get("one_paragraph_description", "") +
            persona.get("primary_use_case", "") +
            persona.get("purchase_context", "")).lower()
    keywords = {
        "student": "budget_student", "budget": "budget_student",
        "commut": "commuter", "travel": "frequent_traveler",
        "remote": "remote_worker", "audiophile": "audiophile",
        "gym": "gym_user", "gam": "gamer", "casual": "casual_listener",
        "iphone": "iphone_user", "android": "android_fast_charge",
        "family": "parent_for_family", "parent": "parent_for_family",
        "safe": "safety_conscious", "multi": "multi_device_household",
        "office": "office_worker",
    }
    for kw, seg in keywords.items():
        if kw in desc:
            return seg
    return "casual_listener"


def main():
    if not SUPPLY_PATH.exists():
        sys.exit(f"Supply file not found: {SUPPLY_PATH}. Run 08 first.")

    df = pd.read_csv(SUPPLY_PATH)
    print(f"Loaded {len(df)} supply rows")

    leakage_rows = []
    for idx, row in df.iterrows():
        text = str(row.get("full_recommendation_package", ""))
        matches = detect_leakage(text)
        if matches:
            leakage_rows.append((idx, matches))

    n_initial = len(leakage_rows)
    print(f"Initial leakage: {n_initial}/{len(df)} ({n_initial / len(df):.1%})")

    if n_initial == 0:
        print("No leakage detected. Copying to clean output.")
        df.to_csv(OUT_PATH, index=False)
        return

    catalogs = {}
    personas = {}
    qual_histories = {}
    for cat in df["category"].unique():
        cat_df = pd.read_csv(CAT_DIR / f"{cat}_catalog.csv")
        catalogs[cat] = cat_df.to_dict("records")
        with open(PERSONA_DIR / f"{cat}_personas.json") as f:
            personas[cat] = json.load(f)
        with open(HIST_DIR / f"{cat}_history_qualitative.json") as f:
            qual_histories[cat] = json.load(f)

    n_fixed = 0
    n_still_leaking = 0

    for idx, matches in leakage_rows:
        row = df.iloc[idx]
        category = row["category"]
        persona_idx = int(row["persona_id"].split("_")[-1]) - 1
        if persona_idx < 0 or persona_idx >= len(personas[category]):
            persona_idx = 0
        persona = personas[category][persona_idx]
        catalog = catalogs[category]
        pid = row["selected_product_id"]
        product = next((p for p in catalog if p["product_id"] == pid), catalog[0])

        segment = _guess_segment(persona)
        qual = qual_histories[category]
        relevant = [q for q in qual if q["product_id"] == pid and q["segment"] == segment]
        if not relevant:
            relevant = [q for q in qual if q["product_id"] == pid]
        hist_summary = relevant[0]["summary"] if relevant else ""

        seed_base = MASTER_SEED + persona_idx * 100 + 50
        exp_cond = row["expression_condition"]

        if exp_cond == "history":
            prompt = build_history_expression_prompt(persona, product, hist_summary, "")
            prompt = STRICT_ANTI_LEAKAGE + "\n\n" + prompt
        else:
            prompt = build_generic_expression_prompt(persona, product, "")

        parsed, raw = ollama_json_call(prompt, seed=seed_base)
        new_rec = parsed.get("recommendation_text", "")
        new_tradeoff = parsed.get("tradeoff_text", "")
        new_full = f"{new_rec}\n\n{new_tradeoff}" if new_tradeoff else new_rec

        if has_leakage(new_full):
            n_still_leaking += 1
            df.at[idx, "leakage_flag"] = True
            df.at[idx, "excluded_from_main"] = True
        else:
            n_fixed += 1
            df.at[idx, "recommendation_text"] = new_rec
            df.at[idx, "tradeoff_text"] = new_tradeoff
            df.at[idx, "persuasion_text"] = parsed.get("persuasion_text", "")
            df.at[idx, "full_recommendation_package"] = new_full
            df.at[idx, "word_count"] = len(new_full.split())
            df.at[idx, "leakage_flag"] = False
            df.at[idx, "regenerated"] = True

    df.to_csv(OUT_PATH, index=False)

    final_leakage = df[df.get("leakage_flag", False) == True].shape[0] if "leakage_flag" in df.columns else n_still_leaking
    print(f"\n=== Leakage Audit Results ===")
    print(f"  Initial leakage: {n_initial}")
    print(f"  Fixed by regeneration: {n_fixed}")
    print(f"  Still leaking (excluded): {n_still_leaking}")
    print(f"  Final leakage rate: {final_leakage}/{len(df)} ({final_leakage / len(df):.1%})")
    print(f"  Target: <2%")
    print(f"  Saved → {OUT_PATH}")

    if final_leakage / len(df) > 0.05:
        print("\n  WARNING: Leakage rate >5%. Review prompts before proceeding.")


if __name__ == "__main__":
    main()
