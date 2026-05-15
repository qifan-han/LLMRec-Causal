"""14 — Unified black-box supply generation for architecture gap (Gamma).

Generates unified (single-call) LLM recommendations for Z=0 (feature-only)
and Z=1 (history-aware) conditions using the same 60 headphones personas
and catalog from the modular experiment.

This produces the mu^BB_0 and mu^BB_1 estimates needed to compute:
  Gamma = tau^BB - tau^MOD

Output:
  data/final_history_shock/unified_bb/unified_bb_supply.csv

Usage:
  python 14_unified_bb_supply.py [--resume]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_local_llm import ollama_json_call, append_jsonl, load_jsonl, DATA_DIR, SUPPLY_MODEL
from utils_parse import has_leakage
from prompts import (
    RECOMMENDER_PERSONA_GENERIC, RECOMMENDER_PERSONA_HISTORY,
    STYLE_INSTRUCTION_GENERIC, STYLE_INSTRUCTION_HISTORY,
    ANTI_LEAKAGE_INSTRUCTION,
    _format_catalog, _format_catalog_with_popularity,
    _format_persona, _format_product_detail_generic, _format_product_detail,
)

CAT_DIR = DATA_DIR / "catalogs"
PERSONA_DIR = DATA_DIR / "personas"
HIST_DIR = DATA_DIR / "history_dgp"
EXEMPLAR_DIR = DATA_DIR / "gpt_exemplars"
OUT_DIR = DATA_DIR / "unified_bb"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = OUT_DIR / "unified_bb_cache.jsonl"
OUT_PATH = OUT_DIR / "unified_bb_supply.csv"

MASTER_SEED = 20260515
CATEGORIES = ["headphones"]


def _guess_segment(persona: dict) -> str:
    desc = (persona.get("one_paragraph_description", "") +
            persona.get("primary_use_case", "") +
            persona.get("purchase_context", "")).lower()
    keywords = {
        "student": "budget_student", "budget": "budget_student",
        "commut": "commuter", "travel": "frequent_traveler",
        "remote": "remote_worker", "work from home": "remote_worker",
        "audiophile": "audiophile", "music quality": "audiophile",
        "gym": "gym_user", "workout": "gym_user", "fitness": "gym_user",
        "gam": "gamer", "casual": "casual_listener",
        "iphone": "casual_listener", "apple": "casual_listener",
        "android": "casual_listener", "fast charg": "casual_listener",
        "family": "casual_listener", "parent": "casual_listener",
        "kid": "casual_listener", "child": "casual_listener",
        "safe": "casual_listener", "multi": "casual_listener",
        "office": "remote_worker",
    }
    for kw, seg in keywords.items():
        if kw in desc:
            return seg
    return "casual_listener"


def get_history_summary(qual_history: list[dict], persona: dict, product_id: str) -> str:
    segment = _guess_segment(persona)
    relevant = [q for q in qual_history
                if q["product_id"] == product_id and q["segment"] == segment]
    if not relevant:
        relevant = [q for q in qual_history if q["product_id"] == product_id]
    if not relevant:
        return "Limited historical data available for this product-consumer combination."
    return relevant[0]["summary"]


def build_unified_generic_prompt(persona: dict, catalog: list[dict],
                                  few_shot: str = "") -> str:
    return f"""{RECOMMENDER_PERSONA_GENERIC}

Recommend the single best product from the catalog for this consumer. Select the product AND write a consumer-facing recommendation in one response.

{STYLE_INSTRUCTION_GENERIC}

Consumer profile:
{_format_persona(persona)}

Product catalog:
{_format_catalog(catalog)}
{few_shot}
Return ONLY valid JSON:
{{
  "selected_product_id": "...",
  "recommendation_text": "3-5 sentence consumer-facing recommendation based on specs",
  "tradeoff_text": "1-2 sentences on the main limitation or tradeoff",
  "persuasion_text": "1 sentence on the strongest reason to buy",
  "selection_rationale": "1-2 sentences on why this product fits best"
}}"""


def build_unified_history_prompt(persona: dict, catalog: list[dict],
                                  history_summary_all: str,
                                  few_shot: str = "") -> str:
    return f"""{RECOMMENDER_PERSONA_HISTORY}

Recommend the single best product from the catalog for this consumer. Select the product AND write a consumer-facing recommendation in one response. Use historical evidence to inform both your selection and your writing.

{STYLE_INSTRUCTION_HISTORY}

Consumer profile:
{_format_persona(persona)}

Internal historical purchase-feedback data (use as background evidence only):
{history_summary_all}

Product catalog (with review counts and popularity):
{_format_catalog_with_popularity(catalog)}
{few_shot}
Consider both the consumer's stated needs AND which products have historically satisfied similar buyers. Write like an experienced advisor who has seen real outcomes.

{ANTI_LEAKAGE_INSTRUCTION}

Return ONLY valid JSON:
{{
  "selected_product_id": "...",
  "recommendation_text": "3-5 sentence recommendation reflecting real-world buyer experience",
  "tradeoff_text": "1-2 sentences on limitations informed by actual user complaints",
  "persuasion_text": "1 sentence on the strongest reason to buy, grounded in buyer track record",
  "selection_rationale": "1-2 sentences on why this product fits best",
  "history_language_used": "none|weak|moderate|strong"
}}"""


def load_inputs(category: str):
    catalog_df = pd.read_csv(CAT_DIR / f"{category}_catalog.csv")
    catalog = catalog_df.to_dict("records")
    with open(PERSONA_DIR / f"{category}_personas.json") as f:
        personas = json.load(f)
    with open(HIST_DIR / f"{category}_history_qualitative.json") as f:
        qual_history = json.load(f)
    few_shot = {}
    fs_path = EXEMPLAR_DIR / "final_few_shot_prompts.json"
    if fs_path.exists():
        with open(fs_path) as f:
            few_shot = json.load(f)
    return catalog, catalog_df, personas, qual_history, few_shot


def generate_unified_pair(category: str, persona: dict, catalog: list[dict],
                          qual_history: list[dict], few_shot: dict,
                          persona_idx: int) -> list[dict]:
    seed_base = MASTER_SEED + persona_idx * 100
    fs_generic = few_shot.get("generic", "")
    fs_history = few_shot.get("history_aware", "")

    # Z=0: unified feature-only
    prompt_z0 = build_unified_generic_prompt(persona, catalog, fs_generic)
    parsed_z0, raw_z0 = ollama_json_call(prompt_z0, seed=seed_base + 50)
    pid_z0 = parsed_z0.get("selected_product_id", catalog[0]["product_id"])

    valid_pids = set(p["product_id"] for p in catalog)
    if pid_z0 not in valid_pids:
        pid_z0 = catalog[0]["product_id"]

    rec_z0 = parsed_z0.get("recommendation_text", "")
    tradeoff_z0 = parsed_z0.get("tradeoff_text", "")
    full_z0 = f"{rec_z0}\n\n{tradeoff_z0}" if tradeoff_z0 else rec_z0

    # Z=1: unified history-aware
    segment = _guess_segment(persona)
    segment_history = [q for q in qual_history if q["segment"] == segment]
    if not segment_history:
        segment_history = qual_history
    segment_history.sort(key=lambda q: q.get("affinity_score", 0), reverse=True)
    hist_summary_all = "\n".join(
        f"- {q['product_id']}: {q['summary']}" for q in segment_history
    )

    prompt_z1 = build_unified_history_prompt(persona, catalog, hist_summary_all, fs_history)
    parsed_z1, raw_z1 = ollama_json_call(prompt_z1, seed=seed_base + 60)
    pid_z1 = parsed_z1.get("selected_product_id", catalog[0]["product_id"])

    if pid_z1 not in valid_pids:
        pid_z1 = catalog[0]["product_id"]

    rec_z1 = parsed_z1.get("recommendation_text", "")
    tradeoff_z1 = parsed_z1.get("tradeoff_text", "")
    full_z1 = f"{rec_z1}\n\n{tradeoff_z1}" if tradeoff_z1 else rec_z1

    cluster_id = f"{category}_{persona_idx:03d}"
    rows = []
    for z_val, pid, parsed, full, rec, tradeoff in [
        (0, pid_z0, parsed_z0, full_z0, rec_z0, tradeoff_z0),
        (1, pid_z1, parsed_z1, full_z1, rec_z1, tradeoff_z1),
    ]:
        row = {
            "cluster_id": cluster_id,
            "category": category,
            "persona_id": persona.get("persona_id", f"{category}_{persona_idx:03d}"),
            "z": z_val,
            "architecture": "unified",
            "selected_product_id": pid,
            "recommendation_text": rec,
            "tradeoff_text": tradeoff,
            "persuasion_text": parsed.get("persuasion_text", ""),
            "full_recommendation_package": full,
            "history_language_used": parsed.get("history_language_used", "none"),
            "local_model": SUPPLY_MODEL,
            "word_count": len(full.split()),
            "parse_failed": parsed.get("_parse_failed", False),
            "leakage_flag": has_leakage(full),
        }
        rows.append(row)
        append_jsonl(CACHE_PATH, row)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    done_clusters = set()
    all_rows = []
    if args.resume and CACHE_PATH.exists():
        all_rows = load_jsonl(CACHE_PATH)
        done_clusters = {r["cluster_id"] for r in all_rows}
        print(f"Resuming: {len(done_clusters)} clusters already done")

    total_clusters = 0
    for category in CATEGORIES:
        with open(PERSONA_DIR / f"{category}_personas.json") as f:
            total_clusters += len(json.load(f))

    done_count = len(done_clusters)
    t0 = time.time()

    for category in CATEGORIES:
        catalog, catalog_df, personas, qual_history, few_shot = load_inputs(category)
        print(f"\n=== Unified BB: {category} ({len(personas)} personas) ===")
        print(f"  Model: {SUPPLY_MODEL}, 2 conditions per persona (Z=0, Z=1)")

        for pi, persona in enumerate(personas):
            cid = f"{category}_{pi:03d}"
            if cid in done_clusters:
                continue

            done_count += 1
            elapsed = time.time() - t0
            rate = elapsed / done_count if done_count > 0 else 0
            remaining = (total_clusters - done_count) * rate

            print(f"  [{done_count}/{total_clusters}] {cid} "
                  f"(~{remaining / 60:.0f}m remaining)", flush=True)

            rows = generate_unified_pair(
                category, persona, catalog, qual_history, few_shot, pi
            )
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)

    n_leakage = df["leakage_flag"].sum() if "leakage_flag" in df.columns else 0
    n_parse_fail = df["parse_failed"].sum() if "parse_failed" in df.columns else 0
    retrieval_diff = (df[df["z"] == 0].set_index("cluster_id")["selected_product_id"]
                      != df[df["z"] == 1].set_index("cluster_id")["selected_product_id"]).sum()

    print(f"\n=== Unified BB Supply Complete ===")
    print(f"  Total rows: {len(df)} ({len(df)//2} clusters x 2 conditions)")
    print(f"  Parse failures: {n_parse_fail}")
    print(f"  Leakage flagged: {n_leakage}")
    print(f"  Product changed (Z=0 vs Z=1): {retrieval_diff}/{len(df)//2}")
    print(f"  Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
