"""08 — Full local supply generation: 120 clusters x 4 cells = 480 packages.

Uses the same logic as the smoke run but for all 60 personas per category.

Output:
  data/final_history_shock/local_supply/final_supply_rows.csv

Usage:
  python 08_run_local_supply_full.py [--resume]
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
    build_generic_retrieval_prompt, build_history_retrieval_prompt,
    build_generic_expression_prompt, build_history_expression_prompt,
)

CAT_DIR = DATA_DIR / "catalogs"
PERSONA_DIR = DATA_DIR / "personas"
HIST_DIR = DATA_DIR / "history_dgp"
EXEMPLAR_DIR = DATA_DIR / "gpt_exemplars"
SUPPLY_DIR = DATA_DIR / "local_supply"
SUPPLY_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PATH = SUPPLY_DIR / "full_supply_cache.jsonl"
OUT_PATH = SUPPLY_DIR / "final_supply_rows.csv"

MASTER_SEED = 20260515
CATEGORIES = ["headphones"]


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


def generate_cluster(category: str, persona: dict, catalog: list[dict],
                     catalog_df: pd.DataFrame, qual_history: list[dict],
                     few_shot: dict, persona_idx: int) -> list[dict]:
    seed_base = MASTER_SEED + persona_idx * 100
    fs_generic = few_shot.get("generic", "")
    fs_history = few_shot.get("history_aware", "")

    prompt_ret_g = build_generic_retrieval_prompt(persona, catalog, fs_generic)
    ret_g, _ = ollama_json_call(prompt_ret_g, seed=seed_base + 1)
    pid_generic = ret_g.get("selected_product_id", catalog[0]["product_id"])

    segment = _guess_segment(persona)
    segment_history = [q for q in qual_history if q["segment"] == segment]
    if not segment_history:
        segment_history = qual_history
    segment_history.sort(key=lambda q: q.get("affinity_score", 0), reverse=True)
    hist_summary_all = "\n".join(
        f"- {q['product_id']}: {q['summary']}"
        for q in segment_history
    )
    prompt_ret_h = build_history_retrieval_prompt(persona, catalog, hist_summary_all, fs_history)
    ret_h, _ = ollama_json_call(prompt_ret_h, seed=seed_base + 2)
    pid_history = ret_h.get("selected_product_id", catalog[0]["product_id"])

    valid_pids = set(p["product_id"] for p in catalog)
    if pid_generic not in valid_pids:
        pid_generic = catalog[0]["product_id"]
    if pid_history not in valid_pids:
        pid_history = catalog[0]["product_id"]

    product_generic = next(p for p in catalog if p["product_id"] == pid_generic)
    product_history = next(p for p in catalog if p["product_id"] == pid_history)

    cells = []
    cell_specs = [
        ("00", pid_generic, product_generic, "generic", "generic"),
        ("10", pid_history, product_history, "history", "generic"),
        ("01", pid_generic, product_generic, "generic", "history"),
        ("11", pid_history, product_history, "history", "history"),
    ]

    for cell, pid, product, ret_cond, exp_cond in cell_specs:
        hist_prod = get_history_summary(qual_history, persona, pid)
        if exp_cond == "generic":
            prompt = build_generic_expression_prompt(persona, product, fs_generic)
        else:
            prompt = build_history_expression_prompt(persona, product, hist_prod, fs_history)

        seed_offset = {"00": 10, "10": 20, "01": 30, "11": 40}[cell]
        parsed, raw = ollama_json_call(prompt, seed=seed_base + seed_offset)

        rec_text = parsed.get("recommendation_text", "")
        tradeoff_text = parsed.get("tradeoff_text", "")
        persuasion_text = parsed.get("persuasion_text", "")
        full_package = f"{rec_text}\n\n{tradeoff_text}" if tradeoff_text else rec_text

        row = {
            "cluster_id": f"{category}_{persona_idx:03d}",
            "category": category,
            "persona_id": persona.get("persona_id", f"{category}_{persona_idx:03d}"),
            "cell": cell,
            "retrieval_condition": ret_cond,
            "expression_condition": exp_cond,
            "selected_product_id": pid,
            "recommendation_text": rec_text,
            "tradeoff_text": tradeoff_text,
            "persuasion_text": persuasion_text,
            "full_recommendation_package": full_package,
            "history_language_used": parsed.get("history_language_used", "none"),
            "local_model": SUPPLY_MODEL,
            "word_count": len(full_package.split()),
            "parse_failed": parsed.get("_parse_failed", False),
            "retrieval_changed": pid_generic != pid_history,
            "leakage_flag": has_leakage(full_package),
        }
        cells.append(row)
        append_jsonl(CACHE_PATH, row)

    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    done_clusters = set()
    if args.resume and CACHE_PATH.exists():
        cached = load_jsonl(CACHE_PATH)
        done_clusters = {r["cluster_id"] for r in cached}
        print(f"Resuming: {len(done_clusters)} clusters already done")

    all_rows = []
    if args.resume and CACHE_PATH.exists():
        all_rows = load_jsonl(CACHE_PATH)

    total_clusters = sum(
        len(json.load(open(PERSONA_DIR / f"{cat}_personas.json")))
        for cat in CATEGORIES
    )
    done_count = len(done_clusters)

    t0 = time.time()
    for category in CATEGORIES:
        catalog, catalog_df, personas, qual_history, few_shot = load_inputs(category)
        print(f"\n=== {category}: {len(personas)} personas ===")

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

            rows = generate_cluster(
                category, persona, catalog, catalog_df,
                qual_history, few_shot, pi,
            )
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)

    n_leakage = df["leakage_flag"].sum() if "leakage_flag" in df.columns else 0
    n_parse_fail = df["parse_failed"].sum() if "parse_failed" in df.columns else 0
    ret_changed = df.groupby("cluster_id")["retrieval_changed"].first().sum()

    print(f"\n=== Full Supply Complete ===")
    print(f"  Total packages: {len(df)}")
    print(f"  Clusters: {df['cluster_id'].nunique()}")
    print(f"  Parse failures: {n_parse_fail}")
    print(f"  Leakage flagged: {n_leakage} ({n_leakage / len(df):.1%})")
    print(f"  Retrieval changed: {ret_changed}/{df['cluster_id'].nunique()}")
    print(f"  Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
