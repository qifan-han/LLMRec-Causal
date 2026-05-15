"""07 — Smoke run: local supply + leakage check for 6 clusters.

2 categories x 3 personas = 6 clusters, 24 recommendation packages.
Validates cell invariant, parse success, and leakage before full run.

Usage:
  python 07_smoke_run_local_supply.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_local_llm import ollama_json_call, DATA_DIR, SUPPLY_MODEL
from utils_parse import has_leakage, detect_leakage
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

MASTER_SEED = 20260515
N_SMOKE_PERSONAS = 3
CATEGORIES = ["headphones"]


def load_inputs(category: str):
    catalog_df = pd.read_csv(CAT_DIR / f"{category}_catalog.csv")
    catalog = catalog_df.to_dict("records")

    with open(PERSONA_DIR / f"{category}_personas.json") as f:
        personas = json.load(f)

    with open(HIST_DIR / f"{category}_history_qualitative.json") as f:
        qual_history = json.load(f)

    few_shot_path = EXEMPLAR_DIR / "final_few_shot_prompts.json"
    few_shot = {}
    if few_shot_path.exists():
        with open(few_shot_path) as f:
            few_shot = json.load(f)

    return catalog, catalog_df, personas, qual_history, few_shot


def get_history_summary(qual_history: list[dict], persona: dict, product_id: str) -> str:
    """Build history summary text for a given persona segment and product."""
    segment = _guess_segment(persona)
    relevant = [q for q in qual_history
                if q["product_id"] == product_id and q["segment"] == segment]
    if not relevant:
        relevant = [q for q in qual_history if q["product_id"] == product_id]
    if not relevant:
        return "Limited historical data available for this product-consumer combination."
    return relevant[0]["summary"]


def _guess_segment(persona: dict) -> str:
    """Map persona to the closest DGP segment."""
    desc = (persona.get("one_paragraph_description", "") +
            persona.get("primary_use_case", "") +
            persona.get("purchase_context", "")).lower()

    keywords = {
        "student": "budget_student",
        "budget": "budget_student",
        "commut": "commuter",
        "travel": "frequent_traveler",
        "remote": "remote_worker",
        "work from home": "remote_worker",
        "audiophile": "audiophile",
        "music quality": "audiophile",
        "gym": "gym_user",
        "workout": "gym_user",
        "fitness": "gym_user",
        "gam": "gamer",
        "casual": "casual_listener",
        "iphone": "iphone_user",
        "apple": "iphone_user",
        "android": "android_fast_charge",
        "fast charg": "android_fast_charge",
        "family": "parent_for_family",
        "parent": "parent_for_family",
        "kid": "parent_for_family",
        "child": "parent_for_family",
        "safe": "safety_conscious",
        "multi": "multi_device_household",
        "office": "office_worker",
    }

    for kw, seg in keywords.items():
        if kw in desc:
            return seg
    return "casual_listener"


def generate_cluster(category: str, persona: dict, catalog: list[dict],
                     catalog_df: pd.DataFrame, qual_history: list[dict],
                     few_shot: dict, persona_idx: int) -> list[dict]:
    """Generate 4 cells for one cluster: 2 retrievals, then 4 expressions."""
    seed_base = MASTER_SEED + persona_idx * 100

    fs_generic = few_shot.get("generic", "")
    fs_history = few_shot.get("history_aware", "")

    # Retrieval: generic
    prompt_ret_g = build_generic_retrieval_prompt(persona, catalog, fs_generic)
    ret_g, raw_ret_g = ollama_json_call(prompt_ret_g, seed=seed_base + 1)
    pid_generic = ret_g.get("selected_product_id", catalog[0]["product_id"])

    # Retrieval: history
    hist_summary_all = "\n".join(
        f"- {q['product_id']}: {q['summary']}"
        for q in qual_history[:10]
    )
    prompt_ret_h = build_history_retrieval_prompt(persona, catalog, hist_summary_all, fs_history)
    ret_h, raw_ret_h = ollama_json_call(prompt_ret_h, seed=seed_base + 2)
    pid_history = ret_h.get("selected_product_id", catalog[0]["product_id"])

    # Validate product IDs exist
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

        cells.append({
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
        })

    return cells


def check_smoke_results(rows: list[dict]):
    df = pd.DataFrame(rows)
    n_clusters = df["cluster_id"].nunique()
    n_parse_fail = df["parse_failed"].sum()
    n_leakage = 0
    leakage_details = []

    for _, row in df.iterrows():
        text = row["full_recommendation_package"]
        if has_leakage(text):
            n_leakage += 1
            matches = detect_leakage(text)
            leakage_details.append({
                "cluster": row["cluster_id"],
                "cell": row["cell"],
                "matches": [m["match"] for m in matches],
            })

    retrieval_changed = df.groupby("cluster_id")["retrieval_changed"].first().sum()

    print(f"\n--- Smoke Run Results ---")
    print(f"  Clusters: {n_clusters}")
    print(f"  Total packages: {len(df)}")
    print(f"  Parse failures: {n_parse_fail}/{len(df)}")
    print(f"  Leakage: {n_leakage}/{len(df)}")
    print(f"  Retrieval changed: {retrieval_changed}/{n_clusters}")

    if leakage_details:
        print(f"\n  Leakage details:")
        for ld in leakage_details:
            print(f"    {ld['cluster']} cell {ld['cell']}: {ld['matches'][:3]}")

    # Check cell invariant
    invariant_ok = True
    for cid, grp in df.groupby("cluster_id"):
        cells = grp.set_index("cell")
        if "00" in cells.index and "01" in cells.index:
            if cells.loc["00", "selected_product_id"] != cells.loc["01", "selected_product_id"]:
                print(f"  FAIL invariant: {cid} cells 00/01 differ in product")
                invariant_ok = False
        if "10" in cells.index and "11" in cells.index:
            if cells.loc["10", "selected_product_id"] != cells.loc["11", "selected_product_id"]:
                print(f"  FAIL invariant: {cid} cells 10/11 differ in product")
                invariant_ok = False

    print(f"  Cell invariant: {'PASSED' if invariant_ok else 'FAILED'}")

    parse_rate = 1 - n_parse_fail / len(df) if len(df) > 0 else 0
    leakage_rate = n_leakage / len(df) if len(df) > 0 else 0

    ok = (invariant_ok and parse_rate >= 0.95 and leakage_rate <= 0.05)
    print(f"\n  Smoke run: {'PASSED' if ok else 'NEEDS REVIEW'}")
    return ok


def main():
    all_rows = []

    for category in CATEGORIES:
        print(f"\n=== {category} (smoke: {N_SMOKE_PERSONAS} personas) ===")
        catalog, catalog_df, personas, qual_history, few_shot = load_inputs(category)

        smoke_personas = personas[:N_SMOKE_PERSONAS]
        for pi, persona in enumerate(smoke_personas):
            print(f"  Cluster {pi + 1}/{N_SMOKE_PERSONAS}: {persona.get('persona_id', pi)}")
            rows = generate_cluster(
                category, persona, catalog, catalog_df,
                qual_history, few_shot, pi,
            )
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_path = SUPPLY_DIR / "smoke_supply_rows.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSmoke supply saved → {out_path}")

    ok = check_smoke_results(all_rows)
    if not ok:
        print("\nReview issues before proceeding to full run.")
    return ok


if __name__ == "__main__":
    main()
