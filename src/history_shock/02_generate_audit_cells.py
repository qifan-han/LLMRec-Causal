"""02 — Four-cell modular audit (selector × writer).

Runs two selectors (generic, history-informed) and four writers to produce
the 2×2 factorial design.  Cells sharing a selector reuse the same
selection; only the writer differs.

  Cell 00: generic selector  → generic writer
  Cell 10: history selector  → generic writer
  Cell 01: generic selector  → history writer
  Cell 11: history selector  → history writer

Usage:
  python 02_generate_audit_cells.py --smoke          # 1 category, 2 consumers
  python 02_generate_audit_cells.py --full            # 3 categories, 10 consumers
  python 02_generate_audit_cells.py --categories headphones --n-consumers 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from utils import (
    CATEGORIES, HIST_DATA, HIST_RESULTS, SUPPLY_MODEL, SUPPLY_TEMPERATURE,
    load_catalog, load_consumers, load_fit_scores, assign_segment,
    get_product_by_id, validate_product_id,
    ollama_generate, parse_json_response, compute_seed,
    append_to_jsonl, load_jsonl_cache,
)
from prompts import (
    build_generic_selector_prompt,
    build_history_selector_prompt,
    build_generic_writer_prompt,
    build_history_writer_prompt,
)

CELLS = ["00", "10", "01", "11"]
CACHE_FILE = HIST_DATA / "raw_supply" / "audit_cache.jsonl"
OUTPUT_CSV = HIST_DATA / "audit_supply.csv"


# ---------------------------------------------------------------------------
# Selector helpers
# ---------------------------------------------------------------------------

def _run_selector(selector_type: str, catalog, consumer, segment,
                  product_hist_df, segment_hist_df,
                  seed: int, cache: dict, cache_key: str) -> dict | None:
    if cache_key in cache:
        return cache[cache_key]["parsed"]

    if selector_type == "generic":
        sys_p, usr_p = build_generic_selector_prompt(catalog, consumer)
    else:
        sys_p, usr_p = build_history_selector_prompt(
            catalog, consumer, product_hist_df, segment_hist_df, segment)

    resp = ollama_generate(sys_p, usr_p, seed=seed, json_mode=True)
    parsed = parse_json_response(resp["response"])

    if parsed and validate_product_id(parsed.get("selected_product_id", ""), catalog):
        entry = {"row_id": cache_key, "raw_response": resp["response"], "parsed": parsed}
        append_to_jsonl(CACHE_FILE, entry)
        cache[cache_key] = entry
        return parsed

    resp2 = ollama_generate(sys_p, usr_p, seed=seed + 1, json_mode=True)
    parsed2 = parse_json_response(resp2["response"])
    if parsed2 and validate_product_id(parsed2.get("selected_product_id", ""), catalog):
        entry = {"row_id": cache_key, "raw_response": resp2["response"], "parsed": parsed2}
        append_to_jsonl(CACHE_FILE, entry)
        cache[cache_key] = entry
        return parsed2

    print(f"  FAIL selector {cache_key}: {resp['response'][:120]}")
    return None


def _run_writer(writer_type: str, product, consumer,
                product_hist_row, segment_hist_row,
                seed: int, cache: dict, cache_key: str) -> str | None:
    if cache_key in cache:
        return cache[cache_key]["parsed"]

    if writer_type == "generic":
        sys_p, usr_p = build_generic_writer_prompt(product, consumer)
    else:
        sys_p, usr_p = build_history_writer_prompt(
            product, consumer, product_hist_row, segment_hist_row)

    resp = ollama_generate(sys_p, usr_p, seed=seed, json_mode=False, num_predict=256)
    text = resp["response"].strip()
    wc = len(text.split())

    if wc < 30:
        resp = ollama_generate(sys_p, usr_p, seed=seed + 1, json_mode=False, num_predict=256)
        text = resp["response"].strip()
        wc = len(text.split())

    entry = {"row_id": cache_key, "raw_response": text, "parsed": text}
    append_to_jsonl(CACHE_FILE, entry)
    cache[cache_key] = entry
    return text


# ---------------------------------------------------------------------------
# Per-consumer processing
# ---------------------------------------------------------------------------

def process_consumer(category: str, consumer: dict, catalog: dict,
                     product_hist_df: pd.DataFrame,
                     segment_hist_df: pd.DataFrame,
                     all_consumers: list[dict],
                     cache: dict) -> list[dict]:
    cid = consumer["consumer_id"]
    segment = assign_segment(consumer, category, all_consumers)
    results = []

    # --- Run selectors (one per type, shared across cells) ---
    generic_sel = _run_selector(
        "generic", catalog, consumer, segment,
        product_hist_df, segment_hist_df,
        seed=compute_seed(cid, "00", "selector"),
        cache=cache,
        cache_key=f"{category}_{cid:03d}_generic_selector",
    )
    history_sel = _run_selector(
        "history", catalog, consumer, segment,
        product_hist_df, segment_hist_df,
        seed=compute_seed(cid, "10", "selector"),
        cache=cache,
        cache_key=f"{category}_{cid:03d}_history_selector",
    )

    if not generic_sel or not history_sel:
        return results

    pid_generic = generic_sel["selected_product_id"]
    pid_history = history_sel["selected_product_id"]
    prod_generic = get_product_by_id(pid_generic, catalog)
    prod_history = get_product_by_id(pid_history, catalog)

    if not prod_generic or not prod_history:
        print(f"  SKIP consumer {cid}: product lookup failed")
        return results

    # --- History lookup for writers ---
    def _hist_row(pid):
        rows = product_hist_df[product_hist_df["product_id"] == pid]
        return rows.to_dict("records")[0] if len(rows) > 0 else None

    def _seg_row(pid):
        rows = segment_hist_df[
            (segment_hist_df["segment_id"] == segment)
            & (segment_hist_df["product_id"] == pid)
        ]
        return rows.to_dict("records")[0] if len(rows) > 0 else None

    # --- Run 4 writers ---
    cell_config = {
        "00": ("generic", prod_generic, pid_generic),
        "10": ("generic", prod_history, pid_history),
        "01": ("history", prod_generic, pid_generic),
        "11": ("history", prod_history, pid_history),
    }

    for cell, (writer_type, product, pid) in cell_config.items():
        ph_row = _hist_row(pid) if writer_type == "history" else None
        sh_row = _seg_row(pid) if writer_type == "history" else None

        text = _run_writer(
            writer_type, product, consumer, ph_row, sh_row,
            seed=compute_seed(cid, cell, "writer"),
            cache=cache,
            cache_key=f"{category}_{cid:03d}_{cell}_writer",
        )
        if text is None:
            continue

        selector_type = "generic" if cell[0] == "0" else "history"
        sel_data = generic_sel if selector_type == "generic" else history_sel

        results.append({
            "row_id": f"{category}_{cid:03d}_{cell}",
            "category": category,
            "consumer_id": cid,
            "cell": cell,
            "selector_type": selector_type,
            "writer_type": writer_type,
            "selected_product_id": pid,
            "shortlist": json.dumps(sel_data.get("shortlist", [])),
            "selection_rationale": sel_data.get("selection_rationale", ""),
            "recommendation_text": text,
            "word_count": len(text.split()),
            "retrieval_changed": int(pid_generic != pid_history),
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="1 category (headphones), 2 consumers")
    parser.add_argument("--full", action="store_true",
                        help="3 categories, 10 consumers each")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--n-consumers", type=int, default=None)
    args = parser.parse_args()

    if args.smoke:
        categories = ["headphones"]
        n_consumers = 2
    elif args.full:
        categories = CATEGORIES
        n_consumers = 10
    else:
        categories = args.categories or CATEGORIES
        n_consumers = args.n_consumers or 10

    cache = load_jsonl_cache(CACHE_FILE)
    all_results = []

    for cat in categories:
        print(f"\n{'='*60}")
        print(f"  Category: {cat}  |  Consumers: {n_consumers}")
        print(f"{'='*60}")

        catalog = load_catalog(cat)
        consumers = load_consumers(cat)
        product_hist_df = pd.read_csv(HIST_DATA / f"{cat}_product_history.csv")
        segment_hist_df = pd.read_csv(HIST_DATA / f"{cat}_segment_history.csv")

        audit_consumers = consumers[:n_consumers]

        for consumer in audit_consumers:
            cid = consumer["consumer_id"]
            print(f"\n  Consumer {cid}:")
            rows = process_consumer(
                cat, consumer, catalog,
                product_hist_df, segment_hist_df,
                consumers, cache,
            )
            for r in rows:
                print(f"    Cell {r['cell']}: {r['selected_product_id']} "
                      f"({r['word_count']} words)")
            all_results.extend(rows)

    df = pd.DataFrame(all_results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows → {OUTPUT_CSV}")

    n_changed = df["retrieval_changed"].sum() // 4 if len(df) > 0 else 0
    n_total = len(df) // 4 if len(df) > 0 else 0
    print(f"Retrieval changed in {n_changed}/{n_total} consumer-category clusters")


if __name__ == "__main__":
    main()
