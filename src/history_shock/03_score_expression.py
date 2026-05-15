"""03 — Score recommendation expression quality (PI / TD / fit_specificity).

Runs the validated evaluator prompt on every row in audit_supply.csv.

Usage:
  python 03_score_expression.py                      # score all rows
  python 03_score_expression.py --input custom.csv   # score a custom CSV
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from utils import (
    HIST_DATA, HIST_RESULTS, SUPPLY_MODEL, SUPPLY_TEMPERATURE,
    load_catalog, load_consumers,
    get_product_by_id, ollama_generate, parse_json_response,
    append_to_jsonl, load_jsonl_cache,
)
from prompts import EVALUATOR_SYSTEM, build_evaluator_prompt

INPUT_CSV = HIST_DATA / "audit_supply.csv"
CACHE_FILE = HIST_DATA / "raw_supply" / "eval_cache.jsonl"
OUTPUT_CSV = HIST_DATA / "evaluator_scores.csv"
EVAL_SEED = 99999


def _parse_eval(text: str) -> dict | None:
    d = parse_json_response(text)
    if d is None:
        return None
    for key in ("fit_specificity", "persuasive_intensity", "tradeoff_disclosure"):
        val = d.get(key)
        if not isinstance(val, (int, float)) or not (1 <= val <= 7):
            return None
        d[key] = int(val)
    d.setdefault("rationale", "")
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args()

    input_csv = Path(args.input) if args.input else INPUT_CSV
    if not input_csv.exists():
        sys.exit(f"Input not found: {input_csv}\nRun 02_generate_audit_cells.py first.")

    df = pd.read_csv(input_csv)
    df["cell"] = df["cell"].astype(str).str.zfill(2)
    cache = load_jsonl_cache(CACHE_FILE)

    catalogs, consumers_lookup = {}, {}
    for cat in df["category"].unique():
        catalogs[cat] = load_catalog(cat)
        cons = load_consumers(cat)
        consumers_lookup[cat] = {c["consumer_id"]: c for c in cons}

    results = []
    n_cached, n_called, n_failed = 0, 0, 0

    for _, row in df.iterrows():
        row_id = row["row_id"]

        if row_id in cache:
            p = cache[row_id]["parsed"]
            results.append({
                "row_id": row_id,
                "persuasive_intensity": p["persuasive_intensity"],
                "tradeoff_disclosure": p["tradeoff_disclosure"],
                "fit_specificity": p["fit_specificity"],
                "eval_rationale": p.get("rationale", ""),
            })
            n_cached += 1
            continue

        cat = row["category"]
        cid = int(row["consumer_id"])
        pid = row["selected_product_id"]

        consumer = consumers_lookup.get(cat, {}).get(cid)
        product = get_product_by_id(pid, catalogs.get(cat, {"products": []}))
        if not consumer or not product:
            print(f"  SKIP {row_id}: missing consumer or product")
            n_failed += 1
            continue

        prompt_text = build_evaluator_prompt(consumer, product, row["recommendation_text"])
        resp = ollama_generate(EVALUATOR_SYSTEM, prompt_text, seed=EVAL_SEED, json_mode=True)
        parsed = _parse_eval(resp["response"])

        if parsed is None:
            resp = ollama_generate(EVALUATOR_SYSTEM, prompt_text, seed=EVAL_SEED + 1, json_mode=True)
            parsed = _parse_eval(resp["response"])

        entry = {"row_id": row_id, "raw_response": resp["response"],
                 "parsed": parsed if parsed else None}
        append_to_jsonl(CACHE_FILE, entry)

        if parsed:
            results.append({
                "row_id": row_id,
                "persuasive_intensity": parsed["persuasive_intensity"],
                "tradeoff_disclosure": parsed["tradeoff_disclosure"],
                "fit_specificity": parsed["fit_specificity"],
                "eval_rationale": parsed.get("rationale", ""),
            })
            n_called += 1
            print(f"  [OK] {row_id}: PI={parsed['persuasive_intensity']} "
                  f"TD={parsed['tradeoff_disclosure']} FS={parsed['fit_specificity']}")
        else:
            print(f"  [FAIL] {row_id}")
            n_failed += 1

    out_df = pd.DataFrame(results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out_df)} rows → {OUTPUT_CSV}")
    print(f"  cached={n_cached}, new={n_called}, failed={n_failed}")


if __name__ == "__main__":
    main()
