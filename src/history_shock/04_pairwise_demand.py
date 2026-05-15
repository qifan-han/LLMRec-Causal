"""04 — Pairwise demand-side LLM comparisons.

For each consumer-category cluster, runs 6 pairwise comparisons among the
4 audit cells.  A/B ordering is randomized per pair to control for
position bias.  The demand model sees only consumer profile + product +
recommendation text — no cell labels, no DGP parameters.

Usage:
  python 04_pairwise_demand.py                       # all rows in audit_supply.csv
  python 04_pairwise_demand.py --input custom.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    HIST_DATA, DEMAND_MODEL, DEMAND_TEMPERATURE, MASTER_SEED,
    load_catalog, load_consumers, get_product_by_id,
    ollama_generate, parse_json_response,
    append_to_jsonl, load_jsonl_cache,
)
from prompts import PAIRWISE_DEMAND_SYSTEM, build_pairwise_demand_prompt

INPUT_CSV = HIST_DATA / "audit_supply.csv"
CACHE_FILE = HIST_DATA / "pairwise_demand_raw" / "demand_cache.jsonl"
OUTPUT_CSV = HIST_DATA / "pairwise_demand.csv"
DEMAND_SEED = 77777

PAIRS = list(itertools.combinations(["00", "10", "01", "11"], 2))


def _parse_pairwise(text: str) -> dict | None:
    d = parse_json_response(text)
    if d is None:
        return None
    if d.get("choice") not in ("A", "B", "tie"):
        return None
    ps = d.get("preference_strength")
    if not isinstance(ps, (int, float)) or not (1 <= ps <= 5):
        return None
    d["preference_strength"] = int(round(ps))
    for key in ("which_has_better_fit", "which_is_more_trustworthy",
                "which_raises_more_tradeoff_concern"):
        if d.get(key) not in ("A", "B", "tie"):
            d[key] = "tie"
    d.setdefault("rationale", "")
    return d


def _remap_choice(val: str, cell_as_a: str, cell_as_b: str) -> str:
    if val == "A":
        return cell_as_a
    if val == "B":
        return cell_as_b
    return "tie"


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

    catalogs, consumers_map = {}, {}
    for cat in df["category"].unique():
        catalogs[cat] = load_catalog(cat)
        cons = load_consumers(cat)
        consumers_map[cat] = {c["consumer_id"]: c for c in cons}

    cell_lookup = {}
    for _, row in df.iterrows():
        cell_lookup[(row["category"], int(row["consumer_id"]), row["cell"])] = row

    rng = np.random.default_rng(MASTER_SEED + 5000)
    results = []
    n_cached, n_called, n_failed = 0, 0, 0

    clusters = df.groupby(["category", "consumer_id"]).size().index.tolist()

    for cat, cid in clusters:
        cid = int(cid)
        consumer = consumers_map.get(cat, {}).get(cid)
        if not consumer:
            continue

        for pair_idx, (cell_i, cell_j) in enumerate(PAIRS):
            pair_id = f"{cat}_{cid:03d}_{cell_i}v{cell_j}"

            if pair_id in cache:
                p = cache[pair_id]["parsed"]
                results.append(cache[pair_id].get("result_row", p))
                n_cached += 1
                continue

            row_i = cell_lookup.get((cat, cid, cell_i))
            row_j = cell_lookup.get((cat, cid, cell_j))
            if row_i is None or row_j is None:
                n_failed += 1
                continue

            pid_i = row_i["selected_product_id"]
            pid_j = row_j["selected_product_id"]
            prod_i = get_product_by_id(pid_i, catalogs[cat])
            prod_j = get_product_by_id(pid_j, catalogs[cat])
            if not prod_i or not prod_j:
                n_failed += 1
                continue

            swap = bool(rng.integers(0, 2))
            if swap:
                cell_as_a, cell_as_b = cell_j, cell_i
                prod_a, text_a = prod_j, row_j["recommendation_text"]
                prod_b, text_b = prod_i, row_i["recommendation_text"]
            else:
                cell_as_a, cell_as_b = cell_i, cell_j
                prod_a, text_a = prod_i, row_i["recommendation_text"]
                prod_b, text_b = prod_j, row_j["recommendation_text"]

            prompt = build_pairwise_demand_prompt(
                consumer, prod_a, text_a, prod_b, text_b)

            seed = DEMAND_SEED + cid * 100 + pair_idx
            resp = ollama_generate(
                PAIRWISE_DEMAND_SYSTEM, prompt,
                seed=seed, json_mode=True,
                model=DEMAND_MODEL, temperature=DEMAND_TEMPERATURE,
            )
            parsed = _parse_pairwise(resp["response"])

            if parsed is None:
                resp = ollama_generate(
                    PAIRWISE_DEMAND_SYSTEM, prompt,
                    seed=seed + 50, json_mode=True,
                    model=DEMAND_MODEL, temperature=DEMAND_TEMPERATURE,
                )
                parsed = _parse_pairwise(resp["response"])

            if parsed is None:
                print(f"  [FAIL] {pair_id}: {resp['response'][:100]}")
                n_failed += 1
                continue

            result_row = {
                "pair_id": pair_id,
                "category": cat,
                "consumer_id": cid,
                "cell_i": cell_i,
                "cell_j": cell_j,
                "cell_as_A": cell_as_a,
                "cell_as_B": cell_as_b,
                "swapped": int(swap),
                "raw_choice": parsed["choice"],
                "choice_winner": _remap_choice(parsed["choice"], cell_as_a, cell_as_b),
                "preference_strength": parsed["preference_strength"],
                "better_fit": _remap_choice(parsed["which_has_better_fit"], cell_as_a, cell_as_b),
                "more_trustworthy": _remap_choice(parsed["which_is_more_trustworthy"], cell_as_a, cell_as_b),
                "more_tradeoff_concern": _remap_choice(parsed["which_raises_more_tradeoff_concern"], cell_as_a, cell_as_b),
                "rationale": parsed.get("rationale", ""),
                "demand_model": DEMAND_MODEL,
            }
            results.append(result_row)

            entry = {"row_id": pair_id, "raw_response": resp["response"],
                     "parsed": parsed, "result_row": result_row}
            append_to_jsonl(CACHE_FILE, entry)
            n_called += 1

            winner = result_row["choice_winner"]
            print(f"  [OK] {pair_id}: winner={winner} strength={parsed['preference_strength']}")

    out_df = pd.DataFrame(results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out_df)} rows → {OUTPUT_CSV}")
    print(f"  cached={n_cached}, new={n_called}, failed={n_failed}")


if __name__ == "__main__":
    main()
