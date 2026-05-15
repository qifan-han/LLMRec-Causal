"""11 — GPT pairwise demand evaluation: main outcome variable.

720 GPT calls: 120 clusters x 6 pairwise comparisons.
Randomized A/B ordering. Multiple outcome dimensions.

Output:
  data/final_history_shock/gpt_eval/pairwise_eval_rows.csv

Usage:
  python 11_gpt_pairwise_eval.py [--resume]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_openai import gpt_json_call, append_jsonl, load_jsonl, DATA_DIR
from prompts import build_gpt_pairwise_eval_prompt, GPT_PAIRWISE_EVAL_SYSTEM

SUPPLY_DIR = DATA_DIR / "local_supply"
EVAL_DIR = DATA_DIR / "gpt_eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
CAT_DIR = DATA_DIR / "catalogs"
PERSONA_DIR = DATA_DIR / "personas"

CACHE_PATH = EVAL_DIR / "pairwise_eval_cache.jsonl"
OUT_PATH = EVAL_DIR / "pairwise_eval_rows.csv"

PAIRS = [
    ("00", "10"), ("00", "01"), ("00", "11"),
    ("10", "01"), ("10", "11"), ("01", "11"),
]
MASTER_SEED = 20260515


def load_supply() -> pd.DataFrame:
    clean = SUPPLY_DIR / "final_supply_rows_clean.csv"
    raw = SUPPLY_DIR / "final_supply_rows.csv"
    path = clean if clean.exists() else raw
    if not path.exists():
        sys.exit(f"Supply not found. Run 08/09 first.")
    df = pd.read_csv(path)
    df["cell"] = df["cell"].astype(str).str.zfill(2)
    if "excluded_from_main" in df.columns:
        df = df[df["excluded_from_main"] != True]
    return df


def _remap_winner(winner: str, cell_as_a: str, cell_as_b: str) -> str:
    winner = winner.strip().upper()
    if winner == "A":
        return cell_as_a
    elif winner == "B":
        return cell_as_b
    return "tie"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    supply = load_supply()
    print(f"Loaded {len(supply)} supply rows")

    catalogs = {}
    personas_map = {}
    for cat in supply["category"].unique():
        cat_df = pd.read_csv(CAT_DIR / f"{cat}_catalog.csv")
        catalogs[cat] = cat_df.to_dict("records")
        with open(PERSONA_DIR / f"{cat}_personas.json") as f:
            personas_map[cat] = {p["persona_id"]: p for p in json.load(f)}

    done_keys = set()
    cached_rows = []
    if args.resume and CACHE_PATH.exists():
        cached_rows = load_jsonl(CACHE_PATH)
        done_keys = {(r["cluster_id"], r["cell_i"], r["cell_j"]) for r in cached_rows}
        print(f"Resuming: {len(done_keys)} already evaluated")

    cluster_ids = supply["cluster_id"].unique()
    total = len(cluster_ids) * len(PAIRS)
    remaining = total - len(done_keys)
    print(f"Total pairs: {total}, remaining: {remaining}")

    all_rows = list(cached_rows)
    count = 0
    t0 = time.time()

    rng = np.random.default_rng(MASTER_SEED + 777)

    for cid in cluster_ids:
        cluster_supply = supply[supply["cluster_id"] == cid]
        cell_lookup = {row["cell"]: row for _, row in cluster_supply.iterrows()}
        category = cluster_supply.iloc[0]["category"]
        persona_id = cluster_supply.iloc[0]["persona_id"]
        persona = personas_map.get(category, {}).get(persona_id, {})

        for cell_i, cell_j in PAIRS:
            key = (cid, cell_i, cell_j)
            if key in done_keys:
                continue

            if cell_i not in cell_lookup or cell_j not in cell_lookup:
                continue

            row_i = cell_lookup[cell_i]
            row_j = cell_lookup[cell_j]

            swap = bool(rng.integers(2))
            if swap:
                cell_as_a, cell_as_b = cell_j, cell_i
                row_a, row_b = row_j, row_i
            else:
                cell_as_a, cell_as_b = cell_i, cell_j
                row_a, row_b = row_i, row_j

            product_a = next(
                (p for p in catalogs.get(category, [])
                 if p["product_id"] == row_a["selected_product_id"]),
                {"brand": "Unknown", "price": 0},
            )
            product_b = next(
                (p for p in catalogs.get(category, [])
                 if p["product_id"] == row_b["selected_product_id"]),
                {"brand": "Unknown", "price": 0},
            )

            package_a = str(row_a.get("full_recommendation_package", ""))
            package_b = str(row_b.get("full_recommendation_package", ""))

            prompt = build_gpt_pairwise_eval_prompt(
                persona, product_a, package_a, product_b, package_b
            )

            parsed, raw = gpt_json_call(prompt, system=GPT_PAIRWISE_EVAL_SYSTEM)

            eval_row = {
                "cluster_id": cid,
                "category": category,
                "persona_id": persona_id,
                "cell_i": cell_i,
                "cell_j": cell_j,
                "cell_as_A": cell_as_a,
                "cell_as_B": cell_as_b,
                "swapped": swap,
            }

            for outcome in ["overall_winner", "purchase_winner",
                            "satisfaction_winner", "trust_winner"]:
                ab_val = parsed.get(outcome, "tie")
                eval_row[f"{outcome}_ab"] = ab_val
                eval_row[f"{outcome}_cell"] = _remap_winner(ab_val, cell_as_a, cell_as_b)

            eval_row["confidence_1_5"] = parsed.get("confidence_1_5", 3)
            eval_row["reason"] = parsed.get("reason", "")
            eval_row["parse_failed"] = parsed.get("_parse_failed", False)
            eval_row["model"] = raw.get("model", "")
            eval_row["elapsed_s"] = raw.get("elapsed_s", 0)

            all_rows.append(eval_row)
            append_jsonl(CACHE_PATH, eval_row)
            count += 1

            if count % 30 == 0 or count == remaining:
                elapsed = time.time() - t0
                rate = elapsed / count
                eta = (remaining - count) * rate
                print(f"  [{count}/{remaining}] {cid} {cell_i}v{cell_j} "
                      f"({eta / 60:.0f}m remaining)")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)

    print(f"\n=== Pairwise Evaluation Complete ===")
    print(f"  Total pairs: {len(df)}")
    print(f"  Parse failures: {df['parse_failed'].sum()}")

    for outcome in ["overall_winner_cell", "purchase_winner_cell"]:
        if outcome not in df.columns:
            continue
        tie_rate = (df[outcome] == "tie").mean()
        print(f"  {outcome} tie rate: {tie_rate:.1%}")

    print(f"  Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
