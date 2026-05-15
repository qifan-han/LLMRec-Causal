"""14b — Pairwise architecture diagnostic: unified BB vs modular diagonal.

120 GPT calls: 60 clusters x 2 pairs.
  Pair 1: unified Z=0 vs modular cell (0,0)
  Pair 2: unified Z=1 vs modular cell (1,1)

Randomized A/B ordering, same pairwise protocol as 11_gpt_pairwise_eval.py.

Output:
  data/final_history_shock/unified_bb/bb_pairwise_diagnostic.csv
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

BB_DIR = DATA_DIR / "unified_bb"
MOD_DIR = DATA_DIR / "local_supply"
CAT_DIR = DATA_DIR / "catalogs"
PERSONA_DIR = DATA_DIR / "personas"

CACHE_PATH = BB_DIR / "bb_pairwise_diagnostic_cache.jsonl"
OUT_PATH = BB_DIR / "bb_pairwise_diagnostic.csv"

PAIRS = [
    (0, "00"),   # unified Z=0  vs  modular cell (0,0)
    (1, "11"),   # unified Z=1  vs  modular cell (1,1)
]

MASTER_SEED = 20260516


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    bb_path = BB_DIR / "unified_bb_supply.csv"
    if not bb_path.exists():
        sys.exit("Run 14_unified_bb_supply.py first.")
    bb = pd.read_csv(bb_path)

    mod_clean = MOD_DIR / "final_supply_rows_clean.csv"
    mod_raw = MOD_DIR / "final_supply_rows.csv"
    mod_path = mod_clean if mod_clean.exists() else mod_raw
    if not mod_path.exists():
        sys.exit("Modular supply not found. Run 08/09 first.")
    mod = pd.read_csv(mod_path)
    mod["cell"] = mod["cell"].astype(str).str.zfill(2)
    if "excluded_from_main" in mod.columns:
        mod = mod[mod["excluded_from_main"] != True]

    print(f"Loaded {len(bb)} unified BB rows, {len(mod)} modular rows")

    catalogs = {}
    personas_map = {}
    for cat in bb["category"].unique():
        cat_df = pd.read_csv(CAT_DIR / f"{cat}_catalog.csv")
        catalogs[cat] = cat_df.to_dict("records")
        with open(PERSONA_DIR / f"{cat}_personas.json") as f:
            personas_map[cat] = {p["persona_id"]: p for p in json.load(f)}

    done_keys = set()
    cached_rows = []
    if args.resume and CACHE_PATH.exists():
        cached_rows = load_jsonl(CACHE_PATH)
        done_keys = {(r["cluster_id"], r["z"], r["mod_cell"]) for r in cached_rows}
        print(f"Resuming: {len(done_keys)} already evaluated")

    cluster_ids = sorted(bb["cluster_id"].unique())
    total = len(cluster_ids) * len(PAIRS)
    remaining = total - len(done_keys)
    print(f"Total comparisons: {total}, remaining: {remaining}")

    all_rows = list(cached_rows)
    count = 0
    t0 = time.time()
    rng = np.random.default_rng(MASTER_SEED)

    for cid in cluster_ids:
        bb_cluster = bb[bb["cluster_id"] == cid]
        mod_cluster = mod[mod["cluster_id"] == cid]
        category = bb_cluster.iloc[0]["category"]
        persona_id = bb_cluster.iloc[0]["persona_id"]
        persona = personas_map.get(category, {}).get(persona_id, {})

        for z_val, mod_cell in PAIRS:
            key = (cid, z_val, mod_cell)
            if key in done_keys:
                continue

            bb_row = bb_cluster[bb_cluster["z"] == z_val]
            mod_row = mod_cluster[mod_cluster["cell"] == mod_cell]
            if bb_row.empty or mod_row.empty:
                continue
            bb_row = bb_row.iloc[0]
            mod_row = mod_row.iloc[0]

            bb_pid = bb_row["selected_product_id"]
            mod_pid = mod_row["selected_product_id"]

            product_bb = next(
                (p for p in catalogs.get(category, []) if p["product_id"] == bb_pid),
                {"brand": "Unknown", "price": 0},
            )
            product_mod = next(
                (p for p in catalogs.get(category, []) if p["product_id"] == mod_pid),
                {"brand": "Unknown", "price": 0},
            )

            package_bb = str(bb_row.get("full_recommendation_package", ""))
            package_mod = str(mod_row.get("full_recommendation_package", ""))

            swap = bool(rng.integers(2))
            if swap:
                prompt = build_gpt_pairwise_eval_prompt(
                    persona, product_mod, package_mod, product_bb, package_bb
                )
            else:
                prompt = build_gpt_pairwise_eval_prompt(
                    persona, product_bb, package_bb, product_mod, package_mod
                )

            parsed, raw = gpt_json_call(prompt, system=GPT_PAIRWISE_EVAL_SYSTEM)

            def remap(winner: str) -> str:
                w = winner.strip().upper()
                if swap:
                    if w == "A": return "modular"
                    if w == "B": return "unified"
                else:
                    if w == "A": return "unified"
                    if w == "B": return "modular"
                return "tie"

            eval_row = {
                "cluster_id": cid,
                "category": category,
                "persona_id": persona_id,
                "z": z_val,
                "mod_cell": mod_cell,
                "bb_product_id": bb_pid,
                "mod_product_id": mod_pid,
                "same_product": bb_pid == mod_pid,
                "swapped": swap,
            }

            for outcome in ["overall_winner", "purchase_winner",
                            "satisfaction_winner", "trust_winner"]:
                ab_val = parsed.get(outcome, "tie")
                eval_row[f"{outcome}_ab"] = ab_val
                eval_row[f"{outcome}"] = remap(ab_val)

            eval_row["confidence_1_5"] = parsed.get("confidence_1_5", 3)
            eval_row["reason"] = parsed.get("reason", "")
            eval_row["parse_failed"] = parsed.get("_parse_failed", False)
            eval_row["model"] = raw.get("model", "")
            eval_row["elapsed_s"] = raw.get("elapsed_s", 0)

            all_rows.append(eval_row)
            append_jsonl(CACHE_PATH, eval_row)
            count += 1

            if count % 10 == 0 or count == remaining:
                elapsed = time.time() - t0
                rate = elapsed / count
                eta = (remaining - count) * rate
                print(f"  [{count}/{remaining}] {cid} Z={z_val} vs cell {mod_cell} "
                      f"({eta / 60:.0f}m remaining)")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)

    print(f"\n=== Pairwise Architecture Diagnostic Complete ===")
    print(f"  Total comparisons: {len(df)}")
    print(f"  Parse failures: {df['parse_failed'].sum()}")

    for z_val, mod_cell in PAIRS:
        sub = df[(df["z"] == z_val) & (df["mod_cell"] == mod_cell)]
        n = len(sub)
        same_prod = sub["same_product"].sum()
        print(f"\n  Z={z_val} vs cell {mod_cell} ({n} clusters):")
        print(f"    Same product selected: {same_prod}/{n} ({same_prod/n:.1%})")
        for outcome in ["overall_winner", "purchase_winner",
                        "satisfaction_winner", "trust_winner"]:
            if outcome not in sub.columns:
                continue
            vc = sub[outcome].value_counts()
            uni = vc.get("unified", 0)
            mod_w = vc.get("modular", 0)
            tie = vc.get("tie", 0)
            print(f"    {outcome}: unified {uni}, modular {mod_w}, tie {tie}")

    print(f"\n  Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
