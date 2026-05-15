"""10 — GPT absolute evaluation of recommendation packages.

480 GPT calls: 120 clusters x 4 cells.
Each package rated independently on fit, purchase probability, satisfaction,
trust, clarity, persuasive intensity, tradeoff disclosure, regret risk.

Output:
  data/final_history_shock/gpt_eval/absolute_eval_rows.csv

Usage:
  python 10_gpt_absolute_eval.py [--resume]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils_openai import gpt_json_call, append_jsonl, load_jsonl, DATA_DIR
from prompts import build_gpt_absolute_eval_prompt, GPT_ABSOLUTE_EVAL_SYSTEM

SUPPLY_DIR = DATA_DIR / "local_supply"
EVAL_DIR = DATA_DIR / "gpt_eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)
CAT_DIR = DATA_DIR / "catalogs"
PERSONA_DIR = DATA_DIR / "personas"

CACHE_PATH = EVAL_DIR / "absolute_eval_cache.jsonl"
OUT_PATH = EVAL_DIR / "absolute_eval_rows.csv"


def load_supply() -> pd.DataFrame:
    clean = SUPPLY_DIR / "final_supply_rows_clean.csv"
    raw = SUPPLY_DIR / "final_supply_rows.csv"
    path = clean if clean.exists() else raw
    if not path.exists():
        sys.exit(f"Supply not found. Run 08/09 first.")
    df = pd.read_csv(path)
    if "excluded_from_main" in df.columns:
        df = df[df["excluded_from_main"] != True]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    supply = load_supply()
    print(f"Loaded {len(supply)} supply rows for evaluation")

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
        done_keys = {(r["cluster_id"], r["cell"]) for r in cached_rows}
        print(f"Resuming: {len(done_keys)} already evaluated")

    total = len(supply)
    remaining = total - len(done_keys)
    print(f"Remaining: {remaining} evaluations")

    all_rows = list(cached_rows)
    count = 0
    t0 = time.time()

    for _, row in supply.iterrows():
        key = (row["cluster_id"], row["cell"])
        if key in done_keys:
            continue

        count += 1
        category = row["category"]
        persona_id = row["persona_id"]
        pid = row["selected_product_id"]

        persona = personas_map.get(category, {}).get(persona_id, {})
        product = next(
            (p for p in catalogs.get(category, []) if p["product_id"] == pid),
            {"brand": "Unknown", "price": 0},
        )

        package_text = str(row.get("full_recommendation_package", ""))
        prompt = build_gpt_absolute_eval_prompt(persona, product, package_text)

        parsed, raw = gpt_json_call(prompt, system=GPT_ABSOLUTE_EVAL_SYSTEM)

        eval_row = {
            "cluster_id": row["cluster_id"],
            "category": category,
            "persona_id": persona_id,
            "cell": row["cell"],
            "selected_product_id": pid,
            **{k: parsed.get(k) for k in [
                "fit_score_1_7", "purchase_probability_0_100",
                "expected_satisfaction_0_100", "trust_score_1_7",
                "clarity_score_1_7", "persuasive_intensity_1_7",
                "tradeoff_disclosure_1_7", "regret_risk_1_7",
            ]},
            "brief_reason": parsed.get("brief_reason", ""),
            "parse_failed": parsed.get("_parse_failed", False),
            "model": raw.get("model", ""),
            "elapsed_s": raw.get("elapsed_s", 0),
        }
        all_rows.append(eval_row)
        append_jsonl(CACHE_PATH, eval_row)

        if count % 20 == 0 or count == remaining:
            elapsed = time.time() - t0
            rate = elapsed / count
            eta = (remaining - count) * rate
            print(f"  [{count}/{remaining}] {row['cluster_id']} cell {row['cell']} "
                  f"({eta / 60:.0f}m remaining)")

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_PATH, index=False)

    print(f"\n=== Absolute Evaluation Complete ===")
    print(f"  Total evaluations: {len(df)}")
    print(f"  Parse failures: {df['parse_failed'].sum()}")
    print(f"  Saved → {OUT_PATH}")

    for col in ["fit_score_1_7", "purchase_probability_0_100", "trust_score_1_7",
                "persuasive_intensity_1_7", "tradeoff_disclosure_1_7"]:
        if col in df.columns:
            by_cell = df.groupby("cell")[col].mean()
            print(f"  {col}: {dict(by_cell.round(2))}")


if __name__ == "__main__":
    main()
