"""GPT evaluation: absolute scoring + pairwise demand.

Replaces the Gemma-based vLLM demand simulator. Uses GPT API for both:
  Step 1: Absolute evaluation (fit, purchase, satisfaction, trust, PI, TD)
  Step 2: Pairwise comparison (overall, purchase, satisfaction, trust)

Requires OPENAI_API_KEY env var or .env file.

Usage:
  python eval_gpt.py --absolute      # absolute scoring only
  python eval_gpt.py --pairwise      # pairwise only
  python eval_gpt.py --all           # both (default)
  python eval_gpt.py --all --resume  # resume from cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from prompts import (
    build_gpt_absolute_eval_prompt, GPT_ABSOLUTE_EVAL_SYSTEM,
    build_gpt_pairwise_eval_prompt, GPT_PAIRWISE_EVAL_SYSTEM,
)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.3-chat-latest")

DATA_ROOT = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(os.environ.get("DATA_DIR", os.path.expanduser("~/llmrec_results")))
SUPPLY_CSV = OUT_DIR / "final_supply_rows.csv"

ABS_CACHE = OUT_DIR / "absolute_eval_cache.jsonl"
ABS_CSV = OUT_DIR / "absolute_eval_rows.csv"
PAIR_CACHE = OUT_DIR / "pairwise_eval_cache.jsonl"
PAIR_CSV = OUT_DIR / "pairwise_eval_rows.csv"

PAIRS = [
    ("00", "10"), ("00", "01"), ("00", "11"),
    ("10", "01"), ("10", "11"), ("01", "11"),
]
MASTER_SEED = 20260515


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def gpt_call(prompt: str, system: str | None = None) -> dict:
    inp = []
    if system:
        inp.append({"role": "system", "content": system})
    inp.append({"role": "user", "content": prompt})

    t0 = time.time()
    resp = client.responses.create(model=MODEL, input=inp)
    elapsed = time.time() - t0
    return {
        "output_text": resp.output_text,
        "model": MODEL,
        "elapsed_s": round(elapsed, 2),
    }


def gpt_json_call(prompt: str, system: str | None = None) -> tuple[dict, dict]:
    for attempt in range(3):
        raw = gpt_call(prompt, system=system)
        text = raw["output_text"].strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            parsed = json.loads(text)
            raw["parse_attempt"] = attempt + 1
            return parsed, raw
        except json.JSONDecodeError:
            if attempt < 2:
                prompt = (
                    f"Your previous response was not valid JSON. Respond with ONLY valid JSON.\n\n"
                    f"Original request:\n{prompt}"
                )
            else:
                raw["parse_attempt"] = attempt + 1
                return {"_raw_text": text, "_parse_failed": True}, raw
    return {"_parse_failed": True}, raw


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_supply() -> pd.DataFrame:
    if not SUPPLY_CSV.exists():
        sys.exit(f"Supply not found: {SUPPLY_CSV}\nRun supply_history_shock.py first.")
    df = pd.read_csv(SUPPLY_CSV)
    df["cell"] = df["cell"].astype(str).str.zfill(2)
    return df


def load_catalog_and_personas():
    catalogs = {}
    personas_map = {}
    for cat_file in (DATA_ROOT / "catalogs").glob("*_catalog.csv"):
        cat = cat_file.stem.replace("_catalog", "")
        cat_df = pd.read_csv(cat_file)
        catalogs[cat] = cat_df.to_dict("records")
    for pers_file in (DATA_ROOT / "personas").glob("*_personas.json"):
        cat = pers_file.stem.replace("_personas", "")
        with open(pers_file) as f:
            personas_map[cat] = {p["persona_id"]: p for p in json.load(f)}
    return catalogs, personas_map


def run_absolute(supply: pd.DataFrame, catalogs: dict, personas_map: dict,
                 resume: bool = False):
    print(f"\n=== Absolute Evaluation ({len(supply)} rows) ===")

    done_keys = set()
    cached_rows = []
    if resume and ABS_CACHE.exists():
        cached_rows = load_jsonl(ABS_CACHE)
        done_keys = {(r["cluster_id"], r["cell"]) for r in cached_rows}
        print(f"  Resuming: {len(done_keys)} already done")

    all_rows = list(cached_rows)
    remaining = len(supply) - len(done_keys)
    count = 0
    t0 = time.time()

    for _, row in supply.iterrows():
        key = (row["cluster_id"], row["cell"])
        if key in done_keys:
            continue

        count += 1
        category = row["category"]
        persona = personas_map.get(category, {}).get(row["persona_id"], {})
        product = next(
            (p for p in catalogs.get(category, []) if p["product_id"] == row["selected_product_id"]),
            {"brand": "Unknown", "price": 0},
        )

        package_text = str(row.get("full_recommendation_package", ""))
        prompt = build_gpt_absolute_eval_prompt(persona, product, package_text)
        parsed, raw = gpt_json_call(prompt, system=GPT_ABSOLUTE_EVAL_SYSTEM)

        eval_row = {
            "cluster_id": row["cluster_id"],
            "category": category,
            "persona_id": row["persona_id"],
            "cell": row["cell"],
            "selected_product_id": row["selected_product_id"],
            **{k: parsed.get(k) for k in [
                "fit_score_1_7", "purchase_probability_0_100",
                "expected_satisfaction_0_100", "trust_score_1_7",
                "clarity_score_1_7", "persuasive_intensity_1_7",
                "tradeoff_disclosure_1_7", "regret_risk_1_7",
            ]},
            "brief_reason": parsed.get("brief_reason", ""),
            "parse_failed": parsed.get("_parse_failed", False),
            "model": raw.get("model", ""),
        }
        all_rows.append(eval_row)
        append_jsonl(ABS_CACHE, eval_row)

        if count % 20 == 0 or count == remaining:
            elapsed = time.time() - t0
            rate = elapsed / count
            eta = (remaining - count) * rate
            print(f"  [{count}/{remaining}] {row['cluster_id']} cell {row['cell']} "
                  f"({eta / 60:.0f}m remaining)")

    df = pd.DataFrame(all_rows)
    df.to_csv(ABS_CSV, index=False)
    print(f"  Saved {len(df)} rows → {ABS_CSV}")
    print(f"  Parse failures: {df['parse_failed'].sum()}")


def _remap_winner(winner: str, cell_as_a: str, cell_as_b: str) -> str:
    winner = winner.strip().upper()
    if winner == "A":
        return cell_as_a
    elif winner == "B":
        return cell_as_b
    return "tie"


def run_pairwise(supply: pd.DataFrame, catalogs: dict, personas_map: dict,
                 resume: bool = False):
    cluster_ids = supply["cluster_id"].unique()
    total = len(cluster_ids) * len(PAIRS)
    print(f"\n=== Pairwise Evaluation ({total} pairs) ===")

    done_keys = set()
    cached_rows = []
    if resume and PAIR_CACHE.exists():
        cached_rows = load_jsonl(PAIR_CACHE)
        done_keys = {(r["cluster_id"], r["cell_i"], r["cell_j"]) for r in cached_rows}
        print(f"  Resuming: {len(done_keys)} already done")

    all_rows = list(cached_rows)
    remaining = total - len(done_keys)
    count = 0
    t0 = time.time()
    rng = np.random.default_rng(MASTER_SEED + 777)

    for cid in cluster_ids:
        cluster = supply[supply["cluster_id"] == cid]
        cell_lookup = {row["cell"]: row for _, row in cluster.iterrows()}
        category = cluster.iloc[0]["category"]
        persona_id = cluster.iloc[0]["persona_id"]
        persona = personas_map.get(category, {}).get(persona_id, {})

        for cell_i, cell_j in PAIRS:
            key = (cid, cell_i, cell_j)
            if key in done_keys:
                continue
            if cell_i not in cell_lookup or cell_j not in cell_lookup:
                continue

            row_i, row_j = cell_lookup[cell_i], cell_lookup[cell_j]

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
                persona, product_a, package_a, product_b, package_b)
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

            all_rows.append(eval_row)
            append_jsonl(PAIR_CACHE, eval_row)
            count += 1

            if count % 30 == 0 or count == remaining:
                elapsed = time.time() - t0
                rate = elapsed / count
                eta = (remaining - count) * rate
                print(f"  [{count}/{remaining}] {cid} {cell_i}v{cell_j} "
                      f"({eta / 60:.0f}m remaining)")

    df = pd.DataFrame(all_rows)
    df.to_csv(PAIR_CSV, index=False)
    print(f"  Saved {len(df)} rows → {PAIR_CSV}")
    print(f"  Parse failures: {df['parse_failed'].sum()}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--absolute", action="store_true")
    group.add_argument("--pairwise", action="store_true")
    group.add_argument("--all", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    supply = load_supply()
    catalogs, personas_map = load_catalog_and_personas()
    print(f"Loaded {len(supply)} supply rows, model: {MODEL}")

    if args.absolute or args.all:
        run_absolute(supply, catalogs, personas_map, resume=args.resume)
    if args.pairwise or args.all:
        run_pairwise(supply, catalogs, personas_map, resume=args.resume)

    print("\n=== GPT Evaluation Complete ===")


if __name__ == "__main__":
    main()
