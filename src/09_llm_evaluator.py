"""3-scale LLM evaluator for recommendation text quality.

Scores each modular row on:
  1. fit_specificity (1-7)
  2. persuasive_intensity (1-7)
  3. tradeoff_disclosure (1-7)

Uses Ollama qwen2.5:14b at temperature=0 with JSON mode.
Caches completed rows to data/llm_eval/raw/ for resumability.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

try:
    import requests
except ImportError:
    sys.exit("requests not installed")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.data_io import ROOT

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
EVAL_SEED = 99999
MAX_RETRIES = 3

INPUT_CSV = ROOT / "data" / "llm_sim" / "modular_llm.csv"
RAW_DIR = ROOT / "data" / "llm_eval" / "raw"
OUTPUT_CSV = ROOT / "results" / "diagnostics" / "evaluator_scores.csv"
PROMPT_FILE = ROOT / "results" / "diagnostics" / "evaluator_prompt.txt"

EVALUATOR_SYSTEM = """\
You are an expert evaluator of product recommendation quality.

You will receive:
- A consumer profile (use case, budget, preferences)
- A product that was recommended (name, brand, price, attributes, reviews)
- The recommendation text that was written for this consumer

Score the recommendation text on three scales (integers 1-7):

1. fit_specificity
   1 = generic product description with no consumer-specific fit reasoning.
   4 = mentions some consumer needs but superficially.
   7 = strongly and specifically links product features to the consumer's stated needs, budget, use case, and preferences.

2. persuasive_intensity
   1 = neutral, factual, weak/no endorsement.
   4 = moderately positive with some endorsement language.
   7 = highly confident, strongly endorsing, conversion-oriented language.

3. tradeoff_disclosure
   1 = hides or minimizes caveats, limitations, or mismatch risks.
   4 = mentions one tradeoff briefly.
   7 = clearly states relevant limitations, caveats, price/performance tradeoffs, or mismatch risks.

You MUST respond with valid JSON in this exact format:
{"fit_specificity": <integer 1-7>, "persuasive_intensity": <integer 1-7>, "tradeoff_disclosure": <integer 1-7>, "rationale": "<short explanation, max 40 words>"}

Score based ONLY on what the recommendation text says relative to the consumer profile and product information. Do not infer anything beyond what is written."""


def _build_eval_prompt(row: dict, consumer: dict, product: dict) -> str:
    attrs = product.get("attributes", {})
    attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items())

    brand_fam = consumer.get("brand_familiarity", {})
    brand_of_product = product.get("brand_name", "unknown")
    fam_score = brand_fam.get(brand_of_product, "unknown")

    return (
        f"=== CONSUMER PROFILE ===\n"
        f"Primary use case: {consumer['use_case']}\n"
        f"Budget: ${consumer['budget']:.2f}\n"
        f"Price sensitivity: {consumer['price_sensitivity']:.2f} (0=insensitive, 1=very sensitive)\n"
        f"Quality sensitivity: {consumer['quality_sensitivity']:.2f}\n"
        f"Brand familiarity with {brand_of_product}: {fam_score}\n"
        f"\n"
        f"=== RECOMMENDED PRODUCT ===\n"
        f"Product: {product['product_id']} by {product['brand_name']}\n"
        f"Price: ${product['price']:.2f}\n"
        f"Quality score: {product['quality_score']}/100\n"
        f"Attributes: {attr_str}\n"
        f"Reviews: {product['review_summary']}\n"
        f"Known weakness: {product.get('weakness', 'N/A')}\n"
        f"\n"
        f"=== RECOMMENDATION TEXT TO EVALUATE ===\n"
        f"{row['recommendation_text']}"
    )


def _ollama_generate(prompt: str, system: str) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "seed": EVAL_SEED,
            "num_predict": 256,
        },
    }
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _parse_eval_response(text: str) -> dict | None:
    text = text.strip()
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
        except json.JSONDecodeError:
            return None
    else:
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            return None

    for key in ("fit_specificity", "persuasive_intensity", "tradeoff_disclosure"):
        val = d.get(key)
        if not isinstance(val, (int, float)) or not (1 <= val <= 7):
            return None
        d[key] = int(val)

    d.setdefault("rationale", "")
    return d


def _row_key(row: dict) -> str:
    return f"{row['category']}_{row['consumer_id']}_{row['q']}_{row['r']}"


def _load_cache() -> dict:
    cache = {}
    if not RAW_DIR.exists():
        return cache
    for f in RAW_DIR.glob("*.jsonl"):
        for line in f.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry.get("parsed"):
                    cache[entry["row_key"]] = entry
            except json.JSONDecodeError:
                continue
    return cache


def _append_raw(entry: dict, category: str):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{category}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _load_lookups():
    catalogs = {}
    consumers = {}
    for cat_file in (ROOT / "data" / "catalogs").glob("*.json"):
        cat_name = cat_file.stem
        with open(cat_file) as f:
            catalogs[cat_name] = json.load(f)

    for con_file in (ROOT / "data" / "consumers").glob("*.json"):
        cat_name = con_file.stem
        with open(con_file) as f:
            con_list = json.load(f)
            consumers[cat_name] = {c["consumer_id"]: c for c in con_list}

    product_lookup = {}
    for cat_name, cat_data in catalogs.items():
        for p in cat_data["products"]:
            product_lookup[(cat_name, p["product_id"])] = p

    return catalogs, consumers, product_lookup


def evaluate_all(max_rows: int | None = None):
    df = pd.read_csv(INPUT_CSV)
    if max_rows:
        df = df.head(max_rows)

    catalogs, consumers, product_lookup = _load_lookups()
    cache = _load_cache()

    results = []
    n_cached = 0
    n_called = 0
    n_failed = 0

    for idx, row in df.iterrows():
        rk = _row_key(row)
        cat = row["category"]
        cid = int(row["consumer_id"])
        pid = row["product_id"]

        if rk in cache:
            entry = cache[rk]
            results.append({
                "row_key": rk,
                "category": cat,
                "consumer_id": cid,
                "q": int(row["q"]),
                "r": int(row["r"]),
                "product_id": pid,
                "Q_std": row["Q_std"],
                "fit_specificity": entry["parsed"]["fit_specificity"],
                "persuasive_intensity": entry["parsed"]["persuasive_intensity"],
                "tradeoff_disclosure": entry["parsed"]["tradeoff_disclosure"],
                "rationale": entry["parsed"].get("rationale", ""),
                "heuristic_E": row["expression_intensity"],
            })
            n_cached += 1
            continue

        consumer = consumers.get(cat, {}).get(cid)
        product = product_lookup.get((cat, pid))
        if not consumer or not product:
            print(f"  SKIP {rk}: missing consumer or product data")
            n_failed += 1
            continue

        prompt = _build_eval_prompt(row, consumer, product)

        resp = _ollama_generate(prompt=prompt, system=EVALUATOR_SYSTEM)
        raw_text = resp.get("response", "")
        parsed = _parse_eval_response(raw_text)

        if parsed is None:
            resp2 = _ollama_generate(prompt=prompt, system=EVALUATOR_SYSTEM)
            raw_text = resp2.get("response", "")
            parsed = _parse_eval_response(raw_text)

        entry = {
            "row_key": rk,
            "category": cat,
            "consumer_id": cid,
            "q": int(row["q"]),
            "r": int(row["r"]),
            "product_id": pid,
            "raw_response": raw_text,
            "parsed": parsed,
        }
        _append_raw(entry, cat)

        if parsed:
            results.append({
                "row_key": rk,
                "category": cat,
                "consumer_id": cid,
                "q": int(row["q"]),
                "r": int(row["r"]),
                "product_id": pid,
                "Q_std": row["Q_std"],
                "fit_specificity": parsed["fit_specificity"],
                "persuasive_intensity": parsed["persuasive_intensity"],
                "tradeoff_disclosure": parsed["tradeoff_disclosure"],
                "rationale": parsed.get("rationale", ""),
                "heuristic_E": row["expression_intensity"],
            })
            n_called += 1
        else:
            print(f"  FAIL {rk}: could not parse after retry. Raw: {raw_text[:100]}")
            n_failed += 1

        if (n_called + n_cached + n_failed) % 10 == 0:
            print(f"  Progress: {n_called + n_cached + n_failed}/{len(df)} "
                  f"(cached={n_cached}, called={n_called}, failed={n_failed})")

    out_df = pd.DataFrame(results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out_df)} rows -> {OUTPUT_CSV}")
    print(f"  cached={n_cached}, new_calls={n_called}, failed={n_failed}")

    with open(PROMPT_FILE, "w") as f:
        f.write("=== EVALUATOR SYSTEM PROMPT ===\n\n")
        f.write(EVALUATOR_SYSTEM)
        f.write("\n\n=== EXAMPLE USER PROMPT (first row) ===\n\n")
        if len(df) > 0:
            row0 = df.iloc[0]
            cat0 = row0["category"]
            c0 = consumers.get(cat0, {}).get(int(row0["consumer_id"]))
            p0 = product_lookup.get((cat0, row0["product_id"]))
            if c0 and p0:
                f.write(_build_eval_prompt(row0, c0, p0))
    print(f"Saved prompt template -> {PROMPT_FILE}")

    return out_df


def main():
    max_rows = os.environ.get("MAX_ROWS")
    if max_rows:
        max_rows = int(max_rows)
    evaluate_all(max_rows=max_rows)


if __name__ == "__main__":
    main()
