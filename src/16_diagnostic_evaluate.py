"""Diagnostic experiment: evaluator (PI, TD scores).

Reads supply_outputs.csv from the diagnostic, scores each recommendation
text using the validated qwen2.5:14b evaluator (same prompt as src/09).

Usage:
  UV_PROJECT_ENVIRONMENT=/tmp/llmrec-venv uv run python src/16_diagnostic_evaluate.py
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

INPUT_CSV = ROOT / "data" / "diagnostic" / "supply_outputs.csv"
RAW_DIR = ROOT / "data" / "diagnostic" / "eval_raw"
OUTPUT_CSV = ROOT / "data" / "diagnostic" / "evaluator_outputs.csv"

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
            t0 = time.time()
            r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=120)
            r.raise_for_status()
            result = r.json()
            result["_latency_ms"] = int((time.time() - t0) * 1000)
            return result
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


def evaluate_all():
    df = pd.read_csv(INPUT_CSV)
    _, consumers, product_lookup = _load_lookups()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    cache = {}
    cache_file = RAW_DIR / "eval_cache.jsonl"
    if cache_file.exists():
        for line in cache_file.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry.get("parsed"):
                    cache[entry["row_id"]] = entry
            except json.JSONDecodeError:
                continue

    results = []
    n_cached = 0
    n_called = 0
    n_failed = 0

    for _, row in df.iterrows():
        row_id = row["row_id"]
        cat = row["category"]
        cid = int(row["consumer_id"])
        pid = row["selected_product_id"]

        if row_id in cache:
            entry = cache[row_id]
            results.append({
                "row_id": row_id,
                "persuasive_intensity": entry["parsed"]["persuasive_intensity"],
                "tradeoff_disclosure": entry["parsed"]["tradeoff_disclosure"],
                "fit_specificity": entry["parsed"]["fit_specificity"],
                "eval_rationale": entry["parsed"].get("rationale", ""),
            })
            n_cached += 1
            continue

        consumer = consumers.get(cat, {}).get(cid)
        product = product_lookup.get((cat, pid))
        if not consumer or not product:
            print(f"  SKIP {row_id}: missing consumer or product")
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
            "row_id": row_id,
            "raw_response": raw_text,
            "parsed": parsed,
        }
        with open(cache_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        if parsed:
            results.append({
                "row_id": row_id,
                "persuasive_intensity": parsed["persuasive_intensity"],
                "tradeoff_disclosure": parsed["tradeoff_disclosure"],
                "fit_specificity": parsed["fit_specificity"],
                "eval_rationale": parsed.get("rationale", ""),
            })
            n_called += 1
            status = f"PI={parsed['persuasive_intensity']} TD={parsed['tradeoff_disclosure']}"
            print(f"  [OK] {row_id}: {status}")
        else:
            print(f"  [FAIL] {row_id}: could not parse")
            n_failed += 1

    out_df = pd.DataFrame(results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out_df)} rows -> {OUTPUT_CSV}")
    print(f"  cached={n_cached}, new_calls={n_called}, failed={n_failed}")
    return out_df


if __name__ == "__main__":
    evaluate_all()
