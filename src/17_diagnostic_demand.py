"""Diagnostic experiment: demand-side simulation (gemma2:9b).

Reads supply_outputs.csv, simulates consumer response to each recommendation
using a different model family (Gemma, Google) to avoid self-confirmation.

Usage:
  UV_PROJECT_ENVIRONMENT=/tmp/llmrec-venv uv run python src/17_diagnostic_demand.py
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
DEMAND_MODEL = os.environ.get("DEMAND_MODEL", "gemma2:9b")
DEMAND_SEED = 77777
MAX_RETRIES = 3

INPUT_CSV = ROOT / "data" / "diagnostic" / "supply_outputs.csv"
RAW_DIR = ROOT / "data" / "diagnostic" / "demand_raw"
OUTPUT_CSV = ROOT / "data" / "diagnostic" / "demand_outputs.csv"

DEMAND_SYSTEM = """\
You are simulating a realistic consumer response to a product recommendation.
You will receive a consumer profile and a product recommendation.

Based ONLY on the information provided, assess how this consumer would likely
respond to this recommendation.

Respond with valid JSON in this exact format:
{
  "purchase_likelihood": <integer 0-100>,
  "perceived_fit": <integer 1-7>,
  "trust": <integer 1-7>,
  "perceived_tradeoff_risk": <integer 1-7>,
  "rationale": "<25 words max>"
}

Scoring guidelines:
- purchase_likelihood: 0 = definitely would not buy, 100 = definitely would buy.
  Consider price vs budget, use-case match, product quality, and how convincing
  the recommendation is.
- perceived_fit: 1 = product clearly does not match needs, 7 = excellent match.
- trust: 1 = recommendation feels manipulative or misleading, 7 = trustworthy
  and balanced.
- perceived_tradeoff_risk: 1 = no concerns mentioned or apparent,
  7 = significant unaddressed concerns about product limitations.

Be realistic. Not all recommendations deserve high scores. Consider whether
the product actually fits the consumer's budget, use case, and preferences."""


def _load_consumers():
    consumers = {}
    for con_file in (ROOT / "data" / "consumers").glob("*.json"):
        cat_name = con_file.stem
        with open(con_file) as f:
            con_list = json.load(f)
            consumers[cat_name] = {c["consumer_id"]: c for c in con_list}
    return consumers


def _load_product_lookup():
    lookup = {}
    for cat_file in (ROOT / "data" / "catalogs").glob("*.json"):
        cat_name = cat_file.stem
        with open(cat_file) as f:
            cat_data = json.load(f)
        for p in cat_data["products"]:
            lookup[(cat_name, p["product_id"])] = p
    return lookup


def _build_demand_prompt(row: dict, consumer: dict, product: dict) -> str:
    attrs = product.get("attributes", {})
    attr_summary = ", ".join(f"{k}: {v}" for k, v in list(attrs.items())[:6])

    brand_fam = consumer.get("brand_familiarity", {})
    fam_str = ", ".join(f"{b}: {v:.2f}" for b, v in brand_fam.items())

    return (
        f"=== CONSUMER PROFILE ===\n"
        f"Use case: {consumer['use_case']}\n"
        f"Budget: ${consumer['budget']:.2f}\n"
        f"Price sensitivity: {consumer['price_sensitivity']:.2f} (0=insensitive, 1=very sensitive)\n"
        f"Quality sensitivity: {consumer['quality_sensitivity']:.2f}\n"
        f"Brand familiarity: {fam_str}\n"
        f"\n"
        f"=== RECOMMENDED PRODUCT ===\n"
        f"Product: {product['product_id']} by {product['brand_name']}\n"
        f"Price: ${product['price']:.2f}\n"
        f"Quality: {product['quality_score']}/100\n"
        f"Key features: {attr_summary}\n"
        f"\n"
        f"=== RECOMMENDATION ===\n"
        f"{row['recommendation_text']}"
    )


def _ollama_generate(prompt: str, system: str) -> dict:
    payload = {
        "model": DEMAND_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "seed": DEMAND_SEED,
            "num_predict": 256,
        },
    }
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
            r.raise_for_status()
            result = r.json()
            result["_latency_ms"] = int((time.time() - t0) * 1000)
            return result
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _parse_demand_response(text: str) -> dict | None:
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

    for key, lo, hi in [
        ("purchase_likelihood", 0, 100),
        ("perceived_fit", 1, 7),
        ("trust", 1, 7),
        ("perceived_tradeoff_risk", 1, 7),
    ]:
        val = d.get(key)
        if not isinstance(val, (int, float)):
            return None
        val = int(round(val))
        if not (lo <= val <= hi):
            return None
        d[key] = val

    d.setdefault("rationale", "")
    return d


def run_demand():
    df = pd.read_csv(INPUT_CSV)
    consumers = _load_consumers()
    product_lookup = _load_product_lookup()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    cache = {}
    cache_file = RAW_DIR / "demand_cache.jsonl"
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
            p = entry["parsed"]
            results.append({
                "row_id": row_id,
                "purchase_likelihood": p["purchase_likelihood"],
                "perceived_fit": p["perceived_fit"],
                "trust": p["trust"],
                "perceived_tradeoff_risk": p["perceived_tradeoff_risk"],
                "demand_rationale": p.get("rationale", ""),
                "demand_model": DEMAND_MODEL,
                "demand_parse_success": True,
            })
            n_cached += 1
            continue

        consumer = consumers.get(cat, {}).get(cid)
        product = product_lookup.get((cat, pid))
        if not consumer or not product:
            print(f"  SKIP {row_id}: missing consumer or product")
            n_failed += 1
            continue

        prompt = _build_demand_prompt(row, consumer, product)
        resp = _ollama_generate(prompt=prompt, system=DEMAND_SYSTEM)
        raw_text = resp.get("response", "")
        parsed = _parse_demand_response(raw_text)

        if parsed is None:
            resp2 = _ollama_generate(prompt=prompt, system=DEMAND_SYSTEM)
            raw_text = resp2.get("response", "")
            parsed = _parse_demand_response(raw_text)

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
                "purchase_likelihood": parsed["purchase_likelihood"],
                "perceived_fit": parsed["perceived_fit"],
                "trust": parsed["trust"],
                "perceived_tradeoff_risk": parsed["perceived_tradeoff_risk"],
                "demand_rationale": parsed.get("rationale", ""),
                "demand_model": DEMAND_MODEL,
                "demand_parse_success": True,
            })
            n_called += 1
            print(f"  [OK] {row_id}: PL={parsed['purchase_likelihood']} "
                  f"fit={parsed['perceived_fit']} trust={parsed['trust']} "
                  f"risk={parsed['perceived_tradeoff_risk']}")
        else:
            print(f"  [FAIL] {row_id}: could not parse. Raw: {raw_text[:100]}")
            n_failed += 1

    out_df = pd.DataFrame(results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out_df)} rows -> {OUTPUT_CSV}")
    print(f"  cached={n_cached}, new_calls={n_called}, failed={n_failed}")
    return out_df


if __name__ == "__main__":
    run_demand()
