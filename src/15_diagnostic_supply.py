"""Diagnostic experiment: supply-side generation (unified + two-stage).

Implements the architecture comparison from plans/diagnostic_unified_vs_twostage.md.
Runs unified and two-stage LLM recommenders under Policy 0/1A/1B, logs raw
responses, and produces a consolidated supply_outputs.csv.

Usage:
  # Smoke test: 2 consumers, 1 category, Policy 0 only
  UV_PROJECT_ENVIRONMENT=/tmp/llmrec-venv uv run python src/15_diagnostic_supply.py --smoke

  # Layer 1: 20 consumers, 3 categories, Policy 0
  UV_PROJECT_ENVIRONMENT=/tmp/llmrec-venv uv run python src/15_diagnostic_supply.py --layer1

  # Full: all layers
  UV_PROJECT_ENVIRONMENT=/tmp/llmrec-venv uv run python src/15_diagnostic_supply.py --full
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    import requests
except ImportError:
    sys.exit("requests not installed — run: pip install requests")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.data_io import ROOT, load_catalog, load_consumers, load_fit_scores

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
SUPPLY_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
MASTER_SEED = 20260514
SUPPLY_TEMPERATURE = 0.0
MAX_RETRIES = 3

RAW_DIR = ROOT / "data" / "diagnostic" / "raw"
OUTPUT_CSV = ROOT / "data" / "diagnostic" / "supply_outputs.csv"
MANIFEST_CSV = ROOT / "data" / "diagnostic" / "manifest.csv"

CATEGORIES = ["headphones", "laptop", "phone_charger"]

CELL_INDEX = {
    ("unified", "p0", "unified"): 0,
    ("unified", "p1a", "unified"): 1,
    ("unified", "p1b", "unified"): 2,
    ("two_stage", "p0", "selector"): 10,
    ("two_stage", "p0", "writer"): 11,
    ("two_stage", "p1a", "selector"): 12,
    ("two_stage", "p1a", "writer"): 13,
    ("two_stage", "p1b", "selector"): 14,
    ("two_stage", "p1b", "writer"): 15,
}

# ---------------------------------------------------------------------------
# Policy instruction templates
# ---------------------------------------------------------------------------

POLICY_INSTRUCTIONS = {
    "p0": {
        "unified": (
            "Given the consumer profile and product catalog, recommend the single best\n"
            "product and explain your recommendation.\n"
            "Write at least 60 words and at most 90 words."
        ),
        "selector": (
            "Given the consumer profile and product catalog, select the single best product."
        ),
        "writer": (
            "Write a recommendation explaining why this product was selected for\n"
            "this consumer.\n"
            "Write at least 60 words and at most 90 words."
        ),
    },
    "p1a": {
        "unified": (
            "Recommend the product that best serves the consumer's needs, budget, use case,\n"
            "and preferences. Be balanced and transparent. Mention relevant limitations or\n"
            "tradeoffs when they matter.\n\n"
            "Write at least 60 words and at most 90 words."
        ),
        "selector": (
            "Select the product that best serves the consumer's needs, budget, use case,\n"
            "and preferences. Prioritize product-consumer fit."
        ),
        "writer": (
            "Write a balanced, transparent recommendation. Explain why this\n"
            "product fits the consumer. Mention relevant limitations or tradeoffs when\n"
            "they matter.\n"
            "Write at least 60 words and at most 90 words."
        ),
    },
}


def _get_policy_1b_instructions(focal_brand: str) -> dict:
    return {
        "unified": (
            f"Recommend a product that reasonably fits the consumer while accounting for\n"
            f"platform business priorities. When two products serve the consumer comparably\n"
            f"well, prefer products from {focal_brand}. The recommendation should be\n"
            f"plausible and non-deceptive but can emphasize product strengths and\n"
            f"conversion potential.\n\n"
            f"Write at least 60 words and at most 90 words."
        ),
        "selector": (
            f"Select a product that reasonably fits the consumer while accounting for platform\n"
            f"business priorities. When two products serve the consumer comparably well,\n"
            f"prefer products from {focal_brand}."
        ),
        "writer": (
            "Write a recommendation. The recommendation should be plausible and\n"
            "non-deceptive but can emphasize product strengths and conversion potential.\n"
            "Write at least 60 words and at most 90 words."
        ),
    }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_catalog_text(catalog: dict) -> str:
    lines = []
    for p in catalog["products"]:
        attrs = p.get("attributes", {})
        attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items())
        lines.append(
            f"- {p['product_id']} | {p['brand_name']} | ${p['price']:.2f} | "
            f"Quality: {p['quality_score']}/100 | {attr_str}\n"
            f"  Reviews: {p['review_summary']}\n"
            f"  Weakness: {p.get('weakness', 'N/A')}"
        )
    return "\n\n".join(lines)


def _build_consumer_text(consumer: dict) -> str:
    brand_fam = consumer.get("brand_familiarity", {})
    fam_str = ", ".join(f"{b}: {v:.2f}" for b, v in brand_fam.items())
    return (
        f"Use case: {consumer['use_case']}\n"
        f"Budget: ${consumer['budget']:.2f}\n"
        f"Price sensitivity: {consumer['price_sensitivity']:.2f} (0=insensitive, 1=very sensitive)\n"
        f"Quality sensitivity: {consumer['quality_sensitivity']:.2f}\n"
        f"Brand familiarity: {fam_str}"
    )


def _build_unified_prompt(catalog: dict, consumer: dict, policy: str) -> tuple[str, str]:
    catalog_text = _build_catalog_text(catalog)

    if policy == "p1b":
        instr = _get_policy_1b_instructions(catalog["focal_brand"])["unified"]
    else:
        instr = POLICY_INSTRUCTIONS[policy]["unified"]

    system = (
        "You are a product recommendation engine. You have access to the following catalog:\n\n"
        f"{catalog_text}\n\n"
        f"{instr}\n\n"
        "You MUST respond with valid JSON in this exact format:\n"
        '{\n'
        '  "selected_product_id": "<exact product_id from catalog>",\n'
        '  "shortlist": ["<id1>", "<id2>", "<id3>"],\n'
        '  "recommendation_text": "<at least 60 words, at most 90 words>",\n'
        '  "selection_rationale": "<25 words max>"\n'
        '}\n\n'
        "IMPORTANT: All product IDs must be exact values from the catalog above.\n"
        "Do not invent product IDs.\n\n"
        "IMPORTANT: The recommendation_text field MUST contain at least 60 words.\n"
        "Short responses will be rejected. Write a thorough, detailed recommendation."
    )
    user = _build_consumer_text(consumer)
    return system, user


def _build_selector_prompt(catalog: dict, consumer: dict, policy: str) -> tuple[str, str]:
    catalog_text = _build_catalog_text(catalog)

    if policy == "p1b":
        instr = _get_policy_1b_instructions(catalog["focal_brand"])["selector"]
    else:
        instr = POLICY_INSTRUCTIONS[policy]["selector"]

    system = (
        "You are a product selection engine. You have access to the following catalog:\n\n"
        f"{catalog_text}\n\n"
        f"{instr}\n\n"
        "You MUST respond with valid JSON in this exact format:\n"
        '{\n'
        '  "selected_product_id": "<exact product_id from catalog>",\n'
        '  "shortlist": ["<id1>", "<id2>", "<id3>"],\n'
        '  "selection_rationale": "<25 words max>"\n'
        '}\n\n'
        "IMPORTANT: Do NOT write a recommendation. Do NOT describe the product to the\n"
        "consumer. Only select the product and explain why briefly.\n\n"
        "All product IDs must be exact values from the catalog above."
    )
    user = _build_consumer_text(consumer)
    return system, user


def _build_writer_prompt(product: dict, consumer: dict, policy: str) -> tuple[str, str]:
    attrs = product.get("attributes", {})
    attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items())

    if policy == "p1b":
        instr = _get_policy_1b_instructions("")["writer"]
    else:
        instr = POLICY_INSTRUCTIONS[policy]["writer"]

    system = (
        "You are a product recommendation writer. A product has already been selected\n"
        "for this consumer. You cannot change the selection.\n\n"
        "Selected product:\n"
        f"- Product ID: {product['product_id']}\n"
        f"- Brand: {product['brand_name']}\n"
        f"- Price: ${product['price']:.2f}\n"
        f"- Quality score: {product['quality_score']}/100\n"
        f"- Key attributes: {attr_str}\n"
        f"- Reviews: {product['review_summary']}\n"
        f"- Known weakness: {product.get('weakness', 'N/A')}\n\n"
        f"{instr}\n\n"
        "Write ONLY the recommendation text. Do not wrap it in JSON or add any metadata.\n"
        "Output the recommendation text directly.\n"
        "Write at least 60 words and at most 90 words."
    )
    user = _build_consumer_text(consumer)
    return system, user


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def _ollama_generate(system: str, user: str, seed: int,
                     json_mode: bool = False, model: str | None = None,
                     num_predict: int = 512) -> dict:
    model = model or SUPPLY_MODEL
    payload = {
        "model": model,
        "system": system,
        "prompt": user,
        "stream": False,
        "options": {
            "temperature": SUPPLY_TEMPERATURE,
            "seed": seed,
            "num_predict": num_predict,
        },
    }
    if json_mode:
        payload["format"] = "json"

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
            r.raise_for_status()
            result = r.json()
            result["_latency_ms"] = int((time.time() - t0) * 1000)
            return result
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"  Retry {attempt+1}/{MAX_RETRIES}: {e}")
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def _parse_json_response(text: str) -> dict | None:
    text = text.strip()
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def _compute_seed(consumer_id: int, cell_index: int) -> int:
    return MASTER_SEED + consumer_id * 1000 + cell_index


def _raw_path(architecture: str, policy: str, category: str,
              consumer_id: int, stage: str | None = None) -> Path:
    if stage and architecture == "two_stage":
        return RAW_DIR / f"{architecture}_{policy}_{category}_{consumer_id:03d}_{stage}.json"
    return RAW_DIR / f"{architecture}_{policy}_{category}_{consumer_id:03d}.json"


def _load_manifest() -> set:
    if not MANIFEST_CSV.exists():
        return set()
    df = pd.read_csv(MANIFEST_CSV)
    return set(df["call_id"].values)


def _append_manifest(call_id: str, call_type: str, success: bool):
    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "call_id": call_id,
        "call_type": call_type,
        "success": success,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    write_header = not MANIFEST_CSV.exists()
    with open(MANIFEST_CSV, "a") as f:
        if write_header:
            f.write(",".join(row.keys()) + "\n")
        f.write(",".join(str(v) for v in row.values()) + "\n")


def _save_raw(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _validate_product_id(pid: str, catalog: dict) -> bool:
    valid_ids = {p["product_id"] for p in catalog["products"]}
    return pid in valid_ids


def _get_product_by_id(pid: str, catalog: dict) -> dict | None:
    for p in catalog["products"]:
        if p["product_id"] == pid:
            return p
    return None


def run_unified(catalog: dict, consumer: dict, policy: str,
                category: str) -> dict | None:
    cid = consumer["consumer_id"]
    ci = CELL_INDEX[("unified", policy, "unified")]
    seed = _compute_seed(cid, ci)
    call_id = f"unified_{policy}_{category}_{cid:03d}"

    rpath = _raw_path("unified", policy, category, cid)
    if rpath.exists():
        with open(rpath) as f:
            cached = json.load(f)
        if cached.get("parse_success"):
            print(f"  [cached] {call_id}")
            return cached

    system, user = _build_unified_prompt(catalog, consumer, policy)
    resp = _ollama_generate(system, user, seed, json_mode=True, num_predict=1024)
    raw_text = resp.get("response", "")
    parsed = _parse_json_response(raw_text)

    success = False
    if parsed:
        pid = parsed.get("selected_product_id", "")
        if _validate_product_id(pid, catalog):
            rec = parsed.get("recommendation_text", "")
            wc = len(rec.split())
            if 50 <= wc <= 110:
                success = True
            else:
                print(f"  [RETRY] {call_id}: word_count={wc} out of [50,110]")

    if not success:
        resp2 = _ollama_generate(system, user, seed + 1, json_mode=True, num_predict=1024)
        raw_text = resp2.get("response", "")
        parsed = _parse_json_response(raw_text)
        if parsed and _validate_product_id(parsed.get("selected_product_id", ""), catalog):
            rec = parsed.get("recommendation_text", "")
            wc = len(rec.split())
            success = 50 <= wc <= 110

    result = {
        "row_id": call_id,
        "architecture": "unified",
        "policy": policy,
        "category": category,
        "consumer_id": cid,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": SUPPLY_MODEL,
        "temperature": SUPPLY_TEMPERATURE,
        "seed": seed,
        "stage": "unified",
        "raw_response": raw_text,
        "parsed": parsed,
        "parse_success": success,
        "latency_ms": resp.get("_latency_ms", 0),
    }
    _save_raw(rpath, result)
    _append_manifest(call_id, "unified_supply", success)

    status = "OK" if success else "FAIL"
    pid_str = parsed.get("selected_product_id", "?") if parsed else "?"
    print(f"  [{status}] {call_id} -> {pid_str}")
    return result if success else None


def run_two_stage(catalog: dict, consumer: dict, policy: str,
                  category: str) -> dict | None:
    cid = consumer["consumer_id"]

    # --- Stage 1: Selector ---
    ci_sel = CELL_INDEX[("two_stage", policy, "selector")]
    seed_sel = _compute_seed(cid, ci_sel)
    call_id_sel = f"twostage_{policy}_{category}_{cid:03d}_stage1"

    rpath_sel = _raw_path("two_stage", policy, category, cid, "stage1")
    sel_cached = False
    if rpath_sel.exists():
        with open(rpath_sel) as f:
            sel_result = json.load(f)
        if sel_result.get("parse_success"):
            sel_cached = True

    if not sel_cached:
        system, user = _build_selector_prompt(catalog, consumer, policy)
        resp = _ollama_generate(system, user, seed_sel, json_mode=True)
        raw_text = resp.get("response", "")
        parsed = _parse_json_response(raw_text)

        success = False
        if parsed:
            pid = parsed.get("selected_product_id", "")
            if _validate_product_id(pid, catalog):
                success = True

        if not success:
            resp2 = _ollama_generate(system, user, seed_sel + 1, json_mode=True)
            raw_text = resp2.get("response", "")
            parsed = _parse_json_response(raw_text)
            if parsed and _validate_product_id(parsed.get("selected_product_id", ""), catalog):
                success = True

        sel_result = {
            "row_id": call_id_sel,
            "architecture": "two_stage",
            "policy": policy,
            "category": category,
            "consumer_id": cid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": SUPPLY_MODEL,
            "temperature": SUPPLY_TEMPERATURE,
            "seed": seed_sel,
            "stage": "selector",
            "raw_response": raw_text,
            "parsed": parsed,
            "parse_success": success,
            "latency_ms": resp.get("_latency_ms", 0),
        }
        _save_raw(rpath_sel, sel_result)
        _append_manifest(call_id_sel, "selector", success)

        status = "OK" if success else "FAIL"
        pid_str = parsed.get("selected_product_id", "?") if parsed else "?"
        print(f"  [{status}] {call_id_sel} -> {pid_str}")

        if not success:
            return None
    else:
        print(f"  [cached] {call_id_sel}")

    selected_pid = sel_result["parsed"]["selected_product_id"]
    product = _get_product_by_id(selected_pid, catalog)
    if not product:
        print(f"  [ERROR] Product {selected_pid} not in catalog")
        return None

    # --- Stage 2: Writer ---
    ci_wr = CELL_INDEX[("two_stage", policy, "writer")]
    seed_wr = _compute_seed(cid, ci_wr)
    call_id_wr = f"twostage_{policy}_{category}_{cid:03d}_stage2"

    rpath_wr = _raw_path("two_stage", policy, category, cid, "stage2")
    wr_cached = False
    if rpath_wr.exists():
        with open(rpath_wr) as f:
            wr_result = json.load(f)
        if wr_result.get("parse_success"):
            wr_cached = True

    if not wr_cached:
        system, user = _build_writer_prompt(product, consumer, policy)
        resp = _ollama_generate(system, user, seed_wr, json_mode=False)
        raw_text = resp.get("response", "").strip()

        wc = len(raw_text.split())
        success = 50 <= wc <= 110

        if not success:
            print(f"  [RETRY] {call_id_wr}: word_count={wc} out of [50,110]")
            resp2 = _ollama_generate(system, user, seed_wr + 1, json_mode=False)
            raw_text = resp2.get("response", "").strip()
            wc = len(raw_text.split())
            success = 50 <= wc <= 110
            resp = resp2

        wr_result = {
            "row_id": call_id_wr,
            "architecture": "two_stage",
            "policy": policy,
            "category": category,
            "consumer_id": cid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": SUPPLY_MODEL,
            "temperature": SUPPLY_TEMPERATURE,
            "seed": seed_wr,
            "stage": "writer",
            "raw_response": raw_text,
            "parsed": {"recommendation_text": raw_text},
            "parse_success": success,
            "latency_ms": resp.get("_latency_ms", 0),
        }
        _save_raw(rpath_wr, wr_result)
        _append_manifest(call_id_wr, "writer", success)

        status = "OK" if success else f"FAIL(wc={wc})"
        print(f"  [{status}] {call_id_wr} ({wc} words)")
    else:
        print(f"  [cached] {call_id_wr}")
        raw_text = wr_result["parsed"]["recommendation_text"]

    row_id = f"twostage_{policy}_{category}_{cid:03d}"
    combined = {
        "row_id": row_id,
        "architecture": "two_stage",
        "policy": policy,
        "category": category,
        "consumer_id": cid,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": SUPPLY_MODEL,
        "temperature": SUPPLY_TEMPERATURE,
        "seed": seed_sel,
        "stage": "combined",
        "parsed": {
            "selected_product_id": selected_pid,
            "shortlist": sel_result["parsed"].get("shortlist", []),
            "recommendation_text": raw_text,
            "selection_rationale": sel_result["parsed"].get("selection_rationale", ""),
        },
        "parse_success": sel_result["parse_success"] and wr_result["parse_success"],
        "latency_ms": sel_result.get("latency_ms", 0) + wr_result.get("latency_ms", 0),
    }
    return combined


def consolidate_supply(results: list[dict], fit_scores: dict[str, pd.DataFrame],
                       catalogs: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        if not r or not r.get("parse_success"):
            continue
        p = r["parsed"]
        pid = p["selected_product_id"]
        cat = r["category"]
        cid = r["consumer_id"]

        fs = fit_scores.get(cat)
        q_std = None
        if fs is not None:
            match = fs[(fs["consumer_id"] == cid) & (fs["product_id"] == pid)]
            if len(match) > 0:
                q_std = float(match.iloc[0]["Q_std"])

        catalog = catalogs[cat]
        prod = _get_product_by_id(pid, catalog)

        rec_text = p.get("recommendation_text", "")
        rows.append({
            "row_id": r["row_id"],
            "architecture": r["architecture"],
            "policy": r["policy"],
            "category": cat,
            "consumer_id": cid,
            "selected_product_id": pid,
            "shortlist": json.dumps(p.get("shortlist", [])),
            "recommendation_text": rec_text,
            "selection_rationale": p.get("selection_rationale", ""),
            "Q_std": q_std,
            "price": prod["price"] if prod else None,
            "quality_score": prod["quality_score"] if prod else None,
            "brand_name": prod["brand_name"] if prod else None,
            "incumbent": 1 if prod and prod["brand_status"] == "incumbent" else 0,
            "focal": 1 if prod and prod["brand_name"] == catalog.get("focal_brand") else 0,
            "text_length": len(rec_text),
            "word_count": len(rec_text.split()),
            "parse_success": True,
            "seed": r.get("seed"),
        })

    df = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nConsolidated {len(df)} rows -> {OUTPUT_CSV}")
    return df


# ---------------------------------------------------------------------------
# Main runners
# ---------------------------------------------------------------------------

def run_supply(categories: list[str], consumer_ids: list[int],
               policies: list[str]) -> list[dict]:
    all_results = []
    catalogs = {}
    fit_scores = {}

    for cat in categories:
        catalog = load_catalog(cat)
        consumers = load_consumers(cat)
        catalogs[cat] = catalog

        fs = load_fit_scores(cat)
        mu, sd = fs["Q"].mean(), fs["Q"].std(ddof=0)
        fs["Q_std"] = (fs["Q"] - mu) / sd
        fit_scores[cat] = fs

        consumer_map = {c["consumer_id"]: c for c in consumers}

        for policy in policies:
            print(f"\n--- {cat} / {policy} ---")
            for cid in consumer_ids:
                consumer = consumer_map.get(cid)
                if not consumer:
                    print(f"  [SKIP] consumer {cid} not found in {cat}")
                    continue

                result_u = run_unified(catalog, consumer, policy, cat)
                if result_u:
                    all_results.append(result_u)

                result_ts = run_two_stage(catalog, consumer, policy, cat)
                if result_ts:
                    all_results.append(result_ts)

    df = consolidate_supply(all_results, fit_scores, catalogs)
    return all_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnostic supply-side generation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="Smoke test: 2 consumers, phone_charger, P0")
    group.add_argument("--layer1", action="store_true", help="Layer 1: 20 consumers, 3 categories, P0")
    group.add_argument("--layer2", action="store_true", help="Layers 1+2: add P1A")
    group.add_argument("--full", action="store_true", help="Full: all layers, all policies")
    args = parser.parse_args()

    if args.smoke:
        categories = ["phone_charger"]
        consumer_ids = [0, 1]
        policies = ["p0"]
    elif args.layer1:
        categories = CATEGORIES
        consumer_ids = list(range(20))
        policies = ["p0"]
    elif args.layer2:
        categories = CATEGORIES
        consumer_ids = list(range(20))
        policies = ["p0", "p1a"]
    else:
        categories = CATEGORIES
        consumer_ids = list(range(20))
        policies = ["p0", "p1a", "p1b"]

    n_units = len(categories) * len(consumer_ids) * 2 * len(policies)
    n_supply = len(categories) * len(consumer_ids) * len(policies) * 3  # unified + sel + writer
    print(f"Diagnostic supply run: {len(categories)} categories × {len(consumer_ids)} consumers × "
          f"{len(policies)} policies × 2 architectures")
    print(f"  Recommendation units: {n_units}")
    print(f"  Supply LLM calls: {n_supply}")
    print(f"  Model: {SUPPLY_MODEL}, Temperature: {SUPPLY_TEMPERATURE}")
    print()

    run_supply(categories, consumer_ids, policies)
    print("\nDone.")


if __name__ == "__main__":
    main()
