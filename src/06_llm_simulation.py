"""Phase 2: LLM-based simulation using Ollama (prompt-engineering pass).

Runs three sub-experiments via qwen2.5:14b on Ollama:
  A. One-shot bundled (z=0 baseline, z=1 brand-forward+persuasive)
  B. Modular retrieval (q=0 baseline, q=1 brand-forward)
  C. Modular expression (r=0 neutral, r=1 persuasive) conditioned on B outputs

Uses the structural DGP outcome layer from Phase 1 to generate Y_i,
with LLM-generated selected_product and expression_intensity replacing
the hand-specified kernels.

Replicability: master_seed=20260514, per-call seed = master_seed + consumer_id*100 + cell_index.
Ollama calls use temperature=0.7 and seed parameter for reproducibility.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.data_io import load_all_categories, CategoryData, ROOT

try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
MASTER_SEED = 20260514
N_CONSUMERS = int(os.environ.get("N_CONSUMERS", "10"))
TEMPERATURE = 0.7
MAX_RETRIES = 3

OUTPUT_DIR = ROOT / "data" / "llm_sim"
RAW_DIR = OUTPUT_DIR / "raw"


def _ollama_generate(prompt: str, system: str, seed: int,
                     json_mode: bool = False) -> dict:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "seed": seed,
            "num_predict": 512,
        },
    }
    if json_mode:
        payload["format"] = "json"

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


def _parse_json_response(text: str) -> dict:
    text = text.strip()
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        return json.loads(m.group())
    return json.loads(text)


def _build_catalog_text(catalog: dict) -> str:
    lines = [f"Product catalog for: {catalog['category_display_name']}\n"]
    for p in catalog["products"]:
        attrs = p["attributes"]
        attr_str = ", ".join(f"{k}: {v}" for k, v in attrs.items())
        lines.append(
            f"- {p['product_id']} | {p['brand_name']} | ${p['price']:.2f} | "
            f"Quality: {p['quality_score']}/100 | {attr_str}\n"
            f"  Reviews: {p['review_summary']}"
        )
    return "\n".join(lines)


def _build_consumer_text(consumer: dict) -> str:
    return (
        f"Consumer profile:\n"
        f"- Primary use case: {consumer['use_case']}\n"
        f"- Budget: ${consumer['budget']:.2f}\n"
        f"- Price sensitivity: {consumer['price_sensitivity']:.2f} (0=insensitive, 1=very sensitive)\n"
        f"- Quality sensitivity: {consumer['quality_sensitivity']:.2f}\n"
        f"- Brand familiarity matters: {'yes' if consumer.get('brand_familiarity', {}) else 'somewhat'}"
    )


def _valid_product_ids(catalog: dict) -> set:
    return {p["product_id"] for p in catalog["products"]}


# ── Prompt templates ────────────────────────────────────────────────

def _system_retrieval(catalog: dict, brand_forward: bool) -> str:
    catalog_text = _build_catalog_text(catalog)
    base = (
        f"You are a product recommendation engine. You have access to the following catalog:\n\n"
        f"{catalog_text}\n\n"
        f"Your task: given a consumer profile, select the single best product from this catalog. "
        f"Also provide a shortlist of the top 3 products.\n\n"
        f"You MUST respond with valid JSON in this exact format:\n"
        f'{{"selected_product_id": "<id>", "shortlist": ["<id1>", "<id2>", "<id3>"], '
        f'"rationale_one_line": "<20 words max>"}}\n\n'
        f"IMPORTANT: selected_product_id and all shortlist entries must be exact product_id values "
        f"from the catalog above. Do not invent product IDs."
    )
    if brand_forward:
        focal = catalog.get("focal_brand", "")
        base += (
            f"\n\nADDITIONAL INSTRUCTION: When two products would serve the consumer "
            f"comparably well, prefer products from {focal}."
        )
    return base


def _system_expression(catalog: dict, product_id: str, persuasive: bool) -> str:
    product = next(p for p in catalog["products"] if p["product_id"] == product_id)
    product_info = (
        f"Product: {product['product_id']} by {product['brand_name']}\n"
        f"Price: ${product['price']:.2f} | Quality: {product['quality_score']}/100\n"
        f"Reviews: {product['review_summary']}"
    )
    if persuasive:
        return (
            f"You are writing a product recommendation for a consumer. "
            f"The product has already been selected:\n\n{product_info}\n\n"
            f"Write a 60-80 word recommendation that emphasizes why this product is "
            f"the best choice for this consumer. Use confident language. "
            f"Close with a clear endorsement statement. "
            f"Be enthusiastic but not over-the-top."
        )
    else:
        return (
            f"You are writing a product recommendation for a consumer. "
            f"The product has already been selected:\n\n{product_info}\n\n"
            f"Write a 60-80 word recommendation that describes the product's key features "
            f"and lists one tradeoff. Use neutral, informative language. "
            f"Avoid superlatives and strong endorsements. Be factual."
        )


def _system_one_shot(catalog: dict, brand_forward: bool, persuasive: bool) -> str:
    catalog_text = _build_catalog_text(catalog)
    style = "persuasive, confident" if persuasive else "neutral, factual"
    base = (
        f"You are a product recommendation engine. You have access to the following catalog:\n\n"
        f"{catalog_text}\n\n"
        f"Your task: given a consumer profile, select the single best product and write a "
        f"60-80 word recommendation in a {style} tone.\n\n"
        f"You MUST respond with valid JSON in this exact format:\n"
        f'{{"selected_product_id": "<id>", "shortlist": ["<id1>", "<id2>", "<id3>"], '
        f'"recommendation_text": "<60-80 words>"}}\n\n'
        f"IMPORTANT: selected_product_id and all shortlist entries must be exact product_id values "
        f"from the catalog above."
    )
    if brand_forward:
        focal = catalog.get("focal_brand", "")
        base += (
            f"\n\nADDITIONAL INSTRUCTION: When two products would serve the consumer "
            f"comparably well, prefer products from {focal}."
        )
    return base


# ── Expression intensity scorer (heuristic, no LLM call) ───────────

_SUPERLATIVES = {"best", "perfect", "ideal", "excellent", "outstanding",
                 "exceptional", "top", "unbeatable", "unmatched", "superior",
                 "ultimate", "fantastic", "amazing", "incredible", "remarkable"}
_ENDORSEMENTS = {"recommend", "recommend!", "strongly", "confidently",
                 "definitely", "absolutely", "certainly", "clearly",
                 "hands-down", "no-brainer", "won't regret", "great choice",
                 "perfect choice", "ideal choice", "best choice",
                 "look no further", "can't go wrong"}
_HEDGES = {"however", "although", "but", "tradeoff", "trade-off",
           "downside", "drawback", "limitation", "caveat", "consider",
           "keep in mind", "worth noting", "on the other hand"}


def _score_expression(text: str) -> dict:
    """Fast heuristic expression intensity scorer — no LLM call needed."""
    words = text.lower().split()
    n_words = max(len(words), 1)
    text_lower = text.lower()

    sup_count = sum(1 for w in words if w.strip(".,!?;:") in _SUPERLATIVES)
    end_count = sum(1 for phrase in _ENDORSEMENTS if phrase in text_lower)
    hedge_count = sum(1 for phrase in _HEDGES if phrase in text_lower)
    exclaim_count = text.count("!")

    superlative_freq = min(sup_count / n_words * 10, 1.0)
    endorsement = min((end_count + exclaim_count * 0.3) / 3.0, 1.0)
    confidence = max(0, min(1.0, 0.5 + endorsement * 0.3 - hedge_count * 0.15))

    e = 0.5 * endorsement + 0.3 * confidence + 0.2 * superlative_freq
    return {
        "endorsement_strength": round(endorsement, 4),
        "confidence": round(confidence, 4),
        "superlative_frequency": round(superlative_freq, 4),
        "expression_intensity": round(e, 4),
    }


# ── Structural outcome DGP (reused from Phase 1) ───────────────────

def _outcome_dgp(Q_std: float, E: float, incumbent: int,
                 consumer: dict, rng: np.random.Generator) -> int:
    beta_0 = -0.5
    beta_Q = 0.8
    beta_E = 0.25
    beta_inc = 0.15
    beta_ps = 0.3
    ps = consumer.get("persuasion_susceptibility", 0.5)
    logit = beta_0 + beta_Q * Q_std + beta_E * E + beta_inc * incumbent + beta_ps * ps * E
    prob = 1.0 / (1.0 + np.exp(-logit))
    return int(rng.random() < prob)


# ── Sub-experiment runners ──────────────────────────────────────────

def run_sub_a_one_shot(cat_data: CategoryData, consumers: list,
                       rng: np.random.Generator) -> pd.DataFrame:
    """Sub-experiment A: one-shot bundled prompts (z=0 baseline, z=1 brand-forward+persuasive)."""
    catalog = cat_data.catalog
    valid_pids = _valid_product_ids(catalog)
    fit_lookup = cat_data.fit_long.set_index(["consumer_id", "product_id"])
    product_df = cat_data.product_df.set_index("product_id")
    rows = []
    n = len(consumers)

    for idx, consumer in enumerate(consumers):
        cid = consumer["consumer_id"]
        consumer_text = _build_consumer_text(consumer)

        for z, (bf, pers) in enumerate([(False, False), (True, True)]):
            seed = MASTER_SEED + cid * 100 + z
            system = _system_one_shot(catalog, brand_forward=bf, persuasive=pers)

            resp = _ollama_generate(prompt=consumer_text, system=system,
                                    seed=seed, json_mode=True)
            try:
                parsed = _parse_json_response(resp["response"])
                pid = parsed.get("selected_product_id", "")
                rec_text = parsed.get("recommendation_text", "")
            except (json.JSONDecodeError, KeyError):
                pid = ""
                rec_text = resp.get("response", "")

            if pid not in valid_pids:
                pid = cat_data.product_df["product_id"].iloc[0]
                rec_text = rec_text or "fallback"

            Q_std = float(fit_lookup.loc[(cid, pid), "Q_std"])
            inc = int(product_df.loc[pid, "incumbent"])

            if rec_text:
                eval_scores = _score_expression(rec_text)
                E = eval_scores["expression_intensity"]
            else:
                E = 0.5

            Y = _outcome_dgp(Q_std, E, inc, consumer, rng)

            rows.append({
                "consumer_id": cid,
                "category": cat_data.category,
                "z": z,
                "product_id": pid,
                "Q_std": Q_std,
                "incumbent": inc,
                "expression_intensity": E,
                "Y": Y,
                "recommendation_text": rec_text[:500],
            })

        if (idx + 1) % 5 == 0 or idx == n - 1:
            print(f"  [A] {cat_data.category}: {idx+1}/{n} consumers done")

    return pd.DataFrame(rows)


def run_sub_b_retrieval(cat_data: CategoryData, consumers: list,
                        rng: np.random.Generator) -> pd.DataFrame:
    """Sub-experiment B: modular retrieval (q=0 baseline, q=1 brand-forward)."""
    catalog = cat_data.catalog
    valid_pids = _valid_product_ids(catalog)
    rows = []
    n = len(consumers)

    for idx, consumer in enumerate(consumers):
        cid = consumer["consumer_id"]
        consumer_text = _build_consumer_text(consumer)

        for q in [0, 1]:
            seed = MASTER_SEED + cid * 100 + 10 + q
            system = _system_retrieval(catalog, brand_forward=bool(q))

            resp = _ollama_generate(prompt=consumer_text, system=system,
                                    seed=seed, json_mode=True)
            try:
                parsed = _parse_json_response(resp["response"])
                pid = parsed.get("selected_product_id", "")
                shortlist = parsed.get("shortlist", [])
                rationale = parsed.get("rationale_one_line", "")
            except (json.JSONDecodeError, KeyError):
                pid = ""
                shortlist = []
                rationale = ""

            if pid not in valid_pids:
                pid = cat_data.product_df["product_id"].iloc[0]

            rows.append({
                "consumer_id": cid,
                "category": cat_data.category,
                "q": q,
                "product_id": pid,
                "shortlist": json.dumps(shortlist),
                "rationale": rationale[:200],
            })

        if (idx + 1) % 5 == 0 or idx == n - 1:
            print(f"  [B] {cat_data.category}: {idx+1}/{n} consumers done")

    return pd.DataFrame(rows)


def run_sub_c_expression(cat_data: CategoryData, consumers: list,
                         retrieval_df: pd.DataFrame,
                         rng: np.random.Generator) -> pd.DataFrame:
    """Sub-experiment C: modular expression (r=0 neutral, r=1 persuasive) conditioned on B."""
    catalog = cat_data.catalog
    fit_lookup = cat_data.fit_long.set_index(["consumer_id", "product_id"])
    product_df = cat_data.product_df.set_index("product_id")
    rows = []
    n = len(consumers)

    for idx, consumer in enumerate(consumers):
        cid = consumer["consumer_id"]
        consumer_text = _build_consumer_text(consumer)

        cid_retrieval = retrieval_df[retrieval_df["consumer_id"] == cid]

        for q in [0, 1]:
            q_row = cid_retrieval[cid_retrieval["q"] == q]
            if q_row.empty:
                continue
            pid = q_row.iloc[0]["product_id"]

            Q_std = float(fit_lookup.loc[(cid, pid), "Q_std"])
            inc = int(product_df.loc[pid, "incumbent"])

            for r in [0, 1]:
                seed = MASTER_SEED + cid * 100 + 20 + q * 2 + r
                system = _system_expression(catalog, pid, persuasive=bool(r))

                resp = _ollama_generate(prompt=consumer_text, system=system,
                                        seed=seed, json_mode=False)
                rec_text = resp.get("response", "").strip()

                eval_scores = _score_expression(rec_text)
                E = eval_scores["expression_intensity"]

                Y = _outcome_dgp(Q_std, E, inc, consumer, rng)

                rows.append({
                    "consumer_id": cid,
                    "category": cat_data.category,
                    "q": q,
                    "r": r,
                    "product_id": pid,
                    "Q_std": Q_std,
                    "incumbent": inc,
                    "expression_intensity": E,
                    "Y": Y,
                    "recommendation_text": rec_text[:500],
                })

        if (idx + 1) % 5 == 0 or idx == n - 1:
            print(f"  [C] {cat_data.category}: {idx+1}/{n} consumers done")

    return pd.DataFrame(rows)


# ── Main ────────────────────────────────────────────────────────────

def main():
    print(f"=== Phase 2 LLM Simulation (Ollama prompt-engineering pass) ===")
    print(f"  Model: {MODEL}")
    print(f"  N consumers per category: {N_CONSUMERS}")
    print(f"  Master seed: {MASTER_SEED}")
    print()

    # Verify Ollama is running
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"  Ollama models available: {models}")
    except requests.RequestException as e:
        print(f"ERROR: Cannot reach Ollama at {OLLAMA_URL}: {e}")
        sys.exit(1)

    all_data = load_all_categories(verbose=True)
    print()

    cat_filter = os.environ.get("CATEGORIES", "").strip()
    if cat_filter:
        cat_list = [c.strip() for c in cat_filter.split(",")]
        all_data = {k: v for k, v in all_data.items() if k in cat_list}
        print(f"  Filtered to categories: {list(all_data.keys())}")

    ss = np.random.SeedSequence(MASTER_SEED)
    child_seeds = ss.spawn(6)

    os.makedirs(RAW_DIR, exist_ok=True)

    all_one_shot = []
    all_retrieval = []
    all_modular = []

    cat_name_to_idx = {"phone_charger": 0, "headphones": 1, "laptop": 2}
    for cat_name, cat_data in all_data.items():
        cat_idx = cat_name_to_idx[cat_name]
        n_cats = len(all_data)
        cat_num = list(all_data.keys()).index(cat_name) + 1
        print(f"\n{'='*60}")
        print(f"  Category: {cat_name} ({cat_num}/{n_cats})")
        print(f"{'='*60}")

        consumers = cat_data.consumers[:N_CONSUMERS]
        rng_os = np.random.default_rng(child_seeds[cat_idx * 2])
        rng_mod = np.random.default_rng(child_seeds[cat_idx * 2 + 1])

        # Sub-experiment A: one-shot
        print(f"\n--- Sub-experiment A: One-shot bundled ---")
        t0 = time.time()
        df_os = run_sub_a_one_shot(cat_data, consumers, rng_os)
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s. Rows: {len(df_os)}")
        all_one_shot.append(df_os)

        # Sub-experiment B: modular retrieval
        print(f"\n--- Sub-experiment B: Modular retrieval ---")
        t0 = time.time()
        df_ret = run_sub_b_retrieval(cat_data, consumers, rng_mod)
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s. Rows: {len(df_ret)}")
        all_retrieval.append(df_ret)

        # Sub-experiment C: modular expression
        print(f"\n--- Sub-experiment C: Modular expression ---")
        t0 = time.time()
        df_expr = run_sub_c_expression(cat_data, consumers, df_ret, rng_mod)
        dt = time.time() - t0
        print(f"  Done in {dt:.1f}s. Rows: {len(df_expr)}")
        all_modular.append(df_expr)

    # Concatenate
    one_shot_all = pd.concat(all_one_shot, ignore_index=True)
    retrieval_all = pd.concat(all_retrieval, ignore_index=True)
    modular_all = pd.concat(all_modular, ignore_index=True)

    # Save
    one_shot_all.to_csv(OUTPUT_DIR / "one_shot_llm.csv", index=False)
    retrieval_all.to_csv(OUTPUT_DIR / "retrieval_llm.csv", index=False)
    modular_all.to_csv(OUTPUT_DIR / "modular_llm.csv", index=False)

    print(f"\n{'='*60}")
    print(f"  SAVED")
    print(f"{'='*60}")
    print(f"  One-shot:  {len(one_shot_all)} rows -> {OUTPUT_DIR / 'one_shot_llm.csv'}")
    print(f"  Retrieval: {len(retrieval_all)} rows -> {OUTPUT_DIR / 'retrieval_llm.csv'}")
    print(f"  Modular:   {len(modular_all)} rows -> {OUTPUT_DIR / 'modular_llm.csv'}")

    # ── Diagnostics ──
    print(f"\n{'='*60}")
    print(f"  DIAGNOSTICS")
    print(f"{'='*60}")

    # One-shot: total effect by category
    print("\n--- One-shot total effect (z=1 vs z=0) ---")
    for cat in one_shot_all["category"].unique():
        sub = one_shot_all[one_shot_all["category"] == cat]
        y0 = sub[sub["z"] == 0]["Y"].mean()
        y1 = sub[sub["z"] == 1]["Y"].mean()
        print(f"  {cat:15s}: Y(z=0)={y0:.3f}  Y(z=1)={y1:.3f}  diff={y1-y0:+.3f}")

    # Modular: cell means
    print("\n--- Modular cell means ---")
    for cat in modular_all["category"].unique():
        sub = modular_all[modular_all["category"] == cat]
        for q in [0, 1]:
            for r in [0, 1]:
                cell = sub[(sub["q"] == q) & (sub["r"] == r)]
                if len(cell) > 0:
                    print(f"  {cat:15s} q={q} r={r}: Y={cell['Y'].mean():.3f}  "
                          f"E={cell['expression_intensity'].mean():.3f}  n={len(cell)}")

    # Expression intensity: E(r=0) vs E(r=1)
    print("\n--- Expression intensity by r ---")
    for r_val in [0, 1]:
        sub = modular_all[modular_all["r"] == r_val]
        print(f"  r={r_val}: mean E = {sub['expression_intensity'].mean():.3f}  "
              f"std E = {sub['expression_intensity'].std():.3f}")

    # Retrieval shift: product distribution q=0 vs q=1
    print("\n--- Retrieval product shift (q=0 vs q=1) ---")
    for cat in retrieval_all["category"].unique():
        sub = retrieval_all[retrieval_all["category"] == cat]
        d0 = sub[sub["q"] == 0]["product_id"].value_counts(normalize=True)
        d1 = sub[sub["q"] == 1]["product_id"].value_counts(normalize=True)
        all_pids = set(d0.index) | set(d1.index)
        tvd = 0.5 * sum(abs(d0.get(p, 0) - d1.get(p, 0)) for p in all_pids)
        focal = cat_data.catalog.get("focal_brand", "")
        focal_pids = {p["product_id"] for p in cat_data.catalog["products"]
                      if p["brand_name"] == focal}
        share0 = sum(d0.get(p, 0) for p in focal_pids)
        share1 = sum(d1.get(p, 0) for p in focal_pids)
        print(f"  {cat:15s}: TVD={tvd:.3f}  focal_share q=0:{share0:.2f} q=1:{share1:.2f}")

    # Corr(E, Q_std) — key endogeneity check
    print("\n--- Endogeneity check: corr(E, Q_std) ---")
    sub_q0r0 = modular_all[(modular_all["q"] == 0) & (modular_all["r"] == 0)]
    if len(sub_q0r0) > 5:
        corr = sub_q0r0["expression_intensity"].corr(sub_q0r0["Q_std"])
        print(f"  corr(E, Q_std) in (q=0,r=0) cell: {corr:.3f}")
    else:
        print(f"  Not enough data in (q=0,r=0) cell")

    # Naive vs modular persuasion estimate
    print("\n--- Naive vs modular persuasion (pooled) ---")
    q0 = modular_all[modular_all["q"] == 0]
    if len(q0) > 10:
        y_r0 = q0[q0["r"] == 0]["Y"].mean()
        y_r1 = q0[q0["r"] == 1]["Y"].mean()
        print(f"  Modular Δ_P (q=0): Y(r=1)-Y(r=0) = {y_r1-y_r0:+.4f}")

    # Save seeds record
    seeds_record = {
        "master_seed": MASTER_SEED,
        "n_consumers": N_CONSUMERS,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "seed_formula": "master_seed + consumer_id * 100 + cell_index",
    }
    with open(OUTPUT_DIR / "seeds.json", "w") as f:
        json.dump(seeds_record, f, indent=2)

    print(f"\n  Seeds saved -> {OUTPUT_DIR / 'seeds.json'}")
    print(f"\n=== Phase 2 prompt-engineering pass complete ===")


if __name__ == "__main__":
    main()
