"""GPU supply generation for history-shock 2x2 design via vLLM.

Batched inference on Qwen2.5-32B-Instruct-AWQ.  Three phases:
  Phase 1: batch all 120 retrieval prompts (60 generic + 60 history)
  Phase 2: parse retrieval, batch all 240 expression prompts
  Phase 3: validate, save CSV + JSONL cache

Same prompts/seeds/schema as the Ollama pipeline (08_run_local_supply_full.py)
but ~10-20x faster on GPU.

Usage:
  export SUPPLY_MODEL=/root/autodl-tmp/Qwen2.5-32B-Instruct-AWQ
  python supply_history_shock.py --smoke   # 3 personas
  python supply_history_shock.py --full    # all 60 personas
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

from prompts import (
    build_generic_retrieval_prompt, build_history_retrieval_prompt,
    build_generic_expression_prompt, build_history_expression_prompt,
)

SUPPLY_MODEL = os.environ.get("SUPPLY_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ")
TENSOR_PARALLEL = int(os.environ.get("TENSOR_PARALLEL", "1"))
MAX_MODEL_LEN = 32768
TEMPERATURE = 0.7
MASTER_SEED = 20260515

MAX_TOKENS_RETRIEVAL = 512
MAX_TOKENS_EXPRESSION = 1024

DATA_ROOT = Path(__file__).resolve().parent / "data"
OUT_DIR = Path(os.environ.get("DATA_DIR", os.path.expanduser("~/llmrec_results")))
CACHE_PATH = OUT_DIR / "supply_cache.jsonl"
OUT_CSV = OUT_DIR / "final_supply_rows.csv"

CATEGORIES = ["headphones"]

LEAKAGE_PATTERNS = [
    r"\b\d+(\.\d+)?%", r"\b\d+\s*(out of|/)\s*\d+",
    r"\b(conversion|satisfaction|return)\s+rate", r"\bNPS\b",
    r"\branked?\s*#?\d+", r"\b(top|bottom)\s*\d+",
    r"\b\d{1,3}(,\d{3})+\s*(units?|sales?|customers?|buyers?)",
    r"\bsample\s+size", r"\bn\s*=\s*\d+",
]
_LEAKAGE_RX = [re.compile(p, re.IGNORECASE) for p in LEAKAGE_PATTERNS]


def has_leakage(text: str) -> bool:
    return any(rx.search(text) for rx in _LEAKAGE_RX)


def load_inputs(category: str):
    catalog_df = pd.read_csv(DATA_ROOT / "catalogs" / f"{category}_catalog.csv")
    catalog = catalog_df.to_dict("records")
    with open(DATA_ROOT / "personas" / f"{category}_personas.json") as f:
        personas = json.load(f)
    with open(DATA_ROOT / "history_dgp" / f"{category}_history_qualitative.json") as f:
        qual_history = json.load(f)
    few_shot = {}
    fs_path = DATA_ROOT / "gpt_exemplars" / "final_few_shot_prompts.json"
    if fs_path.exists():
        with open(fs_path) as f:
            few_shot = json.load(f)
    return catalog, catalog_df, personas, qual_history, few_shot


def _guess_segment(persona: dict) -> str:
    desc = (persona.get("one_paragraph_description", "") +
            persona.get("primary_use_case", "") +
            persona.get("purchase_context", "")).lower()
    keywords = {
        "student": "budget_student", "budget": "budget_student",
        "commut": "commuter", "travel": "frequent_traveler",
        "remote": "remote_worker", "work from home": "remote_worker",
        "audiophile": "audiophile", "music quality": "audiophile",
        "gym": "gym_user", "workout": "gym_user", "fitness": "gym_user",
        "gam": "gamer", "casual": "casual_listener",
    }
    for kw, seg in keywords.items():
        if kw in desc:
            return seg
    return "casual_listener"


def _get_history_summary(qual_history: list[dict], persona: dict, product_id: str) -> str:
    segment = _guess_segment(persona)
    relevant = [q for q in qual_history
                if q["product_id"] == product_id and q["segment"] == segment]
    if not relevant:
        relevant = [q for q in qual_history if q["product_id"] == product_id]
    if not relevant:
        return "Limited historical data available for this product-consumer combination."
    return relevant[0]["summary"]


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
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


def print_env_diagnostics():
    """Print package versions for debugging."""
    print("=" * 60)
    print("  Environment Diagnostics")
    print("=" * 60)
    try:
        import vllm
        print(f"  vllm:         {vllm.__version__}")
    except Exception as e:
        print(f"  vllm:         IMPORT FAILED — {e}")
    try:
        import transformers
        print(f"  transformers: {transformers.__version__}")
    except Exception:
        print(f"  transformers: not installed")
    try:
        import torch
        print(f"  torch:        {torch.__version__}")
        print(f"  CUDA:         {torch.version.cuda or 'N/A'}")
        print(f"  GPU:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    except Exception:
        print(f"  torch:        not installed")
    print(f"  model:        {SUPPLY_MODEL}")
    print(f"  model exists: {os.path.isdir(SUPPLY_MODEL)}")
    print(f"  enforce_eager: True (skip torch.compile)")
    print("=" * 60)


def load_model_and_tokenizer():
    from transformers import AutoTokenizer

    print(f"[MODEL] Loading {SUPPLY_MODEL} ...")
    from vllm import LLM
    llm = LLM(
        model=SUPPLY_MODEL,
        quantization="awq",
        tensor_parallel_size=TENSOR_PARALLEL,
        dtype="auto",
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=False,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(SUPPLY_MODEL)
    print("[MODEL] Loaded.")
    return llm, tokenizer


def apply_chat_template(tokenizer, user_str: str) -> str:
    """Format prompt using Qwen2.5 chat template."""
    messages = [{"role": "user", "content": user_str}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def batch_generate(llm, tokenizer, prompts: list[str], seeds: list[int],
                   max_tokens: int) -> list[str]:
    from vllm import SamplingParams

    formatted = [apply_chat_template(tokenizer, p) for p in prompts]
    params = [SamplingParams(temperature=TEMPERATURE, max_tokens=max_tokens, seed=s)
              for s in seeds]
    outputs = llm.generate(formatted, params)
    return [o.outputs[0].text.strip() for o in outputs]


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="3 personas")
    group.add_argument("--full", action="store_true", help="All 60 personas")
    args = parser.parse_args()

    print_env_diagnostics()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for category in CATEGORIES:
        catalog, catalog_df, personas, qual_history, few_shot = load_inputs(category)
        valid_pids = {p["product_id"] for p in catalog}

        if args.smoke:
            personas = personas[:3]

        n = len(personas)
        fs_generic = few_shot.get("generic", "")
        fs_history = few_shot.get("history_aware", "")

        print(f"\n=== {category}: {n} personas ===")

        llm, tokenizer = load_model_and_tokenizer()

        # --- Phase 1: Retrieval ---
        print(f"\nPhase 1: Building {n * 2} retrieval prompts...")
        ret_generic_prompts = []
        ret_history_prompts = []
        ret_generic_seeds = []
        ret_history_seeds = []

        for pi, persona in enumerate(personas):
            seed_base = MASTER_SEED + pi * 100

            ret_generic_prompts.append(
                build_generic_retrieval_prompt(persona, catalog, fs_generic))
            ret_generic_seeds.append(seed_base + 1)

            segment = _guess_segment(persona)
            segment_history = [q for q in qual_history if q["segment"] == segment]
            if not segment_history:
                segment_history = qual_history
            segment_history.sort(key=lambda q: q.get("affinity_score", 0), reverse=True)
            hist_summary_all = "\n".join(
                f"- {q['product_id']}: {q['summary']}" for q in segment_history)
            ret_history_prompts.append(
                build_history_retrieval_prompt(persona, catalog, hist_summary_all, fs_history))
            ret_history_seeds.append(seed_base + 2)

        print(f"  Generating {n} generic retrieval responses...")
        t0 = time.time()
        ret_generic_texts = batch_generate(
            llm, tokenizer, ret_generic_prompts, ret_generic_seeds, MAX_TOKENS_RETRIEVAL)
        print(f"  Done in {time.time() - t0:.0f}s")

        print(f"  Generating {n} history retrieval responses...")
        t0 = time.time()
        ret_history_texts = batch_generate(
            llm, tokenizer, ret_history_prompts, ret_history_seeds, MAX_TOKENS_RETRIEVAL)
        print(f"  Done in {time.time() - t0:.0f}s")

        pids_generic = []
        pids_history = []
        for pi in range(n):
            parsed_g = _parse_json(ret_generic_texts[pi])
            pid_g = parsed_g.get("selected_product_id", "") if parsed_g else ""
            if pid_g not in valid_pids:
                pid_g = catalog[0]["product_id"]
            pids_generic.append(pid_g)

            parsed_h = _parse_json(ret_history_texts[pi])
            pid_h = parsed_h.get("selected_product_id", "") if parsed_h else ""
            if pid_h not in valid_pids:
                pid_h = catalog[0]["product_id"]
            pids_history.append(pid_h)

        retrieval_changed = sum(1 for g, h in zip(pids_generic, pids_history) if g != h)
        print(f"  Retrieval changed: {retrieval_changed}/{n}")

        # --- Phase 2: Expression ---
        print(f"\nPhase 2: Building {n * 4} expression prompts...")
        expr_prompts = []
        expr_seeds = []
        expr_meta = []

        for pi, persona in enumerate(personas):
            seed_base = MASTER_SEED + pi * 100
            pid_g = pids_generic[pi]
            pid_h = pids_history[pi]
            product_g = next(p for p in catalog if p["product_id"] == pid_g)
            product_h = next(p for p in catalog if p["product_id"] == pid_h)

            cells = [
                ("00", pid_g, product_g, "generic", "generic"),
                ("10", pid_h, product_h, "history", "generic"),
                ("01", pid_g, product_g, "generic", "history"),
                ("11", pid_h, product_h, "history", "history"),
            ]

            for cell, pid, product, ret_cond, exp_cond in cells:
                hist_prod = _get_history_summary(qual_history, persona, pid)
                if exp_cond == "generic":
                    prompt = build_generic_expression_prompt(persona, product, fs_generic)
                else:
                    prompt = build_history_expression_prompt(
                        persona, product, hist_prod, fs_history)

                seed_offset = {"00": 10, "10": 20, "01": 30, "11": 40}[cell]
                expr_prompts.append(prompt)
                expr_seeds.append(seed_base + seed_offset)
                expr_meta.append({
                    "persona_idx": pi, "cell": cell, "pid": pid,
                    "ret_cond": ret_cond, "exp_cond": exp_cond,
                    "retrieval_changed": pid_g != pid_h,
                })

        print(f"  Generating {len(expr_prompts)} expression responses...")
        t0 = time.time()
        expr_texts = batch_generate(
            llm, tokenizer, expr_prompts, expr_seeds, MAX_TOKENS_EXPRESSION)
        print(f"  Done in {time.time() - t0:.0f}s")

        # --- Phase 3: Parse and save ---
        print(f"\nPhase 3: Parsing and saving...")
        all_rows = []
        n_parse_fail = 0
        n_leakage = 0

        for idx, (text, meta) in enumerate(zip(expr_texts, expr_meta)):
            pi = meta["persona_idx"]
            persona = personas[pi]
            parsed = _parse_json(text)

            if parsed and not parsed.get("_parse_failed"):
                rec_text = parsed.get("recommendation_text", "")
                tradeoff_text = parsed.get("tradeoff_text", "")
                persuasion_text = parsed.get("persuasion_text", "")
            else:
                rec_text = text
                tradeoff_text = ""
                persuasion_text = ""
                n_parse_fail += 1

            full_package = f"{rec_text}\n\n{tradeoff_text}" if tradeoff_text else rec_text
            leak = has_leakage(full_package)
            if leak:
                n_leakage += 1

            row = {
                "cluster_id": f"{category}_{pi:03d}",
                "category": category,
                "persona_id": persona.get("persona_id", f"{category}_{pi:03d}"),
                "cell": meta["cell"],
                "retrieval_condition": meta["ret_cond"],
                "expression_condition": meta["exp_cond"],
                "selected_product_id": meta["pid"],
                "recommendation_text": rec_text,
                "tradeoff_text": tradeoff_text,
                "persuasion_text": persuasion_text,
                "full_recommendation_package": full_package,
                "history_language_used": parsed.get("history_language_used", "none") if parsed else "none",
                "local_model": SUPPLY_MODEL,
                "word_count": len(full_package.split()),
                "parse_failed": parsed is None or parsed.get("_parse_failed", False),
                "retrieval_changed": meta["retrieval_changed"],
                "leakage_flag": leak,
            }
            all_rows.append(row)

        df = pd.DataFrame(all_rows)
        df.to_csv(OUT_CSV, index=False)

        with open(CACHE_PATH, "w") as f:
            for row in all_rows:
                f.write(json.dumps(row) + "\n")

        print(f"\n=== Supply Complete ===")
        print(f"  Total packages: {len(df)}")
        print(f"  Clusters: {df['cluster_id'].nunique()}")
        print(f"  Parse failures: {n_parse_fail}")
        print(f"  Leakage flagged: {n_leakage} ({n_leakage / len(df):.1%})")
        print(f"  Retrieval changed: {retrieval_changed}/{n}")
        print(f"  Saved → {OUT_CSV}")


if __name__ == "__main__":
    main()
