"""Shared utilities for the history-shock simulation pipeline."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"
HIST_DATA = ROOT / "data" / "history_shock"
HIST_RESULTS = ROOT / "results" / "history_shock"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
SUPPLY_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
DEMAND_MODEL = os.environ.get("DEMAND_MODEL", "gemma2:9b")
SUPPLY_TEMPERATURE = 0.0
DEMAND_TEMPERATURE = 0.0
MASTER_SEED = 20260515
MAX_RETRIES = 3

CATEGORIES = ["headphones", "laptop", "phone_charger"]

try:
    import requests
except ImportError:
    sys.exit("requests not installed — run: pip install requests")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_catalog(category: str) -> dict:
    with open(DATA_DIR / "catalogs" / f"{category}.json") as f:
        return json.load(f)


def load_consumers(category: str) -> list[dict]:
    with open(DATA_DIR / "consumers" / f"{category}.json") as f:
        return json.load(f)


def load_fit_scores(category: str) -> pd.DataFrame:
    wide = pd.read_csv(DATA_DIR / "fit_scores" / f"{category}.csv")
    long = wide.melt(id_vars="consumer_id", var_name="product_id", value_name="Q")
    mu, sd = long["Q"].mean(), long["Q"].std(ddof=0)
    long["Q_std"] = (long["Q"] - mu) / sd
    return long


def get_product_by_id(pid: str, catalog: dict) -> dict | None:
    for p in catalog["products"]:
        if p["product_id"] == pid:
            return p
    return None


def validate_product_id(pid: str, catalog: dict) -> bool:
    return pid in {p["product_id"] for p in catalog["products"]}


def assign_segment(consumer: dict, category: str, all_consumers: list[dict]) -> str:
    budgets = sorted(c["budget"] for c in all_consumers)
    t1 = budgets[len(budgets) // 3]
    t2 = budgets[2 * len(budgets) // 3]
    b = consumer["budget"]
    if b <= t1:
        return "budget_low"
    elif b <= t2:
        return "budget_mid"
    return "budget_high"


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def ollama_generate(system: str, user: str, seed: int,
                    json_mode: bool = False, model: str | None = None,
                    num_predict: int = 512, temperature: float | None = None) -> dict:
    model = model or SUPPLY_MODEL
    temp = temperature if temperature is not None else SUPPLY_TEMPERATURE
    payload = {
        "model": model,
        "system": system,
        "prompt": user,
        "stream": False,
        "options": {
            "temperature": temp,
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


def parse_json_response(text: str) -> dict | None:
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
# Seed, cache, manifest
# ---------------------------------------------------------------------------

def compute_seed(consumer_id: int, cell_code: str, stage: str) -> int:
    cell_offsets = {"00": 0, "10": 100, "01": 200, "11": 300}
    stage_offsets = {"selector": 0, "writer": 50}
    return MASTER_SEED + consumer_id * 1000 + cell_offsets[cell_code] + stage_offsets[stage]


def save_raw(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_raw(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def append_to_jsonl(path: Path, entry: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_jsonl_cache(path: Path) -> dict:
    cache = {}
    if not path.exists():
        return cache
    for line in path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            if entry.get("parsed"):
                cache[entry["row_id"]] = entry
        except json.JSONDecodeError:
            continue
    return cache
