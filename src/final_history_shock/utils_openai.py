"""Shared OpenAI API utilities for final history-shock simulation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.3-chat-latest")
EVAL_MODEL = os.getenv("OPENAI_EVAL_MODEL", "gpt-5.3-chat-latest")

DATA_DIR = ROOT / "data" / "final_history_shock"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def gpt_call(prompt: str, *, model: str | None = None, temperature: float | None = None,
             system: str | None = None) -> dict:
    """Single GPT call with retry. Returns dict with output_text, usage, etc."""
    m = model or MODEL
    inp = []
    if system:
        inp.append({"role": "system", "content": system})
    inp.append({"role": "user", "content": prompt})

    kwargs = {"model": m, "input": inp}
    if temperature is not None:
        kwargs["temperature"] = temperature

    t0 = time.time()
    resp = client.responses.create(**kwargs)
    elapsed = time.time() - t0

    return {
        "output_text": resp.output_text,
        "usage": resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else str(resp.usage),
        "model": m,
        "elapsed_s": round(elapsed, 2),
        "timestamp": time.time(),
    }


def gpt_json_call(prompt: str, *, model: str | None = None, temperature: float | None = None,
                  system: str | None = None, max_retries: int = 3) -> tuple[dict, dict]:
    """GPT call that parses JSON from output. Returns (parsed_json, raw_meta).
    Retries on parse failure up to max_retries."""
    for attempt in range(max_retries):
        raw = gpt_call(prompt, model=model, temperature=temperature, system=system)
        text = raw["output_text"].strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            parsed = json.loads(text)
            raw["parse_attempt"] = attempt + 1
            return parsed, raw
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                prompt_retry = (
                    f"Your previous response was not valid JSON. Please respond with ONLY valid JSON, "
                    f"no markdown fences or extra text.\n\nOriginal request:\n{prompt}"
                )
                prompt = prompt_retry
            else:
                raw["parse_attempt"] = attempt + 1
                raw["parse_error"] = f"Failed to parse JSON after {max_retries} attempts"
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
