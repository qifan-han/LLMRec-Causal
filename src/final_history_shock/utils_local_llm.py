"""Shared local LLM (Ollama) utilities for final history-shock simulation."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

SUPPLY_MODEL = "qwen2.5:14b"

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "final_history_shock"


def ollama_generate(prompt: str, *, model: str = SUPPLY_MODEL,
                    temperature: float = 0.7, seed: int | None = None,
                    timeout: int = 600) -> dict:
    """Call Ollama and return dict with response, model, elapsed_s."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": temperature},
    }
    if seed is not None:
        payload["options"]["seed"] = seed

    t0 = time.time()
    payload_file = Path("/tmp/ollama_payload.json")
    payload_file.write_text(json.dumps(payload))
    result = subprocess.run(
        ["curl", "-s", "-m", str(timeout),
         "http://localhost:11434/api/generate",
         "-d", f"@{payload_file}"],
        capture_output=True, text=True, timeout=timeout + 10,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(f"Ollama call failed: {result.stderr}")

    resp = json.loads(result.stdout)
    return {
        "response": resp.get("response", ""),
        "model": model,
        "elapsed_s": round(elapsed, 2),
        "eval_count": resp.get("eval_count"),
        "timestamp": time.time(),
    }


def ollama_json_call(prompt: str, *, model: str = SUPPLY_MODEL,
                     temperature: float = 0.7, seed: int | None = None,
                     max_retries: int = 3) -> tuple[dict, dict]:
    """Ollama call that parses JSON from output. Returns (parsed_json, raw_meta)."""
    for attempt in range(max_retries):
        raw = ollama_generate(prompt, model=model, temperature=temperature, seed=seed)
        text = raw["response"].strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            parsed = json.loads(text)
            raw["parse_attempt"] = attempt + 1
            return parsed, raw
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                prompt = (
                    f"Your previous response was not valid JSON. Respond ONLY with valid JSON, "
                    f"no markdown, no extra text.\n\nOriginal request:\n{prompt}"
                )
            else:
                raw["parse_attempt"] = attempt + 1
                raw["parse_error"] = f"Failed after {max_retries} attempts"
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
