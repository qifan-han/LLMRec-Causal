"""00 — OpenAI API test.

Runs 5 simple calls to verify the API key and model work.

Output:
  data/final_history_shock/api_test_results.json

Usage:
  python 00_api_test.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-5.3-chat-latest")
OUT_DIR = ROOT / "data" / "final_history_shock"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TESTS = [
    'Return exactly this JSON and nothing else: {"ok": true}',
    "In one sentence, explain why over-ear headphones differ from earbuds.",
    (
        "A student on a tight budget is choosing between Product A ($29, decent sound, wired) "
        "and Product B ($89, noise cancelling, wireless). Recommend one. "
        'Return JSON: {"product_id": "A" or "B", "reason": "..."}'
    ),
    (
        "You are comparing two product recommendations for the same consumer.\n"
        "Recommendation A: 'The Sony WH-1000XM5 offers excellent noise cancellation for your commute.'\n"
        "Recommendation B: 'The Bose QC45 is a solid choice with long battery life.'\n"
        'Which is better for a daily commuter? Return JSON: {"winner": "A", "B", or "tie", "reason": "..."}'
    ),
    'Return JSON: {"test": "usage", "model_check": true}',
]


def run_tests():
    rows = []
    for i, prompt in enumerate(TESTS):
        print(f"  Test {i + 1}/{len(TESTS)}...", end=" ", flush=True)
        t0 = time.time()
        try:
            resp = client.responses.create(model=MODEL, input=prompt)
            output_text = resp.output_text
            usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else str(resp.usage)
            elapsed = time.time() - t0
            rows.append({
                "i": i,
                "prompt": prompt,
                "output_text": output_text,
                "usage": usage,
                "model": MODEL,
                "elapsed_s": round(elapsed, 2),
                "timestamp": time.time(),
                "error": None,
            })
            print(f"OK ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - t0
            rows.append({
                "i": i,
                "prompt": prompt,
                "output_text": None,
                "usage": None,
                "model": MODEL,
                "elapsed_s": round(elapsed, 2),
                "timestamp": time.time(),
                "error": str(e),
            })
            print(f"FAIL: {e}")

    out_path = OUT_DIR / "api_test_results.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    n_ok = sum(1 for r in rows if r["error"] is None)
    n_fail = len(rows) - n_ok
    print(f"\nAPI test complete: {n_ok}/{len(rows)} passed, {n_fail} failed")
    print(f"Results → {out_path}")

    if n_fail > 0:
        print("\nFailed tests:")
        for r in rows:
            if r["error"]:
                print(f"  Test {r['i']}: {r['error']}")
        sys.exit(1)

    print("\nSample responses:")
    for r in rows[:3]:
        print(f"  Test {r['i']}: {r['output_text'][:120]}...")


if __name__ == "__main__":
    run_tests()
