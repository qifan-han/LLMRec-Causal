"""Master pipeline runner: executes all steps with error recovery.

Saves results at each step. On API error, waits and retries.
Resume-safe: checks for existing outputs before re-running steps.

Usage:
  python run_full_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "final_history_shock"


def run_step(script: str, description: str, timeout: int = 3600,
             extra_args: list[str] | None = None) -> bool:
    """Run a pipeline step. Returns True on success."""
    print(f"\n{'=' * 60}")
    print(f"  STEP: {description}")
    print(f"  Script: {script}")
    print(f"{'=' * 60}\n")

    cmd = [sys.executable, str(SCRIPT_DIR / script)]
    if extra_args:
        cmd.extend(extra_args)

    for attempt in range(3):
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, cwd=str(ROOT),
            )
            print(result.stdout)
            if result.stderr:
                print(f"STDERR: {result.stderr[-500:]}")

            if result.returncode == 0:
                return True

            print(f"  FAILED (exit code {result.returncode})")
            if attempt < 2:
                wait = 120 * (attempt + 1)
                print(f"  Waiting {wait}s before retry...")
                time.sleep(wait)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after {timeout}s")
            if attempt < 2:
                print(f"  Retrying with --resume...")
                extra_args = extra_args or []
                if "--resume" not in extra_args:
                    extra_args.append("--resume")
        except Exception as e:
            print(f"  ERROR: {e}")
            if attempt < 2:
                time.sleep(60)

    print(f"  STEP FAILED after 3 attempts: {description}")
    return False


def check_exists(path: str) -> bool:
    return (DATA_DIR / path).exists()


def main():
    t_start = time.time()
    results = {}

    # Step 1: Catalogs (already done if file exists)
    if check_exists("catalogs/headphones_catalog.csv"):
        print("Catalogs already exist, skipping step 01")
        results["01_catalogs"] = True
    else:
        results["01_catalogs"] = run_step("01_build_or_collect_catalogs.py", "Build catalogs")

    # Step 2: GPT exemplars
    if check_exists("gpt_exemplars/gpt_recommendation_exemplars.jsonl"):
        from utils_openai import load_jsonl
        n = len(load_jsonl(DATA_DIR / "gpt_exemplars" / "gpt_recommendation_exemplars.jsonl"))
        if n >= 180:
            print(f"Exemplars already complete ({n} rows), skipping step 02")
            results["02_exemplars"] = True
        else:
            results["02_exemplars"] = run_step("02_generate_gpt_exemplars.py",
                                               f"GPT exemplars ({n}/180 done)", timeout=7200)
    else:
        results["02_exemplars"] = run_step("02_generate_gpt_exemplars.py",
                                           "GPT exemplars (180 calls)", timeout=7200)

    # Step 3: Personas
    if check_exists("personas/headphones_personas.json"):
        print("Personas already exist, skipping step 03")
        results["03_personas"] = True
    else:
        results["03_personas"] = run_step("03_generate_personas.py",
                                          "Generate personas (12 GPT calls)")

    # Step 4: Validate personas
    results["04_validate"] = run_step("04_validate_personas.py", "Validate personas")

    # Step 5: Historical DGP
    if check_exists("history_dgp/headphones_history_qualitative.json"):
        print("History DGP already exists, skipping step 05")
        results["05_history"] = True
    else:
        results["05_history"] = run_step("05_generate_historical_dgp.py", "Historical DGP")

    # Step 6: Build local prompts
    results["06_prompts"] = run_step("06_build_local_prompts.py", "Build local prompts with few-shot")

    # Step 7: Smoke run
    if check_exists("local_supply/smoke_supply_rows.csv"):
        print("Smoke supply already exists, skipping step 07")
        results["07_smoke"] = True
    else:
        results["07_smoke"] = run_step("07_smoke_run_local_supply.py",
                                       "Smoke run (6 clusters)", timeout=1800)

    # Step 8: Full supply
    if check_exists("local_supply/final_supply_rows.csv"):
        print("Full supply already exists, skipping step 08")
        results["08_supply"] = True
    else:
        results["08_supply"] = run_step("08_run_local_supply_full.py",
                                        "Full supply (60 clusters, 240 packages)",
                                        timeout=7200, extra_args=["--resume"])

    if not results.get("08_supply"):
        print("\nFull supply failed. Cannot continue to evaluation.")
        _print_summary(results, t_start)
        return

    # Step 9: Leakage audit
    results["09_leakage"] = run_step("09_leakage_audit_and_regen.py", "Leakage audit")

    # Step 10: GPT absolute evaluation (secondary priority)
    results["10_absolute"] = run_step("10_gpt_absolute_eval.py",
                                      "GPT absolute evaluation (480 calls)",
                                      timeout=7200, extra_args=["--resume"])

    # Step 11: GPT pairwise evaluation (primary outcome)
    results["11_pairwise"] = run_step("11_gpt_pairwise_eval.py",
                                      "GPT pairwise evaluation (720 calls)",
                                      timeout=7200, extra_args=["--resume"])

    if not results.get("11_pairwise"):
        print("\nPairwise evaluation failed. Running analysis with whatever data exists.")

    # Step 12: Analysis
    results["12_analysis"] = run_step("12_analyze_decomposition.py", "Decomposition analysis")

    # Step 13: Report
    results["13_report"] = run_step("13_write_summary_report.py", "Write summary report")

    _print_summary(results, t_start)


def _print_summary(results: dict, t_start: float):
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE — {elapsed / 60:.0f} minutes")
    print(f"{'=' * 60}")
    for step, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {step}: {status}")

    n_ok = sum(1 for v in results.values() if v)
    n_total = len(results)
    print(f"\n  {n_ok}/{n_total} steps succeeded")

    if all(results.values()):
        print("\n  ALL STEPS PASSED. Check results at:")
        print(f"  {DATA_DIR / 'analysis'}")
        print(f"  {DATA_DIR / 'reports' / 'final_simulation_report.md'}")


if __name__ == "__main__":
    main()
