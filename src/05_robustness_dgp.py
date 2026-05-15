"""Robustness DGP: retrieval-dominant world (fit-aware retrieval policy).

This is the single robustness check requested for the QME working paper. The
purpose is to demonstrate that the modular design recovers the truth in a DGP
where retrieval is the dominant channel — not just in the persuasion-dominant
naive_regression_failure DGP that drives the main results.

Differences from the main DGP:
- retrieval_mode = "fit_aware": q=1 raises the Q-weighting in the retrieval
  kernel by delta_fit_aware, so the brand-forward channel is turned off and
  the alternative retrieval policy is a genuinely better one (it selects
  products with higher fit on average).
- beta_E is slightly lower so persuasion does not dominate retrieval. We
  keep beta_E positive so persuasion is still detectable.

Outputs saved to results/tables/robustness_*.csv. No new figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from importlib import import_module  # noqa: E402

sim = import_module("02_simulate_core_mvp")
est = import_module("03_estimate_core_mvp")

OUT_DIR = ROOT / "results" / "tables"
DATA_DIR = ROOT / "data" / "simulated"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

ROBUST_PARAMS = sim.DEFAULTS.copy()
ROBUST_PARAMS.update(
    {
        "retrieval_mode": "fit_aware",
        "delta_fit_aware": 0.8,   # q=1 weighs fit nearly twice as heavily as q=0
        "a_focal": 0.0,           # turn off the brand-forward channel entirely
        "beta_E": 0.35,           # let retrieval be the leading channel
    }
)


def main():
    print("=== Robustness DGP: retrieval-dominant (fit-aware retrieval) ===")
    print(f"  retrieval_mode    = {ROBUST_PARAMS['retrieval_mode']}")
    print(f"  delta_fit_aware   = {ROBUST_PARAMS['delta_fit_aware']}")
    print(f"  a_focal           = {ROBUST_PARAMS['a_focal']}")
    print(f"  beta_E            = {ROBUST_PARAMS['beta_E']}")
    print()

    one_shot, modular, diagnostics = sim.simulate_main_run(
        master_seed=20260513,
        params=ROBUST_PARAMS,
        verbose=False,
    )

    print("\n=== Diagnostics (per category) ===")
    print(pd.DataFrame(diagnostics["by_category"]).to_string(index=False))
    print("\n=== Pooled ===")
    for k, v in diagnostics["pooled"].items():
        print(f"  {k}: {v}")

    issues = sim.validate_run(one_shot, modular, diagnostics)
    if issues:
        print("\n=== VALIDATION ISSUES ===")
        for x in issues:
            print(f"  - {x}")
        raise SystemExit(1)
    print("\nAll validation checks passed.")

    one_shot.to_csv(DATA_DIR / "one_shot_robustness.csv", index=False)
    modular.to_csv(DATA_DIR / "modular_robustness.csv", index=False)

    t1 = est.table_one_shot_total_effect(one_shot)
    t2 = est.table_modular_decomposition(modular)
    t3 = est.table_naive_vs_oracle(modular)

    p1 = OUT_DIR / "robustness_one_shot_total_effect.csv"
    p2 = OUT_DIR / "robustness_decomposition.csv"
    p3 = OUT_DIR / "robustness_naive_vs_oracle.csv"
    t1.to_csv(p1, index=False)
    t2.to_csv(p2, index=False)
    t3.to_csv(p3, index=False)

    print("\n=== Robustness Table 1: one-shot total effect ===")
    print(t1.to_string(index=False))
    print("\n=== Robustness Table 2: modular decomposition ===")
    print(t2.to_string(index=False))
    print("\n=== Robustness Table 3: naive vs oracle vs modular (persuasion) ===")
    print(t3.to_string(index=False))


if __name__ == "__main__":
    main()
