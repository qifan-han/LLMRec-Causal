"""Tonight-MVP simulation: one-shot bundled policy + modular 2x2 design.

Outputs (both CSVs go in data/simulated/):
  - one_shot_mvp.csv    : columns include z, selected_product_id, Q_selected,
                          Q_selected_std, incumbent, focal_brand,
                          expression_intensity, Y_prob, Y
  - modular_mvp.csv     : same schema but with (q, r) instead of z

The DGP is `naive_regression_failure`: latent fit Q drives both selection and
endogenous expression intensity (lambda_fit > 0), so naive regressions of Y on
realized expression are biased upward.

Bundle asymmetry: the one-shot policy z couples brand-forward retrieval AND
stronger expression in a single backend prompt — the researcher observing
(z, J, E, Y) cannot decompose lift into retrieval vs expression channels. The
modular design randomizes q and r independently and the 2x2 cell means reveal
the components. This operationalizes Proposition 2 of the v2 theory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from core.data_io import (  # noqa: E402
    CategoryData,
    load_all_categories,
)

OUTPUT_DIR = ROOT / "data" / "simulated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# DGP parameter defaults
# -----------------------------------------------------------------------------

DEFAULTS = dict(
    # Retrieval kernel
    retrieval_mode="brand_forward",  # "brand_forward" or "fit_aware"
    a_Q=1.0,
    a_incumbent=0.20,
    a_focal=0.50,           # used only when retrieval_mode == "brand_forward"
    delta_fit_aware=0.6,    # used only when retrieval_mode == "fit_aware" (q=1 adds this to a_Q)
    sigma_R=0.35,
    # Expression kernel
    e0=0.0,
    tau_backend=0.5,  # one-shot only — bundle channel
    tau_R=0.5,        # modular only — randomized expression effect
    lambda_fit=1.0,
    lambda_inc=0.2,
    sigma_E=0.5,
    # Outcome (naive_regression_failure DGP)
    beta_0=-0.8,
    beta_Q=0.7,
    beta_E=0.5,
    beta_QE=0.0,
    beta_inc=0.25,
    beta_incE=-0.15,
)

MASTER_SEED = 20260513


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# -----------------------------------------------------------------------------
# Retrieval: pick a product for every consumer in every policy condition
# -----------------------------------------------------------------------------

def _retrieval_scores(
    fit_long: pd.DataFrame,
    rng: np.random.Generator,
    params: dict,
    q: int,
) -> pd.DataFrame:
    """Compute retrieval scores for every (consumer, product) pair.

    Two retrieval modes are supported via `params["retrieval_mode"]`:
      - "brand_forward": q=1 adds a_focal * focal_brand_j (brand-substitutes-for-fit policy).
      - "fit_aware":     q=1 boosts the Q-weighting by delta_fit_aware (fit-aware policy).

    Returns a long frame with column `score`.
    """
    n = len(fit_long)
    epsilon = rng.normal(0.0, 1.0, size=n)
    mode = params.get("retrieval_mode", "brand_forward")

    if mode == "brand_forward":
        score = (
            params["a_Q"] * fit_long["Q_std"].values
            + params["a_incumbent"] * fit_long["incumbent"].values
            + (params["a_focal"] * fit_long["focal_brand"].values if q == 1 else 0.0)
            + params["sigma_R"] * epsilon
        )
    elif mode == "fit_aware":
        a_Q_eff = params["a_Q"] + (params["delta_fit_aware"] if q == 1 else 0.0)
        score = (
            a_Q_eff * fit_long["Q_std"].values
            + params["a_incumbent"] * fit_long["incumbent"].values
            + params["sigma_R"] * epsilon
        )
    else:
        raise ValueError(f"Unknown retrieval_mode: {mode!r}")

    out = fit_long[["consumer_id", "product_id", "Q", "Q_std", "incumbent", "focal_brand"]].copy()
    out["score"] = score
    return out


def _argmax_select(scored: pd.DataFrame) -> pd.DataFrame:
    """For each consumer, keep the highest-scoring product."""
    idx = scored.groupby("consumer_id")["score"].idxmax()
    sel = scored.loc[idx, ["consumer_id", "product_id", "Q", "Q_std", "incumbent", "focal_brand"]]
    sel = sel.reset_index(drop=True)
    sel.rename(columns={"Q": "Q_selected", "Q_std": "Q_selected_std"}, inplace=True)
    return sel


def simulate_recommendation_for_category(
    data: CategoryData,
    rng_retrieval: np.random.Generator,
    params: dict,
) -> pd.DataFrame:
    """Returns one long frame with all six policy conditions (z={0,1}, q,r∈{0,1}²).

    We share retrieval noise across conditions that use the same kernel: e.g.,
    one-shot z=0 and modular (q=0, *) both use the baseline-retrieval kernel
    on the same epsilon draw, so they select identical products. Likewise
    one-shot z=1 and modular (q=1, *) share brand-forward retrieval noise.
    This lets us define `total_true` from the same retrieval realization on
    the modular cells.
    """
    # Baseline retrieval (z=0, q=0)
    scored_base = _retrieval_scores(data.fit_long, rng_retrieval, params, q=0)
    sel_base = _argmax_select(scored_base)
    sel_base["category"] = data.category

    # Alternative retrieval (z=1, q=1): brand-forward or fit-aware depending on mode
    scored_alt = _retrieval_scores(data.fit_long, rng_retrieval, params, q=1)
    sel_alt = _argmax_select(scored_alt)
    sel_alt["category"] = data.category

    return sel_base, sel_alt


# -----------------------------------------------------------------------------
# Expression: compute intensity for every policy cell
# -----------------------------------------------------------------------------

def _expression_eta(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=n)


def _expression(
    Q_std: np.ndarray,
    incumbent: np.ndarray,
    policy_coef: float,
    eta: np.ndarray,
    params: dict,
    lambda_fit: float | None = None,
) -> np.ndarray:
    """Generic expression equation.

    For one-shot: policy_coef = tau_backend * z
    For modular:  policy_coef = tau_R * r
    """
    lf = params["lambda_fit"] if lambda_fit is None else lambda_fit
    return (
        params["e0"]
        + policy_coef
        + lf * Q_std
        + params["lambda_inc"] * incumbent
        + params["sigma_E"] * eta
    )


# -----------------------------------------------------------------------------
# Outcomes
# -----------------------------------------------------------------------------

def _outcome_prob(Q_std, E, incumbent, params: dict):
    U = (
        params["beta_0"]
        + params["beta_Q"] * Q_std
        + params["beta_E"] * E
        + params["beta_QE"] * Q_std * E
        + params["beta_inc"] * incumbent
        + params["beta_incE"] * incumbent * E
    )
    return _sigmoid(U)


# -----------------------------------------------------------------------------
# Top-level orchestration
# -----------------------------------------------------------------------------

def _build_oneshot_frame(
    sel_base: pd.DataFrame,
    sel_focal: pd.DataFrame,
    rng_expression: np.random.Generator,
    rng_outcome: np.random.Generator,
    params: dict,
    lambda_fit: float | None = None,
) -> pd.DataFrame:
    """Stacks z=0 and z=1 rows for a single category."""
    parts = []
    for z, sel in [(0, sel_base), (1, sel_focal)]:
        n = len(sel)
        eta = _expression_eta(n, rng_expression)
        E = _expression(
            sel["Q_selected_std"].values,
            sel["incumbent"].values,
            policy_coef=params["tau_backend"] * z,
            eta=eta,
            params=params,
            lambda_fit=lambda_fit,
        )
        p = _outcome_prob(sel["Q_selected_std"].values, E, sel["incumbent"].values, params)
        y = rng_outcome.binomial(1, p)
        df = sel.copy()
        df["z"] = z
        df["expression_intensity"] = E
        df["Y_prob"] = p
        df["Y"] = y
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def _build_modular_frame(
    sel_base: pd.DataFrame,
    sel_focal: pd.DataFrame,
    rng_expression: np.random.Generator,
    rng_outcome: np.random.Generator,
    params: dict,
    lambda_fit: float | None = None,
) -> pd.DataFrame:
    """Stacks the four modular cells (q, r) ∈ {0,1}² for a single category."""
    parts = []
    for (q, r), sel in [
        ((0, 0), sel_base),
        ((0, 1), sel_base),
        ((1, 0), sel_focal),
        ((1, 1), sel_focal),
    ]:
        n = len(sel)
        eta = _expression_eta(n, rng_expression)
        E = _expression(
            sel["Q_selected_std"].values,
            sel["incumbent"].values,
            policy_coef=params["tau_R"] * r,
            eta=eta,
            params=params,
            lambda_fit=lambda_fit,
        )
        p = _outcome_prob(sel["Q_selected_std"].values, E, sel["incumbent"].values, params)
        y = rng_outcome.binomial(1, p)
        df = sel.copy()
        df["q"] = q
        df["r"] = r
        df["expression_intensity"] = E
        df["Y_prob"] = p
        df["Y"] = y
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def simulate_main_run(
    master_seed: int = MASTER_SEED,
    params: dict | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run the main simulation across all categories.

    Returns (one_shot_df, modular_df, diagnostics).
    """
    if params is None:
        params = DEFAULTS.copy()

    seq = np.random.SeedSequence(master_seed)
    seeds_retrieval, seeds_expr_os, seeds_out_os, seeds_expr_mod, seeds_out_mod = seq.spawn(5)
    rng_R = np.random.default_rng(seeds_retrieval)
    rng_E_os = np.random.default_rng(seeds_expr_os)
    rng_Y_os = np.random.default_rng(seeds_out_os)
    rng_E_mod = np.random.default_rng(seeds_expr_mod)
    rng_Y_mod = np.random.default_rng(seeds_out_mod)

    cats = load_all_categories(verbose=verbose)

    one_shot_parts = []
    modular_parts = []
    diagnostics_rows = []

    for cat, data in cats.items():
        sel_base, sel_focal = simulate_recommendation_for_category(data, rng_R, params)
        os_df = _build_oneshot_frame(sel_base, sel_focal, rng_E_os, rng_Y_os, params)
        mod_df = _build_modular_frame(sel_base, sel_focal, rng_E_mod, rng_Y_mod, params)
        one_shot_parts.append(os_df)
        modular_parts.append(mod_df)

        # Per-category diagnostics
        diag = {
            "category": cat,
            "n": len(data.consumers),
            "share_focal_baseline": float(sel_base["focal_brand"].mean()),
            "share_focal_brand_forward": float(sel_focal["focal_brand"].mean()),
            "tvd_q0_q1_products": _tvd_product_shares(sel_base, sel_focal),
            "mean_E_q0_r0": float(mod_df.loc[(mod_df["q"] == 0) & (mod_df["r"] == 0), "expression_intensity"].mean()),
            "mean_E_q0_r1": float(mod_df.loc[(mod_df["q"] == 0) & (mod_df["r"] == 1), "expression_intensity"].mean()),
            "mean_Y_q0_r0": float(mod_df.loc[(mod_df["q"] == 0) & (mod_df["r"] == 0), "Y"].mean()),
            "mean_Y_q1_r1": float(mod_df.loc[(mod_df["q"] == 1) & (mod_df["r"] == 1), "Y"].mean()),
            "corr_E_Qstd": float(np.corrcoef(mod_df["expression_intensity"], mod_df["Q_selected_std"])[0, 1]),
        }
        diagnostics_rows.append(diag)

    one_shot = pd.concat(one_shot_parts, ignore_index=True)
    modular = pd.concat(modular_parts, ignore_index=True)

    diagnostics = {
        "params": params,
        "master_seed": master_seed,
        "by_category": diagnostics_rows,
        "pooled": {
            "n_one_shot": len(one_shot),
            "n_modular": len(modular),
            "mean_Y_z0": float(one_shot.loc[one_shot["z"] == 0, "Y"].mean()),
            "mean_Y_z1": float(one_shot.loc[one_shot["z"] == 1, "Y"].mean()),
            "mean_Y_modular_pooled": float(modular["Y"].mean()),
            "corr_E_Qstd_modular": float(
                np.corrcoef(modular["expression_intensity"], modular["Q_selected_std"])[0, 1]
            ),
        },
    }

    return one_shot, modular, diagnostics


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def _tvd_product_shares(sel0: pd.DataFrame, sel1: pd.DataFrame) -> float:
    """Total-variation distance between selected-product distributions."""
    s0 = sel0["product_id"].value_counts(normalize=True)
    s1 = sel1["product_id"].value_counts(normalize=True)
    all_pids = sorted(set(s0.index) | set(s1.index))
    s0 = s0.reindex(all_pids, fill_value=0.0)
    s1 = s1.reindex(all_pids, fill_value=0.0)
    return float(0.5 * np.abs(s0.values - s1.values).sum())


def validate_run(one_shot: pd.DataFrame, modular: pd.DataFrame, diagnostics: dict) -> list[str]:
    """Run the stopping-rule checks. Returns list of violations (empty if OK)."""
    issues = []

    # Cell balance in pooled modular
    cell_counts = modular.groupby(["q", "r"]).size()
    if cell_counts.nunique() != 1:
        issues.append(f"Modular cells unbalanced: {cell_counts.to_dict()}")

    # Outcome rates within [0.05, 0.95]
    pooled_rate = float(modular["Y"].mean())
    if not (0.05 <= pooled_rate <= 0.95):
        issues.append(f"Pooled outcome rate {pooled_rate:.3f} outside [0.05, 0.95]")

    # Expression variance
    if modular["expression_intensity"].std(ddof=0) < 0.05:
        issues.append("Expression intensity has near-zero variance")

    # q=1 vs q=0 retrieval shift
    for diag in diagnostics["by_category"]:
        if diag["tvd_q0_q1_products"] < 0.02:
            issues.append(
                f"[{diag['category']}] q=1 barely changes product shares (TVD={diag['tvd_q0_q1_products']:.3f})"
            )

    # r=1 vs r=0 expression shift
    for diag in diagnostics["by_category"]:
        delta_E = diag["mean_E_q0_r1"] - diag["mean_E_q0_r0"]
        if delta_E < 0.05:
            issues.append(
                f"[{diag['category']}] r=1 does not increase mean expression "
                f"(Δ={delta_E:.3f})"
            )

    # corr(E, Q_std) should be material at lambda_fit > 0.5 (default 1.0)
    corr = diagnostics["pooled"]["corr_E_Qstd_modular"]
    if corr < 0.10:
        issues.append(
            f"corr(E, Q_selected_std) is {corr:.3f}; expected positive endogeneity at lambda_fit > 0"
        )

    # Selected product IDs must be in their catalog
    pids_modular = set(modular["product_id"].unique())
    pids_oneshot = set(one_shot["product_id"].unique())
    pids_used = pids_modular | pids_oneshot
    # Sanity check: every used pid contains its category as prefix (charger_/headphones_/laptop_)
    for pid in pids_used:
        if not any(pid.startswith(pref) for pref in ("charger_", "headphones_", "laptop_")):
            issues.append(f"Suspicious product_id: {pid}")
    return issues


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    one_shot, modular, diagnostics = simulate_main_run(verbose=True)

    print("\n=== Diagnostics (per category) ===")
    print(pd.DataFrame(diagnostics["by_category"]).to_string(index=False))
    print("\n=== Pooled ===")
    for k, v in diagnostics["pooled"].items():
        print(f"  {k}: {v}")

    issues = validate_run(one_shot, modular, diagnostics)
    if issues:
        print("\n=== VALIDATION ISSUES ===")
        for x in issues:
            print(f"  - {x}")
        raise SystemExit(1)
    else:
        print("\nAll validation checks passed.")

    one_shot_path = OUTPUT_DIR / "one_shot_mvp.csv"
    modular_path = OUTPUT_DIR / "modular_mvp.csv"
    one_shot.to_csv(one_shot_path, index=False)
    modular.to_csv(modular_path, index=False)
    print(f"\nSaved {len(one_shot):,} one-shot rows -> {one_shot_path}")
    print(f"Saved {len(modular):,} modular rows  -> {modular_path}")


if __name__ == "__main__":
    main()
