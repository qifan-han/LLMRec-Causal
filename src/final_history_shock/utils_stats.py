"""Shared statistical utilities: Bradley-Terry, bootstrap, decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def fit_bradley_terry(win_matrix: np.ndarray, tie_matrix: np.ndarray | None = None) -> np.ndarray:
    """Fit Bradley-Terry model. win_matrix[i,j] = number of times i beats j.
    Ties split as half-win each. Returns theta with theta[0]=0."""
    K = win_matrix.shape[0]
    if tie_matrix is not None:
        W = win_matrix + 0.5 * tie_matrix
    else:
        W = win_matrix.astype(float)

    def _nll(params):
        theta = np.concatenate([[0.0], params])
        ll = 0.0
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                n_ij = W[i, j] + W[j, i]
                if n_ij == 0:
                    continue
                p_ij = 1.0 / (1.0 + np.exp(theta[j] - theta[i]))
                p_ij = np.clip(p_ij, 1e-10, 1 - 1e-10)
                ll += W[i, j] * np.log(p_ij) + W[j, i] * np.log(1 - p_ij)
        return -ll

    x0 = np.zeros(K - 1)
    res = minimize(_nll, x0, method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-8})
    return np.concatenate([[0.0], res.x])


def decompose_from_utilities(theta: np.ndarray) -> dict:
    """Given theta[00, 10, 01, 11], compute retrieval/expression/total/interaction."""
    return {
        "retrieval": theta[1] - theta[0],
        "expression": theta[2] - theta[0],
        "total": theta[3] - theta[0],
        "interaction": theta[3] - theta[1] - theta[2] + theta[0],
    }


def cluster_bootstrap_bt(pairwise_df: pd.DataFrame, cluster_col: str = "cluster_id",
                         cells: list[str] | None = None, B: int = 1000,
                         seed: int = 42, winner_col: str = "overall_winner_cell") -> pd.DataFrame:
    """Cluster bootstrap of Bradley-Terry decomposition.
    Returns DataFrame with columns: component, mean, se, ci_lo, ci_hi, p_positive."""
    if cells is None:
        cells = ["00", "10", "01", "11"]
    cell_to_idx = {c: i for i, c in enumerate(cells)}
    K = len(cells)

    rng = np.random.default_rng(seed)
    pairwise_df = pairwise_df.copy()
    for col in ["cell_i", "cell_j"]:
        if col in pairwise_df.columns:
            pairwise_df[col] = pairwise_df[col].astype(str).str.zfill(2)
    cluster_ids = pairwise_df[cluster_col].unique()
    n_clusters = len(cluster_ids)

    boot_results = []
    for b in range(B):
        boot_clusters = rng.choice(cluster_ids, size=n_clusters, replace=True)
        boot_df = pd.concat([pairwise_df[pairwise_df[cluster_col] == c] for c in boot_clusters],
                            ignore_index=True)

        win_mat = np.zeros((K, K))
        tie_mat = np.zeros((K, K))
        for _, row in boot_df.iterrows():
            cell_i_val = str(row["cell_i"])
            cell_j_val = str(row["cell_j"])
            ci = cell_to_idx.get(cell_i_val)
            cj = cell_to_idx.get(cell_j_val)
            if ci is None or cj is None:
                continue
            winner = str(row.get(winner_col, row.get("choice_winner", "")))
            if winner == cell_i_val:
                win_mat[ci, cj] += 1
            elif winner == cell_j_val:
                win_mat[cj, ci] += 1
            else:
                tie_mat[ci, cj] += 1
                tie_mat[cj, ci] += 1

        theta = fit_bradley_terry(win_mat, tie_mat)
        decomp = decompose_from_utilities(theta)
        boot_results.append(decomp)

    boot_df = pd.DataFrame(boot_results)
    summary_rows = []
    for comp in ["retrieval", "expression", "total", "interaction"]:
        vals = boot_df[comp].values
        summary_rows.append({
            "component": comp,
            "mean": np.mean(vals),
            "se": np.std(vals),
            "ci_lo": np.percentile(vals, 2.5),
            "ci_hi": np.percentile(vals, 97.5),
            "p_positive": np.mean(vals > 0),
        })

    return pd.DataFrame(summary_rows)


def simple_pairwise_utility(pairwise_df: pd.DataFrame,
                            cluster_col: str = "cluster_id",
                            cells: list[str] | None = None) -> pd.DataFrame:
    """Compute per-cluster simple pairwise score (win=1, tie=0.5, loss=0).
    Returns DataFrame with cluster_id and U_00, U_10, U_01, U_11."""
    if cells is None:
        cells = ["00", "10", "01", "11"]

    rows = []
    for cid, grp in pairwise_df.groupby(cluster_col):
        scores = {c: [] for c in cells}
        for _, row in grp.iterrows():
            ci, cj = row["cell_i"], row["cell_j"]
            winner = row.get("overall_winner_cell", row.get("choice_winner", ""))
            if winner == ci:
                scores.get(ci, []).append(1.0)
                scores.get(cj, []).append(0.0)
            elif winner == cj:
                scores.get(ci, []).append(0.0)
                scores.get(cj, []).append(1.0)
            else:
                scores.get(ci, []).append(0.5)
                scores.get(cj, []).append(0.5)

        row_data = {cluster_col: cid}
        for c in cells:
            row_data[f"U_{c}"] = np.mean(scores[c]) if scores[c] else 0.5
        rows.append(row_data)

    df = pd.DataFrame(rows)
    u00_mean = df["U_00"].mean()
    for c in cells:
        df[f"U_{c}"] = df[f"U_{c}"] - u00_mean
    return df
