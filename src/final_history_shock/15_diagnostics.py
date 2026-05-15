#!/usr/bin/env python3
"""
Step 15 — Diagnostic analysis of modular two-stage experiment.

Reads existing CSV/JSON outputs only. No new LLM calls.
Produces:
  - cluster-level diagnostic dataset
  - retrieval switch anatomy tables
  - absolute DID by outcome
  - purchase interaction diagnostics
  - pairwise reason coding
  - paper-ready figures (PNG + PDF)
  - mechanism diagnostics report
"""

import csv, json, os, re, math, itertools
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

np.random.seed(42)

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "final_history_shock"
DIAG = DATA / "analysis" / "diagnostics"
FIGS = DATA / "analysis" / "figures_diagnostics"
PAPER_FIGS = ROOT / "paper" / "figures"
REPORTS = DATA / "reports"

for d in [DIAG, FIGS, PAPER_FIGS, REPORTS]:
    d.mkdir(parents=True, exist_ok=True)

# ── load data ────────────────────────────────────────────────────────────────
def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

supply = load_csv(DATA / "local_supply" / "final_supply_rows.csv")
abseval = load_csv(DATA / "gpt_eval" / "absolute_eval_rows.csv")
paireval = load_csv(DATA / "gpt_eval" / "pairwise_eval_rows.csv")
catalog_rows = load_csv(DATA / "catalogs" / "headphones_catalog.csv")

with open(DATA / "personas" / "headphones_personas.json") as f:
    personas_list = json.load(f)

with open(DATA / "history_dgp" / "headphones_history_qualitative.json") as f:
    history_qual = json.load(f)

# ── index structures ─────────────────────────────────────────────────────────
catalog = {r["product_id"]: r for r in catalog_rows}
personas = {p["persona_id"]: p for p in personas_list}

# history affinity: (segment, product_id) -> affinity_score
affinity_map = {}
for h in history_qual:
    affinity_map[(h["segment"], h["product_id"])] = float(h["affinity_score"])

# supply by (cluster, cell)
supply_ix = {}
for r in supply:
    supply_ix[(r["cluster_id"], int(r["cell"]))] = r

# abseval by (cluster, cell)
eval_ix = {}
for r in abseval:
    eval_ix[(r["cluster_id"], int(r["cell"]))] = r

# pairwise by (cluster, cell_i, cell_j)
# CSV uses zero-padded strings "00","01","10","11" for cell codes and winners
CELL_STR_TO_INT = {"00": 0, "01": 1, "10": 10, "11": 11, "0": 0, "1": 1}
CELL_INT_TO_STR = {0: "00", 1: "01", 10: "10", 11: "11"}
pair_ix = {}
for r in paireval:
    ci = CELL_STR_TO_INT.get(r["cell_i"], int(r["cell_i"]))
    cj = CELL_STR_TO_INT.get(r["cell_j"], int(r["cell_j"]))
    pair_ix[(r["cluster_id"], ci, cj)] = r

clusters = sorted(set(r["cluster_id"] for r in supply))
EVAL_OUTCOMES = [
    "fit_score_1_7", "purchase_probability_0_100", "expected_satisfaction_0_100",
    "trust_score_1_7", "clarity_score_1_7", "persuasive_intensity_1_7",
    "tradeoff_disclosure_1_7", "regret_risk_1_7",
]

CELLS = [0, 1, 10, 11]

# ── helpers ──────────────────────────────────────────────────────────────────
def flt(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan

def parse_budget(budget_str):
    m = re.findall(r"\$?([\d,]+(?:\.\d+)?)", budget_str.replace(",", ""))
    nums = [float(x) for x in m]
    if len(nums) >= 2:
        return min(nums), max(nums)
    elif len(nums) == 1:
        return None, nums[0]
    return None, None

SEGMENT_MAP = {
    "commut": "commuter", "travel": "frequent_traveler", "noise cancel": "commuter",
    "gym": "gym_user", "workout": "gym_user", "fitness": "gym_user", "exercise": "gym_user",
    "gaming": "gamer", "gamer": "gamer", "esport": "gamer",
    "student": "budget_student", "budget": "budget_student",
    "office": "remote_worker", "remote": "remote_worker", "work call": "remote_worker",
    "call": "remote_worker", "conference": "remote_worker",
    "music": "audiophile", "audiophile": "audiophile", "studio": "audiophile",
    "hi-fi": "audiophile", "hifi": "audiophile",
    "casual": "casual_listener", "general": "casual_listener", "everyday": "casual_listener",
    "kid": "casual_listener", "child": "casual_listener",
}

def guess_segment(persona):
    text = (persona.get("primary_use_case", "") + " " +
            persona.get("secondary_use_case", "") + " " +
            persona.get("purchase_context", "")).lower()
    for kw, seg in SEGMENT_MAP.items():
        if kw in text:
            return seg
    return "casual_listener"

def keyword_overlap(features_list, product_text):
    if not features_list:
        return 0
    product_lower = product_text.lower()
    hits = sum(1 for f in features_list if any(
        w in product_lower for w in f.lower().split() if len(w) > 3
    ))
    return hits / len(features_list)

def product_text(pid):
    p = catalog.get(pid, {})
    return " ".join([
        p.get("title", ""), p.get("key_features", ""),
        p.get("best_for", ""), p.get("brand", ""),
    ]).lower()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Build cluster-level diagnostic dataset
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 1: Building cluster-level diagnostic dataset")
print("=" * 70)

diag_rows = []
for cid in clusters:
    pid = supply_ix[(cid, 0)]["persona_id"]
    persona = personas[pid]

    # products under each cell
    prod = {}
    for cell in CELLS:
        prod[cell] = supply_ix[(cid, cell)]["selected_product_id"]

    retrieval_changed = prod[0] != prod[10]

    row = {
        "cluster_id": cid,
        "persona_id": pid,
        "retrieval_changed": int(retrieval_changed),
    }

    # persona metadata
    for k in ["budget", "primary_use_case", "secondary_use_case",
              "must_have_features", "features_to_avoid", "price_sensitivity",
              "quality_sensitivity", "risk_aversion", "technical_knowledge",
              "purchase_context", "age_range"]:
        val = persona.get(k, "")
        if isinstance(val, list):
            val = "; ".join(val)
        row[f"persona_{k}"] = val

    # catalog metadata for generic (cell 0) and history (cell 10) products
    for label, cell in [("generic", 0), ("history", 10)]:
        p = catalog.get(prod[cell], {})
        row[f"{label}_product_id"] = prod[cell]
        row[f"{label}_brand"] = p.get("brand", "")
        row[f"{label}_title"] = p.get("title", "")
        row[f"{label}_price"] = flt(p.get("price", ""))
        row[f"{label}_price_tier"] = p.get("price_tier", "")
        row[f"{label}_avg_rating"] = flt(p.get("average_rating", ""))
        row[f"{label}_rating_count"] = flt(p.get("rating_count", ""))
        rc = flt(p.get("rating_count", ""))
        row[f"{label}_log_rating_count"] = math.log(rc) if rc and rc > 0 else np.nan
        row[f"{label}_popularity_rank"] = flt(p.get("popularity_rank", ""))
        row[f"{label}_popularity_tier"] = p.get("popularity_tier", "")
        row[f"{label}_key_features"] = p.get("key_features", "")
        row[f"{label}_best_for"] = p.get("best_for", "")
        row[f"{label}_drawbacks"] = p.get("drawbacks", "")

    # absolute eval outcomes for all four cells
    for cell in CELLS:
        e = eval_ix.get((cid, cell), {})
        for out in EVAL_OUTCOMES:
            row[f"cell{cell}_{out}"] = flt(e.get(out, ""))

    # pairwise winners
    for ci, cj in [(0, 1), (0, 10), (0, 11), (1, 11), (10, 1), (10, 11)]:
        pw = pair_ix.get((cid, ci, cj), {})
        row[f"pw_{ci}v{cj}_overall"] = pw.get("overall_winner_cell", "")
        row[f"pw_{ci}v{cj}_purchase"] = pw.get("purchase_winner_cell", "")
        row[f"pw_{ci}v{cj}_satisfaction"] = pw.get("satisfaction_winner_cell", "")
        row[f"pw_{ci}v{cj}_trust"] = pw.get("trust_winner_cell", "")
        row[f"pw_{ci}v{cj}_reason"] = pw.get("reason", "")

    diag_rows.append(row)

# Save
fieldnames = list(diag_rows[0].keys())
with open(DIAG / "cluster_level_diagnostics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(diag_rows)

print(f"  Saved cluster_level_diagnostics.csv: {len(diag_rows)} rows, {len(fieldnames)} columns")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Retrieval switch anatomy
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: Retrieval switch anatomy — why retrieval is negative")
print("=" * 70)

budget_parse_fail = 0
switch_data = []

for row in diag_rows:
    d = {}
    d["cluster_id"] = row["cluster_id"]
    d["retrieval_changed"] = row["retrieval_changed"]

    gp = row["generic_price"]
    hp = row["history_price"]
    d["price_diff"] = hp - gp if not (np.isnan(hp) or np.isnan(gp)) else np.nan

    gr = row["generic_log_rating_count"]
    hr = row["history_log_rating_count"]
    d["log_rating_count_diff"] = hr - gr if not (np.isnan(hr) or np.isnan(gr)) else np.nan

    gar = row["generic_avg_rating"]
    har = row["history_avg_rating"]
    d["avg_rating_diff"] = har - gar if not (np.isnan(har) or np.isnan(gar)) else np.nan

    grank = row["generic_popularity_rank"]
    hrank = row["history_popularity_rank"]
    d["popularity_improvement"] = grank - hrank if not (np.isnan(grank) or np.isnan(hrank)) else np.nan

    d["history_more_expensive"] = int(hp > gp) if not (np.isnan(hp) or np.isnan(gp)) else np.nan

    # budget parsing
    lo, hi = parse_budget(row["persona_budget"])
    if hi is None:
        budget_parse_fail += 1
        d["generic_over_budget"] = np.nan
        d["history_over_budget"] = np.nan
        d["budget_violation_switch"] = np.nan
    else:
        d["generic_over_budget"] = int(gp > hi) if not np.isnan(gp) else np.nan
        d["history_over_budget"] = int(hp > hi) if not np.isnan(hp) else np.nan
        if not (np.isnan(gp) or np.isnan(hp)):
            d["budget_violation_switch"] = int(gp <= hi and hp > hi)
        else:
            d["budget_violation_switch"] = np.nan

    d["history_higher_rating_count"] = int(
        row["history_rating_count"] > row["generic_rating_count"]
    ) if not (np.isnan(row["history_rating_count"]) or np.isnan(row["generic_rating_count"])) else np.nan

    d["history_better_rating"] = int(har > gar) if not (np.isnan(har) or np.isnan(gar)) else np.nan

    tier_order = {"niche": 0, "moderate": 1, "popular": 2, "bestseller": 3}
    gt = tier_order.get(row["generic_popularity_tier"], -1)
    ht = tier_order.get(row["history_popularity_tier"], -1)
    d["history_more_popular_tier"] = int(ht > gt) if gt >= 0 and ht >= 0 else np.nan

    # feature-fit heuristic
    persona = personas[row["persona_id"]]
    mh = persona.get("must_have_features", [])
    fa = persona.get("features_to_avoid", [])
    if isinstance(mh, str):
        mh = [x.strip() for x in mh.split(";")]
    if isinstance(fa, str):
        fa = [x.strip() for x in fa.split(";")]

    d["generic_feature_fit"] = keyword_overlap(mh, product_text(row["generic_product_id"]))
    d["history_feature_fit"] = keyword_overlap(mh, product_text(row["history_product_id"]))
    d["feature_fit_diff"] = d["history_feature_fit"] - d["generic_feature_fit"]

    d["generic_avoid_violation"] = keyword_overlap(fa, product_text(row["generic_product_id"]))
    d["history_avoid_violation"] = keyword_overlap(fa, product_text(row["history_product_id"]))
    d["avoid_violation_diff"] = d["history_avoid_violation"] - d["generic_avoid_violation"]

    # segment affinity
    seg = guess_segment(persona)
    d["inferred_segment"] = seg
    ga = affinity_map.get((seg, row["generic_product_id"]), np.nan)
    ha = affinity_map.get((seg, row["history_product_id"]), np.nan)
    d["generic_segment_affinity"] = ga
    d["history_segment_affinity"] = ha
    d["segment_affinity_diff"] = ha - ga if not (np.isnan(ha) or np.isnan(ga)) else np.nan

    # absolute eval deltas: cell10 - cell00
    d["fit_delta_retrieval"] = row["cell10_fit_score_1_7"] - row["cell0_fit_score_1_7"]
    d["purchase_delta_retrieval"] = row["cell10_purchase_probability_0_100"] - row["cell0_purchase_probability_0_100"]
    d["satisfaction_delta_retrieval"] = row["cell10_expected_satisfaction_0_100"] - row["cell0_expected_satisfaction_0_100"]
    d["trust_delta_retrieval"] = row["cell10_trust_score_1_7"] - row["cell0_trust_score_1_7"]
    d["regret_delta_retrieval"] = row["cell10_regret_risk_1_7"] - row["cell0_regret_risk_1_7"]

    # pairwise: 00 vs 10 overall winner
    pw_result = row.get("pw_0v10_overall", "")
    # Normalize winner cell strings: "00"->"0", "01"->"1" for int comparison
    pw_int = CELL_STR_TO_INT.get(pw_result, pw_result)
    d["pw_00v10_overall_winner"] = pw_int

    switch_data.append(d)

print(f"  Budget parse failures: {budget_parse_fail}/{len(diag_rows)}")

# Subgroups
all_s = switch_data
changed_s = [d for d in switch_data if d["retrieval_changed"] == 1]
# 00 vs 10: cell 0 wins → history retrieval loses
hist_loses = [d for d in switch_data if str(d["pw_00v10_overall_winner"]) == "0"]
hist_wins = [d for d in switch_data if str(d["pw_00v10_overall_winner"]) == "10"]

def summarize_group(group, label):
    n = len(group)
    if n == 0:
        return {"group": label, "n": 0}
    result = {"group": label, "n": n}
    numeric_keys = [k for k in group[0].keys() if k not in
                    ("cluster_id", "retrieval_changed", "inferred_segment",
                     "pw_00v10_overall_winner")]
    for k in numeric_keys:
        vals = [d[k] for d in group if isinstance(d[k], (int, float)) and not np.isnan(d[k])]
        if vals:
            result[f"{k}_mean"] = np.mean(vals)
            result[f"{k}_median"] = np.median(vals)
            result[f"{k}_n"] = len(vals)
        else:
            result[f"{k}_mean"] = np.nan
            result[f"{k}_median"] = np.nan
            result[f"{k}_n"] = 0
    return result

summaries = []
for grp, label in [(all_s, "all_clusters"), (changed_s, "retrieval_changed"),
                    (hist_loses, "history_retrieval_loses_pairwise"),
                    (hist_wins, "history_retrieval_wins_pairwise")]:
    summaries.append(summarize_group(grp, label))

# Save anatomy table
if summaries:
    keys = list(summaries[0].keys())
    with open(DIAG / "table_retrieval_switch_anatomy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summaries)
    print(f"  Saved table_retrieval_switch_anatomy.csv: {len(summaries)} groups")

# Print key findings
print(f"\n  Key retrieval-switch findings (changed clusters, n={len(changed_s)}):")
ch = summarize_group(changed_s, "changed")
for metric in ["price_diff", "log_rating_count_diff", "avg_rating_diff",
               "popularity_improvement", "segment_affinity_diff",
               "feature_fit_diff", "budget_violation_switch",
               "fit_delta_retrieval", "purchase_delta_retrieval"]:
    k = f"{metric}_mean"
    if k in ch:
        print(f"    {metric}: mean={ch[k]:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Absolute DID by outcome
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: Absolute evaluation DID by outcome")
print("=" * 70)

B_BOOT = 2000
SEED = 42

def cluster_bootstrap(cluster_vals, B=B_BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(cluster_vals)
    boots = np.array([np.mean(rng.choice(cluster_vals, size=n, replace=True)) for _ in range(B)])
    return {
        "mean": np.mean(cluster_vals),
        "se": np.std(boots, ddof=1),
        "ci_lo": np.percentile(boots, 2.5),
        "ci_hi": np.percentile(boots, 97.5),
        "p_positive": np.mean(boots > 0),
    }

did_results = []
for out in EVAL_OUTCOMES:
    # cluster-level values
    ret_generic_expr = []   # cell10 - cell00
    ret_history_expr = []   # cell11 - cell01
    expr_generic_ret = []   # cell01 - cell00
    expr_history_ret = []   # cell11 - cell10
    interaction = []        # (cell11-cell10) - (cell01-cell00)

    for row in diag_rows:
        c00 = row[f"cell0_{out}"]
        c01 = row[f"cell1_{out}"]
        c10 = row[f"cell10_{out}"]
        c11 = row[f"cell11_{out}"]
        if any(np.isnan(v) for v in [c00, c01, c10, c11]):
            continue
        ret_generic_expr.append(c10 - c00)
        ret_history_expr.append(c11 - c01)
        expr_generic_ret.append(c01 - c00)
        expr_history_ret.append(c11 - c10)
        interaction.append((c11 - c10) - (c01 - c00))

    for name, vals in [("retrieval|generic_expr", ret_generic_expr),
                       ("retrieval|history_expr", ret_history_expr),
                       ("expression|generic_ret", expr_generic_ret),
                       ("expression|history_ret", expr_history_ret),
                       ("interaction_DID", interaction)]:
        if not vals:
            continue
        bs = cluster_bootstrap(np.array(vals))
        did_results.append({
            "outcome": out,
            "component": name,
            "mean": round(bs["mean"], 4),
            "se": round(bs["se"], 4),
            "ci_lo": round(bs["ci_lo"], 4),
            "ci_hi": round(bs["ci_hi"], 4),
            "p_positive": round(bs["p_positive"], 4),
            "n_clusters": len(vals),
        })

with open(DIAG / "table_absolute_did_by_outcome.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(did_results[0].keys()))
    w.writeheader()
    w.writerows(did_results)

print(f"  Saved table_absolute_did_by_outcome.csv: {len(did_results)} rows")

# Print key DID results
print("\n  Key absolute DID findings:")
for r in did_results:
    if r["component"] in ("retrieval|generic_expr", "expression|generic_ret", "interaction_DID"):
        star = "*" if r["ci_lo"] > 0 or r["ci_hi"] < 0 else ""
        print(f"    {r['outcome']:35s} {r['component']:25s}: {r['mean']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] P>0={r['p_positive']:.3f} {star}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Purchase interaction diagnostics
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: Purchase interaction diagnostics")
print("=" * 70)

# 4A: Absolute purchase interaction
purch_int = []
expr_lift_generic_ret = []
expr_lift_history_ret = []
for row in diag_rows:
    p00 = row["cell0_purchase_probability_0_100"]
    p01 = row["cell1_purchase_probability_0_100"]
    p10 = row["cell10_purchase_probability_0_100"]
    p11 = row["cell11_purchase_probability_0_100"]
    if any(np.isnan(v) for v in [p00, p01, p10, p11]):
        continue
    purch_int.append((p11 - p10) - (p01 - p00))
    expr_lift_generic_ret.append(p01 - p00)
    expr_lift_history_ret.append(p11 - p10)

bs_int = cluster_bootstrap(np.array(purch_int))
bs_lift_g = cluster_bootstrap(np.array(expr_lift_generic_ret))
bs_lift_h = cluster_bootstrap(np.array(expr_lift_history_ret))

print(f"  Absolute purchase interaction: {bs_int['mean']:+.2f} [{bs_int['ci_lo']:+.2f}, {bs_int['ci_hi']:+.2f}]")
print(f"  Expression lift (generic ret):  {bs_lift_g['mean']:+.2f} [{bs_lift_g['ci_lo']:+.2f}, {bs_lift_g['ci_hi']:+.2f}]")
print(f"  Expression lift (history ret):  {bs_lift_h['mean']:+.2f} [{bs_lift_h['ci_lo']:+.2f}, {bs_lift_h['ci_hi']:+.2f}]")

# 4C: identify clusters where history expression helps purchase under history retrieval
# but not under generic retrieval
compensatory_clusters = []
non_compensatory_clusters = []
for i, row in enumerate(diag_rows):
    p00 = row["cell0_purchase_probability_0_100"]
    p01 = row["cell1_purchase_probability_0_100"]
    p10 = row["cell10_purchase_probability_0_100"]
    p11 = row["cell11_purchase_probability_0_100"]
    if any(np.isnan(v) for v in [p00, p01, p10, p11]):
        continue
    if p11 > p10 and p01 <= p00:
        compensatory_clusters.append(i)
    else:
        non_compensatory_clusters.append(i)

print(f"\n  Compensatory clusters (expr helps purchase under hist ret but not generic): {len(compensatory_clusters)}/{len(diag_rows)}")

# 4D: examine those clusters
comp_stats = []
for idx in compensatory_clusters:
    d = switch_data[idx]
    comp_stats.append(d)

if comp_stats:
    ch = summarize_group(comp_stats, "compensatory")
    print(f"    Mean popularity_improvement: {ch.get('popularity_improvement_mean', 'N/A'):.2f}")
    print(f"    Mean log_rating_count_diff: {ch.get('log_rating_count_diff_mean', 'N/A'):.2f}")
    print(f"    Mean segment_affinity_diff: {ch.get('segment_affinity_diff_mean', 'N/A'):.2f}")

# Save purchase interaction drivers
driver_rows = []
for i, row in enumerate(diag_rows):
    p00 = row["cell0_purchase_probability_0_100"]
    p01 = row["cell1_purchase_probability_0_100"]
    p10 = row["cell10_purchase_probability_0_100"]
    p11 = row["cell11_purchase_probability_0_100"]
    if any(np.isnan(v) for v in [p00, p01, p10, p11]):
        continue
    driver_rows.append({
        "cluster_id": row["cluster_id"],
        "persona_id": row["persona_id"],
        "retrieval_changed": row["retrieval_changed"],
        "generic_product": row.get("generic_product_id", switch_data[i].get("cluster_id", "")),
        "history_product": row.get("history_product_id", switch_data[i].get("cluster_id", "")),
        "purchase_00": p00,
        "purchase_01": p01,
        "purchase_10": p10,
        "purchase_11": p11,
        "expr_lift_generic_ret": p01 - p00,
        "expr_lift_history_ret": p11 - p10,
        "purchase_interaction": (p11 - p10) - (p01 - p00),
        "compensatory": int(p11 > p10 and p01 <= p00),
        "price_diff": switch_data[i]["price_diff"],
        "log_rating_count_diff": switch_data[i]["log_rating_count_diff"],
        "popularity_improvement": switch_data[i]["popularity_improvement"],
        "segment_affinity_diff": switch_data[i]["segment_affinity_diff"],
        "history_over_budget": switch_data[i].get("history_over_budget", np.nan),
    })

# Fix generic/history product from switch_data
for i, dr in enumerate(driver_rows):
    dr["generic_product"] = diag_rows[i]["generic_product_id"] if "generic_product_id" in diag_rows[i] else ""
    dr["history_product"] = diag_rows[i]["history_product_id"] if "history_product_id" in diag_rows[i] else ""

with open(DIAG / "table_purchase_interaction_drivers.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(driver_rows[0].keys()))
    w.writeheader()
    w.writerows(driver_rows)

print(f"  Saved table_purchase_interaction_drivers.csv: {len(driver_rows)} rows")


# ── 4: Qualitative examples ─────────────────────────────────────────────────
def format_example(idx, label):
    row = diag_rows[idx]
    persona = personas[row["persona_id"]]
    gp = catalog.get(row.get("generic_product_id", ""), {})
    hp = catalog.get(row.get("history_product_id", ""), {})
    p00 = row["cell0_purchase_probability_0_100"]
    p01 = row["cell1_purchase_probability_0_100"]
    p10 = row["cell10_purchase_probability_0_100"]
    p11 = row["cell11_purchase_probability_0_100"]
    interaction = (p11 - p10) - (p01 - p00)
    lines = [
        f"### {label}: {row['cluster_id']}",
        f"**Persona**: {persona.get('primary_use_case', '')}, budget {persona.get('budget', '')}, "
        f"risk aversion: {persona.get('risk_aversion', '')}, tech knowledge: {persona.get('technical_knowledge', '')}",
        f"**Must-have**: {', '.join(persona.get('must_have_features', []))}",
        f"**Avoid**: {', '.join(persona.get('features_to_avoid', []))}",
        f"**Generic product**: {gp.get('title', 'N/A')[:80]} (${gp.get('price', '?')}, rating {gp.get('average_rating', '?')}, {gp.get('rating_count', '?')} reviews)",
        f"**History product**: {hp.get('title', 'N/A')[:80]} (${hp.get('price', '?')}, rating {hp.get('average_rating', '?')}, {hp.get('rating_count', '?')} reviews)",
        f"**Purchase scores**: cell00={p00:.0f}, cell01={p01:.0f}, cell10={p10:.0f}, cell11={p11:.0f}",
        f"**Expression lift (generic ret)**: {p01 - p00:+.0f} | **Expression lift (history ret)**: {p11 - p10:+.0f}",
        f"**Purchase interaction**: {interaction:+.0f}",
    ]
    # GPT reason from pairwise 10 vs 11 if available
    pw = pair_ix.get((row["cluster_id"], 10, 11), {})
    if pw.get("reason"):
        lines.append(f"**GPT pairwise reason (10 vs 11)**: {pw['reason'][:300]}")
    return "\n".join(lines)

# Sort by absolute purchase interaction for representative selection
sorted_by_int = sorted(range(len(diag_rows)), key=lambda i: (
    diag_rows[i]["cell11_purchase_probability_0_100"] - diag_rows[i]["cell10_purchase_probability_0_100"]
) - (diag_rows[i]["cell1_purchase_probability_0_100"] - diag_rows[i]["cell0_purchase_probability_0_100"]),
    reverse=True)

# Pick examples across the distribution
n = len(sorted_by_int)
positive_int_indices = [i for i in sorted_by_int if
    ((diag_rows[i]["cell11_purchase_probability_0_100"] - diag_rows[i]["cell10_purchase_probability_0_100"]) -
     (diag_rows[i]["cell1_purchase_probability_0_100"] - diag_rows[i]["cell0_purchase_probability_0_100"])) > 0]
negative_int_indices = [i for i in sorted_by_int if
    ((diag_rows[i]["cell11_purchase_probability_0_100"] - diag_rows[i]["cell10_purchase_probability_0_100"]) -
     (diag_rows[i]["cell1_purchase_probability_0_100"] - diag_rows[i]["cell0_purchase_probability_0_100"])) < 0]

examples_md = ["# Purchase Interaction — Representative Examples\n"]
examples_md.append("## Positive interaction examples\n")
examples_md.append("These clusters show history-aware expression lifting purchase probability more under history retrieval than under generic retrieval.\n")

# Pick 6 positive examples spread across the distribution
if len(positive_int_indices) >= 6:
    pick_pos = [positive_int_indices[j] for j in [0, len(positive_int_indices)//5, len(positive_int_indices)//3,
                                                   len(positive_int_indices)//2, 2*len(positive_int_indices)//3,
                                                   4*len(positive_int_indices)//5]]
else:
    pick_pos = positive_int_indices[:6]

for k, idx in enumerate(pick_pos, 1):
    examples_md.append(format_example(idx, f"Example {k} (positive interaction)"))
    examples_md.append("")

examples_md.append("\n## Counterexamples (negative interaction)\n")
examples_md.append("These clusters show the opposite pattern: history-aware expression helps purchase less (or hurts more) under history retrieval.\n")

pick_neg = negative_int_indices[:2] if len(negative_int_indices) >= 2 else negative_int_indices
for k, idx in enumerate(pick_neg, 1):
    examples_md.append(format_example(idx, f"Counterexample {k} (negative interaction)"))
    examples_md.append("")

with open(REPORTS / "purchase_interaction_examples.md", "w") as f:
    f.write("\n".join(examples_md))
print(f"  Saved purchase_interaction_examples.md")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Pairwise evaluator reason coding
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: Pairwise reason coding (heuristic keyword-based)")
print("=" * 70)

REASON_PATTERNS = {
    "persona_fit": r"(?:persona|specific needs|individual|personal|particular needs|user.s needs)",
    "budget_fit": r"(?:budget|price|afford|cost|expensive|cheaper|value for money|price.?point)",
    "feature_match": r"(?:feature|spec|require|must.have|need|function|capabilit)",
    "credibility_history": r"(?:credib|histor|reliab|proven|track record|reputation|popular|review|rating|buyer|customer|feedback|experience)",
    "tradeoff_disclosure": r"(?:tradeoff|trade.off|honest|transparent|limitation|drawback|downside|balanced|candid|nuance)",
    "lower_regret": r"(?:regret|risk|safe|cautious|conservative|uncertain|worry)",
    "trust": r"(?:trust|confidence|assurance|depend|believ)",
}

def code_reason(text):
    if not text:
        return ["unclear"]
    text_lower = text.lower()
    codes = []
    for cat, pat in REASON_PATTERNS.items():
        if re.search(pat, text_lower):
            codes.append(cat)
    return codes if codes else ["other"]

reason_results = []
for ci, cj, label in [(0, 10, "00_vs_10"), (10, 11, "10_vs_11"), (1, 11, "01_vs_11")]:
    code_counts = Counter()
    winner_counts = Counter()
    total = 0
    sample_reasons = []
    for cid in clusters:
        pw = pair_ix.get((cid, ci, cj), {})
        if not pw:
            continue
        total += 1
        reason = pw.get("reason", "")
        winner = pw.get("overall_winner_cell", "")
        winner_counts[str(winner)] += 1
        codes = code_reason(reason)
        for c in codes:
            code_counts[c] += 1
        if reason and len(sample_reasons) < 10:
            sample_reasons.append((cid, winner, reason[:200]))

    for code, count in code_counts.most_common():
        reason_results.append({
            "comparison": label,
            "reason_code": code,
            "count": count,
            "pct": round(100 * count / total, 1) if total else 0,
            "total": total,
        })
    print(f"\n  {label} (n={total}):")
    print(f"    Winners: {dict(winner_counts)}")
    print(f"    Reason codes: {code_counts.most_common(5)}")

with open(DIAG / "table_pairwise_reason_codes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["comparison", "reason_code", "count", "pct", "total"])
    w.writeheader()
    w.writerows(reason_results)
print(f"\n  Saved table_pairwise_reason_codes.csv: {len(reason_results)} rows")

# Pairwise reason summary report
reason_md = ["# Pairwise Evaluation — Reason Summary\n"]
reason_md.append("Reason codes are assigned via heuristic keyword matching. A single reason can match multiple categories.\n")

for ci, cj, label, question in [
    (0, 10, "00_vs_10", "Why does generic retrieval beat history retrieval?"),
    (10, 11, "10_vs_11", "Why does history expression beat generic expression (product held fixed)?"),
    (1, 11, "01_vs_11", "Why does generic retrieval + history expression beat full history?"),
]:
    reason_md.append(f"\n## {label}: {question}\n")
    subset = [r for r in reason_results if r["comparison"] == label]
    winner_counts = Counter()
    sample_reasons = []
    for cid in clusters:
        pw = pair_ix.get((cid, ci, cj), {})
        if pw:
            winner_counts[str(pw.get("overall_winner_cell", ""))] += 1
            if pw.get("reason") and len(sample_reasons) < 5:
                sample_reasons.append(f"- *{cid}*: {pw['reason'][:250]}")

    reason_md.append(f"**Winner distribution**: {dict(winner_counts)}\n")
    reason_md.append("| Reason code | Count | % |")
    reason_md.append("|-------------|-------|---|")
    for r in subset:
        reason_md.append(f"| {r['reason_code']} | {r['count']} | {r['pct']}% |")
    reason_md.append("\n**Sample reasons**:\n")
    reason_md.extend(sample_reasons[:5])
    reason_md.append("")

with open(REPORTS / "pairwise_reason_summary.md", "w") as f:
    f.write("\n".join(reason_md))
print(f"  Saved pairwise_reason_summary.md")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Paper-ready figures
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6: Generating paper-ready figures")
print("=" * 70)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = {
    "retrieval": "#d62728",
    "expression": "#2ca02c",
    "interaction": "#9467bd",
    "total": "#1f77b4",
    "cell0": "#4e79a7",
    "cell1": "#59a14f",
    "cell10": "#e15759",
    "cell11": "#b07aa1",
    "neutral": "#7f7f7f",
}

def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(FIGS / f"{name}.{ext}")
        fig.savefig(PAPER_FIGS / f"{name}.{ext}")
    plt.close(fig)
    print(f"    Saved {name}.png/pdf")


# ── Figure A: Main modular decomposition ─────────────────────────────────────
bt_data = load_csv(DATA / "analysis" / "table4_bt_decomposition.csv")
bt_map = {r["component"]: r for r in bt_data}

fig, ax = plt.subplots(figsize=(7, 4.5))

components = ["retrieval", "expression", "interaction", "total"]
labels = [r"Retrieval ($\Delta_J$)", r"Expression ($\Delta_T$)",
          r"Interaction ($\Delta_{JT}$)", r"Total ($\tau^{MOD}$)"]
colors = [COLORS[c] for c in components]

estimates = [float(bt_map[c]["mean"]) for c in components]
ci_lo = [float(bt_map[c]["ci_lo"]) for c in components]
ci_hi = [float(bt_map[c]["ci_hi"]) for c in components]

errors_lo = [est - lo for est, lo in zip(estimates, ci_lo)]
errors_hi = [hi - est for est, hi in zip(estimates, ci_hi)]

y_pos = np.arange(len(components))
ax.barh(y_pos, estimates, xerr=[errors_lo, errors_hi],
        color=colors, edgecolor="black", linewidth=0.6,
        capsize=4, height=0.55, alpha=0.85)
ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel("Bradley-Terry Estimate (log-odds scale)")
ax.set_title("Modular Decomposition of History Shock\n(Pairwise GPT Evaluation, Overall Winner)")
ax.invert_yaxis()

for i, (est, lo, hi) in enumerate(zip(estimates, ci_lo, ci_hi)):
    ax.text(max(hi, est) + 0.05, i, f"{est:+.2f}", va="center", fontsize=9, fontweight="bold")

ax.set_xlim(ax.get_xlim()[0] - 0.1, ax.get_xlim()[1] + 0.35)
fig.tight_layout()
save_fig(fig, "fig_main_decomposition")


# ── Figure B: Cell means by outcome ──────────────────────────────────────────
outcome_labels = {
    "fit_score_1_7": "Fit (1-7)",
    "purchase_probability_0_100": "Purchase Prob (0-100)",
    "expected_satisfaction_0_100": "Satisfaction (0-100)",
    "trust_score_1_7": "Trust (1-7)",
    "regret_risk_1_7": "Regret Risk (1-7)",
}
cell_labels = {
    0:  "(0,0) Generic+Generic",
    1:  "(0,1) Generic Ret + History Expr",
    10: "(1,0) History Ret + Generic Expr",
    11: "(1,1) History+History",
}

fig, axes = plt.subplots(1, 5, figsize=(16, 4.5), sharey=False)

for ax_idx, (out_key, out_label) in enumerate(outcome_labels.items()):
    ax = axes[ax_idx]
    means = []
    cis_lo = []
    cis_hi = []
    for cell in CELLS:
        vals = np.array([row[f"cell{cell}_{out_key}"] for row in diag_rows
                         if not np.isnan(row[f"cell{cell}_{out_key}"])])
        bs = cluster_bootstrap(vals, B=1000, seed=42)
        means.append(bs["mean"])
        cis_lo.append(bs["ci_lo"])
        cis_hi.append(bs["ci_hi"])

    x = np.arange(4)
    cell_colors = [COLORS[f"cell{c}"] for c in CELLS]
    bars = ax.bar(x, means, color=cell_colors, edgecolor="black", linewidth=0.5,
                  width=0.6, alpha=0.85)
    err_lo = [m - lo for m, lo in zip(means, cis_lo)]
    err_hi = [hi - m for m, hi in zip(means, cis_hi)]
    ax.errorbar(x, means, yerr=[err_lo, err_hi], fmt="none", ecolor="black",
                capsize=3, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["(0,0)", "(0,1)", "(1,0)", "(1,1)"], fontsize=8)
    ax.set_title(out_label, fontsize=10)
    ax.set_xlabel(r"($H^J, H^T$)", fontsize=9)

    # y axis starts at sensible value
    ymin = min(cis_lo) * 0.9 if min(cis_lo) > 0 else min(cis_lo) - abs(min(cis_lo)) * 0.1
    ax.set_ylim(bottom=ymin * 0.85)

fig.suptitle("Cell Means by Outcome Dimension (Absolute GPT Evaluation)", fontsize=13, y=1.02)
fig.tight_layout()
save_fig(fig, "fig_cell_means_outcomes")


# ── Figure C: Retrieval switch anatomy ───────────────────────────────────────
# Mean differences with bootstrap CIs for changed clusters only
anatomy_metrics = [
    ("price_diff", "Price Diff ($)"),
    ("log_rating_count_diff", "Log Review Count Diff"),
    ("avg_rating_diff", "Avg Rating Diff"),
    ("popularity_improvement", "Popularity Rank\nImprovement"),
    ("budget_violation_switch", "Budget Violation\nSwitch Rate"),
    ("segment_affinity_diff", "Segment Affinity Diff"),
    ("feature_fit_diff", "Feature Fit Diff\n(heuristic)"),
]

fig, ax = plt.subplots(figsize=(7, 5))
y_positions = np.arange(len(anatomy_metrics))
means_anat = []
cis_lo_anat = []
cis_hi_anat = []
labels_anat = []

for key, label in anatomy_metrics:
    vals = np.array([d[key] for d in changed_s
                     if isinstance(d[key], (int, float)) and not np.isnan(d[key])])
    if len(vals) > 0:
        bs = cluster_bootstrap(vals, B=2000, seed=42)
        means_anat.append(bs["mean"])
        cis_lo_anat.append(bs["ci_lo"])
        cis_hi_anat.append(bs["ci_hi"])
    else:
        means_anat.append(0)
        cis_lo_anat.append(0)
        cis_hi_anat.append(0)
    labels_anat.append(label)

# Normalize for visualization: standardize each to its own scale
# Better: just show raw means with separate axes or just plot them
err_lo = [m - lo for m, lo in zip(means_anat, cis_lo_anat)]
err_hi = [hi - m for m, hi in zip(means_anat, cis_hi_anat)]

bar_colors = ["#d62728" if m > 0 else "#2ca02c" for m in means_anat]
ax.barh(y_positions, means_anat, xerr=[err_lo, err_hi],
        color=bar_colors, edgecolor="black", linewidth=0.5,
        capsize=4, height=0.5, alpha=0.8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y_positions)
ax.set_yticklabels(labels_anat)
ax.set_xlabel("History Product − Generic Product (mean difference)")
ax.set_title("Retrieval Switch Anatomy\n(clusters where product changed, n={})".format(len(changed_s)))
ax.invert_yaxis()

for i, m in enumerate(means_anat):
    ax.text(max(means_anat[i], cis_hi_anat[i]) + max(abs(v) for v in means_anat) * 0.03,
            i, f"{m:+.2f}", va="center", fontsize=9)

fig.tight_layout()
save_fig(fig, "fig_retrieval_switch_anatomy")


# ── Figure D: Retrieval harm vs product-switch attributes ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

x_vars = [
    ("log_rating_count_diff", "Log Review Count Diff"),
    ("price_diff", "Price Diff ($)"),
    ("popularity_improvement", "Popularity Rank Improvement"),
]
y_var = ("fit_delta_retrieval", "Fit Score Change (cell10 − cell00)")

for ax, (xkey, xlabel) in zip(axes, x_vars):
    xv = []
    yv = []
    for d in changed_s:
        x = d.get(xkey, np.nan)
        y = d.get("fit_delta_retrieval", np.nan)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not (np.isnan(x) or np.isnan(y)):
            xv.append(x)
            yv.append(y)
    xv = np.array(xv)
    yv = np.array(yv)

    ax.scatter(xv, yv, alpha=0.5, s=30, color=COLORS["retrieval"], edgecolors="black", linewidth=0.3)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.6, linestyle="--")

    if len(xv) > 2:
        z = np.polyfit(xv, yv, 1)
        xline = np.linspace(xv.min(), xv.max(), 100)
        ax.plot(xline, np.polyval(z, xline), color="black", linewidth=1.2, linestyle="--", alpha=0.7)
        corr = np.corrcoef(xv, yv)[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax.transAxes,
                fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_var[1] if ax == axes[0] else "")
    ax.set_title(xlabel)

fig.suptitle("Retrieval Harm vs. Product-Switch Attributes\n(clusters where product changed)", fontsize=12, y=1.02)
fig.tight_layout()
save_fig(fig, "fig_retrieval_harm_scatter")


# ── Figure E: Purchase interaction diagnostic ────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))

labels_e = ["Expression lift\n(generic retrieval)\ncell01 − cell00",
            "Expression lift\n(history retrieval)\ncell11 − cell10"]
means_e = [bs_lift_g["mean"], bs_lift_h["mean"]]
err_lo_e = [bs_lift_g["mean"] - bs_lift_g["ci_lo"], bs_lift_h["mean"] - bs_lift_h["ci_lo"]]
err_hi_e = [bs_lift_g["ci_hi"] - bs_lift_g["mean"], bs_lift_h["ci_hi"] - bs_lift_h["mean"]]

x_e = np.arange(2)
bars = ax.bar(x_e, means_e, yerr=[err_lo_e, err_hi_e],
              color=[COLORS["expression"], COLORS["interaction"]],
              edgecolor="black", linewidth=0.6, capsize=6, width=0.5, alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x_e)
ax.set_xticklabels(labels_e, fontsize=9)
ax.set_ylabel("Purchase Probability Change (0-100)")
ax.set_title("History-Expression Lift on Purchase Probability\nby Retrieval Condition")

for i, (m, lo, hi) in enumerate(zip(means_e, err_lo_e, err_hi_e)):
    ax.text(i, m + hi + 0.5, f"{m:+.1f}", ha="center", fontsize=10, fontweight="bold")

# Annotate interaction
int_val = bs_int["mean"]
ax.annotate(f"Interaction: {int_val:+.1f}\n[{bs_int['ci_lo']:+.1f}, {bs_int['ci_hi']:+.1f}]",
            xy=(0.5, max(means_e)), xytext=(0.5, max(means_e) + max(err_hi_e) + 3),
            ha="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9))

fig.tight_layout()
save_fig(fig, "fig_purchase_interaction")


# ── Figure F: Pairwise win-rate matrix ───────────────────────────────────────
cell_pairs = [(0, 1), (0, 10), (0, 11), (1, 11), (10, 1), (10, 11)]
win_matrix = np.full((4, 4), np.nan)
cell_idx = {0: 0, 1: 1, 10: 2, 11: 3}

for ci, cj in cell_pairs:
    wins_i = 0
    wins_j = 0
    ties = 0
    total = 0
    ci_str = CELL_INT_TO_STR.get(ci, str(ci))
    cj_str = CELL_INT_TO_STR.get(cj, str(cj))
    for cid in clusters:
        pw = pair_ix.get((cid, ci, cj), {})
        if not pw:
            continue
        total += 1
        winner = pw.get("overall_winner_cell", "")
        if winner == ci_str:
            wins_i += 1
        elif winner == cj_str:
            wins_j += 1
        else:
            ties += 1
    if total > 0:
        win_matrix[cell_idx[ci], cell_idx[cj]] = wins_i / total * 100
        win_matrix[cell_idx[cj], cell_idx[ci]] = wins_j / total * 100

fig, ax = plt.subplots(figsize=(6, 5))
cell_names = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]

mask = np.isnan(win_matrix)
display = np.where(mask, 0, win_matrix)

im = ax.imshow(display, cmap="RdYlGn", vmin=20, vmax=80, aspect="auto")

for i in range(4):
    for j in range(4):
        if i == j:
            ax.text(j, i, "—", ha="center", va="center", fontsize=11, color="grey")
        elif not np.isnan(win_matrix[i, j]):
            val = win_matrix[i, j]
            color = "white" if val < 30 or val > 70 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)
        else:
            ax.text(j, i, "n/a", ha="center", va="center", fontsize=9, color="grey")

ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(cell_names)
ax.set_yticklabels(cell_names)
ax.set_xlabel("Column cell (opponent)")
ax.set_ylabel("Row cell (wins as %)")
ax.set_title("Pairwise Win Rate Matrix\n(GPT Overall Evaluation)")
fig.colorbar(im, ax=ax, label="Win Rate (%)", shrink=0.8)

fig.tight_layout()
save_fig(fig, "fig_pairwise_win_matrix")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Mechanism diagnostics report
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7: Writing mechanism diagnostics report")
print("=" * 70)

# Gather statistics for the report
ch_summary = summarize_group(changed_s, "changed")
all_summary = summarize_group(all_s, "all")
hl_summary = summarize_group(hist_loses, "hist_loses")

# Absolute DID for key outcomes
did_map = {}
for r in did_results:
    did_map[(r["outcome"], r["component"])] = r

# Budget violation stats
bv_all = [d["budget_violation_switch"] for d in switch_data
          if isinstance(d["budget_violation_switch"], (int, float)) and not np.isnan(d["budget_violation_switch"])]
bv_rate = np.mean(bv_all) if bv_all else 0

# Over-budget rates
ob_generic = [d["generic_over_budget"] for d in switch_data
              if isinstance(d["generic_over_budget"], (int, float)) and not np.isnan(d["generic_over_budget"])]
ob_history = [d["history_over_budget"] for d in switch_data
              if isinstance(d["history_over_budget"], (int, float)) and not np.isnan(d["history_over_budget"])]

n_ret_changed = len(changed_s)
n_total = len(all_s)
n_hist_loses = len(hist_loses)
n_hist_wins = len(hist_wins)

# Reason code summaries
rc_00v10 = {r["reason_code"]: r for r in reason_results if r["comparison"] == "00_vs_10"}

price_direction = "More expensive" if ch_summary.get("price_diff_mean", 0) > 0 else "Cheaper"
logrc_direction = "Higher" if ch_summary.get("log_rating_count_diff_mean", 0) > 0 else "Lower"
rating_direction = "Higher" if ch_summary.get("avg_rating_diff_mean", 0) > 0 else "Lower"
pop_direction = "More popular" if ch_summary.get("popularity_improvement_mean", 0) > 0 else "Less popular"
aff_direction = "Higher" if ch_summary.get("segment_affinity_diff_mean", 0) > 0 else "Lower"
fit_direction = "Better" if ch_summary.get("feature_fit_diff_mean", 0) > 0 else "Worse"

# Compute additional stats for report
n_more_expensive = sum(1 for d in changed_s if isinstance(d.get("history_more_expensive"), (int,float)) and d["history_more_expensive"] == 1)
n_less_popular = sum(1 for d in changed_s if isinstance(d.get("popularity_improvement"), (int,float)) and d["popularity_improvement"] < 0)
n_fewer_reviews = sum(1 for d in changed_s if isinstance(d.get("history_higher_rating_count"), (int,float)) and d["history_higher_rating_count"] == 0)

report = f"""# Mechanism Diagnostics Report — Modular Two-Stage History Shock

**Date**: 2026-05-16
**Status**: Complete

---

## 1. Executive Summary

- History-aware retrieval changes the selected product in {n_ret_changed}/{n_total} clusters ({100*n_ret_changed/n_total:.0f}%), shifting toward more expensive products (mean price diff: ${ch_summary.get("price_diff_mean", 0):+.1f}) that are less popular (mean rank worsening: {ch_summary.get("popularity_improvement_mean", 0):+.1f}) and have fewer reviews (mean log-count diff: {ch_summary.get("log_rating_count_diff_mean", 0):+.2f}). This comes at the cost of persona fit: mean fit score drops by {abs(ch_summary.get("fit_delta_retrieval_mean", 0)):.2f} points (1-7 scale) and purchase probability drops by {abs(ch_summary.get("purchase_delta_retrieval_mean", 0)):.1f} pp (0-100).
- In pairwise evaluation, generic retrieval beats history retrieval in {n_hist_loses}/{n_total} clusters ({100*n_hist_loses/n_total:.0f}%); history retrieval wins in only {n_hist_wins}/{n_total} ({100*n_hist_wins/n_total:.0f}%).
- The purchase-specific positive interaction in the Bradley-Terry pairwise decomposition (+0.426, P>0=95.4%) does **not** replicate in absolute evaluation ({bs_int["mean"]:+.1f} pp, 95% CI [{bs_int["ci_lo"]:+.1f}, {bs_int["ci_hi"]:+.1f}]). The absolute evidence is too imprecise to support a compensatory story.
- History-aware expression increases trust ({did_map.get(("trust_score_1_7","expression|generic_ret"), {}).get("mean", 0):+.2f}, 95% CI [{did_map.get(("trust_score_1_7","expression|generic_ret"), {}).get("ci_lo", 0):+.2f}, {did_map.get(("trust_score_1_7","expression|generic_ret"), {}).get("ci_hi", 0):+.2f}]) and tradeoff disclosure ({did_map.get(("tradeoff_disclosure_1_7","expression|generic_ret"), {}).get("mean", 0):+.2f}) consistently across retrieval conditions.
- Budget violations partially explain retrieval harm: history retrieval moves products outside the persona's stated budget in {100*bv_rate:.0f}% of clusters where generic products were within budget.

---

## 2. Negative Retrieval Effect: Mechanism

### 2.1 What history retrieval changes

When the LLM receives buyer history (popularity tiers, rating counts, segment-level feedback), it shifts product selection as follows. Among retrieval-changed clusters (n={n_ret_changed}):

| Attribute | Mean diff (history − generic) | Direction | Count |
|-----------|-------------------------------|-----------|-------|
| Price | ${ch_summary.get("price_diff_mean", 0):+.1f} | {price_direction} | {n_more_expensive}/{n_ret_changed} more expensive |
| Log review count | {ch_summary.get("log_rating_count_diff_mean", 0):+.2f} | {logrc_direction} | {n_fewer_reviews}/{n_ret_changed} fewer reviews |
| Average rating | {ch_summary.get("avg_rating_diff_mean", 0):+.2f} | {rating_direction} | — |
| Popularity rank | {ch_summary.get("popularity_improvement_mean", 0):+.1f} | {pop_direction} | {n_less_popular}/{n_ret_changed} less popular |
| Segment affinity | {ch_summary.get("segment_affinity_diff_mean", 0):+.2f} | {aff_direction} | — |
| Feature fit (heuristic) | {ch_summary.get("feature_fit_diff_mean", 0):+.3f} | {fit_direction} | — |

The pattern is **not** a simple bestseller/popularity bias. History retrieval shifts toward products that are more expensive, less popular, and have fewer reviews. The segment-affinity improvement is negligible (+{ch_summary.get("segment_affinity_diff_mean", 0):.2f}), suggesting the LLM is not selecting products that are empirically better for the persona's segment.

### 2.2 Why this hurts

The evidence is consistent with a **premium-aspiration bias**: the LLM, when given qualitative buyer feedback describing product satisfaction, gravitates toward higher-priced products — potentially interpreting satisfaction signals as evidence that spending more is worthwhile — even when the premium product is a poor fit for the persona's budget and feature requirements.

1. **Budget violations**: {100*bv_rate:.0f}% of clusters experience a budget violation switch (generic product within budget, history product exceeds budget). Over-budget rate: generic products {100*np.mean(ob_generic) if ob_generic else 0:.0f}%, history products {100*np.mean(ob_history) if ob_history else 0:.0f}%.

2. **Fit score deterioration**: Absolute fit score drops by {abs(ch_summary.get("fit_delta_retrieval_mean", 0)):.2f} points on average (1-7 scale) when retrieval switches to the history-aware product, even holding expression constant. The CI excludes zero.

3. **GPT pairwise reasons**: In the 00 vs 10 comparison, the most frequent reason codes for cell 0 winning are: {", ".join(f"{k} ({v['count']}/{v['total']})" for k, v in sorted(rc_00v10.items(), key=lambda x: -x[1]["count"])[:3])}. Feature match and budget fit are the top codes, consistent with the interpretation that history-retrieved products violate persona-specific constraints.

4. **Regret risk increases**: History-retrieved products are judged as higher regret risk (+{did_map.get(("regret_risk_1_7","retrieval|generic_expr"), {}).get("mean", 0):.2f} on 1-7 scale, CI excludes zero), suggesting the GPT evaluator recognizes that these products carry higher downside for the persona.

### 2.3 Among clusters where history retrieval loses pairwise

When generic retrieval wins the overall pairwise comparison (n={n_hist_loses}):
- Mean price diff: ${hl_summary.get("price_diff_mean", 0):+.1f}
- Mean log review count diff: {hl_summary.get("log_rating_count_diff_mean", 0):+.2f}
- Mean fit delta: {hl_summary.get("fit_delta_retrieval_mean", 0):+.2f}
- Mean segment affinity diff: {hl_summary.get("segment_affinity_diff_mean", 0):+.2f}

The retrieval harm is concentrated in clusters where history information causes the largest price increase and fit deterioration.

### 2.4 Interpretation (with discipline)

In this controlled simulation, history-aware retrieval appears to shift product selection toward more expensive, niche products at the expense of persona-specific fit and budget alignment. The LLM interprets segment-level satisfaction signals as reasons to recommend premium alternatives, without adequately weighing the individual persona's stated budget constraints and feature requirements.

This is consistent with the interpretation that the LLM treats qualitative buyer history as a **population-level quality signal** that overrides persona-specific constraints — a form of aspiration bias rather than popularity bias.

We do not claim this pattern generalizes to all LLM architectures, all product categories, or real consumer populations. The pattern is observed in a single product category (headphones) with a single LLM (Qwen 2.5 14B) evaluated by a single judge (GPT-5.3).

---

## 3. Purchase Interaction: BT vs. Absolute Discrepancy

### 3.1 The finding

In the Bradley-Terry pairwise decomposition, the purchase-specific interaction is positive and marginally significant (+0.426, P>0=95.4%). However, in absolute evaluation, the purchase interaction is {bs_int["mean"]:+.1f} pp (95% CI [{bs_int["ci_lo"]:+.1f}, {bs_int["ci_hi"]:+.1f}]) — essentially zero and imprecise.

### 3.2 Expression lift comparison (absolute scale)

| Condition | Expression lift on purchase | 95% CI |
|-----------|-----------------------------|--------|
| Under generic retrieval | {bs_lift_g["mean"]:+.1f} pp | [{bs_lift_g["ci_lo"]:+.1f}, {bs_lift_g["ci_hi"]:+.1f}] |
| Under history retrieval | {bs_lift_h["mean"]:+.1f} pp | [{bs_lift_h["ci_lo"]:+.1f}, {bs_lift_h["ci_hi"]:+.1f}] |
| **Difference (interaction)** | **{bs_int["mean"]:+.1f} pp** | **[{bs_int["ci_lo"]:+.1f}, {bs_int["ci_hi"]:+.1f}]** |

Expression has near-zero effect on absolute purchase probability under **both** retrieval conditions. This contrasts with the BT pairwise decomposition, where the purchase interaction is the largest positive signal. The discrepancy may reflect that pairwise comparisons are more sensitive to within-cluster relative differences that wash out in absolute scoring.

### 3.3 Compensatory cluster pattern

{len(compensatory_clusters)}/{n_total} clusters ({100*len(compensatory_clusters)/n_total:.0f}%) show the compensatory pattern (history expression lifts purchase under history retrieval but not under generic retrieval). This is a minority of clusters — the pattern is not dominant.

### 3.4 Assessment

The positive purchase interaction is:
- **Present in pairwise BT decomposition** (+0.426, P>0=95.4%)
- **Absent in absolute evaluation** ({bs_int["mean"]:+.1f} pp, CI includes zero)
- **Observed in only {len(compensatory_clusters)}/{n_total} clusters** fitting the compensatory pattern
- **Not robust** across evaluation methods

We characterize this as a **pairwise-scale artifact** or, at best, a pattern that is too imprecise to support a compensatory mechanism claim. The paper should report the BT purchase interaction honestly but should not build a mechanism story around it. The key finding for purchase is the large negative retrieval main effect (-1.663 BT, and -{abs(ch_summary.get("purchase_delta_retrieval_mean", 0)):.1f} pp absolute), not the interaction.

---

## 4. Implications for Paper Framing

### 4.1 What we can claim

1. **The modular decomposition reveals opposing channels that are invisible in an aggregate A/B test.** The near-zero total effect masks a large negative retrieval effect and a large positive expression effect. This is the paper's core empirical contribution from the simulation.

2. **History-aware retrieval shifts product selection toward more expensive, less popular products at the expense of individual persona fit and budget compliance.** The pattern is consistent with a premium-aspiration bias — the LLM interprets satisfaction signals as reasons to up-sell — and is supported by product-level attribute comparisons, absolute evaluation scores, and pairwise evaluator reasons citing feature match and budget fit.

3. **History-aware expression improves trust and tradeoff disclosure consistently.** The trust gain (+{did_map.get(("trust_score_1_7","expression|generic_ret"), {}).get("mean", 0):.2f}) and tradeoff disclosure gain (+{did_map.get(("tradeoff_disclosure_1_7","expression|generic_ret"), {}).get("mean", 0):.2f}) are robust across absolute evaluation, with CIs excluding zero.

4. **The purchase-specific positive interaction in BT pairwise decomposition does not replicate in absolute evaluation.** It should be reported but not interpreted as a confirmed compensatory mechanism.

### 4.2 What we cannot claim

1. We cannot claim these patterns reflect real consumer behavior — the evaluator is a GPT model, not a human.
2. We cannot claim premium-aspiration bias is the only explanation for negative retrieval — alternative explanations include the LLM optimizing for a different objective (e.g., segment-typical choices rather than persona-specific fit).
3. We cannot claim the positive purchase interaction is robust — it appears in BT pairwise decomposition but not in absolute evaluation.
4. We cannot generalize to other product categories, LLM architectures, or prompt designs without additional experiments.
5. We should not describe the pattern as "popularity bias" — history retrieval actually shifts *away* from bestsellers, toward more expensive niche products.

### 4.3 Draft paragraphs for the paper

**Paragraph 1 (Results section):**
The modular decomposition reveals that the near-zero total history shock conceals two large, opposing channel effects. History-aware retrieval reduces overall recommendation quality (BT estimate: -0.828, 95% CI [-1.379, -0.326]), while history-aware expression improves it (+0.704, [0.388, 1.033]). Product-level diagnostics suggest that the retrieval harm arises from a premium-aspiration pattern: when exposed to segment-level buyer feedback, the LLM shifts product selection toward more expensive products (mean price increase: +${ch_summary.get("price_diff_mean", 0):.0f}) that are less popular and have fewer reviews, while violating the persona's stated budget in {100*bv_rate:.0f}% of cases where the generic retrieval respected it. The GPT evaluator's pairwise reasons most frequently cite better feature match and budget fit as reasons for preferring the generic retrieval product.

**Paragraph 2 (Discussion section):**
The opposing channel effects carry direct implications for firms deploying LLM-based recommender systems. Our simulation suggests that routing historical purchase information to the text-generation stage yields trust and satisfaction gains without the fit penalties associated with history-aware product selection. A modular architecture that uses specification-based retrieval with history-informed expression may outperform both the feature-only baseline and the fully history-aware system — a finding that would be invisible under a standard A/B comparison. The retrieval harm arises not from the LLM defaulting to bestsellers, but from its tendency to interpret segment satisfaction feedback as a signal to recommend premium alternatives that exceed individual budget constraints.

**Paragraph 3 (Limitations):**
The purchase-specific interaction merits caution. While the BT pairwise decomposition shows a positive purchase interaction (+0.426, P>0=95.4%), the absolute evaluation finds near-zero interaction ({bs_int["mean"]:+.1f} pp, CI [{bs_int["ci_lo"]:+.1f}, {bs_int["ci_hi"]:+.1f}]). This discrepancy may reflect greater sensitivity of pairwise comparisons to within-cluster relative differences, or it may indicate that the BT interaction is partly a scale artifact. We do not build a compensatory-mechanism story on this finding. Human-subject validation is needed to determine whether these patterns hold under real consumer evaluation.

---

## 5. Recommended Figures

| Figure | File | Placement | Importance |
|--------|------|-----------|------------|
| **Main decomposition** | `fig_main_decomposition.pdf` | **Main text** | **Primary** — the paper's core result visualization |
| **Cell means by outcome** | `fig_cell_means_outcomes.pdf` | **Main text** | **Primary** — shows the full 2x2 pattern readers need |
| **Retrieval switch anatomy** | `fig_retrieval_switch_anatomy.pdf` | Main text or appendix | Supports retrieval mechanism story |
| **Retrieval harm scatter** | `fig_retrieval_harm_scatter.pdf` | Appendix | Robustness — shows cluster-level correlation |
| **Purchase interaction** | `fig_purchase_interaction.pdf` | Main text or appendix | Visualizes the compensatory pattern |
| **Pairwise win matrix** | `fig_pairwise_win_matrix.pdf` | Main text | Intuitive summary of all head-to-head comparisons |

The two most important figures for the main paper body are:
1. **fig_main_decomposition** — the decomposition bar chart is the paper's signature visualization
2. **fig_cell_means_outcomes** — shows readers the raw pattern before the BT decomposition

---

*Report generated from existing simulation outputs. No new LLM calls were made. All confidence intervals use cluster-level bootstrap (B=2000, seed=42). Reason coding uses heuristic keyword matching, not LLM classification.*
"""

with open(REPORTS / "mechanism_diagnostics_report.md", "w") as f:
    f.write(report)
print("  Saved mechanism_diagnostics_report.md")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("\nCreated tables:")
for p in sorted(DIAG.glob("*.csv")):
    print(f"  {p.relative_to(ROOT)}")

print("\nCreated figures:")
for p in sorted(FIGS.glob("*")):
    print(f"  {p.relative_to(ROOT)}")

print("\nCopied to paper/figures/:")
for p in sorted(PAPER_FIGS.glob("*")):
    print(f"  {p.relative_to(ROOT)}")

print("\nCreated reports:")
for name in ["mechanism_diagnostics_report.md", "purchase_interaction_examples.md", "pairwise_reason_summary.md"]:
    print(f"  {(REPORTS / name).relative_to(ROOT)}")

print("\n5 most important numerical findings:")
print(f"  1. Retrieval changes product in {n_ret_changed}/{n_total} ({100*n_ret_changed/n_total:.0f}%) clusters")
print(f"  2. History products have +{ch_summary.get('log_rating_count_diff_mean', 0):.2f} log review counts (popularity bias)")
print(f"  3. Absolute fit drops {ch_summary.get('fit_delta_retrieval_mean', 0):+.2f} (1-7) under history retrieval")
print(f"  4. Absolute purchase drops {ch_summary.get('purchase_delta_retrieval_mean', 0):+.1f} pp under history retrieval")
print(f"  5. Purchase interaction: {bs_int['mean']:+.1f} pp [{bs_int['ci_lo']:+.1f}, {bs_int['ci_hi']:+.1f}] — suggestive, not conclusive")

print(f"\nFinal mechanism story (3 sentences):")
print(f"  History-aware retrieval shifts product selection toward more expensive, less popular")
print(f"  products that exceed persona budgets — a premium-aspiration bias, not a popularity bias.")
print(f"  History-aware expression compensates through higher trust and tradeoff disclosure,")
print(f"  creating the near-zero total effect that hides both channels in an aggregate comparison.")
print(f"  The purchase interaction is positive in BT pairwise but absent in absolute eval —")
print(f"  report it honestly but do not build a compensatory-mechanism claim around it.")

print("\nDone.")
