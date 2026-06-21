#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sensitivity_analysis_exp3.py
============================
Per-model Sobol-Hoeffding variance decomposition for Experiment 3
(prior-research models: SvmW, SvmA, Lstm).

Reuses the analytical SS decomposition from ``sensitivity_analysis_exp2.py``.
Because Exp3 coverage is incomplete for some (model, condition, mode) cells,
each model is reduced to its largest balanced rectangular factorial subset
satisfying ``min seeds-per-(D x G)-subcell >= N_MIN`` (default 12, matching
the 12-seed plotting policy used in ``plot_exp3_condition_seed_summaries.py``).

The goal is to verify, on the completed Exp3 subset, whether the qualitative
claims of Exp2 hold:
  (1) Mode (M) and Rebalancing (R) dominate the systematic variance.
  (2) R x M interaction is the largest two-way effect.
  (3) Distance metric (D) and Domain membership (G) are negligible.

Output
------
  results/analysis/exp3_prior_research/sensitivity_analysis_report.md
  results/analysis/exp3_prior_research/figures/csv/split2/sensitivity/
      sobol_indices_exp3.csv
      sobol_indices_exp3_combined.csv  (RF + per-model exp3 side-by-side)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse exp2 Sobol primitives.
from scripts.python.analysis.domain.sensitivity_analysis_exp2 import (
    compute_ss_decomposition,
    compute_sobol_indices,
    bootstrap_sobol,
    _encode_factors,
    FACTORS,
    FACTOR_NAMES,
    PRIMARY_METRICS,
)

# ---------------------------------------------------------------------------
EXP3_REPORT_DIR = PROJECT_ROOT / "results" / "analysis" / "exp3_prior_research"
EXP3_CSV_BASE = EXP3_REPORT_DIR / "figures" / "csv" / "split2"
EXP3_SOBOL_OUT = EXP3_CSV_BASE / "sensitivity"
EXP3_SOBOL_OUT.mkdir(parents=True, exist_ok=True)

EXP2_SOBOL_CSV = (
    PROJECT_ROOT / "results" / "analysis" / "exp2_domain_shift"
    / "figures" / "csv" / "split2" / "sensitivity" / "sobol_indices.csv"
)

PRIOR_MODELS = ["SvmW", "SvmA", "Lstm"]
N_MIN = 12  # min seeds per (D x G) subcell to consider a (condition, mode) "complete"

# Map exp3 method-tag -> exp2-style condition string
def _condition_tag(method: str, ratio) -> str | None:
    if method == "baseline":
        return "baseline"
    if pd.isna(ratio):
        return None
    rtag = "r" + str(ratio).replace(".", "")
    base = {
        "imbalv3":         "sw_smote",
        "smote_plain":     "smote",
        "undersample_rus": "rus",
    }.get(method)
    return f"{base}_{rtag}" if base else None


def _model_csv_files(model: str) -> dict:
    base = EXP3_CSV_BASE / model
    return {
        "baseline":        base / "baseline"        / f"{model.lower()}_baseline_split2_metrics.csv",
        "imbalv3":         base / "sw_smote"        / f"{model.lower()}_sw_smote_split2_metrics.csv",
        "smote_plain":     base / "smote_plain"     / f"{model.lower()}_smote_split2_metrics.csv",
        "undersample_rus": base / "undersample_rus" / f"{model.lower()}_rus_split2_metrics.csv",
    }


def load_model_data(model: str) -> pd.DataFrame:
    """Load all per-condition CSVs for one prior-research model into one frame
    with an exp2-style ``condition`` column."""
    frames = []
    for method, path in _model_csv_files(model).items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["method"] = method
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["mode"].isin(["source_only", "target_only", "mixed"])].copy()
    df["condition"] = df.apply(lambda r: _condition_tag(r["method"], r.get("ratio")), axis=1)
    df = df[df["condition"].notna()].copy()
    df = df[df["distance"].isin(["mmd", "dtw", "wasserstein"])]
    df = df[df["level"].isin(["in_domain", "out_domain"])]
    return df


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """For each (condition, mode), compute the minimum seed count across the
    6 (distance x level) subcells. Missing subcells count as 0."""
    rows = []
    distances = ["mmd", "dtw", "wasserstein"]
    levels = ["in_domain", "out_domain"]
    full_idx = pd.MultiIndex.from_product([distances, levels], names=["distance", "level"])
    for (cond, mode), g in df.groupby(["condition", "mode"]):
        per_sub = g.groupby(["distance", "level"])["seed"].nunique()
        per_sub = per_sub.reindex(full_idx, fill_value=0)
        rows.append({
            "condition": cond,
            "mode": mode,
            "min_seeds_subcell": int(per_sub.min()),
            "max_seeds_subcell": int(per_sub.max()),
            "total_obs": len(g),
        })
    return pd.DataFrame(rows).sort_values(["condition", "mode"])


def largest_balanced_subset(df: pd.DataFrame, n_min: int) -> tuple[pd.DataFrame, list, list, str]:
    """Return the largest balanced rectangular subset (R x M) such that every
    selected (condition, mode) cell has min subcell-seed-count >= n_min.

    Strategy:
      1. Compute (condition, mode) cells passing n_min.
      2. Build a binary matrix C x M.
      3. Greedy: try keeping all conditions, drop modes that disqualify any.
         If empty, drop conditions with worst coverage and retry.
      4. Pick the rectangle whose area (|conds| * |modes|) is maximal,
         preferring more modes when tied (to preserve M sensitivity).
    """
    cov = coverage_table(df)
    ok = cov[cov["min_seeds_subcell"] >= n_min]
    if ok.empty:
        return pd.DataFrame(), [], [], f"no (condition, mode) cell reaches {n_min} seeds in every (D x G) subcell"

    conds_all = sorted(ok["condition"].unique())
    modes_all = sorted(ok["mode"].unique())
    okset = set(map(tuple, ok[["condition", "mode"]].values))

    best = (0, [], [])  # (area, conds, modes)
    # Try every subset of modes (only 2^3 = 8 options)
    for nmodes in range(len(modes_all), 0, -1):
        from itertools import combinations
        for modes in combinations(modes_all, nmodes):
            # conditions that have ALL these modes covered
            conds_ok = [c for c in conds_all
                        if all((c, m) in okset for m in modes)]
            if not conds_ok:
                continue
            area = len(conds_ok) * len(modes)
            # Tiebreak: prefer more modes, then more conditions
            key = (area, len(modes), len(conds_ok))
            best_key = (best[0], len(best[2]), len(best[1]))
            if key > best_key:
                best = (area, conds_ok, list(modes))

    area, conds, modes = best
    if area == 0:
        return pd.DataFrame(), [], [], f"no balanced rectangle exists at n_min={n_min}"

    sub = df[df["condition"].isin(conds) & df["mode"].isin(modes)].copy()
    note = (f"balanced subset: {len(conds)} conditions x {len(modes)} modes "
            f"(min subcell seeds >= {n_min})")
    return sub, conds, modes, note


def run_sobol_for_model(model: str, n_min: int = N_MIN) -> dict | None:
    df_full = load_model_data(model)
    if df_full.empty:
        print(f"  [{model}] No data."); return None

    cov = coverage_table(df_full)
    sub, conds, modes, note = largest_balanced_subset(df_full, n_min)
    if sub.empty:
        # Try a relaxed n_min for diagnostics
        for n_relax in (8, 5, 3, 2):
            sub2, conds2, modes2, note2 = largest_balanced_subset(df_full, n_relax)
            if not sub2.empty:
                print(f"  [{model}] n_min={n_min} infeasible. Relaxed n_min={n_relax}: {note2}")
                sub, conds, modes, note = sub2, conds2, modes2, note2 + f" (relaxed from n_min={n_min})"
                break
        if sub.empty:
            print(f"  [{model}] No balanced subset even after relaxation. SKIP.")
            return {"model": model, "skipped": True, "coverage": cov,
                    "note": "No balanced subset; insufficient coverage."}

    print(f"  [{model}] {note}; conds={conds}; modes={modes}; N={len(sub)}")

    results = {}
    for metric, mlabel in PRIMARY_METRICS:
        if metric not in sub.columns:
            continue
        sub_m = sub.dropna(subset=[metric])
        if sub_m.empty:
            continue
        res = bootstrap_sobol(sub_m, metric, B=2000)
        results[metric] = res

    return {
        "model": model,
        "skipped": False,
        "coverage": cov,
        "subset_conditions": conds,
        "subset_modes": modes,
        "n_obs": len(sub),
        "note": note,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _exp2_sobol() -> pd.DataFrame | None:
    if EXP2_SOBOL_CSV.exists():
        return pd.read_csv(EXP2_SOBOL_CSV)
    return None


def _hierarchy(res_metric: dict) -> str:
    st = {f: res_metric[f"ST_{f}"][0] for f in FACTORS}
    sorted_f = sorted(st.items(), key=lambda x: -x[1])
    return " > ".join(f"{FACTOR_NAMES[f]}={v:.3f}" for f, v in sorted_f)


def build_report(per_model: dict) -> str:
    rf_sobol = _exp2_sobol()
    out = []
    w = out.append

    w("# Experiment 3 — Sensitivity Analysis on Completed Subset\n")
    w("Per-model Sobol-Hoeffding variance decomposition for the prior-research "
      "models (SvmW, SvmA, Lstm), restricted to the largest balanced rectangular "
      f"factorial subset with >= {N_MIN} seeds in every (distance x level) "
      "subcell.\n")
    w("Reference: Exp2 (RF) hierarchy was Mode > Rebalancing >> Distance ~ Membership, "
      "with R x M as the dominant interaction (>95% systematic variance share).\n")

    # Section per model
    for model in PRIOR_MODELS:
        info = per_model.get(model)
        w(f"\n## {model}\n")
        if info is None:
            w("_No data available._\n"); continue
        if info.get("skipped"):
            w(f"_Skipped: {info['note']}_\n")
            cov = info["coverage"]
            w("\n**Coverage table:**\n")
            w(cov.to_markdown(index=False))
            w("")
            continue

        w(f"- Subset: **{len(info['subset_conditions'])} conditions x "
          f"{len(info['subset_modes'])} modes** (N={info['n_obs']} obs)\n")
        w(f"- Conditions: `{info['subset_conditions']}`\n")
        w(f"- Modes: `{info['subset_modes']}`\n")
        w(f"- Note: {info['note']}\n")

        for metric, mlabel in PRIMARY_METRICS:
            res = info["results"].get(metric)
            if res is None:
                continue
            w(f"\n### {mlabel}\n")
            w(f"**Hierarchy (S_T):** {_hierarchy(res)}\n")
            header_cols = ["Factor", "Exp3 S_i", "Exp3 S_Ti"]
            if rf_sobol is not None:
                header_cols += ["Exp2 (RF) S_i", "Exp2 (RF) S_Ti", "ΔS_Ti (Exp3 − RF)"]
            w("| " + " | ".join(header_cols) + " |")
            w("|" + "|".join(["---"] * len(header_cols)) + "|")
            for f in FACTORS:
                s1, s1_lo, s1_hi = res[f"S1_{f}"]
                st, st_lo, st_hi = res[f"ST_{f}"]
                row = [
                    FACTOR_NAMES[f],
                    f"{s1:.4f} [{s1_lo:.4f}, {s1_hi:.4f}]",
                    f"{st:.4f} [{st_lo:.4f}, {st_hi:.4f}]",
                ]
                if rf_sobol is not None:
                    rf_row = rf_sobol[(rf_sobol["metric"] == mlabel) & (rf_sobol["factor_key"] == f)]
                    if not rf_row.empty:
                        rf_s1 = rf_row["S1"].values[0]
                        rf_st = rf_row["ST"].values[0]
                        row += [f"{rf_s1:.4f}", f"{rf_st:.4f}", f"{st - rf_st:+.4f}"]
                    else:
                        row += ["—", "—", "—"]
                w("| " + " | ".join(row) + " |")
            s_res, _, _ = res["S_residual"]
            w(f"\nResidual (seed): {s_res:.4f}")
            v_rm_tup = res.get("S2_conditionxmode")
            if v_rm_tup is not None:
                v_rm = v_rm_tup[0]
                w(f"R x M (S2) share: **{v_rm:.4f}** ({v_rm*100:.1f}% of total var)")
            w("")

        # Summary line
        if info["results"]:
            f2 = info["results"].get("f2")
            if f2:
                s_M = f2["S1_mode"][0]
                s_R = f2["S1_condition"][0]
                s_RM_tup = f2.get("S2_conditionxmode")
                s_RM = s_RM_tup[0] if s_RM_tup is not None else 0.0
                s_eps = f2["S_residual"][0]
                if (1 - s_eps) > 0:
                    share = (s_M + s_R + s_RM) / (1 - s_eps)
                    w(f"\n**F2 systematic share (M + R + R×M) / (1 - residual) = {share*100:.1f}%**  \n"
                      f"(Exp2 reference: 97.5%)\n")

    # Verdict
    w("\n## Verdict — Do Exp2 conclusions hold in Exp3?\n")
    summary_rows = []
    for model in PRIOR_MODELS:
        info = per_model.get(model)
        if not info or info.get("skipped"):
            summary_rows.append((model, "—", "—", "—", "insufficient coverage"))
            continue
        f2 = info["results"].get("f2")
        if not f2:
            summary_rows.append((model, "—", "—", "—", "no F2 data"))
            continue
        st = {f: f2[f"ST_{f}"][0] for f in FACTORS}
        top = max(st.items(), key=lambda x: x[1])[0]
        # Negligible check: D and G total-orders both < 0.05
        d_g_negligible = (st["distance"] < 0.05 and st["level"] < 0.05)
        rm_dom = st["condition"] > 0.10 and st["mode"] > 0.10
        verdict = "✓" if (top in ("mode", "condition") and d_g_negligible and rm_dom) else "partial"
        summary_rows.append((
            model,
            f"{st['mode']:.3f}",
            f"{st['condition']:.3f}",
            f"D={st['distance']:.3f}, G={st['level']:.3f}",
            verdict,
        ))
    w("| Model | S_T(M) | S_T(R) | S_T(D), S_T(G) | Hierarchy match? |")
    w("|---|---|---|---|---|")
    for r in summary_rows:
        w("| " + " | ".join(r) + " |")
    w("\n_Match criterion: S_T(M) and S_T(R) > 0.10; S_T(D) and S_T(G) < 0.05; "
      "top factor is M or R._\n")
    return "\n".join(out) + "\n"


def _decode_for_metric(info, metric):
    """(Unused; kept for backward compat.) Re-encode subset for SS computation."""
    df = load_model_data(info["model"])
    sub = df[df["condition"].isin(info["subset_conditions"]) &
             df["mode"].isin(info["subset_modes"])].dropna(subset=[metric])
    codes, level_counts = _encode_factors(sub)
    y = sub[metric].to_numpy()
    return y, codes, level_counts


# ---------------------------------------------------------------------------
def export_csv(per_model: dict):
    rows = []
    for model in PRIOR_MODELS:
        info = per_model.get(model)
        if not info or info.get("skipped"):
            continue
        for metric, mlabel in PRIMARY_METRICS:
            res = info["results"].get(metric)
            if res is None: continue
            for f in FACTORS:
                s1, s1_lo, s1_hi = res[f"S1_{f}"]
                st, st_lo, st_hi = res[f"ST_{f}"]
                rows.append({
                    "model": model, "metric": mlabel,
                    "factor": FACTOR_NAMES[f], "factor_key": f,
                    "S1": s1, "S1_lo": s1_lo, "S1_hi": s1_hi,
                    "ST": st, "ST_lo": st_lo, "ST_hi": st_hi,
                    "n_obs": info["n_obs"],
                    "n_conditions": len(info["subset_conditions"]),
                    "n_modes": len(info["subset_modes"]),
                })
    out_path = EXP3_SOBOL_OUT / "sobol_indices_exp3.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")

    # Combined with exp2 RF if available
    rf = _exp2_sobol()
    if rf is not None and rows:
        rf2 = rf.copy(); rf2["model"] = "RF (Exp2)"
        rf2["n_obs"] = np.nan
        rf2["n_conditions"] = 7
        rf2["n_modes"] = 3
        combined = pd.concat([rf2, pd.DataFrame(rows)], ignore_index=True, sort=False)
        comb_path = EXP3_SOBOL_OUT / "sobol_indices_exp3_combined.csv"
        combined.to_csv(comb_path, index=False)
        print(f"Saved: {comb_path.relative_to(PROJECT_ROOT)}")


def main():
    print("=" * 60)
    print("Sensitivity Analysis — Experiment 3 (Prior Research)")
    print(f"N_MIN = {N_MIN} seeds per (distance x level) subcell")
    print("=" * 60)
    per_model = {}
    for model in PRIOR_MODELS:
        print(f"\n--- {model} ---")
        per_model[model] = run_sobol_for_model(model, n_min=N_MIN)

    report = build_report(per_model)
    out_md = EXP3_REPORT_DIR / "sensitivity_analysis_report.md"
    out_md.write_text(report)
    print(f"\nReport: {out_md.relative_to(PROJECT_ROOT)}")
    export_csv(per_model)


if __name__ == "__main__":
    main()
