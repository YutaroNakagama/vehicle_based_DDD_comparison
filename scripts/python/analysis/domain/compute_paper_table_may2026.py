"""Compute paper Table tab:permode (Mean Performance by Rebalancing Strategy
× Training Mode) from May-2026 re-evaluated CSVs.

Aggregation rule
================
For each (strategy R, training mode M) cell, the reported value is the mean
across distance D ∈ {dtw, mmd, wasserstein}, domain G ∈ {in_domain,
out_domain}, and seed (12 seeds).  Cell sample size is therefore n = 72 per
cell when full coverage is available.

Mode mapping
------------
The paper's M ∈ {Within, Mixed, Cross} corresponds to the eval-JSON's
``mode`` ∈ {target_only, mixed, source_only}.

Strategy mapping
----------------
The paper's R rows correspond to the new CSV files as follows:

    Baseline                → baseline_NEW_may2026
    RUS r=0.1, r=0.5        → undersample_rus_NEW_may2026
    SMOTE r=0.1, r=0.5      → smote_plain_NEW_may2026
    SW-SMOTE r=0.1, r=0.5   → sw_smote_NEW_may2026 (imbalv3)

Outputs
-------
    results/analysis/exp2_domain_shift/figures/csv/split2/paper_table_NEW_may2026.csv
    results/analysis/exp2_domain_shift/figures/tex/split2/tab_permode_NEW_may2026.tex
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
CSV_BASE = ROOT / "results/analysis/exp2_domain_shift/figures/csv/split2"
OUT_CSV = CSV_BASE / "paper_table_NEW_may2026.csv"
OUT_TEX = ROOT / "results/analysis/exp2_domain_shift/figures/tex/split2" \
    / "tab_permode_NEW_may2026.tex"
OUT_TEX.parent.mkdir(parents=True, exist_ok=True)

MODE_MAP = {"target_only": "Within", "mixed": "Mixed", "source_only": "Cross"}

SOURCES: list[tuple[str, str | None, Path]] = [
    ("Baseline", None,
     CSV_BASE / "baseline_NEW_may2026/baseline_domain_split2_metrics_NEW_may2026.csv"),
    ("RUS", 0.1,
     CSV_BASE / "undersample_rus_NEW_may2026/undersample_rus_split2_metrics_NEW_may2026.csv"),
    ("RUS", 0.5,
     CSV_BASE / "undersample_rus_NEW_may2026/undersample_rus_split2_metrics_NEW_may2026.csv"),
    ("SMOTE", 0.1,
     CSV_BASE / "smote_plain_NEW_may2026/smote_plain_split2_metrics_NEW_may2026.csv"),
    ("SMOTE", 0.5,
     CSV_BASE / "smote_plain_NEW_may2026/smote_plain_split2_metrics_NEW_may2026.csv"),
    ("SW-SMOTE", 0.1,
     CSV_BASE / "sw_smote_NEW_may2026/sw_smote_split2_metrics_NEW_may2026.csv"),
    ("SW-SMOTE", 0.5,
     CSV_BASE / "sw_smote_NEW_may2026/sw_smote_split2_metrics_NEW_may2026.csv"),
]


def _label(strategy: str, ratio: float | None) -> str:
    if ratio is None:
        return strategy
    return f"{strategy} r={ratio}"


def _load_cell(path: Path, ratio: float | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if ratio is not None:
        df = df[df["ratio"] == ratio]
    return df


def main() -> None:
    rows: list[dict] = []
    for strat, ratio, path in SOURCES:
        if not path.exists():
            print(f"[WARN] missing {path}")
            continue
        df = _load_cell(path, ratio)
        for raw_mode, M in MODE_MAP.items():
            sub = df[df["mode"] == raw_mode]
            if sub.empty:
                continue
            rows.append({
                "strategy": _label(strat, ratio),
                "strategy_short": strat,
                "ratio": ratio if ratio is not None else "",
                "mode": M,
                "n": len(sub),
                "f2_mean": sub["f2"].mean(),
                "auc_mean": sub["auc"].mean(),
                "auc_pr_mean": sub["auc_pr"].mean(),
                "f2_std": sub["f2"].std(),
                "auc_std": sub["auc"].std(),
                "auc_pr_std": sub["auc_pr"].std(),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False, float_format="%.4f")
    print(f"[OK] {OUT_CSV} ({len(out)} rows)")

    # Wide pivot for the LaTeX table: index=strategy, columns=mode, value=metric
    def _wide(metric: str) -> pd.DataFrame:
        return (out.pivot_table(index="strategy", columns="mode",
                                values=f"{metric}_mean", aggfunc="first")
                   .reindex(columns=["Cross", "Within", "Mixed"]))

    f2 = _wide("f2")
    auc = _wide("auc")
    auc_pr = _wide("auc_pr")

    # Strategy display order (matches main.tex)
    order = ["Baseline", "RUS r=0.1", "RUS r=0.5",
             "SMOTE r=0.1", "SMOTE r=0.5",
             "SW-SMOTE r=0.1", "SW-SMOTE r=0.5"]
    f2 = f2.reindex(order)
    auc = auc.reindex(order)
    auc_pr = auc_pr.reindex(order)

    print("\n=== Within / Mixed / Cross averages (mean over D, G, seed) ===")
    print("\n-- F2 --");      print(f2.round(3))
    print("\n-- AUROC --");   print(auc.round(3))
    print("\n-- AUPRC --");   print(auc_pr.round(3))

    # LaTeX rows
    def _tex_label(s: str) -> str:
        if s.startswith("RUS"):
            return s.replace("RUS r=", "RUS $r\\!=\\!")
        if s.startswith("SW-SMOTE"):
            return s.replace("SW-SMOTE r=", "SW-SMOTE $r\\!=\\!")
        if s.startswith("SMOTE"):
            return s.replace("SMOTE r=", "SMOTE $r\\!=\\!")
        return s

    def _close_brace(s: str) -> str:
        return s + "$" if "$r\\!=\\!" in s else s

    # Find best AUROC for bolding
    best_auc_within = auc["Within"].idxmax()
    best_auc_mixed = auc["Mixed"].idxmax()
    best_f2_within = f2["Within"].idxmax()
    best_f2_mixed = f2["Mixed"].idxmax()
    best_pr_within = auc_pr["Within"].idxmax()
    best_pr_mixed = auc_pr["Mixed"].idxmax()

    def _fmt(v: float, bold: bool) -> str:
        s = f"{v:.3f}"
        return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"

    lines = []
    for s in order:
        cells = [
            _fmt(f2.loc[s, "Cross"], False),
            _fmt(f2.loc[s, "Within"], s == best_f2_within),
            _fmt(f2.loc[s, "Mixed"],  s == best_f2_mixed),
            _fmt(auc.loc[s, "Within"], s == best_auc_within),
            _fmt(auc.loc[s, "Mixed"],  s == best_auc_mixed),
            _fmt(auc_pr.loc[s, "Within"], s == best_pr_within),
            _fmt(auc_pr.loc[s, "Mixed"],  s == best_pr_mixed),
        ]
        label = _close_brace(_tex_label(s))
        lines.append(f"{label:<22} & " + " & ".join(cells) + r" \\")

    # Build full table snippet
    tex = (
        r"% Auto-generated by compute_paper_table_may2026.py — DO NOT EDIT BY HAND" "\n"
        r"\begin{tabular}{|l|c|cc|cc|cc|}" "\n"
        r"\hline" "\n"
        r" & & \multicolumn{2}{c|}{F2-score} & \multicolumn{2}{c|}{AUROC} & \multicolumn{2}{c|}{AUPRC} \\" "\n"
        r"Strategy & F2 (Cross) & Within & Mixed & Within & Mixed & Within & Mixed \\" "\n"
        r"\hline" "\n"
        + ("\n" + r"\hline" + "\n").join(lines) + "\n"
        r"\hline" "\n"
        r"\end{tabular}" "\n"
    )
    OUT_TEX.write_text(tex, encoding="utf-8")
    print(f"\n[OK] {OUT_TEX}")
    print("---- table snippet ----")
    print(tex)


if __name__ == "__main__":
    main()
