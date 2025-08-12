#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

METRICS = ["accuracy", "f1", "auc", "precision", "recall"]
SPLIT = "test"

def read_test_metrics(csv_path: Path):
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    df = df[df["split"] == SPLIT].copy()
    if df.empty:
        return None
    row = df.iloc[0]
    return {m: float(row.get(m, np.nan)) for m in METRICS}

def fmt(v):
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)

def make_radar(df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "radar_allgroups_finetune_vs_only10.pdf"
    metrics = ["accuracy","f1","auc","precision","recall"]
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    pdf = PdfPages(pdf_path)
    saved = []
    for _, row in df.iterrows():
        group = str(row["group"])
        fins = [row.get(f"{m}_finetune", np.nan) for m in metrics]
        onls = [row.get(f"{m}_only10",   np.nan) for m in metrics]
        # skip if all nan
        if all(np.isnan(fins)) and all(np.isnan(onls)):
            continue
        fins += fins[:1]; onls += onls[:1]
        fig = plt.figure(figsize=(6,6))
        ax = plt.subplot(111, polar=True)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.0)
        ax.plot(angles, fins, linewidth=2, label="finetune"); ax.fill(angles, fins, alpha=0.1)
        ax.plot(angles, onls, linewidth=2, linestyle="--", label="only10"); ax.fill(angles, onls, alpha=0.1)
        ax.set_title(f"Group: {group}", va="bottom"); ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
        png_path = out_dir / f"radar_{group}.png"
        fig.savefig(png_path, bbox_inches="tight", dpi=150)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        saved.append(png_path.name)
    pdf.close()
    print(f"[radar] Saved {len(saved)} images to {out_dir}")
    print(f"[radar] Combined PDF: {pdf_path}")

def main():
    ap = argparse.ArgumentParser(description="Summarize only10 vs finetune for arbitrary group list")
    prj = Path(__file__).resolve().parents[1]
    ap.add_argument("--names_file", default=str(prj / "misc" / "pretrain_groups" / "group_names_all.txt"))
    ap.add_argument("--model_dir",  default=str(prj / "model" / "common"))
    ap.add_argument("--out_prefix", default="summary_allgroups_only10_vs_finetune")
    ap.add_argument("--model", default="RF", help="model tag in metrics filename (default: RF)")
    ap.add_argument("--make_radar", action="store_true", help="also export radar charts for all groups")
    args = ap.parse_args()

    names_file = Path(args.names_file)
    model_dir  = Path(args.model_dir)
    out_long   = model_dir / f"{args.out_prefix}_long.csv"
    out_wide   = model_dir / f"{args.out_prefix}_wide.csv"
    out_md     = model_dir / f"{args.out_prefix}.md"

    if not names_file.exists():
        raise FileNotFoundError(f"Group names file not found: {names_file}")

    # group_names* の混入行などを除外
    names = [ln.strip() for ln in names_file.read_text().splitlines()
             if ln.strip() and not ln.strip().startswith("group_names")]

    rows = []

    # optional: baseline
    base_path = model_dir / f"metrics_{args.model}.csv"
    base_m = read_test_metrics(base_path)
    if base_m is not None:
        rows.append({"group": "baseline", "scheme": "baseline", **base_m, "source": base_path.name})

    for name in names:
        only10_path  = model_dir / f"metrics_{args.model}_only10_{name}.csv"
        finetune_path = model_dir / f"metrics_{args.model}_finetune_{name}_finetune.csv"

        only10_m = read_test_metrics(only10_path)
        finetune_m = read_test_metrics(finetune_path)

        if only10_m is not None:
            rows.append({"group": name, "scheme": "only10", **only10_m, "source": only10_path.name})
        else:
            print(f"[WARN] missing only10: {only10_path.name}")

        if finetune_m is not None:
            rows.append({"group": name, "scheme": "finetune", **finetune_m, "source": finetune_path.name})
        else:
            print(f"[WARN] missing finetune: {finetune_path.name}")

    if not rows:
        raise RuntimeError("No metrics found. Check files/paths.")

    long_df = pd.DataFrame(rows)
    cat_order = ["baseline"] + names if "baseline" in long_df["group"].values else names
    long_df["group"] = pd.Categorical(long_df["group"], categories=cat_order, ordered=True)
    long_df = long_df.sort_values(["group", "scheme"]).reset_index(drop=True)
    long_df.to_csv(out_long, index=False)
    print(f"Saved: {out_long}")

    # pivot to wide
    wide_src = long_df[long_df["scheme"].isin(["only10", "finetune"])].copy()
    wide = wide_src.pivot_table(index="group", columns="scheme", values=METRICS, aggfunc="first")
    wide.columns = [f"{m}_{sch}" for m, sch in wide.columns]
    wide = wide.reset_index()

    # deltas
    for m in METRICS:
        co, cf = f"{m}_only10", f"{m}_finetune"
        if co in wide.columns and cf in wide.columns:
            wide[f"{m}_delta"] = wide[cf] - wide[co]

    wide.to_csv(out_wide, index=False)
    print(f"Saved: {out_wide}")

    # improvements summary (counts & mean deltas)
    imp_rows = []
    for m in METRICS:
        dcol = f"{m}_delta"
        if dcol not in wide.columns: 
            continue
        vals = wide.loc[wide["group"] != "baseline", dcol] if "baseline" in wide["group"].values else wide[dcol]
        imp_rows.append({
            "metric": m,
            "n_groups": int(vals.notna().sum()),
            "n_improved": int((vals >  0).sum()),
            "n_tied":     int((vals == 0).sum()),
            "n_worse":    int((vals <  0).sum()),
            "mean_delta": float(vals.mean()),
            "median_delta": float(vals.median()),
        })
    improve_df = pd.DataFrame(imp_rows)
    imp_csv = model_dir / f"{args.out_prefix}_improvement_summary.csv"
    improve_df.to_csv(imp_csv, index=False)
    print(f"Saved: {imp_csv}")

    # Markdown (long + wide head + improvements)
    lines = []
    lines.append(f"# Summary: only10 vs finetune (groups={len(names)}, split={SPLIT})\n")
    lines.append("## Improvements summary\n")
    head = ["metric","n_groups","n_improved","n_tied","n_worse","mean_delta","median_delta"]
    lines += ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"]*len(head)) + " |"]
    for _, r in improve_df[head].iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in head) + " |")

    lines.append("\n## Wide (excerpt)\n")
    show_cols = ["group"]
    for m in METRICS:
        for suff in ["only10","finetune","delta"]:
            col = f"{m}_{suff}"
            if col in wide.columns:
                show_cols.append(col)
    lines += ["| " + " | ".join(show_cols) + " |", "| " + " | ".join(["---"]*len(show_cols)) + " |"]
    for _, r in wide[show_cols].iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in show_cols) + " |")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out_md}")

    if args.make_radar:
        make_radar(wide, model_dir / "radar_allgroups")

if __name__ == "__main__":
    main()

