#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model" / "common"
NAMES_FILE = PROJECT_ROOT / "misc" / "pretrain_groups" / "group_names.txt"

# 出力
OUT_LONG = MODEL_DIR / "summary_6groups_only10_vs_finetune_long.csv"
OUT_WIDE = MODEL_DIR / "summary_6groups_only10_vs_finetune_wide.csv"
OUT_MD   = MODEL_DIR / "summary_6groups_only10_vs_finetune.md"

# 参照するテスト分割
SPLIT = "test"

# 集計対象メトリクス
METRICS = ["accuracy", "f1", "auc", "precision", "recall"]

def read_test_metrics(csv_path: Path):
    """metrics_*.csv を読み、split==test の行から必要メトリクスを返す。なければ None"""
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
    out = {}
    for m in METRICS:
        out[m] = float(row.get(m, float("nan")))
    return out

def main():
    # 6グループ名の読み込み
    if not NAMES_FILE.exists():
        raise FileNotFoundError(f"Group names file not found: {NAMES_FILE}")
    names = [ln.strip() for ln in NAMES_FILE.read_text().splitlines() if ln.strip()]
    # 期待ファイル名パターン:
    # only10:  metrics_RF_only10_{name}.csv
    # finetune: metrics_RF_finetune_{name}_finetune.csv  ← (finetuneタグに _finetune が付与される実装)
    rows = []

    # baseline（事前学習・対象指定なしのRF）※存在すれば拾う
    base_path = MODEL_DIR / "metrics_RF.csv"
    base_m = read_test_metrics(base_path)
    if base_m is not None:
        rows.append({
            "group": "baseline",
            "scheme": "baseline",
            **base_m,
            "source": base_path.name
        })

    for name in names:
        # only10
        only10_path = MODEL_DIR / f"metrics_RF_only10_{name}.csv"
        only10_m = read_test_metrics(only10_path)
        if only10_m is not None:
            rows.append({
                "group": name,
                "scheme": "only10",
                **only10_m,
                "source": only10_path.name
            })
        else:
            print(f"[WARN] missing only10 metrics: {only10_path.name}")

        # finetune
        finetune_path = MODEL_DIR / f"metrics_RF_finetune_{name}_finetune.csv"
        finetune_m = read_test_metrics(finetune_path)
        if finetune_m is not None:
            rows.append({
                "group": name,
                "scheme": "finetune",
                **finetune_m,
                "source": finetune_path.name
            })
        else:
            print(f"[WARN] missing finetune metrics: {finetune_path.name}")

    if not rows:
        raise RuntimeError("No metrics found. Check file names and paths.")

    # long形式
    long_df = pd.DataFrame(rows)
    # グループ順はファイル順を維持しつつ baseline を先頭に
    cat_order = ["baseline"] + names if "baseline" in long_df["group"].values else names
    long_df["group"] = pd.Categorical(long_df["group"], categories=cat_order, ordered=True)
    long_df = long_df.sort_values(["group", "scheme"]).reset_index(drop=True)
    long_df.to_csv(OUT_LONG, index=False)
    print(f"Saved: {OUT_LONG}")

    # wide形式（only10とfinetuneを横持ち & delta列）
    # ピボット
    wide_src = long_df[long_df["scheme"].isin(["only10", "finetune"])].copy()
    wide = wide_src.pivot_table(index="group", columns="scheme", values=METRICS, aggfunc="first")
    # 列階層をフラット化： metric_scheme
    wide.columns = [f"{m}_{sch}" for m, sch in wide.columns]
    wide = wide.reset_index()

    # delta列追加（finetune − only10）
    for m in METRICS:
        col_only = f"{m}_only10"
        col_fine = f"{m}_finetune"
        if col_only in wide.columns and col_fine in wide.columns:
            wide[f"{m}_delta"] = wide[col_fine] - wide[col_only]

    wide.to_csv(OUT_WIDE, index=False)
    print(f"Saved: {OUT_WIDE}")

    # Markdown（tabulate無しで生成）
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    md_lines = []
    md_lines.append("# Summary: only10 vs finetune (6 groups, test split)")
    md_lines.append("")
    md_lines.append("## Long format")
    md_cols = ["group", "scheme", *METRICS, "source"]
    md_lines.append("| " + " | ".join(md_cols) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(md_cols)) + " |")
    for _, r in long_df[md_cols].iterrows():
        md_lines.append("| " + " | ".join(fmt(r[c]) for c in md_cols) + " |")
    md_lines.append("")
    md_lines.append("## Wide format (with delta = finetune − only10)")
    wide_cols = ["group"]
    for m in METRICS:
        wide_cols += [f"{m}_only10", f"{m}_finetune", f"{m}_delta"]
    # 存在する列だけ出力
    wide_cols = [c for c in wide_cols if c in wide.columns]
    md_lines.append("| " + " | ".join(wide_cols) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(wide_cols)) + " |")
    for _, r in wide[wide_cols].iterrows():
        md_lines.append("| " + " | ".join(fmt(r[c]) for c in wide_cols) + " |")

    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved: {OUT_MD}")

if __name__ == "__main__":
    main()

# Radar chart visualization for 6 groups comparing finetune vs only10
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---- Config ----
# Default path (change if your file is elsewhere)
csv_path = "/home/s2240011/git/ddd/vehicle_based_DDD_comparison/project/model/common/summary_6groups_only10_vs_finetune_wide.csv"

# If the default path doesn't exist, try the alternate (older) long file; we will pivot it if needed
alt_long_path = "/home/s2240011/git/ddd/vehicle_based_DDD_comparison/project/model/common/summary_6groups_only10_vs_finetune_long.csv"

# Output directory
out_dir = "/mnt/data/radar_charts_6groups"
os.makedirs(out_dir, exist_ok=True)
pdf_path = os.path.join(out_dir, "radar_6groups_finetune_vs_only10.pdf")

# Required metric columns (wide format expected)
required_cols = [
    "group",
    "accuracy_finetune","accuracy_only10",
    "f1_finetune","f1_only10",
    "auc_finetune","auc_only10",
    "precision_finetune","precision_only10",
    "recall_finetune","recall_only10",
]

def load_or_transform():
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df
    elif os.path.exists(alt_long_path):
        # Try to pivot long -> wide (must contain 'group','scheme','metric','value')
        long_df = pd.read_csv(alt_long_path)
        # Normalize scheme labels to match "finetune" or "only10"
        def norm_scheme(s):
            s = str(s).lower()
            if "finetune" in s:
                return "finetune"
            if "only10" in s:
                return "only10"
            return s
        long_df["scheme_norm"] = long_df["scheme"].map(norm_scheme)
        # Keep only our five metrics
        keep = long_df["metric"].isin(["accuracy","f1","auc","precision","recall"])
        long_df = long_df[keep]
        wide = long_df.pivot_table(
            index="group",
            columns=["metric","scheme_norm"],
            values="value",
            aggfunc="mean"
        )
        # Flatten columns
        wide.columns = [f"{m}_{sch}" for (m, sch) in wide.columns]
        wide = wide.reset_index()
        # Rename 'group' index col
        wide.rename(columns={"group":"group"}, inplace=True)
        return wide
    else:
        raise FileNotFoundError("Could not find the summary CSV. Checked:\n"
                                f" - {csv_path}\n"
                                f" - {alt_long_path}\n")

df = load_or_transform()

# Basic validation
available_cols = df.columns.tolist()
missing = [c for c in required_cols if c not in available_cols]
if missing:
    print("The input file is missing required columns:")
    print(missing)
    print("\nAvailable columns:\n", available_cols)
    raise SystemExit

# Radar utilities
def radar_factory(num_vars):
    # Compute angle for each axis
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the loop
    return angles

metrics = ["accuracy","f1","auc","precision","recall"]
angles = radar_factory(len(metrics))

# Create a multi-page PDF
pdf = PdfPages(pdf_path)
saved_images = []

for _, row in df.iterrows():
    group = str(row["group"])

    fin_vals = [row[f"{m}_finetune"] for m in metrics]
    only_vals = [row[f"{m}_only10"] for m in metrics]

    # Close the polygon (repeat first value)
    fin_vals += fin_vals[:1]
    only_vals += only_vals[:1]

    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)

    # Angle ticks with metric names
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    # Set radial limits [0,1] assuming metrics are in [0,1]
    ax.set_ylim(0, 1.0)

    # Plot both series (no explicit colors)
    ax.plot(angles, fin_vals, linewidth=2, label="finetune")
    ax.fill(angles, fin_vals, alpha=0.1)
    ax.plot(angles, only_vals, linewidth=2, linestyle="--", label="only10")
    ax.fill(angles, only_vals, alpha=0.1)

    ax.set_title(f"Group: {group}", va="bottom")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    # Save figure to PNG and PDF
    png_path = os.path.join(out_dir, f"radar_{group}.png")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    saved_images.append(png_path)

pdf.close()

print("Saved individual images:")
for p in saved_images:
    print("-", p)
print("\nSaved combined PDF:", pdf_path)

