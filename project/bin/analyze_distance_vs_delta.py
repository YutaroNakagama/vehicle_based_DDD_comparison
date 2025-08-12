#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
距離と性能差(Δ)の相関分析
- 各グループ G (10名) と補集合 U の平均距離 d(U,G) を距離行列から算出
- G 内の平均ペア距離 disp(G) も算出（群内ばらつき）
- summary_6groups_only10_vs_finetune_long/wide の CSV から Δ 指標を読み込み
- d(U,G) と各 Δ(accuracy,f1,auc,precision,recall) の相関(pearson/spearman)を出力
- 散布図(PNG)を保存
"""

import argparse
import os
import pandas as pd
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def read_group_members(groups_dir, group_names_file):
    names = [ln.strip() for ln in open(group_names_file, encoding="utf-8") if ln.strip()]
    groups = {}
    for name in names:
        path = os.path.join(groups_dir, f"{name}.txt")
        with open(path, encoding="utf-8") as f:
            members = [x for x in f.read().split() if x]
        groups[name] = members
    return groups

def mean_cross_distance(D: pd.DataFrame, A: list, B: list):
    # 共通被験者に制限
    A2 = [a for a in A if a in D.index]
    B2 = [b for b in B if b in D.columns]
    if len(A2) == 0 or len(B2) == 0:
        return np.nan
    sub = D.loc[A2, B2].values
    return float(np.nanmean(sub))

def mean_within_distance(D: pd.DataFrame, A: list):
    A2 = [a for a in A if a in D.index]
    if len(A2) < 2:
        return np.nan
    vals = []
    for i, j in combinations(A2, 2):
        vals.append(D.at[i, j] if (i in D.index and j in D.columns) else np.nan)
        vals.append(D.at[j, i] if (j in D.index and i in D.columns) else np.nan)
    vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else np.nan

def load_summary(summary_csv):
    # wide でも long でもOKにする。wide を推奨。
    df = pd.read_csv(summary_csv)
    cols_lower = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols_lower)
    # wide 形式を想定：group, *_finetune, *_only10, *_delta
    needed_delta = ["accuracy_delta","f1_delta","auc_delta","precision_delta","recall_delta"]
    for c in needed_delta:
        if c not in df.columns:
            raise ValueError(f"{summary_csv} に {c} 列がありません。5指標対応版を指定してください。")
    # group 名
    if "group" not in df.columns:
        raise ValueError("summary に group 列がありません。")
    # group名をキーとして返す
    return df[["group"] + needed_delta].copy()

def annotate_points(ax, x, y, labels):
    for xi, yi, lb in zip(x, y, labels):
        ax.text(xi, yi, lb, fontsize=9, ha="left", va="bottom")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True, help="5指標対応版の wide CSV（例: model/common/summary_6groups_only10_vs_finetune_wide.csv）")
    ap.add_argument("--distance_csv", required=True, help="距離行列CSV（ID×IDの正方行列）")
    ap.add_argument("--groups_dir", required=True, help="グループメンバーtxtが置いてあるディレクトリ（misc/pretrain_groups）")
    ap.add_argument("--group_names_file", required=True, help="group_names.txt（6行：mmd_friendly など）")
    ap.add_argument("--outdir", default="model/common/dist_corr", help="出力先フォルダ")
    ap.add_argument("--subject_list", default=None, help="全被験者リストtxt（未指定なら距離行列のindex/columnsから推定）")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 距離行列読み込み
    D = pd.read_csv(args.distance_csv, index_col=0)
    # 余分な空白を削る
    D.index = D.index.str.strip()
    D.columns = D.columns.str.strip()

    # 被験者全集合
    if args.subject_list and os.path.exists(args.subject_list):
        all_subjects = [ln.strip() for ln in open(args.subject_list, encoding="utf-8") if ln.strip()]
    else:
        # 行列に出てくるIDの和集合
        all_subjects = sorted(set(D.index.tolist()) | set(D.columns.tolist()))

    # グループ読み込み
    groups = read_group_members(args.groups_dir, args.group_names_file)

    # summary（Δ指標）
    summary = load_summary(args.summary_csv)
    # group名の小文字/無駄空白を吸収
    summary["group_norm"] = summary["group"].str.strip().str.lower()

    # グループごとの距離特徴量
    rows = []
    for name, members in groups.items():
        G = members
        U = [s for s in all_subjects if s not in G]
        d_u_g = mean_cross_distance(D, U, G)
        disp_g = mean_within_distance(D, G)
        rows.append({"group": name, "d_UG": d_u_g, "disp_G": disp_g})

    dist_df = pd.DataFrame(rows)
    dist_df["group_norm"] = dist_df["group"].str.strip().str.lower()

    # マージ
#    merged = dist_df.merge(summary, on="group_norm", how="left", suffixes=("",""))
    summary_no_group = summary.drop(columns=["group"], errors="ignore")
    merged = dist_df.merge(summary_no_group, on="group_norm", how="left")
    # 出力：距離特徴＋Δを一覧
    merged_out = merged.drop(columns=["group_norm"]).copy()
    merged_csv = os.path.join(args.outdir, "distance_vs_delta_merged.csv")
    merged_out.to_csv(merged_csv, index=False)
    print(f"Saved: {merged_csv}")

    # 相関（Pearson / Spearman）
    metrics = ["accuracy_delta","f1_delta","auc_delta","precision_delta","recall_delta"]
    corr_rows = []
    for m in metrics:
        x = merged["d_UG"].values
        y = merged[m].values
        if np.all(np.isnan(x)) or np.all(np.isnan(y)):
            p_r = np.nan; p_p = np.nan; s_r = np.nan; s_p = np.nan
        else:
            # 欠損を除外
            mask = ~(np.isnan(x) | np.isnan(y))
            xv, yv = x[mask], y[mask]
            if len(xv) >= 3:
                p_r, p_p = pearsonr(xv, yv)
                s_r, s_p = spearmanr(xv, yv)
            else:
                p_r = p_p = s_r = s_p = np.nan
        corr_rows.append({
            "metric": m,
            "pearson_r": p_r, "pearson_p": p_p,
            "spearman_rho": s_r, "spearman_p": s_p
        })

    corr_df = pd.DataFrame(corr_rows)
    corr_csv = os.path.join(args.outdir, "correlations_dUG_vs_deltas.csv")
    corr_df.to_csv(corr_csv, index=False)
    print(f"Saved: {corr_csv}")

    # 参考：disp_G と Δ の相関も
    corr_rows2 = []
    for m in metrics:
        x = merged["disp_G"].values
        y = merged[m].values
        mask = ~(np.isnan(x) | np.isnan(y))
        xv, yv = x[mask], y[mask]
        if len(xv) >= 3:
            p_r, p_p = pearsonr(xv, yv)
            s_r, s_p = spearmanr(xv, yv)
        else:
            p_r = p_p = s_r = s_p = np.nan
        corr_rows2.append({
            "metric": m,
            "pearson_r": p_r, "pearson_p": p_p,
            "spearman_rho": s_r, "spearman_p": s_p
        })
    corr2_df = pd.DataFrame(corr_rows2)
    corr2_csv = os.path.join(args.outdir, "correlations_dispG_vs_deltas.csv")
    corr2_df.to_csv(corr2_csv, index=False)
    print(f"Saved: {corr2_csv}")

    # 散布図（PNG）: d(U,G) vs Δ
    for m, label in [
        ("accuracy_delta", "Δ Accuracy (finetune − only10)"),
        ("f1_delta", "Δ F1"),
        ("auc_delta", "Δ AUC"),
        ("precision_delta", "Δ Precision"),
        ("recall_delta", "Δ Recall"),
    ]:
        x = merged["d_UG"].values
        y = merged[m].values
        labs = merged["group"].values
        mask = ~(np.isnan(x) | np.isnan(y))
        xv, yv, lv = x[mask], y[mask], labs[mask]
        plt.figure(figsize=(6,4))
        plt.scatter(xv, yv)
        # 最小二乗の回帰直線（点が3以上のとき）
        if len(xv) >= 2:
            a, b = np.polyfit(xv, yv, 1)
            xs = np.linspace(min(xv), max(xv), 100)
            ys = a*xs + b
            plt.plot(xs, ys)
        annotate_points(plt.gca(), xv, yv, lv)
        plt.xlabel("Mean distance d(U, G)")
        plt.ylabel(label)
        plt.title(f"d(U,G) vs {label}")
        plt.tight_layout()
        out_png = os.path.join(args.outdir, f"scatter_dUG_vs_{m}.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"Saved: {out_png}")

    # 参考：disp(G) vs Δ も描く
    for m, label in [
        ("accuracy_delta", "Δ Accuracy"),
        ("f1_delta", "Δ F1"),
        ("auc_delta", "Δ AUC"),
        ("precision_delta", "Δ Precision"),
        ("recall_delta", "Δ Recall"),
    ]:
        x = merged["disp_G"].values
        y = merged[m].values
        labs = merged["group"].values
        mask = ~(np.isnan(x) | np.isnan(y))
        xv, yv, lv = x[mask], y[mask], labs[mask]
        plt.figure(figsize=(6,4))
        plt.scatter(xv, yv)
        if len(xv) >= 2:
            a, b = np.polyfit(xv, yv, 1)
            xs = np.linspace(min(xv), max(xv), 100)
            ys = a*xs + b
            plt.plot(xs, ys)
        annotate_points(plt.gca(), xv, yv, lv)
        plt.xlabel("Within-group dispersion disp(G)")
        plt.ylabel(label)
        plt.title(f"disp(G) vs {label}")
        plt.tight_layout()
        out_png = os.path.join(args.outdir, f"scatter_dispG_vs_{m}.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"Saved: {out_png}")

if __name__ == "__main__":
    main()

