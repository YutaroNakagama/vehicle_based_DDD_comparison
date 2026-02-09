#!/usr/bin/env python3
"""
閾値固定 vs 閾値最適化の比較可視化スクリプト

不均衡対策単体実験において、閾値の違いによるseed毎の結果変動を可視化します。
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

def load_results(base_dir: Path):
    """評価結果を読み込む"""
    methods = [
        "baseline",
        "smote_ratio0.1",
        "smote_ratio0.5",
        "subjectwise_smote_ratio0.1",
        "subjectwise_smote_ratio0.5",
    ]
    seeds = ["s42", "s123"]
    
    results = []
    for method in methods:
        for seed in seeds:
            # 最適化閾値
            fname_opt = f"eval_results_RF_pooled_{method}_{seed}.json"
            fpath_opt = base_dir / fname_opt
            
            # 固定閾値 (0.5) - 新形式 th50
            fname_fix = f"eval_results_RF_pooled_{method}_{seed}_th50.json"
            fpath_fix = base_dir / fname_fix
            
            # 旧形式 th05 もフォールバックで確認
            if not fpath_fix.exists():
                fname_fix = f"eval_results_RF_pooled_{method}_{seed}_th05.json"
                fpath_fix = base_dir / fname_fix
            
            if fpath_opt.exists():
                with open(fpath_opt) as f:
                    d = json.load(f)
                
                # 閾値適用後のメトリクス (_thr サフィックス) を優先使用
                thr = d.get("thr", 0.5)
                acc = d.get("acc_thr", d.get("accuracy", 0))
                prec = d.get("prec_thr", d.get("precision", 0))
                rec = d.get("recall_thr", d.get("recall", 0))
                f1 = d.get("f1_thr", d.get("f1", 0))
                f2 = d.get("f2_thr", 0)
                
                # f2_thrがない場合はCMから計算
                if f2 == 0:
                    cm = d.get("confusion_matrix", [[0,0],[0,0]])
                    tn, fp = cm[0]
                    fn, tp = cm[1]
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    beta = 2
                    f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else 0
                
                results.append({
                    "method": method,
                    "seed": seed,
                    "threshold_type": "optimized",
                    "threshold_value": thr,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "f2": f2,
                })
            
            if fpath_fix.exists():
                with open(fpath_fix) as f:
                    d = json.load(f)
                
                # 閾値適用後のメトリクス (_thr サフィックス) を優先使用
                thr = d.get("thr", 0.5)
                acc = d.get("acc_thr", d.get("accuracy", 0))
                prec = d.get("prec_thr", d.get("precision", 0))
                rec = d.get("recall_thr", d.get("recall", 0))
                f1 = d.get("f1_thr", d.get("f1", 0))
                f2 = d.get("f2_thr", 0)
                
                # f2_thrがない場合はCMから計算
                if f2 == 0:
                    cm = d.get("confusion_matrix", [[0,0],[0,0]])
                    tn, fp = cm[0]
                    fn, tp = cm[1]
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    beta = 2
                    f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else 0
                
                results.append({
                    "method": method,
                    "seed": seed,
                    "threshold_type": "fixed_0.5",
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "f2": f2,
                })
    
    return results


def plot_threshold_comparison(results, output_path: Path):
    """閾値比較のプロット"""
    
    methods = [
        ("baseline", "Baseline"),
        ("smote_ratio0.1", "SMOTE r=0.1"),
        ("smote_ratio0.5", "SMOTE r=0.5"),
        ("subjectwise_smote_ratio0.1", "SW-SMOTE r=0.1"),
        ("subjectwise_smote_ratio0.5", "SW-SMOTE r=0.5"),
    ]
    
    metrics = ["accuracy", "precision", "recall", "f1", "f2"]
    seeds = ["s42", "s123"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 5つのメトリクスをプロット
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        x = np.arange(len(methods))
        width = 0.2
        
        # 各組み合わせのデータを取得
        for i, (seed, color_opt, color_fix) in enumerate([
            ("s42", "#3498db", "#85c1e9"),  # 青系
            ("s123", "#e74c3c", "#f1948a"),  # 赤系
        ]):
            vals_opt = []
            vals_fix = []
            for method_key, _ in methods:
                for r in results:
                    if r["method"] == method_key and r["seed"] == seed:
                        if r["threshold_type"] == "optimized":
                            vals_opt.append(r[metric])
                        elif r["threshold_type"] == "fixed_0.5":
                            vals_fix.append(r[metric])
            
            offset = width * (i * 2 - 1.5)
            ax.bar(x + offset, vals_opt, width, label=f"{seed} optimized", color=color_opt, edgecolor='black', linewidth=0.5)
            ax.bar(x + offset + width, vals_fix, width, label=f"{seed} fixed(0.5)", color=color_fix, edgecolor='black', linewidth=0.5, hatch='//')
        
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} by Method and Threshold')
        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in methods], rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # 6番目のプロットはseed間のばらつきを表示
    ax = axes[5]
    
    # seed間のばらつき（標準偏差的な差分）をメトリクス毎に表示
    for metric in ["f1", "f2", "recall"]:
        diffs_opt = []
        diffs_fix = []
        method_labels = []
        
        for method_key, method_label in methods:
            vals_opt = {seed: None for seed in seeds}
            vals_fix = {seed: None for seed in seeds}
            
            for r in results:
                if r["method"] == method_key:
                    if r["threshold_type"] == "optimized":
                        vals_opt[r["seed"]] = r[metric]
                    elif r["threshold_type"] == "fixed_0.5":
                        vals_fix[r["seed"]] = r[metric]
            
            if vals_opt["s42"] is not None and vals_opt["s123"] is not None:
                diffs_opt.append(abs(vals_opt["s42"] - vals_opt["s123"]))
            else:
                diffs_opt.append(0)
            
            if vals_fix["s42"] is not None and vals_fix["s123"] is not None:
                diffs_fix.append(abs(vals_fix["s42"] - vals_fix["s123"]))
            else:
                diffs_fix.append(0)
            
            method_labels.append(method_label)
    
    # 棒グラフでばらつきを表示
    x = np.arange(len(methods))
    width = 0.35
    
    # F2のばらつきのみ表示
    diffs_opt_f2 = []
    diffs_fix_f2 = []
    for method_key, _ in methods:
        vals_opt = {seed: None for seed in seeds}
        vals_fix = {seed: None for seed in seeds}
        
        for r in results:
            if r["method"] == method_key:
                if r["threshold_type"] == "optimized":
                    vals_opt[r["seed"]] = r["f2"]
                elif r["threshold_type"] == "fixed_0.5":
                    vals_fix[r["seed"]] = r["f2"]
        
        if vals_opt["s42"] is not None and vals_opt["s123"] is not None:
            diffs_opt_f2.append(abs(vals_opt["s42"] - vals_opt["s123"]))
        else:
            diffs_opt_f2.append(0)
        
        if vals_fix["s42"] is not None and vals_fix["s123"] is not None:
            diffs_fix_f2.append(abs(vals_fix["s42"] - vals_fix["s123"]))
        else:
            diffs_fix_f2.append(0)
    
    ax.bar(x - width/2, diffs_opt_f2, width, label='Optimized', color='#3498db')
    ax.bar(x + width/2, diffs_fix_f2, width, label='Fixed(0.5)', color='#e74c3c', hatch='//')
    ax.set_ylabel('|s42 - s123| (F2)')
    ax.set_title('Seed Variation in F2 Score')
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in methods], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_seed_comparison_detail(results, output_path: Path):
    """seed毎の詳細比較プロット"""
    
    methods = [
        ("baseline", "Baseline"),
        ("smote_ratio0.1", "SMOTE r=0.1"),
        ("smote_ratio0.5", "SMOTE r=0.5"),
        ("subjectwise_smote_ratio0.1", "SW-SMOTE r=0.1"),
        ("subjectwise_smote_ratio0.5", "SW-SMOTE r=0.5"),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics_to_plot = [
        ("f2", "F2 Score"),
        ("recall", "Recall"),
        ("precision", "Precision"),
        ("f1", "F1 Score"),
    ]
    
    for ax, (metric, metric_label) in zip(axes.flatten(), metrics_to_plot):
        # 線グラフで seed42 → seed123 の変化を表示
        for method_key, method_label in methods:
            vals_opt = []
            vals_fix = []
            
            for seed in ["s42", "s123"]:
                val_opt = None
                val_fix = None
                for r in results:
                    if r["method"] == method_key and r["seed"] == seed:
                        if r["threshold_type"] == "optimized":
                            val_opt = r[metric]
                        elif r["threshold_type"] == "fixed_0.5":
                            val_fix = r[metric]
                vals_opt.append(val_opt if val_opt is not None else 0)
                vals_fix.append(val_fix if val_fix is not None else 0)
            
            ax.plot([0, 1], vals_opt, 'o-', label=f'{method_label} (opt)', markersize=8)
            ax.plot([0, 1], vals_fix, 's--', label=f'{method_label} (fix)', markersize=8, alpha=0.7)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['seed=42', 'seed=123'])
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label}: Seed Comparison')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=7, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_summary_table(results):
    """結果のサマリーテーブルを出力"""
    print("\n" + "="*100)
    print("THRESHOLD COMPARISON SUMMARY")
    print("="*100)
    
    methods = [
        ("baseline", "Baseline"),
        ("smote_ratio0.1", "SMOTE r=0.1"),
        ("smote_ratio0.5", "SMOTE r=0.5"),
        ("subjectwise_smote_ratio0.1", "SW-SMOTE r=0.1"),
        ("subjectwise_smote_ratio0.5", "SW-SMOTE r=0.5"),
    ]
    
    print(f"\n{'Method':<25} | {'Seed':<6} | {'Threshold':<10} | {'Acc':>6} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'F2':>6}")
    print("-"*100)
    
    for method_key, method_label in methods:
        for seed in ["s42", "s123"]:
            for th_type in ["optimized", "fixed_0.5"]:
                for r in results:
                    if r["method"] == method_key and r["seed"] == seed and r["threshold_type"] == th_type:
                        print(f"{method_label:<25} | {seed:<6} | {th_type:<10} | {r['accuracy']:>6.3f} | {r['precision']:>6.3f} | {r['recall']:>6.3f} | {r['f1']:>6.3f} | {r['f2']:>6.3f}")
        print("-"*100)
    
    # ばらつき分析
    print("\n" + "="*100)
    print("SEED VARIATION ANALYSIS (|s42 - s123|)")
    print("="*100)
    print(f"\n{'Method':<25} | {'Threshold':<10} | {'F2 diff':>8} | {'Recall diff':>11} | {'F1 diff':>8}")
    print("-"*80)
    
    for method_key, method_label in methods:
        for th_type in ["optimized", "fixed_0.5"]:
            vals = {metric: {} for metric in ["f2", "recall", "f1"]}
            for r in results:
                if r["method"] == method_key and r["threshold_type"] == th_type:
                    for metric in ["f2", "recall", "f1"]:
                        vals[metric][r["seed"]] = r[metric]
            
            diffs = {}
            for metric in ["f2", "recall", "f1"]:
                if "s42" in vals[metric] and "s123" in vals[metric]:
                    diffs[metric] = abs(vals[metric]["s42"] - vals[metric]["s123"])
                else:
                    diffs[metric] = 0
            
            print(f"{method_label:<25} | {th_type:<10} | {diffs['f2']:>8.4f} | {diffs['recall']:>11.4f} | {diffs['f1']:>8.4f}")
        print("-"*80)


def main():
    parser = argparse.ArgumentParser(description="閾値固定 vs 最適化の比較可視化")
    parser.add_argument("--output-dir", type=str, 
                       default="results/analysis/exp1_imbalance/local/threshold_comparison",
                       help="出力ディレクトリ")
    args = parser.parse_args()
    
    base_dir = Path("results/outputs/evaluation/RF/local/local[1]")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading results...")
    results = load_results(base_dir)
    print(f"Loaded {len(results)} results")
    
    # サマリーテーブル出力
    print_summary_table(results)
    
    # 可視化
    print("\nGenerating plots...")
    plot_threshold_comparison(results, output_dir / "threshold_comparison.png")
    plot_seed_comparison_detail(results, output_dir / "seed_comparison_detail.png")
    
    print(f"\nDone! Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
