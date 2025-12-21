#!/usr/bin/env python3
"""
PBSログファイルからOptunaのF2スコア収束状況を抽出・可視化するスクリプト

Usage:
    python extract_f2_convergence_from_pbs_logs.py --log_dir PATH [--output_dir PATH]
    
例:
    # ランキング比較実験のログから抽出
    python scripts/python/extract_f2_convergence_from_pbs_logs.py \
        --log_dir scripts/hpc/log \
        --pattern "rank_cmp_" \
        --output_dir results/ranking_convergence
    
    # 不均衡対策実験のログから抽出
    python scripts/python/extract_f2_convergence_from_pbs_logs.py \
        --log_dir scripts/hpc/log \
        --pattern "imbal_v2_" \
        --output_dir results/imbalance_convergence
"""

import argparse
import re
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def parse_pbs_log(log_path: str) -> Tuple[Optional[str], List[Tuple[int, float, float]]]:
    """
    PBSログファイルからタグ名とトライアル情報を抽出
    
    Returns:
        tag: 実験タグ名
        trials: List of (trial_num, value, best_value)
    """
    tag = None
    mode = None
    trials = []
    
    # Optunaトライアルログのパターン
    # [Optuna] Trial   0: value=0.4700, best=0.4700
    trial_pattern = re.compile(
        r'\[Optuna\]\s+Trial\s+(\d+):\s+value=([\d.]+),\s+best=([\d.]+)'
    )
    
    # タグパターン (複数のフォーマットに対応)
    tag_patterns = [
        re.compile(r'tag=([^\s|]+)'),
        re.compile(r'Tag:\s*([^\s]+)'),
        re.compile(r'\[START\].*tag=([^\s|]+)'),
    ]
    
    # モードパターン
    mode_patterns = [
        re.compile(r'mode=(\w+)'),
        re.compile(r'Mode:\s*(\w+)'),
    ]
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # タグを抽出
                if tag is None:
                    for pattern in tag_patterns:
                        match = pattern.search(line)
                        if match:
                            tag = match.group(1)
                            break
                
                # モードを抽出
                if mode is None:
                    for pattern in mode_patterns:
                        match = pattern.search(line)
                        if match:
                            mode = match.group(1)
                            break
                
                # トライアル情報を抽出
                match = trial_pattern.search(line)
                if match:
                    trial_num = int(match.group(1))
                    value = float(match.group(2))
                    best = float(match.group(3))
                    trials.append((trial_num, value, best))
                    
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return None, []
    
    # モード情報をタグに追加（同じタグでsource_only/target_onlyが区別されない問題を解決）
    if tag and mode and mode not in tag:
        tag = f"{tag}_{mode}"
    
    return tag, trials


def find_pbs_logs(log_dir: str, pattern: Optional[str] = None) -> Dict[str, str]:
    """
    PBSログファイルを検索し、タグ名とログパスのマッピングを返す
    """
    log_files = glob.glob(os.path.join(log_dir, "*.OU"))
    # Also search for .log files (custom log files)
    log_files.extend(glob.glob(os.path.join(log_dir, "*.log")))
    
    results = {}
    for log_path in log_files:
        tag, trials = parse_pbs_log(log_path)
        
        if tag is None or len(trials) == 0:
            continue
            
        # パターンフィルタ
        if pattern and pattern not in tag:
            continue
            
        results[tag] = log_path
        
    return results


def extract_all_convergence_data(log_dir: str, pattern: Optional[str] = None) -> pd.DataFrame:
    """
    全ログファイルからF2収束データを抽出してDataFrameに統合
    """
    log_files = glob.glob(os.path.join(log_dir, "*.OU"))
    # Also search for .log files (custom log files)
    log_files.extend(glob.glob(os.path.join(log_dir, "*.log")))
    
    all_data = []
    
    for log_path in log_files:
        tag, trials = parse_pbs_log(log_path)
        
        if tag is None or len(trials) == 0:
            continue
            
        # パターンフィルタ
        if pattern and pattern not in tag:
            continue
        
        # タグから情報を抽出
        for trial_num, value, best in trials:
            all_data.append({
                'tag': tag,
                'trial': trial_num,
                'f2_score': value,
                'best_f2': best,
                'log_file': os.path.basename(log_path)
            })
    
    if not all_data:
        return pd.DataFrame()
        
    return pd.DataFrame(all_data)


def parse_ranking_tag(tag: str) -> Dict[str, str]:
    """
    ランキング比較実験のタグをパース
    例: rank_cmp_centroid_umap_dtw_out_domain_s42
    """
    result = {}
    
    # パターン: rank_cmp_{method}_{metric}_out_domain_s{seed}
    match = re.match(r'rank_cmp_(\w+)_(\w+)_out_domain_s(\d+)', tag)
    if match:
        result['method'] = match.group(1)
        result['metric'] = match.group(2)
        result['seed'] = match.group(3)
        return result
    
    # パターン: rank_cmp_{method}_{metric}_in_domain_s{seed}
    match = re.match(r'rank_cmp_(\w+)_(\w+)_in_domain_s(\d+)', tag)
    if match:
        result['method'] = match.group(1)
        result['metric'] = match.group(2)
        result['seed'] = match.group(3)
        result['domain'] = 'in_domain'
        return result
    
    return result


def parse_imbalance_tag(tag: str) -> Dict[str, str]:
    """
    不均衡対策実験のタグをパース
    例: imbal_v2_smote_balanced_rf_ratio0_1_seed42
    """
    result = {}
    
    # パターン: imbal_v2_{method}_ratio{ratio}_seed{seed}
    match = re.match(r'imbal_v2_(\w+)_ratio([\d_]+)_seed(\d+)', tag)
    if match:
        result['method'] = match.group(1)
        result['ratio'] = match.group(2).replace('_', '.')
        result['seed'] = match.group(3)
        return result
    
    return result


def plot_convergence_curves(df: pd.DataFrame, output_path: str, 
                            group_by: str = 'method',
                            title: str = 'F2 Score Convergence'):
    """
    F2スコアの収束曲線をプロット
    """
    if df.empty:
        print("No data to plot")
        return
    
    # グループごとにプロット
    groups = df.groupby('tag')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左: 全トライアルの値
    ax1 = axes[0]
    for tag, group in groups:
        ax1.plot(group['trial'], group['f2_score'], alpha=0.3, linewidth=0.8)
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('F2 Score')
    ax1.set_title(f'{title} - All Trials')
    ax1.grid(True, alpha=0.3)
    
    # 右: Best値の推移
    ax2 = axes[1]
    colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))
    for (tag, group), color in zip(groups, colors):
        short_tag = tag.split('_')[-2] if len(tag.split('_')) > 2 else tag[:20]
        ax2.plot(group['trial'], group['best_f2'], label=short_tag, 
                color=color, linewidth=1.5)
    ax2.set_xlabel('Trial')
    ax2.set_ylabel('Best F2 Score')
    ax2.set_title(f'{title} - Best F2 Progress')
    ax2.grid(True, alpha=0.3)
    
    # 凡例が多すぎる場合は非表示
    if len(groups) <= 15:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_convergence_by_method(df: pd.DataFrame, output_dir: str, experiment_type: str):
    """
    メソッド別にF2収束をプロット
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if experiment_type == 'ranking':
        # タグから情報を抽出
        df['method'] = df['tag'].apply(lambda x: parse_ranking_tag(x).get('method', 'unknown'))
        df['metric'] = df['tag'].apply(lambda x: parse_ranking_tag(x).get('metric', 'unknown'))
        df['seed'] = df['tag'].apply(lambda x: parse_ranking_tag(x).get('seed', '0'))
        
        # メソッド × メトリック別にプロット
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for idx, metric in enumerate(['dtw', 'mmd', 'wasserstein']):
                ax = axes[idx]
                metric_df = method_df[method_df['metric'] == metric]
                
                if metric_df.empty:
                    ax.set_title(f'{method} - {metric} (No data)')
                    continue
                
                for seed in metric_df['seed'].unique():
                    seed_df = metric_df[metric_df['seed'] == seed]
                    ax.plot(seed_df['trial'], seed_df['best_f2'], 
                           label=f'seed={seed}', linewidth=1.5, alpha=0.8)
                
                ax.set_xlabel('Trial')
                ax.set_ylabel('Best F2 Score')
                ax.set_title(f'{method} - {metric}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'F2 Convergence: {method}', fontsize=14)
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'convergence_{method}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_path}")
            
    elif experiment_type == 'imbalance':
        # タグから情報を抽出
        df['method'] = df['tag'].apply(lambda x: parse_imbalance_tag(x).get('method', 'unknown'))
        df['ratio'] = df['tag'].apply(lambda x: parse_imbalance_tag(x).get('ratio', '1.0'))
        df['seed'] = df['tag'].apply(lambda x: parse_imbalance_tag(x).get('seed', '0'))
        
        # メソッド別にプロット
        for method in df['method'].unique():
            method_df = df[df['method'] == method]
            
            n_ratios = len(method_df['ratio'].unique())
            fig, axes = plt.subplots(1, max(n_ratios, 1), figsize=(5 * max(n_ratios, 1), 5))
            
            if n_ratios == 1:
                axes = [axes]
            
            for idx, ratio in enumerate(sorted(method_df['ratio'].unique())):
                ax = axes[idx] if n_ratios > 1 else axes[0]
                ratio_df = method_df[method_df['ratio'] == ratio]
                
                for seed in ratio_df['seed'].unique():
                    seed_df = ratio_df[ratio_df['seed'] == seed]
                    ax.plot(seed_df['trial'], seed_df['best_f2'],
                           label=f'seed={seed}', linewidth=1.5, alpha=0.8)
                
                ax.set_xlabel('Trial')
                ax.set_ylabel('Best F2 Score')
                ax.set_title(f'ratio={ratio}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'F2 Convergence: {method}', fontsize=14)
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'convergence_{method}.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {output_path}")


def create_summary_table(df: pd.DataFrame, output_path: str):
    """
    収束状況のサマリーテーブルを作成
    """
    if df.empty:
        print("No data for summary")
        return
    
    # タグごとに最終的なbest_f2と収束したトライアル数を集計
    summary = df.groupby('tag').agg({
        'trial': 'max',
        'best_f2': 'max',
        'f2_score': ['mean', 'std']
    }).reset_index()
    
    summary.columns = ['tag', 'n_trials', 'best_f2', 'mean_f2', 'std_f2']
    summary = summary.sort_values('best_f2', ascending=False)
    
    # CSVとして保存
    summary.to_csv(output_path, index=False)
    print(f"Summary saved: {output_path}")
    
    # 上位10件を表示
    print("\n=== Top 10 by Best F2 ===")
    print(summary.head(10).to_string(index=False))
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Extract F2 convergence from PBS logs')
    parser.add_argument('--log_dir', type=str, required=True,
                       help='Directory containing PBS log files (.OU)')
    parser.add_argument('--pattern', type=str, default=None,
                       help='Filter pattern for experiment tags (e.g., "rank_cmp_", "imbal_v2_")')
    parser.add_argument('--output_dir', type=str, default='results/convergence',
                       help='Output directory for plots and CSVs')
    parser.add_argument('--experiment_type', type=str, choices=['ranking', 'imbalance', 'auto'],
                       default='auto', help='Experiment type for parsing tags')
    
    args = parser.parse_args()
    
    print(f"Scanning logs in: {args.log_dir}")
    print(f"Pattern filter: {args.pattern}")
    
    # データ抽出
    df = extract_all_convergence_data(args.log_dir, args.pattern)
    
    if df.empty:
        print("No matching logs found!")
        return
    
    print(f"\nFound {len(df['tag'].unique())} experiments with {len(df)} total trial records")
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 実験タイプを自動判定
    if args.experiment_type == 'auto':
        if args.pattern and 'rank' in args.pattern.lower():
            experiment_type = 'ranking'
        elif args.pattern and 'imbal' in args.pattern.lower():
            experiment_type = 'imbalance'
        elif any('rank_cmp' in t for t in df['tag'].unique()):
            experiment_type = 'ranking'
        else:
            experiment_type = 'imbalance'
    else:
        experiment_type = args.experiment_type
    
    print(f"Experiment type: {experiment_type}")
    
    # CSVとして生データを保存
    raw_csv_path = os.path.join(args.output_dir, 'convergence_raw.csv')
    df.to_csv(raw_csv_path, index=False)
    print(f"Raw data saved: {raw_csv_path}")
    
    # サマリーテーブル作成
    summary_path = os.path.join(args.output_dir, 'convergence_summary.csv')
    create_summary_table(df, summary_path)
    
    # 全体の収束プロット
    overall_path = os.path.join(args.output_dir, 'convergence_overall.png')
    plot_convergence_curves(df, overall_path, title=f'{experiment_type.title()} Experiments')
    
    # メソッド別プロット
    plot_convergence_by_method(df, args.output_dir, experiment_type)
    
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
