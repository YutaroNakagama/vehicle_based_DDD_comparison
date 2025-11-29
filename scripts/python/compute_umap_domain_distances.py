#!/usr/bin/env python3
"""
MDS/t-SNE/UMAPを使ったドメイン中心からの距離計算と可視化

計算内容:
1. 各グループのドメイン中心から、そのグループの被験者それぞれの距離の平均
2. 各グループのドメイン中心から、Middleグループのドメイン中心までの距離
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS, TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

# パス設定
BASE_DIR = Path(__file__).resolve().parents[2]
DISTANCE_DIR = BASE_DIR / "results" / "domain_analysis" / "distance"
OUTPUT_DIR = DISTANCE_DIR / "group-wise" / "intergroup_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["dtw_mean", "mmd_mean", "wasserstein_mean"]
METRIC_DIRS = {"dtw_mean": "dtw", "mmd_mean": "mmd", "wasserstein_mean": "wasserstein"}
LEVELS = ["high", "middle", "low"]


def load_distance_matrix(metric: str) -> np.ndarray:
    """距離行列を読み込む"""
    metric_dir = METRIC_DIRS[metric]
    matrix_path = DISTANCE_DIR / "subject-wise" / metric_dir / f"{metric_dir}_matrix.npy"
    return np.load(matrix_path)


def load_group_subjects(metric: str, level: str, ranking_method: str = "mean_distance") -> list:
    """グループの被験者リストを読み込む
    
    Parameters
    ----------
    metric : str
        距離指標 (dtw_mean, mmd_mean, wasserstein_mean)
    level : str
        グループレベル (high, middle, low)
    ranking_method : str
        ランキング手法 (mean_distance, centroid_mds, centroid_umap, medoid, lof)
    """
    # 新しいフォルダ構造: ranks29/{ranking_method}/{metric}_{level}.txt
    # metric から "_mean" サフィックスを除去
    metric_base = metric.replace("_mean", "")
    group_path = DISTANCE_DIR / "subject-wise" / "ranks" / "ranks29" / ranking_method / f"{metric_base}_{level}.txt"
    
    # フォールバック: 古い形式 (mean_distance_legacy)
    if not group_path.exists():
        group_path = DISTANCE_DIR / "subject-wise" / "ranks" / "ranks29" / "mean_distance_legacy" / f"{metric}_{level}.txt"
    
    with open(group_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_all_subjects(metric: str) -> list:
    """全被験者のリストを読み込む"""
    metric_dir = METRIC_DIRS[metric]
    subject_path = DISTANCE_DIR / "subject-wise" / metric_dir / f"{metric_dir}_subjects.json"
    with open(subject_path, "r") as f:
        return json.load(f)


def get_group_indices(all_subjects: list, group_subjects: list) -> np.ndarray:
    """グループに属する被験者のインデックスを取得"""
    indices = []
    for subj in group_subjects:
        try:
            idx = all_subjects.index(subj)
            indices.append(idx)
        except ValueError:
            print(f"Warning: Subject {subj} not found in all_subjects")
    return np.array(indices)


def compute_projection_with_domain_distances(dist_matrix: np.ndarray, 
                                              group_indices_dict: dict,
                                              method: str = "umap",
                                              n_components: int = 2,
                                              random_state: int = 42) -> dict:
    """次元削減でドメイン中心からの距離を計算
    
    Parameters
    ----------
    method : str
        "mds", "tsne", or "umap"
    
    Returns
    -------
    dict
        - coords: 投影後の座標
        - group_centroids: 各グループの重心座標とメトリクス
        - intra_domain_distances: グループ内の被験者からグループ重心までの平均距離
        - inter_domain_distances: 各グループ重心からMiddle重心までの距離
    """
    method = method.lower()
    
    if method == "mds":
        print("  Running MDS embedding...")
        reducer = MDS(
            n_components=n_components,
            dissimilarity="precomputed",
            random_state=random_state,
            n_init=4,
            max_iter=300
        )
        coords = reducer.fit_transform(dist_matrix)
        print("  ✓ MDS embedding completed")
        
    elif method == "tsne":
        print("  Running t-SNE embedding...")
        reducer = TSNE(
            n_components=n_components,
            metric="precomputed",
            init="random",
            random_state=random_state,
            perplexity=min(30, len(dist_matrix) - 1),
            n_iter=1000
        )
        coords = reducer.fit_transform(dist_matrix)
        print("  ✓ t-SNE embedding completed")
        
    elif method == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        print("  Running UMAP embedding...")
        reducer = umap.UMAP(
            n_components=n_components,
            metric="precomputed",
            random_state=random_state,
            n_neighbors=min(15, len(dist_matrix) - 1),
            verbose=False
        )
        coords = reducer.fit_transform(dist_matrix)
        print("  ✓ UMAP embedding completed")
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'mds', 'tsne', or 'umap'")
    
    # 各グループの重心（ドメイン中心）を計算
    group_centroids = {}
    for level, indices in group_indices_dict.items():
        group_coords = coords[indices]
        centroid = np.mean(group_coords, axis=0)
        group_centroids[level] = {
            "coordinates": centroid,
            "indices": indices,
            "coords": group_coords
        }
    
    # 1. グループ内距離: 各グループのドメイン中心から、そのグループの被験者それぞれの距離の平均
    intra_domain_distances = {}
    for level, data in group_centroids.items():
        centroid = data["coordinates"]
        group_coords = data["coords"]
        
        # 各被験者からグループ重心までの距離
        distances = np.linalg.norm(group_coords - centroid, axis=1)
        
        intra_domain_distances[level] = {
            "mean": float(np.mean(distances)),
            "std": float(np.std(distances)),
            "min": float(np.min(distances)),
            "max": float(np.max(distances)),
            "distances": distances.tolist()
        }
    
    # 2. グループ間距離: 各グループのドメイン中心から、Middleグループのドメイン中心までの距離
    middle_centroid = group_centroids["middle"]["coordinates"]
    
    inter_domain_distances = {}
    for level in LEVELS:
        level_centroid = group_centroids[level]["coordinates"]
        distance = float(np.linalg.norm(level_centroid - middle_centroid))
        inter_domain_distances[level] = distance
    
    return {
        "coords": coords,
        "group_centroids": {
            level: {
                "coordinates": data["coordinates"].tolist(),
                "n_subjects": len(data["indices"])
            }
            for level, data in group_centroids.items()
        },
        "intra_domain_distances": intra_domain_distances,
        "inter_domain_distances": inter_domain_distances
    }


def visualize_projection_with_distances(metric: str,
                                         method: str,
                                         projection_results: dict, 
                                         group_indices_dict: dict):
    """次元削減投影と距離メトリクスを可視化"""
    coords = projection_results["coords"]
    group_centroids = projection_results["group_centroids"]
    intra_distances = projection_results["intra_domain_distances"]
    inter_distances = projection_results["inter_domain_distances"]
    
    method_upper = method.upper()
    
    fig = plt.figure(figsize=(18, 12))
    
    # レイアウト: 上段に投影プロット、下段に距離メトリクス
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
    
    # === 上段: 投影プロット ===
    ax_proj = fig.add_subplot(gs[0, :])
    
    colors = {"high": "red", "middle": "gray", "low": "blue"}
    markers = {"high": "^", "middle": "s", "low": "v"}
    
    # 各グループの被験者をプロット
    for level, indices in group_indices_dict.items():
        group_coords = coords[indices]
        ax_proj.scatter(
            group_coords[:, 0], group_coords[:, 1],
            c=colors[level], marker=markers[level],
            s=100, alpha=0.6, 
            label=f"{level.capitalize()} subjects (n={len(indices)})",
            edgecolors='black', linewidth=0.5
        )
    
    # 各グループのドメイン中心をプロット
    for level, centroid_data in group_centroids.items():
        c = np.array(centroid_data["coordinates"])
        ax_proj.scatter(
            c[0], c[1], 
            c=colors[level], marker='*',
            s=1200, edgecolors='black', linewidth=3,
            label=f"{level.capitalize()} domain center", 
            zorder=10
        )
        
        # グループ名をラベル
        ax_proj.text(
            c[0], c[1] - 0.5, level.upper(),
            fontsize=12, ha='center', fontweight='bold',
            color=colors[level]
        )
    
    # Middle重心から他の重心への線を描画
    middle_centroid = np.array(group_centroids["middle"]["coordinates"])
    for level in ["high", "low"]:
        level_centroid = np.array(group_centroids[level]["coordinates"])
        ax_proj.plot(
            [middle_centroid[0], level_centroid[0]],
            [middle_centroid[1], level_centroid[1]],
            'k--', alpha=0.4, linewidth=2
        )
        
        # 距離をラベル表示
        mid_x = (middle_centroid[0] + level_centroid[0]) / 2
        mid_y = (middle_centroid[1] + level_centroid[1]) / 2
        dist = inter_distances[level]
        ax_proj.text(
            mid_x, mid_y, f"{dist:.2f}",
            fontsize=11, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', 
                     facecolor='yellow', alpha=0.7, edgecolor='black')
        )
    
    ax_proj.set_xlabel(f"{method_upper} Component 1", fontsize=14)
    ax_proj.set_ylabel(f"{method_upper} Component 2", fontsize=14)
    ax_proj.set_title(
        f"{method_upper} Projection with Domain Centers - {metric.upper()}\n"
        f"Distance from each domain center to subjects and Middle domain center",
        fontsize=16, fontweight='bold'
    )
    ax_proj.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
    ax_proj.grid(True, alpha=0.3)
    
    # === 下段左: グループ内距離（ドメイン中心から被験者への平均距離） ===
    ax_intra = fig.add_subplot(gs[1, 0])
    
    levels_list = list(LEVELS)
    intra_means = [intra_distances[level]["mean"] for level in levels_list]
    intra_stds = [intra_distances[level]["std"] for level in levels_list]
    
    bars = ax_intra.bar(
        range(len(levels_list)), intra_means, yerr=intra_stds,
        color=[colors[level] for level in levels_list],
        edgecolor='black', linewidth=2, capsize=10, alpha=0.7
    )
    
    # 数値をバーの上に表示
    for i, (level, mean, std) in enumerate(zip(levels_list, intra_means, intra_stds)):
        ax_intra.text(
            i, mean + std + 0.05 * max(intra_means), 
            f"{mean:.3f}\n±{std:.3f}",
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    
    ax_intra.set_xticks(range(len(levels_list)))
    ax_intra.set_xticklabels([l.capitalize() for l in levels_list], fontsize=12)
    ax_intra.set_ylabel("Average Distance", fontsize=13)
    ax_intra.set_title(
        "Intra-Domain Distance\n"
        "(Domain center → Subjects in same group)",
        fontsize=13, fontweight='bold'
    )
    ax_intra.grid(axis='y', alpha=0.3)
    
    # === 下段右: グループ間距離（各ドメイン中心からMiddle中心への距離） ===
    ax_inter = fig.add_subplot(gs[1, 1])
    
    inter_values = [inter_distances[level] for level in levels_list]
    
    bars = ax_inter.bar(
        range(len(levels_list)), inter_values,
        color=[colors[level] for level in levels_list],
        edgecolor='black', linewidth=2, alpha=0.7
    )
    
    # 数値をバーの上に表示
    for i, (level, value) in enumerate(zip(levels_list, inter_values)):
        ax_inter.text(
            i, value + 0.05 * max(inter_values), 
            f"{value:.3f}",
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    
    ax_inter.set_xticks(range(len(levels_list)))
    ax_inter.set_xticklabels([l.capitalize() for l in levels_list], fontsize=12)
    ax_inter.set_ylabel("Distance to Middle Center", fontsize=13)
    ax_inter.set_title(
        "Inter-Domain Distance\n"
        "(Domain center → Middle domain center)",
        fontsize=13, fontweight='bold'
    )
    ax_inter.grid(axis='y', alpha=0.3)
    
    # Middle自身は0なので、特別にマーク
    ax_inter.text(
        1, inter_values[1] + 0.02, "Self",
        ha='center', va='bottom', fontsize=10, 
        style='italic', color='gray'
    )
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"{metric}_{method}_domain_distances.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_path}")


def create_summary_table(all_results: dict, method: str):
    """全ての距離メトリクスをまとめたテーブルを作成"""
    
    # データフレーム用のデータを準備
    data = []
    
    for metric in METRICS:
        results = all_results[metric]
        intra = results["intra_domain_distances"]
        inter = results["inter_domain_distances"]
        
        for level in LEVELS:
            row = {
                "Metric": metric.replace("_mean", "").upper(),
                "Group": level.capitalize(),
                "Intra_Mean": intra[level]["mean"],
                "Intra_Std": intra[level]["std"],
                "Inter_to_Middle": inter[level]
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    df = df.round(4)
    
    # CSV保存
    csv_path = OUTPUT_DIR / f"{method}_domain_distances_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved summary CSV: {csv_path}")
    
    return df


def visualize_comparison_across_metrics(all_results: dict, method: str):
    """3つの距離指標を比較する可視化"""
    
    method_upper = method.upper()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Domain Distance Comparison Across Metrics ({method_upper})", 
                 fontsize=18, fontweight='bold')
    
    colors = {"high": "red", "middle": "gray", "low": "blue"}
    
    for col_idx, metric in enumerate(METRICS):
        results = all_results[metric]
        intra = results["intra_domain_distances"]
        inter = results["inter_domain_distances"]
        
        metric_name = metric.replace("_mean", "").upper()
        
        # 上段: Intra-domain距離
        ax_intra = axes[0, col_idx]
        levels_list = list(LEVELS)
        intra_means = [intra[level]["mean"] for level in levels_list]
        intra_stds = [intra[level]["std"] for level in levels_list]
        
        ax_intra.bar(
            range(len(levels_list)), intra_means, yerr=intra_stds,
            color=[colors[level] for level in levels_list],
            edgecolor='black', linewidth=1.5, capsize=8, alpha=0.7
        )
        
        for i, (mean, std) in enumerate(zip(intra_means, intra_stds)):
            ax_intra.text(
                i, mean + std * 1.1, f"{mean:.3f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )
        
        ax_intra.set_xticks(range(len(levels_list)))
        ax_intra.set_xticklabels([l.capitalize() for l in levels_list])
        ax_intra.set_title(f"{metric_name}\nIntra-Domain", fontweight='bold')
        ax_intra.set_ylabel("Distance" if col_idx == 0 else "")
        ax_intra.grid(axis='y', alpha=0.3)
        
        # 下段: Inter-domain距離（to Middle）
        ax_inter = axes[1, col_idx]
        inter_values = [inter[level] for level in levels_list]
        
        ax_inter.bar(
            range(len(levels_list)), inter_values,
            color=[colors[level] for level in levels_list],
            edgecolor='black', linewidth=1.5, alpha=0.7
        )
        
        for i, value in enumerate(inter_values):
            ax_inter.text(
                i, value * 1.05, f"{value:.3f}",
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )
        
        ax_inter.set_xticks(range(len(levels_list)))
        ax_inter.set_xticklabels([l.capitalize() for l in levels_list])
        ax_inter.set_title(f"Inter-Domain (to Middle)", fontweight='bold')
        ax_inter.set_ylabel("Distance" if col_idx == 0 else "")
        ax_inter.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"{method}_domain_distances_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved comparison: {output_path}")


def main():
    print("=" * 80)
    print("MDS / t-SNE / UMAP ドメイン距離計算")
    print("=" * 80)
    print()
    print("計算内容:")
    print("  1. Intra-domain: 各グループのドメイン中心から被験者それぞれの距離の平均")
    print("  2. Inter-domain: 各グループのドメイン中心からMiddleドメイン中心の距離")
    print()
    
    # 使用可能なメソッドを確認
    available_methods = ["mds", "tsne"]
    if UMAP_AVAILABLE:
        available_methods.append("umap")
    else:
        print("Note: UMAP not available. Install with: pip install umap-learn")
    
    print(f"Available methods: {', '.join(available_methods).upper()}")
    print()
    
    all_results = {}
    
    for method in available_methods:
        method_results = {}
        
        print(f"\n{'='*80}")
        print(f"METHOD: {method.upper()}")
        print(f"{'='*80}")
        
        for metric in METRICS:
            print(f"\n{'─'*60}")
            print(f"Processing: {metric.upper()} with {method.upper()}")
            print(f"{'─'*60}")
            
            # データ読み込み
            print("  Loading data...")
            dist_matrix = load_distance_matrix(metric)
            all_subjects = load_all_subjects(metric)
            
            # グループインデックスを取得
            group_indices = {}
            for level in LEVELS:
                subjects = load_group_subjects(metric, level)
                group_indices[level] = get_group_indices(all_subjects, subjects)
                print(f"    {level.capitalize()}: {len(group_indices[level])} subjects")
            
            print(f"  Matrix size: {dist_matrix.shape}")
            
            # 次元削減計算
            projection_results = compute_projection_with_domain_distances(
                dist_matrix, group_indices, method=method
            )
            
            # 結果表示
            print("\n  Results:")
            print("  ───────────────────────────────────────")
            print("  Intra-domain distances (mean ± std):")
            for level in LEVELS:
                mean = projection_results["intra_domain_distances"][level]["mean"]
                std = projection_results["intra_domain_distances"][level]["std"]
                print(f"    {level.capitalize():8s}: {mean:.4f} ± {std:.4f}")
            
            print("\n  Inter-domain distances (to Middle center):")
            for level in LEVELS:
                dist = projection_results["inter_domain_distances"][level]
                print(f"    {level.capitalize():8s}: {dist:.4f}")
            
            # 可視化
            print("\n  Generating visualization...")
            visualize_projection_with_distances(metric, method, projection_results, group_indices)
            
            # 結果を保存
            save_results = {
                "method": method,
                "metric": metric,
                "group_centroids": projection_results["group_centroids"],
                "intra_domain_distances": projection_results["intra_domain_distances"],
                "inter_domain_distances": projection_results["inter_domain_distances"]
            }
            
            output_json = OUTPUT_DIR / f"{metric}_{method}_domain_distances.json"
            with open(output_json, "w") as f:
                json.dump(save_results, f, indent=2)
            print(f"  Saved JSON: {output_json}")
            
            method_results[metric] = projection_results
        
        all_results[method] = method_results
    
    # サマリーテーブル作成（全メソッド）
    print(f"\n{'='*80}")
    print("Creating summary tables...")
    print(f"{'='*80}")
    
    for method in available_methods:
        print(f"\n{method.upper()} Summary:")
        df = create_summary_table(all_results[method], method)
        print("\n" + df.to_string(index=False))
    
    # 比較可視化（メソッドごと）
    print(f"\n{'='*80}")
    print("Creating comparison visualizations...")
    print(f"{'='*80}")
    
    for method in available_methods:
        print(f"\n  Creating comparison for {method.upper()}...")
        visualize_comparison_across_metrics(all_results[method], method)
    
    print("\n" + "="*80)
    print("✓ 全メソッドのドメイン距離計算完了")
    print("="*80)
    print()
    print("生成されたファイル:")
    for method in available_methods:
        print(f"\n  {method.upper()}:")
        print(f"    - {{metric}}_{method}_domain_distances.png  : 各指標の詳細可視化")
        print(f"    - {{metric}}_{method}_domain_distances.json : 数値データ")
        print(f"    - {method}_domain_distances_summary.csv   : 全指標のサマリー")
        print(f"    - {method}_domain_distances_comparison.png: 3指標の比較")
    print()
    print(f"保存先: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
