#!/usr/bin/env python3
"""
Inter-group距離とグループ重心を計算するスクリプト

計算内容:
1. Inter-group distances (グループ間距離)
   - High ↔ Middle, High ↔ Low, Middle ↔ Low の平均距離・標準偏差
2. Group centroids (グループ重心)
   - MDS投影空間での各グループの重心座標
   - 重心間のユークリッド距離
3. Intra/Inter比率の計算
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.manifold import MDS, TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import json
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


def load_group_subjects(metric: str, level: str) -> list:
    """グループの被験者リストを読み込む"""
    group_path = DISTANCE_DIR / "subject-wise" / "ranks" / "ranks29" / f"{metric}_{level}.txt"
    with open(group_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_all_subjects(metric: str) -> list:
    """全被験者のリストを読み込む"""
    metric_dir = METRIC_DIRS[metric]
    # subjects.jsonから読み込む
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


def compute_intergroup_distances(dist_matrix: np.ndarray, 
                                  indices_A: np.ndarray, 
                                  indices_B: np.ndarray) -> dict:
    """2つのグループ間の距離統計を計算"""
    # グループAの各要素とグループBの各要素間の距離を抽出
    inter_distances = dist_matrix[np.ix_(indices_A, indices_B)]
    
    return {
        "mean": float(np.mean(inter_distances)),
        "std": float(np.std(inter_distances)),
        "min": float(np.min(inter_distances)),
        "max": float(np.max(inter_distances)),
        "median": float(np.median(inter_distances))
    }


def compute_intragroup_distances(dist_matrix: np.ndarray, 
                                  indices: np.ndarray) -> dict:
    """グループ内の距離統計を計算"""
    intra_distances = dist_matrix[np.ix_(indices, indices)]
    # 対角成分（自己距離=0）を除外
    mask = ~np.eye(len(indices), dtype=bool)
    intra_distances = intra_distances[mask]
    
    return {
        "mean": float(np.mean(intra_distances)),
        "std": float(np.std(intra_distances)),
        "min": float(np.min(intra_distances)),
        "max": float(np.max(intra_distances)),
        "median": float(np.median(intra_distances))
    }


def compute_mds_centroids(dist_matrix: np.ndarray, 
                          group_indices_dict: dict,
                          n_components: int = 2,
                          random_state: int = 42) -> dict:
    """MDS投影してグループ重心を計算"""
    # MDS投影
    mds = MDS(n_components=n_components, 
              dissimilarity="precomputed", 
              random_state=random_state,
              n_init=10,
              max_iter=1000)
    coords = mds.fit_transform(dist_matrix)
    
    # 各グループの重心を計算
    centroids = {}
    for level, indices in group_indices_dict.items():
        group_coords = coords[indices]
        centroid = np.mean(group_coords, axis=0)
        centroids[level] = {
            "coordinates": centroid.tolist(),
            "spread": float(np.mean(np.linalg.norm(group_coords - centroid, axis=1)))
        }
    
    # 重心間の距離を計算
    centroid_distances = {}
    for i, level1 in enumerate(LEVELS):
        for level2 in LEVELS[i+1:]:
            c1 = np.array(centroids[level1]["coordinates"])
            c2 = np.array(centroids[level2]["coordinates"])
            dist = float(np.linalg.norm(c1 - c2))
            centroid_distances[f"{level1}_vs_{level2}"] = dist
    
    return {
        "centroids": centroids,
        "centroid_distances": centroid_distances,
        "mds_coords": coords,
        "stress": float(mds.stress_)
    }


def compute_projection_centroids(dist_matrix: np.ndarray, 
                                   group_indices_dict: dict,
                                   method: str = "mds",
                                   n_components: int = 2,
                                   random_state: int = 42) -> dict:
    """指定した次元削減法で投影してグループ重心を計算
    
    Parameters
    ----------
    dist_matrix : np.ndarray
        距離行列
    group_indices_dict : dict
        グループ名 -> インデックスリストの辞書
    method : str
        投影手法 ("mds", "tsne", "umap")
    n_components : int
        次元数
    random_state : int
        乱数シード
    
    Returns
    -------
    dict
        重心情報、座標、メトリクス
    """
    # 次元削減を実行
    if method == "mds":
        reducer = MDS(n_components=n_components, 
                     dissimilarity="precomputed", 
                     random_state=random_state,
                     n_init=10,
                     max_iter=1000)
        coords = reducer.fit_transform(dist_matrix)
        metric_value = float(reducer.stress_)
        metric_name = "stress"
    
    elif method == "tsne":
        reducer = TSNE(n_components=n_components,
                      metric="precomputed",
                      init="random",  # precomputedの場合はrandomを使用
                      random_state=random_state,
                      perplexity=min(30, len(dist_matrix) - 1),
                      max_iter=1000)
        coords = reducer.fit_transform(dist_matrix)
        metric_value = float(reducer.kl_divergence_)
        metric_name = "kl_divergence"
    
    elif method == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP is not available. Install with: pip install umap-learn")
        reducer = umap.UMAP(n_components=n_components,
                           metric="precomputed",
                           random_state=random_state,
                           n_neighbors=min(15, len(dist_matrix) - 1))
        coords = reducer.fit_transform(dist_matrix)
        metric_value = None  # UMAPには単一のメトリクスがない
        metric_name = "none"
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 全体の重心（ドメイン中心）を計算
    global_centroid = np.mean(coords, axis=0)
    
    # 各グループの重心を計算
    centroids = {}
    for level, indices in group_indices_dict.items():
        group_coords = coords[indices]
        centroid = np.mean(group_coords, axis=0)
        
        # グループ重心からドメイン中心までの距離
        distance_to_global = float(np.linalg.norm(centroid - global_centroid))
        
        # グループ内の広がり
        spread = float(np.mean(np.linalg.norm(group_coords - centroid, axis=1)))
        
        centroids[level] = {
            "coordinates": centroid.tolist(),
            "spread": spread,
            "distance_to_global_centroid": distance_to_global
        }
    
    # 重心間の距離を計算
    centroid_distances = {}
    for i, level1 in enumerate(LEVELS):
        for level2 in LEVELS[i+1:]:
            c1 = np.array(centroids[level1]["coordinates"])
            c2 = np.array(centroids[level2]["coordinates"])
            dist = float(np.linalg.norm(c1 - c2))
            centroid_distances[f"{level1}_vs_{level2}"] = dist
    
    return {
        "method": method,
        "global_centroid": global_centroid.tolist(),
        "centroids": centroids,
        "centroid_distances": centroid_distances,
        "coords": coords,
        "metric_name": metric_name,
        "metric_value": metric_value
    }


def visualize_projection_with_centroids(metric: str, 
                                         projection_results: dict, 
                                         group_indices_dict: dict,
                                         output_suffix: str = ""):
    """投影結果を重心付きで可視化（MDS/t-SNE/UMAP共通）
    
    Parameters
    ----------
    metric : str
        距離指標名
    projection_results : dict
        投影結果（compute_projection_centroidsの出力）
    group_indices_dict : dict
        グループインデックス辞書
    output_suffix : str
        出力ファイル名のサフィックス
    """
    coords = projection_results["coords"]
    centroids = projection_results["centroids"]
    global_centroid = np.array(projection_results["global_centroid"])
    method = projection_results["method"]
    
    fig, ax = plt.subplots(figsize=(14, 11))
    
    # 各グループをプロット
    colors = {"high": "red", "middle": "gray", "low": "blue"}
    markers = {"high": "^", "middle": "s", "low": "v"}
    
    for level, indices in group_indices_dict.items():
        group_coords = coords[indices]
        ax.scatter(group_coords[:, 0], group_coords[:, 1],
                  c=colors[level], marker=markers[level],
                  s=100, alpha=0.6, label=f"{level.capitalize()} group",
                  edgecolors='black', linewidth=0.5)
    
    # ドメイン中心（全体重心）をプロット
    ax.scatter(global_centroid[0], global_centroid[1],
              c='black', marker='X', s=1000, 
              edgecolors='white', linewidth=3,
              label='Global centroid (Domain center)', zorder=15)
    
    # 各グループの重心をプロット
    for level, centroid_data in centroids.items():
        c = np.array(centroid_data["coordinates"])
        ax.scatter(c[0], c[1], 
                  c=colors[level], marker='*',
                  s=800, edgecolors='black', linewidth=2,
                  label=f"{level.capitalize()} centroid", zorder=10)
        
        # ドメイン中心からグループ重心への線を描画
        ax.plot([global_centroid[0], c[0]], [global_centroid[1], c[1]],
               c=colors[level], linestyle='-', alpha=0.5, linewidth=2.5)
        
        # 距離をラベル表示
        dist_to_center = centroid_data["distance_to_global_centroid"]
        mid_x = (global_centroid[0] + c[0]) / 2
        mid_y = (global_centroid[1] + c[1]) / 2
        ax.text(mid_x, mid_y, f"{dist_to_center:.2f}",
               fontsize=10, ha='center', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.4', 
                        facecolor=colors[level], alpha=0.3, edgecolor='black'))
    
    # 軸ラベルとタイトル
    method_upper = method.upper()
    ax.set_xlabel(f"{method_upper} Component 1", fontsize=14)
    ax.set_ylabel(f"{method_upper} Component 2", fontsize=14)
    
    title = f"{method_upper} Projection with Centroids - {metric.upper()}"
    if projection_results["metric_value"] is not None:
        title += f"\n{projection_results['metric_name']}: {projection_results['metric_value']:.4f}"
    ax.set_title(title, fontsize=16)
    
    ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_name = f"{metric}_{method}_centroids{output_suffix}.png"
    output_path = OUTPUT_DIR / output_name
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_centroids(metric: str, mds_results: dict, group_indices_dict: dict):
    """重心付きのMDS投影を可視化（後方互換性のため残す）"""
    # 新しい関数を使用
    projection_results = {
        "coords": mds_results["mds_coords"],
        "centroids": mds_results["centroids"],
        "global_centroid": np.mean(mds_results["mds_coords"], axis=0),
        "method": "mds",
        "metric_name": "stress",
        "metric_value": mds_results["stress"]
    }
    # global_centroidがcentroidsに含まれていない場合、追加計算
    for level in projection_results["centroids"]:
        if "distance_to_global_centroid" not in projection_results["centroids"][level]:
            c = np.array(projection_results["centroids"][level]["coordinates"])
            gc = projection_results["global_centroid"]
            projection_results["centroids"][level]["distance_to_global_centroid"] = float(np.linalg.norm(c - gc))
    
    visualize_projection_with_centroids(metric, projection_results, group_indices_dict)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"{metric}_mds_centroids.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_distance_heatmap(metric: str, results: dict):
    """Intra/Inter距離のヒートマップを作成"""
    # 距離行列を作成
    distance_matrix = np.zeros((3, 3))
    labels = ["High", "Middle", "Low"]
    
    # Intra距離を対角成分に
    for i, level in enumerate(LEVELS):
        distance_matrix[i, i] = results["intragroup"][level]["mean"]
    
    # Inter距離を非対角成分に
    for pair, stats in results["intergroup"].items():
        level1, level2 = pair.split("_vs_")
        i = LEVELS.index(level1)
        j = LEVELS.index(level2)
        distance_matrix[i, j] = stats["mean"]
        distance_matrix[j, i] = stats["mean"]  # 対称行列
    
    # ヒートマップを描画
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(distance_matrix, 
                annot=True, fmt='.4f', 
                xticklabels=labels, yticklabels=labels,
                cmap='YlOrRd', square=True, linewidths=1,
                cbar_kws={'label': 'Distance'},
                ax=ax)
    
    ax.set_title(f"Intra/Inter-group Distance Matrix - {metric.upper()}", 
                fontsize=16, pad=20)
    ax.set_xlabel("Group", fontsize=14)
    ax.set_ylabel("Group", fontsize=14)
    
    # 対角成分（Intra）を強調
    for i in range(3):
        rect = plt.Rectangle((i, i), 1, 1, fill=False, 
                            edgecolor='blue', linewidth=3)
        ax.add_patch(rect)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f"{metric}_distance_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_intra_inter_comparison(all_results: dict):
    """全距離指標でのIntra/Inter比較"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        results = all_results[metric]
        
        # データ準備
        categories = []
        means = []
        stds = []
        types = []  # Intra or Inter
        
        # Intra距離
        for level in LEVELS:
            categories.append(f"{level.capitalize()}\n(Intra)")
            means.append(results["intragroup"][level]["mean"])
            stds.append(results["intragroup"][level]["std"])
            types.append("Intra")
        
        # Inter距離
        for pair in ["high_vs_middle", "high_vs_low", "middle_vs_low"]:
            level1, level2 = pair.split("_vs_")
            categories.append(f"{level1.capitalize()}\nvs\n{level2.capitalize()}")
            means.append(results["intergroup"][pair]["mean"])
            stds.append(results["intergroup"][pair]["std"])
            types.append("Inter")
        
        # 棒グラフ
        colors = ['lightblue' if t == 'Intra' else 'lightcoral' for t in types]
        x = np.arange(len(categories))
        bars = ax.bar(x, means, yerr=stds, capsize=5, 
                     color=colors, edgecolor='black', linewidth=1.5)
        
        # Intra/Interを区別
        for i, (bar, t) in enumerate(zip(bars, types)):
            if t == "Intra":
                bar.set_hatch('//')
        
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylabel("Distance", fontsize=12)
        ax.set_title(f"{metric.upper()}", fontsize=14)
        ax.grid(axis='y', alpha=0.3)
        
        # 凡例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', edgecolor='black', hatch='//', label='Intra-group'),
            Patch(facecolor='lightcoral', edgecolor='black', label='Inter-group')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    plt.suptitle("Intra-group vs Inter-group Distance Comparison", 
                fontsize=16, y=1.02)
    plt.tight_layout()
    output_path = OUTPUT_DIR / "intra_inter_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def compute_intra_inter_ratios(results: dict) -> dict:
    """Intra/Inter比率を計算"""
    ratios = {}
    
    for level in LEVELS:
        intra_mean = results["intragroup"][level]["mean"]
        
        # 他のグループとのInter距離の平均
        inter_means = []
        for pair, stats in results["intergroup"].items():
            if level in pair:
                inter_means.append(stats["mean"])
        
        avg_inter = np.mean(inter_means)
        ratio = intra_mean / avg_inter if avg_inter > 0 else np.nan
        
        ratios[level] = {
            "intra_mean": intra_mean,
            "avg_inter_mean": avg_inter,
            "intra_inter_ratio": ratio
        }
    
    return ratios


def main():
    print("=" * 80)
    print("Inter-group距離とグループ重心の計算")
    print("=" * 80)
    print()
    
    all_results = {}
    
    for metric in METRICS:
        print(f"\n{'='*60}")
        print(f"Processing: {metric.upper()}")
        print(f"{'='*60}")
        
        # データ読み込み
        print("  Loading data...")
        dist_matrix = load_distance_matrix(metric)
        all_subjects = load_all_subjects(metric)
        
        # グループインデックスを取得
        group_indices = {}
        group_subjects = {}
        for level in LEVELS:
            subjects = load_group_subjects(metric, level)
            group_subjects[level] = subjects
            group_indices[level] = get_group_indices(all_subjects, subjects)
            print(f"    {level.capitalize()}: {len(group_indices[level])} subjects")
        
        results = {
            "metric": metric,
            "intragroup": {},
            "intergroup": {},
            "mds_analysis": {},
            "tsne_analysis": {},
            "umap_analysis": {},
            "intra_inter_ratios": {}
        }
        
        # 1. Intra-group距離
        print("\n  Computing intra-group distances...")
        for level in LEVELS:
            intra = compute_intragroup_distances(dist_matrix, group_indices[level])
            results["intragroup"][level] = intra
            print(f"    {level.capitalize()}: mean={intra['mean']:.4f}, std={intra['std']:.4f}")
        
        # 2. Inter-group距離
        print("\n  Computing inter-group distances...")
        for i, level1 in enumerate(LEVELS):
            for level2 in LEVELS[i+1:]:
                inter = compute_intergroup_distances(
                    dist_matrix, 
                    group_indices[level1], 
                    group_indices[level2]
                )
                pair_key = f"{level1}_vs_{level2}"
                results["intergroup"][pair_key] = inter
                print(f"    {level1.capitalize()} vs {level2.capitalize()}: "
                      f"mean={inter['mean']:.4f}, std={inter['std']:.4f}")
        
        # 3. MDS重心分析
        print("\n  Computing MDS centroids...")
        mds_results = compute_projection_centroids(dist_matrix, group_indices, method="mds")
        results["mds_analysis"] = mds_results
        print(f"    MDS stress: {mds_results['metric_value']:.4f}")
        for level, centroid_data in mds_results["centroids"].items():
            dist_to_center = centroid_data["distance_to_global_centroid"]
            print(f"    {level.capitalize()}: spread={centroid_data['spread']:.4f}, "
                  f"dist_to_center={dist_to_center:.4f}")
        
        # 4. t-SNE重心分析
        print("\n  Computing t-SNE centroids...")
        tsne_results = compute_projection_centroids(dist_matrix, group_indices, method="tsne")
        results["tsne_analysis"] = tsne_results
        print(f"    t-SNE KL divergence: {tsne_results['metric_value']:.4f}")
        for level, centroid_data in tsne_results["centroids"].items():
            dist_to_center = centroid_data["distance_to_global_centroid"]
            print(f"    {level.capitalize()}: spread={centroid_data['spread']:.4f}, "
                  f"dist_to_center={dist_to_center:.4f}")
        
        # 5. UMAP重心分析
        if UMAP_AVAILABLE:
            print("\n  Computing UMAP centroids...")
            umap_results = compute_projection_centroids(dist_matrix, group_indices, method="umap")
            results["umap_analysis"] = umap_results
            print(f"    UMAP embedding computed")
            for level, centroid_data in umap_results["centroids"].items():
                dist_to_center = centroid_data["distance_to_global_centroid"]
                print(f"    {level.capitalize()}: spread={centroid_data['spread']:.4f}, "
                      f"dist_to_center={dist_to_center:.4f}")
        else:
            print("\n  Skipping UMAP (not available)")
        
        # 6. Intra/Inter比率
        print("\n  Computing intra/inter ratios...")
        ratios = compute_intra_inter_ratios(results)
        results["intra_inter_ratios"] = ratios
        for level, ratio_data in ratios.items():
            print(f"    {level.capitalize()}: ratio={ratio_data['intra_inter_ratio']:.4f}")
        
        # 結果を保存（coords配列は除外して保存）
        save_results = {k: v for k, v in results.items() 
                       if k not in ["mds_analysis", "tsne_analysis", "umap_analysis"]}
        
        # 投影結果は座標を除いて保存
        for analysis_key, analysis_data in [
            ("mds_analysis", mds_results),
            ("tsne_analysis", tsne_results if "tsne_results" in locals() else None),
            ("umap_analysis", umap_results if UMAP_AVAILABLE and "umap_results" in locals() else None)
        ]:
            if analysis_data is not None:
                save_results[analysis_key] = {
                    "method": analysis_data["method"],
                    "global_centroid": analysis_data["global_centroid"],
                    "centroids": analysis_data["centroids"],
                    "centroid_distances": analysis_data["centroid_distances"],
                    "metric_name": analysis_data["metric_name"],
                    "metric_value": analysis_data["metric_value"]
                }
        
        output_json = OUTPUT_DIR / f"{metric}_intergroup_analysis.json"
        with open(output_json, "w") as f:
            json.dump(save_results, f, indent=2)
        print(f"\n  Saved: {output_json}")
        
        # 可視化
        print("\n  Generating visualizations...")
        visualize_projection_with_centroids(metric, mds_results, group_indices)
        visualize_projection_with_centroids(metric, tsne_results, group_indices)
        if UMAP_AVAILABLE:
            visualize_projection_with_centroids(metric, umap_results, group_indices)
        visualize_distance_heatmap(metric, results)
        
        all_results[metric] = results
    
    # 全体比較の可視化
    print(f"\n{'='*60}")
    print("Generating comparison visualizations...")
    print(f"{'='*60}")
    visualize_intra_inter_comparison(all_results)
    
    # サマリーテーブルを作成
    print("\n" + "="*80)
    print("Summary: Intra-group vs Inter-group Distances")
    print("="*80)
    
    summary_data = []
    for metric in METRICS:
        results = all_results[metric]
        for level in LEVELS:
            row = {
                "metric": metric,
                "group": level,
                "intra_mean": results["intragroup"][level]["mean"],
                "intra_std": results["intragroup"][level]["std"],
            }
            # 平均Inter距離を追加
            inter_means = []
            for pair, stats in results["intergroup"].items():
                if level in pair:
                    inter_means.append(stats["mean"])
            row["avg_inter_mean"] = np.mean(inter_means)
            row["intra_inter_ratio"] = results["intra_inter_ratios"][level]["intra_inter_ratio"]
            summary_data.append(row)
    
    df_summary = pd.DataFrame(summary_data)
    df_summary = df_summary.round(4)
    
    print("\n" + df_summary.to_string(index=False))
    
    csv_path = OUTPUT_DIR / "intra_inter_summary.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSaved summary: {csv_path}")
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print(f"✓ Results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
