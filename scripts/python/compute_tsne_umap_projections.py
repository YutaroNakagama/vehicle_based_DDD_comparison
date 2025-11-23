#!/usr/bin/env python3
"""
t-SNEとUMAPを使ったグループ重心の投影計算
（MDSより計算時間がかかるため、別スクリプトとして分離）
"""

import numpy as np
import json
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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


def compute_projection_centroids(dist_matrix: np.ndarray, 
                                   group_indices_dict: dict,
                                   method: str = "tsne",
                                   n_components: int = 2,
                                   random_state: int = 42) -> dict:
    """指定した次元削減法で投影してグループ重心を計算"""
    
    if method == "tsne":
        print(f"    Running t-SNE (this may take a few minutes)...")
        reducer = TSNE(n_components=n_components,
                      metric="precomputed",
                      init="random",
                      random_state=random_state,
                      perplexity=min(30, len(dist_matrix) - 1),
                      max_iter=1000,
                      verbose=1)
        coords = reducer.fit_transform(dist_matrix)
        metric_value = float(reducer.kl_divergence_)
        metric_name = "kl_divergence"
    
    elif method == "umap":
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP is not available. Install with: pip install umap-learn")
        print(f"    Running UMAP (this may take a few minutes)...")
        reducer = umap.UMAP(n_components=n_components,
                           metric="precomputed",
                           random_state=random_state,
                           n_neighbors=min(15, len(dist_matrix) - 1),
                           verbose=True)
        coords = reducer.fit_transform(dist_matrix)
        metric_value = None
        metric_name = "none"
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'umap'")
    
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
                                         group_indices_dict: dict):
    """投影結果を重心付きで可視化"""
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
    output_name = f"{metric}_{method}_centroids.png"
    output_path = OUTPUT_DIR / output_name
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("=" * 80)
    print("t-SNE/UMAP投影によるグループ重心計算")
    print("=" * 80)
    print()
    print("注意: t-SNEとUMAPは計算に時間がかかります（各指標で数分）")
    print()
    
    methods = ["tsne"]
    if UMAP_AVAILABLE:
        methods.append("umap")
    else:
        print("⚠️ UMAP not available. Install with: pip install umap-learn")
        print()
    
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
        for level in LEVELS:
            subjects = load_group_subjects(metric, level)
            group_indices[level] = get_group_indices(all_subjects, subjects)
        
        print(f"  Matrix size: {dist_matrix.shape}")
        print()
        
        for method in methods:
            print(f"  Computing {method.upper()} centroids...")
            
            try:
                results = compute_projection_centroids(
                    dist_matrix, group_indices, method=method
                )
                
                print(f"    ✓ {method.upper()} completed")
                if results["metric_value"] is not None:
                    print(f"      {results['metric_name']}: {results['metric_value']:.4f}")
                
                for level, centroid_data in results["centroids"].items():
                    dist_to_center = centroid_data["distance_to_global_centroid"]
                    print(f"      {level.capitalize()}: spread={centroid_data['spread']:.4f}, "
                          f"dist_to_center={dist_to_center:.4f}")
                
                # 可視化
                print(f"    Generating visualization...")
                visualize_projection_with_centroids(metric, results, group_indices)
                
                # 結果を保存
                save_results = {
                    "method": results["method"],
                    "global_centroid": results["global_centroid"],
                    "centroids": results["centroids"],
                    "centroid_distances": results["centroid_distances"],
                    "metric_name": results["metric_name"],
                    "metric_value": results["metric_value"]
                }
                
                output_json = OUTPUT_DIR / f"{metric}_{method}_analysis.json"
                with open(output_json, "w") as f:
                    json.dump(save_results, f, indent=2)
                print(f"    Saved: {output_json}")
                print()
                
            except Exception as e:
                print(f"    ✗ Error computing {method.upper()}: {e}")
                print()
    
    print("=" * 80)
    print("✓ t-SNE/UMAP計算完了")
    print("=" * 80)


if __name__ == "__main__":
    main()
