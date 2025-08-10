import os
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

from scipy.stats import wasserstein_distance
from tslearn.metrics import dtw

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
import umap

from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# === MMD Function ===
def compute_mmd(X, Y, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    m = len(X)
    n = len(Y)
    return Kxx.sum() / (m * m) + Kyy.sum() / (n * n) - 2 * Kxy.sum() / (m * n)

# === Feature Loader ===
def load_features(subject_str):
    subject_id, version = subject_str.split("_")
    path = f"data/processed/common/processed_{subject_id}_{version}.csv"
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return None
    df = pd.read_csv(path)
    feature_cols = [
        col for col in df.columns
        if col not in ["Timestamp", "KSS", "subject"]
        and not col.startswith("Channel_")
        and not col.startswith("KSS_")
        and not col.startswith("theta_alpha_over_beta")
    ]
    return df[feature_cols].dropna().values

def load_subject_list(path):
    with open(path) as f:
        return [line.strip() for line in f.readlines()]

def extract_features(subjects):
    features = {}
    for subj in subjects:
        data = load_features(subj)
        if data is not None:
            features[subj] = data
    return features

def compute_mmd_matrix(features):
    valid_subjects = list(features.keys())
    n = len(valid_subjects)
    mmd_matrix = np.zeros((n, n))
    print("Computing MMD matrix...")
    for i in tqdm(range(n), desc="Rows"):
        for j in range(n):
            mmd_matrix[i, j] = compute_mmd(features[valid_subjects[i]], features[valid_subjects[j]])
    return mmd_matrix, valid_subjects

def plot_and_save_mmd_matrix(mmd_matrix, valid_subjects, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(mmd_matrix, xticklabels=valid_subjects, yticklabels=valid_subjects, cmap='viridis')
    plt.title("MMD Distance Between Subjects (Features from Processed CSVs)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/mmd_matrix.png")
    plt.close()
    np.save(f"{out_dir}/mmd_matrix.npy", mmd_matrix)
    with open(f"{out_dir}/mmd_subjects.json", "w") as f:
        json.dump(valid_subjects, f)

def plot_mean_std(mmd_mean, mmd_std, valid_subjects, out_dir, sorted=False):
    plt.figure(figsize=(14, 6))
    if sorted:
        plt.errorbar(valid_subjects, mmd_mean, yerr=mmd_std, fmt='o', capsize=5)
        plt.title("Mean and StdDev of MMD per Subject (Sorted)")
        plt.savefig(f"{out_dir}/mmd_mean_std_sorted.png")
    else:
        plt.errorbar(valid_subjects, mmd_mean, yerr=mmd_std, fmt='o', capsize=5)
        plt.title("Mean and StdDev of MMD per Subject")
        plt.savefig(f"{out_dir}/mmd_mean_std.png")
    plt.xticks(rotation=90)
    plt.ylabel("MMD")
    plt.tight_layout()
    plt.close()

def save_mean_std(mmd_mean, mmd_std, valid_subjects, out_dir, sorted=False):
    if sorted:
        np.save(f"{out_dir}/mmd_mean_sorted.npy", mmd_mean)
        np.save(f"{out_dir}/mmd_std_sorted.npy", mmd_std)
        with open(f"{out_dir}/mmd_subjects_sorted.json", "w") as f:
            json.dump(valid_subjects, f)
    else:
        np.save(f"{out_dir}/mmd_mean.npy", mmd_mean)
        np.save(f"{out_dir}/mmd_std.npy", mmd_std)

# === Feature Loader ===

# === ラベル読み込み ===
def load_subjects(path):
    with open(path) as f:
        return json.load(f)

# === 共通描画関数 ===
def plot_projection(coords, subjects, method_name, dist_name, outdir):
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1])
    for i, label in enumerate(subjects):
        plt.text(coords[i, 0], coords[i, 1], label, fontsize=8)
    plt.title(f"{method_name} Projection - {dist_name}")
    plt.tight_layout()
    save_path = os.path.join(outdir, f"{dist_name.lower()}_{method_name.lower()}_projection.png")
    plt.savefig(save_path)
    plt.close()

def plot_group_heatmap(matrix, group_names, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, xticklabels=group_names, yticklabels=group_names, cmap='viridis', annot=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_mds(matrix, labels, title, save_path):
#    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42,
              normalized_stress=False)
    coords = mds.fit_transform(np.nan_to_num(matrix))
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1])
    for i, label in enumerate(labels):
        plt.text(coords[i, 0], coords[i, 1], label, fontsize=10)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def load_groups(path="../misc/target_groups.txt"):
    with open(path) as f:
        subjects = [line.strip().split() for line in f.readlines()]
    groups = {f"G{idx+1}": group for idx, group in enumerate(subjects)}
    return groups

def compute_group_dist_matrix(matrix, subjects, groups):
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    group_names = list(groups.keys())
    n = len(group_names)
    group_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            a = groups[group_names[i]]
            b = groups[group_names[j]]
            valid = [(subj_to_idx[x], subj_to_idx[y]) for x in a for y in b if x in subj_to_idx and y in subj_to_idx]
            if valid:
                group_matrix[i, j] = np.mean([matrix[p][q] for p, q in valid])
            else:
                group_matrix[i, j] = np.nan
    return group_matrix, group_names
    
def plot_group_projection(matrix, subjects, groups, method_name, save_path):
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=42).fit_transform(matrix)
    subj_to_coords = {s: coords[i] for i, s in enumerate(subjects)}

    plt.figure(figsize=(10, 8))
    for gname, members in groups.items():
        group_coords = [subj_to_coords[s] for s in members if s in subj_to_coords]
        if group_coords:
            group_coords = np.array(group_coords)
            plt.scatter(group_coords[:, 0], group_coords[:, 1], label=gname)
            for i, s in enumerate(members):
                if s in subj_to_coords:
                    plt.text(group_coords[i, 0], group_coords[i, 1], s, fontsize=6)
    plt.title(f"{method_name}: Subject Group Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_group_centroid_distances(centroids):
    group_names = list(centroids.keys())
    n = len(group_names)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(centroids[group_names[i]] - centroids[group_names[j]])
    return dist_matrix, group_names


def plot_centroid_heatmap(matrix, group_names, method_name, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, xticklabels=group_names, yticklabels=group_names,
                annot=True, cmap="Blues", fmt=".2f")
    plt.title(f"{method_name}: Group Centroid Distance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def compute_intra_group_variability(matrix, subjects, groups):
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    results = {}
    for group_name, group_members in groups.items():
        indices = [subj_to_idx[s] for s in group_members if s in subj_to_idx]
        if len(indices) < 2:
            results[group_name] = {"mean": np.nan, "std": np.nan}
            continue
        dists = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                dists.append(matrix[indices[i], indices[j]])
        if dists:
            results[group_name] = {
                "mean": float(np.mean(dists)),
                "std": float(np.std(dists))
            }
        else:
            results[group_name] = {"mean": np.nan, "std": np.nan}
    return results

def compute_intra_inter_stats(matrix, subjects, groups):
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    stats = {}
    for gname, gmembers in groups.items():
        idx_in = [subj_to_idx[s] for s in gmembers if s in subj_to_idx]
        idx_out = [i for i in range(len(subjects)) if i not in idx_in]

        # グループ内距離
        intra_vals = [matrix[i, j] for i in idx_in for j in idx_in if i != j]
        # グループ外距離
        inter_vals = [matrix[i, j] for i in idx_in for j in idx_out]

        stats[gname] = {
            "intra_mean": np.mean(intra_vals) if intra_vals else np.nan,
            "intra_std": np.std(intra_vals) if intra_vals else np.nan,
            "inter_mean": np.mean(inter_vals) if inter_vals else np.nan,
            "inter_std": np.std(inter_vals) if inter_vals else np.nan,
        }
    return stats

# === 可視化（誤差棒付き棒グラフ） ===
def plot_intra_inter(stats, dist_name, save_path):
    group_names = list(stats.keys())
    intra_means = [stats[g]["intra_mean"] for g in group_names]
    intra_stds = [stats[g]["intra_std"] for g in group_names]
    inter_means = [stats[g]["inter_mean"] for g in group_names]
    inter_stds = [stats[g]["inter_std"] for g in group_names]

    x = np.arange(len(group_names))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, intra_means, width, yerr=intra_stds, label="Intra", capsize=5)
    plt.bar(x + width/2, inter_means, width, yerr=inter_stds, label="Inter", capsize=5)
    plt.xticks(x, group_names, rotation=45)
    plt.ylabel("Distance")
    plt.title(f"Intra vs Inter Group Distance ({dist_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def compute_group_centroids_from_distance_matrix(matrix, subjects, groups):
    """
    距離行列→MDS埋め込み→グループ重心dictを返す
    """
    coords = MDS(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(np.nan_to_num(matrix))
    subj_to_coords = {s: coords[i] for i, s in enumerate(subjects)}
    centroids = {}
    for gname, members in groups.items():
        valid = [subj_to_coords[s] for s in members if s in subj_to_coords]
        if valid:
            centroids[gname] = np.mean(valid, axis=0)
    return centroids


def main():
    subject_list_path = "../../dataset/mdapbe/subject_list.txt"
    out_dir = "results/mmd"

    # 1. サブジェクトリスト
    subjects = load_subject_list(subject_list_path)
    # 2. 特徴量抽出
    features = extract_features(subjects)
    # 3. MMD行列
    mmd_matrix, valid_subjects = compute_mmd_matrix(features)
    # 4. 保存とプロット
    plot_and_save_mmd_matrix(mmd_matrix, valid_subjects, out_dir)

    # 5. 平均・分散
    mmd_mean = mmd_matrix.mean(axis=1)
    mmd_std = mmd_matrix.std(axis=1)
    save_mean_std(mmd_mean, mmd_std, valid_subjects, out_dir, sorted=False)
    plot_mean_std(mmd_mean, mmd_std, valid_subjects, out_dir, sorted=False)

    # 6. ソート（降順）
    sorted_indices = np.argsort(-mmd_mean)
    mmd_mean_sorted = mmd_mean[sorted_indices]
    mmd_std_sorted = mmd_std[sorted_indices]
    subjects_sorted = [valid_subjects[i] for i in sorted_indices]
    save_mean_std(mmd_mean_sorted, mmd_std_sorted, subjects_sorted, out_dir, sorted=True)
    plot_mean_std(mmd_mean_sorted, mmd_std_sorted, subjects_sorted, out_dir, sorted=True)

    
    # === Load distance matrix and subject labels ===
    mmd_matrix = np.load("results/mmd/mmd_matrix.npy")
    with open("results/mmd/mmd_subjects.json") as f:
        subjects = json.load(f)
    
    
    # === Load subject list ===
    with open("../../dataset/mdapbe/subject_list.txt") as f:
        subjects = [line.strip() for line in f.readlines()]
    
    features = {}
    for subj in subjects:
        data = load_features(subj)
        if data is not None:
            features[subj] = data
    
    valid_subjects = list(features.keys())
    n = len(valid_subjects)
    
    # === Wasserstein Distance Matrix ===
    print("Computing Wasserstein Distance Matrix (feature-wise average)...")
    wass_matrix = np.zeros((n, n))
    
    MAX_DIST = 1e6  # 例えば100万など、UMAPに支障が出ない範囲で設定
    
    for i in tqdm(range(n)):
        for j in range(n):
            dists = []
            min_dim = min(features[valid_subjects[i]].shape[1], features[valid_subjects[j]].shape[1])
            for k in range(min_dim):
                xi = features[valid_subjects[i]][:, k]
                yj = features[valid_subjects[j]][:, k]
                w = wasserstein_distance(xi, yj)
                if np.isfinite(w):
                    dists.append(min(w, MAX_DIST))  # ← ここで制限
            if dists:
                wass_matrix[i, j] = np.mean(dists)
            else:
                wass_matrix[i, j] = 0.0
    
    # === DTW Distance Matrix (using mean feature vector per subject) ===
    print("Computing DTW Distance Matrix (per-subject mean vector)...")
    dtw_matrix = np.zeros((n, n))
    mean_series = [features[subj].mean(axis=1) for subj in valid_subjects]
    
    for i in tqdm(range(n)):
        for j in range(n):
            try:
                d = dtw(mean_series[i], mean_series[j])
                if np.isfinite(d) and d < 1e6:
                    dtw_matrix[i, j] = d
                else:
                    dtw_matrix[i, j] = 0.0
            except:
                print(f"DTW failed for {i}, {j}")
                dtw_matrix[i, j] = 0.0
    
    # === Save results ===
    os.makedirs("results/distances", exist_ok=True)
    np.save("results/distances/wasserstein_matrix.npy", wass_matrix)
    np.save("results/distances/dtw_matrix.npy", dtw_matrix)
    
    with open("results/distances/subjects.json", "w") as f:
        json.dump(valid_subjects, f)
    
    print("Distance matrices saved to results/distances/")
    
    
    # === 距離行列の種類とファイル対応表 ===
    distance_types = {
        "mmd": "results/mmd/mmd_matrix.npy",
        "wasserstein": "results/distances/wasserstein_matrix.npy",
        "dtw": "results/distances/dtw_matrix.npy"
    }
    
    
    # === 実行処理 ===
    for dist_name, matrix_path in distance_types.items():
        # 対応するsubjects.jsonの場所
        subj_path = "results/mmd/mmd_subjects.json" if dist_name == "mmd" else "results/distances/subjects.json"
        subjects = load_subjects(subj_path)
        matrix = np.load(matrix_path)
    
        if not np.all(np.isfinite(matrix)):
            print(f"[Warning] {dist_name} matrix contains invalid values (inf or NaN)")
            bad_idx = np.argwhere(~np.isfinite(matrix))
            print("Invalid entries at indices:", bad_idx)
            continue  
    
        outdir = f"results/projection/{dist_name.lower()}"
        os.makedirs(outdir, exist_ok=True)
    
    
    for dist_name, matrix_path in distance_types.items():
        matrix = np.load(matrix_path)
        subj_path = "results/mmd/mmd_subjects.json" if dist_name == "mmd" else "results/distances/subjects.json"
        with open(subj_path) as f:
            subjects = json.load(f)
    
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
        mean_vals = matrix.mean(axis=1)
        std_vals = matrix.std(axis=1)
    
        sorted_idx = np.argsort(-mean_vals)
        mean_sorted = mean_vals[sorted_idx]
        std_sorted = std_vals[sorted_idx]
        subj_sorted = [subjects[i] for i in sorted_idx]
    
        save_dir = f"results/{dist_name.lower()}"
        os.makedirs(save_dir, exist_ok=True)
    
        np.save(os.path.join(save_dir, f"{dist_name.lower()}_mean.npy"), mean_vals)
        np.save(os.path.join(save_dir, f"{dist_name.lower()}_std.npy"), std_vals)
        np.save(os.path.join(save_dir, f"{dist_name.lower()}_mean_sorted.npy"), mean_sorted)
        np.save(os.path.join(save_dir, f"{dist_name.lower()}_std_sorted.npy"), std_sorted)
        with open(os.path.join(save_dir, f"{dist_name.lower()}_subjects_sorted.json"), "w") as f:
            json.dump(subj_sorted, f)
    
        plt.figure(figsize=(14, 6))
        plt.errorbar(subj_sorted, mean_sorted, yerr=std_sorted, fmt='o', capsize=5)
        plt.xticks(rotation=90)
        plt.title(f"Mean and StdDev of {dist_name} per Subject (Sorted)")
        plt.ylabel(dist_name)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{dist_name.lower()}_mean_std_sorted.png"))
        plt.close()
    
    
    distance_types = {
        "mmd": "results/mmd/mmd_matrix.npy",
        "wasserstein": "results/distances/wasserstein_matrix.npy",
        "dtw": "results/distances/dtw_matrix.npy"
    }
    
    subject_paths = {
        "mmd": "results/mmd/mmd_subjects.json",
        "wasserstein": "results/distances/subjects.json",
        "dtw": "results/distances/subjects.json"
    }
    
    for dist_name, matrix_path in distance_types.items():
        print(f"Processing: {dist_name}")
        matrix = np.load(matrix_path)
        with open(subject_paths[dist_name]) as f:
            subjects = json.load(f)
    
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
        mean_vals = matrix.mean(axis=1)
        std_vals = matrix.std(axis=1)
        sorted_idx = np.argsort(-mean_vals)
        mean_sorted = mean_vals[sorted_idx]
        std_sorted = std_vals[sorted_idx]
        subj_sorted = [subjects[i] for i in sorted_idx]
    
        save_dir = f"results/{dist_name.lower()}"
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"{dist_name.lower()}_mean.npy"), mean_vals)
        np.save(os.path.join(save_dir, f"{dist_name.lower()}_std.npy"), std_vals)
        np.save(os.path.join(save_dir, f"{dist_name.lower()}_mean_sorted.npy"), mean_sorted)
        np.save(os.path.join(save_dir, f"{dist_name.lower()}_std_sorted.npy"), std_sorted)
        with open(os.path.join(save_dir, f"{dist_name.lower()}_subjects_sorted.json"), "w") as f:
            json.dump(subj_sorted, f)
    
        plt.figure(figsize=(14, 6))
        plt.errorbar(subj_sorted, mean_sorted, yerr=std_sorted, fmt='o', capsize=5)
        plt.xticks(rotation=90)
        plt.title(f"Mean and StdDev of {dist_name} per Subject (Sorted)")
        plt.ylabel(dist_name)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{dist_name.lower()}_mean_std_sorted.png"))
        plt.close()
    
        print(f"  - Group-level analysis for {dist_name}")
        groups = load_groups("../misc/target_groups.txt")
        group_matrix, group_names = compute_group_dist_matrix(matrix, subjects, groups)
    
        group_dir = f"results/group_distances/{dist_name.lower()}"
        os.makedirs(group_dir, exist_ok=True)
        np.save(os.path.join(group_dir, "group_matrix.npy"), group_matrix)
        with open(os.path.join(group_dir, "group_names.json"), "w") as f:
            json.dump(group_names, f)
    
        plot_group_heatmap(group_matrix, group_names,
                           f"{dist_name} Distance Between Groups",
                           os.path.join(group_dir, "group_heatmap.png"))
    
        print("    - MDS for groups...")
    
        # before calling plot_mds
        if np.allclose(group_matrix, 0):
            print(f"[Warning] Group distance matrix for {dist_name} is all zeros. Skipping MDS.")
            continue
        
        if np.any(np.sum(group_matrix, axis=1) == 0):
            print(f"[Warning] Some rows in group_matrix for {dist_name} sum to zero. MDS may be unstable.")
    
        if np.sum(group_matrix) == 0:
            print(f"[Warning] Total distance is zero for {dist_name}. Skipping MDS.")
            continue
    
        plot_mds(np.nan_to_num(group_matrix), group_names,
                 f"{dist_name} - MDS Projection of Groups",
                 os.path.join(group_dir, "group_mds_projection.png"))
    
    
    
    matrix = np.load("results/mmd/mmd_matrix.npy")
    with open("results/mmd/mmd_subjects.json") as f:
        subjects = json.load(f)
    groups = load_groups("../misc/target_groups.txt")
    
    centroids = compute_group_centroids_from_distance_matrix(matrix, subjects, groups)
    if not centroids:
        print("[Warning] No group centroids available (skipping centroid heatmap).")
    else:
        group_dist_matrix, group_names = compute_group_centroid_distances(centroids)
        plot_group_heatmap(group_dist_matrix, group_names,
                           "MMD Group Centroid Distances",
                           "results/mmd/group_centroid_distance_heatmap.png")
    
    for dist_name, matrix_path in distance_types.items():
        print(f"Overlay evaluation: {dist_name}")
        try:
            matrix = np.load(matrix_path)
            subjects = load_subjects(subject_paths[dist_name])
            matrix = np.nan_to_num(matrix)
    
            groups = load_groups("../misc/target_groups.txt")
    
            # グループ分布の射影図
            outdir = f"results/{dist_name.lower()}"
            os.makedirs(outdir, exist_ok=True)
    
            plot_group_projection(matrix, subjects, groups,
                                  method_name=dist_name,
                                  save_path=os.path.join(outdir, "group_overlay_projection.png"))
    
            # 重心と距離行列の算出＋ヒートマップ
            print("\nGroup definitions:")
            for group, members in groups.items():
                print(f"{group}: {members}")
    
            centroids = compute_group_centroids_from_distance_matrix(matrix, subjects, groups)
            if not centroids:
                print("[Warning] No group centroids available (skipping centroid heatmap).")
                continue  # これで以降の処理をスキップ
            centroid_matrix, group_names = compute_group_centroid_distances(centroids)
    
            plot_centroid_heatmap(centroid_matrix, group_names,
                                  method_name=dist_name,
                                  save_path=os.path.join(outdir, "group_centroid_distance_heatmap.png"))
    
        except Exception as e:
            print(f"[{dist_name}] Overlay evaluation failed: {e}")
    
    
    
    # === 各手法に対してグループ評価を実行 ===
    distance_methods = {
        "mmd": "results/mmd/mmd_matrix.npy",
        "wasserstein": "results/distances/wasserstein_matrix.npy",
        "dtw": "results/distances/dtw_matrix.npy"
    }
    
    subject_paths = {
        "mmd": "results/mmd/mmd_subjects.json",
        "wasserstein": "results/distances/subjects.json",
        "dtw": "results/distances/subjects.json"
    }
    
    group_path = "../misc/target_groups.txt"
    groups = load_groups(group_path)
    
    for method, matrix_path in distance_methods.items():
        print(f"[{method.upper()}] Evaluating group structure...")
    
        matrix = np.load(matrix_path)
        with open(subject_paths[method]) as f:
            subjects = json.load(f)
    
        matrix = np.nan_to_num(matrix)
        outdir = f"results/{method}"
    
        # 射影＋グループ色分け
        plot_group_projection(matrix, subjects, groups,
                              method_name=method.upper(),
                              save_path=os.path.join(outdir, "group_overlay_projection.png"))
    
        # 重心距離評価
        centroids = compute_group_centroids_from_distance_matrix(matrix, subjects, groups)
        if not centroids:
            print("[Warning] No group centroids available (skipping centroid heatmap).")
            continue  # これで以降の処理をスキップ
        centroid_matrix, group_names = compute_group_centroid_distances(centroids)
        if centroid_matrix.size == 0:
            print(f"[{method.upper()}] Skipping centroid heatmap (empty matrix).")
        else:
            plot_centroid_heatmap(centroid_matrix, group_names,
                                  method_name=method.upper(),
                                  save_path=os.path.join(outdir, "group_centroid_distance_heatmap.png"))
    
    
    # === 距離ごとのグループ内ばらつき評価と描画 ===
    for dist_name, matrix_path in distance_types.items():
        matrix = np.load(matrix_path)
        with open(subject_paths[dist_name]) as f:
            subjects = json.load(f)
        groups = load_groups("../misc/target_groups.txt")
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
        intra_results = compute_intra_group_variability(matrix, subjects, groups)
    
        # 保存先ディレクトリ
        intra_dir = f"results/group_distances/{dist_name.lower()}/intra"
        os.makedirs(intra_dir, exist_ok=True)
    
        # 保存（JSON & CSV）
        with open(os.path.join(intra_dir, "intra_group_variability.json"), "w") as f:
            json.dump(intra_results, f, indent=2)
    
        pd.DataFrame(intra_results).T.to_csv(os.path.join(intra_dir, "intra_group_variability.csv"))
    
        # 可視化（棒グラフ）
        group_labels = list(intra_results.keys())
        means = [intra_results[g]["mean"] for g in group_labels]
        stds = [intra_results[g]["std"] for g in group_labels]
    
        plt.figure(figsize=(10, 6))
        plt.bar(group_labels, means, yerr=stds, capsize=5)
        plt.title(f"Intra-group Variability - {dist_name}")
        plt.ylabel("Mean Distance")
        plt.xlabel("Group")
        plt.tight_layout()
        plt.savefig(os.path.join(intra_dir, "intra_variability.png"))
        plt.close()
    
    for dist_name, matrix_path in distance_types.items():
        matrix = np.load(matrix_path)
        with open(subject_paths[dist_name]) as f:
            subjects = json.load(f)
        groups = load_groups("../misc/target_groups.txt")
    
        stats = compute_intra_inter_stats(matrix, subjects, groups)
    
        outdir = f"results/group_distances/{dist_name.lower()}/intra_inter"
        os.makedirs(outdir, exist_ok=True)
        plot_intra_inter(stats, dist_name, os.path.join(outdir, "intra_inter_comparison.png"))
    

if __name__ == "__main__":
    main()
