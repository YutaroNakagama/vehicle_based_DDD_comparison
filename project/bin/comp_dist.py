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
from sklearn.manifold import MDS
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

def compute_mmd(X, Y, gamma=None):
    """Compute MMD (Maximum Mean Discrepancy) between two feature matrices."""
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    Kxx = rbf_kernel(X, X, gamma=gamma)
    Kyy = rbf_kernel(Y, Y, gamma=gamma)
    Kxy = rbf_kernel(X, Y, gamma=gamma)
    m, n = len(X), len(Y)
    return Kxx.sum() / (m * m) + Kyy.sum() / (n * n) - 2 * Kxy.sum() / (m * n)

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

def extract_features(subjects):
    return {subj: load_features(subj) for subj in subjects if load_features(subj) is not None}

def load_subject_list(path):
    with open(path) as f:
        return [line.strip() for line in f.readlines()]

def load_json(path):
    with open(path) as f:
        return json.load(f)

def load_groups(path):
    with open(path) as f:
        subjects = [line.strip().split() for line in f.readlines()]
    return {f"G{idx+1}": group for idx, group in enumerate(subjects)}

def compute_mmd_matrix(features):
    valid_subjects = list(features.keys())
    n = len(valid_subjects)
    mmd_matrix = np.zeros((n, n))
    for i in tqdm(range(n), desc="MMD matrix"):
        for j in range(n):
            mmd_matrix[i, j] = compute_mmd(features[valid_subjects[i]], features[valid_subjects[j]])
    return mmd_matrix, valid_subjects

def plot_heatmap(matrix, row_labels, col_labels, title, save_path, annot=True, fmt=".2f"):
    plt.figure(figsize=(max(8, len(row_labels)//2), max(6, len(col_labels)//2)))
    sns.heatmap(matrix, xticklabels=col_labels, yticklabels=row_labels, cmap="viridis", annot=annot, fmt=fmt)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_bar(labels, means, stds, title, ylabel, save_path):
    plt.figure(figsize=(max(12, len(labels)//2), 6))
    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_group_projection(matrix, subjects, groups, method_name, save_path):
    coords = MDS(n_components=2, dissimilarity="precomputed", random_state=42).fit_transform(np.nan_to_num(matrix))
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

def compute_group_dist_matrix(matrix, subjects, groups):
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    group_names = list(groups.keys())
    n = len(group_names)
    group_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a, b = groups[group_names[i]], groups[group_names[j]]
            valid = [(subj_to_idx[x], subj_to_idx[y]) for x in a for y in b if x in subj_to_idx and y in subj_to_idx]
            group_matrix[i, j] = np.mean([matrix[p][q] for p, q in valid]) if valid else np.nan
    return group_matrix, group_names

def compute_group_centroids_from_distance_matrix(matrix, subjects, groups):
    coords = MDS(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(np.nan_to_num(matrix))
    subj_to_coords = {s: coords[i] for i, s in enumerate(subjects)}
    centroids = {gname: np.mean([subj_to_coords[s] for s in members if s in subj_to_coords], axis=0)
                 for gname, members in groups.items() if [subj_to_coords[s] for s in members if s in subj_to_coords]}
    return centroids

def compute_group_centroid_distances(centroids):
    group_names = list(centroids.keys())
    n = len(group_names)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(centroids[group_names[i]] - centroids[group_names[j]])
    return dist_matrix, group_names

def compute_intra_group_variability(matrix, subjects, groups):
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    results = {}
    for group_name, group_members in groups.items():
        indices = [subj_to_idx[s] for s in group_members if s in subj_to_idx]
        if len(indices) < 2:
            results[group_name] = {"mean": np.nan, "std": np.nan}
            continue
        dists = [matrix[i, j] for i in indices for j in indices if i < j]
        results[group_name] = {
            "mean": float(np.mean(dists)) if dists else np.nan,
            "std": float(np.std(dists)) if dists else np.nan
        }
    return results

def compute_intra_inter_stats(matrix, subjects, groups):
    subj_to_idx = {s: i for i, s in enumerate(subjects)}
    stats = {}
    for gname, gmembers in groups.items():
        idx_in = [subj_to_idx[s] for s in gmembers if s in subj_to_idx]
        idx_out = [i for i in range(len(subjects)) if i not in idx_in]
        intra_vals = [matrix[i, j] for i in idx_in for j in idx_in if i != j]
        inter_vals = [matrix[i, j] for i in idx_in for j in idx_out]
        stats[gname] = {
            "intra_mean": np.mean(intra_vals) if intra_vals else np.nan,
            "intra_std": np.std(intra_vals) if intra_vals else np.nan,
            "inter_mean": np.mean(inter_vals) if inter_vals else np.nan,
            "inter_std": np.std(inter_vals) if inter_vals else np.nan,
        }
    return stats

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

def main():
    subject_list_path = "../../dataset/mdapbe/subject_list.txt"
    out_dir = "results/mmd"
    # 1. Subject List
    subjects = load_subject_list(subject_list_path)
    # 2. Feature Extraction
    features = extract_features(subjects)
    # 3. MMD Matrix
    mmd_matrix, valid_subjects = compute_mmd_matrix(features)
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/mmd_matrix.npy", mmd_matrix)
    print(f"Saved MMD matrix to {out_dir}/mmd_matrix.npy")
    with open(f"{out_dir}/mmd_subjects.json", "w") as f:
        json.dump(valid_subjects, f)
    print(f"Saved subject list to {out_dir}/mmd_subjects.json")

    # Wasserstein/DTW計算（同一valid_subjectsを使う）
    n = len(valid_subjects)
    print("Computing Wasserstein and DTW distance matrices...")
    wass_matrix = np.zeros((n, n))
    dtw_matrix = np.zeros((n, n))
    mean_series = [features[subj].mean(axis=1) for subj in valid_subjects]
    MAX_DIST = 1e6

    for i in tqdm(range(n)):
        for j in range(n):
            # Wasserstein
            dists = [min(wasserstein_distance(features[valid_subjects[i]][:, k], features[valid_subjects[j]][:, k]), MAX_DIST)
                     for k in range(min(features[valid_subjects[i]].shape[1], features[valid_subjects[j]].shape[1]))
                     if np.isfinite(wasserstein_distance(features[valid_subjects[i]][:, k], features[valid_subjects[j]][:, k]))]
            wass_matrix[i, j] = np.mean(dists) if dists else 0.0
            # DTW
            try:
                d = dtw(mean_series[i], mean_series[j])
                dtw_matrix[i, j] = d if np.isfinite(d) and d < MAX_DIST else 0.0
            except:
                dtw_matrix[i, j] = 0.0

    os.makedirs("results/distances", exist_ok=True)
    np.save("results/distances/wasserstein_matrix.npy", wass_matrix)
    print("Saved Wasserstein distance matrix to results/distances/wasserstein_matrix.npy")
    np.save("results/distances/dtw_matrix.npy", dtw_matrix)
    print("Saved DTW distance matrix to results/distances/dtw_matrix.npy")
    with open("results/distances/subjects.json", "w") as f:
        json.dump(valid_subjects, f)
    print("Saved subjects list to results/distances/subjects.json")

    # 分析・可視化を1ループで
    distance_types = {
        "mmd": ("results/mmd/mmd_matrix.npy", "results/mmd/mmd_subjects.json"),
        "wasserstein": ("results/distances/wasserstein_matrix.npy", "results/distances/subjects.json"),
        "dtw": ("results/distances/dtw_matrix.npy", "results/distances/subjects.json")
    }
    groups = load_groups("../misc/target_groups.txt")
    for dist_name, (matrix_path, subject_path) in distance_types.items():
        matrix = np.nan_to_num(np.load(matrix_path))
        subjects = load_json(subject_path)
        # 平均・分散
        means, stds = matrix.mean(axis=1), matrix.std(axis=1)
        sorted_idx = np.argsort(-means)
        subj_sorted = [subjects[i] for i in sorted_idx]
        save_dir = f"results/{dist_name.lower()}"
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"{dist_name.lower()}_mean.npy"), means)
        print(f"Saved mean distances ({dist_name}) to {os.path.join(save_dir, f'{dist_name.lower()}_mean.npy')}")
        np.save(os.path.join(save_dir, f"{dist_name.lower()}_std.npy"), stds)
        print(f"Saved standard deviations ({dist_name}) to {os.path.join(save_dir, f'{dist_name.lower()}_std.npy')}")
        np.save(os.path.join(save_dir, f"{dist_name.lower()}_mean_sorted.npy"), means[sorted_idx])
        print(f"Saved sorted mean distances ({dist_name}) to {os.path.join(save_dir, f'{dist_name.lower()}_mean_sorted.npy')}")
        np.save(os.path.join(save_dir, f"{dist_name.lower()}_std_sorted.npy"), stds[sorted_idx])
        print(f"Saved sorted standard deviations ({dist_name}) to {os.path.join(save_dir, f'{dist_name.lower()}_std_sorted.npy')}")
        with open(os.path.join(save_dir, f"{dist_name.lower()}_subjects_sorted.json"), "w") as f:
            json.dump(subj_sorted, f)
        print(f"Saved sorted subject list ({dist_name}) to {os.path.join(save_dir, f'{dist_name.lower()}_subjects_sorted.json')}")
        plot_bar(subj_sorted, means[sorted_idx], stds[sorted_idx], f"Mean and StdDev of {dist_name} per Subject (Sorted)", dist_name, os.path.join(save_dir, f"{dist_name.lower()}_mean_std_sorted.png"))
        print(f"Saved sorted bar plot ({dist_name}) to {os.path.join(save_dir, f'{dist_name.lower()}_mean_std_sorted.png')}")

        # グループ間距離・重心・プロット
        group_matrix, group_names = compute_group_dist_matrix(matrix, subjects, groups)
        group_dir = f"results/group_distances/{dist_name.lower()}"
        os.makedirs(group_dir, exist_ok=True)
        np.save(os.path.join(group_dir, "group_matrix.npy"), group_matrix)
        print(f"Saved group distance matrix ({dist_name}) to {os.path.join(group_dir, 'group_matrix.npy')}")
        with open(os.path.join(group_dir, "group_names.json"), "w") as f:
            json.dump(group_names, f)
        print(f"Saved group names to {os.path.join(group_dir, 'group_names.json')}")
        plot_heatmap(group_matrix, group_names, group_names, 
                     f"{dist_name} Distance Between Groups", 
                     os.path.join(group_dir, "group_heatmap.png"))
        print(f"Saved group heatmap ({dist_name}) to {os.path.join(group_dir, 'group_heatmap.png')}")
        plot_group_projection(matrix, subjects, groups, dist_name, 
                              os.path.join(save_dir, "group_overlay_projection.png"))
        print(f"Saved group overlay projection ({dist_name}) to {os.path.join(save_dir, 'group_overlay_projection.png')}")

        centroids = compute_group_centroids_from_distance_matrix(matrix, subjects, groups)
        if centroids:
            centroid_matrix, centroid_names = compute_group_centroid_distances(centroids)
            centroid_heatmap_path = os.path.join(save_dir, "group_centroid_distance_heatmap.png")
            plot_heatmap(centroid_matrix, centroid_names, centroid_names, 
                         f"{dist_name}: Group Centroid Distance", 
                         centroid_heatmap_path)
            print(f"Saved centroid distance heatmap ({dist_name}) to {centroid_heatmap_path}")

        # グループ内ばらつき・intra/inter可視化
        intra_results = compute_intra_group_variability(matrix, subjects, groups)
        intra_dir = f"results/group_distances/{dist_name.lower()}/intra"
        os.makedirs(intra_dir, exist_ok=True)
        intra_json_path = os.path.join(intra_dir, "intra_group_variability.json")
        with open(intra_json_path, "w") as f:
            json.dump(intra_results, f, indent=2)
        print(f"Saved intra-group variability JSON ({dist_name}) to {intra_json_path}")
        intra_csv_path = os.path.join(intra_dir, "intra_group_variability.csv")
        pd.DataFrame(intra_results).T.to_csv(intra_csv_path)
        print(f"Saved intra-group variability CSV ({dist_name}) to {intra_csv_path}")
        intra_plot_path = os.path.join(intra_dir, "intra_variability.png")
        plot_bar(list(intra_results.keys()), 
                 [intra_results[g]["mean"] for g in intra_results], 
                 [intra_results[g]["std"] for g in intra_results], 
                 f"Intra-group Variability - {dist_name}", "Mean Distance", 
                 intra_plot_path)
        print(f"Saved intra-group variability plot ({dist_name}) to {intra_plot_path}")

        stats = compute_intra_inter_stats(matrix, subjects, groups)
        intra_inter_dir = f"results/group_distances/{dist_name.lower()}/intra_inter"
        os.makedirs(intra_inter_dir, exist_ok=True)
        intra_inter_plot_path = os.path.join(intra_inter_dir, "intra_inter_comparison.png")
        plot_intra_inter(stats, dist_name, intra_inter_plot_path)
        print(f"Saved intra-inter comparison plot ({dist_name}) to {intra_inter_plot_path}")

if __name__ == "__main__":
    main()

