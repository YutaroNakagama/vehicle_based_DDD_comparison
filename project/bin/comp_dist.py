import os
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from tslearn.metrics import dtw
from tqdm import tqdm

# === Feature Loader ===
def load_features(subject_str):
    subject_id, version = subject_str.split("_")
    path = f"data/processed/common/processed_{subject_id}_{version}.csv"
    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return None
    df = pd.read_csv(path)
    feature_cols = [col for col in df.columns if col not in ["Timestamp", "KSS", "subject"]]
    return df[feature_cols].dropna().values

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

for i in tqdm(range(n)):
    for j in range(n):
        dists = []
        min_dim = min(features[valid_subjects[i]].shape[1], features[valid_subjects[j]].shape[1])
        for k in range(min_dim):
            xi = features[valid_subjects[i]][:, k]
            yj = features[valid_subjects[j]][:, k]
            dists.append(wasserstein_distance(xi, yj))
        wass_matrix[i, j] = np.mean(dists)

# === DTW Distance Matrix (using mean feature vector per subject) ===
print("Computing DTW Distance Matrix (per-subject mean vector)...")
dtw_matrix = np.zeros((n, n))
mean_series = [features[subj].mean(axis=1) for subj in valid_subjects]

for i in tqdm(range(n)):
    for j in range(n):
        dtw_matrix[i, j] = dtw(mean_series[i], mean_series[j])

# === Save results ===
os.makedirs("results/distances", exist_ok=True)
np.save("results/distances/wasserstein_matrix.npy", wass_matrix)
np.save("results/distances/dtw_matrix.npy", dtw_matrix)

with open("results/distances/subjects.json", "w") as f:
    import json
    json.dump(valid_subjects, f)

print("Distance matrices saved to results/distances/")

import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.manifold import MDS, TSNE
import umap
import os

# === 距離行列の種類とファイル対応表 ===
distance_types = {
    "MMD": "results/mmd/mmd_matrix.npy",
    "Wasserstein": "results/distances/wasserstein_matrix.npy",
    "DTW": "results/distances/dtw_matrix.npy"
}

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

# === 実行処理 ===
for dist_name, matrix_path in distance_types.items():
    # 対応するsubjects.jsonの場所
    subj_path = "results/mmd/mmd_subjects.json" if dist_name == "MMD" else "results/distances/subjects.json"
    subjects = load_subjects(subj_path)
    matrix = np.load(matrix_path)

    outdir = f"results/projection/{dist_name.lower()}"
    os.makedirs(outdir, exist_ok=True)

    print(f"Processing: {dist_name}")

    # --- MDS ---
    print("  - MDS...")
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    coords_mds = mds.fit_transform(matrix)
    plot_projection(coords_mds, subjects, "MDS", dist_name, outdir)

    # --- t-SNE ---
    print("  - t-SNE...")
    tsne = TSNE(n_components=2, metric="precomputed", random_state=42)
    coords_tsne = tsne.fit_transform(matrix)
    plot_projection(coords_tsne, subjects, "t-SNE", dist_name, outdir)

    # --- UMAP ---
    print("  - UMAP...")
    reducer = umap.UMAP(n_components=2, metric='precomputed', random_state=42)
    coords_umap = reducer.fit_transform(matrix)
    plot_projection(coords_umap, subjects, "UMAP", dist_name, outdir)

print("✅ All projections completed and saved in 'results/projection/'")

