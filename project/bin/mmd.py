import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm  # for progress bar

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
    # Exclude non-feature columns
    feature_cols = [col for col in df.columns if col not in ["Timestamp", "KSS", "subject"]]
    return df[feature_cols].dropna().values

# === Load subject list ===
with open("../../dataset/mdapbe/subject_list.txt") as f:
    subjects = [line.strip() for line in f.readlines()]

# === Feature extraction ===
features = {}
for subj in subjects:
    data = load_features(subj)
    if data is not None:
        features[subj] = data

# === MMD Matrix with progress ===
valid_subjects = list(features.keys())
n = len(valid_subjects)
mmd_matrix = np.zeros((n, n))

print("Computing MMD matrix...")
for i in tqdm(range(n), desc="Rows"):
    for j in range(n):
        mmd_matrix[i, j] = compute_mmd(features[valid_subjects[i]], features[valid_subjects[j]])

# === Plot and Save ===
plt.figure(figsize=(12, 10))
sns.heatmap(mmd_matrix, xticklabels=valid_subjects, yticklabels=valid_subjects, cmap='viridis')
plt.title("MMD Distance Between Subjects (Features from Processed CSVs)")
plt.tight_layout()

# 保存先ディレクトリを作成して保存
os.makedirs("results", exist_ok=True)

import json

# === Save MMD matrix and subject labels ===
os.makedirs("results/mmd", exist_ok=True)

np.save("results/mmd/mmd_matrix.npy", mmd_matrix)
with open("results/mmd/mmd_subjects.json", "w") as f:
    json.dump(valid_subjects, f)

plt.savefig("results/mmd/mmd_matrix.png")

# === Compute mean and stddev per subject ===
mmd_mean = mmd_matrix.mean(axis=1)
mmd_std = mmd_matrix.std(axis=1)

# === Save mean/std data ===
np.save("results/mmd/mmd_mean.npy", mmd_mean)
np.save("results/mmd/mmd_std.npy", mmd_std)

# === Plot mean and std ===
plt.figure(figsize=(14, 6))
plt.errorbar(valid_subjects, mmd_mean, yerr=mmd_std, fmt='o', capsize=5)
plt.xticks(rotation=90)
plt.title("Mean and StdDev of MMD per Subject")
plt.ylabel("MMD")
plt.tight_layout()
plt.savefig("results/mmd/mmd_mean_std.png")
plt.close()

# === Sort by mean MMD (descending) ===
sorted_indices = np.argsort(-mmd_mean)
mmd_mean_sorted = mmd_mean[sorted_indices]
mmd_std_sorted = mmd_std[sorted_indices]
subjects_sorted = [valid_subjects[i] for i in sorted_indices]

# Save sorted data
np.save("results/mmd/mmd_mean_sorted.npy", mmd_mean_sorted)
np.save("results/mmd/mmd_std_sorted.npy", mmd_std_sorted)
with open("results/mmd/mmd_subjects_sorted.json", "w") as f:
    json.dump(subjects_sorted, f)

# Plot sorted bar chart with error bars
plt.figure(figsize=(14, 6))
plt.errorbar(subjects_sorted, mmd_mean_sorted, yerr=mmd_std_sorted, fmt='o', capsize=5)
plt.xticks(rotation=90)
plt.title("Mean and StdDev of MMD per Subject (Sorted)")
plt.ylabel("MMD")
plt.tight_layout()
plt.savefig("results/mmd/mmd_mean_std_sorted.png")
plt.close()

import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.manifold import MDS, TSNE
import umap

# === Load distance matrix and subject labels ===
mmd_matrix = np.load("results/mmd/mmd_matrix.npy")
with open("results/mmd/mmd_subjects.json") as f:
    subjects = json.load(f)

# === Create output dir if needed ===
import os
os.makedirs("results/mmd/projection", exist_ok=True)

# === Common plotting function ===
def plot_projection(coords, method_name, save_path):
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1])
    for i, label in enumerate(subjects):
        plt.text(coords[i, 0], coords[i, 1], label, fontsize=8)
    plt.title(f"{method_name} Projection of MMD Distances")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# === MDS ===
print("Running MDS...")
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
coords_mds = mds.fit_transform(mmd_matrix)
plot_projection(coords_mds, "MDS", "results/mmd/projection/mds_projection.png")

# === t-SNE ===
print("Running t-SNE...")
tsne = TSNE(n_components=2, metric="precomputed", random_state=42)
coords_tsne = tsne.fit_transform(mmd_matrix)
plot_projection(coords_tsne, "t-SNE", "results/mmd/projection/tsne_projection.png")

# === UMAP ===
print("Running UMAP...")
reducer = umap.UMAP(n_components=2, metric='precomputed', random_state=42)
coords_umap = reducer.fit_transform(mmd_matrix)
plot_projection(coords_umap, "UMAP", "results/mmd/projection/umap_projection.png")

print("Projections saved to results/mmd/projection/")

#import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#import json
#import os
#
## Load
#matrix = np.load("results/mmd/mmd_matrix.npy")
#with open("results/mmd/mmd_subjects.json") as f:
#    subjects = json.load(f)
#
## Plot
#plt.figure(figsize=(12, 10))
#sns.heatmap(matrix, xticklabels=subjects, yticklabels=subjects, cmap='viridis')
#plt.title("MMD Distance Between Subjects (Reloaded)")
#plt.tight_layout()
#plt.savefig("results/mmd/mmd_matrix_reloaded.png")
#plt.show()
#

#import numpy as np
#import matplotlib.pyplot as plt
#import json
#
## Load data
#mmd_mean = np.load("results/mmd/mmd_mean.npy")
#mmd_std = np.load("results/mmd/mmd_std.npy")
#with open("results/mmd/mmd_subjects.json") as f:
#    subjects = json.load(f)
#
## Plot
#plt.figure(figsize=(14, 6))
#plt.errorbar(subjects, mmd_mean, yerr=mmd_std, fmt='o', capsize=5)
#plt.xticks(rotation=90)
#plt.title("Mean and StdDev of MMD per Subject (Reloaded)")
#plt.ylabel("MMD")
#plt.tight_layout()
#plt.savefig("results/mmd/mmd_mean_std_reloaded.png")
#plt.show()
#

#import numpy as np
#import matplotlib.pyplot as plt
#import json
#
## Load sorted data
#mmd_mean = np.load("results/mmd/mmd_mean_sorted.npy")
#mmd_std = np.load("results/mmd/mmd_std_sorted.npy")
#with open("results/mmd/mmd_subjects_sorted.json") as f:
#    subjects = json.load(f)
#
## Plot
#plt.figure(figsize=(14, 6))
#plt.errorbar(subjects, mmd_mean, yerr=mmd_std, fmt='o', capsize=5)
#plt.xticks(rotation=90)
#plt.title("Mean and StdDev of MMD per Subject (Sorted & Reloaded)")
#plt.ylabel("MMD")
#plt.tight_layout()
#plt.savefig("results/mmd/mmd_mean_std_sorted_reloaded.png")
#plt.show()
#
