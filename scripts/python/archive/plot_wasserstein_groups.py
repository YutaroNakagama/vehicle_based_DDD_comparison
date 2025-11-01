"""
Plot distributions of selected features across Wasserstein groups (High, Middle, Low).
Generates both violin and box plots with linear and log scales.
"""

import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def load_subjects(group_file):
    """Load subject IDs from a group file (space-separated)."""
    with open(group_file) as f:
        return f.read().strip().split()

# Define groups
groups = {
    "High": load_subjects("results/ranks10/wasserstein_mean_high.txt"),
    "Middle": load_subjects("results/ranks10/wasserstein_mean_middle.txt"),
    "Low": load_subjects("results/ranks10/wasserstein_mean_low.txt"),
}

# Output directory
out_dir = Path("reports/figures/wasserstein_selected")
out_dir.mkdir(parents=True, exist_ok=True)

# Load all subject CSVs
data_root = Path("data/processed/common")
dfs = []
for label, subjects in groups.items():
    for subj in subjects:
        csv_path = list(data_root.glob(f"processed_{subj}.csv"))[0]
        df = pd.read_csv(csv_path)
        df["Group"] = label
        dfs.append(df)
all_data = pd.concat(dfs, ignore_index=True)

# Load selected features from .pkl files
feat_dir = Path("models/common")
pkls = [
    feat_dir / "selected_features_train_RF_only_target_rank_wasserstein_mean_high.pkl",
    feat_dir / "selected_features_train_RF_only_target_rank_wasserstein_mean_middle.pkl",
    feat_dir / "selected_features_train_RF_only_target_rank_wasserstein_mean_low.pkl",
]

selected_features = set()
for pkl in pkls:
    with open(pkl, "rb") as f:
        feats = pickle.load(f)
        selected_features.update(feats)

selected_features = sorted(selected_features)
print(f"Number of selected features: {len(selected_features)}")

# Plot distributions for each feature
for col in tqdm(selected_features, desc="Plotting features"):
    if col not in all_data.columns:
        print(f"[warn] {col} not found in dataset, skipped.")
        continue

    for scale in ["linear", "log"]:
        suffix = "linear" if scale == "linear" else "log"

        # Violin plot
        plt.figure(figsize=(6, 4))
        sns.violinplot(data=all_data, x="Group", y=col, density_norm="width", cut=0)
        if scale == "log":
            plt.yscale("log")
        plt.title(f"{col} (Violin, {suffix} scale)")
        plt.tight_layout()
        plt.savefig(out_dir / f"{col}_violin_{suffix}.png")
        plt.close()

        # Box plot
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=all_data, x="Group", y=col)
        if scale == "log":
            plt.yscale("log")
        plt.title(f"{col} (Boxplot, {suffix} scale)")
        plt.tight_layout()
        plt.savefig(out_dir / f"{col}_boxplot_{suffix}.png")
        plt.close()

print(f"Finished: {len(selected_features)*4} PNGs saved in {out_dir}")

