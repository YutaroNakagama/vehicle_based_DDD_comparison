# Domain Distance Analysis with MDS, t-SNE, and UMAP

## Overview
This directory contains results of domain distance analysis using three dimensionality reduction methods (MDS, t-SNE, UMAP).

## Computed Metrics

### 1. Intra-domain Distance
Mean distance from each group's domain center to each subject in that group

- **High group**: High-distance group (29 subjects)
- **Middle group**: Middle-distance group (29 subjects)
- **Low group**: Low-distance group (29 subjects)

### 2. Inter-domain Distance
Distance from each group's domain center to the Middle group's domain center

## File Structure

### Detailed Visualizations (per metric x per method)
- `{metric}_{method}_domain_distances.png` (9 files)
  - 3-panel layout:
    1. Dimensionality reduction projection (with domain centers)
    2. Intra-domain distance bar chart
    3. Inter-domain distance bar chart

### Numeric Data (JSON format)
- `{metric}_{method}_domain_distances.json` (9 files)
  - Domain center coordinates
  - Intra-domain distance (mean, std, min, max)
  - Inter-domain distance

### Summary CSV (per method)
- `mds_domain_distances_summary.csv`
- `tsne_domain_distances_summary.csv`
- `umap_domain_distances_summary.csv`

CSV columns:
- Metric: Distance metric (DTW, MMD, WASSERSTEIN)
- Group: Group name (High, Middle, Low)
- Intra_Mean: Mean intra-group distance
- Intra_Std: Intra-group distance standard deviation
- Inter_to_Middle: Distance to Middle domain center

### Comparative Visualizations (per method)
- `mds_domain_distances_comparison.png`
- `tsne_domain_distances_comparison.png`
- `umap_domain_distances_comparison.png`

2x3 grid layout:
- Top row: Intra-domain distance comparison across 3 metrics
- Bottom row: Inter-domain distance comparison across 3 metrics

## Key Findings

### DTW Distance Results
- **MDS**: High group has the largest spread (Intra: 6.02±4.56) and is most isolated (Inter: 5.97)
- **t-SNE**: High group is most isolated (Inter: 4.00)
- **UMAP**: High group is most isolated (Inter: 2.50)

### MMD Distance Results
- High group shows the largest intra-domain distance across all methods
- High group has the largest inter-domain distance in UMAP (1.33)

### Wasserstein Distance Results
- High group shows the largest intra-domain and inter-domain distances across all methods
- Most prominent separation in t-SNE (High Inter: 2.80)

## Interpretation

1. **High group characteristics**:
   - Largest intra-domain distance across all methods → high within-group diversity
   - Largest inter-domain distance across all methods → most isolated from Middle group
   - This diversity and isolation may contribute to higher Recall

2. **Low group characteristics**:
   - Smallest intra-domain distance → high within-group consistency
   - Closest to Middle group → typical subject patterns

3. **Comparison of dimensionality reduction methods**:
   - **MDS**: Preserves original distance structure most faithfully (especially prominent for DTW)
   - **t-SNE**: Emphasizes local structure, clear cluster separation
   - **UMAP**: Good balance between global and local structure

## Generated
November 23, 2024

## Scripts
`scripts/python/compute_umap_domain_distances.py`
