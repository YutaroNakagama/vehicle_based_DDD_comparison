# KNN Ranking + Imbalance Handling Comparison

## Overview

This experiment compares different imbalance handling methods on KNN-ranked subject groups.

### Experiment Design

| Factor | Values |
|--------|--------|
| Ranking Method | KNN (best performing) |
| Distance Metric | MMD |
| Domain Levels | out_domain, mid_domain, in_domain |
| Training Modes | target_only, source_only |
| Imbalance Methods | 5 methods (see below) |

### Imbalance Methods (Ordered by Complexity)

1. **baseline** - No resampling (control)
2. **undersample_rus** - Random Under-Sampling only
3. **undersample_tomek** - Tomek Links cleaning only
4. **smote_rus** - SMOTE + Random Under-Sampling
5. **smote_tomek** - SMOTE + Tomek Links cleaning

### Total Jobs

- Baseline: 2 modes × 3 levels = **6 jobs**
- Imbalance: 2 modes × 3 levels × 4 methods = **24 jobs**
- Evaluation: **30 jobs**

## Usage

### 1. Submit Training Jobs

```bash
# Submit all jobs
./launch_knn_imbalance.sh all

# Or submit separately
./launch_knn_imbalance.sh baseline  # Baseline only (6 jobs)
./launch_knn_imbalance.sh imbal     # Imbalance methods (24 jobs)
```

### 2. Submit Evaluation Jobs (after training completes)

```bash
./launch_knn_imbalance.sh eval
```

### 3. Collect Results

```bash
# Summary only
python scripts/python/domain_analysis/collect_knn_imbalance_results.py

# With plots
python scripts/python/domain_analysis/collect_knn_imbalance_results.py --plot
```

## Expected Output

Results will be saved to:
- `results/domain_analysis/summary/knn_imbalance/knn_imbalance_results.csv`
- `results/domain_analysis/summary/knn_imbalance/png/knn_imbalance_comparison_*.png`

## Research Questions

1. **Does imbalance handling improve performance on KNN-ranked groups?**
   - Compare baseline vs resampling methods

2. **Which imbalance method works best for each domain level?**
   - out_domain (outliers) may need different handling than in_domain (typical)

3. **Is simple under-sampling sufficient, or is SMOTE needed?**
   - Compare RUS vs SMOTE+RUS

4. **Does Tomek cleaning improve over random under-sampling?**
   - Compare RUS vs Tomek

## Interpretation Guide

| Level | Subjects | Expected Behavior |
|-------|----------|-------------------|
| out_domain | Atypical drivers | High variability, may benefit from oversampling |
| mid_domain | Intermediate | Moderate performance |
| in_domain | Typical drivers | Most stable, baseline may be sufficient |
