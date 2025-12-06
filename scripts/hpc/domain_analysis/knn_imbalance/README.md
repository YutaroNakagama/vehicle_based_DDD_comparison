# KNN Ranking + Imbalance Handling Comparison

## Overview

This experiment compares different imbalance handling methods on KNN-ranked subject groups.

### Full Experiment Design (135 jobs)

| Factor | Values | Count |
|--------|--------|-------|
| Distance Metrics | mmd, wasserstein, dtw | 3 |
| Training Modes | pooled, source_only, target_only | 3 |
| Domain Levels | out_domain, mid_domain, in_domain | 3 |
| Imbalance Methods | 5 methods (see below) | 5 |

**Total: 3 × 3 × 3 × 5 = 135 jobs**

### Imbalance Methods (Ordered by Complexity)

1. **baseline** - No resampling (control)
2. **undersample_rus** - Random Under-Sampling only
3. **undersample_tomek** - Tomek Links cleaning only
4. **smote_rus** - SMOTE + Random Under-Sampling
5. **smote_tomek** - SMOTE + Tomek Links cleaning

## Usage

### 1. Submit Full Training (Recommended)

```bash
./launch_knn_imbalance.sh full
```

This submits 135 jobs covering all combinations.

### 2. Submit Evaluation (after training completes)

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

## Output Structure

Results will be saved to:
- `results/domain_analysis/summary/knn_imbalance/knn_imbalance_results.csv`
- `results/domain_analysis/summary/knn_imbalance/png/`

## Research Questions

1. **Distance metric effect**: Does the choice of distance metric (MMD/Wasserstein/DTW) affect imbalance handling?

2. **Mode effect**: How does training mode interact with imbalance handling?
   - pooled: Train on all subjects
   - source_only: Train on non-target, eval on target
   - target_only: Train and eval on target subjects

3. **Domain level effect**: Do outlier subjects (out_domain) need different handling than typical subjects (in_domain)?

4. **Method comparison**: 
   - Is simple RUS sufficient, or is SMOTE needed?
   - Does Tomek cleaning improve over random under-sampling?

## Expected Results

Based on previous experiments:
- **KNN + MMD + target_only** was the best ranking configuration
- **SMOTE+RUS** showed best AUPRC in general imbalance experiments
- **BalancedRF** showed best F2 at default threshold

## Legacy Scripts

For backward compatibility, the old scripts are still available:
- `pbs_knn_baseline.sh` - Baseline only (6 jobs, MMD only)
- `pbs_knn_imbalance.sh` - Imbalance only (24 jobs, MMD only)
