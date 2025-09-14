# Rank-based RF Analysis (Temporary)

This document describes how to analyze the results obtained from
`pbs_rank_10.sh` with the RF model.

## 1. Training jobs

Submit the array job:

```bash
qsub jobs/pbs_rank_10.sh
````

* The job runs for each target group listed in `misc/rank_names_10.txt`.
* Each index corresponds to a text file in `results/ranks10/`.
* Model: RandomForest (`RF`)
* Modes executed: `only_general → finetune → only_target`

## 2. Result files

Training artifacts are saved under:

```
project/
  model/common/
    RF_rank_*.pkl
    metrics_RF_rank_*.csv
    pr_*_RF_rank_*.{csv,png}
    roc_*_RF_rank_*.{csv,png}
    cm_*_RF_rank_*.{csv,png}
```

Suffix examples: `mmd_mean_high`, `mmd_mean_middle`, `mmd_mean_low`.

## 3. Aggregation & analysis

Run the helper script:

```bash
python misc/aggregate_summary_40cases.py
```

This script:

* Collects metrics across all runs.
* Produces summary CSV/plots for comparison.
* Currently tailored to RF model results.

Output files:

```
summary_only10_vs_finetune.csv
table_only10_vs_finetune_wide.csv
```

## 4. Notes

* Scripts in `misc/` are **temporary** and can be refactored into
  `src/analysis/` + `bin/` if they become part of the main workflow.
* For quick re-runs, adjust `rank_names_10.txt` to control which groups are processed.
* The fastest metric to compute (among DTW, Wasserstein, MMD) is **MMD**,
  so prefer MMD-only runs when testing.

