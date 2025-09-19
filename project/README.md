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

---

## 2. Train-only jobs (skip evaluation)

When you want to **pre-train models only** (save model, features, scaler)
without running validation/test, set the environment variable `TRAIN_ONLY=true`:

```bash
qsub -v TRAIN_ONLY=true jobs/pbs_rank_10_was_general_vs_target.sh
```

* Modes executed: `only_general (train_only)` and `only_target (train_only)`
* No evaluation metrics are produced.
* Saved artifacts (for later eval) include:

```
model/common/
  RF_rank_*.pkl
  selected_features_train_RF_*.pkl
  scaler_RF_*.pkl
```

---

## 3. Eval-only jobs (skip retraining)

When you want to **load pre-trained models** and run only validation & test
(without retraining), set the environment variable `EVAL_ONLY=true` and use
the job script with eval support:

```bash
qsub -v EVAL_ONLY=true jobs/pbs_rank_10_was_general_vs_target.sh
```

* Modes executed: `only_general (eval_only)` and `only_target (eval_only)`
* Metrics are saved under:

```
model/common/
  metrics_RF_rank_*_evalonly_on_targets.csv
```

---

## 4. Result files

Training or eval-only artifacts are saved under:

```
project/
  model/common/
    RF_rank_*.pkl
    metrics_RF_rank_*.csv
    metrics_RF_rank_*_evalonly_on_targets.csv
    pr_*_RF_rank_*.{csv,png}
    roc_*_RF_rank_*.{csv,png}
    cm_*_RF_rank_*.{csv,png}
```

Suffix examples: `mmd_mean_high`, `mmd_mean_middle`, `mmd_mean_low`.

---

## 5. Aggregation & analysis

Run the helper script:

```bash
python misc/aggregate_summary_40cases.py
```

This script:

* Collects metrics across all runs.
* Produces summary CSVs for comparison.

Output files:

```
results/analysis/summary_40cases_all_splits.csv
results/analysis/summary_40cases_test.csv
results/analysis/summary_40cases_test_mode_compare.csv
```

---

## 6. Plotting

To generate figures from the aggregated results:

```bash
python misc/plot_summary_metrics_40.py
```

This produces:

```
results/analysis/summary_metrics_40_mean_tri_bar.png
results/analysis/diff_heatmap_auc_ap_40_tri.png
```

* `summary_metrics_40_mean_tri_bar.png`: tri-bar plots comparing
  General / Target / Finetune across distance metrics.
* `diff_heatmap_auc_ap_40_tri.png`: heatmaps of AUC/AP differences
  (General–Target, Finetune–Target, General–Finetune).

---

## 7. Notes

* Scripts in `misc/` are **temporary** and can be refactored into
  `src/analysis/` + `bin/` if they become part of the main workflow.
* For quick re-runs, adjust `rank_names_10.txt` to control which groups are processed.
* The fastest metric to compute (among DTW, Wasserstein, MMD) is **MMD**,
  so prefer MMD-only runs when testing.


