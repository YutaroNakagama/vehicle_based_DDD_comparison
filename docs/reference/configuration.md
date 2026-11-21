# Configuration Reference

This document describes the main parameters in `src/config.py`. Values marked "env" can be
overridden with the environment variable given.

## Path Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SUBJECT_LIST_PATH` | `config/subjects/subject_list.txt` (env `DDD_SUBJECT_LIST_PATH`) | The 87 subject-sessions |
| `DATASET_PATH` | `../dataset/mdapbe/physio` (env `DDD_DATASET_PATH`) | Raw upstream data (SIMlsl, EEG) |
| `INTRIM_CSV_PATH` | `./data/interim` | Interim per-signal CSVs |
| `PROCESS_CSV_COMMON_PATH` | `./data/processed/common` | Feature tables used by the RF |
| `MODEL_PKL_PATH` | `./models/` (env `DDD_MODEL_PKL_PATH`) | Trained models (not distributed) |
| `RESULTS_OUTPUTS_TRAINING_PATH` | `./results/outputs/training` | Per-run training records |
| `RESULTS_OUTPUTS_EVALUATION_PATH` | `./results/outputs/evaluation` | Per-run evaluation records |
| `RESULTS_ANALYSIS_EXP2_PATH` | `./results/analysis/exp2_domain_shift` | Metric tables, distances and figures of the paper |

## Signal Processing

| Variable | Value | Description |
|----------|-------|-------------|
| `SAMPLE_RATE_SIMLSL` | 60 Hz | Vehicle-dynamics signals |
| `SAMPLE_RATE_EEG` | 500 Hz | EEG signals (used only for the labels) |
| `WAVELET_LEV` | 3 | Wavelet-packet decomposition level |

The paper uses the `common` window configuration of `MODEL_WINDOW_CONFIG`: windows of
3 s with a 1.5 s step (`{"window_sec": 3, "step_sec": 1.5}`).

## Labeling Scheme

```python
KSS_BIN_LABELS = [1, 2, 3, 4, 5, 8, 9]
KSS_LABEL_MAP  = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 8: 1, 9: 1}   # 0 = alert, 1 = drowsy
```

KSS levels 6 and 7 are excluded. Setting `KSS_SIMPLIFIED=1` switches to a test scheme
(1–3 alert, 8–9 drowsy) that was not used in the paper.

## Data Split

| Variable | Default | Description |
|----------|---------|-------------|
| `TRAIN_RATIO` / `VAL_RATIO` / `TEST_RATIO` | 0.6 / 0.2 / 0.2 (env) | Train/validation/test ratios of the time-stratified split (`--time_stratify_labels`) |
| `TIME_STRATIFY_TOLERANCE` | 0.02 | Tolerance for the positive-class ratio in the time-stratified split |
| `DEFAULT_RANDOM_SEED` | 42 | Seed when `--seed` is not given |

## Training Parameters

### Feature selection

| Variable | Default | Description |
|----------|---------|-------------|
| `TOP_K_FEATURES` | 10 | Features kept by the RF importance ranking |

### Optuna hyperparameter tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `N_TRIALS` | 50 (env `N_TRIALS_OVERRIDE`) | Optuna trials; the paper's job script sets 100 |
| `OPTUNA_N_STARTUP_TRIALS` | 5 | Random trials before TPE |
| `OPTUNA_N_WARMUP_STEPS` | 1 | Pruner warm-up steps |
| `OPTUNA_INTERVAL_STEPS` | 1 | Pruning check interval |

### Cross-validation

| Variable | Default | Description |
|----------|---------|-------------|
| `CV_FOLDS_OPTUNA` | 3 | CV folds during optimization |
| `CV_FOLDS_OPTUNA_DATA_LEAK` | 2 | CV folds of the leakage-check variant |
| `CV_FOLDS_CALIBRATION` | 5 | CV folds for probability calibration |

### Model hyperparameters

| Variable | Default | Description |
|----------|---------|-------------|
| `RF_CLASS_WEIGHT` | `{0: 1.0, 1: 3.0}` | Random-forest class weights |

## Evaluation

| Variable | Default | Description |
|----------|---------|-------------|
| `FBETA_SCORE_BETA` | 2.0 | β of the F-β score |
| `MIN_RECALL_THRESHOLD` | 0.70 | Minimum recall for the precision-at-recall threshold |
| `BASELINE_AUPRC` | 0.039 | AUPRC of a random classifier at the drowsy prevalence |

## Domain Analysis

| Variable | Value | Description |
|----------|-------|-------------|
| `DISTANCE_METRICS` | `["mmd", "wasserstein", "dtw"]` | Subject distance metrics |

The In/Out subject lists of the paper are distributed in the Zenodo bundle
(`results/analysis/exp2_domain_shift/distance/rankings/split2/knn/`); see
`DATADICTIONARY.md`, Section 5.

## Overriding parameters

Prefer the environment variables above. For other parameters, override the module
attributes before the pipeline is imported, e.g.

```python
import src.config as cfg
cfg.CV_FOLDS_OPTUNA = 2
```
