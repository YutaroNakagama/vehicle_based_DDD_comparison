# Scripts Overview

This directory contains **all executable scripts** for the Driver Drowsiness Detection (DDD) project.  
It includes both **Python entry points** for local runs and **PBS job scripts** for cluster execution.

---

## Structure

```
scripts/
├── python/              # Python entry points (wrappers around src/*)
│   ├── train/           # Training scripts
│   ├── evaluation/      # Evaluation scripts
│   ├── preprocess/      # Data preprocessing scripts
│   ├── analysis/        # Post-hoc analysis scripts
│   ├── visualization/   # Plotting and visualization scripts
│   ├── setup/           # One-time setup and validation scripts
│   └── archive/         # Deprecated/legacy scripts
├── hpc/                 # HPC job scripts for JAIST KAGAYAKI cluster
│   ├── jobs/            # PBS job scripts (.sh)
│   │   ├── train/       # Model training jobs
│   │   ├── evaluate/    # Model evaluation jobs
│   │   ├── preprocess/  # Data preprocessing jobs
│   │   ├── imbalance/   # Imbalance method experiments
│   │   └── domain_analysis/  # Domain analysis jobs
│   ├── launchers/       # Batch job submission scripts
│   ├── lib/             # Shared shell functions
│   ├── templates/       # PBS job templates (examples)
│   └── logs/            # Job logs (*.OU, *.ER) - gitignored
```

---

## Python Entry Points (`scripts/python/`)

These are thin wrappers around `src/` modules, designed to be run as CLI tools.

### Main Scripts

- **`train/train.py`** → Model training (baseline, domain generalization, feature selection)
- **`evaluation/evaluate.py`** → Evaluation (cross-validation, subject-wise split, metrics)
- **`preprocess/preprocess.py`** → Data preprocessing (single/multi-process, augmentation)

### Analysis (`analysis/`)

- **`analyze.py`** → General analysis utilities
- **`analyze_imbalance_v3_results.py`** → Analyze imbalance method experiment results
- **`analyze_ranking_convergence.py`** → Analyze ranking convergence
- **`summarize_confusion_matrices.py`** → Aggregate confusion matrix results
- **`distance/`** → Distance-based analysis scripts
- **`domain/`** → Domain-specific analysis scripts
- **`imbalance/`** → Imbalance-related analysis scripts

### Visualization (`visualization/`)

- **`imbalance_visualize.py`** → Visualize imbalance experiment results
- **`plot_f2_boxplot.py`** → F2 score box plots
- **`plot_auprc_boxplot.py`** → AUPRC box plots
- **`plot_confusion_matrices.py`** → Confusion matrix visualizations
- **`visualize_baseline_metrics.py`** → Baseline metrics visualization
- **`visualize_hyperparams.py`** → Hyperparameter visualization
- **`visualize_multiseed_results.py`** → Multi-seed experiment results
- **`visualize_optuna_convergence.py`** → Optuna convergence plots

### Setup and Validation (`setup/`)

- **`generate_target_groups.py`** → Create target subject groups from master list
- **`generate_pretrain_group.py`** → Generate pretrain (general) subject list
- **`make_ranked_groups.py`** → Create ranked subject groups
- **`check_feature_columns.py`** → Validate feature consistency across subjects
- **`check_only_modes.py`** → Compare metrics between training modes

Note: ROC visualization is available via `python -m src.utils.visualization.plot_roc_cli`

Example:
```bash
python scripts/python/train/train.py --model RF --subject_wise_split
```

---

## HPC Job Scripts (`scripts/hpc/`)

PBS job scripts for submitting large-scale experiments to the **JAIST KAGAYAKI cluster**.
They activate the appropriate conda environment, set safe threading variables, and call the Python entry points.

### Directory Structure

* **`jobs/`**: PBS job scripts organized by task type
  * `train/` - Model training jobs
  * `evaluate/` - Model evaluation jobs
  * `preprocess/` - Data preprocessing jobs
  * `imbalance/` - Imbalance method experiments
  * `domain_analysis/` - Domain analysis jobs
* **`launchers/`**: Batch submission scripts for launching multiple jobs
* **`lib/`**: Shared shell functions (e.g., `common.sh`)
* **`templates/`**: PBS job templates (for creating new jobs)
* **`logs/`**: PBS log outputs (`*.OU`, `*.ER`) - gitignored

### Example Submissions

```bash
# Preprocessing
qsub scripts/hpc/jobs/preprocess/pbs_preprocess.sh

# Training (RF model)
qsub -v MODEL=RF scripts/hpc/jobs/train/pbs_train.sh

# Evaluation
qsub -v MODEL=RF,TAG=coral scripts/hpc/jobs/evaluate/pbs_evaluate.sh

# Batch launch all training jobs
bash scripts/hpc/launchers/launch_train_all.sh
```

---

## Notes

* Logs are always written under `scripts/hpc/logs/`.
* The `logs/` directory is **git-ignored**, but contains `README.md` for documentation.
* Actual results are saved under:
  * `results/` → metrics, CSVs, NumPy arrays
  * `models/` → trained model files
  * `models/` → trained models and scalers

