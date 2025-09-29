# Scripts Overview

This directory contains **all executable scripts** for the Driver Drowsiness Detection (DDD) project.  
It includes both **Python entry points** for local runs and **PBS job scripts** for cluster execution.

---

## Structure

```

scripts/
├── python/      # Python entry points (wrappers around src/*)
├── hpc/         # HPC job scripts for JAIST KAGAYAKI cluster
│   ├── preprocess/       # Data preprocessing jobs
│   ├── train/            # Model training jobs
│   ├── evaluate/         # Model evaluation jobs
│   ├── domain_gen/       # Domain generalization experiments
│   ├── templates/        # PBS job templates (examples)
│   └── log/              # Job logs (*.oXXXX, *.eXXXX)

````

---

## Python Entry Points (`scripts/python/`)

These are thin wrappers around `src/` modules, designed to be run as CLI tools.

- **`preprocess.py`** → Data preprocessing (single/multi-process, augmentation)  
- **`train.py`** → Model training (baseline, only10, finetune, domain generalization, feature selection)  
- **`evaluate.py`** → Evaluation (cross-validation, subject-wise split, metrics)  
- **`analyze.py`** → Post-hoc analysis (distances, correlations, summary tables, rankings)  
- **`plot.py`** → ROC and metric visualizations  
- **`make_pretrain_groups.py`** → Wide-format comparison of *only10* vs *finetune*  

Example:
```bash
python scripts/python/train.py --model RF --subject_wise_split
````

---

## HPC Job Scripts (`scripts/hpc/`)

PBS job scripts for submitting large-scale experiments to the **JAIST KAGAYAKI cluster**.
They activate the appropriate conda environment, set safe threading variables, and call the Python entry points.

### Categories

* **preprocess/**: Data preparation jobs
* **train/**: Model training jobs
* **evaluate/**: Model evaluation jobs
* **domain_gen/**: Domain generalization jobs
* **templates/**: PBS job templates (for creating new jobs)
* **log/**: PBS log outputs (`*.oXXXX`, `*.eXXXX`)

Example submissions:

```bash
# Preprocessing
qsub scripts/hpc/preprocess/pbs_preprocess.sh

# Training (RF model)
qsub -v MODEL=RF scripts/hpc/train/pbs_train.sh

# Evaluation
qsub -v MODEL=RF,TAG=coral scripts/hpc/evaluate/pbs_evaluate.sh
```

---

## Notes

* Logs are always written under `scripts/hpc/log/`.
* Most HPC scripts are **git-ignored**, except for `templates/` and README.
* Actual results are saved under:

  * `results/` → metrics, CSVs, NumPy arrays
  * `models/` → trained models and scalers

