# docs/ directory

This directory contains the **Sphinx documentation** for the Vehicle-Based DDD Comparison project.

---

## Build Instructions

### 1. Install dependencies
```bash
cd docs
pip install -r requirements.txt
````

### 2. Build HTML documentation

```bash
make html
```

The generated documentation will be available at:

```
docs/_build/html/index.html
```

---

## Content

* **analysis.rst** → Analysis tools (distance, correlation, summaries)
* **data\_pipeline.rst** → Preprocessing pipelines & feature extraction
* **evaluation.rst** → Evaluation framework
* **models.rst** → Model architectures & pipelines
* **utils.rst** → Utility modules (I/O, visualization, domain generalization)
* **bin/** → Command-line entry points (`preprocess`, `train`, `evaluate`, etc.)

---

## Notes

* Use `autodoc` to automatically extract docstrings from `project/src/`.
* Style can be customized in `_static/` and `conf.py`.
* To add new modules, update `index.rst` and rebuild.

# misc/ directory

This directory contains utility scripts and configuration files used in experiments.

---

## Files
- **make_pretrain_group.py** → Generate pretraining subject groups  
- **make_target_groups.py** → Generate target subject lists  
- **run.sh** → Example shell script for launching jobs  
- **unzip.sh** → Utility to unpack dataset archives  
- **filelist.txt** → File listing for processed datasets  
- **subject_list.txt** → Master list of subjects  
- **target_groups.txt** → Target subject group definitions  
- **requirements.txt** → Runtime dependencies for training & evaluation  

---

## Notes
These scripts are not part of the main pipelines but help organize datasets and experiments.

---

## IO Overview (Input / Output Summary)

This section summarises the inputs and outputs of all major tasks in the project.

| Task (Script)        | Input                                                                 | Output                                                                 | Notes                                                                 |
|-----------------------|----------------------------------------------------------------------|------------------------------------------------------------------------|----------------------------------------------------------------------|
| **Preprocessing**<br>`bin/preprocess.py` | - Raw dataset (`dataset/mdapbe/*.mat`)<br>- Config (`src/config.py`) | - Processed CSV (`project/data/processed/[model]/`)<br>- Logs (`jobs/log/`) | Converts EEG, SIMlsl, and vehicle data into features; supports jittering |
| **Training**<br>`bin/train.py` | - Processed data (`data/processed/`)<br>- Model type (RF, LSTM, SvmA, SvmW, etc.)<br>- Split strategy / subject lists | - Trained model (`model/[type]/model.pkl`)<br>- Scaler (`scaler.pkl`)<br>- Selected features (`selected_features_train.pkl`)<br>- Feature metadata (`feature_meta.json`) | Uses Optuna for hyperparameter tuning; supports domain mixup, CORAL, VAE |
| **Evaluation**<br>`bin/evaluate.py` | - Trained models (`model/[type]/`)<br>- Test data / subject splits | - Metrics report (Accuracy, F1, AUC, AP)<br>- Evaluation CSVs (`results/`) | Can run subject-wise evaluation; outputs threshold-optimised F1 |
| **Analysis**<br>`bin/analyze.py` | - Summary CSVs (`model/common/summary_*.csv`)<br>- Distance matrices (MMD, Wasserstein, DTW)<br>- Group files (`misc/target_groups.txt`) | - Correlation results<br>- Distance heatmaps<br>- Plots (`results/analysis/`) | Computes domain distances and correlations vs. finetune gains |
| **PBS Jobs**<br>`jobs/*.sh` | - Job scripts (`pbs_*.sh`)<br>- Environment (Python, requirements.txt)<br>- Subject/group lists (`misc/`) | - HPC logs (`jobs/log/*.o*`, `*.e*`)<br>- Result files in `results/` and `model/` | Automates preprocessing, training, evaluation, and analysis on HPC cluster |

