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

## Data Flow: Modules and IO Summary

| Module | Input | Output | Description |
|--------|-------|--------|-------------|
| **Preprocess**<br>`bin/preprocess.py` + `src/data_pipeline/*` | - Raw dataset (EEG, steering, physio, KSS, etc.)<br>- Subject list (`misc/subject_list.txt`)<br>- CLI args: `--model`, `--jittering`, `--multi_process` | - Processed CSV files (`data/processed/[model]/processed_*.csv`)<br>- Logs of preprocessing steps | Extract features (time–frequency, wavelet, EEG, KSS, etc.), optional jittering, merge per subject. |
| **Train**<br>`bin/train.py` + `src/models/model_pipeline.py` | - Processed CSVs<br>- Subject list(s)<br>- CLI args: `--model`, `--mode`, `--feature_selection`, `--domain_mixup`, `--coral`, `--vae`, etc. | - Trained model (`model/[type]/*.pkl` / `.keras`)<br>- Scaler (`scaler_*.pkl`)<br>- Selected features (`selected_features_*.pkl`)<br>- Training metrics (CSV/log) | Splits dataset (random / subject-wise / finetune). Performs feature selection, scaling, domain generalization, and trains models (LSTM, SvmA, RF, etc.). |
| **Evaluate**<br>`bin/evaluate.py` + `src/evaluation/eval_pipeline.py` | - Trained model artifacts (`model/[type]/*.pkl`, `.keras`, scalers, features)<br>- Processed CSVs<br>- CLI args: `--model`, `--tag`, `--subject_wise_split`, etc. | - Evaluation metrics (`results/[model]/metrics_*.json`)<br>- Confusion matrices (log)<br>- Validation/Test scores | Loads trained models, aligns features, applies scaler, evaluates performance. Supports LSTM, SvmA, and classical ML models. |
| **Analysis**<br>`bin/analyze.py` + `src/analysis/*` | - Evaluation results (`model/[type]/metrics_*.csv` / `.json`)<br>- Distance matrices (`results/mmd/*.npy`, `results/distances/*.npy`)<br>- Group definitions (`misc/pretrain_groups/*.txt`)<br>- CLI args: subcommand (`comp-dist`, `corr`, `summarize`, etc.) | - Distance matrices (MMD, Wasserstein, DTW)<br>- Correlation CSV/PNG<br>- Summaries (`summary_*.csv`)<br>- Comparison tables (`table_*.csv`)<br>- Ranking lists (`results/ranks/*.csv`)<br>- Visualization (`heatmap.png`, `radar.png`) | Unified CLI for research analysis: compute distances, correlate with evaluation deltas, summarize only10 vs finetune, build tables, rank subjects, and generate plots. |


## End-to-End Data Flow

```mermaid
flowchart TD
    %% --- Input Stage ---
    subgraph RawData["Raw Data Sources"]
        R1[EEG Signals]
        R2[Vehicle Dynamics<br>(Steering, Lane Offset)]
        R3[Physiological Signals<br>(Pupil, PERCLOS, etc.)]
        R4[KSS Labels]
        R5[Subject List / Groups]
    end

    %% --- Preprocess Stage ---
    subgraph Preprocess["Preprocess (bin/preprocess.py)"]
        P1[Feature Extraction<br>- Time-Freq / Wavelet / EEG / Physio / KSS]
        P2[Optional Augmentation<br>(Jittering)]
        P3[Merge per Subject<br>+ Save Processed CSV]
    end

    R1 --> P1
    R2 --> P1
    R3 --> P1
    R4 --> P1
    R5 --> P1
    P1 --> P2 --> P3

    %% --- Train Stage ---
    subgraph Train["Train (bin/train.py)"]
        T1[Data Split<br>(Random / Subject-wise / Finetune)]
        T2[Domain Generalization<br>(Mixup / CORAL / VAE)]
        T3[Feature Selection<br>(RF / MI / ANOVA)]
        T4[Scaling (StandardScaler)]
        T5[Model Training<br>(RF, SvmA/W, LSTM, etc.)]
        T6[Save Artifacts<br>(model.pkl / scaler.pkl / features.pkl)]
    end

    P3 --> T1 --> T2 --> T3 --> T4 --> T5 --> T6

    %% --- Evaluate Stage ---
    subgraph Evaluate["Evaluate (bin/evaluate.py)"]
        E1[Load Artifacts<br>(model, scaler, features)]
        E2[Split Data<br>(random / subject-wise)]
        E3[Align Features + Scaling]
        E4[Compute Metrics<br>(Accuracy, F1, AUC, AP, ConfMat)]
        E5[Save Metrics JSON/CSV]
    end

    T6 --> E1
    P3 --> E2
    E1 --> E3 --> E4 --> E5

    %% --- Analysis Stage ---
    subgraph Analysis["Analysis (bin/analyze.py)"]
        A1[comp-dist<br>Compute MMD/Wasserstein/DTW]
        A2[corr<br>d(U,G) vs Δmetrics]
        A3[summarize<br>only10 vs finetune]
        A4[summarize-metrics<br>Aggregate CSVs]
        A5[make-table<br>Wide Comparison]
        A6[report-pretrain-groups<br>Intra/Inter/NN stats]
        A7[corr-collect<br>Heatmap]
        A8[rank-export<br>Top/Bottom-k lists]
    end

    E5 --> A2
    A1 --> A2
    A1 --> A6
    A1 --> A8
    R5 --> A1
    R5 --> A3
    A2 -->|Correlation CSV/PNG| Analysis
    A3 -->|Summary CSV| Analysis
    A4 -->|Aggregated CSV| Analysis
    A5 -->|Table CSV| Analysis
    A6 -->|Summary JSON/CSV + Radar| Analysis
    A7 -->|Heatmap PNG| Analysis
    A8 -->|Ranking CSV| Analysis

    %% --- Outputs ---
    subgraph Results["Outputs"]
        O1[Processed CSVs]
        O2[Model Artifacts<br>(.pkl / .keras / scaler / features)]
        O3[Evaluation Metrics<br>(CSV/JSON)]
        O4[Distance Matrices<br>(.npy / .json)]
        O5[Summaries / Tables<br>(.csv)]
        O6[Visualizations<br>(heatmap.png / radar.png)]
        O7[Rankings<br>(top-k, bottom-k .csv)]
    end

    P3 --> O1
    T6 --> O2
    E5 --> O3
    A1 --> O4
    A2 --> O5
    A3 --> O5
    A4 --> O5
    A5 --> O5
    A6 --> O5
    A6 --> O6
    A7 --> O6
    A8 --> O7

## Analysis Subflow (Detail)

```mermaid
flowchart TD
    %% Inputs
    M1[Evaluation Metrics<br>(results/[model]/metrics_*.csv)]
    M2[Distance Matrices<br>(results/mmd/*.npy, results/distances/*.npy)]
    M3[Group Definitions<br>(misc/pretrain_groups/*.txt)]

    %% comp-dist
    A1[comp-dist<br>Compute distances (MMD / Wasserstein / DTW)]
    M1 --> A1
    M2 --> A1
    M3 --> A1
    A1 --> D1[Distance Outputs<br>(.npy, .json)]

    %% corr
    A2[corr<br>Correlate d(U,G), disp(G) with Δmetrics]
    M1 --> A2
    A1 --> A2
    A2 --> D2[Correlation CSV<br>& Heatmap PNG]

    %% summarize
    A3[summarize<br>only10 vs finetune<br>(radar plot optional)]
    M1 --> A3
    M3 --> A3
    A3 --> D3[Summary CSV<br>& Radar Plot]

    %% summarize-metrics
    A4[summarize-metrics<br>Aggregate metrics_*.csv]
    M1 --> A4
    A4 --> D4[Summary (long-form) CSV]

    %% make-table
    A5[make-table<br>Wide comparison table]
    A4 --> A5
    A3 --> A5
    A5 --> D5[Wide Table CSV]

    %% report-pretrain-groups
    A6[report-pretrain-groups<br>Intra / Inter / NN stats]
    A1 --> A6
    M3 --> A6
    A6 --> D6[Group Summary<br>JSON & CSV]

    %% corr-collect
    A7[corr-collect<br>Merge correlations & heatmap]
    A2 --> A7
    A6 --> A7
    A7 --> D7[Correlation Heatmap PNG<br>& CSV]

    %% rank-export
    A8[rank-export<br>Top/Bottom-k subject lists]
    A1 --> A8
    A8 --> D8[Rankings CSV<br>(top-k, bottom-k)]

    %% Outputs
    subgraph Outputs
        D1
        D2
        D3
        D4
        D5
        D6
        D7
        D8
    end

