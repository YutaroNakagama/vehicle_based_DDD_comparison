# Prior Research Replication

This document describes the **prior research replication experiments** implemented in this repository. We replicate three representative approaches from the literature to establish baselines for comparison with our proposed methods.

## 1. Overview

Three prior research methods are implemented:

| Model | Architecture | Reference Focus | Input Features |
|-------|-------------|-----------------|----------------|
| **SvmA** | ANFIS + PSO-optimized SVM | Feature weighting with fuzzy inference | Time-frequency + EEG |
| **SvmW** | Standard SVM with RBF kernel | Wavelet-based feature extraction | Wavelet + EEG |
| **Lstm** | Bidirectional LSTM + Attention | Sequence-based deep learning | Smooth STD PE + EEG |

## 2. Model Architectures

### 2.1 SvmA (ANFIS-based Feature Weighting)

**Location:** `src/models/architectures/SvmA.py`

This model combines:
- **ANFIS-style feature importance**: Weights features using Fisher Index, Correlation, T-test, and Mutual Information
- **PSO optimization**: Particle Swarm Optimization for hyperparameter tuning
- **SVM classifier**: RBF kernel with optimized C and gamma

**Key Components:**

```
Feature Indices → ANFIS Weighting → Feature Selection → PSO-optimized SVM
     │                │                    │                    │
  Fisher          Weighted Sum        Threshold            C, gamma
  Correlation     (importance > 0.75)  Selection          optimization
  T-test
  Mutual Info
```

**Importance Degree Calculation:**
- Score > 0.75 → High importance (1.0)
- Score > 0.40 → Medium importance (0.5)
- Score ≤ 0.40 → Low importance (0.0)

Only high-importance features are selected for final SVM training.

### 2.2 SvmW (Wavelet-based SVM)

**Location:** Standard SVM via `src/models/training/model_factory.py`

This model uses:
- **Wavelet decomposition features**: Extracted from steering wheel and lane offset signals
- **RBF kernel SVM**: Standard implementation with Optuna hyperparameter tuning

**Feature Pipeline:**
```
Raw Signals → Wavelet Transform → Statistical Features → SVM (RBF)
      │              │                    │                 │
  Steering       DWT levels          Mean, Std,        Optuna-tuned
  Lane Offset    Coefficients        Energy, etc.      C, gamma
```

### 2.3 Lstm (Bidirectional LSTM with Attention)

**Location:** `src/models/architectures/lstm.py`

This model implements:
- **Bidirectional LSTM**: Captures temporal patterns in both directions
- **Custom Attention Layer**: Focuses on important time steps
- **Dense layers**: For final classification

**Architecture:**
```
Input (timesteps, features)
        ↓
Bidirectional LSTM (50 units)
        ↓
Attention Layer (learned weights)
        ↓
Dense (20 units, ReLU)
        ↓
Dense (1 unit, Sigmoid)
        ↓
Binary Output
```

**Training Configuration:**
- Optimizer: Adam
- Loss: Binary Cross-Entropy
- K-Fold Cross-Validation (default: 5 folds)
- Early Stopping with patience

## 3. Data Preprocessing

Each model requires specific preprocessing:

### 3.1 Feature Sets by Model

| Model | Feature Source | Window Size | Step Size |
|-------|---------------|-------------|-----------|
| SvmA | time_freq_domain + EEG | 3 sec | 1.5 sec |
| SvmW | wavelet + EEG | 3 sec | 1.5 sec |
| Lstm | smooth_std_pe + EEG | 5 sec | 2.5 sec |

### 3.2 Preprocessing Pipeline

```bash
# Preprocess for specific model
python scripts/python/preprocess/preprocess.py \
    --model_name SvmA \
    --subject_id 1
```

## 4. Running Experiments

### 4.1 Local Execution

```bash
# Train SvmA with seed 42
python scripts/python/train/train.py \
    --model SvmA \
    --mode pooled \
    --subject_wise_split \
    --seed 42 \
    --time_stratify_labels \
    --tag "prior_research_s42"

# Train SvmW with seed 42
python scripts/python/train/train.py \
    --model SvmW \
    --mode pooled \
    --subject_wise_split \
    --seed 42 \
    --time_stratify_labels \
    --tag "prior_research_s42"

# Train Lstm with seed 42
python scripts/python/train/train.py \
    --model Lstm \
    --mode pooled \
    --subject_wise_split \
    --seed 42 \
    --time_stratify_labels \
    --tag "prior_research_s42"
```

### 4.2 HPC Execution (PBS)

```bash
# Single submission example (CPU — SvmW/SvmA)
qsub -N "prior_SvmW_test" \
     -l select=1:ncpus=8:mem=16gb -l walltime=12:00:00 -q SINGLE \
     -v MODEL=SvmW,CONDITION=baseline,DISTANCE=mmd,DOMAIN=out_domain,SEED=42 \
     scripts/hpc/jobs/train/pbs_prior_research_unified.sh

# Single submission example (GPU — Lstm)
qsub -N "prior_Lstm_test" \
     -l select=1:ncpus=4:mem=16gb:ngpus=1 -l walltime=16:00:00 -q GPU-1 \
     -v MODEL=Lstm,CONDITION=baseline,DISTANCE=mmd,DOMAIN=out_domain,SEED=42 \
     scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh
```

**Bulk execution (automatic submission via unified daemon):**

exp3 has 84 jobs per model (252 total), and due to queue limits,
a unified daemon process automatically submits remaining jobs.

```bash
# Start unified daemon (supports all models, automatic GPU queue routing)
nohup bash scripts/hpc/launchers/auto_resub_unified_v2.sh &
```

The daemon picks unsubmitted jobs from the remaining job list and runs `qsub` based on queue availability.
Lstm jobs are automatically routed to GPU queues (GPU-1, GPU-1A, GPU-S, GPU-L, GPU-LA).
See [reproducibility.md](../experiments/reproducibility.md#automatic-submission-via-daemon) for details.

### 4.3 Domain Split Experiments (domain_train)

In domain_train mode, each domain's data is split 70/15/15 (train/val/test),
and two evaluations are performed for each training run (using same-domain train/val):
- **within**: Evaluated on same-domain test (15%)
- **cross**: Evaluated on opposite-domain test (15%)

```bash
# Submit all 252 jobs at once
bash scripts/hpc/launchers/launch_prior_research_unified.sh

# Dry run to preview jobs
bash scripts/hpc/launchers/launch_prior_research_unified.sh --dry-run
```

**PBS Job Scripts:**
- CPU (SvmW/SvmA): `scripts/hpc/jobs/train/pbs_prior_research_unified.sh`
- GPU (Lstm): `scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh`

## 5. Output Artifacts

### 5.1 Model Files

**SvmW / SvmA:**
```
models/{MODEL}/{JOB_ID}/{JOB_ID}[1]/
├── {MODEL}_pooled_*.pkl                    # Trained model
├── scaler_{MODEL}_pooled_*.pkl             # Feature scaler
├── selected_features_{MODEL}_pooled_*.pkl  # Selected features info
├── feature_meta_{MODEL}_pooled_*.json      # Feature metadata
└── threshold_{MODEL}_pooled_*.json         # Classification threshold
```

**Lstm:**
```
models/Lstm/{JOB_ID}/{JOB_ID}[1]/
├── Lstm_pooled_*.keras                      # Final model (Keras format)
├── Lstm_pooled_*_fold_{0-4}.keras            # Per-fold models (K=5)
├── training_history_Lstm_pooled_*.json       # Training history (loss, accuracy)
├── scaler_Lstm_pooled_*.pkl                  # Feature scaler
├── selected_features_Lstm_pooled_*.pkl       # Selected features info
├── feature_meta_Lstm_pooled_*.json           # Feature metadata
└── threshold_Lstm_pooled_*.json              # Classification threshold
```

### 5.2 Result Files

```
results/outputs/training/{MODEL}/{JOB_ID}/{JOB_ID}[1]/
├── train_results_{MODEL}_pooled_*.json  # Metrics (train/val/test)
└── train_results_{MODEL}_pooled_*.csv   # Detailed results
```

### 5.3 Optuna Study (SvmW only)

SvmW uses Optuna for hyperparameter tuning, so the study object is saved.
SvmA uses PSO and Lstm uses K-Fold CV, so no Optuna study is saved for them.

```
models/SvmW/{JOB_ID}/
├── optuna_SvmW_pooled_*_study.pkl         # Optuna study object
├── optuna_SvmW_pooled_*_trials.csv        # Trial history
└── optuna_SvmW_pooled_*_convergence.json  # Convergence data
```

## 6. Key Differences from Proposed Methods

| Aspect | Prior Research | Proposed Methods |
|--------|---------------|------------------|
| Imbalance Handling | None (class imbalance ignored) | SMOTE variants, RUS, BalancedRF |
| Feature Selection | ANFIS (SvmA), Manual (SvmW/Lstm) | Optuna-tuned, domain ranking |
| Cross-Validation | K-Fold (Lstm), None (SvmA/SvmW) | Subject-wise split (no data leakage) |
| Hyperparameter Tuning | PSO (SvmA), Grid (SvmW) | Optuna Bayesian optimization |

## 7. Known Issues and Limitations

### 7.1 Generalization Gap

All models show significant validation-test performance gap due to:
- Subject-wise split: Models trained on some subjects, tested on completely different subjects
- Individual variability: Drowsiness patterns vary significantly between drivers
- Class imbalance: Majority of data is "awake" class

### 7.2 Reproducibility Notes

- **Random seeds**: Set via `--seed` parameter
- **Thread limits**: Controlled via environment variables in PBS script
- **TensorFlow**: Lstm runs in GPU mode (A40/A100). `configure_gpu()` configures GPU memory growth

## 8. References

The implementations are based on representative approaches from the driver drowsiness detection literature:

- **ANFIS + SVM (SvmA)**: Combines fuzzy inference for feature weighting with SVM classification
- **Wavelet + SVM (SvmW)**: Uses wavelet decomposition for signal processing with SVM
- **Bidirectional LSTM (Lstm)**: Deep learning approach with attention mechanism for sequential data

For specific paper citations, see the original publications in the project bibliography.

---

## Related Documents

- [Developer Guide](developer_guide.md) — Overall repository architecture
- [Domain Generalization Pipeline](domain_generalization.md) — Domain analysis workflow
- [Imbalance Methods](../reference/imbalance_methods.md) — Imbalance handling strategies
