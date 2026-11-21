# Local Experiment Scripts

This directory contains shell scripts for running machine learning experiments on a local workstation.

## Scripts

| Script | Description |
|--------|-------------|
| `run_preprocess.sh` | Feature extraction pipeline (87 subjects in parallel) |
| `run_domain_parallel.sh` | Domain analysis experiments (18-36 parallel jobs) |

---

## 0. Preprocessing (`run_preprocess.sh`)

Extracts features from raw data for all subjects in parallel.

### Usage

```bash
# Default: common model (all features), background
./scripts/local/run_preprocess.sh

# Specific model
./scripts/local/run_preprocess.sh --model SvmA

# Custom parallel jobs
./scripts/local/run_preprocess.sh --jobs 10

# With jittering augmentation
./scripts/local/run_preprocess.sh --jittering

# Foreground execution
./scripts/local/run_preprocess.sh --fg
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model NAME` | Model type (common, SvmA, SvmW, Lstm) | common |
| `--jobs N` | Number of parallel jobs | auto (nproc - 2) |
| `--jittering` | Enable jittering augmentation | disabled |
| `--fg` | Run in foreground | background |

### Output

**Logs:**
```
scripts/local/logs/preprocess/
└── preprocess_YYYYMMDD_HHMMSS.log
```

**Data:**
```
data/processed/common/  # or data/processed/<model>/
└── *.parquet           # Feature files per subject
```

---

## 1. Domain Analysis (`run_domain_parallel.sh`)

Runs domain analysis experiments with different distance metrics and domain selections.

### Experiments

- **Modes**: `source_only`, `target_only` (2)
- **Distances**: `mmd`, `wasserstein`, `dtw` (3)
- **Domains**: `in_domain`, `mid_domain`, `out_domain` (3)
- **Total**: 18 experiments per condition

### Usage

```bash
# Default: SMOTE only (18 experiments, background)
./scripts/local/run_domain_parallel.sh

# Baseline only (no oversampling)
./scripts/local/run_domain_parallel.sh --baseline

# Both SMOTE and Baseline (36 experiments)
./scripts/local/run_domain_parallel.sh --both

# Custom trial count
./scripts/local/run_domain_parallel.sh --trials 50

# Foreground execution
./scripts/local/run_domain_parallel.sh --fg
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--smote` | Run with Subject-wise SMOTE | ✓ (default) |
| `--baseline` | Run without oversampling | |
| `--both` | Run both conditions | |
| `--trials N` | Number of Optuna trials | 10 |
| `--eval` | Run evaluation after training | |
| `--fg` | Run in foreground | background |

### Output

**Logs:**
```
scripts/local/logs/domain/
├── domain_parallel_YYYYMMDD_HHMMSS.log  # Main execution log
├── imbalv3_knn_*.log                     # SMOTE experiment logs
├── imbalv3_knn_*_eval.log                # SMOTE evaluation logs
├── baseline_domain_knn_*.log             # Baseline experiment logs
└── baseline_domain_knn_*_eval.log        # Baseline evaluation logs
```

**Results:**
- Models saved to `models/` directory
- Metrics saved to `results/` directory

### Monitoring

```bash
# Check running processes
watch "ps aux | grep train.py | grep -v grep | wc -l"

# Follow main log
tail -f scripts/local/logs/domain/domain_parallel_*.log

# Check progress
watch "grep -E 'Trial|best=' scripts/local/logs/domain/*.log | tail -20"
```

### Stop Execution

```bash
pkill -f 'train.py.*(imbalv3|baseline_domain)'
```

---

## Common Configuration

### Default Settings

| Parameter | Value |
|-----------|-------|
| Model | RandomForest (RF) |
| Optuna Trials | 10 |
| Random Seed | 42 |
| SMOTE Ratio | 0.5 |
| CV Folds | 3 |

### Environment

- Python virtual environment: `.venv-linux`
- Automatically activated by scripts
- Uses all available CPU cores for parallel execution

### Logs Directory Structure

```
scripts/local/logs/
├── preprocess/  # Preprocessing logs
├── domain/      # Domain analysis logs
├── smote/       # (legacy)
└── training/    # (legacy)
```

---

## Quick Reference

```bash
# Preprocessing (feature extraction for 87 subjects)
./scripts/local/run_preprocess.sh

# Domain analysis with SMOTE (18 experiments)
./scripts/local/run_domain_parallel.sh

# Domain analysis with evaluation
./scripts/local/run_domain_parallel.sh --eval

# Domain analysis: Baseline vs SMOTE (36 experiments)
./scripts/local/run_domain_parallel.sh --both

# Domain analysis: Baseline vs SMOTE with evaluation
./scripts/local/run_domain_parallel.sh --both --eval

# Check all running experiments
ps aux | grep train.py | grep -v grep

# Stop all experiments
pkill -f train.py
```
