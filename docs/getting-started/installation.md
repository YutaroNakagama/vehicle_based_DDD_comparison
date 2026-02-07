# Installation Guide

This guide covers setting up the Vehicle-Based Driver Drowsiness Detection project.

## Prerequisites

- **Python**: 3.10+ (3.13 recommended)
- **Git**: For version control
- **CUDA** (optional): For GPU-accelerated training with TensorFlow

## Quick Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YutaroNakagama/vehicle_based_DDD_comparison.git
cd vehicle_based_DDD_comparison
```

### 2. Create Virtual Environment

Using conda (recommended):

```bash
conda create -n ddd python=3.13
conda activate ddd
```

Or using venv:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Package Dependencies

The main dependencies include:

| Category | Packages |
|----------|----------|
| **ML Core** | scikit-learn, numpy, pandas, scipy |
| **Boosting** | lightgbm, xgboost, catboost |
| **Imbalanced Learning** | imbalanced-learn |
| **Hyperparameter Tuning** | optuna |
| **Deep Learning** | tensorflow (optional) |
| **Visualization** | matplotlib, seaborn |
| **Testing** | pytest, pytest-cov |

## Dataset Setup

### Download the Dataset

The project uses the open dataset from *Multi-modal Data Acquisition Platform for Behavioral Evaluation* (Aygun et al., 2024).

**DOI:** [10.7910/DVN/HMZ5RG](https://doi.org/10.7910/DVN/HMZ5RG)

```bash
# Download via dataverse API
curl -L -O -J "https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/HMZ5RG"

# Extract to data/ directory
unzip dataverse_files.zip -d data/raw/
```

### Directory Structure After Setup

```
data/
├── raw/           # Original dataset files (.mat)
├── interim/       # Intermediate processed data
│   ├── eeg/
│   ├── merged/
│   └── smooth_std_pe/
└── processed/     # Final feature matrices
    └── common/
```

## Verify Installation

Run the smoke tests to verify the installation:

```bash
python -m pytest tests/ -m "smoke" -v
```

## IDE Setup (VS Code)

Recommended extensions:

- Python (ms-python.python)
- Pylance (ms-python.vscode-pylance)
- Jupyter (ms-toolsai.jupyter)

Add to `.vscode/settings.json`:

```json
{
    "python.analysis.extraPaths": ["${workspaceFolder}/src"],
    "python.testing.pytestArgs": ["tests"],
    "python.testing.pytestEnabled": true
}
```

## Troubleshooting

### Import Errors

If you encounter `ModuleNotFoundError`, ensure the `src` directory is in your Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

Or add to your script:

```python
import sys
sys.path.insert(0, '/path/to/vehicle_based_DDD_comparison')
```

### TensorFlow GPU Issues

For GPU support, install CUDA-compatible TensorFlow:

```bash
pip install tensorflow[and-cuda]
```

### Memory Issues

For large datasets, consider:

1. Reduce batch sizes in training
2. Use data generators instead of loading all data
3. Enable garbage collection between experiments

## Next Steps

- [Quickstart Guide](quickstart.md) — Run your first experiment
- [Configuration Reference](../reference/configuration.md) — Customize parameters
- [Developer Guide](../architecture/developer_guide.md) — Understand the codebase
