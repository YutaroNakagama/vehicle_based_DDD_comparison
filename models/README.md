# models/

Directory for storing trained models, scalers, and feature selection results.

## Directory Structure

```
models/
├── RF/                    # Random Forest
├── BalancedRF/            # Balanced Random Forest
├── EasyEnsemble/          # EasyEnsemble Classifier
├── Lstm/                  # LSTM (deep learning)
├── SvmA/                  # SVM (all features)
└── SvmW/                  # SVM (wavelet features)
```

## Job-based Organization

Each model directory is organized by PBS Job ID:

```
RF/
├── <jobID>/               # PBS job ID
│   └── <jobID>[<idx>]/    # Array job index (optional)
│       ├── RF_<tag>.pkl           # Trained model
│       ├── scaler_<tag>.pkl       # StandardScaler
│       ├── selected_features_<tag>.pkl  # Feature selection result
│       ├── feature_meta_<tag>.json      # Feature metadata
│       └── threshold_<tag>.json         # Classification threshold
└── latest_job.txt         # Latest completed job ID
```

## File Naming Convention

`<model>_<tag>.pkl` / `.json`

Tag format: `<mode>_<experiment>_<details>_<jobID>_<arrayIdx>_<seed>`

Example: `RF_source_only_imbalance_knn_mmd_out_domain_smote_14572594_1_1.pkl`

## Model Types

| Model | Description |
|-------|-------------|
| RF | Random Forest (sklearn) |
| BalancedRF | Balanced Random Forest (imbalanced-learn) |
| EasyEnsemble | EasyEnsemble Classifier (imbalanced-learn) |
| Lstm | LSTM neural network (TensorFlow/Keras) |
| SvmA | SVM with all features |
| SvmW | SVM with wavelet features |

## Notes

- Model files (*.pkl, *.h5) are excluded from Git via `.gitignore`
- `latest_job.txt` records the most recent completed job ID
