# data/

Raw and processed datasets (ignored by git).

## Directory Structure

```
data/
├── interim/               # Intermediate data (during preprocessing pipeline)
│   ├── eeg/              # EEG signal data
│   │   ├── common/       # Common preprocessed
│   │   ├── Lstm/         # LSTM-specific processing
│   │   ├── SvmA/         # SVM (all features)
│   │   └── SvmW/         # SVM (wavelet features)
│   ├── merged/           # EEG + vehicle data merged
│   │   ├── common/
│   │   ├── Lstm/
│   │   ├── SvmA/
│   │   └── SvmW/
│   ├── smooth_std_pe/    # Smoothing, standardization, PE processing
│   ├── time_freq_domain/ # Time-frequency domain features
│   └── wavelet/          # Wavelet transform coefficients
└── processed/            # Final processed data (model input ready)
    ├── common/           # Common feature set
    ├── Lstm/             # LSTM sequence data
    ├── SvmA/             # SVM features (all features)
    └── SvmW/             # SVM features (wavelet)
```

## File Naming Convention

- `processed_S0XXX_Y.csv` - Subject ID `S0XXX`, Session `Y` (1 or 2)
- Example: `processed_S0113_1.csv` → Subject S0113, Session 1

## Data Flow

1. **Raw** → `interim/eeg/` (EEG preprocessing)
2. `interim/eeg/` + vehicle data → `interim/merged/`
3. `interim/merged/` → feature extraction → `interim/smooth_std_pe/`, `time_freq_domain/`, `wavelet/`
4. Feature integration → `processed/` (split by model type)

## Notes

- All data files are excluded from Git via `.gitignore`
- Actual data is managed on JAIST KAGAYAKI cluster
