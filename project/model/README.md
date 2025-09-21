# model/ directory

This directory stores **trained models, scalers, and feature selection metadata** for different experiment setups.

---

## Structure
```

model/
├── common/        # Baseline trained models
├── common\_k10/    # 10-fold cross-validation results
├── common\_k20/    # 20-fold cross-validation results
├── common\_k40/    # 40-fold cross-validation results
└── ...            # (future experiments may add more)

```

---

## Contents
Each subdirectory typically contains:
- `model.pkl` → Trained model object (pickle)
- `scaler.pkl` → Feature scaler used during training
- `selected_features_train.pkl` → Feature indices selected during training
- `feature_meta.json` → Metadata about features (names, types, transformations)
- Additional artifacts (Optuna logs, threshold values, etc.)

---

## Notes
- The directory naming convention `common_kX` refers to **k-fold cross-validation** results.  
  - Example: `common_k10/` = models trained under 10-fold CV.  
- For fine-tuning experiments, results are typically linked to **target subject groups** defined in `misc/target_groups.txt`.  

---

## Tips
- Store model training configs alongside artifacts for reproducibility.  
- Consider archiving large models separately (e.g., Git LFS or external storage).  
