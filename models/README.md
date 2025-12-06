# models/

学習済みモデル、スケーラー、特徴量選択結果を保存するディレクトリ。

## Directory Structure

```
models/
├── RF/                    # Random Forest
├── BalancedRF/            # Balanced Random Forest
├── EasyEnsemble/          # EasyEnsemble Classifier
├── Lstm/                  # LSTM (deep learning)
├── SvmA/                  # SVM (all features)
├── SvmW/                  # SVM (wavelet features)
└── RF_backup_*/           # バックアップ（特定実験の保存）
```

## Job-based Organization

各モデルディレクトリはPBS Job IDで整理:

```
RF/
├── <jobID>/               # PBS job ID
│   └── <jobID>[<idx>]/    # Array job index
│       ├── RF_<tag>.pkl           # 学習済みモデル
│       ├── scaler_<tag>.pkl       # StandardScaler
│       ├── selected_features_<tag>.pkl  # 特徴量選択結果
│       ├── feature_meta_<tag>.json      # 特徴量メタ情報
│       └── threshold_<tag>.json         # 分類閾値
├── latest_job.txt         # 最新ジョブID
└── imbalance_train_job.txt # 進行中の不均衡実験ジョブ
```

## File Naming Convention

`<model>_<tag>.pkl` / `.json`

Tag format: `<mode>_<experiment>_<details>_<jobID>_<arrayIdx>_<seed>`

例: `RF_source_only_imbalance_knn_mmd_out_domain_smote_14572594_1_1.pkl`

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

- モデルファイル (*.pkl, *.h5) は `.gitignore` により Git 管理対象外
- `latest_job.txt` には最新の完了ジョブIDを記録
- バックアップディレクトリは重要な実験結果を保護
