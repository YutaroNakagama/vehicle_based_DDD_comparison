# results/

This directory stores all experiment results.  
The structure follows a consistent policy to ensure reproducibility, clarity, and reusability.

## Directory Structure

```
results/
├── domain_analysis/           # ドメイン分析結果
│   ├── distance/              # 距離行列・可視化
│   │   ├── subject-wise/      # 被験者間距離
│   │   │   ├── mmd/           # MMD距離
│   │   │   └── wasserstein/   # Wasserstein距離
│   │   └── group-wise/        # グループ間距離
│   ├── rankings/              # ランキング結果
│   │   ├── centroid_umap/
│   │   ├── lof/
│   │   └── mean_distance/
│   ├── summary/               # 要約テーブル・可視化
│   │   ├── csv/
│   │   └── png/
│   └── archive/               # 古い実験結果
│       └── knn_imbalance/
├── evaluation/                # モデル評価結果（Job IDベース）
│   ├── RF/                    # Random Forest
│   ├── BalancedRF/            # Balanced Random Forest
│   ├── EasyEnsemble/          # EasyEnsemble
│   └── ensemble/              # アンサンブル評価
└── imbalance_analysis/        # 不均衡データ分析
    ├── v1/                    # 初期分析結果
    └── v2/                    # 改良版分析結果
```

## Naming Conventions

### Per-job Results (evaluation/)

各Job IDディレクトリに評価結果を保存:
- `<jobID>/<jobID>[<idx>]/` - Array job index別

### Summary Files

- **Single-job summary:** `summary_<target>_<jobID>.csv`
- **Multi-job comparison:** `compare_<target>_<analysisType>_<date>.csv`
- **Global summary:** `summary_all_<target>_<date>.csv`

## Policy

- **CSV** for numerical data, **PNG** for visualizations
- Job results are never overwritten: each run is placed in its own job ID folder
- No PDF/SVG — PNG is the single standard format
- Old experiments are moved to `archive/` subdirectory
