# results/

This directory stores all experiment results.  
The structure follows a consistent policy to ensure reproducibility, clarity, and reusability.

## Directory Structure

```
results/
├── domain_analysis/       # ドメイン分析結果
│   ├── distance/          # 距離行列 (group-wise/, subject-wise/)
│   ├── rankings/          # ランキング結果 (centroid_umap/, lof/, mean_distance/)
│   ├── summary/           # 要約テーブル・可視化 (csv/, png/)
│   └── knn_imbalance/     # [archive] 旧KNN不均衡実験
├── evaluation/            # モデル評価結果
│   ├── RF/                # Random Forest評価
│   ├── BalancedRF/        # Balanced Random Forest評価
│   ├── EasyEnsemble/      # EasyEnsemble評価
│   └── ensemble/          # アンサンブル評価
└── imbalance_analysis/    # 不均衡データ分析
    ├── *.png              # 可視化結果
    ├── *.csv              # 数値結果
    └── v2/                # v2実験結果
```

## Naming Conventions

### Per-job Results (evaluation/, domain_analysis/)

- **Single-job summary:**  
  `summary_<target>_<jobID>.csv / .png`

- **Multi-job comparison:**  
  `compare_<target>_<analysisType>_<date>.csv / .png`

- **Global summary:**  
  `summary_all_<target>_<date>.csv / .png`

### Files in Each Subdirectory

- `metrics/` → Per-job raw performance metrics (CSV/JSON)
- `predictions/` → Per-job prediction outputs (ROC, PR, CM; CSV + PNG)
- `distances/` → Per-job distance matrices (DTW, MMD, Wasserstein)
- `ranks/` → Per-job ranking results (mean, std, top10, top20)

## Policy

- **CSV** for numerical data, **PNG** for visualizations
- Job results are never overwritten: each run is placed in its own job ID folder
- No PDF/SVG — PNG is the single standard format
