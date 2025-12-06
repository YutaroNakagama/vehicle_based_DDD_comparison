# Imbalance Comparison V2

不均衡データ対策手法の比較実験（改善版）

## 実験条件

| 項目 | V1（前回） | V2（今回） |
|------|-----------|-----------|
| Optuna試行回数 | 50 | **100** |
| CV分割数 | 3-fold | **5-fold** |
| 目的関数 | Precision@Recall≥70% | **F2スコア** |
| データ分割 | ランダム層化 | **時間ベース層化** |

## 比較手法

| # | 手法 | モデル | サンプリング | 特徴 |
|---|------|-------|-------------|------|
| 1 | Baseline | RF | なし | ベースライン |
| 2 | SMOTE | RF | SMOTE | オーバーサンプリングのみ |
| 3 | SMOTE+Tomek | RF | SMOTE → Tomek Links | 境界クリーニング |
| 4 | SMOTE+ENN | RF | SMOTE → ENN | ノイズ除去 |
| 5 | SMOTE+RUS | RF | SMOTE → RandomUnderSampler | ハイブリッド |
| 6 | BalancedRF | BalancedRF | 内部処理 | 各木でバランシング |
| 7 | EasyEnsemble | EasyEnsemble | 内部処理 | アンサンブル |

## 使用方法

### 全ジョブ一括投入
```bash
cd scripts/hpc/imbalance_comparison_v2
chmod +x launch_all.sh
./launch_all.sh
```

### 状況確認
```bash
qstat -u $USER
```

### 結果確認
```bash
# ジョブID確認
cat scripts/hpc/imbalance_comparison_v2/job_ids_v2.txt

# 評価結果確認
python3 << 'EOF'
import json
import glob

for f in sorted(glob.glob("results/evaluation/*/*/eval_results_*.json")):
    if "imbal_v2" in f:
        with open(f) as fp:
            d = json.load(fp)
        print(f"{f}:")
        print(f"  Precision: {d.get('precision', 0)*100:.2f}%")
        print(f"  Recall: {d.get('recall', 0)*100:.2f}%")
        print(f"  F1: {d.get('f1', 0)*100:.2f}%")
        print()
EOF
```

## 予想実行時間

| 手法 | 予想時間 |
|------|---------|
| Baseline | 4時間 |
| SMOTE+Tomek | 15時間 |
| SMOTE+ENN | 11時間 |
| BalancedRF | 1.5時間 |
| SMOTE+RUS | 3時間 |
| EasyEnsemble | 2.5時間 |

**並列実行時**: 約15-16時間（最長ジョブ基準）

## ファイル構成

```
imbalance_comparison_v2/
├── launch_all.sh              # 一括投入スクリプト
├── pbs_train_baseline.sh      # Baseline訓練
├── pbs_train_smote.sh         # SMOTE単体訓練
├── pbs_train_smote_tomek.sh   # SMOTE+Tomek訓練
├── pbs_train_smote_enn.sh     # SMOTE+ENN訓練
├── pbs_train_smote_rus.sh     # SMOTE+RUS訓練
├── pbs_train_balanced_rf.sh   # BalancedRF訓練
├── pbs_train_easy_ensemble.sh # EasyEnsemble訓練
├── pbs_evaluate.sh            # 評価スクリプト
├── job_ids_v2.txt             # ジョブID記録（自動生成）
└── README.md                  # このファイル
```

## 期待される改善

- **F2スコア最適化**: Recall重視でFN削減
- **5-fold CV**: より安定したパラメータ選択
- **時間ベース分割**: データリーク防止
- **新手法追加**: BalancedRF, SMOTE+RUS, EasyEnsembleによる性能向上
