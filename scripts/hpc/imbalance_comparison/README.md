# Imbalance Comparison Experiments

このディレクトリには、不均衡データ対処法の比較実験用スクリプトが含まれています。

## 概要

Driver Drowsiness Detection (DDD) データセットは約4%の陽性率（眠気状態）という極端な不均衡を持っています。
このディレクトリのスクリプトは、異なるオーバーサンプリング手法の効果を比較検証します。

## 比較手法

| 手法 | 説明 | 期待される効果 |
|------|------|----------------|
| **Baseline** | オーバーサンプリングなし | ベースライン（比較基準） |
| **SMOTE + Tomek Links** | SMOTEで増やした後、境界ペアを除去 | Precisionの改善 |
| **SMOTE + ENN** | SMOTEで増やした後、ノイズを積極的に除去 | よりクリーンな決定境界 |

## ファイル構成

```
imbalance_comparison/
├── pbs_train_baseline.sh    # Baseline訓練ジョブ
├── pbs_train_smote_tomek.sh # SMOTE+Tomek訓練ジョブ
├── pbs_train_smote_enn.sh   # SMOTE+ENN訓練ジョブ
├── pbs_evaluate.sh          # 評価ジョブ
├── launch_all.sh            # 全ジョブ一括投入スクリプト
└── README.md                # このファイル
```

## 使用方法

### 全ジョブ一括投入

```bash
cd /home/s2240011/git/ddd/vehicle_based_DDD_comparison
bash scripts/hpc/imbalance_comparison/launch_all.sh
```

### 個別ジョブ投入

```bash
# Baselineのみ
qsub -v MODEL=RF,SEED=42 scripts/hpc/imbalance_comparison/pbs_train_baseline.sh

# SMOTE + Tomekのみ
qsub -v MODEL=RF,SEED=42 scripts/hpc/imbalance_comparison/pbs_train_smote_tomek.sh

# SMOTE + ENNのみ
qsub -v MODEL=RF,SEED=42 scripts/hpc/imbalance_comparison/pbs_train_smote_enn.sh
```

## 評価指標

- **Precision**: 陽性予測の正確性
- **Recall**: 実際の陽性の検出率
- **F1 Score**: PrecisionとRecallの調和平均
- **F2 Score**: Recallを重視したF-measure（眠気検知では重要）
- **AUC-ROC**: 全体的な識別能力
- **AUC-PR**: 不均衡データでより信頼性の高い指標

## 結果の確認

訓練・評価完了後、以下で結果を確認できます：

```bash
# モデル成果物
ls models/RF/

# 評価結果
ls results/evaluation/RF/
```

## 注意事項

- 各ジョブは約8〜12時間かかる可能性があります
- 評価ジョブは訓練完了後に自動的に開始されます（依存関係設定済み）
- メール通知が設定されています（開始・終了・エラー時）
