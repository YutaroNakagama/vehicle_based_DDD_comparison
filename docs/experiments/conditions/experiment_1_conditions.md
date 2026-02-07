# 実験1（クラス不均衡）の実験条件

このファイルは「実験1：クラス不均衡対策の比較」で使用した実験条件の一覧を示します。

## 概要

- **目的**: プール訓練（ドメイン分割なし）における不均衡対策手法の性能比較
- **モデル**: RF（RandomForestClassifier）
- **ランチャー**: `scripts/hpc/launchers/launch_paper_imbalance.sh`
- **ジョブスクリプト**: `scripts/hpc/jobs/imbalance/pbs_imbalance_comparison.sh`

## パラメータ一覧

| パラメータ | 値 |
|---|---|
| 分類モデル | RF |
| Optuna 試行回数 | 100 |
| Optuna 目的関数 | F2 スコア |
| 乱数シード (SEED) | 42, 123（2 種） |
| ターゲット比率 (RATIO) | 0.1, 0.5（2 種、比率ベース手法のみ） |

## 不均衡対策手法

| 手法 (METHOD) | 説明 | RATIO 使用 |
|---|---|---|
| `baseline` | class_weight のみ（オーバーサンプリングなし） | ✗ |
| `smote` | Plain SMOTE | ✓ |
| `smote_subjectwise` | Subject-wise SMOTE | ✓ |
| `undersample_rus` | Random Under-Sampling (RUS) | ✓ |
| `balanced_rf` | BalancedRandomForestClassifier | ✗ |

## ジョブ数の計算

| 手法 | シード | 比率 | ジョブ数 |
|---|---|---|---|
| baseline | 2 | — | 2 |
| smote | 2 | 2 | 4 |
| smote_subjectwise | 2 | 2 | 4 |
| undersample_rus | 2 | 2 | 4 |
| balanced_rf | 2 | — | 2 |
| **合計** | | | **16** |

## HPC リソース設定

| 手法 | CPU | メモリ | 制限時間 | キュー |
|---|---|---|---|---|
| balanced_rf | 8 | 8 GB | 08:00:00 | LONG |
| smote / smote_subjectwise | 4 | 8 GB | 08:00:00 | SINGLE |
| baseline / undersample_rus | 4 | 8 GB | 04:00:00 | SINGLE |

## ジョブ例（PBS 環境）

```bash
# Baseline（比率なし）
qsub -N baseli_s42 -l select=1:ncpus=4:mem=8gb -l walltime=04:00:00 -q SINGLE \
    -v METHOD=baseline,SEED=42,N_TRIALS=100 \
    scripts/hpc/jobs/imbalance/pbs_imbalance_comparison.sh

# SMOTE（比率あり）
qsub -N smote_r0.1_s42 -l select=1:ncpus=4:mem=8gb -l walltime=08:00:00 -q SINGLE \
    -v METHOD=smote,RATIO=0.1,SEED=42,N_TRIALS=100 \
    scripts/hpc/jobs/imbalance/pbs_imbalance_comparison.sh
```

## 出力先

- 評価結果: `results/outputs/evaluation/RF/`
- 訓練ログ: `results/outputs/training/RF/`
- モデル: `models/RF/{JOB_ID}/`

## 関連ドキュメント

- [不均衡手法リファレンス](../../reference/imbalance_methods.md)
- [実験結果](../results/imbalance_results.md)
- [再現手順](../reproducibility.md#experiment-1-imbalance-analysis)

---

作成日: 2026-02-07
