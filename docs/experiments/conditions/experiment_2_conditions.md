# 実験2（ドメインシフト）の実験条件

このファイルは「実験2：ドメインシフト (split2)」で使用した実験条件の一覧を示します。

## 概要

- **目的**: split2 ドメイン分割による RF モデルのドメインシフト耐性評価
- **モデル**: RF（BalancedRF は不均衡対策手法として含む）
- **データ分割**: `split2`（`in_domain`: 44 名、`out_domain`: 43 名）
- **ランチャー**: `scripts/hpc/launchers/launch_paper_domain_split2.sh`
- **ジョブスクリプト**: `scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh`

## パラメータ一覧

| パラメータ | 値 |
|---|---|
| 距離指標 (DISTANCE) | `mmd`, `dtw`, `wasserstein`（3 種） |
| ドメイン (DOMAIN) | `in_domain`, `out_domain`（2 種） |
| 訓練モード (MODE) | `source_only`, `target_only`（2 種） |
| 乱数シード (SEED) | 42, 123（2 種） |
| ターゲット比率 (RATIO) | 0.1, 0.5（2 種、比率ベース手法のみ） |
| ランキング手法 (RANKING) | `knn` |
| Optuna 試行回数 | 100 |
| Optuna 目的関数 | F2 スコア |

## 訓練モードの意味

| MODE | 説明 | 訓練データ | 評価データ |
|---|---|---|---|
| `source_only` | Cross-domain | ターゲットの逆ドメイン | 指定ドメイン |
| `target_only` | Single-domain | 指定ドメイン内 | 指定ドメイン内 |

### Cross-Domain ロジック（source_only の場合）

| DOMAIN 指定 | 訓練データ | 評価データ |
|---|---|---|
| `out_domain` | in_domain（44 名） | out_domain（43 名） |
| `in_domain` | out_domain（43 名） | in_domain（44 名） |

## 不均衡対策手法

| 手法 (CONDITION) | 説明 | RATIO 使用 |
|---|---|---|
| `baseline` | 不均衡対策なし | ✗ |
| `smote_plain` | Plain SMOTE | ✓ |
| `smote` | Subject-wise SMOTE | ✓ |
| `undersample` | Random Under-Sampling | ✓ |
| `balanced_rf` | BalancedRandomForestClassifier | ✗ |

## ジョブ数の計算

1 ループ（DISTANCE × DOMAIN × MODE × SEED）あたりの条件数:

| 手法 | ジョブ数 |
|---|---|
| baseline（比率なし） | 1 |
| smote_plain × 2 比率 | 2 |
| smote × 2 比率 | 2 |
| undersample × 2 比率 | 2 |
| balanced_rf（比率なし） | 1 |
| **小計** | **8** |

ループ数: 3 (DISTANCE) × 2 (DOMAIN) × 2 (MODE) × 2 (SEED) = **24**

**合計: 24 × 8 = 192 ジョブ**

## HPC リソース設定

| 手法 | CPU | メモリ | 制限時間 | キュー |
|---|---|---|---|---|
| balanced_rf | 8 | 12 GB | 08:00:00 | LONG |
| smote / smote_plain | 4 | 10 GB | 08:00:00 | SINGLE |
| baseline / undersample | 4 | 8 GB | 06:00:00 | SINGLE |

## ジョブ例（PBS 環境）

```bash
# Baseline
qsub -N bs_mo_s_s42 -l select=1:ncpus=4:mem=8gb -l walltime=06:00:00 -q SINGLE \
    -v CONDITION=baseline,MODE=source_only,DISTANCE=mmd,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true \
    scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh

# SMOTE with ratio
qsub -N sm_mo_s_r0.1_s42 -l select=1:ncpus=4:mem=10gb -l walltime=08:00:00 -q SINGLE \
    -v CONDITION=smote,MODE=source_only,DISTANCE=mmd,DOMAIN=out_domain,RATIO=0.1,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true \
    scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh
```

## 出力先

- ランキング入力: `results/analysis/domain/distance/subject-wise/ranks/split2/{ranking_method}/`
- 評価結果: `results/outputs/evaluation/BalancedRF/`
- 訓練ログ: `results/outputs/training/BalancedRF/`
- モデル: `models/BalancedRF/{JOB_ID}/`

## 備考

- 実行前に `results/analysis/domain/` にランキングファイル（`*_in_domain.txt` / `*_out_domain.txt`）が生成されていることを確認してください
- 実行用のランチャーやリソース設定は `scripts/hpc/launchers/` に依存します

## 関連ドキュメント

- [ドメイン汎化パイプライン](../../architecture/domain_generalization.md)
- [実験結果](../results/domain_results.md)
- [再現手順](../reproducibility.md#experiment-2-domain-shift-rf-split2)

---

作成日: 2026-02-07
