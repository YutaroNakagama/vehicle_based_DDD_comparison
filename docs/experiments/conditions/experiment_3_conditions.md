# 実験3（先行研究再現）の実験条件

このファイルは「実験3：先行研究モデルの再現（split2 ドメイン分割版）」で使用した実験条件の一覧を示します。

## 概要

- **目的**: 先行研究のモデル（SvmA, SvmW, Lstm）を split2 ドメイン分割で再現し、ドメインシフト耐性を評価
- **モデル**: SvmW, SvmA, Lstm（3 種）
- **データ分割**: `split2`（`in_domain`: 44 名、`out_domain`: 43 名）
- **ランチャー**: `scripts/hpc/launchers/launch_prior_research_split2.sh`
- **ジョブスクリプト**: `scripts/hpc/jobs/train/pbs_prior_research_split2.sh`

## パラメータ一覧

| パラメータ | 値 |
|---|---|
| 距離指標 (DISTANCE) | `mmd`, `dtw`, `wasserstein`（3 種） |
| ドメイン (DOMAIN) | `in_domain`, `out_domain`（2 種） |
| 訓練モード (MODE) | `source_only`, `target_only`（2 種） |
| 乱数シード (SEED) | 42, 123（2 種） |
| ターゲット比率 (RATIO) | 0.1, 0.5（2 種、比率ベース手法のみ） |
| ランキング手法 (RANKING) | `knn` |
| Optuna 試行回数 | 100（SvmW のみ） |

## 訓練モードの意味

| MODE | 説明 | 訓練データ | 評価データ |
|---|---|---|---|
| `source_only` | Cross-domain | ターゲットの逆ドメイン | 指定ドメイン |
| `target_only` | Within-domain | 指定ドメイン内 | 指定ドメイン内 |

### Cross-Domain ロジック（source_only の場合）

| DOMAIN 指定 | 訓練データ | 評価データ |
|---|---|---|
| `out_domain` | in_domain（44 名） | out_domain（43 名） |
| `in_domain` | out_domain（43 名） | in_domain（44 名） |

## モデル別の不均衡対策手法

各モデルで適用可能な手法が異なります。

### SvmW（Optuna 最適化）

| 手法 (CONDITION) | 説明 | RATIO 使用 |
|---|---|---|
| `baseline` | 不均衡対策なし | ✗ |
| `smote_plain` | Plain SMOTE | ✓ |
| `smote` | Subject-wise SMOTE | ✓ |
| `undersample` | Random Under-Sampling | ✓ |
| `balanced_rf` | Balanced RF（SvmW と併用） | ✓ |

### SvmA（PSO 最適化）

| 手法 (CONDITION) | 説明 | RATIO 使用 |
|---|---|---|
| `baseline` | 不均衡対策なし | ✗ |
| `smote_plain` | Plain SMOTE | ✓ |
| `smote` | Subject-wise SMOTE | ✓ |
| `undersample` | Random Under-Sampling | ✓ |

> ※ Balanced RF は SvmA には適用不可

### Lstm（Deep Learning, CPU モード）

| 手法 (CONDITION) | 説明 | RATIO 使用 |
|---|---|---|
| `baseline` | 不均衡対策なし | ✗ |
| `smote_plain` | Plain SMOTE | ✓ |
| `smote` | Subject-wise SMOTE | ✓ |
| `undersample` | Random Under-Sampling | ✓ |

> ※ Balanced RF は Lstm には適用不可

## ジョブ数の計算

各モデルの 1 ループ（DISTANCE × DOMAIN × MODE × SEED）あたりの条件数:

| モデル | baseline | 比率ベース手法 × 比率数 | 合計/ループ |
|---|---|---|---|
| SvmW | 1 | 4 手法 × 2 比率 = 8 | 9 |
| SvmA | 1 | 3 手法 × 2 比率 = 6 | 7 |
| Lstm | 1 | 3 手法 × 2 比率 = 6 | 7 |

ループ数: 3 (DISTANCE) × 2 (DOMAIN) × 2 (MODE) × 2 (SEED) = **24**

| モデル | 条件数/ループ | ループ数 | ジョブ数 |
|---|---|---|---|
| SvmW | 9 | 24 | **216** |
| SvmA | 7 | 24 | **168** |
| Lstm | 7 | 24 | **168** |
| **合計** | | | **552** |

## HPC リソース設定

| モデル | CPU | メモリ | 制限時間 | キュー分散 |
|---|---|---|---|---|
| SvmA | 8 | 32 GB | 24:00:00 | SINGLE / LONG / DEFAULT（ラウンドロビン） |
| SvmW | 8 | 16 GB | 12:00:00 | SINGLE / LONG / DEFAULT（ラウンドロビン） |
| Lstm | 8 | 32 GB | 16:00:00 | SINGLE / LONG / DEFAULT（ラウンドロビン） |

> 複数キューへのラウンドロビン分散投入（`USE_MULTI_QUEUE=true`）を使用。

## ジョブ例（PBS 環境）

```bash
# SvmW baseline
qsub -N Sv_bs_mo_s_s42 -l select=1:ncpus=8:mem=16gb -l walltime=12:00:00 -q SINGLE \
    -v MODEL=SvmW,CONDITION=baseline,MODE=source_only,DISTANCE=mmd,DOMAIN=out_domain,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true \
    scripts/hpc/jobs/train/pbs_prior_research_split2.sh

# SvmA SMOTE with ratio
qsub -N Sv_sm_mo_s_r0.1_s42 -l select=1:ncpus=8:mem=32gb -l walltime=24:00:00 -q LONG \
    -v MODEL=SvmA,CONDITION=smote,MODE=source_only,DISTANCE=mmd,DOMAIN=out_domain,RATIO=0.1,SEED=42,N_TRIALS=100,RANKING=knn,RUN_EVAL=true \
    scripts/hpc/jobs/train/pbs_prior_research_split2.sh
```

## 出力先

- 評価結果: `results/outputs/evaluation/{SvmW,SvmA,Lstm}/`
- 訓練ログ: `results/outputs/training/{SvmW,SvmA,Lstm}/`
- モデル: `models/{SvmW,SvmA,Lstm}/{JOB_ID}/`

## 備考

- SvmA は PSO（Particle Swarm Optimization）による最適化のため、実行時間が最も長い
- Lstm は `CUDA_VISIBLE_DEVICES=""` で CPU モード強制
- 全ジョブで `OMP_NUM_THREADS=1` 等を設定し、決定性を確保

## 関連ドキュメント

- [先行研究パイプライン](../../architecture/prior_research.md)
- [実験結果](../results/prior_research_results.md)
- [再現手順](../reproducibility.md#experiment-3-prior-research-replication-split2)

---

作成日: 2026-02-07
