# 実験3（先行研究再現）の実験条件

このファイルは「実験3：先行研究モデルの再現（domain_train 統一版）」で使用した実験条件の一覧を示す。

> **改訂履歴:** 旧 split2 版（source_only/target_only モード、504 ジョブ）から domain_train 統一版（252 ジョブ）へ移行。
> 旧版では各ドメインに対し source_only と target_only で同一モデルを2回訓練していたが、
> domain_train モードでは各ドメインを1回だけ訓練し、2回評価（within-domain / cross-domain）する設計に変更。

---

## 概要

- **目的**: 先行研究のモデル（SvmA, SvmW, Lstm）を split2 ドメイン分割で再現し、ドメインシフト耐性を評価
- **モデル**: SvmW, SvmA, Lstm（3 種）
- **訓練モード**: `domain_train`（各ドメインのデータを 70/15/15 で分割し1回訓練、2回評価）
- **データ分割**: `split2`（`in_domain`: 44 名、`out_domain`: 43 名）
- **ランチャー**: `scripts/hpc/launchers/launch_prior_research_unified.sh`
- **ジョブスクリプト**:
  - CPU（SvmW / SvmA）: `scripts/hpc/jobs/train/pbs_prior_research_unified.sh`
  - GPU（Lstm）: `scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh`
- **自動投入デーモン**: `scripts/hpc/launchers/auto_resub_unified_v2.sh`（全モデル統一、GPU キュー自動振り分け）

---

## 実験マトリクス

| パラメータ | 値 |
|---|---|
| モデル | SvmW, SvmA, Lstm (3) |
| 距離指標 | mmd, dtw, wasserstein (3) |
| ドメイングループ | in_domain (44名), out_domain (43名) (2) |
| 訓練モード | domain_train (1) |
| シード | 42, 123 (2) |
| 不均衡対策手法 | baseline, smote_plain, smote, undersample (4) |
| ターゲット比率 | 0.1, 0.5 (ratio-based のみ) (2) |
| 距離ランキング | knn |
| Optuna trials | 100 (SvmW のみ) |

### 訓練モード

| モード | 説明 | 訓練データ | 評価（within） | 評価（cross） |
|---|---|---|---|---|
| domain_train | 統一モード | ターゲットドメインの train(70%) | 同ドメインの test(15%) | 逆ドメインの test(15%) |

> **旧 split2 版との違い:**
> - 旧版: `source_only`（逆ドメインで訓練→ターゲットで評価）+ `target_only`（同ドメインで訓練→同ドメインで評価）の2モード → 同一モデルが2回訓練される
> - 新版: `domain_train` の1モードで各ドメインのデータを 70/15/15 分割 → 1回訓練、2回評価（within + cross）
> - 効果: ジョブ数が半減（504 → 252）、同一の学習結果から within/cross 両方の比較が可能

### 不均衡対策手法

| 手法 | 比率パラメータ | ジョブ数/メトリクス組 |
|---|---|---|
| baseline | なし | 1 |
| smote_plain | 0.1, 0.5 | 2 |
| smote | 0.1, 0.5 | 2 |
| undersample | 0.1, 0.5 | 2 |

1メトリクス組 = 1距離 × 1ドメイン × 1モード × 1シード

### ジョブ数の計算

各モデルとも同一の4条件（baseline, smote_plain, smote, undersample）を使用する。
domain_train モードは1モード（旧版の source_only/target_only 2モードから統一）。

```
baseline:    3 距離 × 2 ドメイン × 1 モード × 2 シード × 1           = 12
smote_plain: 3 距離 × 2 ドメイン × 1 モード × 2 シード × 2 比率      = 24
smote:       3 距離 × 2 ドメイン × 1 モード × 2 シード × 2 比率      = 24
undersample: 3 距離 × 2 ドメイン × 1 モード × 2 シード × 2 比率      = 24
────────────────────────────────────────────────────────
合計: 84 ジョブ/モデル × 3 モデル = 252 ジョブ
```

> **Note:** 旧 split2 版では 168 ジョブ/モデル × 3 = 504 ジョブだったが、domain_train への移行で半減。

---

## HPC リソース設定

### 統一ランチャーのデフォルト設定

| モデル | CPU | メモリ | Walltime | GPU | 備考 |
|---|---|---|---|---|---|
| SvmW | 8 | 16 GB | 12:00:00 | なし | Optuna 最適化 |
| SvmA | 8 | 32 GB | 24:00:00 | なし | PSO 最適化 |
| Lstm | 4 | 16 GB | 16:00:00 | 1 (A40/A100) | GPU 高速化 |

> **Lstm GPU 対応:** Lstm は GPU PBS スクリプト（`pbs_prior_research_unified_gpu.sh`）で実行。
> `module load hpc_sdk/22.2` で CUDA ライブラリをロードし、`configure_gpu()` で TensorFlow の GPU メモリ成長を設定。

### SMOTE 系条件の walltime 増加

SMOTE 系条件（smote, smote_plain）で walltime 超過が頻発したため、以下のように増加。

| モデル | 条件 | 旧 Walltime | 新 Walltime | CPU | メモリ |
|---|---|---|---|---|---|
| SvmW | baseline, undersample | 12:00:00 | 12:00:00 | 8 | 16 GB |
| SvmW | smote, smote_plain | 12:00:00 | **24:00:00** | 8 | 16 GB |
| SvmA | baseline, undersample | 24:00:00 | 24:00:00 | 8 | 32 GB |
| SvmA | smote, smote_plain | 24:00:00 | **48:00:00** | 8 | 32 GB |
| Lstm | baseline | 16:00:00 | 16:00:00 | 4 | 16 GB |
| Lstm | smote, smote_plain, undersample | 16:00:00 | **24:00:00** | 4 | 16 GB |

### キュー配分

自動投入デーモン（`auto_resub_unified_v2.sh`）はモデルに応じて CPU / GPU キューを自動振り分けする。

**CPU キュー（SvmW / SvmA）:**

| キュー | max_run/user | 用途 |
|---|---|---|
| SINGLE | 10 | メイン投入先 |
| DEFAULT | 20 | メイン投入先 |
| SMALL | 7 | 補助 |
| LONG | 2 | 長時間ジョブ |

**GPU キュー（Lstm）:**

| キュー | GPU | max_run/user | 用途 |
|---|---|---|---|
| GPU-1 | A40 | 4 | メインGPU |
| GPU-1A | A100 | 2 | 高速GPU |
| GPU-S | A40 | 2 | 補助GPU |
| GPU-L | A40 | 1 | 補助GPU |
| GPU-LA | A100 | 1 | 補助GPU |

> デーモンは `qstat` でキュー空き状況を確認し、空きのあるキューにラウンドロビンで投入する。
> CPU 最大同時実行: 39 ジョブ、GPU 最大同時実行: 10 ジョブ。

---

## タグ命名規則

```
prior_{MODEL}_{CONDITION}_{RANKING}_{DISTANCE}_{DOMAIN}_domain_train_split2_s{SEED}
```

例: `prior_SvmW_baseline_knn_mmd_out_domain_domain_train_split2_s42`

比率指定がある場合:
```
prior_{MODEL}_{CONDITION}_{RATIO}_{RANKING}_{DISTANCE}_{DOMAIN}_domain_train_split2_s{SEED}
```

例: `prior_SvmA_smote_0.1_knn_dtw_in_domain_domain_train_split2_s123`

---

## 出力アーティファクト

### モデルファイル

```
models/{MODEL}/{JOB_ID}/{JOB_ID}[1]/
├── {MODEL}_{mode}_{tag}_{jobid}_1.keras      # 最終モデル (Lstm: .keras)
├── {MODEL}_{mode}_{tag}_{jobid}_1.pkl         # 最終モデル (SvmW/SvmA: .pkl)
├── {MODEL}_fold{N}_{jobid}_1.keras            # Fold モデル (Lstm のみ, N=1-5)
├── scaler_{MODEL}_{mode}_{tag}_{jobid}_1.pkl  # スケーラー
├── scaler_{MODEL}_fold{N}_{jobid}_1.pkl       # Fold スケーラー (Lstm)
├── selected_features_*.pkl                    # 選択特徴量
├── feature_meta_*.json                        # 特徴量メタデータ
├── threshold_*.json                           # 分類閾値 (SvmW)
└── training_history_*.json                    # 学習履歴 (Lstm)
```

### Optuna Study (SvmW のみ)

```
models/SvmW/{JOB_ID}/
├── optuna_SvmW_{mode}_{tag}_study.pkl         # Optuna study
├── optuna_SvmW_{mode}_{tag}_trials.csv        # Trial 履歴
└── optuna_SvmW_{mode}_{tag}_convergence.json  # 収束データ
```

### 評価結果

domain_train モードでは各ジョブに対し2回の評価（within / cross）が実行される。
ファイル名にサフィックス `_within` / `_cross` が付与される。

```
results/outputs/evaluation/{MODEL}/{JOB_ID}/{JOB_ID}[1]/
├── eval_results_{MODEL}_{mode}_{tag}_within.json   # Within-domain 評価（同ドメインの test 15%）
├── eval_results_{MODEL}_{mode}_{tag}_within.csv
├── eval_results_{MODEL}_{mode}_{tag}_cross.json    # Cross-domain 評価（逆ドメインの test 15%）
└── eval_results_{MODEL}_{mode}_{tag}_cross.csv
```

> **Note:** 旧 split2 版ではサフィックスなしの単一ファイルだった。
> domain_train 移行時に `eval_type` がファイル名に含まれるよう `savers.py` を修正。

---

## 関連ドキュメント

- [再現性ガイド](../reproducibility.md) — 実験の再現方法
- [結果](../results/03-prior-research-results.md) — 実験3の結果
- [先行研究モデル](../../architecture/prior_research.md) — モデルアーキテクチャ詳細
