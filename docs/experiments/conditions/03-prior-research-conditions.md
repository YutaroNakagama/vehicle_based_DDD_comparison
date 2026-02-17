# 実験3（先行研究再現）の実験条件

このファイルは「実験3：先行研究モデルの再現（split2 ドメイン分割版）」で使用した実験条件の一覧を示す。

---

## 概要

- **目的**: 先行研究のモデル（SvmA, SvmW, Lstm）を split2 ドメイン分割で再現し、ドメインシフト耐性を評価
- **モデル**: SvmW, SvmA, Lstm（3 種）
- **データ分割**: `split2`（`in_domain`: 44 名、`out_domain`: 43 名）
- **ランチャー**: `scripts/hpc/launchers/launch_prior_research_split2.sh`
- **ジョブスクリプト**: `scripts/hpc/jobs/train/pbs_prior_research_split2.sh`
- **自動投入デーモン**:
  - `scripts/hpc/launchers/auto_resub_svmw_zhao2009.sh`
  - `scripts/hpc/launchers/auto_resub_svma_arefnezhad2019.sh`
  - `scripts/hpc/launchers/auto_resub_lstm_wang2022.sh`
- **ジョブリスト生成**:
  - `scripts/hpc/launchers/gen_svmw_joblist.sh`
  - `scripts/hpc/launchers/gen_svma_joblist.sh`
  - `scripts/hpc/launchers/gen_lstm_joblist.sh`

---

## 実験マトリクス

| パラメータ | 値 |
|---|---|
| モデル | SvmW, SvmA, Lstm (3) |
| 距離指標 | mmd, dtw, wasserstein (3) |
| ドメイングループ | in_domain (44名), out_domain (43名) (2) |
| 訓練モード | source_only, target_only (2) |
| シード | 42, 123 (2) |
| 不均衡対策手法 | baseline, smote_plain, smote, undersample (4) |
| ターゲット比率 | 0.1, 0.5 (ratio-based のみ) (2) |
| 距離ランキング | knn |
| Optuna trials | 100 (SvmW のみ) |

### 訓練モード

| モード | 説明 | 訓練データ | 評価データ |
|---|---|---|---|
| source_only | Cross-domain | ターゲットの逆ドメイン | ターゲットドメイン |
| target_only | Within-domain | ターゲットドメイン内 | ターゲットドメイン内 |

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

```
baseline:    3 距離 × 2 ドメイン × 2 モード × 2 シード × 1           = 24
smote_plain: 3 距離 × 2 ドメイン × 2 モード × 2 シード × 2 比率      = 48
smote:       3 距離 × 2 ドメイン × 2 モード × 2 シード × 2 比率      = 48
undersample: 3 距離 × 2 ドメイン × 2 モード × 2 シード × 2 比率      = 48
────────────────────────────────────────────────────────
合計: 168 ジョブ/モデル × 3 モデル = 504 ジョブ
```

> **Note:** ランチャーのフッタ計算では SvmW=216 と表示されるが、これは旧設定（`balanced_rf` 条件を含む）の名残。現在は `balanced_rf` は除外されており、全モデル共通で 168 ジョブ。

---

## HPC リソース設定

### メインランチャーのデフォルト設定

| モデル | CPU | メモリ | Walltime | 備考 |
|---|---|---|---|---|
| SvmW | 8 | 16 GB | 12:00:00 | Optuna 最適化 |
| SvmA | 8 | 32 GB | 24:00:00 | PSO 最適化 |
| Lstm | 8 | 32 GB | 16:00:00 | Deep Learning (CPU) |

### 自動投入デーモンでの更新設定（commit 49cf96e 以降）

SMOTE 系条件（smote, smote_plain）で walltime 超過が頻発したため、ジョブリスト生成スクリプトで以下のように増加させた。

| モデル | 条件 | 旧 Walltime | 新 Walltime | CPU | メモリ |
|---|---|---|---|---|---|
| SvmW | baseline, undersample | 12:00:00 | 12:00:00 | 8 | 16 GB |
| SvmW | smote, smote_plain | 12:00:00 | **24:00:00** | 8 | 16 GB |
| SvmA | baseline, undersample | 24:00:00 | 24:00:00 | 8 | 32 GB |
| SvmA | smote, smote_plain | 24:00:00 | **48:00:00** | 8 | 32 GB |
| Lstm | baseline | 16:00:00 | 16:00:00 | 4 | 16 GB |
| Lstm | smote, smote_plain, undersample | 16:00:00 | **24:00:00** | 4 | 16 GB |

> **Note:** Lstm のデーモンは `ncpus=4`（メインランチャーの 8 から削減）。

### キュー配分

デーモンはキュー空き状況を `qstat` で確認し、以下の上限内でラウンドロビン投入する。

| キュー | 上限 |
|---|---|
| SINGLE | 40 |
| DEFAULT | 40 |
| LONG | 15 |
| SMALL | 30 |

---

## タグ命名規則

```
prior_{MODEL}_{CONDITION}_{RANKING}_{DISTANCE}_{DOMAIN}_{MODE}_split2_s{SEED}
```

例: `prior_SvmW_baseline_knn_mmd_out_domain_source_only_split2_s42`

比率指定がある場合:
```
prior_{MODEL}_{CONDITION}_{RATIO}_{RANKING}_{DISTANCE}_{DOMAIN}_{MODE}_split2_s{SEED}
```

例: `prior_SvmA_smote_0.1_knn_dtw_in_domain_target_only_split2_s123`

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

```
results/outputs/evaluation/{MODEL}/{JOB_ID}/{JOB_ID}[1]/
├── eval_results_{MODEL}_{mode}_{tag}.json     # 評価メトリクス
└── eval_results_{MODEL}_{mode}_{tag}.csv      # 詳細結果
```

---

## 関連ドキュメント

- [再現性ガイド](../reproducibility.md) — 実験の再現方法
- [結果](../results/03-prior-research-results.md) — 実験3の結果
- [先行研究モデル](../../architecture/prior_research.md) — モデルアーキテクチャ詳細
