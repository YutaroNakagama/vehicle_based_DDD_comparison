# 実験2（ドメインシフト）の実験条件

このファイルは「実験2：ドメインシフト (split2)」で使用した実験条件の一覧を示します。

---

## 概要

- **目的**: split2 ドメイン分割による RF モデルのドメインシフト耐性評価
- **モデル**: RF（BalancedRF は不均衡対策手法として含む）
- **データ分割**: `split2`（`in_domain`: 44 名、`out_domain`: 43 名）
- **ランチャー**:
  - Cross/Within-domain: `scripts/hpc/launchers/launch_paper_domain_split2.sh`
  - Multi-domain: `scripts/hpc/launchers/launch_exp2_mixed.sh`
  - 追加シード投入: `scripts/hpc/launchers/exp2_10seeds_submit.sh`
  - 失敗ジョブ再投入: `scripts/hpc/launchers/resubmit_failed_exp2.sh`
- **ジョブスクリプト**: `scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh`

## 実験パラメータ

| パラメータ | 値 |
|-----------|-----|
| Distance metrics | mmd, dtw, wasserstein (3) |
| Domain groups | in_domain (44 名), out_domain (43 名) (2) |
| Training modes | source_only, target_only, mixed (3) |
| Seeds | 0, 1, 7, 13, 42, 123, 256, 512, 1337, 2024 (10) |
| Ranking method | knn |
| Optuna trials | 100 |
| CV strategy | StratifiedKFold (3-fold) |
| 最適化指標 | F2 score |

> **Note:** 初期実験は seed=42, 123 の 2 種で実施。その後 8 シードを追加し、
> 最終的に 10 シードでの安定性評価を完了（balanced_rf のみ 2 シード）。

## 不均衡対策条件（5 種 → 8 jobs/combo）

| # | CONDITION | 説明 | ratio | jobs/combo |
|---|-----------|------|-------|------------|
| 1 | `baseline` | 不均衡対策なし（class_weight のみ） | なし | 1 |
| 2 | `smote_plain` | グローバル SMOTE（全被験者プール後に適用） | 0.1, 0.5 | 2 |
| 3 | `smote` | Subject-wise SMOTE（被験者単位で SMOTE 適用） | 0.1, 0.5 | 2 |
| 4 | `undersample` | Random Under-Sampling (RUS) | 0.1, 0.5 | 2 |
| 5 | `balanced_rf` | BalancedRandomForestClassifier（内部バランシング） | なし | 1 |

> **smote_plain vs smote の違い:**
> - `smote_plain`: 全被験者のデータをプールした後に SMOTE を適用（標準的な SMOTE）
> - `smote`: 各被験者のデータに個別に SMOTE を適用してからプール（Subject-wise SMOTE）
>
> Subject-wise SMOTE は被験者間のデータ分布の違いを保持する利点がある。

## ジョブ数の計算

```
10 シード版（baseline, smote_plain, sw_smote, undersample_rus）:
  Cross/Within-domain:  3 dist × 2 dom × 2 mode × 10 seed × 7 cond  = 840 jobs
  Multi-domain (mixed): 3 dist × 2 dom × 1 mode × 10 seed × 7 cond  = 420 jobs
  小計: 1,260 jobs

2 シード版（balanced_rf のみ）:
  Cross/Within-domain:  3 dist × 2 dom × 2 mode × 2 seed × 1 cond   = 24 jobs
  Multi-domain (mixed): 3 dist × 2 dom × 1 mode × 2 seed × 1 cond   = 12 jobs
  小計: 36 jobs

合計: 1,260 + 36 = 1,296 jobs
```

> **Note:** 初期は 2 シード (42, 123) で 288 jobs として実行。
> 8 シード追加後に追加投入し、最終的に ~1,296 jobs を完了。

## Training Mode の定義

| Mode | 説明 | 訓練データ | 評価データ |
|------|------|-----------|-----------|
| `source_only` | Cross-domain | 反対ドメイン | 対象ドメイン |
| `target_only` | Within-domain | 同一ドメイン | 同一ドメイン |
| `mixed` | Multi-domain | 全 87 名（プール） | 対象ドメイン |

### split2 のデータ分割詳細

| Mode | Domain | 訓練データ | 評価データ |
|------|--------|-----------|-----------|
| source_only | out_domain | in_domain (44 名) | out_domain (43 名) |
| source_only | in_domain | out_domain (43 名) | in_domain (44 名) |
| target_only | out_domain | out_domain (43 名) | out_domain (43 名) |
| target_only | in_domain | in_domain (44 名) | in_domain (44 名) |
| mixed | out_domain | 全 87 名 | out_domain (43 名) |
| mixed | in_domain | 全 87 名 | in_domain (44 名) |

## HPC リソース設定

| Condition | CPUs | Memory | Walltime | Queue |
|-----------|------|--------|----------|-------|
| baseline / undersample | 4 | 8 GB | 06:00:00 | SINGLE |
| smote / smote_plain | 4 | 10 GB | 08:00:00 | SINGLE |
| balanced_rf | 8 | 12 GB | 08:00:00 | LONG |

> **Known issue:** smote_plain の ratio=0.5 + out_domain + seed=123 設定で
> DEFAULT queue (10h) の walltime を超過するケースがあり、LONG queue (15h) で再投入。

## 関連ドキュメント

- [実験結果](../results/02-domain-results.md)
- [再現性ガイド](../reproducibility.md)
- [Domain Generalization Pipeline](../../architecture/domain_generalization.md)
- [不均衡対策手法](../../reference/imbalance_methods.md)
