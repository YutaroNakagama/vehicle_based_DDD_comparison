# Domain Shift Experiment Results (Experiment 2)

> **Note:** For pipeline details, see [Domain Generalization Pipeline](../../architecture/domain_generalization.md).
> For condition definitions, see [Domain Conditions](../conditions/02-domain-conditions.md).

---

## Experiment 2: Domain Generalization with Subject-Split (split2)

**Last updated:** 2026-02-17

### Overview

Experiment 2 investigates domain generalization performance of **RF モデル** under a vehicle-based subject split (split2).
Training and evaluation are performed for all combinations of training modes, distance metrics, domain groups,
imbalance handling methods, and random seeds.

### Experiment Matrix

| Parameter | Values | Count |
|-----------|--------|-------|
| Distance metrics | mmd, dtw, wasserstein | 3 |
| Domain groups | in_domain (44), out_domain (43) | 2 |
| Training modes | source_only, target_only, mixed | 3 |
| Seeds | 42, 123 | 2 |
| Conditions | baseline, smote_plain, smote (sw_smote), undersample, balanced_rf | 5 (→8 jobs/combo) |

**Total expected configurations:** 288 (192 cross/within-domain + 96 mixed)

### Experiment Status (2026-02-17)

| Condition | Model | Expected | Completed | Status |
|-----------|-------|----------|-----------|--------|
| baseline | RF | 36 | 36 | ✅ |
| undersample (RUS) | RF | 72 | 72 | ✅ |
| smote_plain | RF | 72 | 70 | ⏳ (2 jobs in LONG queue) |
| smote (sw_smote) | RF | 72 | 72 | ✅ |
| balanced_rf | BalancedRF | 36 | 36 | ✅ |
| **Total** | | **288** | **286** | **99.3%** |

> **Note:** smote_plain の残り 2 件は walltime 超過のため LONG queue で再実行中。
> auto_retry daemon (PID 2554345) が LONG queue 空き待ちで自動投入。

### Condition 命名規則

コード上の `CONDITION` パラメータと実際のタグ名の対応:

| CONDITION (launcher) | eval ファイル中のタグ | 処理内容 |
|---------------------|---------------------|---------|
| `baseline` | `baseline_domain_*` | 不均衡対策なし（class_weight のみ） |
| `smote_plain` | `smote_plain_*` | グローバル SMOTE |
| `smote` | `imbalv3_*` / `swsmote_*` | Subject-wise SMOTE（被験者単位） |
| `undersample` | `undersample_rus_*` | Random Under-Sampling |
| `balanced_rf` | `balanced_rf_*` | BalancedRandomForestClassifier |

> **注意:** sw_smote のタグ名は開発過程で `swsmote_*` → `imbalv3_*` と変遷している。
> 両方の命名が結果ディレクトリに混在するが、同じ処理を指す。

### HPC リソース設定

| Condition | CPUs | Memory | Walltime | Queue |
|-----------|------|--------|----------|-------|
| baseline / undersample | 4 | 8 GB | 06:00:00 | SINGLE |
| smote / smote_plain | 4 | 10 GB | 08:00:00 | SINGLE |
| balanced_rf | 8 | 12 GB | 08:00:00 | LONG |

> **Known issue:** smote_plain の ratio=0.5 + out_domain 設定で DEFAULT queue (10h) の walltime を超過。
> LONG queue (15h) で再投入済み。

### Output Structure

```
results/outputs/evaluation/
├── RF/                    # baseline, smote_plain, sw_smote, undersample
│   └── {JOB_ID}/
│       └── {JOB_ID}[1]/
│           └── eval_results_*.json
└── BalancedRF/            # balanced_rf condition
    └── {JOB_ID}/
        └── {JOB_ID}[1]/
            └── eval_results_*.json
```

(内容は元ファイルから移行されています — 旧ファイル名: `domain_results.md`)
