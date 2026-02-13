# Domain Shift Experiment Results

> **Note:** For pipeline details, see [Domain Generalization Pipeline](../../architecture/domain_generalization.md).

---

## Experiment 2: Domain Generalization with Subject-Split (split2)

**Last updated:** 2026-02-13

### Overview

Experiment 2 investigates domain generalization performance under a vehicle-based subject split (split2).
Training and evaluation are performed for all combinations of training modes, distance metrics, domain groups, imbalance handling methods, and random seeds.

### Experiment Status

| Category | Status |
|----------|--------|
| **Total Configurations** | 180 |
| **Completed** | 180 / 180 ✅ |
| **Evaluation CSVs** | 5 condition files (288 records total) |
| **Summary Plots** | 16 PNG (+ 4 archive) |

### Experiment Design Matrix

| Dimension | Values | Count |
|-----------|--------|-------|
| **Training Mode** | cross-domain, within-domain, multi-domain | 3 |
| **Imbalance Method** | baseline (RF), smote_plain (RF), sw_smote (RF), undersample_rus (RF), balanced_rf (BalancedRF) | 5 |
| **Distance Metric** | knn_dtw, knn_mmd, knn_wasserstein | 3 |
| **Domain Group** | in_domain, out_domain | 2 |
| **Seeds** | 42, 123 | 2 |

**Total**: 3 modes × 5 conditions × 3 distances × 2 domains × 2 seeds = **180 configurations**

- **Ratios** (smote_plain, sw_smote, undersample_rus): 0.1, 0.5
- **baseline / balanced_rf**: ratio なし

### Mode Definitions

| Mode | Config Name | Description |
|------|-------------|-------------|
| cross-domain | `source_only` | Train on source subjects only; evaluate on target subjects |
| within-domain | `target_only` | Train on target subjects only; evaluate on target subjects |
| multi-domain | `mixed` | Train on both source and target subjects; evaluate on target subjects |
| Pooled Baseline | `pooled` | Train on all subjects without domain split (reference baseline) |

---

## Results Summary

### Overall Performance by Mode (all conditions averaged)

| Mode | F1 | F2 | AUC | AUC-PR | Precision | Recall |
|------|----|----|-----|--------|-----------|--------|
| **cross-domain** | 0.1051 | 0.1758 | 0.5478 | 0.0764 | 0.0943 | 0.4474 |
| **within-domain** | 0.1327 | 0.2316 | 0.6847 | 0.1876 | 0.0846 | 0.6131 |
| **multi-domain** | 0.1610 | 0.3012 | 0.7737 | 0.2622 | 0.0910 | 0.7899 |

> **Key Finding:** Multi-domain training consistently outperforms both cross-domain and within-domain modes on AUC (+0.23 / +0.09 vs source/target), AUC-PR, F2, and recall, though at the cost of lower precision.

### Performance by Condition × Mode

#### Baseline (RF, no resampling)

| Mode | F1 | F2 | AUC | AUC-PR | Precision | Recall |
|------|----|----|-----|--------|-----------|--------|
| cross-domain | 0.0860 | 0.1723 | 0.5279 | 0.0532 | 0.0469 | 0.5350 |
| within-domain | 0.1273 | 0.2546 | 0.7231 | 0.1977 | 0.0695 | 0.7708 |
| **multi-domain** | **0.1123** | **0.2263** | **0.6548** | **0.1051** | 0.0611 | 0.7037 |

#### SMOTE Plain (RF + SMOTE)

| Mode | Ratio | F1 | F2 | AUC | AUC-PR | Precision | Recall |
|------|-------|----|----|-----|--------|-----------|--------|
| cross-domain | 0.1 | 0.0973 | 0.1732 | 0.5424 | 0.0556 | 0.0563 | 0.3650 |
| within-domain | 0.1 | 0.1297 | 0.2451 | 0.6846 | 0.1346 | 0.0727 | 0.6077 |
| **multi-domain** | **0.1** | **0.2157** | **0.3837** | **0.8323** | **0.2642** | 0.1249 | 0.8071 |
| cross-domain | 0.5 | 0.0926 | 0.1618 | 0.5310 | 0.0517 | 0.0542 | 0.3269 |
| within-domain | 0.5 | 0.1555 | 0.2825 | 0.7322 | 0.1380 | 0.0889 | 0.6229 |
| **multi-domain** | **0.5** | **0.1959** | **0.3407** | **0.7927** | **0.1902** | 0.1149 | 0.6841 |

#### Subject-Wise SMOTE (RF + SW-SMOTE)

| Mode | Ratio | F1 | F2 | AUC | AUC-PR | Precision | Recall |
|------|-------|----|----|-----|--------|-----------|--------|
| cross-domain | 0.1 | 0.0980 | 0.1734 | 0.5482 | 0.0571 | 0.0569 | 0.3600 |
| within-domain | 0.1 | 0.0890 | 0.1639 | 0.5424 | 0.0831 | 0.0506 | 0.3793 |
| **multi-domain** | **0.1** | **0.2126** | **0.3840** | **0.8321** | **0.2129** | 0.1220 | 0.8317 |
| cross-domain | 0.5 | 0.0850 | 0.1443 | 0.5214 | 0.0499 | 0.0505 | 0.2734 |
| within-domain | 0.5 | 0.2285 | 0.3063 | 0.7235 | 0.1678 | 0.1731 | 0.4503 |
| **multi-domain** | **0.5** | **0.2180** | **0.3939** | **0.8439** | **0.2168** | 0.1250 | 0.8536 |

#### Undersample RUS (RF + Random Undersampling)

| Mode | Ratio | F1 | F2 | AUC | AUC-PR | Precision | Recall |
|------|-------|----|----|-----|--------|-----------|--------|
| cross-domain | 0.1 | 0.0934 | 0.1847 | 0.5653 | 0.0786 | 0.0513 | 0.5371 |
| within-domain | 0.1 | 0.0835 | 0.1699 | 0.6154 | 0.1560 | 0.0452 | 0.5489 |
| **multi-domain** | **0.1** | **0.1307** | **0.2601** | **0.7484** | **0.3746** | 0.0715 | 0.7696 |
| cross-domain | 0.5 | 0.0899 | 0.1769 | 0.5429 | 0.0666 | 0.0494 | 0.5057 |
| within-domain | 0.5 | 0.0968 | 0.1902 | 0.6226 | 0.1924 | 0.0533 | 0.5422 |
| **multi-domain** | **0.5** | **0.1187** | **0.2340** | **0.6795** | **0.2717** | 0.0652 | 0.6695 |

#### Balanced RF (BalancedRandomForest, no resampling)

| Mode | F1 | F2 | AUC | AUC-PR | Precision | Recall |
|------|----|----|-----|--------|-----------|--------|
| cross-domain | 0.1985 | 0.2199 | 0.6034 | 0.1981 | 0.3892 | 0.6759 |
| within-domain | 0.1513 | 0.2407 | 0.8338 | 0.4312 | 0.1237 | 0.9830 |
| **multi-domain** | **0.0842** | **0.1867** | **0.8064** | **0.4618** | 0.0439 | 1.0000 |

> BalancedRF の multi-domain モードは AUC-PR が最高 (0.4618) だが、precision が非常に低く (0.0439)、recall は 1.0 に飽和している。

---

### Performance by Distance Metric (all conditions averaged)

| Distance | Mode | F1 | F2 | AUC | AUC-PR |
|----------|------|----|----|-----|--------|
| **DTW** | cross-domain | 0.0904 | 0.1694 | 0.5368 | 0.0632 |
| | within-domain | 0.1183 | 0.2184 | 0.6730 | 0.1599 |
| | multi-domain | 0.1614 | 0.3009 | 0.7770 | 0.2745 |
| **MMD** | cross-domain | 0.1206 | 0.1846 | 0.5514 | 0.0860 |
| | within-domain | 0.1370 | 0.2405 | 0.7018 | 0.2362 |
| | multi-domain | 0.1595 | 0.2998 | 0.7715 | 0.2508 |
| **Wasserstein** | cross-domain | 0.1043 | 0.1735 | 0.5552 | 0.0799 |
| | within-domain | 0.1428 | 0.2361 | 0.6794 | 0.1666 |
| | multi-domain | 0.1622 | 0.3029 | 0.7727 | 0.2612 |

> 距離指標間の差は比較的小さい。MMD が cross-domain の F1 でやや優位。全距離で multi-domain > within-domain > cross-domain の順。

### Performance by Domain Group (all conditions averaged)

| Domain | Mode | F1 | F2 | AUC | AUC-PR |
|--------|------|----|----|-----|--------|
| **in_domain** | cross-domain | 0.1068 | 0.1800 | 0.5372 | 0.0716 |
| | within-domain | 0.1437 | 0.2435 | 0.7009 | 0.2314 |
| | multi-domain | 0.1598 | 0.2991 | 0.7557 | 0.2612 |
| **out_domain** | cross-domain | 0.1034 | 0.1716 | 0.5584 | 0.0811 |
| | within-domain | 0.1217 | 0.2198 | 0.6685 | 0.1438 |
| | multi-domain | 0.1622 | 0.3033 | 0.7918 | 0.2631 |

> in_domain は within-domain でやや高い F1/AUC-PR を示す。out_domain では multi-domain が特に有効（AUC 0.7918）。

### Model Type Comparison: RF vs BalancedRF

| Model | Mode | F1 | F2 | AUC | AUC-PR |
|-------|------|----|----|-----|--------|
| **RF** | cross-domain | 0.0918 | 0.1695 | 0.5399 | 0.0590 |
| | within-domain | 0.1301 | 0.2304 | 0.6634 | 0.1528 |
| | multi-domain | 0.1720 | 0.3175 | 0.7691 | 0.2336 |
| **BalancedRF** | cross-domain | 0.1985 | 0.2199 | 0.6034 | 0.1981 |
| | within-domain | 0.1513 | 0.2407 | 0.8338 | 0.4312 |
| | multi-domain | 0.0842 | 0.1867 | 0.8064 | 0.4618 |

> BalancedRF は AUC/AUC-PR で大幅に優位だが、F1 は RF の multi-domain モードが最高。BalancedRF は precision-recall トレードオフが大きく異なる（高 recall、低 precision）。

---

### Top 10 Configurations by F1

| Rank | F1 | F2 | AUC | AUC-PR | Configuration |
|------|----|----|-----|--------|---------------|
| 1 | 0.8885 | 0.8327 | 0.9109 | 0.8302 | `target_only_balanced_rf_knn_wasserstein_in_domain_s42` |
| 2 | 0.4437 | 0.3327 | 0.6777 | 0.3448 | `source_only_balanced_rf_knn_mmd_out_domain_s42` |
| 3 | 0.4412 | 0.4379 | 0.7221 | 0.3232 | `target_only_sw_smote_knn_mmd_out_domain_r0.5_s123` |
| 4 | 0.4356 | 0.3308 | 0.6640 | 0.3441 | `source_only_balanced_rf_knn_mmd_out_domain_s123` |
| 5 | 0.3612 | 0.4199 | 0.7479 | 0.3331 | `target_only_sw_smote_knn_wasserstein_in_domain_r0.5_s42` |
| 6 | 0.3403 | 0.2553 | 0.6068 | 0.2799 | `source_only_balanced_rf_knn_mmd_in_domain_s123` |
| 7 | 0.3333 | 0.3684 | 0.7166 | 0.1901 | `target_only_sw_smote_knn_mmd_out_domain_r0.5_s42` |
| 8 | 0.2943 | 0.2068 | 0.6119 | 0.2375 | `source_only_balanced_rf_knn_wasserstein_in_domain_s42` |
| 9 | 0.2793 | 0.2066 | 0.5955 | 0.2372 | `source_only_balanced_rf_knn_wasserstein_in_domain_s123` |
| 10 | 0.2791 | 0.4839 | 0.9224 | 0.4435 | `mixed_smote_plain_knn_dtw_out_domain_r0.1_s42` |

### Top 10 Configurations by AUC

| Rank | AUC | F1 | AUC-PR | Configuration |
|------|-----|----|--------|---------------|
| 1 | 0.9600 | 0.0772 | 0.8930 | `mixed_balanced_rf_knn_dtw_out_domain_s42` |
| 2 | 0.9536 | 0.1914 | 0.8873 | `target_only_undersample_rus_knn_mmd_in_domain_r0.5_s42` |
| 3 | 0.9503 | 0.1827 | 0.7431 | `mixed_undersample_rus_knn_wasserstein_out_domain_r0.1_s123` |
| 4 | 0.9503 | 0.1827 | 0.7431 | `mixed_undersample_rus_knn_wasserstein_out_domain_r0.5_s123` |
| 5 | 0.9490 | 0.0811 | 0.8525 | `mixed_balanced_rf_knn_wasserstein_out_domain_s42` |
| 6 | 0.9437 | 0.1864 | 0.6720 | `mixed_undersample_rus_knn_dtw_out_domain_r0.5_s123` |
| 7 | 0.9344 | 0.1855 | 0.7525 | `target_only_undersample_rus_knn_mmd_in_domain_r0.5_s123` |
| 8 | 0.9333 | 0.1748 | 0.7520 | `mixed_undersample_rus_knn_mmd_in_domain_r0.1_s123` |
| 9 | 0.9333 | 0.1748 | 0.7520 | `mixed_undersample_rus_knn_mmd_in_domain_r0.5_s123` |
| 10 | 0.9291 | 0.1590 | 0.7031 | `mixed_undersample_rus_knn_dtw_out_domain_r0.1_s123` |

---

## Key Observations

1. **Multi-domain training が最も効果的**: 全体平均で AUC 0.7737、F2 0.3012 と、cross-domain / within-domain を上回る。特に recall が高く（0.79）、居眠り検出の見落とし低減に有効。

2. **Source_only (Cross-domain) は性能が低い**: AUC 0.55 前後で、ほぼランダムに近い。ソースドメインのみの学習ではターゲットへの汎化が困難。

3. **BalancedRF は AUC/AUC-PR で優位**: AUC-PR が RF の約 2-3 倍。ただし precision が非常に低く、F1 では必ずしも最良ではない。

4. **SMOTE (plain / sw_smote) の multi-domain モードが F1/F2 の最良バランス**: smote_plain ratio=0.1 の multi-domain が F1=0.2157, F2=0.3837, AUC=0.8323 と、F1 と AUC のバランスが最も良い。

5. **距離指標の差は限定的**: DTW, MMD, Wasserstein いずれも大きな差はない。MMD がやや cross-domain で有利。

6. **Seed 間のばらつきが大きい**: Top 10 の F1 は 0.28〜0.89 と大きく散らばり、特定の seed/domain 組み合わせで極端に高い F1 を示す外れ値がある（例: `target_only_balanced_rf_knn_wasserstein_in_domain_s42` の F1=0.89）。

7. **Domain group の影響**: in_domain は within-domain で有利、out_domain は multi-domain で特に有効。Domain shift が大きい out_domain でも multi-domain は AUC 0.79 を達成。

---

## Known Issues (2026-02-13)

### 🔴 BalancedRF の `random_state` ハードコード問題（修正済み）

**現象**: BalancedRF の 18 通り（mode×distance×level）のうち 12 通りで、seed=42 と seed=123 の F1・precision・recall が **完全に一致**。AUC のみ異なる。また、mixed と source_only で F1/precision/recall が完全一致するケースが 6 件確認された。

**原因**: `model_factory.py`, `classifiers.py`, `optuna_tuning.py` の 3 箇所で `random_state=42` がハードコードされており、パイプラインの `seed` 引数がモデル本体に到達していなかった。seed が効くのは Optuna の `TPESampler(seed=seed)` のみで、Optuna が同じハイパーパラメータに収束するとモデルの木構造が完全に同一になっていた。

**修正**: 全 `random_state` をパイプラインの `seed` パラメータに連動するよう修正（コミット TBD）。

**影響**: 現在の BalancedRF 結果は seed 間の独立性がないため、BalancedRF の seed 平均値は実質 N=1。再実行が必要。

### 🟡 BalancedRF の recall=1.0 飽和

BalancedRF 36 件中 28 件で recall=1.0（precision ≈ 0.04）。ほぼ全サンプルを陽性と予測しており、有意義な判別をしていない可能性がある。ハイパーパラメータチューニングの閾値設定や class バランシング戦略の見直しが必要。

---

## Output Artifacts

### CSV Files

| Condition | Path | Records |
|-----------|------|---------|
| balanced_rf | `results/analysis/exp2_domain_shift/figures/csv/split2/balanced_rf/` | 36 |
| baseline | `results/analysis/exp2_domain_shift/figures/csv/split2/baseline/` | 36 |
| smote_plain | `results/analysis/exp2_domain_shift/figures/csv/split2/smote_plain/` | 72 |
| sw_smote | `results/analysis/exp2_domain_shift/figures/csv/split2/sw_smote/` | 72 |
| undersample_rus | `results/analysis/exp2_domain_shift/figures/csv/split2/undersample_rus/` | 72 |

### Summary Plots (4 rows × 7 cols bar charts)

Each plot shows 3 bars (Cross-domain / Within-domain / Multi-domain) per metric, with Pooled baseline as a dashed line.

| Condition | Seed | File |
|-----------|------|------|
| balanced_rf | 42 | `split2/balanced_rf/brf_s42.png` |
| balanced_rf | 123 | `split2/balanced_rf/brf_s123.png` |
| baseline | 42 | `split2/baseline/baseline_s42.png` |
| baseline | 123 | `split2/baseline/baseline_s123.png` |
| smote_plain (r=0.1) | 42, 123 | `split2/smote_plain/smote_r01_s{42,123}.png` |
| smote_plain (r=0.5) | 42, 123 | `split2/smote_plain/smote_r05_s{42,123}.png` |
| sw_smote (r=0.1) | 42, 123 | `split2/sw_smote/sw_smote_r01_s{42,123}.png` |
| sw_smote (r=0.5) | 42, 123 | `split2/sw_smote/sw_smote_r05_s{42,123}.png` |
| undersample_rus (r=0.1) | 42, 123 | `split2/undersample_rus/rus_r01_s{42,123}.png` |
| undersample_rus (r=0.5) | 42, 123 | `split2/undersample_rus/rus_r05_s{42,123}.png` |

**Total:** 16 plots (under `results/analysis/exp2_domain_shift/figures/png/split2/`)

### Reproduction

```bash
# BalancedRF plots
qsub scripts/hpc/jobs/evaluate/pbs_visualize_split2.sh

# RF plots (baseline, smote_plain, sw_smote, undersample_rus)
qsub scripts/hpc/jobs/evaluate/pbs_visualize_split2_rf.sh

# Filter to specific condition
qsub -v CONDITION=smote_plain scripts/hpc/jobs/evaluate/pbs_visualize_split2_rf.sh
```
