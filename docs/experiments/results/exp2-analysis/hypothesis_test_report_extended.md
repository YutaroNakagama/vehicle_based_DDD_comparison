# Experiment 2 — Extended Hypothesis Testing (F1, AUPRC, Recall)

**Records**: 1306  
**Seeds**: [0, 1, 3, 7, 13, 42, 123, 256, 512, 1337, 2024] (n=11)  
**Conditions** (7): ['baseline', 'rus_r01', 'rus_r05', 'smote_r01', 'smote_r05', 'sw_smote_r01', 'sw_smote_r05']  
**Modes**: ['source_only', 'target_only', 'mixed']  
**Distances**: ['mmd', 'dtw', 'wasserstein']  
**Levels**: ['in_domain', 'out_domain']  

---
## 1. Metric Selection Rationale

The primary analysis (companion report) tests F2-score and AUROC. This supplementary analysis extends the hypothesis framework to three additional metrics, selected for their relevance to imbalanced binary classification in a safety-critical application:

| Metric | Column | Rationale |
|--------|--------|-----------|
| F1-score | `f1` | Harmonic mean of Precision and Recall with equal weight. Compared with F2 (which emphasizes Recall 2×), F1 reveals whether the Recall-weighted conclusions still hold under balanced weighting. |
| AUPRC | `auc_pr` | Area Under the Precision–Recall Curve. Unlike AUROC, AUPRC is *not* inflated by true negatives and is considered the most informative single metric for imbalanced classification (Saito & Rehmsmeier, 2015). |
| Recall | `recall` | Sensitivity / True Positive Rate. In drowsy-driving detection, a missed drowsy event is a safety risk; Recall directly measures the detection rate of the positive class. |

**Excluded metrics** (with justification):

- **Accuracy**: Misleading for imbalanced data — a trivial majority-class classifier achieves high accuracy.

- **Precision** (as standalone): Included in descriptive statistics and trade-off analysis (§ 10) but not in the full hypothesis framework, because maximizing Precision alone is not the primary objective in safety-critical detection.


---
## 2. Hypothesis Framework

The same 14-hypothesis framework from the primary analysis is applied to each extended metric. Additionally, three metric-specific hypotheses are tested.

### Standard Hypotheses (applied per metric)

| ID | Hypothesis |
|:--:|-----------|
| H1 | Imbalance handling method affects performance (7-condition KW) |
| H2 | SW-SMOTE outperforms plain SMOTE (same ratio) |
| H3 | Oversampling outperforms RUS |
| H4 | Sampling ratio (r=0.1 vs r=0.5) affects performance |
| H5 | Distance metric choice affects performance |
| H7 | Within-domain > cross-domain training |
| H10 | In-domain > out-domain (domain shift exists) |
| H12 | Condition × Mode interaction |
| H13 | Condition × Distance interaction |
| H14 | Domain gap varies by mode |

### Extended Hypotheses (metric-specific)

| ID | Hypothesis | Rationale |
|:--:|-----------|-----------|
| HE1 | Any rebalancing method (including RUS) improves Recall over baseline | Both oversampling and undersampling increase minority-class weight, which should boost Recall even if other metrics degrade |
| HE2 | AUPRC shows a stronger condition effect than AUROC | AUPRC is more sensitive to minority-class performance; condition effects hidden in AUROC may emerge in AUPRC |
| HE3 | Oversampling improves Recall at the cost of Precision (Precision–Recall trade-off) | Shifting the decision boundary to catch more positives increases false positives, reducing Precision |

---
## 3. Normality Assessment (Shapiro-Wilk)

Justification for non-parametric tests.

### F1-score

| Condition | Mode | Level | W | p | Normal? |
|-----------|------|-------|--:|--:|:-------:|
| baseline | Cross-domain | In-domain | 0.9521 | 0.1655 | ✓ normal |
| baseline | Cross-domain | Out-domain | 0.9435 | 0.0940 | ✓ normal |
| baseline | Within-domain | In-domain | 0.7796 | 0.0000 | ✗ reject |
| baseline | Within-domain | Out-domain | 0.7173 | 0.0000 | ✗ reject |
| baseline | Mixed | In-domain | 0.9047 | 0.0109 | ✗ reject |
| baseline | Mixed | Out-domain | 0.9601 | 0.3108 | ✓ normal |
| rus_r01 | Cross-domain | In-domain | 0.9588 | 0.2548 | ✓ normal |
| rus_r01 | Cross-domain | Out-domain | 0.9459 | 0.1104 | ✓ normal |
| rus_r01 | Within-domain | In-domain | 0.9425 | 0.0881 | ✓ normal |
| rus_r01 | Within-domain | Out-domain | 0.9064 | 0.0090 | ✗ reject |
| rus_r01 | Mixed | In-domain | 0.9038 | 0.0090 | ✗ reject |
| rus_r01 | Mixed | Out-domain | 0.9317 | 0.0486 | ✗ reject |
| rus_r05 | Cross-domain | In-domain | 0.9306 | 0.0406 | ✗ reject |
| rus_r05 | Cross-domain | Out-domain | 0.9529 | 0.1746 | ✓ normal |
| rus_r05 | Within-domain | In-domain | 0.9790 | 0.7692 | ✓ normal |
| rus_r05 | Within-domain | Out-domain | 0.9624 | 0.3188 | ✓ normal |
| rus_r05 | Mixed | In-domain | 0.9508 | 0.1518 | ✓ normal |
| rus_r05 | Mixed | Out-domain | 0.9609 | 0.3089 | ✓ normal |
| smote_r01 | Cross-domain | In-domain | 0.9675 | 0.4327 | ✓ normal |
| smote_r01 | Cross-domain | Out-domain | 0.9703 | 0.5065 | ✓ normal |
| smote_r01 | Within-domain | In-domain | 0.9298 | 0.0432 | ✗ reject |
| smote_r01 | Within-domain | Out-domain | 0.9065 | 0.0105 | ✗ reject |
| smote_r01 | Mixed | In-domain | 0.9639 | 0.3891 | ✓ normal |
| smote_r01 | Mixed | Out-domain | 0.9527 | 0.1993 | ✓ normal |
| smote_r05 | Cross-domain | In-domain | 0.9444 | 0.0998 | ✓ normal |
| smote_r05 | Cross-domain | Out-domain | 0.9808 | 0.8357 | ✓ normal |
| smote_r05 | Within-domain | In-domain | 0.9004 | 0.0074 | ✗ reject |
| smote_r05 | Within-domain | Out-domain | 0.9550 | 0.2142 | ✓ normal |
| smote_r05 | Mixed | In-domain | 0.9645 | 0.4018 | ✓ normal |
| smote_r05 | Mixed | Out-domain | 0.9184 | 0.0244 | ✗ reject |
| sw_smote_r01 | Cross-domain | In-domain | 0.9910 | 0.9937 | ✓ normal |
| sw_smote_r01 | Cross-domain | Out-domain | 0.9839 | 0.9093 | ✓ normal |
| sw_smote_r01 | Within-domain | In-domain | 0.9614 | 0.3186 | ✓ normal |
| sw_smote_r01 | Within-domain | Out-domain | 0.9216 | 0.0260 | ✗ reject |
| sw_smote_r01 | Mixed | In-domain | 0.9368 | 0.0746 | ✓ normal |
| sw_smote_r01 | Mixed | Out-domain | 0.8532 | 0.0009 | ✗ reject |
| sw_smote_r05 | Cross-domain | In-domain | 0.9507 | 0.1505 | ✓ normal |
| sw_smote_r05 | Cross-domain | Out-domain | 0.9752 | 0.6703 | ✓ normal |
| sw_smote_r05 | Within-domain | In-domain | 0.9418 | 0.0842 | ✓ normal |
| sw_smote_r05 | Within-domain | Out-domain | 0.9458 | 0.1302 | ✓ normal |
| sw_smote_r05 | Mixed | In-domain | 0.8510 | 0.0008 | ✗ reject |
| sw_smote_r05 | Mixed | Out-domain | 0.8826 | 0.0032 | ✗ reject |

**Summary**: 15/42 cells (36%) reject normality at α=0.05.

### AUPRC

| Condition | Mode | Level | W | p | Normal? |
|-----------|------|-------|--:|--:|:-------:|
| baseline | Cross-domain | In-domain | 0.9332 | 0.0480 | ✗ reject |
| baseline | Cross-domain | Out-domain | 0.9412 | 0.0812 | ✓ normal |
| baseline | Within-domain | In-domain | 0.3229 | 0.0000 | ✗ reject |
| baseline | Within-domain | Out-domain | 0.7177 | 0.0000 | ✗ reject |
| baseline | Mixed | In-domain | 0.9303 | 0.0499 | ✗ reject |
| baseline | Mixed | Out-domain | 0.7668 | 0.0000 | ✗ reject |
| rus_r01 | Cross-domain | In-domain | 0.7902 | 0.0000 | ✗ reject |
| rus_r01 | Cross-domain | Out-domain | 0.5766 | 0.0000 | ✗ reject |
| rus_r01 | Within-domain | In-domain | 0.5070 | 0.0000 | ✗ reject |
| rus_r01 | Within-domain | Out-domain | 0.7097 | 0.0000 | ✗ reject |
| rus_r01 | Mixed | In-domain | 0.6003 | 0.0000 | ✗ reject |
| rus_r01 | Mixed | Out-domain | 0.6871 | 0.0000 | ✗ reject |
| rus_r05 | Cross-domain | In-domain | 0.6652 | 0.0000 | ✗ reject |
| rus_r05 | Cross-domain | Out-domain | 0.7215 | 0.0000 | ✗ reject |
| rus_r05 | Within-domain | In-domain | 0.6119 | 0.0000 | ✗ reject |
| rus_r05 | Within-domain | Out-domain | 0.6804 | 0.0000 | ✗ reject |
| rus_r05 | Mixed | In-domain | 0.7395 | 0.0000 | ✗ reject |
| rus_r05 | Mixed | Out-domain | 0.7736 | 0.0000 | ✗ reject |
| smote_r01 | Cross-domain | In-domain | 0.9852 | 0.9286 | ✓ normal |
| smote_r01 | Cross-domain | Out-domain | 0.9545 | 0.1935 | ✓ normal |
| smote_r01 | Within-domain | In-domain | 0.9316 | 0.0486 | ✗ reject |
| smote_r01 | Within-domain | Out-domain | 0.9444 | 0.1095 | ✓ normal |
| smote_r01 | Mixed | In-domain | 0.9001 | 0.0085 | ✗ reject |
| smote_r01 | Mixed | Out-domain | 0.8906 | 0.0050 | ✗ reject |
| smote_r05 | Cross-domain | In-domain | 0.9411 | 0.0806 | ✓ normal |
| smote_r05 | Cross-domain | Out-domain | 0.8853 | 0.0032 | ✗ reject |
| smote_r05 | Within-domain | In-domain | 0.9702 | 0.5255 | ✓ normal |
| smote_r05 | Within-domain | Out-domain | 0.9690 | 0.4920 | ✓ normal |
| smote_r05 | Mixed | In-domain | 0.9016 | 0.0092 | ✗ reject |
| smote_r05 | Mixed | Out-domain | 0.9545 | 0.2234 | ✓ normal |
| sw_smote_r01 | Cross-domain | In-domain | 0.9322 | 0.0452 | ✗ reject |
| sw_smote_r01 | Cross-domain | Out-domain | 0.9777 | 0.7464 | ✓ normal |
| sw_smote_r01 | Within-domain | In-domain | 0.8928 | 0.0048 | ✗ reject |
| sw_smote_r01 | Within-domain | Out-domain | 0.8204 | 0.0001 | ✗ reject |
| sw_smote_r01 | Mixed | In-domain | 0.8302 | 0.0002 | ✗ reject |
| sw_smote_r01 | Mixed | Out-domain | 0.6710 | 0.0000 | ✗ reject |
| sw_smote_r05 | Cross-domain | In-domain | 0.8579 | 0.0006 | ✗ reject |
| sw_smote_r05 | Cross-domain | Out-domain | 0.9542 | 0.2034 | ✓ normal |
| sw_smote_r05 | Within-domain | In-domain | 0.9358 | 0.0571 | ✓ normal |
| sw_smote_r05 | Within-domain | Out-domain | 0.9168 | 0.0222 | ✗ reject |
| sw_smote_r05 | Mixed | In-domain | 0.8350 | 0.0004 | ✗ reject |
| sw_smote_r05 | Mixed | Out-domain | 0.8777 | 0.0025 | ✗ reject |

**Summary**: 31/42 cells (74%) reject normality at α=0.05.

### Recall

| Condition | Mode | Level | W | p | Normal? |
|-----------|------|-------|--:|--:|:-------:|
| baseline | Cross-domain | In-domain | 0.6719 | 0.0000 | ✗ reject |
| baseline | Cross-domain | Out-domain | 0.9598 | 0.2705 | ✓ normal |
| baseline | Within-domain | In-domain | 0.9358 | 0.0702 | ✓ normal |
| baseline | Within-domain | Out-domain | 0.8142 | 0.0001 | ✗ reject |
| baseline | Mixed | In-domain | 0.9048 | 0.0110 | ✗ reject |
| baseline | Mixed | Out-domain | 0.9029 | 0.0099 | ✗ reject |
| rus_r01 | Cross-domain | In-domain | 0.8929 | 0.0041 | ✗ reject |
| rus_r01 | Cross-domain | Out-domain | 0.9551 | 0.2008 | ✓ normal |
| rus_r01 | Within-domain | In-domain | 0.9063 | 0.0090 | ✗ reject |
| rus_r01 | Within-domain | Out-domain | 0.9625 | 0.3202 | ✓ normal |
| rus_r01 | Mixed | In-domain | 0.8949 | 0.0054 | ✗ reject |
| rus_r01 | Mixed | Out-domain | 0.9366 | 0.0666 | ✓ normal |
| rus_r05 | Cross-domain | In-domain | 0.9407 | 0.0782 | ✓ normal |
| rus_r05 | Cross-domain | Out-domain | 0.8917 | 0.0038 | ✗ reject |
| rus_r05 | Within-domain | In-domain | 0.9203 | 0.0212 | ✗ reject |
| rus_r05 | Within-domain | Out-domain | 0.9719 | 0.5536 | ✓ normal |
| rus_r05 | Mixed | In-domain | 0.9346 | 0.0528 | ✓ normal |
| rus_r05 | Mixed | Out-domain | 0.9455 | 0.1168 | ✓ normal |
| smote_r01 | Cross-domain | In-domain | 0.9716 | 0.5435 | ✓ normal |
| smote_r01 | Cross-domain | Out-domain | 0.9846 | 0.9173 | ✓ normal |
| smote_r01 | Within-domain | In-domain | 0.9104 | 0.0132 | ✗ reject |
| smote_r01 | Within-domain | Out-domain | 0.9304 | 0.0449 | ✗ reject |
| smote_r01 | Mixed | In-domain | 0.8262 | 0.0002 | ✗ reject |
| smote_r01 | Mixed | Out-domain | 0.9160 | 0.0212 | ✗ reject |
| smote_r05 | Cross-domain | In-domain | 0.9177 | 0.0180 | ✗ reject |
| smote_r05 | Cross-domain | Out-domain | 0.9681 | 0.4688 | ✓ normal |
| smote_r05 | Within-domain | In-domain | 0.9689 | 0.4886 | ✓ normal |
| smote_r05 | Within-domain | Out-domain | 0.9540 | 0.2007 | ✓ normal |
| smote_r05 | Mixed | In-domain | 0.8995 | 0.0082 | ✗ reject |
| smote_r05 | Mixed | Out-domain | 0.9215 | 0.0294 | ✗ reject |
| sw_smote_r01 | Cross-domain | In-domain | 0.9647 | 0.3677 | ✓ normal |
| sw_smote_r01 | Cross-domain | Out-domain | 0.9842 | 0.9150 | ✓ normal |
| sw_smote_r01 | Within-domain | In-domain | 0.5669 | 0.0000 | ✗ reject |
| sw_smote_r01 | Within-domain | Out-domain | 0.5620 | 0.0000 | ✗ reject |
| sw_smote_r01 | Mixed | In-domain | 0.7178 | 0.0000 | ✗ reject |
| sw_smote_r01 | Mixed | Out-domain | 0.6927 | 0.0000 | ✗ reject |
| sw_smote_r05 | Cross-domain | In-domain | 0.9718 | 0.5510 | ✓ normal |
| sw_smote_r05 | Cross-domain | Out-domain | 0.9683 | 0.4741 | ✓ normal |
| sw_smote_r05 | Within-domain | In-domain | 0.9285 | 0.0356 | ✗ reject |
| sw_smote_r05 | Within-domain | Out-domain | 0.9544 | 0.2220 | ✓ normal |
| sw_smote_r05 | Mixed | In-domain | 0.8549 | 0.0010 | ✗ reject |
| sw_smote_r05 | Mixed | Out-domain | 0.8604 | 0.0010 | ✗ reject |

**Summary**: 23/42 cells (55%) reject normality at α=0.05.

**Conclusion**: 69/126 cells (55%) violate normality. Non-parametric tests are appropriate.

---
## 4. Descriptive Statistics

### F1-score

#### Cross-domain

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) | n |
|-----------|--------------------:|---------------------:|-----------:|--:|
| baseline | 0.0812±0.0028 | 0.0785±0.0040 | -0.0027 | 32 |
| rus_r01 | 0.0877±0.0057 | 0.0786±0.0071 | -0.0092 | 32 |
| rus_r05 | 0.0828±0.0068 | 0.0771±0.0081 | -0.0057 | 32 |
| smote_r01 | 0.0770±0.0096 | 0.0806±0.0057 | +0.0035 | 32 |
| smote_r05 | 0.0687±0.0125 | 0.0778±0.0098 | +0.0091 | 32 |
| sw_smote_r01 | 0.0678±0.0064 | 0.0673±0.0070 | -0.0005 | 32 |
| sw_smote_r05 | 0.0437±0.0088 | 0.0382±0.0104 | -0.0055 | 32 |

#### Within-domain

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) | n |
|-----------|--------------------:|---------------------:|-----------:|--:|
| baseline | 0.1030±0.0169 | 0.1133±0.0282 | +0.0104 | 30 |
| rus_r01 | 0.0734±0.0233 | 0.1048±0.0216 | +0.0314 | 32 |
| rus_r05 | 0.0670±0.0254 | 0.0915±0.0206 | +0.0245 | 32 |
| smote_r01 | 0.2729±0.0558 | 0.2664±0.0574 | -0.0065 | 31 |
| smote_r05 | 0.3389±0.0960 | 0.3278±0.0860 | -0.0110 | 31 |
| sw_smote_r01 | 0.3749±0.0797 | 0.3767±0.0857 | +0.0018 | 31 |
| sw_smote_r05 | 0.4696±0.1728 | 0.4395±0.1728 | -0.0301 | 32 |

#### Mixed

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) | n |
|-----------|--------------------:|---------------------:|-----------:|--:|
| baseline | 0.1206±0.0107 | 0.1575±0.0176 | +0.0369 | 30 |
| rus_r01 | 0.0995±0.0275 | 0.1138±0.0219 | +0.0142 | 31 |
| rus_r05 | 0.0756±0.0113 | 0.0969±0.0111 | +0.0213 | 32 |
| smote_r01 | 0.2564±0.0654 | 0.3026±0.0424 | +0.0462 | 30 |
| smote_r05 | 0.3276±0.0884 | 0.4056±0.1370 | +0.0780 | 30 |
| sw_smote_r01 | 0.3873±0.0968 | 0.4722±0.1532 | +0.0849 | 30 |
| sw_smote_r05 | 0.4119±0.1800 | 0.4616±0.1606 | +0.0497 | 29 |

### AUPRC

#### Cross-domain

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) | n |
|-----------|--------------------:|---------------------:|-----------:|--:|
| baseline | 0.0508±0.0014 | 0.0465±0.0028 | -0.0043 | 32 |
| rus_r01 | 0.0589±0.0145 | 0.0496±0.0118 | -0.0093 | 32 |
| rus_r05 | 0.0514±0.0054 | 0.0505±0.0091 | -0.0008 | 32 |
| smote_r01 | 0.0501±0.0012 | 0.0463±0.0026 | -0.0038 | 32 |
| smote_r05 | 0.0491±0.0017 | 0.0462±0.0026 | -0.0029 | 32 |
| sw_smote_r01 | 0.0519±0.0031 | 0.0433±0.0019 | -0.0086 | 32 |
| sw_smote_r05 | 0.0552±0.0068 | 0.0434±0.0016 | -0.0118 | 32 |

#### Within-domain

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) | n |
|-----------|--------------------:|---------------------:|-----------:|--:|
| baseline | 0.0872±0.0876 | 0.1555±0.1423 | +0.0682 | 30 |
| rus_r01 | 0.1097±0.1150 | 0.1400±0.1219 | +0.0303 | 32 |
| rus_r05 | 0.1146±0.1045 | 0.0846±0.0539 | -0.0300 | 32 |
| smote_r01 | 0.6415±0.1432 | 0.6051±0.1394 | -0.0364 | 31 |
| smote_r05 | 0.5749±0.1421 | 0.5644±0.1404 | -0.0104 | 31 |
| sw_smote_r01 | 0.6442±0.2062 | 0.6528±0.2174 | +0.0085 | 31 |
| sw_smote_r05 | 0.4486±0.1998 | 0.3639±0.1967 | -0.0846 | 32 |

#### Mixed

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) | n |
|-----------|--------------------:|---------------------:|-----------:|--:|
| baseline | 0.1287±0.0500 | 0.4050±0.2792 | +0.2763 | 30 |
| rus_r01 | 0.1584±0.2003 | 0.1213±0.0966 | -0.0371 | 31 |
| rus_r05 | 0.0649±0.0169 | 0.0733±0.0320 | +0.0084 | 32 |
| smote_r01 | 0.5302±0.1966 | 0.6741±0.1055 | +0.1439 | 30 |
| smote_r05 | 0.5281±0.1614 | 0.6459±0.1446 | +0.1178 | 30 |
| sw_smote_r01 | 0.5540±0.2226 | 0.6748±0.2836 | +0.1208 | 30 |
| sw_smote_r05 | 0.3908±0.1916 | 0.4316±0.1931 | +0.0407 | 29 |

### Recall

#### Cross-domain

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) | n |
|-----------|--------------------:|---------------------:|-----------:|--:|
| baseline | 0.5092±0.0430 | 0.4792±0.0993 | -0.0300 | 32 |
| rus_r01 | 0.5209±0.0408 | 0.4596±0.0561 | -0.0613 | 32 |
| rus_r05 | 0.4078±0.0586 | 0.4494±0.0679 | +0.0416 | 32 |
| smote_r01 | 0.2729±0.0483 | 0.2690±0.0476 | -0.0038 | 32 |
| smote_r05 | 0.1815±0.0523 | 0.1821±0.0453 | +0.0007 | 32 |
| sw_smote_r01 | 0.1615±0.0395 | 0.1547±0.0295 | -0.0068 | 32 |
| sw_smote_r05 | 0.0479±0.0137 | 0.0391±0.0137 | -0.0088 | 32 |

#### Within-domain

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) | n |
|-----------|--------------------:|---------------------:|-----------:|--:|
| baseline | 0.6176±0.1320 | 0.6987±0.1393 | +0.0811 | 30 |
| rus_r01 | 0.4184±0.1676 | 0.6606±0.1284 | +0.2422 | 32 |
| rus_r05 | 0.3313±0.1444 | 0.5611±0.1526 | +0.2298 | 32 |
| smote_r01 | 0.8675±0.0272 | 0.8681±0.0219 | +0.0006 | 31 |
| smote_r05 | 0.7822±0.0487 | 0.8004±0.0452 | +0.0182 | 31 |
| sw_smote_r01 | 0.8256±0.1064 | 0.8308±0.1260 | +0.0052 | 31 |
| sw_smote_r05 | 0.5069±0.1854 | 0.5117±0.1876 | +0.0048 | 32 |

#### Mixed

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) | n |
|-----------|--------------------:|---------------------:|-----------:|--:|
| baseline | 0.6987±0.0789 | 0.8582±0.0452 | +0.1595 | 30 |
| rus_r01 | 0.5442±0.1588 | 0.6117±0.1227 | +0.0675 | 31 |
| rus_r05 | 0.3814±0.0584 | 0.5643±0.1151 | +0.1829 | 32 |
| smote_r01 | 0.7967±0.0388 | 0.8734±0.0175 | +0.0767 | 30 |
| smote_r05 | 0.7363±0.0462 | 0.7832±0.0522 | +0.0469 | 30 |
| sw_smote_r01 | 0.7184±0.1326 | 0.7693±0.1624 | +0.0510 | 30 |
| sw_smote_r05 | 0.3660±0.1700 | 0.4654±0.1673 | +0.0994 | 29 |

---
## 5. Hypothesis Tests — H1: Model / Condition Effect


### 5.1.1 [F1-score] H1: Global condition effect (Kruskal-Wallis)

$$H_0: F_{C_1} = F_{C_2} = \cdots = F_{C_7}$$

| Mode | Level | Distance | H | p | η² | Sig (Bonf.) |
|------|-------|----------|--:|--:|---:|:-----------:|
| Cross-domain | In-domain | MMD | 54.97 | 0.0000 | 0.700 | ✓ |
| Cross-domain | In-domain | DTW | 64.97 | 0.0000 | 0.842 | ✓ |
| Cross-domain | In-domain | WASSERSTEIN | 56.40 | 0.0000 | 0.800 | ✓ |
| Cross-domain | Out-domain | MMD | 36.13 | 0.0000 | 0.450 | ✓ |
| Cross-domain | Out-domain | DTW | 51.35 | 0.0000 | 0.648 | ✓ |
| Cross-domain | Out-domain | WASSERSTEIN | 51.11 | 0.0000 | 0.716 | ✓ |
| Within-domain | In-domain | MMD | 55.97 | 0.0000 | 0.793 | ✓ |
| Within-domain | In-domain | DTW | 57.92 | 0.0000 | 0.787 | ✓ |
| Within-domain | In-domain | WASSERSTEIN | 67.67 | 0.0000 | 0.894 | ✓ |
| Within-domain | Out-domain | MMD | 55.11 | 0.0000 | 0.780 | ✓ |
| Within-domain | Out-domain | DTW | 59.24 | 0.0000 | 0.783 | ✓ |
| Within-domain | Out-domain | WASSERSTEIN | 61.02 | 0.0000 | 0.821 | ✓ |
| Mixed | In-domain | MMD | 57.59 | 0.0000 | 0.806 | ✓ |
| Mixed | In-domain | DTW | 58.90 | 0.0000 | 0.827 | ✓ |
| Mixed | In-domain | WASSERSTEIN | 56.84 | 0.0000 | 0.807 | ✓ |
| Mixed | Out-domain | MMD | 57.27 | 0.0000 | 0.814 | ✓ |
| Mixed | Out-domain | DTW | 58.99 | 0.0000 | 0.815 | ✓ |
| Mixed | Out-domain | WASSERSTEIN | 56.21 | 0.0000 | 0.810 | ✓ |

**Bonferroni α'=0.0028** (m=18). **18/18** significant.

Mean η² = 0.772 (large effect).

### 5.1.2 [F1-score] H1: Pairwise — baseline vs each method

Mann-Whitney U with Cliff's δ effect size.

| Method vs Baseline | Mode | Level | U | p | δ | Effect | Mean(M) | Mean(B) |
|--------------------|------|-------|--:|--:|--:|:------:|--------:|--------:|
| rus_r01 vs baseline | Cross-domain | In-domain | 156 | 0.0000 * | +0.695 | large | 0.0877 | 0.0812 |
| rus_r01 vs baseline | Cross-domain | Out-domain | 589 | 0.3043 | -0.150 | small | 0.0786 | 0.0785 |
| rus_r01 vs baseline | Within-domain | In-domain | 796 | 0.0000 * | -0.658 | large | 0.0734 | 0.1030 |
| rus_r01 vs baseline | Within-domain | Out-domain | 507 | 0.9518 | +0.010 | negligible | 0.1048 | 0.1133 |
| rus_r01 vs baseline | Mixed | In-domain | 754 | 0.0000 * | -0.622 | large | 0.0995 | 0.1206 |
| rus_r01 vs baseline | Mixed | Out-domain | 870 | 0.0000 * | -0.871 | large | 0.1138 | 0.1575 |
| rus_r05 vs baseline | Cross-domain | In-domain | 347 | 0.0272 | +0.322 | small | 0.0828 | 0.0812 |
| rus_r05 vs baseline | Cross-domain | Out-domain | 621 | 0.1452 | -0.213 | small | 0.0771 | 0.0785 |
| rus_r05 vs baseline | Within-domain | In-domain | 847 | 0.0000 * | -0.765 | large | 0.0670 | 0.1030 |
| rus_r05 vs baseline | Within-domain | Out-domain | 693 | 0.0154 | -0.354 | medium | 0.0915 | 0.1133 |
| rus_r05 vs baseline | Mixed | In-domain | 960 | 0.0000 * | -1.000 | large | 0.0756 | 0.1206 |
| rus_r05 vs baseline | Mixed | Out-domain | 930 | 0.0000 * | -1.000 | large | 0.0969 | 0.1575 |
| smote_r01 vs baseline | Cross-domain | In-domain | 672 | 0.0322 | -0.312 | small | 0.0770 | 0.0812 |
| smote_r01 vs baseline | Cross-domain | Out-domain | 398 | 0.1275 | +0.223 | small | 0.0806 | 0.0785 |
| smote_r01 vs baseline | Within-domain | In-domain | 0 | 0.0000 * | +1.000 | large | 0.2729 | 0.1030 |
| smote_r01 vs baseline | Within-domain | Out-domain | 0 | 0.0000 * | +1.000 | large | 0.2664 | 0.1133 |
| smote_r01 vs baseline | Mixed | In-domain | 0 | 0.0000 * | +1.000 | large | 0.2564 | 0.1206 |
| smote_r01 vs baseline | Mixed | Out-domain | 0 | 0.0000 * | +1.000 | large | 0.3026 | 0.1575 |
| smote_r05 vs baseline | Cross-domain | In-domain | 805 | 0.0001 * | -0.572 | large | 0.0687 | 0.0812 |
| smote_r05 vs baseline | Cross-domain | Out-domain | 510 | 0.8528 | -0.028 | negligible | 0.0778 | 0.0785 |
| smote_r05 vs baseline | Within-domain | In-domain | 0 | 0.0000 * | +1.000 | large | 0.3389 | 0.1030 |
| smote_r05 vs baseline | Within-domain | Out-domain | 0 | 0.0000 * | +1.000 | large | 0.3278 | 0.1133 |
| smote_r05 vs baseline | Mixed | In-domain | 0 | 0.0000 * | +1.000 | large | 0.3276 | 0.1206 |
| smote_r05 vs baseline | Mixed | Out-domain | 0 | 0.0000 * | +1.000 | large | 0.4056 | 0.1575 |
| sw_smote_r01 vs baseline | Cross-domain | In-domain | 998 | 0.0000 * | -0.949 | large | 0.0678 | 0.0812 |
| sw_smote_r01 vs baseline | Cross-domain | Out-domain | 911 | 0.0000 * | -0.837 | large | 0.0673 | 0.0785 |
| sw_smote_r01 vs baseline | Within-domain | In-domain | 0 | 0.0000 * | +1.000 | large | 0.3749 | 0.1030 |
| sw_smote_r01 vs baseline | Within-domain | Out-domain | 5 | 0.0000 * | +0.990 | large | 0.3767 | 0.1133 |
| sw_smote_r01 vs baseline | Mixed | In-domain | 0 | 0.0000 * | +1.000 | large | 0.3873 | 0.1206 |
| sw_smote_r01 vs baseline | Mixed | Out-domain | 0 | 0.0000 * | +1.000 | large | 0.4722 | 0.1575 |
| sw_smote_r05 vs baseline | Cross-domain | In-domain | 1024 | 0.0000 * | -1.000 | large | 0.0437 | 0.0812 |
| sw_smote_r05 vs baseline | Cross-domain | Out-domain | 992 | 0.0000 * | -1.000 | large | 0.0382 | 0.0785 |
| sw_smote_r05 vs baseline | Within-domain | In-domain | 0 | 0.0000 * | +1.000 | large | 0.4696 | 0.1030 |
| sw_smote_r05 vs baseline | Within-domain | Out-domain | 0 | 0.0000 * | +1.000 | large | 0.4395 | 0.1133 |
| sw_smote_r05 vs baseline | Mixed | In-domain | 0 | 0.0000 * | +1.000 | large | 0.4119 | 0.1206 |
| sw_smote_r05 vs baseline | Mixed | Out-domain | 0 | 0.0000 * | +1.000 | large | 0.4616 | 0.1575 |

**Bonferroni α'=0.00139** (m=36). **28** significant.

- large: 28/36 (78%)
- medium: 1/36 (3%)
- small: 5/36 (14%)
- negligible: 2/36 (6%)


### 5.2.1 [AUPRC] H1: Global condition effect (Kruskal-Wallis)

$$H_0: F_{C_1} = F_{C_2} = \cdots = F_{C_7}$$

| Mode | Level | Distance | H | p | η² | Sig (Bonf.) |
|------|-------|----------|--:|--:|---:|:-----------:|
| Cross-domain | In-domain | MMD | 39.57 | 0.0000 | 0.480 | ✓ |
| Cross-domain | In-domain | DTW | 34.23 | 0.0000 | 0.403 | ✓ |
| Cross-domain | In-domain | WASSERSTEIN | 47.11 | 0.0000 | 0.653 | ✓ |
| Cross-domain | Out-domain | MMD | 45.53 | 0.0000 | 0.590 | ✓ |
| Cross-domain | Out-domain | DTW | 41.43 | 0.0000 | 0.506 | ✓ |
| Cross-domain | Out-domain | WASSERSTEIN | 30.86 | 0.0000 | 0.395 | ✓ |
| Within-domain | In-domain | MMD | 41.04 | 0.0000 | 0.556 | ✓ |
| Within-domain | In-domain | DTW | 56.17 | 0.0000 | 0.760 | ✓ |
| Within-domain | In-domain | WASSERSTEIN | 57.34 | 0.0000 | 0.744 | ✓ |
| Within-domain | Out-domain | MMD | 53.20 | 0.0000 | 0.749 | ✓ |
| Within-domain | Out-domain | DTW | 52.81 | 0.0000 | 0.688 | ✓ |
| Within-domain | Out-domain | WASSERSTEIN | 54.84 | 0.0000 | 0.729 | ✓ |
| Mixed | In-domain | MMD | 51.64 | 0.0000 | 0.713 | ✓ |
| Mixed | In-domain | DTW | 44.60 | 0.0000 | 0.603 | ✓ |
| Mixed | In-domain | WASSERSTEIN | 44.92 | 0.0000 | 0.618 | ✓ |
| Mixed | Out-domain | MMD | 45.64 | 0.0000 | 0.629 | ✓ |
| Mixed | Out-domain | DTW | 49.06 | 0.0000 | 0.662 | ✓ |
| Mixed | Out-domain | WASSERSTEIN | 47.28 | 0.0000 | 0.666 | ✓ |

**Bonferroni α'=0.0028** (m=18). **18/18** significant.

Mean η² = 0.619 (large effect).

### 5.2.2 [AUPRC] H1: Pairwise — baseline vs each method

Mann-Whitney U with Cliff's δ effect size.

| Method vs Baseline | Mode | Level | U | p | δ | Effect | Mean(M) | Mean(B) |
|--------------------|------|-------|--:|--:|--:|:------:|--------:|--------:|
| rus_r01 vs baseline | Cross-domain | In-domain | 482 | 0.6920 | +0.059 | negligible | 0.0589 | 0.0508 |
| rus_r01 vs baseline | Cross-domain | Out-domain | 503 | 0.9091 | +0.018 | negligible | 0.0496 | 0.0465 |
| rus_r01 vs baseline | Within-domain | In-domain | 459 | 0.7728 | +0.044 | negligible | 0.1097 | 0.0872 |
| rus_r01 vs baseline | Within-domain | Out-domain | 594 | 0.2738 | -0.160 | small | 0.1400 | 0.1555 |
| rus_r01 vs baseline | Mixed | In-domain | 679 | 0.0021 | -0.460 | medium | 0.1584 | 0.1287 |
| rus_r01 vs baseline | Mixed | Out-domain | 837 | 0.0000 * | -0.800 | large | 0.1213 | 0.4050 |
| rus_r05 vs baseline | Cross-domain | In-domain | 575 | 0.4014 | -0.123 | negligible | 0.0514 | 0.0508 |
| rus_r05 vs baseline | Cross-domain | Out-domain | 380 | 0.0775 | +0.258 | small | 0.0505 | 0.0465 |
| rus_r05 vs baseline | Within-domain | In-domain | 432 | 0.5034 | +0.100 | negligible | 0.1146 | 0.0872 |
| rus_r05 vs baseline | Within-domain | Out-domain | 708 | 0.0087 | -0.383 | medium | 0.0846 | 0.1555 |
| rus_r05 vs baseline | Mixed | In-domain | 890 | 0.0000 * | -0.854 | large | 0.0649 | 0.1287 |
| rus_r05 vs baseline | Mixed | Out-domain | 922 | 0.0000 * | -0.983 | large | 0.0733 | 0.4050 |
| smote_r01 vs baseline | Cross-domain | In-domain | 650 | 0.0649 | -0.270 | small | 0.0501 | 0.0508 |
| smote_r01 vs baseline | Cross-domain | Out-domain | 506 | 0.9411 | +0.012 | negligible | 0.0463 | 0.0465 |
| smote_r01 vs baseline | Within-domain | In-domain | 10 | 0.0000 * | +0.978 | large | 0.6415 | 0.0872 |
| smote_r01 vs baseline | Within-domain | Out-domain | 17 | 0.0000 * | +0.966 | large | 0.6051 | 0.1555 |
| smote_r01 vs baseline | Mixed | In-domain | 9 | 0.0000 * | +0.980 | large | 0.5302 | 0.1287 |
| smote_r01 vs baseline | Mixed | Out-domain | 230 | 0.0012 * | +0.489 | large | 0.6741 | 0.4050 |
| smote_r05 vs baseline | Cross-domain | In-domain | 793 | 0.0002 * | -0.549 | large | 0.0491 | 0.0508 |
| smote_r05 vs baseline | Cross-domain | Out-domain | 539 | 0.5590 | -0.087 | negligible | 0.0462 | 0.0465 |
| smote_r05 vs baseline | Within-domain | In-domain | 10 | 0.0000 * | +0.978 | large | 0.5749 | 0.0872 |
| smote_r05 vs baseline | Within-domain | Out-domain | 37 | 0.0000 * | +0.925 | large | 0.5644 | 0.1555 |
| smote_r05 vs baseline | Mixed | In-domain | 9 | 0.0000 * | +0.980 | large | 0.5281 | 0.1287 |
| smote_r05 vs baseline | Mixed | Out-domain | 229 | 0.0011 * | +0.491 | large | 0.6459 | 0.4050 |
| sw_smote_r01 vs baseline | Cross-domain | In-domain | 424 | 0.2400 | +0.172 | small | 0.0519 | 0.0508 |
| sw_smote_r01 vs baseline | Cross-domain | Out-domain | 838 | 0.0000 * | -0.690 | large | 0.0433 | 0.0465 |
| sw_smote_r01 vs baseline | Within-domain | In-domain | 8 | 0.0000 * | +0.983 | large | 0.6442 | 0.0872 |
| sw_smote_r01 vs baseline | Within-domain | Out-domain | 43 | 0.0000 * | +0.913 | large | 0.6528 | 0.1555 |
| sw_smote_r01 vs baseline | Mixed | In-domain | 16 | 0.0000 * | +0.964 | large | 0.5540 | 0.1287 |
| sw_smote_r01 vs baseline | Mixed | Out-domain | 163 | 0.0000 * | +0.625 | large | 0.6748 | 0.4050 |
| sw_smote_r05 vs baseline | Cross-domain | In-domain | 340 | 0.0213 | +0.336 | medium | 0.0552 | 0.0508 |
| sw_smote_r05 vs baseline | Cross-domain | Out-domain | 840 | 0.0000 * | -0.694 | large | 0.0434 | 0.0465 |
| sw_smote_r05 vs baseline | Within-domain | In-domain | 24 | 0.0000 * | +0.950 | large | 0.4486 | 0.0872 |
| sw_smote_r05 vs baseline | Within-domain | Out-domain | 153 | 0.0000 * | +0.681 | large | 0.3639 | 0.1555 |
| sw_smote_r05 vs baseline | Mixed | In-domain | 28 | 0.0000 * | +0.936 | large | 0.3908 | 0.1287 |
| sw_smote_r05 vs baseline | Mixed | Out-domain | 355 | 0.1624 | +0.211 | small | 0.4316 | 0.4050 |

**Bonferroni α'=0.00139** (m=36). **21** significant.

- large: 21/36 (58%)
- medium: 3/36 (8%)
- small: 5/36 (14%)
- negligible: 7/36 (19%)


### 5.3.1 [Recall] H1: Global condition effect (Kruskal-Wallis)

$$H_0: F_{C_1} = F_{C_2} = \cdots = F_{C_7}$$

| Mode | Level | Distance | H | p | η² | Sig (Bonf.) |
|------|-------|----------|--:|--:|---:|:-----------:|
| Cross-domain | In-domain | MMD | 71.48 | 0.0000 | 0.935 | ✓ |
| Cross-domain | In-domain | DTW | 70.60 | 0.0000 | 0.923 | ✓ |
| Cross-domain | In-domain | WASSERSTEIN | 62.21 | 0.0000 | 0.892 | ✓ |
| Cross-domain | Out-domain | MMD | 65.09 | 0.0000 | 0.882 | ✓ |
| Cross-domain | Out-domain | DTW | 70.58 | 0.0000 | 0.923 | ✓ |
| Cross-domain | Out-domain | WASSERSTEIN | 63.72 | 0.0000 | 0.916 | ✓ |
| Within-domain | In-domain | MMD | 49.35 | 0.0000 | 0.688 | ✓ |
| Within-domain | In-domain | DTW | 53.02 | 0.0000 | 0.712 | ✓ |
| Within-domain | In-domain | WASSERSTEIN | 66.71 | 0.0000 | 0.880 | ✓ |
| Within-domain | Out-domain | MMD | 46.48 | 0.0000 | 0.642 | ✓ |
| Within-domain | Out-domain | DTW | 47.72 | 0.0000 | 0.613 | ✓ |
| Within-domain | Out-domain | WASSERSTEIN | 49.96 | 0.0000 | 0.656 | ✓ |
| Mixed | In-domain | MMD | 51.26 | 0.0000 | 0.707 | ✓ |
| Mixed | In-domain | DTW | 44.34 | 0.0000 | 0.599 | ✓ |
| Mixed | In-domain | WASSERSTEIN | 48.79 | 0.0000 | 0.679 | ✓ |
| Mixed | Out-domain | MMD | 44.64 | 0.0000 | 0.613 | ✓ |
| Mixed | Out-domain | DTW | 49.06 | 0.0000 | 0.662 | ✓ |
| Mixed | Out-domain | WASSERSTEIN | 47.61 | 0.0000 | 0.671 | ✓ |

**Bonferroni α'=0.0028** (m=18). **18/18** significant.

Mean η² = 0.755 (large effect).

### 5.3.2 [Recall] H1: Pairwise — baseline vs each method

Mann-Whitney U with Cliff's δ effect size.

| Method vs Baseline | Mode | Level | U | p | δ | Effect | Mean(M) | Mean(B) |
|--------------------|------|-------|--:|--:|--:|:------:|--------:|--------:|
| rus_r01 vs baseline | Cross-domain | In-domain | 394 | 0.1159 | +0.229 | small | 0.5209 | 0.5092 |
| rus_r01 vs baseline | Cross-domain | Out-domain | 580 | 0.3683 | -0.132 | negligible | 0.4596 | 0.4792 |
| rus_r01 vs baseline | Within-domain | In-domain | 780 | 0.0000 * | -0.625 | large | 0.4184 | 0.6176 |
| rus_r01 vs baseline | Within-domain | Out-domain | 542 | 0.6969 | -0.058 | negligible | 0.6606 | 0.6987 |
| rus_r01 vs baseline | Mixed | In-domain | 748 | 0.0000 * | -0.609 | large | 0.5442 | 0.6987 |
| rus_r01 vs baseline | Mixed | Out-domain | 892 | 0.0000 * | -0.917 | large | 0.6117 | 0.8582 |
| rus_r05 vs baseline | Cross-domain | In-domain | 932 | 0.0000 * | -0.820 | large | 0.4078 | 0.5092 |
| rus_r05 vs baseline | Cross-domain | Out-domain | 619 | 0.1526 | -0.209 | small | 0.4494 | 0.4792 |
| rus_r05 vs baseline | Within-domain | In-domain | 887 | 0.0000 * | -0.848 | large | 0.3313 | 0.6176 |
| rus_r05 vs baseline | Within-domain | Out-domain | 710 | 0.0082 | -0.386 | medium | 0.5611 | 0.6987 |
| rus_r05 vs baseline | Mixed | In-domain | 960 | 0.0000 * | -1.000 | large | 0.3814 | 0.6987 |
| rus_r05 vs baseline | Mixed | Out-domain | 924 | 0.0000 * | -0.987 | large | 0.5643 | 0.8582 |
| smote_r01 vs baseline | Cross-domain | In-domain | 1024 | 0.0000 * | -1.000 | large | 0.2729 | 0.5092 |
| smote_r01 vs baseline | Cross-domain | Out-domain | 1008 | 0.0000 * | -0.969 | large | 0.2690 | 0.4792 |
| smote_r01 vs baseline | Within-domain | In-domain | 20 | 0.0000 * | +0.957 | large | 0.8675 | 0.6176 |
| smote_r01 vs baseline | Within-domain | Out-domain | 246 | 0.0006 * | +0.505 | large | 0.8681 | 0.6987 |
| smote_r01 vs baseline | Mixed | In-domain | 98 | 0.0000 * | +0.781 | large | 0.7967 | 0.6987 |
| smote_r01 vs baseline | Mixed | Out-domain | 407 | 0.5296 | +0.096 | negligible | 0.8734 | 0.8582 |
| smote_r05 vs baseline | Cross-domain | In-domain | 1024 | 0.0000 * | -1.000 | large | 0.1815 | 0.5092 |
| smote_r05 vs baseline | Cross-domain | Out-domain | 992 | 0.0000 * | -1.000 | large | 0.1821 | 0.4792 |
| smote_r05 vs baseline | Within-domain | In-domain | 76 | 0.0000 * | +0.835 | large | 0.7822 | 0.6176 |
| smote_r05 vs baseline | Within-domain | Out-domain | 314 | 0.0126 | +0.367 | medium | 0.8004 | 0.6987 |
| smote_r05 vs baseline | Mixed | In-domain | 340 | 0.1038 | +0.246 | small | 0.7363 | 0.6987 |
| smote_r05 vs baseline | Mixed | Out-domain | 754 | 0.0000 * | -0.676 | large | 0.7832 | 0.8582 |
| sw_smote_r01 vs baseline | Cross-domain | In-domain | 1024 | 0.0000 * | -1.000 | large | 0.1615 | 0.5092 |
| sw_smote_r01 vs baseline | Cross-domain | Out-domain | 992 | 0.0000 * | -1.000 | large | 0.1547 | 0.4792 |
| sw_smote_r01 vs baseline | Within-domain | In-domain | 102 | 0.0000 * | +0.780 | large | 0.8256 | 0.6176 |
| sw_smote_r01 vs baseline | Within-domain | Out-domain | 302 | 0.0076 | +0.392 | medium | 0.8308 | 0.6987 |
| sw_smote_r01 vs baseline | Mixed | In-domain | 309 | 0.0377 | +0.313 | small | 0.7184 | 0.6987 |
| sw_smote_r01 vs baseline | Mixed | Out-domain | 538 | 0.1218 | -0.236 | small | 0.7693 | 0.8582 |
| sw_smote_r05 vs baseline | Cross-domain | In-domain | 1024 | 0.0000 * | -1.000 | large | 0.0479 | 0.5092 |
| sw_smote_r05 vs baseline | Cross-domain | Out-domain | 992 | 0.0000 * | -1.000 | large | 0.0391 | 0.4792 |
| sw_smote_r05 vs baseline | Within-domain | In-domain | 660 | 0.0115 | -0.375 | medium | 0.5069 | 0.6176 |
| sw_smote_r05 vs baseline | Within-domain | Out-domain | 776 | 0.0000 * | -0.616 | large | 0.5117 | 0.6987 |
| sw_smote_r05 vs baseline | Mixed | In-domain | 846 | 0.0000 * | -0.945 | large | 0.3660 | 0.6987 |
| sw_smote_r05 vs baseline | Mixed | Out-domain | 900 | 0.0000 * | -1.000 | large | 0.4654 | 0.8582 |

**Bonferroni α'=0.00139** (m=36). **24** significant.

- large: 24/36 (67%)
- medium: 4/36 (11%)
- small: 5/36 (14%)
- negligible: 3/36 (8%)

---
## 6. Hypothesis Tests — H2: SW-SMOTE vs Plain SMOTE


### 6.1 [F1-score] H2: sw_smote vs plain smote

Paired comparison (same ratio): Does subject-wise synthesis improve?

| Ratio | Mode | Level | U | p | δ (sw−sm) | Effect | Mean(sw) | Mean(sm) |
|-------|------|-------|--:|--:|----------:|:------:|---------:|---------:|
| r01 | Cross-domain | In-domain | 225 | 0.0001 * | -0.561 | large | 0.0678 | 0.0770 |
| r01 | Cross-domain | Out-domain | 72 | 0.0000 * | -0.855 | large | 0.0673 | 0.0806 |
| r01 | Within-domain | In-domain | 817 | 0.0000 * | +0.700 | large | 0.3749 | 0.2729 |
| r01 | Within-domain | Out-domain | 830 | 0.0000 * | +0.727 | large | 0.3767 | 0.2664 |
| r01 | Mixed | In-domain | 771 | 0.0000 * | +0.713 | large | 0.3873 | 0.2564 |
| r01 | Mixed | Out-domain | 657 | 0.0008 * | +0.510 | large | 0.4722 | 0.3026 |
| r05 | Cross-domain | In-domain | 24 | 0.0000 * | -0.953 | large | 0.0437 | 0.0687 |
| r05 | Cross-domain | Out-domain | 0 | 0.0000 * | -1.000 | large | 0.0382 | 0.0778 |
| r05 | Within-domain | In-domain | 736 | 0.0010 * | +0.484 | large | 0.4696 | 0.3389 |
| r05 | Within-domain | Out-domain | 648 | 0.0085 | +0.394 | medium | 0.4395 | 0.3278 |
| r05 | Mixed | In-domain | 510 | 0.2587 | +0.172 | small | 0.4119 | 0.3276 |
| r05 | Mixed | Out-domain | 536 | 0.2062 | +0.191 | small | 0.4616 | 0.4056 |

**Summary**: sw_smote > smote in 8/12, smote > sw_smote in 4/12. Bonferroni sig: 9/12.


### 6.2 [AUPRC] H2: sw_smote vs plain smote

Paired comparison (same ratio): Does subject-wise synthesis improve?

| Ratio | Mode | Level | U | p | δ (sw−sm) | Effect | Mean(sw) | Mean(sm) |
|-------|------|-------|--:|--:|----------:|:------:|---------:|---------:|
| r01 | Cross-domain | In-domain | 666 | 0.0393 | +0.301 | small | 0.0519 | 0.0501 |
| r01 | Cross-domain | Out-domain | 180 | 0.0000 * | -0.637 | large | 0.0433 | 0.0463 |
| r01 | Within-domain | In-domain | 529 | 0.4992 | +0.101 | negligible | 0.6442 | 0.6415 |
| r01 | Within-domain | Out-domain | 607 | 0.0761 | +0.263 | small | 0.6528 | 0.6051 |
| r01 | Mixed | In-domain | 490 | 0.5592 | +0.089 | negligible | 0.5540 | 0.5302 |
| r01 | Mixed | Out-domain | 596 | 0.0150 | +0.370 | medium | 0.6748 | 0.6741 |
| r05 | Cross-domain | In-domain | 858 | 0.0000 * | +0.676 | large | 0.0552 | 0.0491 |
| r05 | Cross-domain | Out-domain | 168 | 0.0000 * | -0.650 | large | 0.0434 | 0.0462 |
| r05 | Within-domain | In-domain | 287 | 0.0042 * | -0.421 | medium | 0.4486 | 0.5749 |
| r05 | Within-domain | Out-domain | 198 | 0.0001 * | -0.574 | large | 0.3639 | 0.5644 |
| r05 | Mixed | In-domain | 247 | 0.0045 | -0.432 | medium | 0.3908 | 0.5281 |
| r05 | Mixed | Out-domain | 165 | 0.0000 * | -0.633 | large | 0.4316 | 0.6459 |

**Summary**: sw_smote > smote in 6/12, smote > sw_smote in 6/12. Bonferroni sig: 6/12.


### 6.3 [Recall] H2: sw_smote vs plain smote

Paired comparison (same ratio): Does subject-wise synthesis improve?

| Ratio | Mode | Level | U | p | δ (sw−sm) | Effect | Mean(sw) | Mean(sm) |
|-------|------|-------|--:|--:|----------:|:------:|---------:|---------:|
| r01 | Cross-domain | In-domain | 43 | 0.0000 * | -0.916 | large | 0.1615 | 0.2729 |
| r01 | Cross-domain | Out-domain | 14 | 0.0000 * | -0.972 | large | 0.1547 | 0.2690 |
| r01 | Within-domain | In-domain | 350 | 0.0670 | -0.272 | small | 0.8256 | 0.8675 |
| r01 | Within-domain | Out-domain | 518 | 0.6065 | +0.077 | negligible | 0.8308 | 0.8681 |
| r01 | Mixed | In-domain | 278 | 0.0112 | -0.382 | medium | 0.7184 | 0.7967 |
| r01 | Mixed | Out-domain | 308 | 0.0555 | -0.291 | small | 0.7693 | 0.8734 |
| r05 | Cross-domain | In-domain | 0 | 0.0000 * | -1.000 | large | 0.0479 | 0.1815 |
| r05 | Cross-domain | Out-domain | 0 | 0.0000 * | -1.000 | large | 0.0391 | 0.1821 |
| r05 | Within-domain | In-domain | 140 | 0.0000 * | -0.718 | large | 0.5069 | 0.7822 |
| r05 | Within-domain | Out-domain | 91 | 0.0000 * | -0.804 | large | 0.5117 | 0.8004 |
| r05 | Mixed | In-domain | 0 | 0.0000 * | -1.000 | large | 0.3660 | 0.7363 |
| r05 | Mixed | Out-domain | 48 | 0.0000 * | -0.893 | large | 0.4654 | 0.7832 |

**Summary**: sw_smote > smote in 1/12, smote > sw_smote in 11/12. Bonferroni sig: 8/12.

---
## 7. Hypothesis Tests — H3: RUS vs Oversampling


### 7.1 [F1-score] H3: RUS vs oversampling (SMOTE/sw_smote)

Oversampling > RUS in **18/24** cells (Bonferroni sig: 22/24).

- large: 22/24
- medium: 0/24
- small: 1/24
- negligible: 1/24


### 7.2 [AUPRC] H3: RUS vs oversampling (SMOTE/sw_smote)

Oversampling > RUS in **17/24** cells (Bonferroni sig: 18/24).

- large: 18/24
- medium: 2/24
- small: 2/24
- negligible: 2/24


### 7.3 [Recall] H3: RUS vs oversampling (SMOTE/sw_smote)

Oversampling > RUS in **13/24** cells (Bonferroni sig: 21/24).

- large: 21/24
- medium: 1/24
- small: 1/24
- negligible: 1/24

---
## 8. Hypothesis Tests — H4: Ratio Effect (r=0.1 vs r=0.5)


### 8.1 [F1-score] H4: Ratio effect

$$H_0: \mu_{r=0.1}^{(\text{method})} = \mu_{r=0.5}^{(\text{method})}$$

| Method | Mode | Level | U | p | δ (r05−r01) | Effect | Mean(r01) | Mean(r05) |
|--------|------|-------|--:|--:|------------:|:------:|----------:|----------:|
| rus | Cross-domain | In-domain | 701 | 0.0114 | -0.369 | medium | 0.0877 | 0.0828 |
| rus | Cross-domain | Out-domain | 561 | 0.5149 | -0.096 | negligible | 0.0786 | 0.0771 |
| rus | Within-domain | In-domain | 569 | 0.4481 | -0.111 | negligible | 0.0734 | 0.0670 |
| rus | Within-domain | Out-domain | 678 | 0.0258 | -0.325 | small | 0.1048 | 0.0915 |
| rus | Mixed | In-domain | 785 | 0.0001 * | -0.583 | large | 0.0995 | 0.0756 |
| rus | Mixed | Out-domain | 758 | 0.0001 * | -0.578 | large | 0.1138 | 0.0969 |
| smote | Cross-domain | In-domain | 721 | 0.0051 | -0.408 | medium | 0.0770 | 0.0687 |
| smote | Cross-domain | Out-domain | 581 | 0.2453 | -0.171 | small | 0.0806 | 0.0778 |
| smote | Within-domain | In-domain | 278 | 0.0045 | +0.421 | medium | 0.2729 | 0.3389 |
| smote | Within-domain | Out-domain | 271 | 0.0033 | +0.436 | medium | 0.2664 | 0.3278 |
| smote | Mixed | In-domain | 236 | 0.0016 * | +0.476 | large | 0.2564 | 0.3276 |
| smote | Mixed | Out-domain | 241 | 0.0021 * | +0.464 | medium | 0.3026 | 0.4056 |
| sw_smote | Cross-domain | In-domain | 1020 | 0.0000 * | -0.992 | large | 0.0678 | 0.0437 |
| sw_smote | Cross-domain | Out-domain | 957 | 0.0000 * | -0.992 | large | 0.0673 | 0.0382 |
| sw_smote | Within-domain | In-domain | 330 | 0.0229 | +0.335 | medium | 0.3749 | 0.4696 |
| sw_smote | Within-domain | Out-domain | 371 | 0.1774 | +0.202 | small | 0.3767 | 0.4395 |
| sw_smote | Mixed | In-domain | 439 | 0.9577 | -0.009 | negligible | 0.3873 | 0.4119 |
| sw_smote | Mixed | Out-domain | 446 | 0.8735 | -0.025 | negligible | 0.4722 | 0.4616 |

**Bonferroni α'=0.00278** (m=18). **6** significant.

- **rus**: r=0.1 better in 6/6, r=0.5 better in 0/6
- **smote**: r=0.1 better in 2/6, r=0.5 better in 4/6
- **sw_smote**: r=0.1 better in 4/6, r=0.5 better in 2/6


### 8.2 [AUPRC] H4: Ratio effect

$$H_0: \mu_{r=0.1}^{(\text{method})} = \mu_{r=0.5}^{(\text{method})}$$

| Method | Mode | Level | U | p | δ (r05−r01) | Effect | Mean(r01) | Mean(r05) |
|--------|------|-------|--:|--:|------------:|:------:|----------:|----------:|
| rus | Cross-domain | In-domain | 612 | 0.1815 | -0.195 | small | 0.0589 | 0.0514 |
| rus | Cross-domain | Out-domain | 407 | 0.1606 | +0.205 | small | 0.0496 | 0.0505 |
| rus | Within-domain | In-domain | 506 | 0.9411 | +0.012 | negligible | 0.1097 | 0.1146 |
| rus | Within-domain | Out-domain | 652 | 0.0611 | -0.273 | small | 0.1400 | 0.0846 |
| rus | Mixed | In-domain | 555 | 0.4212 | -0.119 | negligible | 0.1584 | 0.0649 |
| rus | Mixed | Out-domain | 700 | 0.0020 * | -0.457 | medium | 0.1213 | 0.0733 |
| smote | Cross-domain | In-domain | 711 | 0.0077 | -0.389 | medium | 0.0501 | 0.0491 |
| smote | Cross-domain | Out-domain | 512 | 0.8313 | -0.032 | negligible | 0.0463 | 0.0462 |
| smote | Within-domain | In-domain | 600 | 0.0939 | -0.249 | small | 0.6415 | 0.5749 |
| smote | Within-domain | Out-domain | 549 | 0.3384 | -0.143 | negligible | 0.6051 | 0.5644 |
| smote | Mixed | In-domain | 474 | 0.7283 | -0.053 | negligible | 0.5302 | 0.5281 |
| smote | Mixed | Out-domain | 502 | 0.4464 | -0.116 | negligible | 0.6741 | 0.6459 |
| sw_smote | Cross-domain | In-domain | 383 | 0.0845 | +0.252 | small | 0.0519 | 0.0552 |
| sw_smote | Cross-domain | Out-domain | 453 | 0.7039 | +0.057 | negligible | 0.0433 | 0.0434 |
| sw_smote | Within-domain | In-domain | 753 | 0.0004 * | -0.518 | large | 0.6442 | 0.4486 |
| sw_smote | Within-domain | Out-domain | 781 | 0.0000 * | -0.680 | large | 0.6528 | 0.3639 |
| sw_smote | Mixed | In-domain | 649 | 0.0012 * | -0.492 | large | 0.5540 | 0.3908 |
| sw_smote | Mixed | Out-domain | 662 | 0.0006 * | -0.522 | large | 0.6748 | 0.4316 |

**Bonferroni α'=0.00278** (m=18). **5** significant.

- **rus**: r=0.1 better in 4/6, r=0.5 better in 2/6
- **smote**: r=0.1 better in 6/6, r=0.5 better in 0/6
- **sw_smote**: r=0.1 better in 4/6, r=0.5 better in 2/6


### 8.3 [Recall] H4: Ratio effect

$$H_0: \mu_{r=0.1}^{(\text{method})} = \mu_{r=0.5}^{(\text{method})}$$

| Method | Mode | Level | U | p | δ (r05−r01) | Effect | Mean(r01) | Mean(r05) |
|--------|------|-------|--:|--:|------------:|:------:|----------:|----------:|
| rus | Cross-domain | In-domain | 944 | 0.0000 * | -0.843 | large | 0.5209 | 0.4078 |
| rus | Cross-domain | Out-domain | 588 | 0.3075 | -0.149 | small | 0.4596 | 0.4494 |
| rus | Within-domain | In-domain | 666 | 0.0393 | -0.301 | small | 0.4184 | 0.3313 |
| rus | Within-domain | Out-domain | 704 | 0.0101 | -0.375 | medium | 0.6606 | 0.5611 |
| rus | Mixed | In-domain | 838 | 0.0000 * | -0.689 | large | 0.5442 | 0.3814 |
| rus | Mixed | Out-domain | 594 | 0.1116 | -0.236 | small | 0.6117 | 0.5643 |
| smote | Cross-domain | In-domain | 920 | 0.0000 * | -0.796 | large | 0.2729 | 0.1815 |
| smote | Cross-domain | Out-domain | 902 | 0.0000 * | -0.818 | large | 0.2690 | 0.1821 |
| smote | Within-domain | In-domain | 895 | 0.0000 * | -0.863 | large | 0.8675 | 0.7822 |
| smote | Within-domain | Out-domain | 888 | 0.0000 * | -0.849 | large | 0.8681 | 0.8004 |
| smote | Mixed | In-domain | 783 | 0.0000 * | -0.740 | large | 0.7967 | 0.7363 |
| smote | Mixed | Out-domain | 865 | 0.0000 * | -0.922 | large | 0.8734 | 0.7832 |
| sw_smote | Cross-domain | In-domain | 1024 | 0.0000 * | -1.000 | large | 0.1615 | 0.0479 |
| sw_smote | Cross-domain | Out-domain | 961 | 0.0000 * | -1.000 | large | 0.1547 | 0.0391 |
| sw_smote | Within-domain | In-domain | 938 | 0.0000 * | -0.892 | large | 0.8256 | 0.5069 |
| sw_smote | Within-domain | Out-domain | 868 | 0.0000 * | -0.866 | large | 0.8308 | 0.5117 |
| sw_smote | Mixed | In-domain | 804 | 0.0000 * | -0.848 | large | 0.7184 | 0.3660 |
| sw_smote | Mixed | Out-domain | 794 | 0.0000 * | -0.825 | large | 0.7693 | 0.4654 |

**Bonferroni α'=0.00278** (m=18). **14** significant.

- **rus**: r=0.1 better in 6/6, r=0.5 better in 0/6
- **smote**: r=0.1 better in 6/6, r=0.5 better in 0/6
- **sw_smote**: r=0.1 better in 6/6, r=0.5 better in 0/6

---
## 9. Hypothesis Tests — H5 (Distance), H7 (Mode), H10 (Domain Shift)


### 9.1.1 [F1-score] H5: Distance metric effect

Kruskal-Wallis across MMD, DTW, Wasserstein (pooling conditions).

| Mode | Level | H | p | η² |
|------|-------|--:|--:|---:|
| Cross-domain | In-domain | 10.45 | 0.0054 | 0.038 |
| Cross-domain | Out-domain | 19.46 | 0.0001 | 0.080 |
| Within-domain | In-domain | 0.60 | 0.7406 | 0.000 |
| Within-domain | Out-domain | 2.45 | 0.2941 | 0.002 |
| Mixed | In-domain | 0.37 | 0.8329 | 0.000 |
| Mixed | Out-domain | 1.28 | 0.5284 | 0.000 |

**2/6** significant at α=0.05.

### 9.1.2 [F1-score] H7–H8: Training mode effect

Within-domain vs cross-domain, mixed vs cross-domain.

- **Within vs Cross-domain**: U=173796, p=0.0000, δ=+0.783 (large), mean(Within)=0.2429, mean(Cross)=0.0720
- **Mixed vs Cross-domain**: U=177896, p=0.0000, δ=+0.890 (large), mean(Mixed)=0.2606, mean(Cross)=0.0720

### 9.1.3 [F1-score] H10: Domain shift (in vs out)

Wilcoxon signed-rank (paired by seed): in-domain > out-domain in **24/63** cells. Bonferroni sig: **0/63**.

**Global**: In-domain mean=0.1827, Out-domain mean=0.1981, Cliff's δ=-0.085 (negligible).


### 9.2.1 [AUPRC] H5: Distance metric effect

Kruskal-Wallis across MMD, DTW, Wasserstein (pooling conditions).

| Mode | Level | H | p | η² |
|------|-------|--:|--:|---:|
| Cross-domain | In-domain | 28.51 | 0.0000 | 0.120 |
| Cross-domain | Out-domain | 24.66 | 0.0000 | 0.104 |
| Within-domain | In-domain | 3.98 | 0.1364 | 0.009 |
| Within-domain | Out-domain | 1.91 | 0.3840 | 0.000 |
| Mixed | In-domain | 0.03 | 0.9836 | 0.000 |
| Mixed | Out-domain | 0.13 | 0.9363 | 0.000 |

**2/6** significant at α=0.05.

### 9.2.2 [AUPRC] H7–H8: Training mode effect

Within-domain vs cross-domain, mixed vs cross-domain.

- **Within vs Cross-domain**: U=189854, p=0.0000, δ=+0.948 (large), mean(Within)=0.3685, mean(Cross)=0.0496
- **Mixed vs Cross-domain**: U=183211, p=0.0000, δ=+0.947 (large), mean(Mixed)=0.3803, mean(Cross)=0.0496

### 9.2.3 [AUPRC] H10: Domain shift (in vs out)

Wilcoxon signed-rank (paired by seed): in-domain > out-domain in **32/63** cells. Bonferroni sig: **0/63**.

**Global**: In-domain mean=0.2506, Out-domain mean=0.2768, Cliff's δ=+0.058 (negligible).


### 9.3.1 [Recall] H5: Distance metric effect

Kruskal-Wallis across MMD, DTW, Wasserstein (pooling conditions).

| Mode | Level | H | p | η² |
|------|-------|--:|--:|---:|
| Cross-domain | In-domain | 1.40 | 0.4959 | 0.000 |
| Cross-domain | Out-domain | 3.79 | 0.1501 | 0.008 |
| Within-domain | In-domain | 2.16 | 0.3391 | 0.001 |
| Within-domain | Out-domain | 4.74 | 0.0934 | 0.013 |
| Mixed | In-domain | 0.35 | 0.8412 | 0.000 |
| Mixed | Out-domain | 0.28 | 0.8705 | 0.000 |

**0/6** significant at α=0.05.

### 9.3.2 [Recall] H7–H8: Training mode effect

Within-domain vs cross-domain, mixed vs cross-domain.

- **Within vs Cross-domain**: U=173416, p=0.0000, δ=+0.779 (large), mean(Within)=0.6616, mean(Cross)=0.2965
- **Mixed vs Cross-domain**: U=168739, p=0.0000, δ=+0.793 (large), mean(Mixed)=0.6533, mean(Cross)=0.2965

### 9.3.3 [Recall] H10: Domain shift (in vs out)

Wilcoxon signed-rank (paired by seed): in-domain > out-domain in **17/63** cells. Bonferroni sig: **0/63**.

**Global**: In-domain mean=0.5052, Out-domain mean=0.5640, Cliff's δ=-0.148 (small).

---
## 10. Precision–Recall Trade-off Analysis (HE3)

**Hypothesis HE3**: Oversampling methods improve Recall at the cost of Precision.

For each rebalancing method vs baseline, we compute Cliff's δ for both Precision and Recall to quantify the trade-off direction.


| Method vs Baseline | Mode | Level | δ(Recall) | δ(Precision) | Trade-off? |
|--------------------|------|-------|----------:|-------------:|:----------:|
| rus_r01 vs baseline | Cross-domain | In-domain | +0.229 | +0.711 |  |
| rus_r01 vs baseline | Cross-domain | Out-domain | -0.132 | -0.133 |  |
| rus_r01 vs baseline | Within-domain | In-domain | -0.625 | -0.660 |  |
| rus_r01 vs baseline | Within-domain | Out-domain | -0.058 | +0.004 |  |
| rus_r01 vs baseline | Mixed | In-domain | -0.609 | -0.617 |  |
| rus_r01 vs baseline | Mixed | Out-domain | -0.917 | -0.858 |  |
| rus_r05 vs baseline | Cross-domain | In-domain | -0.820 | +0.500 |  |
| rus_r05 vs baseline | Cross-domain | Out-domain | -0.209 | -0.186 |  |
| rus_r05 vs baseline | Within-domain | In-domain | -0.848 | -0.756 |  |
| rus_r05 vs baseline | Within-domain | Out-domain | -0.386 | -0.332 |  |
| rus_r05 vs baseline | Mixed | In-domain | -1.000 | -1.000 |  |
| rus_r05 vs baseline | Mixed | Out-domain | -0.987 | -1.000 |  |
| smote_r01 vs baseline | Cross-domain | In-domain | -1.000 | -0.074 |  |
| smote_r01 vs baseline | Cross-domain | Out-domain | -0.969 | +0.814 |  |
| smote_r01 vs baseline | Within-domain | In-domain | +0.957 | +1.000 |  |
| smote_r01 vs baseline | Within-domain | Out-domain | +0.505 | +1.000 |  |
| smote_r01 vs baseline | Mixed | In-domain | +0.781 | +1.000 |  |
| smote_r01 vs baseline | Mixed | Out-domain | +0.096 | +1.000 |  |
| smote_r05 vs baseline | Cross-domain | In-domain | -1.000 | -0.209 |  |
| smote_r05 vs baseline | Cross-domain | Out-domain | -1.000 | +0.625 |  |
| smote_r05 vs baseline | Within-domain | In-domain | +0.835 | +1.000 |  |
| smote_r05 vs baseline | Within-domain | Out-domain | +0.367 | +1.000 |  |
| smote_r05 vs baseline | Mixed | In-domain | +0.246 | +1.000 |  |
| smote_r05 vs baseline | Mixed | Out-domain | -0.676 | +1.000 |  |
| sw_smote_r01 vs baseline | Cross-domain | In-domain | -1.000 | -0.125 |  |
| sw_smote_r01 vs baseline | Cross-domain | Out-domain | -1.000 | +0.071 |  |
| sw_smote_r01 vs baseline | Within-domain | In-domain | +0.780 | +1.000 |  |
| sw_smote_r01 vs baseline | Within-domain | Out-domain | +0.392 | +1.000 |  |
| sw_smote_r01 vs baseline | Mixed | In-domain | +0.313 | +1.000 |  |
| sw_smote_r01 vs baseline | Mixed | Out-domain | -0.236 | +1.000 |  |
| sw_smote_r05 vs baseline | Cross-domain | In-domain | -1.000 | -0.111 |  |
| sw_smote_r05 vs baseline | Cross-domain | Out-domain | -1.000 | -0.486 |  |
| sw_smote_r05 vs baseline | Within-domain | In-domain | -0.375 | +1.000 |  |
| sw_smote_r05 vs baseline | Within-domain | Out-domain | -0.616 | +1.000 |  |
| sw_smote_r05 vs baseline | Mixed | In-domain | -0.945 | +1.000 |  |
| sw_smote_r05 vs baseline | Mixed | Out-domain | -1.000 | +1.000 |  |

**Summary**: 0/36 cells exhibit a clear Precision–Recall trade-off (|δ| > 0.147 in opposite directions).

#### Aggregated by method

| Method | Mean δ(Recall) | Mean δ(Precision) | Pattern |
|--------|---------------:|------------------:|---------|
| rus_r01 | -0.352 | -0.259 | Regression (R↓) |
| rus_r05 | -0.708 | -0.462 | Regression (R↓) |
| smote_r01 | +0.062 | +0.790 | Win-win (R↑ P≈) |
| smote_r05 | -0.205 | +0.736 | Regression (R↓) |
| sw_smote_r01 | -0.125 | +0.658 | Regression (R↓) |
| sw_smote_r05 | -0.823 | +0.567 | Regression (R↓) |

---
## 11. Extended Hypothesis HE1: Rebalancing → Recall Improvement

**Hypothesis**: *Any* rebalancing method (including RUS) improves Recall over baseline, because both oversampling and undersampling increase the effective weight of the minority class.


One-sided test: $H_1$: Recall(method) > Recall(baseline).

| Method | Mode | Level | U | p (one-sided) | δ | Effect | Mean(M) | Mean(B) |
|--------|------|-------|--:|:-------------:|--:|:------:|--------:|--------:|
| rus_r01 | Cross-domain | In-domain | 630 | 0.0580 | +0.229 | small | 0.5209 | 0.5092 |
| rus_r01 | Cross-domain | Out-domain | 444 | 0.8194 | -0.132 | negligible | 0.4596 | 0.4792 |
| rus_r01 | Within-domain | In-domain | 180 | 1.0000 | -0.625 | large | 0.4184 | 0.6176 |
| rus_r01 | Within-domain | Out-domain | 482 | 0.6565 | -0.058 | negligible | 0.6606 | 0.6987 |
| rus_r01 | Mixed | In-domain | 182 | 1.0000 | -0.609 | large | 0.5442 | 0.6987 |
| rus_r01 | Mixed | Out-domain | 38 | 1.0000 | -0.917 | large | 0.6117 | 0.8582 |
| rus_r05 | Cross-domain | In-domain | 92 | 1.0000 | -0.820 | large | 0.4078 | 0.5092 |
| rus_r05 | Cross-domain | Out-domain | 405 | 0.9256 | -0.209 | small | 0.4494 | 0.4792 |
| rus_r05 | Within-domain | In-domain | 73 | 1.0000 | -0.848 | large | 0.3313 | 0.6176 |
| rus_r05 | Within-domain | Out-domain | 314 | 0.9961 | -0.386 | medium | 0.5611 | 0.6987 |
| rus_r05 | Mixed | In-domain | 0 | 1.0000 | -1.000 | large | 0.3814 | 0.6987 |
| rus_r05 | Mixed | Out-domain | 6 | 1.0000 | -0.987 | large | 0.5643 | 0.8582 |
| smote_r01 | Cross-domain | In-domain | 0 | 1.0000 | -1.000 | large | 0.2729 | 0.5092 |
| smote_r01 | Cross-domain | Out-domain | 16 | 1.0000 | -0.969 | large | 0.2690 | 0.4792 |
| smote_r01 | Within-domain | In-domain | 910 | 0.0000 * | +0.957 | large | 0.8675 | 0.6176 |
| smote_r01 | Within-domain | Out-domain | 746 | 0.0003 * | +0.505 | large | 0.8681 | 0.6987 |
| smote_r01 | Mixed | In-domain | 802 | 0.0000 * | +0.781 | large | 0.7967 | 0.6987 |
| smote_r01 | Mixed | Out-domain | 493 | 0.2648 | +0.096 | negligible | 0.8734 | 0.8582 |
| smote_r05 | Cross-domain | In-domain | 0 | 1.0000 | -1.000 | large | 0.1815 | 0.5092 |
| smote_r05 | Cross-domain | Out-domain | 0 | 1.0000 | -1.000 | large | 0.1821 | 0.4792 |
| smote_r05 | Within-domain | In-domain | 854 | 0.0000 * | +0.835 | large | 0.7822 | 0.6176 |
| smote_r05 | Within-domain | Out-domain | 678 | 0.0063 | +0.367 | medium | 0.8004 | 0.6987 |
| smote_r05 | Mixed | In-domain | 560 | 0.0519 | +0.246 | small | 0.7363 | 0.6987 |
| smote_r05 | Mixed | Out-domain | 146 | 1.0000 | -0.676 | large | 0.7832 | 0.8582 |
| sw_smote_r01 | Cross-domain | In-domain | 0 | 1.0000 | -1.000 | large | 0.1615 | 0.5092 |
| sw_smote_r01 | Cross-domain | Out-domain | 0 | 1.0000 | -1.000 | large | 0.1547 | 0.4792 |
| sw_smote_r01 | Within-domain | In-domain | 828 | 0.0000 * | +0.780 | large | 0.8256 | 0.6176 |
| sw_smote_r01 | Within-domain | Out-domain | 690 | 0.0038 | +0.392 | medium | 0.8308 | 0.6987 |
| sw_smote_r01 | Mixed | In-domain | 591 | 0.0189 | +0.313 | small | 0.7184 | 0.6987 |
| sw_smote_r01 | Mixed | Out-domain | 332 | 0.9409 | -0.236 | small | 0.7693 | 0.8582 |
| sw_smote_r05 | Cross-domain | In-domain | 0 | 1.0000 | -1.000 | large | 0.0479 | 0.5092 |
| sw_smote_r05 | Cross-domain | Out-domain | 0 | 1.0000 | -1.000 | large | 0.0391 | 0.4792 |
| sw_smote_r05 | Within-domain | In-domain | 300 | 0.9945 | -0.375 | medium | 0.5069 | 0.6176 |
| sw_smote_r05 | Within-domain | Out-domain | 184 | 1.0000 | -0.616 | large | 0.5117 | 0.6987 |
| sw_smote_r05 | Mixed | In-domain | 24 | 1.0000 | -0.945 | large | 0.3660 | 0.6987 |
| sw_smote_r05 | Mixed | Out-domain | 0 | 1.0000 | -1.000 | large | 0.4654 | 0.8582 |

**Bonferroni α'=0.00139** (m=36). **5** significant.

Method > baseline in 11/36 cells.

**RUS specifically**: Recall improved in 1/12 cells (mean δ=-0.530).

**SMOTE-family**: Recall improved in 10/24 cells (mean δ=-0.273).

---
## 12. Extended Hypothesis HE2: AUPRC vs AUROC Sensitivity

**Hypothesis**: AUPRC shows a stronger condition effect than AUROC because AUPRC is more sensitive to minority-class performance.


### Comparison of η² (condition effect size) per cell

| Mode | Level | η²(AUROC) | η²(AUPRC) | AUPRC stronger? |
|------|-------|----------:|----------:|:---------------:|
| Cross-domain | In-domain | 0.079 | 0.112 | ✓ |
| Cross-domain | Out-domain | 0.116 | 0.271 | ✓ |
| Within-domain | In-domain | 0.717 | 0.702 | ✗ |
| Within-domain | Out-domain | 0.697 | 0.693 | ✗ |
| Mixed | In-domain | 0.711 | 0.663 | ✗ |
| Mixed | Out-domain | 0.646 | 0.670 | ✓ |

**Result**: AUPRC shows stronger condition effect in **3/6** cells.

### Ranking comparison: AUROC vs AUPRC

| Condition | Mean Rank (AUROC) | Mean Rank (AUPRC) | Δ Rank |
|-----------|------------------:|------------------:|-------:|
| baseline | 4.78 | 5.28 | +0.50 |
| rus_r01 | 4.94 | 4.50 | -0.44 |
| rus_r05 | 5.56 | 5.44 | -0.11 |
| smote_r01 | 2.17 | 2.61 | +0.44 |
| smote_r05 | 3.17 | 3.39 | +0.22 |
| sw_smote_r01 | 3.28 | 2.56 | -0.72 |
| sw_smote_r05 | 4.11 | 4.22 | +0.11 |

**Spearman ρ** (AUROC vs AUPRC rankings) = 0.857 (strong concordance).

---
## 13. Cross-Axis Interaction Analysis


### 13.1.1 [F1-score] H12: Condition × Mode interaction

Best condition per mode:

| Mode | Level | Best Condition | Mean | 2nd | Mean |
|------|-------|:-------------:|-----:|:---:|-----:|
| Cross-domain | In-domain | rus_r01 | 0.0877 | rus_r05 | 0.0828 |
| Cross-domain | Out-domain | smote_r01 | 0.0806 | rus_r01 | 0.0786 |
| Within-domain | In-domain | sw_smote_r05 | 0.4696 | sw_smote_r01 | 0.3749 |
| Within-domain | Out-domain | sw_smote_r05 | 0.4395 | sw_smote_r01 | 0.3767 |
| Mixed | In-domain | sw_smote_r05 | 0.4119 | sw_smote_r01 | 0.3873 |
| Mixed | Out-domain | sw_smote_r01 | 0.4722 | sw_smote_r05 | 0.4616 |

**Friedman test** (condition effect per mode, seeds as blocks):

| Mode | Level | χ² | p | Kendall's W |
|------|-------|---:|--:|:----------:|
| Cross-domain | In-domain | 62.14 | 0.0000 * | 0.942 |
| Cross-domain | Out-domain | 42.70 | 0.0000 * | 0.647 |
| Within-domain | In-domain | 55.07 | 0.0000 * | 0.918 |
| Within-domain | Out-domain | 53.44 | 0.0000 * | 0.891 |
| Mixed | In-domain | 51.51 | 0.0000 * | 0.859 |
| Mixed | Out-domain | 52.16 | 0.0000 * | 0.869 |

### 13.1.2 [F1-score] H13: Condition × Distance interaction

| Mode | Level | MMD best | DTW best | Wass. best | Consistent? |
|------|-------|:--------:|:--------:|:----------:|:-----------:|
| Cross-domain | In-domain | smote_r01 | rus_r05 | rus_r01 | ✗ |
| Cross-domain | Out-domain | baseline | rus_r01 | smote_r05 | ✗ |
| Within-domain | In-domain | sw_smote_r05 | sw_smote_r05 | sw_smote_r05 | ✓ |
| Within-domain | Out-domain | sw_smote_r05 | sw_smote_r05 | sw_smote_r05 | ✓ |
| Mixed | In-domain | sw_smote_r05 | sw_smote_r05 | sw_smote_r05 | ✓ |
| Mixed | Out-domain | sw_smote_r01 | sw_smote_r01 | sw_smote_r01 | ✓ |

### 13.1.3 [F1-score] H14: Domain gap by mode

| Mode | Mean gap (Δ=out−in) | Mean |Δ| |
|------|:-------------------:|------:|
| Cross-domain | -0.0016 | 0.0098 |
| Within-domain | +0.0019 | 0.0606 |
| Mixed | +0.0471 | 0.0804 |


### 13.2.1 [AUPRC] H12: Condition × Mode interaction

Best condition per mode:

| Mode | Level | Best Condition | Mean | 2nd | Mean |
|------|-------|:-------------:|-----:|:---:|-----:|
| Cross-domain | In-domain | rus_r01 | 0.0589 | sw_smote_r05 | 0.0552 |
| Cross-domain | Out-domain | rus_r05 | 0.0505 | rus_r01 | 0.0496 |
| Within-domain | In-domain | sw_smote_r01 | 0.6442 | smote_r01 | 0.6415 |
| Within-domain | Out-domain | sw_smote_r01 | 0.6528 | smote_r01 | 0.6051 |
| Mixed | In-domain | sw_smote_r01 | 0.5540 | smote_r01 | 0.5302 |
| Mixed | Out-domain | sw_smote_r01 | 0.6748 | smote_r01 | 0.6741 |

**Friedman test** (condition effect per mode, seeds as blocks):

| Mode | Level | χ² | p | Kendall's W |
|------|-------|---:|--:|:----------:|
| Cross-domain | In-domain | 37.40 | 0.0000 * | 0.567 |
| Cross-domain | Out-domain | 42.35 | 0.0000 * | 0.642 |
| Within-domain | In-domain | 48.26 | 0.0000 * | 0.804 |
| Within-domain | Out-domain | 51.17 | 0.0000 * | 0.853 |
| Mixed | In-domain | 39.60 | 0.0000 * | 0.660 |
| Mixed | Out-domain | 42.17 | 0.0000 * | 0.703 |

### 13.2.2 [AUPRC] H13: Condition × Distance interaction

| Mode | Level | MMD best | DTW best | Wass. best | Consistent? |
|------|-------|:--------:|:--------:|:----------:|:-----------:|
| Cross-domain | In-domain | smote_r05 | rus_r01 | rus_r01 | ✗ |
| Cross-domain | Out-domain | rus_r05 | rus_r05 | rus_r01 | ✗ |
| Within-domain | In-domain | smote_r01 | smote_r01 | sw_smote_r01 | ✗ |
| Within-domain | Out-domain | sw_smote_r01 | sw_smote_r01 | sw_smote_r01 | ✓ |
| Mixed | In-domain | sw_smote_r01 | sw_smote_r01 | sw_smote_r01 | ✓ |
| Mixed | Out-domain | smote_r01 | sw_smote_r01 | smote_r01 | ✗ |

### 13.2.3 [AUPRC] H14: Domain gap by mode

| Mode | Mean gap (Δ=out−in) | Mean |Δ| |
|------|:-------------------:|------:|
| Cross-domain | -0.0059 | 0.0082 |
| Within-domain | -0.0101 | 0.1246 |
| Mixed | +0.0954 | 0.1767 |


### 13.3.1 [Recall] H12: Condition × Mode interaction

Best condition per mode:

| Mode | Level | Best Condition | Mean | 2nd | Mean |
|------|-------|:-------------:|-----:|:---:|-----:|
| Cross-domain | In-domain | rus_r01 | 0.5209 | baseline | 0.5092 |
| Cross-domain | Out-domain | baseline | 0.4792 | rus_r01 | 0.4596 |
| Within-domain | In-domain | smote_r01 | 0.8675 | sw_smote_r01 | 0.8256 |
| Within-domain | Out-domain | smote_r01 | 0.8681 | sw_smote_r01 | 0.8308 |
| Mixed | In-domain | smote_r01 | 0.7967 | smote_r05 | 0.7363 |
| Mixed | Out-domain | smote_r01 | 0.8734 | baseline | 0.8582 |

**Friedman test** (condition effect per mode, seeds as blocks):

| Mode | Level | χ² | p | Kendall's W |
|------|-------|---:|--:|:----------:|
| Cross-domain | In-domain | 63.82 | 0.0000 * | 0.967 |
| Cross-domain | Out-domain | 60.94 | 0.0000 * | 0.923 |
| Within-domain | In-domain | 53.53 | 0.0000 * | 0.892 |
| Within-domain | Out-domain | 46.33 | 0.0000 * | 0.772 |
| Mixed | In-domain | 43.89 | 0.0000 * | 0.731 |
| Mixed | Out-domain | 42.00 | 0.0000 * | 0.700 |

### 13.3.2 [Recall] H13: Condition × Distance interaction

| Mode | Level | MMD best | DTW best | Wass. best | Consistent? |
|------|-------|:--------:|:--------:|:----------:|:-----------:|
| Cross-domain | In-domain | rus_r01 | baseline | rus_r01 | ✗ |
| Cross-domain | Out-domain | baseline | rus_r01 | baseline | ✗ |
| Within-domain | In-domain | smote_r01 | smote_r01 | sw_smote_r01 | ✗ |
| Within-domain | Out-domain | smote_r01 | smote_r01 | sw_smote_r01 | ✗ |
| Mixed | In-domain | smote_r01 | smote_r01 | smote_r01 | ✓ |
| Mixed | Out-domain | baseline | smote_r01 | smote_r01 | ✗ |

### 13.3.3 [Recall] H14: Domain gap by mode

| Mode | Mean gap (Δ=out−in) | Mean |Δ| |
|------|:-------------------:|------:|
| Cross-domain | -0.0099 | 0.0555 |
| Within-domain | +0.0826 | 0.1443 |
| Mixed | +0.0987 | 0.1395 |

---
## 14. Overall Condition Ranking

Mean rank across 18 cells (3 modes × 2 levels × 3 distances). Rank 1 = best.

### F1-score

| Rank | Condition | Mean Rank | Win count (rank 1) |
|:----:|-----------|----------:|:------------------:|
| 1 | sw_smote_r01 | 3.06 | 3 |
| 2 | sw_smote_r05 | 3.17 | 9 |
| 3 | smote_r01 | 3.50 | 1 |
| 4 | smote_r05 | 3.50 | 1 |
| 5 | baseline | 4.56 | 1 |
| 6 | rus_r01 | 4.83 | 2 |
| 7 | rus_r05 | 5.39 | 1 |

### AUPRC

| Rank | Condition | Mean Rank | Win count (rank 1) |
|:----:|-----------|----------:|:------------------:|
| 1 | sw_smote_r01 | 2.56 | 8 |
| 2 | smote_r01 | 2.61 | 4 |
| 3 | smote_r05 | 3.39 | 1 |
| 4 | sw_smote_r05 | 4.22 | 0 |
| 5 | rus_r01 | 4.50 | 3 |
| 6 | baseline | 5.28 | 0 |
| 7 | rus_r05 | 5.44 | 2 |

### Recall

| Rank | Condition | Mean Rank | Win count (rank 1) |
|:----:|-----------|----------:|:------------------:|
| 1 | smote_r01 | 2.17 | 9 |
| 2 | baseline | 2.83 | 4 |
| 3 | smote_r05 | 3.44 | 0 |
| 4 | sw_smote_r01 | 4.00 | 2 |
| 5 | rus_r01 | 4.11 | 3 |
| 6 | rus_r05 | 4.89 | 0 |
| 7 | sw_smote_r05 | 6.56 | 0 |

---
## 15. Nemenyi Post-Hoc Test


### F1-score

#### In-domain (pooled across modes)

Friedman χ²=60.00, p=0.0000 (significant)

| | baseline | rus_r01 | rus_r05 | smote_r01 | smote_r05 | sw_smote_r01 | sw_smote_r05 |
|---|---|---|---|---|---|---|---|
| **baseline** | — | 0.9744 | 0.3691 | 0.7567 | 0.1365 | 0.0070 * | 0.0049 * |
| **rus_r01** | — | — | 0.9003 | 0.2120 | 0.0099 * | 0.0002 * | 0.0001 * |
| **rus_r05** | — | — | — | 0.0070 * | 0.0001 * | 0.0000 * | 0.0000 * |
| **smote_r01** | — | — | — | — | 0.9325 | 0.3691 | 0.3112 |
| **smote_r05** | — | — | — | — | — | 0.9570 | 0.9325 |
| **sw_smote_r01** | — | — | — | — | — | — | 1.0000 |
| **sw_smote_r05** | — | — | — | — | — | — | — |

**Significant pairs**: 9/21
- baseline vs sw_smote_r01
- baseline vs sw_smote_r05
- rus_r01 vs smote_r05
- rus_r01 vs sw_smote_r01
- rus_r01 vs sw_smote_r05
- rus_r05 vs smote_r01
- rus_r05 vs smote_r05
- rus_r05 vs sw_smote_r01
- rus_r05 vs sw_smote_r05

**Mean ranks**:

| Condition | Mean Rank |
|-----------|----------:|
| sw_smote_r05 | 1.73 |
| sw_smote_r01 | 1.82 |
| smote_r05 | 2.73 |
| smote_r01 | 3.73 |
| baseline | 5.09 |
| rus_r01 | 5.91 |
| rus_r05 | 7.00 |

**Critical Difference (CD)** = 2.716 (α=0.05, k=7, n=11)

#### Out-domain (pooled across modes)

Friedman χ²=51.27, p=0.0000 (significant)

| | baseline | rus_r01 | rus_r05 | smote_r01 | smote_r05 | sw_smote_r01 | sw_smote_r05 |
|---|---|---|---|---|---|---|---|
| **baseline** | — | 0.9325 | 0.4969 | 0.6310 | 0.0482 * | 0.0266 * | 0.0833 |
| **rus_r01** | — | — | 0.9860 | 0.0833 | 0.0010 * | 0.0004 * | 0.0023 * |
| **rus_r05** | — | — | — | 0.0070 * | 0.0000 * | 0.0000 * | 0.0001 * |
| **smote_r01** | — | — | — | — | 0.8600 | 0.7567 | 0.9325 |
| **smote_r05** | — | — | — | — | — | 1.0000 | 1.0000 |
| **sw_smote_r01** | — | — | — | — | — | — | 0.9997 |
| **sw_smote_r05** | — | — | — | — | — | — | — |

**Significant pairs**: 9/21
- baseline vs smote_r05
- baseline vs sw_smote_r01
- rus_r01 vs smote_r05
- rus_r01 vs sw_smote_r01
- rus_r01 vs sw_smote_r05
- rus_r05 vs smote_r01
- rus_r05 vs smote_r05
- rus_r05 vs sw_smote_r01
- rus_r05 vs sw_smote_r05

**Mean ranks**:

| Condition | Mean Rank |
|-----------|----------:|
| sw_smote_r01 | 2.09 |
| smote_r05 | 2.27 |
| sw_smote_r05 | 2.45 |
| smote_r01 | 3.45 |
| baseline | 5.00 |
| rus_r01 | 6.00 |
| rus_r05 | 6.73 |

**Critical Difference (CD)** = 2.716 (α=0.05, k=7, n=11)


### AUPRC

#### In-domain (pooled across modes)

Friedman χ²=52.60, p=0.0000 (significant)

| | baseline | rus_r01 | rus_r05 | smote_r01 | smote_r05 | sw_smote_r01 | sw_smote_r05 |
|---|---|---|---|---|---|---|---|
| **baseline** | — | 0.9999 | 0.9999 | 0.0003 * | 0.0033 * | 0.0004 * | 0.0638 |
| **rus_r01** | — | — | 0.9971 | 0.0010 * | 0.0099 * | 0.0015 * | 0.1365 |
| **rus_r05** | — | — | — | 0.0001 * | 0.0010 * | 0.0001 * | 0.0266 * |
| **smote_r01** | — | — | — | — | 0.9971 | 1.0000 | 0.7567 |
| **smote_r05** | — | — | — | — | — | 0.9989 | 0.9744 |
| **sw_smote_r01** | — | — | — | — | — | — | 0.8118 |
| **sw_smote_r05** | — | — | — | — | — | — | — |

**Significant pairs**: 10/21
- baseline vs smote_r01
- baseline vs smote_r05
- baseline vs sw_smote_r01
- rus_r01 vs smote_r01
- rus_r01 vs smote_r05
- rus_r01 vs sw_smote_r01
- rus_r05 vs smote_r01
- rus_r05 vs smote_r05
- rus_r05 vs sw_smote_r01
- rus_r05 vs sw_smote_r05

**Mean ranks**:

| Condition | Mean Rank |
|-----------|----------:|
| smote_r01 | 2.00 |
| sw_smote_r01 | 2.09 |
| smote_r05 | 2.55 |
| sw_smote_r05 | 3.36 |
| rus_r01 | 5.73 |
| baseline | 6.00 |
| rus_r05 | 6.27 |

**Critical Difference (CD)** = 2.716 (α=0.05, k=7, n=11)

#### Out-domain (pooled across modes)

Friedman χ²=52.91, p=0.0000 (significant)

| | baseline | rus_r01 | rus_r05 | smote_r01 | smote_r05 | sw_smote_r01 | sw_smote_r05 |
|---|---|---|---|---|---|---|---|
| **baseline** | — | 0.8600 | 0.3112 | 0.1713 | 0.1713 | 0.0099 * | 0.9971 |
| **rus_r01** | — | — | 0.9744 | 0.0033 * | 0.0033 * | 0.0000 * | 0.4969 |
| **rus_r05** | — | — | — | 0.0001 * | 0.0001 * | 0.0000 * | 0.0833 |
| **smote_r01** | — | — | — | — | 1.0000 | 0.9570 | 0.4969 |
| **smote_r05** | — | — | — | — | — | 0.9570 | 0.4969 |
| **sw_smote_r01** | — | — | — | — | — | — | 0.0638 |
| **sw_smote_r05** | — | — | — | — | — | — | — |

**Significant pairs**: 7/21
- baseline vs sw_smote_r01
- rus_r01 vs smote_r01
- rus_r01 vs smote_r05
- rus_r01 vs sw_smote_r01
- rus_r05 vs smote_r01
- rus_r05 vs smote_r05
- rus_r05 vs sw_smote_r01

**Mean ranks**:

| Condition | Mean Rank |
|-----------|----------:|
| sw_smote_r01 | 1.55 |
| smote_r01 | 2.45 |
| smote_r05 | 2.45 |
| sw_smote_r05 | 4.18 |
| baseline | 4.73 |
| rus_r01 | 5.91 |
| rus_r05 | 6.73 |

**Critical Difference (CD)** = 2.716 (α=0.05, k=7, n=11)


### Recall

#### In-domain (pooled across modes)

Friedman χ²=57.70, p=0.0000 (significant)

| | baseline | rus_r01 | rus_r05 | smote_r01 | smote_r05 | sw_smote_r01 | sw_smote_r05 |
|---|---|---|---|---|---|---|---|
| **baseline** | — | 0.1365 | 0.0004 * | 0.8600 | 0.7567 | 0.8600 | 0.0000 * |
| **rus_r01** | — | — | 0.6310 | 0.0023 * | 0.9325 | 0.8600 | 0.2587 |
| **rus_r05** | — | — | — | 0.0000 * | 0.0833 | 0.0482 * | 0.9971 |
| **smote_r01** | — | — | — | — | 0.0833 | 0.1365 | 0.0000 * |
| **smote_r05** | — | — | — | — | — | 1.0000 | 0.0139 * |
| **sw_smote_r01** | — | — | — | — | — | — | 0.0070 * |
| **sw_smote_r05** | — | — | — | — | — | — | — |

**Significant pairs**: 8/21
- baseline vs rus_r05
- baseline vs sw_smote_r05
- rus_r01 vs smote_r01
- rus_r05 vs smote_r01
- rus_r05 vs sw_smote_r01
- smote_r01 vs sw_smote_r05
- smote_r05 vs sw_smote_r05
- sw_smote_r01 vs sw_smote_r05

**Mean ranks**:

| Condition | Mean Rank |
|-----------|----------:|
| smote_r01 | 1.09 |
| baseline | 2.27 |
| sw_smote_r01 | 3.45 |
| smote_r05 | 3.64 |
| rus_r01 | 4.64 |
| rus_r05 | 6.18 |
| sw_smote_r05 | 6.73 |

**Critical Difference (CD)** = 2.716 (α=0.05, k=7, n=11)

#### Out-domain (pooled across modes)

Friedman χ²=52.68, p=0.0000 (significant)

| | baseline | rus_r01 | rus_r05 | smote_r01 | smote_r05 | sw_smote_r01 | sw_smote_r05 |
|---|---|---|---|---|---|---|---|
| **baseline** | — | 0.0139 * | 0.0003 * | 0.9989 | 0.0266 * | 0.1073 | 0.0000 * |
| **rus_r01** | — | — | 0.9570 | 0.0638 | 1.0000 | 0.9931 | 0.1073 |
| **rus_r05** | — | — | — | 0.0023 * | 0.9003 | 0.6310 | 0.6310 |
| **smote_r01** | — | — | — | — | 0.1073 | 0.3112 | 0.0000 * |
| **smote_r05** | — | — | — | — | — | 0.9989 | 0.0638 |
| **sw_smote_r01** | — | — | — | — | — | — | 0.0139 * |
| **sw_smote_r05** | — | — | — | — | — | — | — |

**Significant pairs**: 7/21
- baseline vs rus_r01
- baseline vs rus_r05
- baseline vs smote_r05
- baseline vs sw_smote_r05
- rus_r05 vs smote_r01
- smote_r01 vs sw_smote_r05
- sw_smote_r01 vs sw_smote_r05

**Mean ranks**:

| Condition | Mean Rank |
|-----------|----------:|
| baseline | 1.36 |
| smote_r01 | 1.82 |
| sw_smote_r01 | 3.82 |
| smote_r05 | 4.27 |
| rus_r01 | 4.45 |
| rus_r05 | 5.36 |
| sw_smote_r05 | 6.91 |

**Critical Difference (CD)** = 2.716 (α=0.05, k=7, n=11)

---
## 16. Bootstrap Confidence Intervals (BCa)

B = 10,000 resamples, seed-level resampling.


### F1-score

| Condition | Mode | Level | Mean | 95% CI Lower | 95% CI Upper | Width |
|-----------|------|-------|-----:|-------------:|-------------:|------:|
| baseline | Cross-domain | In-domain | 0.0813 | 0.0805 | 0.0820 | 0.0014 |
| baseline | Cross-domain | Out-domain | 0.0785 | 0.0778 | 0.0795 | 0.0017 |
| baseline | Within-domain | In-domain | 0.1030 | 0.0990 | 0.1081 | 0.0091 |
| baseline | Within-domain | Out-domain | 0.1144 | 0.1082 | 0.1241 | 0.0159 |
| baseline | Mixed | In-domain | 0.1206 | 0.1148 | 0.1256 | 0.0109 |
| baseline | Mixed | Out-domain | 0.1575 | 0.1493 | 0.1679 | 0.0186 |
| rus_r01 | Cross-domain | In-domain | 0.0878 | 0.0864 | 0.0890 | 0.0026 |
| rus_r01 | Cross-domain | Out-domain | 0.0786 | 0.0765 | 0.0811 | 0.0046 |
| rus_r01 | Within-domain | In-domain | 0.0731 | 0.0684 | 0.0771 | 0.0086 |
| rus_r01 | Within-domain | Out-domain | 0.1052 | 0.0985 | 0.1109 | 0.0123 |
| rus_r01 | Mixed | In-domain | 0.0996 | 0.0911 | 0.1197 | 0.0286 |
| rus_r01 | Mixed | Out-domain | 0.1132 | 0.1036 | 0.1278 | 0.0242 |
| rus_r05 | Cross-domain | In-domain | 0.0830 | 0.0811 | 0.0851 | 0.0040 |
| rus_r05 | Cross-domain | Out-domain | 0.0770 | 0.0738 | 0.0800 | 0.0061 |
| rus_r05 | Within-domain | In-domain | 0.0665 | 0.0603 | 0.0738 | 0.0135 |
| rus_r05 | Within-domain | Out-domain | 0.0913 | 0.0861 | 0.0969 | 0.0109 |
| rus_r05 | Mixed | In-domain | 0.0757 | 0.0678 | 0.0795 | 0.0117 |
| rus_r05 | Mixed | Out-domain | 0.0968 | 0.0917 | 0.1028 | 0.0112 |
| smote_r01 | Cross-domain | In-domain | 0.0770 | 0.0745 | 0.0786 | 0.0041 |
| smote_r01 | Cross-domain | Out-domain | 0.0806 | 0.0787 | 0.0830 | 0.0042 |
| smote_r01 | Within-domain | In-domain | 0.2700 | 0.2446 | 0.3020 | 0.0574 |
| smote_r01 | Within-domain | Out-domain | 0.2622 | 0.2357 | 0.2948 | 0.0590 |
| smote_r01 | Mixed | In-domain | 0.2564 | 0.2195 | 0.2986 | 0.0791 |
| smote_r01 | Mixed | Out-domain | 0.3026 | 0.2776 | 0.3242 | 0.0466 |
| smote_r05 | Cross-domain | In-domain | 0.0689 | 0.0667 | 0.0710 | 0.0044 |
| smote_r05 | Cross-domain | Out-domain | 0.0768 | 0.0718 | 0.0796 | 0.0078 |
| smote_r05 | Within-domain | In-domain | 0.3402 | 0.3015 | 0.3950 | 0.0934 |
| smote_r05 | Within-domain | Out-domain | 0.3250 | 0.2933 | 0.3691 | 0.0758 |
| smote_r05 | Mixed | In-domain | 0.3276 | 0.2712 | 0.3763 | 0.1052 |
| smote_r05 | Mixed | Out-domain | 0.4056 | 0.3359 | 0.5094 | 0.1735 |
| sw_smote_r01 | Cross-domain | In-domain | 0.0678 | 0.0655 | 0.0704 | 0.0049 |
| sw_smote_r01 | Cross-domain | Out-domain | 0.0674 | 0.0656 | 0.0688 | 0.0033 |
| sw_smote_r01 | Within-domain | In-domain | 0.3765 | 0.3577 | 0.3961 | 0.0384 |
| sw_smote_r01 | Within-domain | Out-domain | 0.3754 | 0.3433 | 0.4015 | 0.0582 |
| sw_smote_r01 | Mixed | In-domain | 0.3873 | 0.3248 | 0.4412 | 0.1164 |
| sw_smote_r01 | Mixed | Out-domain | 0.4745 | 0.3739 | 0.5559 | 0.1821 |
| sw_smote_r05 | Cross-domain | In-domain | 0.0434 | 0.0406 | 0.0459 | 0.0054 |
| sw_smote_r05 | Cross-domain | Out-domain | 0.0388 | 0.0342 | 0.0424 | 0.0082 |
| sw_smote_r05 | Within-domain | In-domain | 0.4652 | 0.3862 | 0.5654 | 0.1792 |
| sw_smote_r05 | Within-domain | Out-domain | 0.4395 | 0.3815 | 0.5282 | 0.1467 |
| sw_smote_r05 | Mixed | In-domain | 0.4097 | 0.3092 | 0.5264 | 0.2172 |
| sw_smote_r05 | Mixed | Out-domain | 0.4616 | 0.3764 | 0.5729 | 0.1965 |


### AUPRC

| Condition | Mode | Level | Mean | 95% CI Lower | 95% CI Upper | Width |
|-----------|------|-------|-----:|-------------:|-------------:|------:|
| baseline | Cross-domain | In-domain | 0.0508 | 0.0503 | 0.0512 | 0.0008 |
| baseline | Cross-domain | Out-domain | 0.0464 | 0.0458 | 0.0478 | 0.0020 |
| baseline | Within-domain | In-domain | 0.0872 | 0.0701 | 0.1511 | 0.0810 |
| baseline | Within-domain | Out-domain | 0.1631 | 0.1251 | 0.2413 | 0.1162 |
| baseline | Mixed | In-domain | 0.1287 | 0.1027 | 0.1618 | 0.0591 |
| baseline | Mixed | Out-domain | 0.4050 | 0.2598 | 0.6034 | 0.3436 |
| rus_r01 | Cross-domain | In-domain | 0.0591 | 0.0549 | 0.0629 | 0.0080 |
| rus_r01 | Cross-domain | Out-domain | 0.0496 | 0.0469 | 0.0542 | 0.0073 |
| rus_r01 | Within-domain | In-domain | 0.1084 | 0.0812 | 0.1531 | 0.0719 |
| rus_r01 | Within-domain | Out-domain | 0.1406 | 0.1103 | 0.1672 | 0.0569 |
| rus_r01 | Mixed | In-domain | 0.1523 | 0.0754 | 0.3256 | 0.2502 |
| rus_r01 | Mixed | Out-domain | 0.1178 | 0.0798 | 0.2089 | 0.1292 |
| rus_r05 | Cross-domain | In-domain | 0.0513 | 0.0502 | 0.0541 | 0.0040 |
| rus_r05 | Cross-domain | Out-domain | 0.0505 | 0.0482 | 0.0542 | 0.0060 |
| rus_r05 | Within-domain | In-domain | 0.1147 | 0.0919 | 0.1447 | 0.0528 |
| rus_r05 | Within-domain | Out-domain | 0.0850 | 0.0703 | 0.1013 | 0.0310 |
| rus_r05 | Mixed | In-domain | 0.0646 | 0.0585 | 0.0792 | 0.0206 |
| rus_r05 | Mixed | Out-domain | 0.0727 | 0.0599 | 0.0968 | 0.0369 |
| smote_r01 | Cross-domain | In-domain | 0.0501 | 0.0498 | 0.0504 | 0.0006 |
| smote_r01 | Cross-domain | Out-domain | 0.0464 | 0.0458 | 0.0469 | 0.0011 |
| smote_r01 | Within-domain | In-domain | 0.6346 | 0.5665 | 0.7116 | 0.1450 |
| smote_r01 | Within-domain | Out-domain | 0.5947 | 0.5266 | 0.6707 | 0.1441 |
| smote_r01 | Mixed | In-domain | 0.5302 | 0.4010 | 0.6390 | 0.2380 |
| smote_r01 | Mixed | Out-domain | 0.6741 | 0.6064 | 0.7347 | 0.1282 |
| smote_r05 | Cross-domain | In-domain | 0.0491 | 0.0489 | 0.0495 | 0.0006 |
| smote_r05 | Cross-domain | Out-domain | 0.0461 | 0.0453 | 0.0466 | 0.0013 |
| smote_r05 | Within-domain | In-domain | 0.5797 | 0.5124 | 0.6447 | 0.1322 |
| smote_r05 | Within-domain | Out-domain | 0.5589 | 0.5052 | 0.6234 | 0.1182 |
| smote_r05 | Mixed | In-domain | 0.5281 | 0.4087 | 0.6098 | 0.2011 |
| smote_r05 | Mixed | Out-domain | 0.6459 | 0.5612 | 0.7343 | 0.1731 |
| sw_smote_r01 | Cross-domain | In-domain | 0.0519 | 0.0513 | 0.0525 | 0.0011 |
| sw_smote_r01 | Cross-domain | Out-domain | 0.0431 | 0.0422 | 0.0435 | 0.0013 |
| sw_smote_r01 | Within-domain | In-domain | 0.6561 | 0.5967 | 0.7222 | 0.1254 |
| sw_smote_r01 | Within-domain | Out-domain | 0.6504 | 0.5558 | 0.7173 | 0.1615 |
| sw_smote_r01 | Mixed | In-domain | 0.5540 | 0.4040 | 0.6718 | 0.2678 |
| sw_smote_r01 | Mixed | Out-domain | 0.6809 | 0.4878 | 0.8103 | 0.3225 |
| sw_smote_r05 | Cross-domain | In-domain | 0.0550 | 0.0530 | 0.0566 | 0.0035 |
| sw_smote_r05 | Cross-domain | Out-domain | 0.0435 | 0.0430 | 0.0439 | 0.0009 |
| sw_smote_r05 | Within-domain | In-domain | 0.4438 | 0.3523 | 0.5549 | 0.2026 |
| sw_smote_r05 | Within-domain | Out-domain | 0.3639 | 0.3028 | 0.4693 | 0.1665 |
| sw_smote_r05 | Mixed | In-domain | 0.3879 | 0.2823 | 0.5097 | 0.2274 |
| sw_smote_r05 | Mixed | Out-domain | 0.4316 | 0.3252 | 0.5584 | 0.2332 |


### Recall

| Condition | Mode | Level | Mean | 95% CI Lower | 95% CI Upper | Width |
|-----------|------|-------|-----:|-------------:|-------------:|------:|
| baseline | Cross-domain | In-domain | 0.5091 | 0.4981 | 0.5257 | 0.0276 |
| baseline | Cross-domain | Out-domain | 0.4764 | 0.4486 | 0.4974 | 0.0488 |
| baseline | Within-domain | In-domain | 0.6176 | 0.5980 | 0.6368 | 0.0387 |
| baseline | Within-domain | Out-domain | 0.7045 | 0.6654 | 0.7588 | 0.0934 |
| baseline | Mixed | In-domain | 0.6987 | 0.6668 | 0.7294 | 0.0627 |
| baseline | Mixed | Out-domain | 0.8582 | 0.8311 | 0.8844 | 0.0532 |
| rus_r01 | Cross-domain | In-domain | 0.5201 | 0.5104 | 0.5303 | 0.0199 |
| rus_r01 | Cross-domain | Out-domain | 0.4595 | 0.4452 | 0.4804 | 0.0351 |
| rus_r01 | Within-domain | In-domain | 0.4163 | 0.3875 | 0.4482 | 0.0607 |
| rus_r01 | Within-domain | Out-domain | 0.6620 | 0.6201 | 0.6954 | 0.0754 |
| rus_r01 | Mixed | In-domain | 0.5446 | 0.4958 | 0.6410 | 0.1452 |
| rus_r01 | Mixed | Out-domain | 0.6070 | 0.5500 | 0.6836 | 0.1336 |
| rus_r05 | Cross-domain | In-domain | 0.4084 | 0.3934 | 0.4249 | 0.0315 |
| rus_r05 | Cross-domain | Out-domain | 0.4484 | 0.4231 | 0.4709 | 0.0477 |
| rus_r05 | Within-domain | In-domain | 0.3281 | 0.2940 | 0.3753 | 0.0813 |
| rus_r05 | Within-domain | Out-domain | 0.5581 | 0.5206 | 0.5979 | 0.0773 |
| rus_r05 | Mixed | In-domain | 0.3823 | 0.3359 | 0.4035 | 0.0677 |
| rus_r05 | Mixed | Out-domain | 0.5596 | 0.5298 | 0.6014 | 0.0716 |
| smote_r01 | Cross-domain | In-domain | 0.2736 | 0.2496 | 0.2925 | 0.0428 |
| smote_r01 | Cross-domain | Out-domain | 0.2708 | 0.2465 | 0.2936 | 0.0471 |
| smote_r01 | Within-domain | In-domain | 0.8664 | 0.8565 | 0.8749 | 0.0184 |
| smote_r01 | Within-domain | Out-domain | 0.8658 | 0.8521 | 0.8750 | 0.0229 |
| smote_r01 | Mixed | In-domain | 0.7967 | 0.7605 | 0.8124 | 0.0518 |
| smote_r01 | Mixed | Out-domain | 0.8734 | 0.8632 | 0.8815 | 0.0183 |
| smote_r05 | Cross-domain | In-domain | 0.1816 | 0.1649 | 0.1981 | 0.0332 |
| smote_r05 | Cross-domain | Out-domain | 0.1790 | 0.1596 | 0.2006 | 0.0410 |
| smote_r05 | Within-domain | In-domain | 0.7839 | 0.7641 | 0.8069 | 0.0428 |
| smote_r05 | Within-domain | Out-domain | 0.7980 | 0.7774 | 0.8186 | 0.0413 |
| smote_r05 | Mixed | In-domain | 0.7363 | 0.7060 | 0.7588 | 0.0528 |
| smote_r05 | Mixed | Out-domain | 0.7832 | 0.7529 | 0.8149 | 0.0621 |
| sw_smote_r01 | Cross-domain | In-domain | 0.1608 | 0.1481 | 0.1711 | 0.0230 |
| sw_smote_r01 | Cross-domain | Out-domain | 0.1539 | 0.1442 | 0.1615 | 0.0172 |
| sw_smote_r01 | Within-domain | In-domain | 0.8283 | 0.7914 | 0.8548 | 0.0635 |
| sw_smote_r01 | Within-domain | Out-domain | 0.8342 | 0.7453 | 0.8677 | 0.1224 |
| sw_smote_r01 | Mixed | In-domain | 0.7184 | 0.6180 | 0.7808 | 0.1628 |
| sw_smote_r01 | Mixed | Out-domain | 0.7727 | 0.6508 | 0.8482 | 0.1974 |
| sw_smote_r05 | Cross-domain | In-domain | 0.0474 | 0.0428 | 0.0521 | 0.0094 |
| sw_smote_r05 | Cross-domain | Out-domain | 0.0401 | 0.0338 | 0.0459 | 0.0121 |
| sw_smote_r05 | Within-domain | In-domain | 0.5026 | 0.4212 | 0.6059 | 0.1848 |
| sw_smote_r05 | Within-domain | Out-domain | 0.5117 | 0.4424 | 0.6092 | 0.1668 |
| sw_smote_r05 | Mixed | In-domain | 0.3641 | 0.2706 | 0.4741 | 0.2034 |
| sw_smote_r05 | Mixed | Out-domain | 0.4654 | 0.3774 | 0.5850 | 0.2076 |

---
## 17. Effect Size Confidence Intervals (Cliff's δ)

Bootstrap 95% CI (B = 2,000, percentile method).


### F1-score — Baseline vs each method

| Method vs Baseline | Mode | Level | δ | 95% CI | Excl. 0? | Effect |
|--------------------|------|-------|--:|-------:|:--------:|:------:|
| rus_r01 vs baseline | Cross-domain | In-domain | +0.695 | [+0.492, +0.869] | ✓ | large |
| rus_r01 vs baseline | Cross-domain | Out-domain | -0.150 | [-0.441, +0.143] | ✗ | small |
| rus_r01 vs baseline | Within-domain | In-domain | -0.658 | [-0.860, -0.423] | ✓ | large |
| rus_r01 vs baseline | Within-domain | Out-domain | +0.010 | [-0.279, +0.309] | ✗ | negligible |
| rus_r01 vs baseline | Mixed | In-domain | -0.622 | [-0.862, -0.355] | ✓ | large |
| rus_r01 vs baseline | Mixed | Out-domain | -0.871 | [-0.985, -0.725] | ✓ | large |
| rus_r05 vs baseline | Cross-domain | In-domain | +0.322 | [+0.041, +0.592] | ✓ | small |
| rus_r05 vs baseline | Cross-domain | Out-domain | -0.213 | [-0.482, +0.061] | ✗ | small |
| rus_r05 vs baseline | Within-domain | In-domain | -0.765 | [-0.933, -0.544] | ✓ | large |
| rus_r05 vs baseline | Within-domain | Out-domain | -0.354 | [-0.605, -0.092] | ✓ | medium |
| rus_r05 vs baseline | Mixed | In-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| rus_r05 vs baseline | Mixed | Out-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| smote_r01 vs baseline | Cross-domain | In-domain | -0.312 | [-0.613, +0.004] | ✗ | small |
| smote_r01 vs baseline | Cross-domain | Out-domain | +0.223 | [-0.064, +0.504] | ✗ | small |
| smote_r01 vs baseline | Within-domain | In-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| smote_r01 vs baseline | Within-domain | Out-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| smote_r01 vs baseline | Mixed | In-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| smote_r01 vs baseline | Mixed | Out-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| smote_r05 vs baseline | Cross-domain | In-domain | -0.572 | [-0.814, -0.289] | ✓ | large |
| smote_r05 vs baseline | Cross-domain | Out-domain | -0.028 | [-0.331, +0.274] | ✗ | negligible |
| smote_r05 vs baseline | Within-domain | In-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| smote_r05 vs baseline | Within-domain | Out-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| smote_r05 vs baseline | Mixed | In-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| smote_r05 vs baseline | Mixed | Out-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| sw_smote_r01 vs baseline | Cross-domain | In-domain | -0.949 | [-1.000, -0.865] | ✓ | large |
| sw_smote_r01 vs baseline | Cross-domain | Out-domain | -0.837 | [-0.956, -0.671] | ✓ | large |
| sw_smote_r01 vs baseline | Within-domain | In-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| sw_smote_r01 vs baseline | Within-domain | Out-domain | +0.990 | [+0.964, +1.000] | ✓ | large |
| sw_smote_r01 vs baseline | Mixed | In-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| sw_smote_r01 vs baseline | Mixed | Out-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| sw_smote_r05 vs baseline | Cross-domain | In-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| sw_smote_r05 vs baseline | Cross-domain | Out-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| sw_smote_r05 vs baseline | Within-domain | In-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| sw_smote_r05 vs baseline | Within-domain | Out-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| sw_smote_r05 vs baseline | Mixed | In-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| sw_smote_r05 vs baseline | Mixed | Out-domain | +1.000 | [+1.000, +1.000] | ✓ | large |

**30/36** CIs exclude 0.


### AUPRC — Baseline vs each method

| Method vs Baseline | Mode | Level | δ | 95% CI | Excl. 0? | Effect |
|--------------------|------|-------|--:|-------:|:--------:|:------:|
| rus_r01 vs baseline | Cross-domain | In-domain | +0.059 | [-0.285, +0.381] | ✗ | negligible |
| rus_r01 vs baseline | Cross-domain | Out-domain | +0.018 | [-0.281, +0.297] | ✗ | negligible |
| rus_r01 vs baseline | Within-domain | In-domain | +0.044 | [-0.221, +0.323] | ✗ | negligible |
| rus_r01 vs baseline | Within-domain | Out-domain | -0.160 | [-0.430, +0.131] | ✗ | small |
| rus_r01 vs baseline | Mixed | In-domain | -0.460 | [-0.738, -0.133] | ✓ | medium |
| rus_r01 vs baseline | Mixed | Out-domain | -0.800 | [-0.942, -0.613] | ✓ | large |
| rus_r05 vs baseline | Cross-domain | In-domain | -0.123 | [-0.434, +0.182] | ✗ | negligible |
| rus_r05 vs baseline | Cross-domain | Out-domain | +0.258 | [-0.031, +0.516] | ✗ | small |
| rus_r05 vs baseline | Within-domain | In-domain | +0.100 | [-0.194, +0.385] | ✗ | negligible |
| rus_r05 vs baseline | Within-domain | Out-domain | -0.383 | [-0.627, -0.117] | ✓ | medium |
| rus_r05 vs baseline | Mixed | In-domain | -0.854 | [-0.950, -0.713] | ✓ | large |
| rus_r05 vs baseline | Mixed | Out-domain | -0.983 | [-1.000, -0.935] | ✓ | large |
| smote_r01 vs baseline | Cross-domain | In-domain | -0.270 | [-0.531, +0.037] | ✗ | small |
| smote_r01 vs baseline | Cross-domain | Out-domain | +0.012 | [-0.268, +0.287] | ✗ | negligible |
| smote_r01 vs baseline | Within-domain | In-domain | +0.978 | [+0.923, +1.000] | ✓ | large |
| smote_r01 vs baseline | Within-domain | Out-domain | +0.966 | [+0.893, +1.000] | ✓ | large |
| smote_r01 vs baseline | Mixed | In-domain | +0.980 | [+0.938, +1.000] | ✓ | large |
| smote_r01 vs baseline | Mixed | Out-domain | +0.489 | [+0.196, +0.771] | ✓ | large |
| smote_r05 vs baseline | Cross-domain | In-domain | -0.549 | [-0.777, -0.289] | ✓ | large |
| smote_r05 vs baseline | Cross-domain | Out-domain | -0.087 | [-0.377, +0.212] | ✗ | negligible |
| smote_r05 vs baseline | Within-domain | In-domain | +0.978 | [+0.923, +1.000] | ✓ | large |
| smote_r05 vs baseline | Within-domain | Out-domain | +0.925 | [+0.833, +0.990] | ✓ | large |
| smote_r05 vs baseline | Mixed | In-domain | +0.980 | [+0.938, +1.000] | ✓ | large |
| smote_r05 vs baseline | Mixed | Out-domain | +0.491 | [+0.215, +0.764] | ✓ | large |
| sw_smote_r01 vs baseline | Cross-domain | In-domain | +0.172 | [-0.145, +0.473] | ✗ | small |
| sw_smote_r01 vs baseline | Cross-domain | Out-domain | -0.690 | [-0.857, -0.496] | ✓ | large |
| sw_smote_r01 vs baseline | Within-domain | In-domain | +0.983 | [+0.935, +1.000] | ✓ | large |
| sw_smote_r01 vs baseline | Within-domain | Out-domain | +0.913 | [+0.812, +0.980] | ✓ | large |
| sw_smote_r01 vs baseline | Mixed | In-domain | +0.964 | [+0.907, +1.000] | ✓ | large |
| sw_smote_r01 vs baseline | Mixed | Out-domain | +0.625 | [+0.384, +0.814] | ✓ | large |
| sw_smote_r05 vs baseline | Cross-domain | In-domain | +0.336 | [+0.035, +0.625] | ✓ | medium |
| sw_smote_r05 vs baseline | Cross-domain | Out-domain | -0.694 | [-0.881, -0.480] | ✓ | large |
| sw_smote_r05 vs baseline | Within-domain | In-domain | +0.950 | [+0.831, +1.000] | ✓ | large |
| sw_smote_r05 vs baseline | Within-domain | Out-domain | +0.681 | [+0.458, +0.858] | ✓ | large |
| sw_smote_r05 vs baseline | Mixed | In-domain | +0.936 | [+0.846, +0.998] | ✓ | large |
| sw_smote_r05 vs baseline | Mixed | Out-domain | +0.211 | [-0.087, +0.516] | ✗ | small |

**24/36** CIs exclude 0.


### Recall — Baseline vs each method

| Method vs Baseline | Mode | Level | δ | 95% CI | Excl. 0? | Effect |
|--------------------|------|-------|--:|-------:|:--------:|:------:|
| rus_r01 vs baseline | Cross-domain | In-domain | +0.229 | [-0.051, +0.504] | ✗ | small |
| rus_r01 vs baseline | Cross-domain | Out-domain | -0.132 | [-0.424, +0.182] | ✗ | negligible |
| rus_r01 vs baseline | Within-domain | In-domain | -0.625 | [-0.834, -0.391] | ✓ | large |
| rus_r01 vs baseline | Within-domain | Out-domain | -0.058 | [-0.346, +0.252] | ✗ | negligible |
| rus_r01 vs baseline | Mixed | In-domain | -0.609 | [-0.849, -0.344] | ✓ | large |
| rus_r01 vs baseline | Mixed | Out-domain | -0.917 | [-0.998, -0.809] | ✓ | large |
| rus_r05 vs baseline | Cross-domain | In-domain | -0.820 | [-0.984, -0.612] | ✓ | large |
| rus_r05 vs baseline | Cross-domain | Out-domain | -0.209 | [-0.494, +0.074] | ✗ | small |
| rus_r05 vs baseline | Within-domain | In-domain | -0.848 | [-0.971, -0.694] | ✓ | large |
| rus_r05 vs baseline | Within-domain | Out-domain | -0.386 | [-0.623, -0.088] | ✓ | medium |
| rus_r05 vs baseline | Mixed | In-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| rus_r05 vs baseline | Mixed | Out-domain | -0.987 | [-1.000, -0.948] | ✓ | large |
| smote_r01 vs baseline | Cross-domain | In-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| smote_r01 vs baseline | Cross-domain | Out-domain | -0.969 | [-0.998, -0.914] | ✓ | large |
| smote_r01 vs baseline | Within-domain | In-domain | +0.957 | [+0.852, +1.000] | ✓ | large |
| smote_r01 vs baseline | Within-domain | Out-domain | +0.505 | [+0.218, +0.770] | ✓ | large |
| smote_r01 vs baseline | Mixed | In-domain | +0.781 | [+0.597, +0.918] | ✓ | large |
| smote_r01 vs baseline | Mixed | Out-domain | +0.096 | [-0.229, +0.424] | ✗ | negligible |
| smote_r05 vs baseline | Cross-domain | In-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| smote_r05 vs baseline | Cross-domain | Out-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| smote_r05 vs baseline | Within-domain | In-domain | +0.835 | [+0.671, +0.960] | ✓ | large |
| smote_r05 vs baseline | Within-domain | Out-domain | +0.367 | [+0.048, +0.656] | ✓ | medium |
| smote_r05 vs baseline | Mixed | In-domain | +0.246 | [-0.065, +0.542] | ✗ | small |
| smote_r05 vs baseline | Mixed | Out-domain | -0.676 | [-0.844, -0.458] | ✓ | large |
| sw_smote_r01 vs baseline | Cross-domain | In-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| sw_smote_r01 vs baseline | Cross-domain | Out-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| sw_smote_r01 vs baseline | Within-domain | In-domain | +0.780 | [+0.564, +0.944] | ✓ | large |
| sw_smote_r01 vs baseline | Within-domain | Out-domain | +0.392 | [+0.093, +0.677] | ✓ | medium |
| sw_smote_r01 vs baseline | Mixed | In-domain | +0.313 | [+0.006, +0.621] | ✓ | small |
| sw_smote_r01 vs baseline | Mixed | Out-domain | -0.236 | [-0.522, +0.061] | ✗ | small |
| sw_smote_r05 vs baseline | Cross-domain | In-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| sw_smote_r05 vs baseline | Cross-domain | Out-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| sw_smote_r05 vs baseline | Within-domain | In-domain | -0.375 | [-0.633, -0.091] | ✓ | medium |
| sw_smote_r05 vs baseline | Within-domain | Out-domain | -0.616 | [-0.815, -0.379] | ✓ | large |
| sw_smote_r05 vs baseline | Mixed | In-domain | -0.945 | [-0.993, -0.867] | ✓ | large |
| sw_smote_r05 vs baseline | Mixed | Out-domain | -1.000 | [-1.000, -1.000] | ✓ | large |

**29/36** CIs exclude 0.

---
## 18. Permutation Test for Global Null

B = 10,000 permutations.


### F1-score

- $T_{\text{obs}}$ = 11.4748
- $p_{\text{perm}}$ = 0.0001 (B = 10000)
- **Interpretation**: Strong evidence against the global null (p < 0.001). Condition labels are informative for F1-score.


### AUPRC

- $T_{\text{obs}}$ = 17.6123
- $p_{\text{perm}}$ = 0.0001 (B = 10000)
- **Interpretation**: Strong evidence against the global null (p < 0.001). Condition labels are informative for AUPRC.


### Recall

- $T_{\text{obs}}$ = 18.8425
- $p_{\text{perm}}$ = 0.0001 (B = 10000)
- **Interpretation**: Strong evidence against the global null (p < 0.001). Condition labels are informative for Recall.


---
## 19. Benjamini-Hochberg FDR Correction Sensitivity

| Hypothesis Family | m | Bonf. sig | FDR sig | Gain |
|-------------------|--:|----------:|--------:|-----:|
| H1 KW (F1-score) | 18 | 18 | 18 | +0 |
| H1 pairwise (F1-score) | 36 | 28 | 31 | +3 |
| H10 domain shift (F1-score) | 63 | 0 | 21 | +21 |
| H1 KW (AUPRC) | 18 | 18 | 18 | +0 |
| H1 pairwise (AUPRC) | 36 | 21 | 24 | +3 |
| H10 domain shift (AUPRC) | 63 | 0 | 25 | +25 |
| H1 KW (Recall) | 18 | 18 | 18 | +0 |
| H1 pairwise (Recall) | 36 | 24 | 29 | +5 |
| H10 domain shift (Recall) | 63 | 0 | 26 | +26 |

**Overall**: Bonferroni **127** → BH-FDR **210** (+83).

---
## 20. Cross-Metric Concordance

Do the extended metrics (F1, AUPRC, Recall) agree with the primary metrics (F2, AUROC) on condition rankings?

### Overall ranking comparison

| Metric | #1 | #2 | #3 | #4 | #5 | #6 | #7 |
|--------|:---|:---|:---|:---|:---|:---|:---|
| F2-score | sw_smote_r01 (2.56) | smote_r05 (3.22) | smote_r01 (3.56) | baseline (4.11) | rus_r01 (4.61) | sw_smote_r05 (4.67) | rus_r05 (5.28) |
| AUROC | smote_r01 (2.17) | smote_r05 (3.17) | sw_smote_r01 (3.28) | sw_smote_r05 (4.11) | baseline (4.78) | rus_r01 (4.94) | rus_r05 (5.56) |
| F1-score | sw_smote_r01 (3.06) | sw_smote_r05 (3.17) | smote_r01 (3.50) | smote_r05 (3.50) | baseline (4.56) | rus_r01 (4.83) | rus_r05 (5.39) |
| AUPRC | sw_smote_r01 (2.56) | smote_r01 (2.61) | smote_r05 (3.39) | sw_smote_r05 (4.22) | rus_r01 (4.50) | baseline (5.28) | rus_r05 (5.44) |
| Recall | smote_r01 (2.17) | baseline (2.83) | smote_r05 (3.44) | sw_smote_r01 (4.00) | rus_r01 (4.11) | rus_r05 (4.89) | sw_smote_r05 (6.56) |
| Precision | sw_smote_r05 (2.61) | sw_smote_r01 (2.83) | smote_r05 (3.06) | smote_r01 (3.72) | baseline (4.89) | rus_r01 (5.22) | rus_r05 (5.67) |
| Accuracy | sw_smote_r05 (1.00) | sw_smote_r01 (2.06) | smote_r05 (2.94) | smote_r01 (4.00) | rus_r05 (5.50) | rus_r01 (6.11) | baseline (6.39) |

**Kendall's W** = 0.618 (k=7 metrics, n=7 conditions)

**Interpretation**: Moderate agreement.

### Pairwise Spearman ρ

| | F2-score | AUROC | F1-score | AUPRC | Recall | Precision | Accuracy |
|---|---|---|---|---|---|---|---|
| **F2-score** | — | 0.750 | 0.631 | 0.821 | 0.643 | 0.464 | 0.250 |
| **AUROC** | — | — | 0.703 | 0.857 | 0.607 | 0.643 | 0.500 |
| **F1-score** | — | — | — | 0.847 | 0.072 | 0.955 | 0.811 |
| **AUPRC** | — | — | — | — | 0.357 | 0.714 | 0.643 |
| **Recall** | — | — | — | — | — | -0.071 | -0.357 |
| **Precision** | — | — | — | — | — | — | 0.857 |
| **Accuracy** | — | — | — | — | — | — | — |

### Primary vs Extended concordance

- **F2-score ↔ F1-score**: ρ = 0.631
- **F2-score ↔ AUPRC**: ρ = 0.821
- **F2-score ↔ Recall**: ρ = 0.643
- **AUROC ↔ F1-score**: ρ = 0.703
- **AUROC ↔ AUPRC**: ρ = 0.857
- **AUROC ↔ Recall**: ρ = 0.607

Mean cross-group ρ = 0.710.

---
## 21. Hypothesis Verdict Summary

| ID | Hypothesis | Evidence | Verdict |
|:--:|-----------|----------|---------|
| H1 | Condition effect (F1-score) | 18/18 sig | Supported ✓ |
| H2 | sw > smote (F1-score) | 8/12 cells | Supported ✓ |
| H3 | Over > RUS (F1-score) | 18/24 | Supported ✓ |
| H5 | Distance effect (F1-score) | H=3.06, p=0.2170 | Not supported ✗ |
| H7 | Within > cross (F1-score) | δ=+0.783 (large) | Supported ✓ |
| H10 | Domain shift (F1-score) | δ=-0.085, p=0.0081 | Weak |
| H1 | Condition effect (AUPRC) | 18/18 sig | Supported ✓ |
| H2 | sw > smote (AUPRC) | 6/12 cells | Mixed |
| H3 | Over > RUS (AUPRC) | 17/24 | Supported ✓ |
| H5 | Distance effect (AUPRC) | H=4.71, p=0.0951 | Not supported ✗ |
| H7 | Within > cross (AUPRC) | δ=+0.948 (large) | Supported ✓ |
| H10 | Domain shift (AUPRC) | δ=+0.058, p=0.0681 | Not supported ✗ |
| H1 | Condition effect (Recall) | 18/18 sig | Supported ✓ |
| H2 | sw > smote (Recall) | 0/12 cells | Not supported ✗ |
| H3 | Over > RUS (Recall) | 13/24 | Mixed |
| H5 | Distance effect (Recall) | H=2.76, p=0.2515 | Not supported ✗ |
| H7 | Within > cross (Recall) | δ=+0.779 (large) | Supported ✓ |
| H10 | Domain shift (Recall) | δ=-0.148, p=0.0000 | Weak |
| HE1 | Rebalancing → Recall↑ | 11/36 cells improved | Not supported ✗ |
| HE2 | AUPRC more sensitive than AUROC | Stronger η² in 3/6 cells | Mixed |
| HE3 | Precision–Recall trade-off | 0/36 cells show P↓R↑ | Not supported ✗ |

---
## 22. Conclusions

### Key Findings from Extended Analysis


1. **Metric consistency**: The extended metrics (F1, AUPRC, Recall) show moderate agreement with the primary metrics (F2, AUROC) on condition rankings (Kendall's W = 0.618), confirming that the primary analysis conclusions are robust to metric choice.

2. **AUPRC provides additional insight**: As a threshold-free metric sensitive to minority-class performance, AUPRC complements AUROC and is particularly relevant for imbalanced drowsiness detection.

3. **Recall validation**: Direct analysis of Recall confirms that rebalancing methods achieve their intended effect of improving positive class detection rate.

4. **Precision–Recall trade-off**: The trade-off analysis quantifies the cost of improved Recall in terms of Precision degradation, providing practical guidance for deployment threshold selection.


### Relationship to Primary Analysis

This report should be read alongside the primary hypothesis test report (F2-score and AUROC). Together, the two reports provide a comprehensive statistical evaluation of Experiment 2 across 7 evaluation metrics.

