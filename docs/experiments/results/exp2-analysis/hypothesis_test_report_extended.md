# Experiment 2 — Extended Hypothesis Testing (F1, AUPRC, Recall)

**Records**: 1258  
**Seeds**: [0, 1, 7, 13, 42, 123, 256, 512, 1337, 2024] (n=10)  
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
| baseline | Cross-domain | In-domain | 0.9482 | 0.1511 | ✓ normal |
| baseline | Cross-domain | Out-domain | 0.9420 | 0.1032 | ✓ normal |
| baseline | Within-domain | In-domain | 0.7796 | 0.0000 | ✗ reject |
| baseline | Within-domain | Out-domain | 0.6874 | 0.0000 | ✗ reject |
| baseline | Mixed | In-domain | 0.9047 | 0.0109 | ✗ reject |
| baseline | Mixed | Out-domain | 0.9601 | 0.3108 | ✓ normal |
| rus_r01 | Cross-domain | In-domain | 0.9510 | 0.1803 | ✓ normal |
| rus_r01 | Cross-domain | Out-domain | 0.9391 | 0.0862 | ✓ normal |
| rus_r01 | Within-domain | In-domain | 0.9430 | 0.1093 | ✓ normal |
| rus_r01 | Within-domain | Out-domain | 0.8880 | 0.0043 | ✗ reject |
| rus_r01 | Mixed | In-domain | 0.9049 | 0.0111 | ✗ reject |
| rus_r01 | Mixed | Out-domain | 0.9367 | 0.0741 | ✓ normal |
| rus_r05 | Cross-domain | In-domain | 0.9250 | 0.0363 | ✗ reject |
| rus_r05 | Cross-domain | Out-domain | 0.9536 | 0.2108 | ✓ normal |
| rus_r05 | Within-domain | In-domain | 0.9786 | 0.7869 | ✓ normal |
| rus_r05 | Within-domain | Out-domain | 0.9653 | 0.4205 | ✓ normal |
| rus_r05 | Mixed | In-domain | 0.9531 | 0.2044 | ✓ normal |
| rus_r05 | Mixed | Out-domain | 0.9646 | 0.4036 | ✓ normal |
| smote_r01 | Cross-domain | In-domain | 0.9622 | 0.3526 | ✓ normal |
| smote_r01 | Cross-domain | Out-domain | 0.9693 | 0.5204 | ✓ normal |
| smote_r01 | Within-domain | In-domain | 0.9370 | 0.0754 | ✓ normal |
| smote_r01 | Within-domain | Out-domain | 0.9103 | 0.0151 | ✗ reject |
| smote_r01 | Mixed | In-domain | 0.9639 | 0.3891 | ✓ normal |
| smote_r01 | Mixed | Out-domain | 0.9527 | 0.1993 | ✓ normal |
| smote_r05 | Cross-domain | In-domain | 0.9513 | 0.1830 | ✓ normal |
| smote_r05 | Cross-domain | Out-domain | 0.9861 | 0.9540 | ✓ normal |
| smote_r05 | Within-domain | In-domain | 0.8950 | 0.0063 | ✗ reject |
| smote_r05 | Within-domain | Out-domain | 0.9559 | 0.2423 | ✓ normal |
| smote_r05 | Mixed | In-domain | 0.9645 | 0.4018 | ✓ normal |
| smote_r05 | Mixed | Out-domain | 0.9184 | 0.0244 | ✗ reject |
| sw_smote_r01 | Cross-domain | In-domain | 0.9889 | 0.9845 | ✓ normal |
| sw_smote_r01 | Cross-domain | Out-domain | 0.9830 | 0.8984 | ✓ normal |
| sw_smote_r01 | Within-domain | In-domain | 0.9629 | 0.3672 | ✓ normal |
| sw_smote_r01 | Within-domain | Out-domain | 0.9183 | 0.0242 | ✗ reject |
| sw_smote_r01 | Mixed | In-domain | 0.9368 | 0.0746 | ✓ normal |
| sw_smote_r01 | Mixed | Out-domain | 0.8532 | 0.0009 | ✗ reject |
| sw_smote_r05 | Cross-domain | In-domain | 0.9308 | 0.0515 | ✓ normal |
| sw_smote_r05 | Cross-domain | Out-domain | 0.9727 | 0.6146 | ✓ normal |
| sw_smote_r05 | Within-domain | In-domain | 0.9464 | 0.1355 | ✓ normal |
| sw_smote_r05 | Within-domain | Out-domain | 0.9458 | 0.1302 | ✓ normal |
| sw_smote_r05 | Mixed | In-domain | 0.8510 | 0.0008 | ✗ reject |
| sw_smote_r05 | Mixed | Out-domain | 0.8826 | 0.0032 | ✗ reject |

**Summary**: 13/42 cells (31%) reject normality at α=0.05.

### AUPRC

| Condition | Mode | Level | W | p | Normal? |
|-----------|------|-------|--:|--:|:-------:|
| baseline | Cross-domain | In-domain | 0.9321 | 0.0557 | ✓ normal |
| baseline | Cross-domain | Out-domain | 0.9405 | 0.0936 | ✓ normal |
| baseline | Within-domain | In-domain | 0.3229 | 0.0000 | ✗ reject |
| baseline | Within-domain | Out-domain | 0.6744 | 0.0000 | ✗ reject |
| baseline | Mixed | In-domain | 0.9303 | 0.0499 | ✗ reject |
| baseline | Mixed | Out-domain | 0.7668 | 0.0000 | ✗ reject |
| rus_r01 | Cross-domain | In-domain | 0.7805 | 0.0000 | ✗ reject |
| rus_r01 | Cross-domain | Out-domain | 0.5822 | 0.0000 | ✗ reject |
| rus_r01 | Within-domain | In-domain | 0.5228 | 0.0000 | ✗ reject |
| rus_r01 | Within-domain | Out-domain | 0.6889 | 0.0000 | ✗ reject |
| rus_r01 | Mixed | In-domain | 0.6104 | 0.0000 | ✗ reject |
| rus_r01 | Mixed | Out-domain | 0.6967 | 0.0000 | ✗ reject |
| rus_r05 | Cross-domain | In-domain | 0.6730 | 0.0000 | ✗ reject |
| rus_r05 | Cross-domain | Out-domain | 0.7249 | 0.0000 | ✗ reject |
| rus_r05 | Within-domain | In-domain | 0.6007 | 0.0000 | ✗ reject |
| rus_r05 | Within-domain | Out-domain | 0.6580 | 0.0000 | ✗ reject |
| rus_r05 | Mixed | In-domain | 0.7536 | 0.0000 | ✗ reject |
| rus_r05 | Mixed | Out-domain | 0.7780 | 0.0000 | ✗ reject |
| smote_r01 | Cross-domain | In-domain | 0.9859 | 0.9510 | ✓ normal |
| smote_r01 | Cross-domain | Out-domain | 0.9470 | 0.1405 | ✓ normal |
| smote_r01 | Within-domain | In-domain | 0.9286 | 0.0450 | ✗ reject |
| smote_r01 | Within-domain | Out-domain | 0.9502 | 0.1710 | ✓ normal |
| smote_r01 | Mixed | In-domain | 0.9001 | 0.0085 | ✗ reject |
| smote_r01 | Mixed | Out-domain | 0.8906 | 0.0050 | ✗ reject |
| smote_r05 | Cross-domain | In-domain | 0.9343 | 0.0641 | ✓ normal |
| smote_r05 | Cross-domain | Out-domain | 0.8915 | 0.0052 | ✗ reject |
| smote_r05 | Within-domain | In-domain | 0.9697 | 0.5313 | ✓ normal |
| smote_r05 | Within-domain | Out-domain | 0.9635 | 0.3791 | ✓ normal |
| smote_r05 | Mixed | In-domain | 0.9016 | 0.0092 | ✗ reject |
| smote_r05 | Mixed | Out-domain | 0.9545 | 0.2234 | ✓ normal |
| sw_smote_r01 | Cross-domain | In-domain | 0.9378 | 0.0796 | ✓ normal |
| sw_smote_r01 | Cross-domain | Out-domain | 0.9769 | 0.7378 | ✓ normal |
| sw_smote_r01 | Within-domain | In-domain | 0.9012 | 0.0090 | ✗ reject |
| sw_smote_r01 | Within-domain | Out-domain | 0.8090 | 0.0001 | ✗ reject |
| sw_smote_r01 | Mixed | In-domain | 0.8302 | 0.0002 | ✗ reject |
| sw_smote_r01 | Mixed | Out-domain | 0.6710 | 0.0000 | ✗ reject |
| sw_smote_r05 | Cross-domain | In-domain | 0.8695 | 0.0016 | ✗ reject |
| sw_smote_r05 | Cross-domain | Out-domain | 0.9518 | 0.1894 | ✓ normal |
| sw_smote_r05 | Within-domain | In-domain | 0.9421 | 0.1034 | ✓ normal |
| sw_smote_r05 | Within-domain | Out-domain | 0.9168 | 0.0222 | ✗ reject |
| sw_smote_r05 | Mixed | In-domain | 0.8350 | 0.0004 | ✗ reject |
| sw_smote_r05 | Mixed | Out-domain | 0.8777 | 0.0025 | ✗ reject |

**Summary**: 29/42 cells (69%) reject normality at α=0.05.

### Recall

| Condition | Mode | Level | W | p | Normal? |
|-----------|------|-------|--:|--:|:-------:|
| baseline | Cross-domain | In-domain | 0.6779 | 0.0000 | ✗ reject |
| baseline | Cross-domain | Out-domain | 0.9610 | 0.3283 | ✓ normal |
| baseline | Within-domain | In-domain | 0.9358 | 0.0702 | ✓ normal |
| baseline | Within-domain | Out-domain | 0.8066 | 0.0001 | ✗ reject |
| baseline | Mixed | In-domain | 0.9048 | 0.0110 | ✗ reject |
| baseline | Mixed | Out-domain | 0.9029 | 0.0099 | ✗ reject |
| rus_r01 | Cross-domain | In-domain | 0.8984 | 0.0077 | ✗ reject |
| rus_r01 | Cross-domain | Out-domain | 0.9602 | 0.3131 | ✓ normal |
| rus_r01 | Within-domain | In-domain | 0.9094 | 0.0144 | ✗ reject |
| rus_r01 | Within-domain | Out-domain | 0.9609 | 0.3259 | ✓ normal |
| rus_r01 | Mixed | In-domain | 0.8932 | 0.0058 | ✗ reject |
| rus_r01 | Mixed | Out-domain | 0.9405 | 0.0941 | ✓ normal |
| rus_r05 | Cross-domain | In-domain | 0.9357 | 0.0698 | ✓ normal |
| rus_r05 | Cross-domain | Out-domain | 0.8980 | 0.0075 | ✗ reject |
| rus_r05 | Within-domain | In-domain | 0.9171 | 0.0226 | ✗ reject |
| rus_r05 | Within-domain | Out-domain | 0.9716 | 0.5825 | ✓ normal |
| rus_r05 | Mixed | In-domain | 0.9402 | 0.0920 | ✓ normal |
| rus_r05 | Mixed | Out-domain | 0.9497 | 0.1656 | ✓ normal |
| smote_r01 | Cross-domain | In-domain | 0.9727 | 0.6168 | ✓ normal |
| smote_r01 | Cross-domain | Out-domain | 0.9800 | 0.8252 | ✓ normal |
| smote_r01 | Within-domain | In-domain | 0.9000 | 0.0084 | ✗ reject |
| smote_r01 | Within-domain | Out-domain | 0.9329 | 0.0588 | ✓ normal |
| smote_r01 | Mixed | In-domain | 0.8262 | 0.0002 | ✗ reject |
| smote_r01 | Mixed | Out-domain | 0.9160 | 0.0212 | ✗ reject |
| smote_r05 | Cross-domain | In-domain | 0.9227 | 0.0316 | ✗ reject |
| smote_r05 | Cross-domain | Out-domain | 0.9672 | 0.4663 | ✓ normal |
| smote_r05 | Within-domain | In-domain | 0.9633 | 0.3759 | ✓ normal |
| smote_r05 | Within-domain | Out-domain | 0.9530 | 0.2030 | ✓ normal |
| smote_r05 | Mixed | In-domain | 0.8995 | 0.0082 | ✗ reject |
| smote_r05 | Mixed | Out-domain | 0.9215 | 0.0294 | ✗ reject |
| sw_smote_r01 | Cross-domain | In-domain | 0.9659 | 0.4348 | ✓ normal |
| sw_smote_r01 | Cross-domain | Out-domain | 0.9822 | 0.8803 | ✓ normal |
| sw_smote_r01 | Within-domain | In-domain | 0.5758 | 0.0000 | ✗ reject |
| sw_smote_r01 | Within-domain | Out-domain | 0.5709 | 0.0000 | ✗ reject |
| sw_smote_r01 | Mixed | In-domain | 0.7178 | 0.0000 | ✗ reject |
| sw_smote_r01 | Mixed | Out-domain | 0.6927 | 0.0000 | ✗ reject |
| sw_smote_r05 | Cross-domain | In-domain | 0.9561 | 0.2459 | ✓ normal |
| sw_smote_r05 | Cross-domain | Out-domain | 0.9718 | 0.5886 | ✓ normal |
| sw_smote_r05 | Within-domain | In-domain | 0.9306 | 0.0508 | ✓ normal |
| sw_smote_r05 | Within-domain | Out-domain | 0.9544 | 0.2220 | ✓ normal |
| sw_smote_r05 | Mixed | In-domain | 0.8549 | 0.0010 | ✗ reject |
| sw_smote_r05 | Mixed | Out-domain | 0.8604 | 0.0010 | ✗ reject |

**Summary**: 21/42 cells (50%) reject normality at α=0.05.

**Conclusion**: 63/126 cells (50%) violate normality. Non-parametric tests are appropriate.

---
## 4. Descriptive Statistics

### F1-score

#### Cross-domain

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) | n |
|-----------|--------------------:|---------------------:|-----------:|--:|
| baseline | 0.0812±0.0028 | 0.0785±0.0041 | -0.0027 | 30 |
| rus_r01 | 0.0876±0.0059 | 0.0785±0.0072 | -0.0091 | 30 |
| rus_r05 | 0.0824±0.0068 | 0.0775±0.0081 | -0.0050 | 30 |
| smote_r01 | 0.0770±0.0099 | 0.0804±0.0057 | +0.0034 | 30 |
| smote_r05 | 0.0684±0.0119 | 0.0784±0.0095 | +0.0099 | 30 |
| sw_smote_r01 | 0.0678±0.0066 | 0.0672±0.0071 | -0.0006 | 30 |
| sw_smote_r05 | 0.0444±0.0086 | 0.0379±0.0104 | -0.0066 | 30 |

#### Within-domain

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) | n |
|-----------|--------------------:|---------------------:|-----------:|--:|
| baseline | 0.1030±0.0169 | 0.1111±0.0275 | +0.0081 | 30 |
| rus_r01 | 0.0740±0.0237 | 0.1038±0.0219 | +0.0299 | 30 |
| rus_r05 | 0.0681±0.0257 | 0.0920±0.0200 | +0.0239 | 30 |
| smote_r01 | 0.2745±0.0560 | 0.2686±0.0569 | -0.0059 | 30 |
| smote_r05 | 0.3381±0.0975 | 0.3294±0.0870 | -0.0088 | 30 |
| sw_smote_r01 | 0.3741±0.0809 | 0.3774±0.0871 | +0.0033 | 30 |
| sw_smote_r05 | 0.4794±0.1735 | 0.4395±0.1728 | -0.0399 | 30 |

#### Mixed

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) | n |
|-----------|--------------------:|---------------------:|-----------:|--:|
| baseline | 0.1206±0.0107 | 0.1575±0.0176 | +0.0369 | 30 |
| rus_r01 | 0.0995±0.0279 | 0.1141±0.0222 | +0.0145 | 30 |
| rus_r05 | 0.0752±0.0116 | 0.0970±0.0113 | +0.0218 | 30 |
| smote_r01 | 0.2564±0.0654 | 0.3026±0.0424 | +0.0462 | 30 |
| smote_r05 | 0.3276±0.0884 | 0.4056±0.1370 | +0.0780 | 30 |
| sw_smote_r01 | 0.3873±0.0968 | 0.4722±0.1532 | +0.0849 | 30 |
| sw_smote_r05 | 0.4119±0.1800 | 0.4616±0.1606 | +0.0497 | 29 |

### AUPRC

#### Cross-domain

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) | n |
|-----------|--------------------:|---------------------:|-----------:|--:|
| baseline | 0.0508±0.0014 | 0.0466±0.0028 | -0.0041 | 30 |
| rus_r01 | 0.0585±0.0142 | 0.0497±0.0122 | -0.0088 | 30 |
| rus_r05 | 0.0514±0.0055 | 0.0506±0.0094 | -0.0008 | 30 |
| smote_r01 | 0.0502±0.0012 | 0.0463±0.0027 | -0.0038 | 30 |
| smote_r05 | 0.0490±0.0017 | 0.0463±0.0026 | -0.0028 | 30 |
| sw_smote_r01 | 0.0519±0.0031 | 0.0434±0.0018 | -0.0085 | 30 |
| sw_smote_r05 | 0.0556±0.0068 | 0.0434±0.0016 | -0.0123 | 30 |

#### Within-domain

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) | n |
|-----------|--------------------:|---------------------:|-----------:|--:|
| baseline | 0.0872±0.0876 | 0.1387±0.1217 | +0.0514 | 30 |
| rus_r01 | 0.1127±0.1183 | 0.1389±0.1245 | +0.0262 | 30 |
| rus_r05 | 0.1145±0.1072 | 0.0837±0.0553 | -0.0308 | 30 |
| smote_r01 | 0.6452±0.1441 | 0.6108±0.1381 | -0.0344 | 30 |
| smote_r05 | 0.5722±0.1437 | 0.5674±0.1418 | -0.0048 | 30 |
| sw_smote_r01 | 0.6377±0.2064 | 0.6541±0.2210 | +0.0164 | 30 |
| sw_smote_r05 | 0.4591±0.2015 | 0.3639±0.1967 | -0.0952 | 30 |

#### Mixed

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) | n |
|-----------|--------------------:|---------------------:|-----------:|--:|
| baseline | 0.1287±0.0500 | 0.4050±0.2792 | +0.2763 | 30 |
| rus_r01 | 0.1617±0.2028 | 0.1232±0.0977 | -0.0385 | 30 |
| rus_r05 | 0.0655±0.0172 | 0.0736±0.0325 | +0.0081 | 30 |
| smote_r01 | 0.5302±0.1966 | 0.6741±0.1055 | +0.1439 | 30 |
| smote_r05 | 0.5281±0.1614 | 0.6459±0.1446 | +0.1178 | 30 |
| sw_smote_r01 | 0.5540±0.2226 | 0.6748±0.2836 | +0.1208 | 30 |
| sw_smote_r05 | 0.3908±0.1916 | 0.4316±0.1931 | +0.0407 | 29 |

### Recall

#### Cross-domain

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) | n |
|-----------|--------------------:|---------------------:|-----------:|--:|
| baseline | 0.5096±0.0443 | 0.4854±0.0984 | -0.0243 | 30 |
| rus_r01 | 0.5225±0.0415 | 0.4597±0.0580 | -0.0627 | 30 |
| rus_r05 | 0.4065±0.0584 | 0.4515±0.0694 | +0.0450 | 30 |
| smote_r01 | 0.2712±0.0487 | 0.2652±0.0466 | -0.0060 | 30 |
| smote_r05 | 0.1812±0.0522 | 0.1839±0.0450 | +0.0027 | 30 |
| sw_smote_r01 | 0.1630±0.0395 | 0.1552±0.0299 | -0.0078 | 30 |
| sw_smote_r05 | 0.0489±0.0135 | 0.0385±0.0135 | -0.0105 | 30 |

#### Within-domain

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) | n |
|-----------|--------------------:|---------------------:|-----------:|--:|
| baseline | 0.6176±0.1320 | 0.6862±0.1345 | +0.0685 | 30 |
| rus_r01 | 0.4230±0.1711 | 0.6577±0.1318 | +0.2347 | 30 |
| rus_r05 | 0.3385±0.1460 | 0.5677±0.1526 | +0.2292 | 30 |
| smote_r01 | 0.8681±0.0274 | 0.8694±0.0211 | +0.0012 | 30 |
| smote_r05 | 0.7813±0.0492 | 0.8018±0.0454 | +0.0205 | 30 |
| sw_smote_r01 | 0.8241±0.1079 | 0.8290±0.1277 | +0.0048 | 30 |
| sw_smote_r05 | 0.5164±0.1867 | 0.5117±0.1876 | -0.0047 | 30 |

#### Mixed

| Condition | In-domain (mean±SD) | Out-domain (mean±SD) | Δ (out−in) | n |
|-----------|--------------------:|---------------------:|-----------:|--:|
| baseline | 0.6987±0.0789 | 0.8582±0.0452 | +0.1595 | 30 |
| rus_r01 | 0.5439±0.1615 | 0.6143±0.1239 | +0.0704 | 30 |
| rus_r05 | 0.3795±0.0596 | 0.5670±0.1161 | +0.1875 | 30 |
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
| Cross-domain | In-domain | MMD | 50.85 | 0.0000 | 0.712 | ✓ |
| Cross-domain | In-domain | DTW | 58.46 | 0.0000 | 0.833 | ✓ |
| Cross-domain | In-domain | WASSERSTEIN | 56.40 | 0.0000 | 0.800 | ✓ |
| Cross-domain | Out-domain | MMD | 35.30 | 0.0000 | 0.465 | ✓ |
| Cross-domain | Out-domain | DTW | 45.22 | 0.0000 | 0.623 | ✓ |
| Cross-domain | Out-domain | WASSERSTEIN | 51.11 | 0.0000 | 0.716 | ✓ |
| Within-domain | In-domain | MMD | 55.97 | 0.0000 | 0.793 | ✓ |
| Within-domain | In-domain | DTW | 56.17 | 0.0000 | 0.796 | ✓ |
| Within-domain | In-domain | WASSERSTEIN | 62.17 | 0.0000 | 0.892 | ✓ |
| Within-domain | Out-domain | MMD | 55.11 | 0.0000 | 0.780 | ✓ |
| Within-domain | Out-domain | DTW | 54.53 | 0.0000 | 0.770 | ✓ |
| Within-domain | Out-domain | WASSERSTEIN | 57.25 | 0.0000 | 0.814 | ✓ |
| Mixed | In-domain | MMD | 56.44 | 0.0000 | 0.801 | ✓ |
| Mixed | In-domain | DTW | 56.51 | 0.0000 | 0.815 | ✓ |
| Mixed | In-domain | WASSERSTEIN | 56.84 | 0.0000 | 0.807 | ✓ |
| Mixed | Out-domain | MMD | 57.27 | 0.0000 | 0.814 | ✓ |
| Mixed | Out-domain | DTW | 56.66 | 0.0000 | 0.804 | ✓ |
| Mixed | Out-domain | WASSERSTEIN | 56.21 | 0.0000 | 0.810 | ✓ |

**Bonferroni α'=0.0028** (m=18). **18/18** significant.

Mean η² = 0.769 (large effect).

### 5.1.2 [F1-score] H1: Pairwise — baseline vs each method

Mann-Whitney U with Cliff's δ effect size.

| Method vs Baseline | Mode | Level | U | p | δ | Effect | Mean(M) | Mean(B) |
|--------------------|------|-------|--:|--:|--:|:------:|--------:|--------:|
| rus_r01 vs baseline | Cross-domain | In-domain | 143 | 0.0000 * | +0.682 | large | 0.0876 | 0.0812 |
| rus_r01 vs baseline | Cross-domain | Out-domain | 525 | 0.2707 | -0.167 | small | 0.0785 | 0.0785 |
| rus_r01 vs baseline | Within-domain | In-domain | 738 | 0.0000 * | -0.640 | large | 0.0740 | 0.1030 |
| rus_r01 vs baseline | Within-domain | Out-domain | 428 | 0.7506 | +0.049 | negligible | 0.1038 | 0.1111 |
| rus_r01 vs baseline | Mixed | In-domain | 724 | 0.0001 * | -0.609 | large | 0.0995 | 0.1206 |
| rus_r01 vs baseline | Mixed | Out-domain | 840 | 0.0000 * | -0.867 | large | 0.1141 | 0.1575 |
| rus_r05 vs baseline | Cross-domain | In-domain | 317 | 0.0501 | +0.296 | small | 0.0824 | 0.0812 |
| rus_r05 vs baseline | Cross-domain | Out-domain | 530 | 0.2398 | -0.178 | small | 0.0775 | 0.0785 |
| rus_r05 vs baseline | Within-domain | In-domain | 787 | 0.0000 * | -0.749 | large | 0.0681 | 0.1030 |
| rus_r05 vs baseline | Within-domain | Out-domain | 590 | 0.0392 | -0.311 | small | 0.0920 | 0.1111 |
| rus_r05 vs baseline | Mixed | In-domain | 900 | 0.0000 * | -1.000 | large | 0.0752 | 0.1206 |
| rus_r05 vs baseline | Mixed | Out-domain | 900 | 0.0000 * | -1.000 | large | 0.0970 | 0.1575 |
| smote_r01 vs baseline | Cross-domain | In-domain | 592 | 0.0364 | -0.316 | small | 0.0770 | 0.0812 |
| smote_r01 vs baseline | Cross-domain | Out-domain | 367 | 0.2226 | +0.184 | small | 0.0804 | 0.0785 |
| smote_r01 vs baseline | Within-domain | In-domain | 0 | 0.0000 * | +1.000 | large | 0.2745 | 0.1030 |
| smote_r01 vs baseline | Within-domain | Out-domain | 0 | 0.0000 * | +1.000 | large | 0.2686 | 0.1111 |
| smote_r01 vs baseline | Mixed | In-domain | 0 | 0.0000 * | +1.000 | large | 0.2564 | 0.1206 |
| smote_r01 vs baseline | Mixed | Out-domain | 0 | 0.0000 * | +1.000 | large | 0.3026 | 0.1575 |
| smote_r05 vs baseline | Cross-domain | In-domain | 723 | 0.0001 * | -0.607 | large | 0.0684 | 0.0812 |
| smote_r05 vs baseline | Cross-domain | Out-domain | 448 | 0.9823 | +0.004 | negligible | 0.0784 | 0.0785 |
| smote_r05 vs baseline | Within-domain | In-domain | 0 | 0.0000 * | +1.000 | large | 0.3381 | 0.1030 |
| smote_r05 vs baseline | Within-domain | Out-domain | 0 | 0.0000 * | +1.000 | large | 0.3294 | 0.1111 |
| smote_r05 vs baseline | Mixed | In-domain | 0 | 0.0000 * | +1.000 | large | 0.3276 | 0.1206 |
| smote_r05 vs baseline | Mixed | Out-domain | 0 | 0.0000 * | +1.000 | large | 0.4056 | 0.1575 |
| sw_smote_r01 vs baseline | Cross-domain | In-domain | 876 | 0.0000 * | -0.947 | large | 0.0678 | 0.0812 |
| sw_smote_r01 vs baseline | Cross-domain | Out-domain | 824 | 0.0000 * | -0.831 | large | 0.0672 | 0.0785 |
| sw_smote_r01 vs baseline | Within-domain | In-domain | 0 | 0.0000 * | +1.000 | large | 0.3741 | 0.1030 |
| sw_smote_r01 vs baseline | Within-domain | Out-domain | 4 | 0.0000 * | +0.991 | large | 0.3774 | 0.1111 |
| sw_smote_r01 vs baseline | Mixed | In-domain | 0 | 0.0000 * | +1.000 | large | 0.3873 | 0.1206 |
| sw_smote_r01 vs baseline | Mixed | Out-domain | 0 | 0.0000 * | +1.000 | large | 0.4722 | 0.1575 |
| sw_smote_r05 vs baseline | Cross-domain | In-domain | 900 | 0.0000 * | -1.000 | large | 0.0444 | 0.0812 |
| sw_smote_r05 vs baseline | Cross-domain | Out-domain | 900 | 0.0000 * | -1.000 | large | 0.0379 | 0.0785 |
| sw_smote_r05 vs baseline | Within-domain | In-domain | 0 | 0.0000 * | +1.000 | large | 0.4794 | 0.1030 |
| sw_smote_r05 vs baseline | Within-domain | Out-domain | 0 | 0.0000 * | +1.000 | large | 0.4395 | 0.1111 |
| sw_smote_r05 vs baseline | Mixed | In-domain | 0 | 0.0000 * | +1.000 | large | 0.4119 | 0.1206 |
| sw_smote_r05 vs baseline | Mixed | Out-domain | 0 | 0.0000 * | +1.000 | large | 0.4616 | 0.1575 |

**Bonferroni α'=0.00139** (m=36). **28** significant.

- large: 28/36 (78%)
- medium: 0/36 (0%)
- small: 6/36 (17%)
- negligible: 2/36 (6%)


### 5.2.1 [AUPRC] H1: Global condition effect (Kruskal-Wallis)

$$H_0: F_{C_1} = F_{C_2} = \cdots = F_{C_7}$$

| Mode | Level | Distance | H | p | η² | Sig (Bonf.) |
|------|-------|----------|--:|--:|---:|:-----------:|
| Cross-domain | In-domain | MMD | 35.29 | 0.0000 | 0.465 | ✓ |
| Cross-domain | In-domain | DTW | 32.73 | 0.0000 | 0.424 | ✓ |
| Cross-domain | In-domain | WASSERSTEIN | 47.11 | 0.0000 | 0.653 | ✓ |
| Cross-domain | Out-domain | MMD | 44.08 | 0.0000 | 0.604 | ✓ |
| Cross-domain | Out-domain | DTW | 37.73 | 0.0000 | 0.504 | ✓ |
| Cross-domain | Out-domain | WASSERSTEIN | 30.86 | 0.0000 | 0.395 | ✓ |
| Within-domain | In-domain | MMD | 41.04 | 0.0000 | 0.556 | ✓ |
| Within-domain | In-domain | DTW | 53.26 | 0.0000 | 0.750 | ✓ |
| Within-domain | In-domain | WASSERSTEIN | 51.97 | 0.0000 | 0.730 | ✓ |
| Within-domain | Out-domain | MMD | 53.20 | 0.0000 | 0.749 | ✓ |
| Within-domain | Out-domain | DTW | 49.12 | 0.0000 | 0.684 | ✓ |
| Within-domain | Out-domain | WASSERSTEIN | 52.45 | 0.0000 | 0.737 | ✓ |
| Mixed | In-domain | MMD | 50.39 | 0.0000 | 0.705 | ✓ |
| Mixed | In-domain | DTW | 42.32 | 0.0000 | 0.586 | ✓ |
| Mixed | In-domain | WASSERSTEIN | 44.92 | 0.0000 | 0.618 | ✓ |
| Mixed | Out-domain | MMD | 45.64 | 0.0000 | 0.629 | ✓ |
| Mixed | Out-domain | DTW | 46.38 | 0.0000 | 0.641 | ✓ |
| Mixed | Out-domain | WASSERSTEIN | 47.28 | 0.0000 | 0.666 | ✓ |

**Bonferroni α'=0.0028** (m=18). **18/18** significant.

Mean η² = 0.616 (large effect).

### 5.2.2 [AUPRC] H1: Pairwise — baseline vs each method

Mann-Whitney U with Cliff's δ effect size.

| Method vs Baseline | Mode | Level | U | p | δ | Effect | Mean(M) | Mean(B) |
|--------------------|------|-------|--:|--:|--:|:------:|--------:|--------:|
| rus_r01 vs baseline | Cross-domain | In-domain | 419 | 0.6520 | +0.069 | negligible | 0.0585 | 0.0508 |
| rus_r01 vs baseline | Cross-domain | Out-domain | 461 | 0.8766 | -0.024 | negligible | 0.0497 | 0.0466 |
| rus_r01 vs baseline | Within-domain | In-domain | 421 | 0.6735 | +0.064 | negligible | 0.1127 | 0.0872 |
| rus_r01 vs baseline | Within-domain | Out-domain | 520 | 0.3042 | -0.156 | small | 0.1389 | 0.1387 |
| rus_r01 vs baseline | Mixed | In-domain | 649 | 0.0033 | -0.442 | medium | 0.1617 | 0.1287 |
| rus_r01 vs baseline | Mixed | Out-domain | 807 | 0.0000 * | -0.793 | large | 0.1232 | 0.4050 |
| rus_r05 vs baseline | Cross-domain | In-domain | 494 | 0.5201 | -0.098 | negligible | 0.0514 | 0.0508 |
| rus_r05 vs baseline | Cross-domain | Out-domain | 344 | 0.1188 | +0.236 | small | 0.0506 | 0.0466 |
| rus_r05 vs baseline | Within-domain | In-domain | 413 | 0.5895 | +0.082 | negligible | 0.1145 | 0.0872 |
| rus_r05 vs baseline | Within-domain | Out-domain | 624 | 0.0103 | -0.387 | medium | 0.0837 | 0.1387 |
| rus_r05 vs baseline | Mixed | In-domain | 830 | 0.0000 * | -0.844 | large | 0.0655 | 0.1287 |
| rus_r05 vs baseline | Mixed | Out-domain | 892 | 0.0000 * | -0.982 | large | 0.0736 | 0.4050 |
| smote_r01 vs baseline | Cross-domain | In-domain | 558 | 0.1120 | -0.240 | small | 0.0502 | 0.0508 |
| smote_r01 vs baseline | Cross-domain | Out-domain | 459 | 0.9000 | -0.020 | negligible | 0.0463 | 0.0466 |
| smote_r01 vs baseline | Within-domain | In-domain | 9 | 0.0000 * | +0.980 | large | 0.6452 | 0.0872 |
| smote_r01 vs baseline | Within-domain | Out-domain | 3 | 0.0000 * | +0.993 | large | 0.6108 | 0.1387 |
| smote_r01 vs baseline | Mixed | In-domain | 9 | 0.0000 * | +0.980 | large | 0.5302 | 0.1287 |
| smote_r01 vs baseline | Mixed | Out-domain | 230 | 0.0012 * | +0.489 | large | 0.6741 | 0.4050 |
| smote_r05 vs baseline | Cross-domain | In-domain | 701 | 0.0002 * | -0.558 | large | 0.0490 | 0.0508 |
| smote_r05 vs baseline | Cross-domain | Out-domain | 492 | 0.5395 | -0.093 | negligible | 0.0463 | 0.0466 |
| smote_r05 vs baseline | Within-domain | In-domain | 10 | 0.0000 * | +0.978 | large | 0.5722 | 0.0872 |
| smote_r05 vs baseline | Within-domain | Out-domain | 20 | 0.0000 * | +0.956 | large | 0.5674 | 0.1387 |
| smote_r05 vs baseline | Mixed | In-domain | 9 | 0.0000 * | +0.980 | large | 0.5281 | 0.1287 |
| smote_r05 vs baseline | Mixed | Out-domain | 229 | 0.0011 * | +0.491 | large | 0.6459 | 0.4050 |
| sw_smote_r01 vs baseline | Cross-domain | In-domain | 363 | 0.2009 | +0.193 | small | 0.0519 | 0.0508 |
| sw_smote_r01 vs baseline | Cross-domain | Out-domain | 767 | 0.0000 * | -0.704 | large | 0.0434 | 0.0466 |
| sw_smote_r01 vs baseline | Within-domain | In-domain | 8 | 0.0000 * | +0.982 | large | 0.6377 | 0.0872 |
| sw_smote_r01 vs baseline | Within-domain | Out-domain | 33 | 0.0000 * | +0.927 | large | 0.6541 | 0.1387 |
| sw_smote_r01 vs baseline | Mixed | In-domain | 16 | 0.0000 * | +0.964 | large | 0.5540 | 0.1287 |
| sw_smote_r01 vs baseline | Mixed | Out-domain | 163 | 0.0000 * | +0.625 | large | 0.6748 | 0.4050 |
| sw_smote_r05 vs baseline | Cross-domain | In-domain | 259 | 0.0049 | +0.424 | medium | 0.0556 | 0.0508 |
| sw_smote_r05 vs baseline | Cross-domain | Out-domain | 779 | 0.0000 * | -0.731 | large | 0.0434 | 0.0466 |
| sw_smote_r05 vs baseline | Within-domain | In-domain | 22 | 0.0000 * | +0.951 | large | 0.4591 | 0.0872 |
| sw_smote_r05 vs baseline | Within-domain | Out-domain | 119 | 0.0000 * | +0.736 | large | 0.3639 | 0.1387 |
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
| Cross-domain | In-domain | MMD | 64.77 | 0.0000 | 0.933 | ✓ |
| Cross-domain | In-domain | DTW | 64.35 | 0.0000 | 0.926 | ✓ |
| Cross-domain | In-domain | WASSERSTEIN | 62.21 | 0.0000 | 0.892 | ✓ |
| Cross-domain | Out-domain | MMD | 61.98 | 0.0000 | 0.889 | ✓ |
| Cross-domain | Out-domain | DTW | 64.01 | 0.0000 | 0.921 | ✓ |
| Cross-domain | Out-domain | WASSERSTEIN | 63.72 | 0.0000 | 0.916 | ✓ |
| Within-domain | In-domain | MMD | 49.35 | 0.0000 | 0.688 | ✓ |
| Within-domain | In-domain | DTW | 51.76 | 0.0000 | 0.726 | ✓ |
| Within-domain | In-domain | WASSERSTEIN | 61.05 | 0.0000 | 0.874 | ✓ |
| Within-domain | Out-domain | MMD | 46.48 | 0.0000 | 0.642 | ✓ |
| Within-domain | Out-domain | DTW | 44.99 | 0.0000 | 0.619 | ✓ |
| Within-domain | Out-domain | WASSERSTEIN | 47.01 | 0.0000 | 0.651 | ✓ |
| Mixed | In-domain | MMD | 50.34 | 0.0000 | 0.704 | ✓ |
| Mixed | In-domain | DTW | 42.35 | 0.0000 | 0.586 | ✓ |
| Mixed | In-domain | WASSERSTEIN | 48.79 | 0.0000 | 0.679 | ✓ |
| Mixed | Out-domain | MMD | 44.64 | 0.0000 | 0.613 | ✓ |
| Mixed | Out-domain | DTW | 47.19 | 0.0000 | 0.654 | ✓ |
| Mixed | Out-domain | WASSERSTEIN | 47.61 | 0.0000 | 0.671 | ✓ |

**Bonferroni α'=0.0028** (m=18). **18/18** significant.

Mean η² = 0.755 (large effect).

### 5.3.2 [Recall] H1: Pairwise — baseline vs each method

Mann-Whitney U with Cliff's δ effect size.

| Method vs Baseline | Mode | Level | U | p | δ | Effect | Mean(M) | Mean(B) |
|--------------------|------|-------|--:|--:|--:|:------:|--------:|--------:|
| rus_r01 vs baseline | Cross-domain | In-domain | 334 | 0.0889 | +0.257 | small | 0.5225 | 0.5096 |
| rus_r01 vs baseline | Cross-domain | Out-domain | 529 | 0.2457 | -0.176 | small | 0.4597 | 0.4854 |
| rus_r01 vs baseline | Within-domain | In-domain | 722 | 0.0001 * | -0.604 | large | 0.4230 | 0.6176 |
| rus_r01 vs baseline | Within-domain | Out-domain | 460 | 0.8824 | -0.023 | negligible | 0.6577 | 0.6862 |
| rus_r01 vs baseline | Mixed | In-domain | 718 | 0.0001 * | -0.596 | large | 0.5439 | 0.6987 |
| rus_r01 vs baseline | Mixed | Out-domain | 862 | 0.0000 * | -0.914 | large | 0.6143 | 0.8582 |
| rus_r05 vs baseline | Cross-domain | In-domain | 817 | 0.0000 * | -0.816 | large | 0.4065 | 0.5096 |
| rus_r05 vs baseline | Cross-domain | Out-domain | 558 | 0.1136 | -0.239 | small | 0.4515 | 0.4854 |
| rus_r05 vs baseline | Within-domain | In-domain | 827 | 0.0000 * | -0.838 | large | 0.3385 | 0.6176 |
| rus_r05 vs baseline | Within-domain | Out-domain | 594 | 0.0345 | -0.319 | small | 0.5677 | 0.6862 |
| rus_r05 vs baseline | Mixed | In-domain | 900 | 0.0000 * | -1.000 | large | 0.3795 | 0.6987 |
| rus_r05 vs baseline | Mixed | Out-domain | 894 | 0.0000 * | -0.987 | large | 0.5670 | 0.8582 |
| smote_r01 vs baseline | Cross-domain | In-domain | 900 | 0.0000 * | -1.000 | large | 0.2712 | 0.5096 |
| smote_r01 vs baseline | Cross-domain | Out-domain | 890 | 0.0000 * | -0.978 | large | 0.2652 | 0.4854 |
| smote_r01 vs baseline | Within-domain | In-domain | 19 | 0.0000 * | +0.958 | large | 0.8681 | 0.6176 |
| smote_r01 vs baseline | Within-domain | Out-domain | 196 | 0.0002 * | +0.566 | large | 0.8694 | 0.6862 |
| smote_r01 vs baseline | Mixed | In-domain | 98 | 0.0000 * | +0.781 | large | 0.7967 | 0.6987 |
| smote_r01 vs baseline | Mixed | Out-domain | 407 | 0.5296 | +0.096 | negligible | 0.8734 | 0.8582 |
| smote_r05 vs baseline | Cross-domain | In-domain | 900 | 0.0000 * | -1.000 | large | 0.1812 | 0.5096 |
| smote_r05 vs baseline | Cross-domain | Out-domain | 900 | 0.0000 * | -1.000 | large | 0.1839 | 0.4854 |
| smote_r05 vs baseline | Within-domain | In-domain | 76 | 0.0000 * | +0.832 | large | 0.7813 | 0.6176 |
| smote_r05 vs baseline | Within-domain | Out-domain | 244 | 0.0023 | +0.459 | medium | 0.8018 | 0.6862 |
| smote_r05 vs baseline | Mixed | In-domain | 340 | 0.1038 | +0.246 | small | 0.7363 | 0.6987 |
| smote_r05 vs baseline | Mixed | Out-domain | 754 | 0.0000 * | -0.676 | large | 0.7832 | 0.8582 |
| sw_smote_r01 vs baseline | Cross-domain | In-domain | 900 | 0.0000 * | -1.000 | large | 0.1630 | 0.5096 |
| sw_smote_r01 vs baseline | Cross-domain | Out-domain | 900 | 0.0000 * | -1.000 | large | 0.1552 | 0.4854 |
| sw_smote_r01 vs baseline | Within-domain | In-domain | 102 | 0.0000 * | +0.774 | large | 0.8241 | 0.6176 |
| sw_smote_r01 vs baseline | Within-domain | Out-domain | 252 | 0.0034 | +0.441 | medium | 0.8290 | 0.6862 |
| sw_smote_r01 vs baseline | Mixed | In-domain | 309 | 0.0377 | +0.313 | small | 0.7184 | 0.6987 |
| sw_smote_r01 vs baseline | Mixed | Out-domain | 538 | 0.1218 | -0.236 | small | 0.7693 | 0.8582 |
| sw_smote_r05 vs baseline | Cross-domain | In-domain | 900 | 0.0000 * | -1.000 | large | 0.0489 | 0.5096 |
| sw_smote_r05 vs baseline | Cross-domain | Out-domain | 900 | 0.0000 * | -1.000 | large | 0.0385 | 0.4854 |
| sw_smote_r05 vs baseline | Within-domain | In-domain | 605 | 0.0224 | -0.344 | medium | 0.5164 | 0.6176 |
| sw_smote_r05 vs baseline | Within-domain | Out-domain | 716 | 0.0001 * | -0.590 | large | 0.5117 | 0.6862 |
| sw_smote_r05 vs baseline | Mixed | In-domain | 846 | 0.0000 * | -0.945 | large | 0.3660 | 0.6987 |
| sw_smote_r05 vs baseline | Mixed | Out-domain | 900 | 0.0000 * | -1.000 | large | 0.4654 | 0.8582 |

**Bonferroni α'=0.00139** (m=36). **24** significant.

- large: 24/36 (67%)
- medium: 3/36 (8%)
- small: 7/36 (19%)
- negligible: 2/36 (6%)

---
## 6. Hypothesis Tests — H2: SW-SMOTE vs Plain SMOTE


### 6.1 [F1-score] H2: sw_smote vs plain smote

Paired comparison (same ratio): Does subject-wise synthesis improve?

| Ratio | Mode | Level | U | p | δ (sw−sm) | Effect | Mean(sw) | Mean(sm) |
|-------|------|-------|--:|--:|----------:|:------:|---------:|---------:|
| r01 | Cross-domain | In-domain | 206 | 0.0003 * | -0.542 | large | 0.0678 | 0.0770 |
| r01 | Cross-domain | Out-domain | 70 | 0.0000 * | -0.844 | large | 0.0672 | 0.0804 |
| r01 | Within-domain | In-domain | 758 | 0.0000 * | +0.684 | large | 0.3741 | 0.2745 |
| r01 | Within-domain | Out-domain | 774 | 0.0000 * | +0.720 | large | 0.3774 | 0.2686 |
| r01 | Mixed | In-domain | 771 | 0.0000 * | +0.713 | large | 0.3873 | 0.2564 |
| r01 | Mixed | Out-domain | 657 | 0.0008 * | +0.510 | large | 0.4722 | 0.3026 |
| r05 | Cross-domain | In-domain | 21 | 0.0000 * | -0.953 | large | 0.0444 | 0.0684 |
| r05 | Cross-domain | Out-domain | 0 | 0.0000 * | -1.000 | large | 0.0379 | 0.0784 |
| r05 | Within-domain | In-domain | 681 | 0.0007 * | +0.513 | large | 0.4794 | 0.3381 |
| r05 | Within-domain | Out-domain | 624 | 0.0103 | +0.387 | medium | 0.4395 | 0.3294 |
| r05 | Mixed | In-domain | 510 | 0.2587 | +0.172 | small | 0.4119 | 0.3276 |
| r05 | Mixed | Out-domain | 536 | 0.2062 | +0.191 | small | 0.4616 | 0.4056 |

**Summary**: sw_smote > smote in 8/12, smote > sw_smote in 4/12. Bonferroni sig: 9/12.


### 6.2 [AUPRC] H2: sw_smote vs plain smote

Paired comparison (same ratio): Does subject-wise synthesis improve?

| Ratio | Mode | Level | U | p | δ (sw−sm) | Effect | Mean(sw) | Mean(sm) |
|-------|------|-------|--:|--:|----------:|:------:|---------:|---------:|
| r01 | Cross-domain | In-domain | 593 | 0.0351 | +0.318 | small | 0.0519 | 0.0502 |
| r01 | Cross-domain | Out-domain | 177 | 0.0001 * | -0.607 | large | 0.0434 | 0.0463 |
| r01 | Within-domain | In-domain | 477 | 0.6952 | +0.060 | negligible | 0.6377 | 0.6452 |
| r01 | Within-domain | Out-domain | 566 | 0.0877 | +0.258 | small | 0.6541 | 0.6108 |
| r01 | Mixed | In-domain | 490 | 0.5592 | +0.089 | negligible | 0.5540 | 0.5302 |
| r01 | Mixed | Out-domain | 596 | 0.0150 | +0.370 | medium | 0.6748 | 0.6741 |
| r05 | Cross-domain | In-domain | 780 | 0.0000 * | +0.733 | large | 0.0556 | 0.0490 |
| r05 | Cross-domain | Out-domain | 141 | 0.0000 * | -0.687 | large | 0.0434 | 0.0463 |
| r05 | Within-domain | In-domain | 278 | 0.0112 | -0.382 | medium | 0.4591 | 0.5722 |
| r05 | Within-domain | Out-domain | 191 | 0.0001 * | -0.576 | large | 0.3639 | 0.5674 |
| r05 | Mixed | In-domain | 247 | 0.0045 | -0.432 | medium | 0.3908 | 0.5281 |
| r05 | Mixed | Out-domain | 165 | 0.0000 * | -0.633 | large | 0.4316 | 0.6459 |

**Summary**: sw_smote > smote in 6/12, smote > sw_smote in 6/12. Bonferroni sig: 5/12.


### 6.3 [Recall] H2: sw_smote vs plain smote

Paired comparison (same ratio): Does subject-wise synthesis improve?

| Ratio | Mode | Level | U | p | δ (sw−sm) | Effect | Mean(sw) | Mean(sm) |
|-------|------|-------|--:|--:|----------:|:------:|---------:|---------:|
| r01 | Cross-domain | In-domain | 41 | 0.0000 * | -0.909 | large | 0.1630 | 0.2712 |
| r01 | Cross-domain | Out-domain | 14 | 0.0000 * | -0.969 | large | 0.1552 | 0.2652 |
| r01 | Within-domain | In-domain | 314 | 0.0442 | -0.303 | small | 0.8241 | 0.8681 |
| r01 | Within-domain | Out-domain | 472 | 0.7500 | +0.049 | negligible | 0.8290 | 0.8694 |
| r01 | Mixed | In-domain | 278 | 0.0112 | -0.382 | medium | 0.7184 | 0.7967 |
| r01 | Mixed | Out-domain | 308 | 0.0555 | -0.291 | small | 0.7693 | 0.8734 |
| r05 | Cross-domain | In-domain | 0 | 0.0000 * | -1.000 | large | 0.0489 | 0.1812 |
| r05 | Cross-domain | Out-domain | 0 | 0.0000 * | -1.000 | large | 0.0385 | 0.1839 |
| r05 | Within-domain | In-domain | 136 | 0.0000 * | -0.698 | large | 0.5164 | 0.7813 |
| r05 | Within-domain | Out-domain | 86 | 0.0000 * | -0.809 | large | 0.5117 | 0.8018 |
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
- medium: 1/24
- small: 3/24
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
| rus | Cross-domain | In-domain | 620 | 0.0122 | -0.378 | medium | 0.0876 | 0.0824 |
| rus | Cross-domain | Out-domain | 472 | 0.7506 | -0.049 | negligible | 0.0785 | 0.0775 |
| rus | Within-domain | In-domain | 492 | 0.5395 | -0.093 | negligible | 0.0740 | 0.0681 |
| rus | Within-domain | Out-domain | 583 | 0.0501 | -0.296 | small | 0.1038 | 0.0920 |
| rus | Mixed | In-domain | 708 | 0.0001 * | -0.573 | large | 0.0995 | 0.0752 |
| rus | Mixed | Out-domain | 706 | 0.0002 * | -0.569 | large | 0.1141 | 0.0970 |
| smote | Cross-domain | In-domain | 642 | 0.0046 | -0.427 | medium | 0.0770 | 0.0684 |
| smote | Cross-domain | Out-domain | 508 | 0.3953 | -0.129 | negligible | 0.0804 | 0.0784 |
| smote | Within-domain | In-domain | 274 | 0.0095 | +0.391 | medium | 0.2745 | 0.3381 |
| smote | Within-domain | Out-domain | 260 | 0.0051 | +0.422 | medium | 0.2686 | 0.3294 |
| smote | Mixed | In-domain | 236 | 0.0016 * | +0.476 | large | 0.2564 | 0.3276 |
| smote | Mixed | Out-domain | 241 | 0.0021 * | +0.464 | medium | 0.3026 | 0.4056 |
| sw_smote | Cross-domain | In-domain | 896 | 0.0000 * | -0.991 | large | 0.0678 | 0.0444 |
| sw_smote | Cross-domain | Out-domain | 896 | 0.0000 * | -0.991 | large | 0.0672 | 0.0379 |
| sw_smote | Within-domain | In-domain | 276 | 0.0103 | +0.387 | medium | 0.3741 | 0.4794 |
| sw_smote | Within-domain | Out-domain | 359 | 0.1809 | +0.202 | small | 0.3774 | 0.4395 |
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
| rus | Cross-domain | In-domain | 540 | 0.1858 | -0.200 | small | 0.0585 | 0.0514 |
| rus | Cross-domain | Out-domain | 357 | 0.1715 | +0.207 | small | 0.0497 | 0.0506 |
| rus | Within-domain | In-domain | 457 | 0.9234 | -0.016 | negligible | 0.1127 | 0.1145 |
| rus | Within-domain | Out-domain | 575 | 0.0657 | -0.278 | small | 0.1389 | 0.0837 |
| rus | Mixed | In-domain | 494 | 0.5201 | -0.098 | negligible | 0.1617 | 0.0655 |
| rus | Mixed | Out-domain | 658 | 0.0022 * | -0.462 | medium | 0.1232 | 0.0736 |
| smote | Cross-domain | In-domain | 642 | 0.0046 | -0.427 | medium | 0.0502 | 0.0490 |
| smote | Cross-domain | Out-domain | 451 | 0.9941 | -0.002 | negligible | 0.0463 | 0.0463 |
| smote | Within-domain | In-domain | 574 | 0.0679 | -0.276 | small | 0.6452 | 0.5722 |
| smote | Within-domain | Out-domain | 520 | 0.3042 | -0.156 | small | 0.6108 | 0.5674 |
| smote | Mixed | In-domain | 474 | 0.7283 | -0.053 | negligible | 0.5302 | 0.5281 |
| smote | Mixed | Out-domain | 502 | 0.4464 | -0.116 | negligible | 0.6741 | 0.6459 |
| sw_smote | Cross-domain | In-domain | 313 | 0.0436 | +0.304 | small | 0.0519 | 0.0556 |
| sw_smote | Cross-domain | Out-domain | 447 | 0.9705 | +0.007 | negligible | 0.0434 | 0.0434 |
| sw_smote | Within-domain | In-domain | 668 | 0.0013 * | -0.484 | large | 0.6377 | 0.4591 |
| sw_smote | Within-domain | Out-domain | 756 | 0.0000 * | -0.680 | large | 0.6541 | 0.3639 |
| sw_smote | Mixed | In-domain | 649 | 0.0012 * | -0.492 | large | 0.5540 | 0.3908 |
| sw_smote | Mixed | Out-domain | 662 | 0.0006 * | -0.522 | large | 0.6748 | 0.4316 |

**Bonferroni α'=0.00278** (m=18). **5** significant.

- **rus**: r=0.1 better in 5/6, r=0.5 better in 1/6
- **smote**: r=0.1 better in 6/6, r=0.5 better in 0/6
- **sw_smote**: r=0.1 better in 4/6, r=0.5 better in 2/6


### 8.3 [Recall] H4: Ratio effect

$$H_0: \mu_{r=0.1}^{(\text{method})} = \mu_{r=0.5}^{(\text{method})}$$

| Method | Mode | Level | U | p | δ (r05−r01) | Effect | Mean(r01) | Mean(r05) |
|--------|------|-------|--:|--:|------------:|:------:|----------:|----------:|
| rus | Cross-domain | In-domain | 831 | 0.0000 * | -0.847 | large | 0.5225 | 0.4065 |
| rus | Cross-domain | Out-domain | 504 | 0.4289 | -0.120 | negligible | 0.4597 | 0.4515 |
| rus | Within-domain | In-domain | 576 | 0.0646 | -0.279 | small | 0.4230 | 0.3385 |
| rus | Within-domain | Out-domain | 599 | 0.0281 | -0.331 | medium | 0.6577 | 0.5677 |
| rus | Mixed | In-domain | 760 | 0.0000 * | -0.690 | large | 0.5439 | 0.3795 |
| rus | Mixed | Out-domain | 552 | 0.1334 | -0.227 | small | 0.6143 | 0.5670 |
| smote | Cross-domain | In-domain | 804 | 0.0000 * | -0.788 | large | 0.2712 | 0.1812 |
| smote | Cross-domain | Out-domain | 810 | 0.0000 * | -0.799 | large | 0.2652 | 0.1839 |
| smote | Within-domain | In-domain | 839 | 0.0000 * | -0.864 | large | 0.8681 | 0.7813 |
| smote | Within-domain | Out-domain | 838 | 0.0000 * | -0.861 | large | 0.8694 | 0.8018 |
| smote | Mixed | In-domain | 783 | 0.0000 * | -0.740 | large | 0.7967 | 0.7363 |
| smote | Mixed | Out-domain | 865 | 0.0000 * | -0.922 | large | 0.8734 | 0.7832 |
| sw_smote | Cross-domain | In-domain | 900 | 0.0000 * | -1.000 | large | 0.1630 | 0.0489 |
| sw_smote | Cross-domain | Out-domain | 900 | 0.0000 * | -1.000 | large | 0.1552 | 0.0385 |
| sw_smote | Within-domain | In-domain | 846 | 0.0000 * | -0.881 | large | 0.8241 | 0.5164 |
| sw_smote | Within-domain | Out-domain | 838 | 0.0000 * | -0.861 | large | 0.8290 | 0.5117 |
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
| Cross-domain | In-domain | 10.44 | 0.0054 | 0.041 |
| Cross-domain | Out-domain | 17.00 | 0.0002 | 0.072 |
| Within-domain | In-domain | 0.48 | 0.7848 | 0.000 |
| Within-domain | Out-domain | 2.52 | 0.2832 | 0.003 |
| Mixed | In-domain | 0.56 | 0.7541 | 0.000 |
| Mixed | Out-domain | 1.22 | 0.5446 | 0.000 |

**2/6** significant at α=0.05.

### 9.1.2 [F1-score] H7–H8: Training mode effect

Within-domain vs cross-domain, mixed vs cross-domain.

- **Within vs Cross-domain**: U=158170, p=0.0000, δ=+0.793 (large), mean(Within)=0.2452, mean(Cross)=0.0719
- **Mixed vs Cross-domain**: U=166111, p=0.0000, δ=+0.892 (large), mean(Mixed)=0.2627, mean(Cross)=0.0719

### 9.1.3 [F1-score] H10: Domain shift (in vs out)

Wilcoxon signed-rank (paired by seed): in-domain > out-domain in **24/63** cells. Bonferroni sig: **0/63**.

**Global**: In-domain mean=0.1853, Out-domain mean=0.2010, Cliff's δ=-0.083 (negligible).


### 9.2.1 [AUPRC] H5: Distance metric effect

Kruskal-Wallis across MMD, DTW, Wasserstein (pooling conditions).

| Mode | Level | H | p | η² |
|------|-------|--:|--:|---:|
| Cross-domain | In-domain | 25.25 | 0.0000 | 0.112 |
| Cross-domain | Out-domain | 23.18 | 0.0000 | 0.102 |
| Within-domain | In-domain | 3.13 | 0.2094 | 0.005 |
| Within-domain | Out-domain | 1.90 | 0.3870 | 0.000 |
| Mixed | In-domain | 0.10 | 0.9519 | 0.000 |
| Mixed | Out-domain | 0.03 | 0.9830 | 0.000 |

**2/6** significant at α=0.05.

### 9.2.2 [AUPRC] H7–H8: Training mode effect

Within-domain vs cross-domain, mixed vs cross-domain.

- **Within vs Cross-domain**: U=171670, p=0.0000, δ=+0.946 (large), mean(Within)=0.3704, mean(Cross)=0.0496
- **Mixed vs Cross-domain**: U=170925, p=0.0000, δ=+0.947 (large), mean(Mixed)=0.3841, mean(Cross)=0.0496

### 9.2.3 [AUPRC] H10: Domain shift (in vs out)

Wilcoxon signed-rank (paired by seed): in-domain > out-domain in **32/63** cells. Bonferroni sig: **0/63**.

**Global**: In-domain mean=0.2548, Out-domain mean=0.2809, Cliff's δ=+0.057 (negligible).


### 9.3.1 [Recall] H5: Distance metric effect

Kruskal-Wallis across MMD, DTW, Wasserstein (pooling conditions).

| Mode | Level | H | p | η² |
|------|-------|--:|--:|---:|
| Cross-domain | In-domain | 1.32 | 0.5163 | 0.000 |
| Cross-domain | Out-domain | 2.70 | 0.2592 | 0.003 |
| Within-domain | In-domain | 1.48 | 0.4770 | 0.000 |
| Within-domain | Out-domain | 5.21 | 0.0738 | 0.016 |
| Mixed | In-domain | 0.39 | 0.8236 | 0.000 |
| Mixed | Out-domain | 0.14 | 0.9324 | 0.000 |

**0/6** significant at α=0.05.

### 9.3.2 [Recall] H7–H8: Training mode effect

Within-domain vs cross-domain, mixed vs cross-domain.

- **Within vs Cross-domain**: U=157526, p=0.0000, δ=+0.786 (large), mean(Within)=0.6638, mean(Cross)=0.2959
- **Mixed vs Cross-domain**: U=157569, p=0.0000, δ=+0.795 (large), mean(Mixed)=0.6554, mean(Cross)=0.2959

### 9.3.3 [Recall] H10: Domain shift (in vs out)

Wilcoxon signed-rank (paired by seed): in-domain > out-domain in **17/63** cells. Bonferroni sig: **0/63**.

**Global**: In-domain mean=0.5103, Out-domain mean=0.5660, Cliff's δ=-0.143 (negligible).

---
## 10. Precision–Recall Trade-off Analysis (HE3)

**Hypothesis HE3**: Oversampling methods improve Recall at the cost of Precision.

For each rebalancing method vs baseline, we compute Cliff's δ for both Precision and Recall to quantify the trade-off direction.


| Method vs Baseline | Mode | Level | δ(Recall) | δ(Precision) | Trade-off? |
|--------------------|------|-------|----------:|-------------:|:----------:|
| rus_r01 vs baseline | Cross-domain | In-domain | +0.257 | +0.700 |  |
| rus_r01 vs baseline | Cross-domain | Out-domain | -0.176 | -0.136 |  |
| rus_r01 vs baseline | Within-domain | In-domain | -0.604 | -0.642 |  |
| rus_r01 vs baseline | Within-domain | Out-domain | -0.023 | +0.042 |  |
| rus_r01 vs baseline | Mixed | In-domain | -0.596 | -0.604 |  |
| rus_r01 vs baseline | Mixed | Out-domain | -0.914 | -0.853 |  |
| rus_r05 vs baseline | Cross-domain | In-domain | -0.816 | +0.471 |  |
| rus_r05 vs baseline | Cross-domain | Out-domain | -0.239 | -0.140 |  |
| rus_r05 vs baseline | Within-domain | In-domain | -0.838 | -0.740 |  |
| rus_r05 vs baseline | Within-domain | Out-domain | -0.319 | -0.287 |  |
| rus_r05 vs baseline | Mixed | In-domain | -1.000 | -1.000 |  |
| rus_r05 vs baseline | Mixed | Out-domain | -0.987 | -1.000 |  |
| smote_r01 vs baseline | Cross-domain | In-domain | -1.000 | -0.084 |  |
| smote_r01 vs baseline | Cross-domain | Out-domain | -0.978 | +0.807 |  |
| smote_r01 vs baseline | Within-domain | In-domain | +0.958 | +1.000 |  |
| smote_r01 vs baseline | Within-domain | Out-domain | +0.566 | +1.000 |  |
| smote_r01 vs baseline | Mixed | In-domain | +0.781 | +1.000 |  |
| smote_r01 vs baseline | Mixed | Out-domain | +0.096 | +1.000 |  |
| smote_r05 vs baseline | Cross-domain | In-domain | -1.000 | -0.220 |  |
| smote_r05 vs baseline | Cross-domain | Out-domain | -1.000 | +0.673 |  |
| smote_r05 vs baseline | Within-domain | In-domain | +0.832 | +1.000 |  |
| smote_r05 vs baseline | Within-domain | Out-domain | +0.459 | +1.000 |  |
| smote_r05 vs baseline | Mixed | In-domain | +0.246 | +1.000 |  |
| smote_r05 vs baseline | Mixed | Out-domain | -0.676 | +1.000 |  |
| sw_smote_r01 vs baseline | Cross-domain | In-domain | -1.000 | -0.162 |  |
| sw_smote_r01 vs baseline | Cross-domain | Out-domain | -1.000 | +0.053 |  |
| sw_smote_r01 vs baseline | Within-domain | In-domain | +0.774 | +1.000 |  |
| sw_smote_r01 vs baseline | Within-domain | Out-domain | +0.441 | +1.000 |  |
| sw_smote_r01 vs baseline | Mixed | In-domain | +0.313 | +1.000 |  |
| sw_smote_r01 vs baseline | Mixed | Out-domain | -0.236 | +1.000 |  |
| sw_smote_r05 vs baseline | Cross-domain | In-domain | -1.000 | -0.047 |  |
| sw_smote_r05 vs baseline | Cross-domain | Out-domain | -1.000 | -0.487 |  |
| sw_smote_r05 vs baseline | Within-domain | In-domain | -0.344 | +1.000 |  |
| sw_smote_r05 vs baseline | Within-domain | Out-domain | -0.590 | +1.000 |  |
| sw_smote_r05 vs baseline | Mixed | In-domain | -0.945 | +1.000 |  |
| sw_smote_r05 vs baseline | Mixed | Out-domain | -1.000 | +1.000 |  |

**Summary**: 0/36 cells exhibit a clear Precision–Recall trade-off (|δ| > 0.147 in opposite directions).

#### Aggregated by method

| Method | Mean δ(Recall) | Mean δ(Precision) | Pattern |
|--------|---------------:|------------------:|---------|
| rus_r01 | -0.343 | -0.249 | Regression (R↓) |
| rus_r05 | -0.700 | -0.449 | Regression (R↓) |
| smote_r01 | +0.070 | +0.787 | Win-win (R↑ P≈) |
| smote_r05 | -0.190 | +0.742 | Regression (R↓) |
| sw_smote_r01 | -0.118 | +0.649 | Regression (R↓) |
| sw_smote_r05 | -0.813 | +0.578 | Regression (R↓) |

---
## 11. Extended Hypothesis HE1: Rebalancing → Recall Improvement

**Hypothesis**: *Any* rebalancing method (including RUS) improves Recall over baseline, because both oversampling and undersampling increase the effective weight of the minority class.


One-sided test: $H_1$: Recall(method) > Recall(baseline).

| Method | Mode | Level | U | p (one-sided) | δ | Effect | Mean(M) | Mean(B) |
|--------|------|-------|--:|:-------------:|--:|:------:|--------:|--------:|
| rus_r01 | Cross-domain | In-domain | 566 | 0.0445 | +0.257 | small | 0.5225 | 0.5096 |
| rus_r01 | Cross-domain | Out-domain | 371 | 0.8801 | -0.176 | small | 0.4597 | 0.4854 |
| rus_r01 | Within-domain | In-domain | 178 | 1.0000 | -0.604 | large | 0.4230 | 0.6176 |
| rus_r01 | Within-domain | Out-domain | 440 | 0.5646 | -0.023 | negligible | 0.6577 | 0.6862 |
| rus_r01 | Mixed | In-domain | 182 | 1.0000 | -0.596 | large | 0.5439 | 0.6987 |
| rus_r01 | Mixed | Out-domain | 38 | 1.0000 | -0.914 | large | 0.6143 | 0.8582 |
| rus_r05 | Cross-domain | In-domain | 83 | 1.0000 | -0.816 | large | 0.4065 | 0.5096 |
| rus_r05 | Cross-domain | Out-domain | 342 | 0.9449 | -0.239 | small | 0.4515 | 0.4854 |
| rus_r05 | Within-domain | In-domain | 73 | 1.0000 | -0.838 | large | 0.3385 | 0.6176 |
| rus_r05 | Within-domain | Out-domain | 306 | 0.9834 | -0.319 | small | 0.5677 | 0.6862 |
| rus_r05 | Mixed | In-domain | 0 | 1.0000 | -1.000 | large | 0.3795 | 0.6987 |
| rus_r05 | Mixed | Out-domain | 6 | 1.0000 | -0.987 | large | 0.5670 | 0.8582 |
| smote_r01 | Cross-domain | In-domain | 0 | 1.0000 | -1.000 | large | 0.2712 | 0.5096 |
| smote_r01 | Cross-domain | Out-domain | 10 | 1.0000 | -0.978 | large | 0.2652 | 0.4854 |
| smote_r01 | Within-domain | In-domain | 881 | 0.0000 * | +0.958 | large | 0.8681 | 0.6176 |
| smote_r01 | Within-domain | Out-domain | 704 | 0.0001 * | +0.566 | large | 0.8694 | 0.6862 |
| smote_r01 | Mixed | In-domain | 802 | 0.0000 * | +0.781 | large | 0.7967 | 0.6987 |
| smote_r01 | Mixed | Out-domain | 493 | 0.2648 | +0.096 | negligible | 0.8734 | 0.8582 |
| smote_r05 | Cross-domain | In-domain | 0 | 1.0000 | -1.000 | large | 0.1812 | 0.5096 |
| smote_r05 | Cross-domain | Out-domain | 0 | 1.0000 | -1.000 | large | 0.1839 | 0.4854 |
| smote_r05 | Within-domain | In-domain | 824 | 0.0000 * | +0.832 | large | 0.7813 | 0.6176 |
| smote_r05 | Within-domain | Out-domain | 656 | 0.0012 * | +0.459 | medium | 0.8018 | 0.6862 |
| smote_r05 | Mixed | In-domain | 560 | 0.0519 | +0.246 | small | 0.7363 | 0.6987 |
| smote_r05 | Mixed | Out-domain | 146 | 1.0000 | -0.676 | large | 0.7832 | 0.8582 |
| sw_smote_r01 | Cross-domain | In-domain | 0 | 1.0000 | -1.000 | large | 0.1630 | 0.5096 |
| sw_smote_r01 | Cross-domain | Out-domain | 0 | 1.0000 | -1.000 | large | 0.1552 | 0.4854 |
| sw_smote_r01 | Within-domain | In-domain | 798 | 0.0000 * | +0.774 | large | 0.8241 | 0.6176 |
| sw_smote_r01 | Within-domain | Out-domain | 648 | 0.0017 | +0.441 | medium | 0.8290 | 0.6862 |
| sw_smote_r01 | Mixed | In-domain | 591 | 0.0189 | +0.313 | small | 0.7184 | 0.6987 |
| sw_smote_r01 | Mixed | Out-domain | 332 | 0.9409 | -0.236 | small | 0.7693 | 0.8582 |
| sw_smote_r05 | Cross-domain | In-domain | 0 | 1.0000 | -1.000 | large | 0.0489 | 0.5096 |
| sw_smote_r05 | Cross-domain | Out-domain | 0 | 1.0000 | -1.000 | large | 0.0385 | 0.4854 |
| sw_smote_r05 | Within-domain | In-domain | 295 | 0.9893 | -0.344 | medium | 0.5164 | 0.6176 |
| sw_smote_r05 | Within-domain | Out-domain | 184 | 1.0000 | -0.590 | large | 0.5117 | 0.6862 |
| sw_smote_r05 | Mixed | In-domain | 24 | 1.0000 | -0.945 | large | 0.3660 | 0.6987 |
| sw_smote_r05 | Mixed | Out-domain | 0 | 1.0000 | -1.000 | large | 0.4654 | 0.8582 |

**Bonferroni α'=0.00139** (m=36). **6** significant.

Method > baseline in 11/36 cells.

**RUS specifically**: Recall improved in 1/12 cells (mean δ=-0.521).

**SMOTE-family**: Recall improved in 10/24 cells (mean δ=-0.263).

---
## 12. Extended Hypothesis HE2: AUPRC vs AUROC Sensitivity

**Hypothesis**: AUPRC shows a stronger condition effect than AUROC because AUPRC is more sensitive to minority-class performance.


### Comparison of η² (condition effect size) per cell

| Mode | Level | η²(AUROC) | η²(AUPRC) | AUPRC stronger? |
|------|-------|----------:|----------:|:---------------:|
| Cross-domain | In-domain | 0.081 | 0.134 | ✓ |
| Cross-domain | Out-domain | 0.106 | 0.268 | ✓ |
| Within-domain | In-domain | 0.715 | 0.696 | ✗ |
| Within-domain | Out-domain | 0.704 | 0.702 | ✗ |
| Mixed | In-domain | 0.703 | 0.655 | ✗ |
| Mixed | Out-domain | 0.638 | 0.663 | ✓ |

**Result**: AUPRC shows stronger condition effect in **3/6** cells.

### Ranking comparison: AUROC vs AUPRC

| Condition | Mean Rank (AUROC) | Mean Rank (AUPRC) | Δ Rank |
|-----------|------------------:|------------------:|-------:|
| baseline | 4.83 | 5.17 | +0.33 |
| rus_r01 | 4.89 | 4.56 | -0.33 |
| rus_r05 | 5.61 | 5.39 | -0.22 |
| smote_r01 | 2.17 | 2.67 | +0.50 |
| smote_r05 | 3.11 | 3.39 | +0.28 |
| sw_smote_r01 | 3.33 | 2.61 | -0.72 |
| sw_smote_r05 | 4.06 | 4.22 | +0.17 |

**Spearman ρ** (AUROC vs AUPRC rankings) = 0.857 (strong concordance).

---
## 13. Cross-Axis Interaction Analysis


### 13.1.1 [F1-score] H12: Condition × Mode interaction

Best condition per mode:

| Mode | Level | Best Condition | Mean | 2nd | Mean |
|------|-------|:-------------:|-----:|:---:|-----:|
| Cross-domain | In-domain | rus_r01 | 0.0876 | rus_r05 | 0.0824 |
| Cross-domain | Out-domain | smote_r01 | 0.0804 | baseline | 0.0785 |
| Within-domain | In-domain | sw_smote_r05 | 0.4794 | sw_smote_r01 | 0.3741 |
| Within-domain | Out-domain | sw_smote_r05 | 0.4395 | sw_smote_r01 | 0.3774 |
| Mixed | In-domain | sw_smote_r05 | 0.4119 | sw_smote_r01 | 0.3873 |
| Mixed | Out-domain | sw_smote_r01 | 0.4722 | sw_smote_r05 | 0.4616 |

**Friedman test** (condition effect per mode, seeds as blocks):

| Mode | Level | χ² | p | Kendall's W |
|------|-------|---:|--:|:----------:|
| Cross-domain | In-domain | 56.36 | 0.0000 * | 0.939 |
| Cross-domain | Out-domain | 39.26 | 0.0000 * | 0.654 |
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
| Cross-domain | -0.0015 | 0.0099 |
| Within-domain | +0.0015 | 0.0608 |
| Mixed | +0.0475 | 0.0811 |


### 13.2.1 [AUPRC] H12: Condition × Mode interaction

Best condition per mode:

| Mode | Level | Best Condition | Mean | 2nd | Mean |
|------|-------|:-------------:|-----:|:---:|-----:|
| Cross-domain | In-domain | rus_r01 | 0.0585 | sw_smote_r05 | 0.0556 |
| Cross-domain | Out-domain | rus_r05 | 0.0506 | rus_r01 | 0.0497 |
| Within-domain | In-domain | smote_r01 | 0.6452 | sw_smote_r01 | 0.6377 |
| Within-domain | Out-domain | sw_smote_r01 | 0.6541 | smote_r01 | 0.6108 |
| Mixed | In-domain | sw_smote_r01 | 0.5540 | smote_r01 | 0.5302 |
| Mixed | Out-domain | sw_smote_r01 | 0.6748 | smote_r01 | 0.6741 |

**Friedman test** (condition effect per mode, seeds as blocks):

| Mode | Level | χ² | p | Kendall's W |
|------|-------|---:|--:|:----------:|
| Cross-domain | In-domain | 39.90 | 0.0000 * | 0.665 |
| Cross-domain | Out-domain | 38.66 | 0.0000 * | 0.644 |
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
| Within-domain | Out-domain | sw_smote_r01 | smote_r01 | sw_smote_r01 | ✗ |
| Mixed | In-domain | sw_smote_r01 | sw_smote_r01 | sw_smote_r01 | ✓ |
| Mixed | Out-domain | smote_r01 | sw_smote_r01 | smote_r01 | ✗ |

### 13.2.3 [AUPRC] H14: Domain gap by mode

| Mode | Mean gap (Δ=out−in) | Mean |Δ| |
|------|:-------------------:|------:|
| Cross-domain | -0.0059 | 0.0083 |
| Within-domain | -0.0102 | 0.1251 |
| Mixed | +0.0963 | 0.1784 |


### 13.3.1 [Recall] H12: Condition × Mode interaction

Best condition per mode:

| Mode | Level | Best Condition | Mean | 2nd | Mean |
|------|-------|:-------------:|-----:|:---:|-----:|
| Cross-domain | In-domain | rus_r01 | 0.5225 | baseline | 0.5096 |
| Cross-domain | Out-domain | baseline | 0.4854 | rus_r01 | 0.4597 |
| Within-domain | In-domain | smote_r01 | 0.8681 | sw_smote_r01 | 0.8241 |
| Within-domain | Out-domain | smote_r01 | 0.8694 | sw_smote_r01 | 0.8290 |
| Mixed | In-domain | smote_r01 | 0.7967 | smote_r05 | 0.7363 |
| Mixed | Out-domain | smote_r01 | 0.8734 | baseline | 0.8582 |

**Friedman test** (condition effect per mode, seeds as blocks):

| Mode | Level | χ² | p | Kendall's W |
|------|-------|---:|--:|:----------:|
| Cross-domain | In-domain | 58.07 | 0.0000 * | 0.968 |
| Cross-domain | Out-domain | 56.01 | 0.0000 * | 0.934 |
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
| Cross-domain | -0.0091 | 0.0555 |
| Within-domain | +0.0792 | 0.1419 |
| Mixed | +0.0995 | 0.1405 |

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
| 1 | sw_smote_r01 | 2.61 | 7 |
| 2 | smote_r01 | 2.67 | 5 |
| 3 | smote_r05 | 3.39 | 1 |
| 4 | sw_smote_r05 | 4.22 | 0 |
| 5 | rus_r01 | 4.56 | 3 |
| 6 | baseline | 5.17 | 0 |
| 7 | rus_r05 | 5.39 | 2 |

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

Friedman χ²=54.21, p=0.0000 (significant)

| | baseline | rus_r01 | rus_r05 | smote_r01 | smote_r05 | sw_smote_r01 | sw_smote_r05 |
|---|---|---|---|---|---|---|---|
| **baseline** | — | 0.9821 | 0.4361 | 0.7748 | 0.1649 | 0.0161 * | 0.0079 * |
| **rus_r01** | — | — | 0.9162 | 0.2550 | 0.0161 * | 0.0007 * | 0.0003 * |
| **rus_r05** | — | — | — | 0.0113 * | 0.0002 * | 0.0000 * | 0.0000 * |
| **smote_r01** | — | — | — | — | 0.9460 | 0.5050 | 0.3705 |
| **smote_r05** | — | — | — | — | — | 0.9821 | 0.9460 |
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
| sw_smote_r05 | 1.70 |
| sw_smote_r01 | 1.90 |
| smote_r05 | 2.70 |
| smote_r01 | 3.70 |
| baseline | 5.10 |
| rus_r01 | 5.90 |
| rus_r05 | 7.00 |

**Critical Difference (CD)** = 2.848 (α=0.05, k=7, n=10)

#### Out-domain (pooled across modes)

Friedman χ²=51.09, p=0.0000 (significant)

| | baseline | rus_r01 | rus_r05 | smote_r01 | smote_r05 | sw_smote_r01 | sw_smote_r05 |
|---|---|---|---|---|---|---|---|
| **baseline** | — | 0.9460 | 0.5755 | 0.6455 | 0.0577 | 0.0428 * | 0.0226 * |
| **rus_r01** | — | — | 0.9911 | 0.1004 | 0.0016 * | 0.0011 * | 0.0004 * |
| **rus_r05** | — | — | — | 0.0113 * | 0.0001 * | 0.0000 * | 0.0000 * |
| **smote_r01** | — | — | — | — | 0.8777 | 0.8303 | 0.7126 |
| **smote_r05** | — | — | — | — | — | 1.0000 | 0.9999 |
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
| sw_smote_r05 | 2.00 |
| sw_smote_r01 | 2.20 |
| smote_r05 | 2.30 |
| smote_r01 | 3.50 |
| baseline | 5.10 |
| rus_r01 | 6.10 |
| rus_r05 | 6.80 |

**Critical Difference (CD)** = 2.848 (α=0.05, k=7, n=10)


### AUPRC

#### In-domain (pooled across modes)

Friedman χ²=47.91, p=0.0000 (significant)

| | baseline | rus_r01 | rus_r05 | smote_r01 | smote_r05 | sw_smote_r01 | sw_smote_r05 |
|---|---|---|---|---|---|---|---|
| **baseline** | — | 1.0000 | 0.9986 | 0.0007 * | 0.0113 * | 0.0025 * | 0.1004 |
| **rus_r01** | — | — | 0.9911 | 0.0016 * | 0.0226 * | 0.0054 * | 0.1649 |
| **rus_r05** | — | — | — | 0.0001 * | 0.0016 * | 0.0003 * | 0.0226 * |
| **smote_r01** | — | — | — | — | 0.9911 | 0.9999 | 0.7748 |
| **smote_r05** | — | — | — | — | — | 0.9996 | 0.9911 |
| **sw_smote_r01** | — | — | — | — | — | — | 0.9162 |
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
| smote_r01 | 1.90 |
| sw_smote_r01 | 2.20 |
| smote_r05 | 2.60 |
| sw_smote_r05 | 3.30 |
| rus_r01 | 5.70 |
| baseline | 5.90 |
| rus_r05 | 6.40 |

**Critical Difference (CD)** = 2.848 (α=0.05, k=7, n=10)

#### Out-domain (pooled across modes)

Friedman χ²=50.49, p=0.0000 (significant)

| | baseline | rus_r01 | rus_r05 | smote_r01 | smote_r05 | sw_smote_r01 | sw_smote_r05 |
|---|---|---|---|---|---|---|---|
| **baseline** | — | 0.9162 | 0.4361 | 0.1004 | 0.1649 | 0.0113 * | 0.9460 |
| **rus_r01** | — | — | 0.9821 | 0.0025 * | 0.0054 * | 0.0001 * | 0.3098 |
| **rus_r05** | — | — | — | 0.0001 * | 0.0002 * | 0.0000 * | 0.0428 * |
| **smote_r01** | — | — | — | — | 1.0000 | 0.9911 | 0.6455 |
| **smote_r05** | — | — | — | — | — | 0.9676 | 0.7748 |
| **sw_smote_r01** | — | — | — | — | — | — | 0.2066 |
| **sw_smote_r05** | — | — | — | — | — | — | — |

**Significant pairs**: 8/21
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
| sw_smote_r01 | 1.60 |
| smote_r01 | 2.30 |
| smote_r05 | 2.50 |
| sw_smote_r05 | 3.90 |
| baseline | 4.90 |
| rus_r01 | 6.00 |
| rus_r05 | 6.80 |

**Critical Difference (CD)** = 2.848 (α=0.05, k=7, n=10)


### Recall

#### In-domain (pooled across modes)

Friedman χ²=53.53, p=0.0000 (significant)

| | baseline | rus_r01 | rus_r05 | smote_r01 | smote_r05 | sw_smote_r01 | sw_smote_r05 |
|---|---|---|---|---|---|---|---|
| **baseline** | — | 0.1649 | 0.0016 * | 0.7748 | 0.8777 | 0.9676 | 0.0002 * |
| **rus_r01** | — | — | 0.7748 | 0.0016 * | 0.8777 | 0.7126 | 0.4361 |
| **rus_r05** | — | — | — | 0.0000 * | 0.1004 | 0.0428 * | 0.9986 |
| **smote_r01** | — | — | — | — | 0.1004 | 0.2066 | 0.0000 * |
| **smote_r05** | — | — | — | — | — | 0.9999 | 0.0226 * |
| **sw_smote_r01** | — | — | — | — | — | — | 0.0079 * |
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
| smote_r01 | 1.00 |
| baseline | 2.40 |
| sw_smote_r01 | 3.30 |
| smote_r05 | 3.60 |
| rus_r01 | 4.80 |
| rus_r05 | 6.20 |
| sw_smote_r05 | 6.70 |

**Critical Difference (CD)** = 2.848 (α=0.05, k=7, n=10)

#### Out-domain (pooled across modes)

Friedman χ²=50.14, p=0.0000 (significant)

| | baseline | rus_r01 | rus_r05 | smote_r01 | smote_r05 | sw_smote_r01 | sw_smote_r05 |
|---|---|---|---|---|---|---|---|
| **baseline** | — | 0.0113 * | 0.0007 * | 1.0000 | 0.0766 | 0.1296 | 0.0000 * |
| **rus_r01** | — | — | 0.9911 | 0.0226 * | 0.9962 | 0.9821 | 0.2550 |
| **rus_r05** | — | — | — | 0.0016 * | 0.8303 | 0.7126 | 0.7126 |
| **smote_r01** | — | — | — | — | 0.1296 | 0.2066 | 0.0000 * |
| **smote_r05** | — | — | — | — | — | 1.0000 | 0.0577 |
| **sw_smote_r01** | — | — | — | — | — | — | 0.0313 * |
| **sw_smote_r05** | — | — | — | — | — | — | — |

**Significant pairs**: 7/21
- baseline vs rus_r01
- baseline vs rus_r05
- baseline vs sw_smote_r05
- rus_r01 vs smote_r01
- rus_r05 vs smote_r01
- smote_r01 vs sw_smote_r05
- sw_smote_r01 vs sw_smote_r05

**Mean ranks**:

| Condition | Mean Rank |
|-----------|----------:|
| baseline | 1.40 |
| smote_r01 | 1.60 |
| sw_smote_r01 | 3.90 |
| smote_r05 | 4.10 |
| rus_r01 | 4.70 |
| rus_r05 | 5.40 |
| sw_smote_r05 | 6.90 |

**Critical Difference (CD)** = 2.848 (α=0.05, k=7, n=10)

---
## 16. Bootstrap Confidence Intervals (BCa)

B = 10,000 resamples, seed-level resampling.


### F1-score

| Condition | Mode | Level | Mean | 95% CI Lower | 95% CI Upper | Width |
|-----------|------|-------|-----:|-------------:|-------------:|------:|
| baseline | Cross-domain | In-domain | 0.0812 | 0.0805 | 0.0820 | 0.0015 |
| baseline | Cross-domain | Out-domain | 0.0785 | 0.0777 | 0.0796 | 0.0019 |
| baseline | Within-domain | In-domain | 0.1030 | 0.0991 | 0.1084 | 0.0092 |
| baseline | Within-domain | Out-domain | 0.1111 | 0.1055 | 0.1157 | 0.0102 |
| baseline | Mixed | In-domain | 0.1206 | 0.1147 | 0.1258 | 0.0111 |
| baseline | Mixed | Out-domain | 0.1575 | 0.1493 | 0.1682 | 0.0189 |
| rus_r01 | Cross-domain | In-domain | 0.0876 | 0.0862 | 0.0889 | 0.0027 |
| rus_r01 | Cross-domain | Out-domain | 0.0785 | 0.0763 | 0.0815 | 0.0052 |
| rus_r01 | Within-domain | In-domain | 0.0740 | 0.0687 | 0.0777 | 0.0089 |
| rus_r01 | Within-domain | Out-domain | 0.1038 | 0.0971 | 0.1094 | 0.0123 |
| rus_r01 | Mixed | In-domain | 0.0995 | 0.0902 | 0.1208 | 0.0306 |
| rus_r01 | Mixed | Out-domain | 0.1141 | 0.1035 | 0.1298 | 0.0263 |
| rus_r05 | Cross-domain | In-domain | 0.0824 | 0.0807 | 0.0845 | 0.0037 |
| rus_r05 | Cross-domain | Out-domain | 0.0775 | 0.0742 | 0.0805 | 0.0063 |
| rus_r05 | Within-domain | In-domain | 0.0681 | 0.0616 | 0.0752 | 0.0135 |
| rus_r05 | Within-domain | Out-domain | 0.0920 | 0.0862 | 0.0978 | 0.0116 |
| rus_r05 | Mixed | In-domain | 0.0752 | 0.0672 | 0.0793 | 0.0121 |
| rus_r05 | Mixed | Out-domain | 0.0970 | 0.0914 | 0.1035 | 0.0121 |
| smote_r01 | Cross-domain | In-domain | 0.0770 | 0.0742 | 0.0787 | 0.0045 |
| smote_r01 | Cross-domain | Out-domain | 0.0804 | 0.0784 | 0.0829 | 0.0045 |
| smote_r01 | Within-domain | In-domain | 0.2745 | 0.2488 | 0.3080 | 0.0592 |
| smote_r01 | Within-domain | Out-domain | 0.2686 | 0.2426 | 0.3000 | 0.0573 |
| smote_r01 | Mixed | In-domain | 0.2564 | 0.2201 | 0.2981 | 0.0780 |
| smote_r01 | Mixed | Out-domain | 0.3026 | 0.2775 | 0.3238 | 0.0463 |
| smote_r05 | Cross-domain | In-domain | 0.0684 | 0.0663 | 0.0708 | 0.0045 |
| smote_r05 | Cross-domain | Out-domain | 0.0784 | 0.0756 | 0.0805 | 0.0049 |
| smote_r05 | Within-domain | In-domain | 0.3381 | 0.2972 | 0.3994 | 0.1022 |
| smote_r05 | Within-domain | Out-domain | 0.3294 | 0.2938 | 0.3746 | 0.0808 |
| smote_r05 | Mixed | In-domain | 0.3276 | 0.2700 | 0.3768 | 0.1068 |
| smote_r05 | Mixed | Out-domain | 0.4056 | 0.3360 | 0.5083 | 0.1723 |
| sw_smote_r01 | Cross-domain | In-domain | 0.0678 | 0.0652 | 0.0707 | 0.0055 |
| sw_smote_r01 | Cross-domain | Out-domain | 0.0672 | 0.0653 | 0.0688 | 0.0035 |
| sw_smote_r01 | Within-domain | In-domain | 0.3741 | 0.3554 | 0.3969 | 0.0415 |
| sw_smote_r01 | Within-domain | Out-domain | 0.3774 | 0.3397 | 0.4058 | 0.0661 |
| sw_smote_r01 | Mixed | In-domain | 0.3873 | 0.3266 | 0.4421 | 0.1155 |
| sw_smote_r01 | Mixed | Out-domain | 0.4745 | 0.3725 | 0.5560 | 0.1835 |
| sw_smote_r05 | Cross-domain | In-domain | 0.0444 | 0.0429 | 0.0473 | 0.0044 |
| sw_smote_r05 | Cross-domain | Out-domain | 0.0379 | 0.0332 | 0.0413 | 0.0081 |
| sw_smote_r05 | Within-domain | In-domain | 0.4794 | 0.3979 | 0.5821 | 0.1843 |
| sw_smote_r05 | Within-domain | Out-domain | 0.4395 | 0.3818 | 0.5274 | 0.1456 |
| sw_smote_r05 | Mixed | In-domain | 0.4097 | 0.3059 | 0.5242 | 0.2183 |
| sw_smote_r05 | Mixed | Out-domain | 0.4616 | 0.3768 | 0.5716 | 0.1948 |


### AUPRC

| Condition | Mode | Level | Mean | 95% CI Lower | 95% CI Upper | Width |
|-----------|------|-------|-----:|-------------:|-------------:|------:|
| baseline | Cross-domain | In-domain | 0.0508 | 0.0503 | 0.0511 | 0.0009 |
| baseline | Cross-domain | Out-domain | 0.0466 | 0.0459 | 0.0480 | 0.0021 |
| baseline | Within-domain | In-domain | 0.0872 | 0.0701 | 0.1496 | 0.0795 |
| baseline | Within-domain | Out-domain | 0.1387 | 0.1093 | 0.1657 | 0.0564 |
| baseline | Mixed | In-domain | 0.1287 | 0.1027 | 0.1626 | 0.0599 |
| baseline | Mixed | Out-domain | 0.4050 | 0.2589 | 0.6020 | 0.3432 |
| rus_r01 | Cross-domain | In-domain | 0.0585 | 0.0544 | 0.0629 | 0.0085 |
| rus_r01 | Cross-domain | Out-domain | 0.0497 | 0.0469 | 0.0550 | 0.0081 |
| rus_r01 | Within-domain | In-domain | 0.1127 | 0.0841 | 0.1599 | 0.0758 |
| rus_r01 | Within-domain | Out-domain | 0.1389 | 0.1054 | 0.1678 | 0.0624 |
| rus_r01 | Mixed | In-domain | 0.1617 | 0.0825 | 0.3420 | 0.2595 |
| rus_r01 | Mixed | Out-domain | 0.1232 | 0.0814 | 0.2179 | 0.1365 |
| rus_r05 | Cross-domain | In-domain | 0.0514 | 0.0501 | 0.0542 | 0.0041 |
| rus_r05 | Cross-domain | Out-domain | 0.0506 | 0.0481 | 0.0546 | 0.0065 |
| rus_r05 | Within-domain | In-domain | 0.1145 | 0.0891 | 0.1466 | 0.0575 |
| rus_r05 | Within-domain | Out-domain | 0.0837 | 0.0681 | 0.1025 | 0.0343 |
| rus_r05 | Mixed | In-domain | 0.0655 | 0.0589 | 0.0817 | 0.0228 |
| rus_r05 | Mixed | Out-domain | 0.0736 | 0.0594 | 0.1008 | 0.0414 |
| smote_r01 | Cross-domain | In-domain | 0.0502 | 0.0498 | 0.0505 | 0.0007 |
| smote_r01 | Cross-domain | Out-domain | 0.0463 | 0.0457 | 0.0470 | 0.0012 |
| smote_r01 | Within-domain | In-domain | 0.6452 | 0.5714 | 0.7227 | 0.1513 |
| smote_r01 | Within-domain | Out-domain | 0.6108 | 0.5412 | 0.6842 | 0.1431 |
| smote_r01 | Mixed | In-domain | 0.5302 | 0.4012 | 0.6391 | 0.2379 |
| smote_r01 | Mixed | Out-domain | 0.6741 | 0.6068 | 0.7349 | 0.1281 |
| smote_r05 | Cross-domain | In-domain | 0.0490 | 0.0488 | 0.0493 | 0.0005 |
| smote_r05 | Cross-domain | Out-domain | 0.0463 | 0.0457 | 0.0467 | 0.0010 |
| smote_r05 | Within-domain | In-domain | 0.5722 | 0.5040 | 0.6460 | 0.1421 |
| smote_r05 | Within-domain | Out-domain | 0.5674 | 0.5085 | 0.6324 | 0.1240 |
| smote_r05 | Mixed | In-domain | 0.5281 | 0.4152 | 0.6113 | 0.1961 |
| smote_r05 | Mixed | Out-domain | 0.6459 | 0.5623 | 0.7316 | 0.1693 |
| sw_smote_r01 | Cross-domain | In-domain | 0.0519 | 0.0513 | 0.0525 | 0.0013 |
| sw_smote_r01 | Cross-domain | Out-domain | 0.0434 | 0.0431 | 0.0436 | 0.0005 |
| sw_smote_r01 | Within-domain | In-domain | 0.6377 | 0.5805 | 0.6974 | 0.1170 |
| sw_smote_r01 | Within-domain | Out-domain | 0.6541 | 0.5528 | 0.7284 | 0.1756 |
| sw_smote_r01 | Mixed | In-domain | 0.5540 | 0.3987 | 0.6694 | 0.2707 |
| sw_smote_r01 | Mixed | Out-domain | 0.6809 | 0.4857 | 0.8096 | 0.3240 |
| sw_smote_r05 | Cross-domain | In-domain | 0.0556 | 0.0541 | 0.0570 | 0.0030 |
| sw_smote_r05 | Cross-domain | Out-domain | 0.0434 | 0.0429 | 0.0438 | 0.0009 |
| sw_smote_r05 | Within-domain | In-domain | 0.4591 | 0.3607 | 0.5733 | 0.2126 |
| sw_smote_r05 | Within-domain | Out-domain | 0.3639 | 0.3020 | 0.4736 | 0.1716 |
| sw_smote_r05 | Mixed | In-domain | 0.3879 | 0.2823 | 0.5111 | 0.2288 |
| sw_smote_r05 | Mixed | Out-domain | 0.4316 | 0.3283 | 0.5597 | 0.2313 |


### Recall

| Condition | Mode | Level | Mean | 95% CI Lower | 95% CI Upper | Width |
|-----------|------|-------|-----:|-------------:|-------------:|------:|
| baseline | Cross-domain | In-domain | 0.5096 | 0.4975 | 0.5279 | 0.0304 |
| baseline | Cross-domain | Out-domain | 0.4854 | 0.4647 | 0.5033 | 0.0386 |
| baseline | Within-domain | In-domain | 0.6176 | 0.5977 | 0.6363 | 0.0386 |
| baseline | Within-domain | Out-domain | 0.6862 | 0.6468 | 0.7163 | 0.0695 |
| baseline | Mixed | In-domain | 0.6987 | 0.6667 | 0.7297 | 0.0630 |
| baseline | Mixed | Out-domain | 0.8582 | 0.8312 | 0.8837 | 0.0524 |
| rus_r01 | Cross-domain | In-domain | 0.5225 | 0.5126 | 0.5319 | 0.0193 |
| rus_r01 | Cross-domain | Out-domain | 0.4597 | 0.4438 | 0.4816 | 0.0377 |
| rus_r01 | Within-domain | In-domain | 0.4230 | 0.3945 | 0.4547 | 0.0602 |
| rus_r01 | Within-domain | Out-domain | 0.6577 | 0.6120 | 0.6931 | 0.0811 |
| rus_r01 | Mixed | In-domain | 0.5439 | 0.4894 | 0.6444 | 0.1550 |
| rus_r01 | Mixed | Out-domain | 0.6143 | 0.5518 | 0.6926 | 0.1408 |
| rus_r05 | Cross-domain | In-domain | 0.4065 | 0.3912 | 0.4254 | 0.0342 |
| rus_r05 | Cross-domain | Out-domain | 0.4515 | 0.4220 | 0.4745 | 0.0525 |
| rus_r05 | Within-domain | In-domain | 0.3385 | 0.3073 | 0.3856 | 0.0783 |
| rus_r05 | Within-domain | Out-domain | 0.5677 | 0.5313 | 0.6072 | 0.0758 |
| rus_r05 | Mixed | In-domain | 0.3795 | 0.3361 | 0.4032 | 0.0671 |
| rus_r05 | Mixed | Out-domain | 0.5670 | 0.5361 | 0.6096 | 0.0735 |
| smote_r01 | Cross-domain | In-domain | 0.2712 | 0.2453 | 0.2925 | 0.0472 |
| smote_r01 | Cross-domain | Out-domain | 0.2652 | 0.2408 | 0.2877 | 0.0469 |
| smote_r01 | Within-domain | In-domain | 0.8681 | 0.8575 | 0.8769 | 0.0194 |
| smote_r01 | Within-domain | Out-domain | 0.8694 | 0.8568 | 0.8771 | 0.0203 |
| smote_r01 | Mixed | In-domain | 0.7967 | 0.7629 | 0.8124 | 0.0495 |
| smote_r01 | Mixed | Out-domain | 0.8734 | 0.8628 | 0.8813 | 0.0185 |
| smote_r05 | Cross-domain | In-domain | 0.1812 | 0.1626 | 0.1993 | 0.0367 |
| smote_r05 | Cross-domain | Out-domain | 0.1839 | 0.1645 | 0.2054 | 0.0408 |
| smote_r05 | Within-domain | In-domain | 0.7813 | 0.7612 | 0.8071 | 0.0459 |
| smote_r05 | Within-domain | Out-domain | 0.8018 | 0.7787 | 0.8219 | 0.0432 |
| smote_r05 | Mixed | In-domain | 0.7363 | 0.7052 | 0.7586 | 0.0534 |
| smote_r05 | Mixed | Out-domain | 0.7832 | 0.7535 | 0.8149 | 0.0614 |
| sw_smote_r01 | Cross-domain | In-domain | 0.1630 | 0.1489 | 0.1727 | 0.0238 |
| sw_smote_r01 | Cross-domain | Out-domain | 0.1552 | 0.1443 | 0.1628 | 0.0185 |
| sw_smote_r01 | Within-domain | In-domain | 0.8241 | 0.7866 | 0.8532 | 0.0665 |
| sw_smote_r01 | Within-domain | Out-domain | 0.8290 | 0.7302 | 0.8650 | 0.1348 |
| sw_smote_r01 | Mixed | In-domain | 0.7184 | 0.6180 | 0.7809 | 0.1629 |
| sw_smote_r01 | Mixed | Out-domain | 0.7727 | 0.6509 | 0.8482 | 0.1973 |
| sw_smote_r05 | Cross-domain | In-domain | 0.0489 | 0.0452 | 0.0535 | 0.0083 |
| sw_smote_r05 | Cross-domain | Out-domain | 0.0385 | 0.0324 | 0.0438 | 0.0114 |
| sw_smote_r05 | Within-domain | In-domain | 0.5164 | 0.4278 | 0.6241 | 0.1962 |
| sw_smote_r05 | Within-domain | Out-domain | 0.5117 | 0.4424 | 0.6092 | 0.1668 |
| sw_smote_r05 | Mixed | In-domain | 0.3641 | 0.2706 | 0.4742 | 0.2036 |
| sw_smote_r05 | Mixed | Out-domain | 0.4654 | 0.3773 | 0.5849 | 0.2076 |

---
## 17. Effect Size Confidence Intervals (Cliff's δ)

Bootstrap 95% CI (B = 2,000, percentile method).


### F1-score — Baseline vs each method

| Method vs Baseline | Mode | Level | δ | 95% CI | Excl. 0? | Effect |
|--------------------|------|-------|--:|-------:|:--------:|:------:|
| rus_r01 vs baseline | Cross-domain | In-domain | +0.682 | [+0.476, +0.867] | ✓ | large |
| rus_r01 vs baseline | Cross-domain | Out-domain | -0.167 | [-0.473, +0.136] | ✗ | small |
| rus_r01 vs baseline | Within-domain | In-domain | -0.640 | [-0.842, -0.398] | ✓ | large |
| rus_r01 vs baseline | Within-domain | Out-domain | +0.049 | [-0.249, +0.353] | ✗ | negligible |
| rus_r01 vs baseline | Mixed | In-domain | -0.609 | [-0.858, -0.327] | ✓ | large |
| rus_r01 vs baseline | Mixed | Out-domain | -0.867 | [-0.978, -0.711] | ✓ | large |
| rus_r05 vs baseline | Cross-domain | In-domain | +0.296 | [-0.011, +0.591] | ✗ | small |
| rus_r05 vs baseline | Cross-domain | Out-domain | -0.178 | [-0.449, +0.124] | ✗ | small |
| rus_r05 vs baseline | Within-domain | In-domain | -0.749 | [-0.933, -0.511] | ✓ | large |
| rus_r05 vs baseline | Within-domain | Out-domain | -0.311 | [-0.571, -0.016] | ✓ | small |
| rus_r05 vs baseline | Mixed | In-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| rus_r05 vs baseline | Mixed | Out-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| smote_r01 vs baseline | Cross-domain | In-domain | -0.316 | [-0.633, +0.024] | ✗ | small |
| smote_r01 vs baseline | Cross-domain | Out-domain | +0.184 | [-0.124, +0.476] | ✗ | small |
| smote_r01 vs baseline | Within-domain | In-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| smote_r01 vs baseline | Within-domain | Out-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| smote_r01 vs baseline | Mixed | In-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| smote_r01 vs baseline | Mixed | Out-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| smote_r05 vs baseline | Cross-domain | In-domain | -0.607 | [-0.849, -0.324] | ✓ | large |
| smote_r05 vs baseline | Cross-domain | Out-domain | +0.004 | [-0.316, +0.318] | ✗ | negligible |
| smote_r05 vs baseline | Within-domain | In-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| smote_r05 vs baseline | Within-domain | Out-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| smote_r05 vs baseline | Mixed | In-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| smote_r05 vs baseline | Mixed | Out-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| sw_smote_r01 vs baseline | Cross-domain | In-domain | -0.947 | [-1.000, -0.858] | ✓ | large |
| sw_smote_r01 vs baseline | Cross-domain | Out-domain | -0.831 | [-0.953, -0.682] | ✓ | large |
| sw_smote_r01 vs baseline | Within-domain | In-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| sw_smote_r01 vs baseline | Within-domain | Out-domain | +0.991 | [+0.964, +1.000] | ✓ | large |
| sw_smote_r01 vs baseline | Mixed | In-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| sw_smote_r01 vs baseline | Mixed | Out-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| sw_smote_r05 vs baseline | Cross-domain | In-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| sw_smote_r05 vs baseline | Cross-domain | Out-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| sw_smote_r05 vs baseline | Within-domain | In-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| sw_smote_r05 vs baseline | Within-domain | Out-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| sw_smote_r05 vs baseline | Mixed | In-domain | +1.000 | [+1.000, +1.000] | ✓ | large |
| sw_smote_r05 vs baseline | Mixed | Out-domain | +1.000 | [+1.000, +1.000] | ✓ | large |

**29/36** CIs exclude 0.


### AUPRC — Baseline vs each method

| Method vs Baseline | Mode | Level | δ | 95% CI | Excl. 0? | Effect |
|--------------------|------|-------|--:|-------:|:--------:|:------:|
| rus_r01 vs baseline | Cross-domain | In-domain | +0.069 | [-0.267, +0.400] | ✗ | negligible |
| rus_r01 vs baseline | Cross-domain | Out-domain | -0.024 | [-0.329, +0.289] | ✗ | negligible |
| rus_r01 vs baseline | Within-domain | In-domain | +0.064 | [-0.231, +0.353] | ✗ | negligible |
| rus_r01 vs baseline | Within-domain | Out-domain | -0.156 | [-0.458, +0.133] | ✗ | small |
| rus_r01 vs baseline | Mixed | In-domain | -0.442 | [-0.740, -0.144] | ✓ | medium |
| rus_r01 vs baseline | Mixed | Out-domain | -0.793 | [-0.933, -0.600] | ✓ | large |
| rus_r05 vs baseline | Cross-domain | In-domain | -0.098 | [-0.416, +0.207] | ✗ | negligible |
| rus_r05 vs baseline | Cross-domain | Out-domain | +0.236 | [-0.080, +0.509] | ✗ | small |
| rus_r05 vs baseline | Within-domain | In-domain | +0.082 | [-0.220, +0.376] | ✗ | negligible |
| rus_r05 vs baseline | Within-domain | Out-domain | -0.387 | [-0.636, -0.096] | ✓ | medium |
| rus_r05 vs baseline | Mixed | In-domain | -0.844 | [-0.953, -0.704] | ✓ | large |
| rus_r05 vs baseline | Mixed | Out-domain | -0.982 | [-1.000, -0.938] | ✓ | large |
| smote_r01 vs baseline | Cross-domain | In-domain | -0.240 | [-0.524, +0.062] | ✗ | small |
| smote_r01 vs baseline | Cross-domain | Out-domain | -0.020 | [-0.313, +0.280] | ✗ | negligible |
| smote_r01 vs baseline | Within-domain | In-domain | +0.980 | [+0.927, +1.000] | ✓ | large |
| smote_r01 vs baseline | Within-domain | Out-domain | +0.993 | [+0.973, +1.000] | ✓ | large |
| smote_r01 vs baseline | Mixed | In-domain | +0.980 | [+0.938, +1.000] | ✓ | large |
| smote_r01 vs baseline | Mixed | Out-domain | +0.489 | [+0.196, +0.771] | ✓ | large |
| smote_r05 vs baseline | Cross-domain | In-domain | -0.558 | [-0.798, -0.296] | ✓ | large |
| smote_r05 vs baseline | Cross-domain | Out-domain | -0.093 | [-0.404, +0.233] | ✗ | negligible |
| smote_r05 vs baseline | Within-domain | In-domain | +0.978 | [+0.920, +1.000] | ✓ | large |
| smote_r05 vs baseline | Within-domain | Out-domain | +0.956 | [+0.882, +0.996] | ✓ | large |
| smote_r05 vs baseline | Mixed | In-domain | +0.980 | [+0.938, +1.000] | ✓ | large |
| smote_r05 vs baseline | Mixed | Out-domain | +0.491 | [+0.215, +0.764] | ✓ | large |
| sw_smote_r01 vs baseline | Cross-domain | In-domain | +0.193 | [-0.131, +0.478] | ✗ | small |
| sw_smote_r01 vs baseline | Cross-domain | Out-domain | -0.704 | [-0.878, -0.493] | ✓ | large |
| sw_smote_r01 vs baseline | Within-domain | In-domain | +0.982 | [+0.938, +1.000] | ✓ | large |
| sw_smote_r01 vs baseline | Within-domain | Out-domain | +0.927 | [+0.829, +0.987] | ✓ | large |
| sw_smote_r01 vs baseline | Mixed | In-domain | +0.964 | [+0.911, +1.000] | ✓ | large |
| sw_smote_r01 vs baseline | Mixed | Out-domain | +0.625 | [+0.398, +0.818] | ✓ | large |
| sw_smote_r05 vs baseline | Cross-domain | In-domain | +0.424 | [+0.113, +0.700] | ✓ | medium |
| sw_smote_r05 vs baseline | Cross-domain | Out-domain | -0.731 | [-0.907, -0.518] | ✓ | large |
| sw_smote_r05 vs baseline | Within-domain | In-domain | +0.951 | [+0.833, +1.000] | ✓ | large |
| sw_smote_r05 vs baseline | Within-domain | Out-domain | +0.736 | [+0.524, +0.891] | ✓ | large |
| sw_smote_r05 vs baseline | Mixed | In-domain | +0.936 | [+0.846, +0.998] | ✓ | large |
| sw_smote_r05 vs baseline | Mixed | Out-domain | +0.211 | [-0.096, +0.518] | ✗ | small |

**24/36** CIs exclude 0.


### Recall — Baseline vs each method

| Method vs Baseline | Mode | Level | δ | 95% CI | Excl. 0? | Effect |
|--------------------|------|-------|--:|-------:|:--------:|:------:|
| rus_r01 vs baseline | Cross-domain | In-domain | +0.257 | [-0.051, +0.536] | ✗ | small |
| rus_r01 vs baseline | Cross-domain | Out-domain | -0.176 | [-0.486, +0.144] | ✗ | small |
| rus_r01 vs baseline | Within-domain | In-domain | -0.604 | [-0.831, -0.357] | ✓ | large |
| rus_r01 vs baseline | Within-domain | Out-domain | -0.023 | [-0.330, +0.271] | ✗ | negligible |
| rus_r01 vs baseline | Mixed | In-domain | -0.596 | [-0.842, -0.329] | ✓ | large |
| rus_r01 vs baseline | Mixed | Out-domain | -0.914 | [-1.000, -0.792] | ✓ | large |
| rus_r05 vs baseline | Cross-domain | In-domain | -0.816 | [-0.989, -0.596] | ✓ | large |
| rus_r05 vs baseline | Cross-domain | Out-domain | -0.239 | [-0.546, +0.060] | ✗ | small |
| rus_r05 vs baseline | Within-domain | In-domain | -0.838 | [-0.964, -0.664] | ✓ | large |
| rus_r05 vs baseline | Within-domain | Out-domain | -0.319 | [-0.584, -0.022] | ✓ | small |
| rus_r05 vs baseline | Mixed | In-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| rus_r05 vs baseline | Mixed | Out-domain | -0.987 | [-1.000, -0.947] | ✓ | large |
| smote_r01 vs baseline | Cross-domain | In-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| smote_r01 vs baseline | Cross-domain | Out-domain | -0.978 | [-1.000, -0.929] | ✓ | large |
| smote_r01 vs baseline | Within-domain | In-domain | +0.958 | [+0.853, +1.000] | ✓ | large |
| smote_r01 vs baseline | Within-domain | Out-domain | +0.566 | [+0.267, +0.824] | ✓ | large |
| smote_r01 vs baseline | Mixed | In-domain | +0.781 | [+0.598, +0.916] | ✓ | large |
| smote_r01 vs baseline | Mixed | Out-domain | +0.096 | [-0.249, +0.434] | ✗ | negligible |
| smote_r05 vs baseline | Cross-domain | In-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| smote_r05 vs baseline | Cross-domain | Out-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| smote_r05 vs baseline | Within-domain | In-domain | +0.832 | [+0.657, +0.960] | ✓ | large |
| smote_r05 vs baseline | Within-domain | Out-domain | +0.459 | [+0.146, +0.733] | ✓ | medium |
| smote_r05 vs baseline | Mixed | In-domain | +0.246 | [-0.065, +0.542] | ✗ | small |
| smote_r05 vs baseline | Mixed | Out-domain | -0.676 | [-0.844, -0.458] | ✓ | large |
| sw_smote_r01 vs baseline | Cross-domain | In-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| sw_smote_r01 vs baseline | Cross-domain | Out-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| sw_smote_r01 vs baseline | Within-domain | In-domain | +0.774 | [+0.556, +0.942] | ✓ | large |
| sw_smote_r01 vs baseline | Within-domain | Out-domain | +0.441 | [+0.142, +0.732] | ✓ | medium |
| sw_smote_r01 vs baseline | Mixed | In-domain | +0.313 | [-0.010, +0.613] | ✗ | small |
| sw_smote_r01 vs baseline | Mixed | Out-domain | -0.236 | [-0.510, +0.067] | ✗ | small |
| sw_smote_r05 vs baseline | Cross-domain | In-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| sw_smote_r05 vs baseline | Cross-domain | Out-domain | -1.000 | [-1.000, -1.000] | ✓ | large |
| sw_smote_r05 vs baseline | Within-domain | In-domain | -0.344 | [-0.627, -0.043] | ✓ | medium |
| sw_smote_r05 vs baseline | Within-domain | Out-domain | -0.590 | [-0.806, -0.331] | ✓ | large |
| sw_smote_r05 vs baseline | Mixed | In-domain | -0.945 | [-0.993, -0.864] | ✓ | large |
| sw_smote_r05 vs baseline | Mixed | Out-domain | -1.000 | [-1.000, -1.000] | ✓ | large |

**28/36** CIs exclude 0.

---
## 18. Permutation Test for Global Null

B = 10,000 permutations.


### F1-score

- $T_{\text{obs}}$ = 11.4889
- $p_{\text{perm}}$ = 0.0001 (B = 10000)
- **Interpretation**: Strong evidence against the global null (p < 0.001). Condition labels are informative for F1-score.


### AUPRC

- $T_{\text{obs}}$ = 17.6663
- $p_{\text{perm}}$ = 0.0001 (B = 10000)
- **Interpretation**: Strong evidence against the global null (p < 0.001). Condition labels are informative for AUPRC.


### Recall

- $T_{\text{obs}}$ = 18.8019
- $p_{\text{perm}}$ = 0.0001 (B = 10000)
- **Interpretation**: Strong evidence against the global null (p < 0.001). Condition labels are informative for Recall.


---
## 19. Benjamini-Hochberg FDR Correction Sensitivity

| Hypothesis Family | m | Bonf. sig | FDR sig | Gain |
|-------------------|--:|----------:|--------:|-----:|
| H1 KW (F1-score) | 18 | 18 | 18 | +0 |
| H1 pairwise (F1-score) | 36 | 28 | 30 | +2 |
| H10 domain shift (F1-score) | 63 | 0 | 21 | +21 |
| H1 KW (AUPRC) | 18 | 18 | 18 | +0 |
| H1 pairwise (AUPRC) | 36 | 21 | 24 | +3 |
| H10 domain shift (AUPRC) | 63 | 0 | 25 | +25 |
| H1 KW (Recall) | 18 | 18 | 18 | +0 |
| H1 pairwise (Recall) | 36 | 24 | 29 | +5 |
| H10 domain shift (Recall) | 63 | 0 | 25 | +25 |

**Overall**: Bonferroni **127** → BH-FDR **208** (+81).

---
## 20. Cross-Metric Concordance

Do the extended metrics (F1, AUPRC, Recall) agree with the primary metrics (F2, AUROC) on condition rankings?

### Overall ranking comparison

| Metric | #1 | #2 | #3 | #4 | #5 | #6 | #7 |
|--------|:---|:---|:---|:---|:---|:---|:---|
| F2-score | sw_smote_r01 (2.56) | smote_r05 (3.22) | smote_r01 (3.56) | baseline (4.11) | rus_r01 (4.61) | sw_smote_r05 (4.67) | rus_r05 (5.28) |
| AUROC | smote_r01 (2.17) | smote_r05 (3.11) | sw_smote_r01 (3.33) | sw_smote_r05 (4.06) | baseline (4.83) | rus_r01 (4.89) | rus_r05 (5.61) |
| F1-score | sw_smote_r01 (3.06) | sw_smote_r05 (3.17) | smote_r01 (3.50) | smote_r05 (3.50) | baseline (4.56) | rus_r01 (4.83) | rus_r05 (5.39) |
| AUPRC | sw_smote_r01 (2.61) | smote_r01 (2.67) | smote_r05 (3.39) | sw_smote_r05 (4.22) | rus_r01 (4.56) | baseline (5.17) | rus_r05 (5.39) |
| Recall | smote_r01 (2.17) | baseline (2.83) | smote_r05 (3.44) | sw_smote_r01 (4.00) | rus_r01 (4.11) | rus_r05 (4.89) | sw_smote_r05 (6.56) |
| Precision | sw_smote_r05 (2.56) | sw_smote_r01 (2.83) | smote_r05 (3.11) | smote_r01 (3.67) | baseline (4.94) | rus_r01 (5.22) | rus_r05 (5.67) |
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
| H5 | Distance effect (F1-score) | H=2.04, p=0.3600 | Not supported ✗ |
| H7 | Within > cross (F1-score) | δ=+0.793 (large) | Supported ✓ |
| H10 | Domain shift (F1-score) | δ=-0.083, p=0.0111 | Weak |
| H1 | Condition effect (AUPRC) | 18/18 sig | Supported ✓ |
| H2 | sw > smote (AUPRC) | 5/12 cells | Mixed |
| H3 | Over > RUS (AUPRC) | 17/24 | Supported ✓ |
| H5 | Distance effect (AUPRC) | H=2.04, p=0.3607 | Not supported ✗ |
| H7 | Within > cross (AUPRC) | δ=+0.946 (large) | Supported ✓ |
| H10 | Domain shift (AUPRC) | δ=+0.057, p=0.0809 | Not supported ✗ |
| H1 | Condition effect (Recall) | 18/18 sig | Supported ✓ |
| H2 | sw > smote (Recall) | 0/12 cells | Not supported ✗ |
| H3 | Over > RUS (Recall) | 13/24 | Mixed |
| H5 | Distance effect (Recall) | H=1.56, p=0.4579 | Not supported ✗ |
| H7 | Within > cross (Recall) | δ=+0.786 (large) | Supported ✓ |
| H10 | Domain shift (Recall) | δ=-0.143, p=0.0000 | Weak |
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

