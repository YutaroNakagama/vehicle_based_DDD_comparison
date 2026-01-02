# Evaluation Metrics for Time-Series Anomaly Detection

This document explains the evaluation metrics used for time-series anomaly detection (such as driver drowsiness detection).

## Overview

In anomaly detection with imbalanced data, the choice of metrics is critical.
In this project, the positive rate (Drowsy) is approximately 4%, so appropriate metrics must be selected.

## Recommended Metrics Priority

| Priority | Metric | Usage |
|----------|--------|-------|
| ⭐⭐⭐ | **AUPRC** | Primary metric for model comparison (threshold-independent) |
| ⭐⭐⭐ | **Recall** | Anomaly detection rate (prevents missed detections) |
| ⭐⭐ | **F2 Score** | Recall-focused balance metric |
| ⭐ | Precision | False alarm rate evaluation |
| ⭐ | F1 Score | Balance metric (equal weight to Recall/Precision) |
| △ | AUROC | Tends to be overestimated with imbalanced data |
| △ | Accuracy | Inappropriate for imbalanced data |

---

## Metric Details

### 1. Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Characteristics:**
- Proportion of correct predictions out of all predictions
- **Inappropriate for imbalanced data** - High values can be achieved by simply predicting all samples as the majority class

**Example (this project):**
- With 4% positive rate, predicting "all negative" → Accuracy = 96%
- Appears high but is worthless for anomaly detection

**Recommendation:** Use only as a reference. Do not use as the primary metric.

---

### 2. Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Characteristics:**
- Proportion of actual positives among samples predicted as positive
- Evaluates the **reduction of false alarms (False Positives)**
- Important when avoiding "crying wolf" behavior

**Use Cases:**
- When false alarm costs are high (e.g., manufacturing line shutdown)
- When user experience is prioritized

**Value in this project:** Approximately 4% (low = many false alarms)

---

### 3. Recall (Sensitivity)

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Characteristics:**
- Proportion of correctly detected positives among actual positives
- Evaluates the **reduction of missed detections (False Negatives)**
- **Most important for safety-critical anomaly detection**

**Use Cases:**
- Drowsy driving detection - missed detections are fatal
- Medical diagnosis - missing a disease is dangerous
- Security - missing intrusions is critical

**Target in this project:** 50% or higher

---

### 4. F1 Score

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Characteristics:**
- Harmonic mean of Precision and Recall
- Equal weight to both
- Recall tends to be sacrificed with imbalanced data

**Recommendation:** Use when Recall and Precision are equally important

---

### 5. F2 Score ⭐Recommended

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

With $\beta = 2$, **Recall is weighted twice as much**.

**Characteristics:**
- Prioritizes reducing missed detections
- Widely used in safety-critical applications
- More suitable for anomaly detection than F1

**Usage in this project:** Used as the objective function for threshold optimization

---

### 6. AUROC (Area Under ROC Curve)

$$\text{AUROC} = \int_0^1 \text{TPR}(FPR) \, d(\text{FPR})$$

**Characteristics:**
- Area under the ROC curve
- Threshold-independent overall evaluation
- 0.5 = random, 1.0 = perfect

**Issues (imbalanced data):**
- **Overestimated when True Negative Rate (TNR) is high**
- FPR tends to be low when the majority class is dominant
- Even 0.9+ may not be practical with imbalanced data

**Recommendation:** Use as reference. Prioritize AUPRC.

---

### 7. AUPRC (Area Under Precision-Recall Curve) ⭐⭐⭐Most Important

$$\text{AUPRC} = \int_0^1 \text{Precision}(Recall) \, d(\text{Recall})$$

**Characteristics:**
- Area under the PR curve
- **Most reliable metric for imbalanced data**
- Random classifier AUPRC = positive rate (approximately 4% in this project)

**Why AUPRC is important:**
1. Not affected by True Negatives
2. Accurately reflects minority class detection performance
3. Best for model comparison before threshold selection

**Values in this project:**
- Random baseline: 0.039 (3.9%)
- Current best: 0.073 (SMOTE+RUS)

**Interpretation guidelines:**
| AUPRC | Evaluation |
|-------|------------|
| < 0.039 | Below random |
| 0.039-0.06 | Room for improvement |
| 0.06-0.10 | Somewhat effective |
| 0.10-0.20 | Effective |
| > 0.20 | Good |

---

## Metric Selection Guidelines

### Recommended Metrics by Use Case

| Use Case | Primary Metric | Secondary Metric | Reason |
|----------|----------------|------------------|--------|
| **Safety-critical** (drowsy detection, medical) | Recall, F2 | AUPRC | Missed detections are fatal |
| **Cost-focused** (manufacturing anomaly detection) | Precision, F1 | AUPRC | Cost of false alarm shutdowns |
| **Model comparison** (R&D) | AUPRC | Recall, F2 | Threshold-independent overall evaluation |
| **Operational monitoring** | Recall@fixed FPR | - | Reflects operational constraints |

### Evaluation Policy in This Project

1. **Model selection**: Compare using AUPRC
2. **Threshold optimization**: Maximize F2 score
3. **Final evaluation**: Report Recall (detection rate)

---

## Confusion Matrix

```
                 Predicted
              Negative  Positive
Actual Negative    TN        FP      ← False alarm
       Positive    FN        TP      ← Missed detection
```

- **TP (True Positive)**: Correctly predicted as positive
- **TN (True Negative)**: Correctly predicted as negative
- **FP (False Positive)**: False alarm (negative incorrectly predicted as positive)
- **FN (False Negative)**: Missed detection (positive incorrectly predicted as negative)

**Importance in anomaly detection:**
- If FN (missed detection) > FP (false alarm) → Focus on Recall
- If FP (false alarm) > FN (missed detection) → Focus on Precision

---

## Threshold Selection

Probabilistic classifiers' performance varies with threshold.

### Threshold Selection Methods

1. **Default (0.5)**: Common but inappropriate for imbalanced data
2. **F2 maximization**: Automatically selects a Recall-focused threshold
3. **Fixed Recall**: Selects threshold to achieve target Recall (e.g., 80%)
4. **Cost minimization**: Define costs for false alarms/missed detections and minimize

### Threshold Selection in This Project

```python
# Select threshold that maximizes F2 score on validation data
from sklearn.metrics import fbeta_score
best_threshold = max(thresholds, key=lambda t: fbeta_score(y_val, y_pred > t, beta=2))
```

---

## References

1. Davis, J., & Goadrich, M. (2006). "The relationship between Precision-Recall and ROC curves." *ICML*.
2. Saito, T., & Rehmsmeier, M. (2015). "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets." *PLOS ONE*.
3. He, H., & Garcia, E. A. (2009). "Learning from Imbalanced Data." *IEEE TKDE*.

---

## Related Documents

- [Imbalance Handling Methods](imbalance_methods.md)
- [Evaluation Pipeline](evaluation.rst)
