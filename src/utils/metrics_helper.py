"""Shared metrics calculation utilities for model training and evaluation.

This module consolidates repeated metric calculations (accuracy, precision, recall, F1, 
confusion matrix, ROC/PR curves) used across training, evaluation, and analysis pipelines.

Unified naming: all functions use `y_true`, `y_pred`, `y_score` consistently.
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_division: int = 0,
) -> Dict[str, float]:
    """Calculate standard classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    zero_division : int, default=0
        Value to return when there is a zero division in precision/recall/F1.

    Returns
    -------
    dict
        Dictionary containing:
        - ``"accuracy"`` : float
        - ``"precision"`` : float
        - ``"recall"`` : float
        - ``"f1"`` : float
        - ``"confusion_matrix"`` : list of list
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=zero_division)),
        "recall": float(recall_score(y_true, y_pred, zero_division=zero_division)),
        "f1": float(f1_score(y_true, y_pred, zero_division=zero_division)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def calculate_roc_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    invert_check: bool = True,
) -> Dict[str, Any]:
    """Calculate ROC curve and AUC.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted probabilities or decision scores for the positive class.
    invert_check : bool, default=True
        If True and AUC < 0.5, invert y_score and recalculate (handles reversed probability).

    Returns
    -------
    dict
        Dictionary containing:
        - ``"fpr"`` : list of float
        - ``"tpr"`` : list of float
        - ``"auc"`` : float
        - ``"inverted"`` : bool (True if scores were inverted)
    """
    y_score = np.asarray(y_score)
    inverted = False

    if invert_check:
        try:
            auc_check = roc_auc_score(y_true, y_score)
            if auc_check < 0.5:
                logging.warning(
                    f"[ROC] AUC={auc_check:.3f} < 0.5 → Inverting predicted scores"
                )
                y_score = 1.0 - y_score
                inverted = True
        except Exception as e:
            logging.warning(f"[ROC] AUC check failed: {e}")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": float(roc_auc),
        "inverted": inverted,
    }


def calculate_pr_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, Any]:
    """Calculate Precision-Recall curve and average precision (AUPRC).

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted probabilities or decision scores for the positive class.

    Returns
    -------
    dict
        Dictionary containing:
        - ``"precision"`` : list of float
        - ``"recall"`` : list of float
        - ``"auc_pr"`` : float (average precision score)
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auc_pr = average_precision_score(y_true, y_score)

    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "auc_pr": float(auc_pr),
    }


def calculate_extended_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    zero_division: int = 0,
    include_roc: bool = True,
    include_pr: bool = True,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics including ROC and PR curves.

    Unified function for training and evaluation pipelines.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_score : array-like, optional
        Predicted probabilities or decision scores. Required for ROC/PR curves.
    zero_division : int, default=0
        Value to return for undefined precision/recall/F1.
    include_roc : bool, default=True
        Whether to compute ROC curve metrics (requires y_score).
    include_pr : bool, default=True
        Whether to compute PR curve metrics (requires y_score).

    Returns
    -------
    dict
        Merged dictionary containing classification metrics, ROC curve, and PR curve.
        Includes keys: accuracy, precision, recall, f1, confusion_matrix,
        roc_curve (if y_score provided), pr_curve (if y_score provided).
    """
    metrics = calculate_classification_metrics(y_true, y_pred, zero_division=zero_division)

    if y_score is not None:
        if include_roc:
            try:
                roc_metrics = calculate_roc_metrics(y_true, y_score, invert_check=True)
                metrics["roc_curve"] = roc_metrics
                metrics["roc_auc"] = roc_metrics["auc"]
            except Exception as e:
                logging.warning(f"[METRICS] ROC calculation failed: {e}")
                metrics["roc_auc"] = None

        if include_pr:
            try:
                pr_metrics = calculate_pr_metrics(y_true, y_score)
                metrics["pr_curve"] = pr_metrics
                metrics["auc_pr"] = pr_metrics["auc_pr"]
            except Exception as e:
                logging.warning(f"[METRICS] PR calculation failed: {e}")
                metrics["auc_pr"] = None

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    beta: float = 1.0,
) -> Tuple[float, float]:
    """Find optimal classification threshold by maximizing F-beta score.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_score : array-like
        Predicted probabilities or decision scores.
    beta : float, default=1.0
        Beta parameter for F-beta score (1.0 = F1, 2.0 = F2 emphasizing recall).

    Returns
    -------
    float
        Optimal threshold value.
    float
        Best F-beta score at that threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # precision_recall_curve returns len(thresholds) = len(precision) - 1
    prec_t = precision[:-1]
    rec_t = recall[:-1]

    denom = (beta**2) * prec_t + rec_t + 1e-8
    fbeta_scores = (1 + beta**2) * (prec_t * rec_t) / denom

    if fbeta_scores.size == 0:
        return 0.5, 0.0

    best_idx = int(np.argmax(fbeta_scores))
    best_threshold = thresholds[best_idx] if thresholds.size else 0.5
    best_fbeta = fbeta_scores[best_idx]

    return float(best_threshold), float(best_fbeta)


def apply_threshold(
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Apply classification threshold to continuous scores.

    Parameters
    ----------
    y_score : array-like
        Predicted probabilities or decision scores.
    threshold : float, default=0.5
        Decision threshold.

    Returns
    -------
    np.ndarray
        Binary predictions (0 or 1).
    """
    return (np.asarray(y_score) >= threshold).astype(int)


def calculate_class_specific_metrics(
    conf_matrix: np.ndarray,
    classification_report_dict: Optional[Dict] = None,
) -> Dict[str, Optional[float]]:
    """Extract per-class metrics and specificity from confusion matrix.

    Parameters
    ----------
    conf_matrix : array-like of shape (2, 2)
        Confusion matrix for binary classification.
    classification_report_dict : dict, optional
        Scikit-learn classification_report output_dict=True.

    Returns
    -------
    dict
        Dictionary containing:
        - ``"precision_pos"``, ``"recall_pos"``, ``"f1_pos"``, ``"support_pos"``
        - ``"precision_neg"``, ``"recall_neg"``, ``"specificity"``
    """
    result = {}

    # Extract from classification report if provided
    if classification_report_dict:
        pos = classification_report_dict.get("1", {})
        neg = classification_report_dict.get("0", {})

        result["precision_pos"] = float(pos.get("precision")) if "precision" in pos else None
        result["recall_pos"] = float(pos.get("recall")) if "recall" in pos else None
        result["f1_pos"] = float(pos.get("f1-score")) if "f1-score" in pos else None
        result["support_pos"] = int(pos.get("support")) if "support" in pos else None

        result["precision_neg"] = float(neg.get("precision")) if "precision" in neg else None
        result["recall_neg"] = float(neg.get("recall")) if "recall" in neg else None

    # Calculate specificity from confusion matrix
    try:
        tn, fp, fn, tp = conf_matrix.ravel()
        result["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    except Exception:
        result["specificity"] = None

    return result
