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


def precision_at_min_recall(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_recall: float = 0.70,
) -> float:
    """Calculate maximum precision at or above minimum recall.

    This metric is useful for Precision-focused optimization while
    maintaining a minimum safety threshold (Recall).

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    min_recall : float, default=0.70
        Minimum recall threshold. Returns max precision where recall >= min_recall.

    Returns
    -------
    float
        Maximum precision at recall >= min_recall. Returns 0.0 if no threshold
        satisfies the minimum recall constraint.
    """
    if len(y_true) == 0 or len(y_proba) == 0:
        return 0.0

    # Handle edge case: only one class present
    if len(np.unique(y_true)) < 2:
        return 0.0

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Find indices where recall >= min_recall
    valid_idx = recalls >= min_recall
    if not np.any(valid_idx):
        return 0.0  # No threshold satisfies min_recall

    # Return maximum precision in valid range
    return float(np.max(precisions[valid_idx]))


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


# === JSON Evaluation Metric Extraction ===

def get_positive_class_block(classification_report: Dict) -> Optional[Dict]:
    """Extract the positive-class metrics block from a classification report.
    
    Handles historical key variations such as "1", "1.0", "True", "pos", "positive".
    
    Parameters
    ----------
    classification_report : dict
        Classification report dictionary from sklearn or custom evaluator.
    
    Returns
    -------
    dict or None
        Positive class metrics block if found, else None.
    
    Examples
    --------
    >>> cr = {"0": {...}, "1": {"precision": 0.8, "recall": 0.7}}
    >>> block = get_positive_class_block(cr)
    >>> block["precision"]
    0.8
    """
    if not isinstance(classification_report, dict):
        return None
    
    # Try common positive class keys
    for key in ("1", "1.0", "True", "pos", "positive"):
        block = classification_report.get(key)
        if isinstance(block, dict):
            return block
    
    return None


def get_metric_from_positive_class(
    classification_report: Dict,
    field: str
) -> Optional[float]:
    """Extract a specific metric from the positive class block.
    
    Parameters
    ----------
    classification_report : dict
        Classification report dictionary.
    field : str
        Metric field name (e.g., 'precision', 'recall', 'f1-score').
    
    Returns
    -------
    float or None
        Metric value if found, else None.
    
    Examples
    --------
    >>> cr = {"1": {"precision": 0.8, "recall": 0.7, "f1-score": 0.74}}
    >>> get_metric_from_positive_class(cr, "f1-score")
    0.74
    """
    block = get_positive_class_block(classification_report)
    if block is None:
        return None
    
    value = block.get(field)
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def estimate_positive_rate(classification_report: Dict) -> Optional[float]:
    """Estimate positive class rate from classification report support counts.
    
    Parameters
    ----------
    classification_report : dict
        Classification report with class-wise support counts.
    
    Returns
    -------
    float or None
        Positive class rate (proportion of positive samples) if found, else None.
    
    Examples
    --------
    >>> cr = {"0": {"support": 970}, "1": {"support": 30}}
    >>> estimate_positive_rate(cr)
    0.03
    """
    if not isinstance(classification_report, dict):
        return None
    
    # Try common key pairs for negative/positive classes
    key_pairs = [
        ("0", "1"),
        ("0.0", "1.0"),
        ("False", "True"),
        ("neg", "pos"),
        ("negative", "positive"),
    ]
    
    for neg_key, pos_key in key_pairs:
        neg_block = classification_report.get(neg_key)
        pos_block = classification_report.get(pos_key)
        
        if (neg_block and pos_block and 
            "support" in neg_block and "support" in pos_block):
            try:
                n_neg = float(neg_block["support"])
                n_pos = float(pos_block["support"])
                total = n_neg + n_pos
                if total > 0:
                    return n_pos / total
            except (ValueError, TypeError):
                continue
    
    return None


def compute_f2_score_from_pr(precision: float, recall: float) -> Optional[float]:
    """Compute F2 score from precision and recall.
    
    F2 score weighs recall higher than precision: F2 = 5*P*R / (4*P + R)
    
    Parameters
    ----------
    precision : float
        Precision value.
    recall : float
        Recall value.
    
    Returns
    -------
    float or None
        F2 score if inputs are valid, else None.
    
    Examples
    --------
    >>> compute_f2_score_from_pr(0.8, 0.7)
    0.7291666666666666
    """
    if precision is None or recall is None:
        return None
    
    try:
        denominator = 4 * precision + recall
        if denominator == 0:
            return None
        return 5 * precision * recall / denominator
    except (ValueError, TypeError):
        return None


def extract_metrics_from_eval_json(
    data: Dict[str, Any],
    filename: str = ""
) -> Dict[str, Any]:
    """Extract standardized metrics from an evaluation JSON file.
    
    This function handles backward compatibility with various JSON formats
    and extracts metrics consistently regardless of format variations.
    
    Parameters
    ----------
    data : dict
        Parsed evaluation JSON data.
    filename : str, optional
        Source filename for logging purposes.
    
    Returns
    -------
    dict
        Dictionary with extracted metrics including:
        - model, mode, distance, level
        - auc, auc_pr, f1, f2, accuracy, precision, recall, specificity
        - pos_rate (positive class proportion)
        - threshold-optimized metrics (*_thr variants)
    
    Examples
    --------
    >>> data = {"auc": 0.85, "classification_report": {"1": {"f1-score": 0.7}}}
    >>> metrics = extract_metrics_from_eval_json(data)
    >>> metrics["auc"]
    0.85
    """
    cr = data.get("classification_report", {}) or {}
    
    # Extract positive rate
    pos_rate = estimate_positive_rate(cr)
    if pos_rate is None:
        pos_rate = data.get("pos_rate") or data.get("positive_rate")
    
    # Extract distance and level from filename if not in data
    # Example: eval_results_RF_pooled_rank_dtw_mean_out_domain_mean_test.json
    # New format: eval_results_RF_pooled_rank_mean_distance_dtw_out_domain_test.json
    distance = data.get("distance")
    level = data.get("level")
    ranking_method = data.get("ranking_method")
    
    # If level is a split name (test/val/train) instead of out_domain/mid_domain/in_domain, extract from filename
    if level in ("test", "val", "train", "validation"):
        level = None
    
    # Extract from filename if distance/level not properly set
    if ((not distance or distance == "unknown") or not level) and filename:
        import re
        
        # New pattern: rank_{method}_{metric}_{level}
        # Methods: mean_distance, centroid_umap, lof
        match_new = re.search(
            r'rank_(mean_distance|centroid_umap|lof|centroid_mds|medoid)_(dtw|mmd|wasserstein)_(out_domain|mid_domain|in_domain)',
            filename
        )
        if match_new:
            method = match_new.group(1)
            metric = match_new.group(2)
            level_extracted = match_new.group(3)
            if not ranking_method:
                ranking_method = method
            if not distance or distance == "unknown":
                distance = f"{metric}_{level_extracted}"
            if not level:
                level = level_extracted
        else:
            # Legacy pattern: rank_{metric}_mean_{level}
            match = re.search(r'rank_(dtw|mmd|wasserstein)_mean_(out_domain|mid_domain|in_domain)', filename)
            if match:
                metric = match.group(1)
                level_extracted = match.group(2)
                if not ranking_method:
                    ranking_method = "mean_distance"
                if not distance or distance == "unknown":
                    distance = f"{metric}_mean_{level_extracted}"
                if not level:
                    level = level_extracted
    
    # Ensure level is set even if distance extraction failed
    if not level or level == "unknown":
        if distance and '_' in distance:
            parts = distance.split('_')
            if 'high' in parts:
                level = 'high'
            elif 'middle' in parts:
                level = 'middle'
            elif 'low' in parts:
                level = 'low'
    
    # Extract core metrics with fallback chain
    metrics = {
        "file": filename,
        "model": data.get("model", "RF"),
        "mode": data.get("mode", "source_only"),
        "ranking_method": ranking_method if ranking_method else "mean_distance",
        "distance": distance if distance else "unknown",
        "level": level if level else None,
        "pos_rate": pos_rate,
        
        # AUC metrics
        "auc": (
            data.get("auc") or 
            data.get("roc_auc") or 
            data.get("metrics", {}).get("auc")
        ),
        "auc_pr": (
            data.get("auc_pr") or 
            data.get("metrics", {}).get("auc_pr") or 
            data.get("pr_curve", {}).get("auc_pr")
        ),
        
        # Classification metrics
        "f1": (
            data.get("f1_pos") or 
            get_metric_from_positive_class(cr, "f1-score") or 
            cr.get("macro avg", {}).get("f1-score")
        ),
        "accuracy": data.get("accuracy") or cr.get("accuracy"),
        "precision": (
            data.get("precision_pos") or 
            get_metric_from_positive_class(cr, "precision") or 
            cr.get("macro avg", {}).get("precision")
        ),
        "recall": (
            data.get("recall_pos") or 
            get_metric_from_positive_class(cr, "recall") or 
            cr.get("macro avg", {}).get("recall")
        ),
        "specificity": data.get("specificity"),
        "mse": data.get("mse") or data.get("metrics", {}).get("mse"),
        
        # Threshold-optimized metrics
        "precision_thr": (
            data.get("precision_thr_pos") or data.get("prec_thr")
        ),
        "recall_thr": (
            data.get("recall_thr_pos") or data.get("recall_thr")
        ),
        "f1_thr": (
            data.get("f1_thr_pos") or data.get("f1_thr")
        ),
        "f2_thr": (
            data.get("f2_thr_pos") or data.get("f2_thr")
        ),
        "specificity_thr": data.get("specificity_thr"),
        
        "split": "test",  # Default for compatibility
    }
    
    # Compute F2 score if not present
    if metrics.get("f2") is None:
        precision = metrics.get("precision")
        recall = metrics.get("recall")
        metrics["f2"] = compute_f2_score_from_pr(precision, recall)
    else:
        metrics["f2"] = data.get("f2")
    
    return metrics

