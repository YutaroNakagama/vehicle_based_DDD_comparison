"""Evaluation helpers for model training pipeline.

This module provides functions for evaluating trained classifiers
and optimizing classification thresholds.
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
from sklearn.metrics import classification_report, fbeta_score

from src.evaluation.metrics import (
    calculate_extended_metrics,
    find_optimal_threshold,
    apply_threshold,
)


def evaluate_classifier(
    clf,
    X_train_scaled: np.ndarray,
    X_val_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    model: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Evaluate classifier on train/val/test splits.

    Parameters
    ----------
    clf : Classifier
        Trained classifier.
    X_train_scaled, X_val_scaled, X_test_scaled : np.ndarray
        Scaled feature matrices.
    y_train, y_val, y_test : np.ndarray
        Labels.
    model : str
        Model name for logging.

    Returns
    -------
    Tuple[Dict, Dict, Dict]
        Metrics dictionaries for train, val, test splits.
        Each contains internal keys (_y_true, _y_pred, _proba) for threshold optimization.
    """
    m_train = _eval_single_split(clf, X_train_scaled, y_train)
    m_val = _eval_single_split(clf, X_val_scaled, y_val)
    m_test = _eval_single_split(clf, X_test_scaled, y_test)

    logging.info(
        f"{model} (Optuna) metrics: "
        f"train acc={m_train['accuracy']:.3f}, val acc={m_val['accuracy']:.3f}, test acc={m_test['accuracy']:.3f}"
    )

    # Log test classification report
    y_pred_test = clf.predict(X_test_scaled)
    logging.info("Test classification report:\n" + classification_report(y_test, y_pred_test))

    return m_train, m_val, m_test


def _eval_single_split(clf, X_scaled: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Evaluate classifier on a single split.

    Parameters
    ----------
    clf : Classifier
        Trained classifier.
    X_scaled : np.ndarray
        Scaled features.
    y : np.ndarray
        Labels.

    Returns
    -------
    Dict[str, Any]
        Metrics dictionary with internal keys for downstream processing.
    """
    yhat = clf.predict(X_scaled)

    y_score = None
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_scaled)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_scaled)

    out = calculate_extended_metrics(y, yhat, y_score, zero_division=0)
    out["_y_true"] = y
    out["_y_pred"] = yhat
    out["_proba"] = y_score

    return out


def optimize_threshold(
    y_val: np.ndarray,
    proba_val: np.ndarray,
    y_test: Optional[np.ndarray] = None,
    proba_test: Optional[np.ndarray] = None,
    beta: float = 2.0,
) -> Optional[float]:
    """Optimize classification threshold on validation set.

    Uses F-beta score for threshold selection. F2 (beta=2.0) gives
    4x weight to Recall over Precision, appropriate for DDD safety.

    Parameters
    ----------
    y_val : np.ndarray
        Validation labels.
    proba_val : np.ndarray
        Validation probabilities.
    y_test : np.ndarray, optional
        Test labels for reporting.
    proba_test : np.ndarray, optional
        Test probabilities for reporting.
    beta : float, default=2.0
        F-beta score parameter.

    Returns
    -------
    Optional[float]
        Optimal threshold, or None if probabilities not available.
    """
    if proba_val is None:
        logging.warning("Threshold optimization skipped: model does not support probability estimation.")
        return None

    best_threshold, best_f2 = find_optimal_threshold(y_val, proba_val, beta=beta)
    logging.info(f"Optimal threshold for F{beta} (β={beta}): {best_threshold:.3f} (F{beta}={best_f2:.3f})")

    # Report validation metrics at optimal threshold
    thr_val = _apply_threshold_metrics(proba_val, y_val, best_threshold, beta)
    logging.info(f"Validation (F{beta}-opt threshold) metrics: " + json.dumps(thr_val))

    # Report test metrics if available
    if proba_test is not None and y_test is not None:
        thr_test = _apply_threshold_metrics(proba_test, y_test, best_threshold, beta)
        logging.info(f"Test (F{beta}-opt threshold from Val) metrics: " + json.dumps(thr_test))

    return best_threshold


def _apply_threshold_metrics(
    proba: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    beta: float = 2.0,
) -> Dict[str, float]:
    """Calculate metrics at a specific threshold.

    Parameters
    ----------
    proba : np.ndarray
        Predicted probabilities.
    y_true : np.ndarray
        True labels.
    threshold : float
        Classification threshold.
    beta : float, default=2.0
        F-beta parameter.

    Returns
    -------
    Dict[str, float]
        Metrics dictionary.
    """
    yhat = apply_threshold(proba, threshold)
    metrics = calculate_extended_metrics(y_true, yhat, proba, zero_division=0, include_roc=False, include_pr=False)
    metrics[f"f{int(beta)}"] = float(fbeta_score(y_true, yhat, beta=beta, zero_division=0))
    return metrics


def prepare_results_dict(
    m_train: Dict[str, Any],
    m_val: Dict[str, Any],
    m_test: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Prepare results dictionary for saving.

    Removes internal keys (starting with '_') from metrics.

    Parameters
    ----------
    m_train, m_val, m_test : Dict[str, Any]
        Metrics dictionaries.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Cleaned results dictionary.
    """
    return {
        "train": {k: v for k, v in m_train.items() if not k.startswith("_")},
        "val": {k: v for k, v in m_val.items() if not k.startswith("_")},
        "test": {k: v for k, v in m_test.items() if not k.startswith("_")},
    }
