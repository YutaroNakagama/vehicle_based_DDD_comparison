"""Evaluation module for classical ML models using saved pickle classifiers.

This module loads a pre-trained model and its selected features, applies scaling,
and computes classification metrics (MSE, ROC AUC, report) on the test data.
"""

import pickle
import numpy as np
import logging
from sklearn.metrics import classification_report, roc_curve, auc, mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from src.config import MODEL_PKL_PATH

from src.utils.metrics_helper import calculate_extended_metrics, calculate_class_specific_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def common_eval(
    X_test,
    y_test,
    model_name: str,
    clf
) -> dict:
    """
    Evaluate a classical ML model using test data and a trained classifier.

    Assumes that ``X_test`` has already been scaled and filtered to the selected
    features, and that ``y_test`` is aligned with ``X_test``.

    Parameters
    ----------
    X_test : array-like of shape (n_samples, n_features)
        Feature matrix for test data. Must be preprocessed and aligned with the
        selected features used during training.
    y_test : array-like of shape (n_samples,)
        True labels corresponding to ``X_test``.
    model_name : str
        Unified model identifier used across pipeline (was `model_type`).
    clf : sklearn.base.BaseEstimator
        Trained classifier object (loaded from pickle).

    Returns
    -------
    dict
        Dictionary containing evaluation results, including:

        - ``"model"`` : str
            Model name.
        - ``"mse"`` : float
            Mean squared error between predictions and true labels.
        - ``"roc_auc"`` : float or None
            Area under the ROC curve (if probabilities are available).
        - ``"classification_report"`` : dict
            Classification metrics in scikit-learn report format.
        - ``"confusion_matrix"`` : list of list of int
            Confusion matrix as a nested list.
        - ``"roc_curve"`` : dict, optional
            Contains ``"fpr"``, ``"tpr"``, and ``"auc"`` if ROC data is available.
    """
    y_pred = clf.predict(X_test)

    # Get probability scores for ROC/PR curves
    y_score = None
    if hasattr(clf, "predict_proba"):
        classes = getattr(clf, "classes_", np.array([0, 1]))
        if 1 in classes:
            pos_idx = int(np.where(classes == 1)[0][0])
        else:
            # fallback: assume the second column corresponds to positive
            pos_idx = 1 if len(classes) > 1 else 0

        y_score = clf.predict_proba(X_test)[:, pos_idx]

        logging.info(
            f"[EVAL] Using predict_proba column index {pos_idx} for positive class "
            f"(classes={classes.tolist()})"
        )

        # --- Check for inverted probability direction (AUC < 0.5) ---
        from sklearn.metrics import roc_auc_score
        try:
            auc_check = roc_auc_score(y_test, y_score)
            if auc_check < 0.5:
                logging.warning(f"[EVAL] AUC={auc_check:.3f} < 0.5 → Inverting predicted probabilities")
                y_score = 1.0 - y_score
        except Exception as e:
            logging.warning(f"[EVAL] AUC check failed: {e}")

    # Use unified metrics calculation
    result = calculate_extended_metrics(
        y_test, y_pred, y_score, 
        zero_division=0, 
        include_roc=True, 
        include_pr=True
    )

    # Add model name and MSE for backward compatibility
    from sklearn.metrics import mean_squared_error
    result["model"] = model_name
    result["mse"] = float(mean_squared_error(y_test, y_pred))

    # Extract class-specific metrics
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = np.array(result["confusion_matrix"])
    class_metrics = calculate_class_specific_metrics(conf_matrix, report_dict)
    result.update(class_metrics)

    # Add classification report for backward compatibility
    result["classification_report"] = report_dict

    # Default threshold diagnostics (@0.5) if probabilities available
    if y_score is not None:
        y_hat_05 = (y_score >= 0.5).astype(int)
        pos_rate_pred_05 = float(y_hat_05.mean())
        result["decision_threshold_default"] = 0.5
        result["pred_pos_rate_at_0p5"] = pos_rate_pred_05
        result["y_pred_proba"] = y_score.tolist()

    logging.info(f"Model: {model_name}")
    logging.info(f"MSE: {result['mse']:.4f}")
    if result.get("roc_auc"):
        logging.info(f"ROC AUC: {result['roc_auc']:.4f}")
    if result.get("auc_pr"):
        logging.info(f"AUPRC (Average Precision): {result['auc_pr']:.4f}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    return result
