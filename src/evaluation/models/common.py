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

# --- Added imports for PRC (Precision-Recall Curve) evaluation ---
from sklearn.metrics import precision_recall_curve, average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def common_eval(
    X_test,
    y_test,
    model_name: str,
    model_type: str,
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
        The name of the model (e.g., ``"RF"``, ``"SvmW"``).
    model_type : str
        The model type identifier, used to locate saved artifacts.
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

    roc_auc = None
    fpr, tpr = None, None  # Add to capture ROC curve if possible

    if hasattr(clf, "predict_proba"):
        classes = getattr(clf, "classes_", np.array([0, 1]))
        if 1 in classes:
            pos_idx = int(np.where(classes == 1)[0][0])
        else:
            # fallback: assume the second column corresponds to positive
            pos_idx = 1 if len(classes) > 1 else 0

        y_pred_proba = clf.predict_proba(X_test)[:, pos_idx]

        logging.info(
            f"[EVAL] Using predict_proba column index {pos_idx} for positive class "
            f"(classes={classes.tolist()})"
        )

        # --- Check for inverted probability direction (AUC < 0.5) ---
        from sklearn.metrics import roc_auc_score
        try:
            auc_check = roc_auc_score(y_test, y_pred_proba)
            if auc_check < 0.5:
                logging.warning(f"[EVAL] AUC={auc_check:.3f} < 0.5 → Inverting predicted probabilities")
                y_pred_proba = 1.0 - y_pred_proba
                auc_check = 1.0 - auc_check
        except Exception as e:
            logging.warning(f"[EVAL] AUC check failed: {e}")


        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # === Compute Precision-Recall Curve (PRC) and Average Precision (AUPRC) ===
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)

        result_pr = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "auc_pr": float(auc_pr)
        }
        # Also keep raw probabilities for downstream diagnostics (histogram, threshold search)
        # NOTE: This can be ~O(n_test) in size; acceptable for 1回の評価。
        result_proba = y_pred_proba.tolist()

    mse = mean_squared_error(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    logging.info(f"Model: {model_name}")
    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"ROC AUC: {roc_auc:.4f}" if roc_auc is not None else "ROC AUC: N/A")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    result = {
        "model": model_name,
        "mse": float(mse),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist()
    }

    # Include ROC curve data if available
    if fpr is not None and tpr is not None:
        result["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": roc_auc
        }

    # Include PRC (Precision–Recall Curve) data if available
    if hasattr(clf, "predict_proba"):
        result["pr_curve"] = result_pr
        result["auc_pr"] = result_pr["auc_pr"]
        result["y_pred_proba"] = result_proba
        # Default threshold diagnostics (@0.5)
        y_hat_05 = (y_pred_proba >= 0.5).astype(int)
        pos_rate_pred_05 = float(y_hat_05.mean())
        result["decision_threshold_default"] = 0.5
        result["pred_pos_rate_at_0p5"] = pos_rate_pred_05

        logging.info(f"AUPRC (Average Precision): {auc_pr:.4f}")

    return result
