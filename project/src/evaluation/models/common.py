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

    Assumes:
    - X_test has already been scaled and filtered to selected features.
    - y_test is aligned with X_test.

    Returns:
        dict: Evaluation results suitable for JSON output.
    """
    y_pred = clf.predict(X_test)

    roc_auc = None
    if hasattr(clf, "predict_proba"):
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

    mse = mean_squared_error(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    logging.info(f"Model: {model_name}")
    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"ROC AUC: {roc_auc:.4f}" if roc_auc is not None else "ROC AUC: N/A")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    return {
        "model": model_name,
        "mse": float(mse),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist()
    }

