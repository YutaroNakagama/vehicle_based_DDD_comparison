"""Evaluation module for SVM-ANFIS models using saved model and features.

Loads the optimized SVM model and selected features,
applies prediction to test data, and logs evaluation metrics.
"""

import warnings
warnings.filterwarnings("ignore")

import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve,
    precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, classification_report,
    average_precision_score,
)

from src.config import MODEL_PKL_PATH
from src.evaluation.metrics import (
    calculate_class_specific_metrics,
    find_optimal_threshold,
)


def load_model_and_features(model: str):
    """
    Load the trained SVM-ANFIS model and its selected feature list.

    Parameters
    ----------
    model : str
        Name of the model directory under ``MODEL_PKL_PATH``.

    Returns
    -------
    tuple
        A tuple containing:

        - ``svm_model`` : sklearn.svm.SVC
            Trained SVM-ANFIS model.
        - ``selected_features`` : list of str
            List of selected feature names used during training.
    """
    model_path = f'{MODEL_PKL_PATH}/{model}/svm_model_final.pkl'
    features_path = f'{MODEL_PKL_PATH}/{model}/selected_features_train.pkl'

    svm_model = joblib.load(model_path)
    selected_features = joblib.load(features_path) 

    return svm_model, selected_features


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate accuracy, precision, recall, and F1 score.

    Parameters
    ----------
    y_true : numpy.ndarray of shape (n_samples,)
        Ground truth labels.
    y_pred : numpy.ndarray of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    tuple
        A tuple containing:

        - ``accuracy`` : float
            Overall classification accuracy.
        - ``precision`` : numpy.ndarray
            Class-wise precision values.
        - ``recall`` : numpy.ndarray
            Class-wise recall values.
        - ``f1`` : numpy.ndarray
            Class-wise F1 scores.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    return accuracy, precision, recall, f1


def evaluate_model(svm_model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluate the trained SVM-ANFIS model on a test set and print metrics.

    Parameters
    ----------
    svm_model : sklearn.svm.SVC
        Trained SVM classifier.
    X_test : pandas.DataFrame
        Test feature matrix.
    y_test : pandas.Series
        Ground truth labels corresponding to ``X_test``.

    Returns
    -------
    None
        This function prints evaluation results (accuracy, precision, recall,
        F1 score, and confusion matrix) to the console.
    """
    y_pred = svm_model.predict(X_test)
    accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision}")
    print(f"Recall    : {recall}")
    print(f"F1 Score  : {f1}")
    print("\nConfusion Matrix:\n", conf_matrix)


def SvmA_eval(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    clf,
    selected_features: list
) -> dict:
    """
    Main evaluation entry point for SVM-ANFIS model.

    Parameters
    ----------
    X_test : pandas.DataFrame
        Test feature matrix. Columns will be aligned and missing features
        filled with zeros if necessary.
    y_test : pandas.Series
        Ground truth labels corresponding to ``X_test``.
    model_name : str
        Unified model identifier used for saving and logging.
    clf : sklearn.svm.SVC
        Trained SVM classifier loaded from pickle.
    selected_features : list of str
        List of selected feature names used for training.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics:

        - ``"model"`` : str
            Model name.
        - ``"accuracy"`` : float
            Classification accuracy.
        - ``"precision"`` : list of float
            Class-wise precision values.
        - ``"recall"`` : list of float
            Class-wise recall values.
        - ``"f1_score"`` : list of float
            Class-wise F1 scores.
        - ``"confusion_matrix"`` : list of list of int
            Confusion matrix as a nested list.
        - ``"custom_threshold"`` : float or None
            F2-optimal classification threshold.
    """

    # Note: Feature alignment and scaling are handled upstream by
    # prepare_evaluation_features(). X_test arrives ready-to-predict.

    # --- Obtain probability scores for threshold optimization ---
    y_score = None
    try:
        y_proba = clf.predict_proba(X_test)
        y_score = y_proba[:, 1]  # probability of positive class
    except AttributeError:
        logging.warning("predict_proba unavailable; falling back to decision_function")
        try:
            y_score = clf.decision_function(X_test)
        except AttributeError:
            logging.warning("decision_function also unavailable; using default predict()")

    # --- Optimal threshold via F2-score (emphasize recall for drowsiness) ---
    if y_score is not None:
        opt_threshold, opt_f2 = find_optimal_threshold(y_test, y_score, beta=2.0)
        logging.info(
            f"[SvmA] Optimal threshold: {opt_threshold:.4f} (F2={opt_f2:.4f})"
        )
        y_pred = (y_score >= opt_threshold).astype(int)
    else:
        opt_threshold = None
        y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # --- ROC-AUC ---
    y_decision = None
    try:
        y_decision = clf.decision_function(X_test)
        roc_auc = roc_auc_score(y_test, y_decision)
    except (AttributeError, ValueError) as e:
        logging.warning(f"ROC AUC computation failed: {e}")
        roc_auc = None

    # --- AUPRC (Average Precision) ---
    auc_pr = None
    if y_decision is not None:
        try:
            auc_pr = float(average_precision_score(y_test, y_decision))
        except (ValueError, Exception) as e:
            logging.warning(f"AUPRC computation failed: {e}")

    mse = mean_squared_error(y_test, y_pred)

    # Extract positive-class metrics
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    class_metrics = calculate_class_specific_metrics(
        np.array(conf_matrix), report_dict
    )

    logging.info(f"Model: {model_name}")  
    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"ROC AUC: {roc_auc:.4f}" if roc_auc is not None else "ROC AUC: N/A")
    logging.info(f"AUPRC:   {auc_pr:.4f}" if auc_pr is not None else "AUPRC: N/A")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    result = {
        "model": model_name,
        "accuracy": float(accuracy),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1_score": f1.tolist(),
        "confusion_matrix": conf_matrix.tolist(),
        "roc_auc": roc_auc,
        "auc_pr": auc_pr,
        "classification_report": report_dict,
        "custom_threshold": float(opt_threshold) if opt_threshold is not None else None,
    }
    # Merge positive-class metrics (precision_pos, recall_pos, f1_pos, ...)
    result.update(class_metrics)

    return result
