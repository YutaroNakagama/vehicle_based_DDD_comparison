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
    roc_auc_score, mean_squared_error, classification_report
)

from src.config import MODEL_PKL_PATH


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
    selected_features = joblib.load(features_path)  # これは list[str]

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
    model: str,
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
    model : str
        Model name used for saving and logging.
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
    """

    # Clean column names
    X_test.columns = X_test.columns.str.strip()

    # Fill missing columns
    missing_cols = set(selected_features) - set(X_test.columns)
    if missing_cols:
        print(f"Warning: Missing columns in X_test: {missing_cols}")
        for col in missing_cols:
            X_test[col] = 0.0

    # Align feature order
    X_test = X_test[selected_features]

    # Predict and compute metrics
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None)
    recall = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred, average=None)
    conf_matrix = confusion_matrix(y_test, y_pred)

    try:
        roc_auc = roc_auc_score(y_test, clf.decision_function(X_test))
    except AttributeError:
        roc_auc = None
    
    mse = mean_squared_error(y_test, y_pred)

    logging.info(f"Model: {model}")  
    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"ROC AUC: {roc_auc:.4f}" if roc_auc is not None else "ROC AUC: N/A")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    return {
        "model": model,
        "accuracy": float(accuracy),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1_score": f1.tolist(),
        "confusion_matrix": conf_matrix.tolist()
    }
