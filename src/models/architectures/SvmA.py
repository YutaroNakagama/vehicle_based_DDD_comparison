"""SVM model with ANFIS-based feature weighting and PSO optimization.

This module implements:
- ANFIS-style feature importance calculation
- Feature selection using learned importance
- PSO-based hyperparameter and feature weighting optimization
- SVM training and evaluation

Trained models and selected features are saved using `joblib`.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from pyswarm import pso

from src.config import MODEL_PKL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def calculate_importance_degree(params: list[float], indices_df: pd.DataFrame) -> np.ndarray:
    """
    Compute importance degree per feature using ANFIS-style weighted indices.

    Parameters
    ----------
    params : list of float
        Weight parameters for each index in the order:
        [Fisher, Correlation, T-test, Mutual Information].
    indices_df : pandas.DataFrame
        DataFrame containing feature index scores.

    Returns
    -------
    numpy.ndarray
        Importance degree array, where values are:
        - 1   : High importance
        - 0.5 : Medium importance
        - 0   : Low importance
    """
    # --- Defensive conversion (must come first) ---
    if not isinstance(indices_df, pd.DataFrame):
        try:
            # Handle list/array/dict gracefully
            if isinstance(indices_df, (list, np.ndarray)):
                df = pd.DataFrame(indices_df)
                ncols = df.shape[1] if df.ndim > 1 else 1
                # Assign column names safely
                if ncols == 4:
                    df.columns = ["Fisher_Index", "Correlation_Index", "T-test_Index", "Mutual_Information_Index"]
                else:
                    # If only 1 column → copy it 4 times
                    if ncols == 1:
                        df = pd.concat([df] * 4, axis=1)
                    elif ncols > 4:
                        df = df.iloc[:, :4]
                    else:
                        df = pd.concat([df] * (4 // ncols), axis=1)
                    df.columns = ["Fisher_Index", "Correlation_Index", "T-test_Index", "Mutual_Information_Index"]
                indices_df = df
                logging.warning(f"indices_df auto-converted from {type(indices_df)} with shape {indices_df.shape}")
            elif isinstance(indices_df, dict):
                indices_df = pd.DataFrame(indices_df)
                missing = set(["Fisher_Index", "Correlation_Index", "T-test_Index", "Mutual_Information_Index"]) - set(indices_df.columns)
                for m in missing:
                    indices_df[m] = 0.0
                indices_df = indices_df[["Fisher_Index", "Correlation_Index", "T-test_Index", "Mutual_Information_Index"]]
                logging.warning("indices_df auto-built from dict keys.")
            else:
                raise TypeError(f"indices_df must be DataFrame, list, array, or dict — got {type(indices_df)}")

        except Exception as e:
            raise TypeError(
                f"indices_df could not be converted properly. Original type: {type(indices_df)}"
            ) from e

    # Ensure expected columns exist
    expected = ["Fisher_Index", "Correlation_Index", "T-test_Index", "Mutual_Information_Index"]
    for col in expected:
        if col not in indices_df.columns:
            indices_df[col] = 0.0

    # --- Ensure numeric dtype for all index columns ---
    for col in expected:
        if col in indices_df.columns:
            indices_df[col] = pd.to_numeric(indices_df[col], errors="coerce").fillna(0.0)
    # --- Compute weighted scores ---
    weighted_scores = (
        indices_df["Fisher_Index"] * params[0] +
        indices_df["Correlation_Index"] * params[1] +
        indices_df["T-test_Index"] * params[2] +
        indices_df["Mutual_Information_Index"] * params[3]
    )
    return np.where(weighted_scores > 0.75, 1, np.where(weighted_scores > 0.4, 0.5, 0))


def select_features(features_df: pd.DataFrame, importance_degree: np.ndarray) -> pd.DataFrame:
    """
    Select features based on importance threshold.

    Parameters
    ----------
    features_df : pandas.DataFrame
        Input feature matrix.
    importance_degree : numpy.ndarray
        Importance levels per feature (0, 0.5, or 1).

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only features with
        importance equal to 1.
    """
    return features_df.loc[:, importance_degree == 1]


def optimize_svm_anfis(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    indices_df: pd.DataFrame
) -> list[float]:
    """
    Optimize ANFIS feature weights and SVM hyperparameters using PSO.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix.
    y_train : pandas.Series
        Training labels.
    X_val : pandas.DataFrame
        Validation feature matrix.
    y_val : pandas.Series
        Validation labels.
    indices_df : pandas.DataFrame
        Feature importance index values.

    Returns
    -------
    list of float
        Optimal parameters in the form:
        [w1, w2, w3, w4, C, gamma].
    """
    def objective(params):
        importance_degree = calculate_importance_degree(params[:4], indices_df)
        X_train_sel = select_features(X_train, importance_degree)
        X_val_sel = select_features(X_val, importance_degree)

        if X_train_sel.shape[1] == 0:
            return 1.0  # Penalty for empty selection

        model = SVC(kernel='rbf', C=params[4], gamma=params[5])
        model.fit(X_train_sel, y_train)
        accuracy = accuracy_score(y_val, model.predict(X_val_sel))
        return -accuracy

    lb = [0, 0, 0, 0, 0.1, 0.001]
    ub = [1, 1, 1, 1, 10, 1]

    optimal_params, _ = pso(objective, lb, ub, swarmsize=3, maxiter=3)
    return optimal_params


def evaluate_model(model: SVC, X: pd.DataFrame, y: pd.Series, dataset_name: str) -> None:
    """
    Log classification metrics for a given dataset.

    Parameters
    ----------
    model : sklearn.svm.SVC
        Trained SVM model.
    X : pandas.DataFrame
        Input feature matrix.
    y : pandas.Series
        Ground truth labels.
    dataset_name : str
        Label used in log messages (e.g., ``"Training"``).

    Returns
    -------
    None
    """
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average=None)
    recall = recall_score(y, y_pred, average=None)
    f1 = f1_score(y, y_pred, average=None)
    conf_matrix = confusion_matrix(y, y_pred)

    logging.info(f"{dataset_name} Accuracy: {accuracy}")
    logging.info(f"{dataset_name} Precision: {precision}")
    logging.info(f"{dataset_name} Recall: {recall}")
    logging.info(f"{dataset_name} F1 Score: {f1}")
    logging.info(f"{dataset_name} Confusion Matrix:\n{conf_matrix}")


def SvmA_train(
    X_train: pd.DataFrame, X_val: pd.DataFrame,
    y_train: pd.Series, y_val: pd.Series,
    indices_df: pd.DataFrame, model: str
) -> None:
    """
    Train SVM using ANFIS-based feature weighting and PSO optimization.

    This function selects features, trains an SVM with optimized
    hyperparameters, and saves the model and selected features.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix.
    X_val : pandas.DataFrame
        Validation feature matrix.
    y_train : pandas.Series
        Training labels.
    y_val : pandas.Series
        Validation labels.
    indices_df : pandas.DataFrame
        Feature selection index scores.
    model : str
        Model name used for saving artifacts.

    Returns
    -------
    None
        Trained model and selected features are saved to disk.
    """
    logging.info("Starting SVM-ANFIS optimization...")

    optimal_params = optimize_svm_anfis(X_train, y_train, X_val, y_val, indices_df)
    best_anfis_params, best_C, best_gamma = optimal_params[:4], optimal_params[4], optimal_params[5]

    importance_degree = calculate_importance_degree(best_anfis_params, indices_df)
    X_train_sel = select_features(X_train, importance_degree)
    X_val_sel = select_features(X_val, importance_degree)

    # --- Safeguard: fallback if no features selected ---
    if X_train_sel.shape[1] == 0:
        logging.warning("[WARN] No features selected by ANFIS importance → using all input features instead.")
        X_train_sel = X_train.copy()
        X_val_sel = X_val.copy()

    svm_final = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
    svm_final.fit(X_train_sel, y_train)

    model_dir = f"{MODEL_PKL_PATH}/{model}"
    os.makedirs(model_dir, exist_ok=True)
    
    # --- Unified saving rules ---
    joblib.dump(svm_final, f"{model_dir}/{model}.pkl")
    joblib.dump(X_train_sel.columns.tolist(), f"{model_dir}/selected_features_{model}.pkl")

    # Save scaler (for consistency, even if identity here)
    from sklearn.preprocessing import StandardScaler
    dummy_scaler = StandardScaler().fit(X_train_sel)  # mimic preprocessing step
    joblib.dump(dummy_scaler, f"{model_dir}/scaler_{model}.pkl")
    
    logging.info("Model and features saved successfully.")

    evaluate_model(svm_final, X_train_sel, y_train, "Training")
    evaluate_model(svm_final, X_val_sel, y_val, "Validation")

    logging.info(f"Optimal ANFIS Parameters: {best_anfis_params}")
    logging.info(f"Optimal SVM Parameters (C, gamma): ({best_C}, {best_gamma})")

