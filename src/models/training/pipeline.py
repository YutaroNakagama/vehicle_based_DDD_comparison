"""Training pipeline for classical ML models using Optuna-based hyperparameter tuning.

This module supports ANFIS-based feature selection and trains a RandomForest classifier.
The model is saved as a `.pkl` file and selected features are stored for reproducibility.

This is a refactored version with logic split into submodules:
- sampling.oversampling: Sampling strategies for class imbalance
- training.optuna_tuning: Hyperparameter optimization with Optuna
- training.classifiers: Classifier instantiation and calibration
- training.evaluation: Evaluation and threshold optimization
"""

import os
import logging
import json
import numpy as np
import pandas as pd
import warnings
import contextlib
import sys
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import (
    classification_report, roc_curve, auc, make_scorer, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix, precision_recall_curve, average_precision_score
)

from src.config import (
    MODEL_PKL_PATH,
    configure_blas_threads,
    MIN_RECALL_THRESHOLD,
    FBETA_SCORE_BETA,
)
from src.utils.io.savers import save_artifacts
from src.evaluation.metrics import (
    calculate_extended_metrics,
    find_optimal_threshold,
    apply_threshold,
)
from src.utils.io.model_artifacts import load_model_artifacts

# Import refactored submodules
from src.models.sampling.oversampling import apply_oversampling
from src.models.training.optuna_tuning import (
    create_optuna_objective,
    run_optuna_optimization,
    OPTUNA_SUPPORTED_MODELS,
)
from src.models.training.classifiers import (
    create_classifier,
    apply_rf_calibration,
    fit_classifier,
    make_sample_weight,
)
from src.models.training.evaluation import (
    evaluate_classifier,
    optimize_threshold,
    prepare_results_dict,
    _eval_single_split,
)

# --- Limit CPU threads globally (important for PBS environments) ---
configure_blas_threads(n_threads=1)

# --- Disable joblib internal parallelization ---
import joblib
joblib.parallel_backend("sequential")

import gc

def common_train(
    X_train: "pd.DataFrame",
    X_val: "pd.DataFrame",
    X_test: "pd.DataFrame",
    y_train: "pd.Series",
    y_val: "pd.Series",
    y_test: "pd.Series",
    selected_features: list,
    model: str,
    model_name: str,
    mode: str,
    clf: object = None,
    scaler: object = None,
    suffix: str = "",
    data_leak: bool = False,
    use_oversampling: bool = False,
    oversample_method: str = "smote",
    target_ratio: float = 0.33,
    seed: int = 42,
) -> dict:
    """Train a classical ML model using Optuna and ANFIS-based feature selection.

    This function:
    - Performs feature importance estimation via ANFIS membership functions.
    - Uses Optuna to tune hyperparameters and feature selection threshold.
    - Trains the final model and saves it to disk.
    - Logs classification performance.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix.
    X_val : pandas.DataFrame
        Validation feature matrix.
    X_test : pandas.DataFrame
        Test feature matrix.
    y_train : pandas.Series
        Training labels.
    y_val : pandas.Series
        Validation labels.
    y_test : pandas.Series
        Test labels.
    selected_features : list of str
        List of selected feature names.
    model : str
        Model name (used for file naming).
    model_name : str
        Model name (previously `model_type`); unified naming across pipeline.
    clf : object, optional
        Classifier to train. If ``None``, a default model is selected internally.
    scaler : sklearn.preprocessing.StandardScaler, optional
        Pre-fitted scaler for feature normalization.
    suffix : str, default=""
        Suffix appended to saved file names (e.g., tags, strategies).
    data_leak : bool, default=False
        Whether to allow intentional data leakage (for ablation studies).
    use_oversampling : bool, default=False
        Apply oversampling to minority class.
    oversample_method : str, default="smote"
        Oversampling method to use.
    target_ratio : float, default=0.33
        Target minority/majority ratio for oversampling.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (best_clf, scaler, best_threshold, feature_meta, results)
    """
    # ====== Data preparation ======
    X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    X_val = X_val.loc[:, ~X_val.columns.duplicated()]
    X_test = X_test.loc[:, ~X_test.columns.duplicated()]

    # ====== Apply oversampling if requested ======
    if use_oversampling:
        X_train, y_train = apply_oversampling(
            X_train, y_train,
            method=oversample_method,
            target_ratio=target_ratio,
            random_state=seed,
        )

    # ====== Scaler validation ======
    if scaler is None:
        raise ValueError("Scaler must be provided (pre-fitted in pipeline).")

    # ====== Scale features ======
    X_train_scaled = scaler.transform(X_train[selected_features])
    X_val_scaled = scaler.transform(X_val[selected_features])
    X_test_scaled = scaler.transform(X_test[selected_features])

    # ====== Optuna hyperparameter optimization ======
    objective = create_optuna_objective(
        model=model,
        X_train_scaled=X_train_scaled,
        y_train=y_train.to_numpy() if hasattr(y_train, 'to_numpy') else y_train,
        scaler=scaler,
        selected_features=selected_features,
        data_leak=data_leak,
        X_test_scaled=X_test_scaled if data_leak else None,
        y_test=y_test if data_leak else None,
    )

    best_params = run_optuna_optimization(
        model=model,
        objective=objective,
        seed=seed,
        suffix=suffix,
        mode=mode,
    )

    logging.info(f"Selected features (from input): {selected_features}")

    # ====== Create and train classifier ======
    best_clf = create_classifier(model, best_params)

    # Prepare sample weights
    sw_train = make_sample_weight(y_train)
    sw_val = make_sample_weight(y_val)
    sw_test = make_sample_weight(y_test)

    # Track if classifier is already fitted (e.g., after RF calibration)
    already_fitted = False

    # Special handling for RF: apply calibration
    if model == "RF":
        best_clf, already_fitted = apply_rf_calibration(
            best_clf,
            X_train_scaled, X_val_scaled,
            y_train, y_val,
            sw_train, sw_val,
        )

    # Final fit (skip if already fitted)
    if not already_fitted:
        fit_classifier(
            best_clf, X_train_scaled, y_train, sw_train,
            data_leak=data_leak,
            X_val_scaled=X_val_scaled,
            X_test_scaled=X_test_scaled,
            y_val=y_val,
            y_test=y_test,
            sw_val=sw_val,
            sw_test=sw_test,
        )

    # ====== Prepare feature metadata ======
    feature_meta = {
        "selected_features": selected_features,
        "feature_source": model_name,
    }

    # ====== Evaluate classifier ======
    m_train, m_val, m_test = evaluate_classifier(
        best_clf,
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        model,
    )

    # ====== Prepare results ======
    results = prepare_results_dict(m_train, m_val, m_test)

    # ====== Save artifacts ======
    save_artifacts(
        model_name=model,
        suffix=suffix,
        best_clf=best_clf,
        scaler=scaler,
        selected_features=selected_features,
        feature_meta=feature_meta,
    )

    # ====== Threshold optimization ======
    best_threshold = optimize_threshold(
        y_val, m_val.get("_proba"),
        y_test, m_test.get("_proba"),
        beta=FBETA_SCORE_BETA,
    )

    return best_clf, scaler, best_threshold, feature_meta, results
