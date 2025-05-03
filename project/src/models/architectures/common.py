"""Training pipeline for classical ML models using Optuna-based hyperparameter tuning.

This module supports ANFIS-based feature selection and trains a RandomForest classifier.
The model is saved as a `.pkl` file and selected features are stored for reproducibility.
"""

import os
import optuna
import pickle
import logging
import json
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import classification_report, roc_curve, auc, make_scorer, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.models.feature_selection.anfis import calculate_id
from src.config import MODEL_PKL_PATH, N_TRIALS

def common_train(
    X_train, X_test, y_train, y_test,
    selected_features,  
    model: str, model_type: str,
    clf=None, suffix: str = ""
):
    """Train a classical ML model (RandomForest) using Optuna and ANFIS-based feature selection.

    This function:
    - Performs feature importance estimation via ANFIS membership functions.
    - Uses Optuna to tune hyperparameters and feature selection threshold.
    - Trains the final model and saves it to disk.
    - Logs classification performance.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        X_test (pd.DataFrame): Testing feature matrix.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
        feature_indices (pd.DataFrame): Feature selection scores per feature (e.g., Fisher, MI).
        model (str): Model name (used for file naming).
        model_type (str): Model group (e.g., 'common', 'SvmA').
        clf: Optional classifier (currently unused; default is RandomForest).

    Returns:
        None
    """

    # Remove duplicated column names while preserving order
    X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    X_test = X_test.loc[:, ~X_test.columns.duplicated()]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "random_state": 42,
        }

        clf = RandomForestClassifier(**params, class_weight="balanced")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[selected_features])
        roc_auc = make_scorer(roc_auc_score, needs_proba=True)
        score = cross_val_score(clf, X_train_scaled, y_train, cv=3, scoring=roc_auc).mean()
        return score

    # Run Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    best_params = study.best_params

    logging.info(f"Best hyperparameters: {best_params}")
    logging.info(f"Selected features (from input): {selected_features}")

    # Final training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[selected_features])
    X_test_scaled = scaler.transform(X_test[selected_features])

    best_clf = RandomForestClassifier(**best_params, class_weight="balanced")
    best_clf.fit(X_train_scaled, y_train)

    # Save model and features
    logging.info(f"Saving {model} with {len(selected_features)} features.")
    os.makedirs(f"{MODEL_PKL_PATH}/{model_type}", exist_ok=True)
    with open(f"{MODEL_PKL_PATH}/{model_type}/{model}{suffix}.pkl", "wb") as f:
        pickle.dump(best_clf, f)
    with open(f"{MODEL_PKL_PATH}/{model_type}/selected_features_train_{model}{suffix}.pkl", "wb") as f:
        pickle.dump(selected_features, f)
    # Save fitted scaler
    with open(f"{MODEL_PKL_PATH}/{model_type}/scaler_{model}{suffix}.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save metadata (feature source info)
    feature_meta = {
        "selected_features": selected_features,
        "feature_source": model_type  # e.g., "common"
    }
    with open(f"{MODEL_PKL_PATH}/{model_type}/feature_meta_{model}{suffix}.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    # Evaluate
    y_pred = best_clf.predict(X_test_scaled)
    roc_auc = 0
    if hasattr(best_clf, "predict_proba"):
        y_pred_proba = best_clf.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

    logging.info(f"{model} (Optuna) trained with ROC AUC: {roc_auc:.2f}")
    logging.info(classification_report(y_test, y_pred))

