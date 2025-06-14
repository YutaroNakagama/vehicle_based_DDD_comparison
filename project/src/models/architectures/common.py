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
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_curve, auc, make_scorer, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

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
        if model == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,  
            }
            clf = LGBMClassifier(**params)

        elif model == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 5),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0),
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1, 
                "tree_method": "hist",  
            }
            clf = XGBClassifier(**params)

        elif model == "RF":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "random_state": 42,
                "class_weight": "balanced_subsample",
                "n_jobs": -1,  
            }
            clf = RandomForestClassifier(**params)

        elif model == "BalancedRF":
            sampling_strategy = trial.suggest_categorical(
                "sampling_strategy", ["auto", "majority", "not majority", "not minority", "all", 1.0]
            )
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "sampling_strategy": sampling_strategy,  
                "replacement": trial.suggest_categorical("replacement", [True, False]),
                "random_state": 42,
                # NOTE: class_weight is not needed; B-RF handles balancing internally
            }
            clf = BalancedRandomForestClassifier(**params)

        elif model == "CatBoost":
            params = {
                "iterations": trial.suggest_int("iterations", 100, 300),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "random_seed": 42,
                "verbose": 0,
                "eval_metric": "AUC",
                "loss_function": "Logloss"
            }
            clf = CatBoostClassifier(**params)

        elif model == "LogisticRegression":
            params = {
                "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 1000,
                "random_state": 42,
                "class_weight": "balanced"
            }
            clf = LogisticRegression(**params)

        elif model == "SVM":
            params = {
                "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "kernel": "linear",
                "probability": True,
                "random_state": 42,
                "class_weight": "balanced"
            }
            clf = SVC(**params)

        elif model == "DecisionTree":
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "random_state": 42,
                "class_weight": "balanced"
            }
            clf = DecisionTreeClassifier(**params)

        elif model == "AdaBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
                "random_state": 42,
            }
            clf = AdaBoostClassifier(**params)

        elif model == "GradientBoosting":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "random_state": 42,
            }
            clf = GradientBoostingClassifier(**params)

        elif model == "K-Nearest Neighbors":
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 3, 15),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            }
            clf = KNeighborsClassifier(**params)

        elif model == "MLP":
            params = {
                "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (100, 50)]),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
                "max_iter": 500,
                "random_state": 42
            }
            clf = MLPClassifier(**params)

        else:
            raise ValueError(f"Optuna tuning not implemented for model: {model}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[selected_features])
        roc_auc = make_scorer(roc_auc_score, needs_proba=True)

        try:
            score = cross_val_score(clf, X_train_scaled, y_train, cv=2, scoring=roc_auc, n_jobs=1).mean()
        except Exception as e:
            logging.warning(f"Scoring failed: {e}")
            return 0.0 

        return score

    # Run Optuna study
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)
    )
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)

    best_params = study.best_params

    logging.info(f"Best hyperparameters: {best_params}")
    logging.info(f"Selected features (from input): {selected_features}")

    # Final training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[selected_features])
    X_test_scaled = scaler.transform(X_test[selected_features])

    if model == "LightGBM":
        best_clf = LGBMClassifier(**best_params)

    elif model == "XGBoost":
        best_clf = XGBClassifier(**best_params)

    elif model == "CatBoost":
        best_clf = CatBoostClassifier(**best_params)

    elif model == "RF":
        if "class_weight" in best_params:
            best_params.pop("class_weight")
        best_clf = RandomForestClassifier(**best_params, class_weight="balanced_subsample")

    elif model == "BalancedRF":
        best_clf = BalancedRandomForestClassifier(**best_params)

    elif model == "LogisticRegression":
        if "class_weight" in best_params:
                best_params.pop("class_weight")
        best_clf = LogisticRegression(**best_params, class_weight="balanced")

    elif model == "SVM":
        best_clf = SVC(**best_params, probability=True, class_weight="balanced", random_state=42)

    elif model == "DecisionTree":
        if "class_weight" in best_params:
            best_params.pop("class_weight")
        best_clf = DecisionTreeClassifier(**best_params, class_weight="balanced")

    elif model == "AdaBoost":
        best_clf = AdaBoostClassifier(**best_params)

    elif model == "GradientBoosting":
        best_clf = GradientBoostingClassifier(**best_params)

    elif model == "K-Nearest Neighbors":
        best_clf = KNeighborsClassifier(**best_params)

    elif model == "MLP":
        best_clf = MLPClassifier(**best_params)

    else:
        raise ValueError(f"Unknown model: {model}")

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
    elif hasattr(best_clf, "decision_function"):
        y_pred_proba = best_clf.decision_function(X_test_scaled)
    else:
        y_pred_proba = None
    
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

    logging.info(f"{model} (Optuna) trained with ROC AUC: {roc_auc:.2f}")
    logging.info(classification_report(y_test, y_pred))

    # Threshold optimization (F1-score)
    if y_pred_proba is not None:
        from sklearn.metrics import precision_recall_curve, f1_score

        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        logging.info(f"Optimal threshold for F1: {best_threshold:.3f}")
        
        y_pred_opt = (y_pred_proba >= best_threshold).astype(int)
        logging.info("Classification Report with optimized threshold:")
        logging.info(classification_report(y_test, y_pred_opt))

        # Save threshold
        threshold_meta = {
            "model": model,
            "threshold": best_threshold,
            "metric": "F1-optimal",
        }
        with open(f"{MODEL_PKL_PATH}/{model_type}/threshold_{model}{suffix}.json", "w") as f:
            json.dump(threshold_meta, f, indent=2)

    else:
        logging.warning("Threshold optimization skipped: model does not support probability estimation.")

