"""Model instantiation and calibration helpers.

This module provides functions for creating classifier instances
with optimized hyperparameters and applying calibration.
"""

import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

from src.config import RF_CLASS_WEIGHT, CV_FOLDS_CALIBRATION


def create_classifier(
    model: str,
    best_params: Dict[str, Any],
) -> Any:
    """Create a classifier instance with optimized hyperparameters.

    Parameters
    ----------
    model : str
        Model name.
    best_params : Dict[str, Any]
        Best hyperparameters from Optuna optimization.

    Returns
    -------
    Classifier instance.

    Raises
    ------
    ValueError
        If model is unknown.
    """
    if model == "LightGBM":
        return LGBMClassifier(**best_params)

    elif model == "XGBoost":
        return XGBClassifier(**best_params)

    elif model == "CatBoost":
        return CatBoostClassifier(**best_params)

    elif model == "RF":
        # Remove class_weight from params if present (we override it)
        params = {k: v for k, v in best_params.items() if k != "class_weight"}
        return RandomForestClassifier(
            **params,
            class_weight=RF_CLASS_WEIGHT,
            n_jobs=1,
            random_state=42,
        )

    elif model == "BalancedRF":
        return BalancedRandomForestClassifier(**best_params, random_state=42)

    elif model == "EasyEnsemble":
        return EasyEnsembleClassifier(**best_params, random_state=42)

    elif model == "LogisticRegression":
        params = {k: v for k, v in best_params.items() if k != "class_weight"}
        return LogisticRegression(**params, class_weight="balanced", random_state=42)

    elif model in ["SVM", "SvmW"]:
        return SVC(**best_params, probability=True, class_weight="balanced", random_state=42)

    elif model == "DecisionTree":
        params = {k: v for k, v in best_params.items() if k != "class_weight"}
        return DecisionTreeClassifier(**params, class_weight="balanced", random_state=42)

    elif model == "AdaBoost":
        return AdaBoostClassifier(**best_params, random_state=42)

    elif model == "GradientBoosting":
        return GradientBoostingClassifier(**best_params, random_state=42)

    elif model == "K-Nearest Neighbors":
        return KNeighborsClassifier(**best_params)

    elif model == "MLP":
        return MLPClassifier(**best_params, random_state=42)

    else:
        raise ValueError(f"Unknown model: {model}")


def apply_rf_calibration(
    clf: RandomForestClassifier,
    X_train_scaled: np.ndarray,
    X_val_scaled: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    sw_train: np.ndarray,
    sw_val: np.ndarray,
) -> Tuple[CalibratedClassifierCV, bool]:
    """Apply sigmoid calibration to Random Forest.

    Parameters
    ----------
    clf : RandomForestClassifier
        Base classifier.
    X_train_scaled : np.ndarray
        Scaled training features.
    X_val_scaled : np.ndarray
        Scaled validation features.
    y_train : np.ndarray
        Training labels.
    y_val : np.ndarray
        Validation labels.
    sw_train : np.ndarray
        Training sample weights.
    sw_val : np.ndarray
        Validation sample weights.

    Returns
    -------
    Tuple[CalibratedClassifierCV, bool]
        Calibrated classifier and flag indicating it's already fitted.
    """
    logging.info("[CALIBRATION] Performing single-step calibration (Sigmoid, 5-fold CV) using train+val combined...")

    X_combined = np.vstack([X_train_scaled, X_val_scaled])
    y_combined = np.concatenate([y_train, y_val])
    sw_combined = np.concatenate([sw_train, sw_val]).astype(float)

    # Train the base RF
    clf.fit(X_combined, y_combined, sample_weight=sw_combined)

    # Apply sigmoid calibration
    cv_calib = StratifiedKFold(n_splits=CV_FOLDS_CALIBRATION, shuffle=True, random_state=42)
    calib = CalibratedClassifierCV(clf, cv=cv_calib, method='sigmoid')

    try:
        calib.fit(X_combined, y_combined, sample_weight=sw_combined)
    except TypeError:
        logging.warning("[CALIBRATION] sample_weight not supported. Fitting without weights.")
        calib.fit(X_combined, y_combined)

    logging.info("[CALIBRATION] Completed single-step sigmoid calibration successfully.")

    # Log number of trees
    _log_forest_size(calib)

    return calib, True


def _log_forest_size(clf) -> None:
    """Log the number of trees in a calibrated forest classifier."""
    try:
        n_trees = None
        if hasattr(clf, "calibrated_classifiers_") and clf.calibrated_classifiers_:
            base = clf.calibrated_classifiers_[0].estimator
            if hasattr(base, "estimators_"):
                n_trees = len(base.estimators_)
        elif hasattr(clf, "base_estimator_") and hasattr(clf.base_estimator_, "estimators_"):
            n_trees = len(clf.base_estimator_.estimators_)
        if n_trees is not None:
            logging.info(f"Number of trees in the forest: {n_trees}")
    except Exception as e:
        logging.debug(f"[CALIBRATION] Could not read n_trees: {e}")


def fit_classifier(
    clf,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    sw_train: np.ndarray,
    data_leak: bool = False,
    X_val_scaled: Optional[np.ndarray] = None,
    X_test_scaled: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    sw_val: Optional[np.ndarray] = None,
    sw_test: Optional[np.ndarray] = None,
) -> None:
    """Fit classifier with sample weights.

    Parameters
    ----------
    clf : Classifier
        Classifier to fit.
    X_train_scaled : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Training labels.
    sw_train : np.ndarray
        Training sample weights.
    data_leak : bool, default=False
        Whether to include all data in training (for ablation).
    X_val_scaled, X_test_scaled : np.ndarray, optional
        Additional data for data_leak mode.
    y_val, y_test : np.ndarray, optional
        Additional labels for data_leak mode.
    sw_val, sw_test : np.ndarray, optional
        Additional weights for data_leak mode.
    """
    try:
        if data_leak and X_val_scaled is not None and X_test_scaled is not None:
            X_all = np.vstack([X_train_scaled, X_val_scaled, X_test_scaled])
            y_all = np.concatenate([y_train, y_val, y_test])
            sw_all = np.concatenate([sw_train, sw_val, sw_test]).astype(float)
            clf.fit(X_all, y_all, sample_weight=sw_all)
        else:
            clf.fit(X_train_scaled, y_train, sample_weight=sw_train)
    except TypeError:
        # Some estimators don't accept sample_weight
        if data_leak and X_val_scaled is not None and X_test_scaled is not None:
            X_all = np.vstack([X_train_scaled, X_val_scaled, X_test_scaled])
            y_all = np.concatenate([y_train, y_val, y_test])
            clf.fit(X_all, y_all)
        else:
            clf.fit(X_train_scaled, y_train)


def make_sample_weight(y: np.ndarray) -> np.ndarray:
    """Return class-balanced sample weights.

    Parameters
    ----------
    y : np.ndarray
        Label array.

    Returns
    -------
    np.ndarray
        Sample weights.
    """
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_all = len(y)
    if n_pos == 0 or n_neg == 0 or n_all == 0:
        return np.ones_like(y, dtype=float)
    w_pos = n_all / (2.0 * n_pos)
    w_neg = n_all / (2.0 * n_neg)
    return np.where(y == 1, w_pos, w_neg).astype(float)
