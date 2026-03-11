"""Optuna-based hyperparameter optimization for ML models.

This module provides the Optuna objective function and optimization
logic for hyperparameter tuning of various classifiers.
"""

import logging
import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

from src.config import (
    N_TRIALS,
    OPTUNA_N_STARTUP_TRIALS,
    OPTUNA_N_WARMUP_STEPS,
    OPTUNA_INTERVAL_STEPS,
    CV_FOLDS_OPTUNA,
    CV_FOLDS_OPTUNA_DATA_LEAK,
    LIGHTGBM_N_ESTIMATORS_RANGE,
    LIGHTGBM_LEARNING_RATE_RANGE,
    FBETA_SCORE_BETA,
)
from src.evaluation.metrics import find_optimal_threshold


# List of models that support Optuna tuning
OPTUNA_SUPPORTED_MODELS = [
    "LightGBM", "XGBoost", "CatBoost",
    "RF", "BalancedRF", "EasyEnsemble",
    "LogisticRegression", "SVM", "SvmW",
    "DecisionTree", "AdaBoost", "GradientBoosting",
    "K-Nearest Neighbors", "MLP",
]

# Timeout for Optuna optimization (0 = no timeout)
OPTUNA_TIMEOUT_SEC = int(os.getenv("OPTUNA_TIMEOUT_SEC", "0"))

# Number of parallel jobs for models (default 1, can be overridden via environment)
# Use -1 for all cores, or specify a number like 4, 8, etc.
N_JOBS_OVERRIDE = int(os.getenv("N_JOBS_OVERRIDE", "1"))


def create_optuna_objective(
    model: str,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    scaler,
    selected_features: List[str],
    data_leak: bool = False,
    X_test_scaled: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    seed: int = 42,
):
    """Create an Optuna objective function for the given model.

    Parameters
    ----------
    model : str
        Model name (e.g., "RF", "LightGBM").
    X_train_scaled : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Training labels.
    scaler : StandardScaler
        Fitted scaler object.
    selected_features : List[str]
        List of feature names.
    data_leak : bool, default=False
        Whether to allow data leakage (for ablation studies).
    X_test_scaled : np.ndarray, optional
        Scaled test features (used in data_leak mode).
    y_test : np.ndarray, optional
        Test labels (used in data_leak mode).
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Callable
        Optuna objective function.
    """

    def objective(trial):
        clf = _suggest_hyperparameters(trial, model, seed=seed)
        if clf is None:
            return 0.0

        try:
            score = _evaluate_with_cv(
                clf, X_train_scaled, y_train,
                data_leak, X_test_scaled, y_test,
                model=model,
            )
        except Exception as e:
            msg = str(e)
            if "not in index" in msg or "Scoring failed" in msg or "only one class" in msg:
                return 0.0
            logging.debug(f"[cross_val_score outer exception suppressed] {msg}")
            return 0.0

        return score

    return objective


def _suggest_hyperparameters(trial: optuna.Trial, model: str, seed: int = 42):
    """Suggest hyperparameters based on model type.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object.
    model : str
        Model name.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Classifier instance or None if model not supported.
    """
    if model == "LightGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *LIGHTGBM_N_ESTIMATORS_RANGE),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "learning_rate": trial.suggest_float("learning_rate", *LIGHTGBM_LEARNING_RATE_RANGE, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": N_JOBS_OVERRIDE,
        }
        return LGBMClassifier(**params)

    elif model == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *LIGHTGBM_N_ESTIMATORS_RANGE),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "learning_rate": trial.suggest_float("learning_rate", *LIGHTGBM_LEARNING_RATE_RANGE, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0),
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": seed,
            "n_jobs": N_JOBS_OVERRIDE,
            "tree_method": "hist",
        }
        return XGBClassifier(**params)

    elif model == "RF":
        # Expanded search space based on analysis (2026-01-10 update):
        # - max_depth: Expanded range since 69% of best were at upper boundary (100 or None)
        #   Previous: [3, 5, 7, 10, 15, 20, 30, None]
        #   New: [5, 10, 20, 30, 50, 100, None] - allows deeper trees
        # - max_features: Added 0.05 since 67% of best were at min (0.1)
        # - n_estimators: Expanded to 1500 since 12% were at upper boundary
        # - min_samples_leaf: Minimum 5 to prevent overfitting (was 1)
        # - min_samples_split: Minimum 10 to prevent overfitting (was 2)
        # - oob_score: True to enable out-of-bag score for generalization monitoring
        max_depth_choice = trial.suggest_categorical("max_depth", [5, 10, 20, 30, 50, 100, None])
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
            "max_depth": max_depth_choice,
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 100),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 50),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.05, 0.1, 0.2, 0.3, 0.5]),
            "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.1),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
            "random_state": seed,
            "n_jobs": N_JOBS_OVERRIDE,
            "max_samples": trial.suggest_categorical("max_samples", [None, 0.5, 0.7, 0.9]),
            "warm_start": False,
            "bootstrap": True,
            "oob_score": True,
        }
        return RandomForestClassifier(**params)

    elif model == "BalancedRF":
        # Updated 2026-01-10: Expanded max_depth range to allow deeper trees
        sampling_strategy = trial.suggest_categorical(
            "sampling_strategy", ["auto", "majority", "not majority", "not minority", "all", 1.0]
        )
        max_depth_choice = trial.suggest_categorical("max_depth", [10, 20, 30, 50, 100, 150, None])
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": max_depth_choice,
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
            "sampling_strategy": sampling_strategy,
            "replacement": trial.suggest_categorical("replacement", [True, False]),
            "random_state": seed,
        }
        return BalancedRandomForestClassifier(**params)

    elif model == "EasyEnsemble":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 5, 100),
            "sampling_strategy": trial.suggest_categorical(
                "sampling_strategy", ["auto", "majority", "not majority", "all", 0.5, 0.8, 1.0]
            ),
            "replacement": trial.suggest_categorical("replacement", [False, True]),
            "random_state": seed,
            "n_jobs": N_JOBS_OVERRIDE,
        }
        return EasyEnsembleClassifier(**params)

    elif model == "CatBoost":
        params = {
            "iterations": trial.suggest_int("iterations", 100, 300),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_seed": seed,
            "verbose": 0,
            "eval_metric": "AUC",
            "loss_function": "Logloss",
        }
        return CatBoostClassifier(**params)

    elif model == "LogisticRegression":
        params = {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": seed,
            "class_weight": "balanced",
        }
        return LogisticRegression(**params)

    elif model == "SvmW":
        # Zhao et al. 2009: RBF kernel (C now tuned instead of fixed at 300)
        params = {
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "kernel": "rbf",
            "gamma": trial.suggest_float("gamma", 1e-5, 1.0, log=True),
            "probability": True,
            "random_state": seed,
            "class_weight": "balanced",
        }
        return SVC(**params)

    elif model == "SVM":
        params = {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "kernel": "linear",
            "probability": True,
            "random_state": seed,
            "class_weight": "balanced",
        }
        return SVC(**params)

    elif model == "DecisionTree":
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "random_state": seed,
            "class_weight": "balanced",
        }
        return DecisionTreeClassifier(**params)

    elif model == "AdaBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0),
            "random_state": seed,
        }
        return AdaBoostClassifier(**params)

    elif model == "GradientBoosting":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "random_state": seed,
        }
        return GradientBoostingClassifier(**params)

    elif model == "K-Nearest Neighbors":
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 15),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        }
        return KNeighborsClassifier(**params)

    elif model == "MLP":
        params = {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (100, 50)]),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "max_iter": 500,
            "random_state": seed,
        }
        return MLPClassifier(**params)

    else:
        raise ValueError(f"Optuna tuning not implemented for model: {model}")


def _make_sample_weight(y: np.ndarray) -> np.ndarray:
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


def _evaluate_with_cv(
    clf,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    data_leak: bool = False,
    X_test_scaled: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    model: str = "",
) -> float:
    """Evaluate classifier using cross-validation.

    Parameters
    ----------
    clf : Classifier
        Classifier instance.
    X_train_scaled : np.ndarray
        Scaled training features.
    y_train : np.ndarray
        Training labels.
    data_leak : bool
        Whether to use data leakage mode.
    X_test_scaled : np.ndarray, optional
        Test features for data_leak mode.
    y_test : np.ndarray, optional
        Test labels for data_leak mode.
    model : str, default=""
        Model name. Used to skip sample_weight for internally balanced
        models (BalancedRF, EasyEnsemble).

    Returns
    -------
    float
        Cross-validation score (F2).
    """
    from src.models.training.classifiers import INTERNALLY_BALANCED_MODELS
    skip_sw = model in INTERNALLY_BALANCED_MODELS
    if data_leak and X_test_scaled is not None and y_test is not None:
        X_all = np.vstack([X_train_scaled, X_test_scaled])
        y_all = np.concatenate([y_train, y_test])
        cv = StratifiedKFold(n_splits=CV_FOLDS_OPTUNA_DATA_LEAK, shuffle=True, random_state=42)

        # Check for single-class folds
        for i, (_, va_idx) in enumerate(cv.split(X_all, y_all)):
            bincounts = np.bincount(y_all[va_idx].astype(int))
            if len(bincounts) < 2 or np.any(bincounts == 0):
                return 0.0

        scores = []
        for tr_idx, va_idx in cv.split(X_all, y_all):
            try:
                clf.fit(X_all[tr_idx], y_all[tr_idx])
            except TypeError:
                clf.fit(X_all[tr_idx], y_all[tr_idx])
            if hasattr(clf, "predict_proba"):
                p = clf.predict_proba(X_all[va_idx])[:, 1]
            else:
                p = clf.decision_function(X_all[va_idx])
            opt_thr, f2 = find_optimal_threshold(y_all[va_idx], p, beta=2.0)
            scores.append(f2)
        return float(np.nanmean(scores)) if scores else 0.0

    # Normal mode: 3-fold CV on training data
    cv = StratifiedKFold(n_splits=CV_FOLDS_OPTUNA, shuffle=True, random_state=42)
    scores = []

    for i, (tr_idx, va_idx) in enumerate(cv.split(X_train_scaled, y_train)):
        y_tr = y_train[tr_idx] if isinstance(y_train, np.ndarray) else y_train.to_numpy()[tr_idx]
        y_va = y_train[va_idx] if isinstance(y_train, np.ndarray) else y_train.to_numpy()[va_idx]

        try:
            if skip_sw:
                clf.fit(X_train_scaled[tr_idx], y_tr)
            else:
                sw_tr = _make_sample_weight(y_tr)
                clf.fit(X_train_scaled[tr_idx], y_tr, sample_weight=sw_tr)
        except TypeError:
            clf.fit(X_train_scaled[tr_idx], y_tr)

        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(X_train_scaled[va_idx])[:, 1]
        else:
            p = clf.decision_function(X_train_scaled[va_idx])

        opt_thr, f2 = find_optimal_threshold(y_va, p, beta=2.0)
        logging.debug(f"[CV] Fold {i}: F2 = {f2:.6f} (thr={opt_thr:.3f})")
        scores.append(f2)

    if not scores or np.any(np.isnan(scores)):
        return 0.0
    return float(np.nanmean(scores))


def run_optuna_optimization(
    model: str,
    objective,
    seed: int = 42,
    suffix: str = "",
    mode: str = "pooled",
) -> Dict[str, Any]:
    """Run Optuna optimization and return best parameters.

    Parameters
    ----------
    model : str
        Model name.
    objective : Callable
        Optuna objective function.
    seed : int, default=42
        Random seed.
    suffix : str, default=""
        Experiment suffix.
    mode : str, default="pooled"
        Training mode.

    Returns
    -------
    Dict[str, Any]
        Best hyperparameters.
    """
    if model not in OPTUNA_SUPPORTED_MODELS:
        logging.warning(f"Optuna tuning skipped: not implemented for model={model}")
        return {}

    logging.info(f"[Optuna] Creating study with TPESampler(seed={seed}), N_TRIALS={N_TRIALS}")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=OPTUNA_N_STARTUP_TRIALS,
            n_warmup_steps=OPTUNA_N_WARMUP_STEPS,
            interval_steps=OPTUNA_INTERVAL_STEPS,
        ),
    )

    def _optuna_logging_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            best_value = study.best_value
            logging.info(
                f"[Optuna] Trial {trial.number:3d}: "
                f"value={trial.value:.4f}, best={best_value:.4f}"
            )

    optimize_kwargs = dict(
        n_trials=N_TRIALS,
        n_jobs=1,
        gc_after_trial=True,
        callbacks=[_optuna_logging_callback],
    )
    if OPTUNA_TIMEOUT_SEC > 0:
        optimize_kwargs["timeout"] = OPTUNA_TIMEOUT_SEC

    study.optimize(objective, **optimize_kwargs)

    _log_convergence_summary(study)
    _save_optuna_study(study, model, mode, suffix, seed)

    logging.info(f"Best hyperparameters: {study.best_params}")
    return study.best_params


def _log_convergence_summary(study: optuna.Study) -> None:
    """Log convergence summary for the Optuna study."""
    trials = study.trials
    values = [t.value for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(values) >= 10:
        last_10_best = max(values[-10:])
        overall_best = max(values)
        improvement = overall_best - last_10_best
        logging.info(
            f"[Optuna Convergence] Total trials: {len(trials)}, "
            f"Best: {overall_best:.4f}, Last 10 best: {last_10_best:.4f}, "
            f"Improvement in last 10: {improvement:.4f}"
        )


def _save_optuna_study(
    study: optuna.Study,
    model: str,
    mode: str,
    suffix: str,
    seed: int,
) -> None:
    """Save Optuna study artifacts (pickle, CSV, JSON)."""
    try:
        pbs_jobid = os.getenv('PBS_JOBID', os.getenv('PBS_ARRAY_INDEX', 'local'))
        if '.' in str(pbs_jobid):
            pbs_jobid = str(pbs_jobid).split('.')[0]
        if '[' in str(pbs_jobid):
            pbs_jobid = str(pbs_jobid).split('[')[0]

        optuna_dir = Path(f"models/{model}/{pbs_jobid}")
        optuna_dir.mkdir(parents=True, exist_ok=True)

        safe_suffix = suffix.replace('/', '_').replace(' ', '_') if suffix else 'default'
        safe_mode = mode if isinstance(mode, str) else 'unknown'
        base_name = f"optuna_{model}_{safe_mode}_{safe_suffix}_s{seed}"

        # Save trials dataframe
        trials_df = study.trials_dataframe()
        trials_df['model'] = model if isinstance(model, str) else str(model)
        trials_df['mode'] = safe_mode
        trials_df['seed'] = seed
        trials_df['suffix'] = suffix if suffix else ''
        optuna_csv_path = optuna_dir / f"{base_name}_trials.csv"
        trials_df.to_csv(optuna_csv_path, index=False)
        logging.info(f"[Optuna] Study trials saved to: {optuna_csv_path}")

        # Save study object
        optuna_pkl_path = optuna_dir / f"{base_name}_study.pkl"
        with open(optuna_pkl_path, 'wb') as f:
            pickle.dump(study, f)
        logging.info(f"[Optuna] Study object saved to: {optuna_pkl_path}")

        # Save convergence history
        convergence_history = []
        best_so_far = float('-inf')
        for t in study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            if t.value is not None and t.value > best_so_far:
                best_so_far = t.value
            convergence_history.append({
                'trial_number': t.number,
                'value': t.value,
                'best_so_far': best_so_far,
                'params': t.params,
                'datetime_start': t.datetime_start.isoformat() if t.datetime_start else None,
                'datetime_complete': t.datetime_complete.isoformat() if t.datetime_complete else None,
                'duration_seconds': (t.datetime_complete - t.datetime_start).total_seconds()
                    if t.datetime_start and t.datetime_complete else None,
            })

        optuna_json_path = optuna_dir / f"{base_name}_convergence.json"
        with open(optuna_json_path, 'w') as f:
            json.dump({
                'metadata': {
                    'model': model if isinstance(model, str) else str(model),
                    'mode': safe_mode,
                    'suffix': suffix if suffix else '',
                    'seed': seed,
                    'n_trials': N_TRIALS,
                    'best_value': study.best_value,
                    'best_params': study.best_params,
                },
                'trials': convergence_history,
            }, f, indent=2, default=str)
        logging.info(f"[Optuna] Convergence history saved to: {optuna_json_path}")

        # Log per-trial hyperparameters
        logging.info(f"[Optuna] === Per-trial Hyperparameter History ===")
        for entry in convergence_history:
            logging.info(
                f"[Optuna] Trial {entry['trial_number']:3d}: "
                f"F2={entry['value']:.4f}, best_so_far={entry['best_so_far']:.4f}, "
                f"params={entry['params']}"
            )

    except Exception as e:
        logging.warning(f"[Optuna] Failed to save study data: {e}")
