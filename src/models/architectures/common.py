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
import warnings
import contextlib
import sys
from sklearn.exceptions import FitFailedWarning
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    classification_report, roc_curve, auc, make_scorer, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix, precision_recall_curve, average_precision_score
)


# ========== Custom metric: Precision at minimum Recall ==========
def precision_at_min_recall(y_true, y_proba, min_recall=0.70):
    """Calculate maximum precision at or above minimum recall.
    
    This metric is useful for Precision-focused optimization while
    maintaining a minimum safety threshold (Recall).
    
    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities for the positive class.
    min_recall : float, default=0.70
        Minimum recall threshold. Returns max precision where recall >= min_recall.
    
    Returns
    -------
    float
        Maximum precision at recall >= min_recall. Returns 0.0 if no threshold
        satisfies the minimum recall constraint.
    """
    if len(y_true) == 0 or len(y_proba) == 0:
        return 0.0
    
    # Handle edge case: only one class present
    if len(np.unique(y_true)) < 2:
        return 0.0
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find indices where recall >= min_recall
    valid_idx = recalls >= min_recall
    if not np.any(valid_idx):
        return 0.0  # No threshold satisfies min_recall
    
    # Return maximum precision in valid range
    return float(np.max(precisions[valid_idx]))
# ================================================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

from src.data_pipeline.augmentation import augment_minority_class

from src.config import (
    MODEL_PKL_PATH,
    N_TRIALS,
    OPTUNA_N_STARTUP_TRIALS,
    OPTUNA_N_WARMUP_STEPS,
    OPTUNA_INTERVAL_STEPS,
    configure_blas_threads,
)

from src.utils.io.savers import save_artifacts
from src.utils.evaluation.metrics import (
    calculate_extended_metrics,
    find_optimal_threshold,
    apply_threshold,
)
from src.utils.io.model_artifacts import load_model_artifacts

# --- Limit CPU threads globally (important for PBS environments) ---
configure_blas_threads(n_threads=1)

# --- Disable joblib and Optuna internal parallelization ---
import joblib
joblib.parallel_backend("sequential")  # Prevent joblib from spawning extra workers

# --- Ensure Optuna runs trials strictly serially ---
optuna.logging.set_verbosity(optuna.logging.WARNING)
OPTUNA_TIMEOUT_SEC = int(os.getenv("OPTUNA_TIMEOUT_SEC", "0"))  # 0 = no timeout

import gc

def common_train(
    X_train, X_val, X_test, y_train, y_val, y_test,
    selected_features,
    model: str, model_name: str,
    mode: str,
    clf=None, scaler=None, suffix: str = "",
    data_leak: bool = False,
    eval_only: bool = False,
    train_only: bool = False,
    use_oversampling: bool = False,
    oversample_method: str = "smote",
):
    """
    Train a classical ML model using Optuna and ANFIS-based feature selection.

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

    Returns
    -------
    None
        The trained model, selected features, scaler, and evaluation metrics
        are saved to disk as pickle, JSON, and CSV artifacts.
    """

    if eval_only:
        # ====== eval_only mode ======
        logging.info("[EVAL_ONLY] Loading pre-trained model and scaler...")
        # define out_dir (was referenced before assignment)
        out_dir = f"{MODEL_PKL_PATH}/{model_name}"

        # Use unified artifact loader
        best_clf, scaler, selected_features = load_model_artifacts(
            model_name=model,
            base_dir=out_dir,
            suffix=suffix,
            mode=mode,
            use_joblib=True
        )

        if best_clf is None or scaler is None or selected_features is None:
            logging.error("[EVAL_ONLY] Failed to load required artifacts. Aborting.")
            return {}

        # Scaling
        X_val_scaled  = scaler.transform(X_val[selected_features])
        X_test_scaled = scaler.transform(X_test[selected_features])

        # Reuse evaluation function (unified metrics)
        def _eval_split(Xs, ys):
            yhat = best_clf.predict(Xs)
            proba = best_clf.predict_proba(Xs)[:, 1] if hasattr(best_clf, "predict_proba") else None
            return calculate_extended_metrics(ys, yhat, proba, zero_division=0)

        m_val  = _eval_split(X_val_scaled, y_val)
        m_test = _eval_split(X_test_scaled, y_test)

        logging.info(f"[EVAL_ONLY] Validation metrics: {json.dumps(m_val, indent=2)}")
        logging.info(f"[EVAL_ONLY] Test metrics: {json.dumps(m_test, indent=2)}")

        # ===== Save metrics (eval only) =====
        rows = []
        rows.append({"split": "val",  **m_val})
        rows.append({"split": "test", **m_test})
        os.makedirs(f"{MODEL_PKL_PATH}/{model_name}", exist_ok=True)
    
        # Ensure mode is included in the suffix
        eval_suffix = suffix + f"_{model_name}_evalonly"
        pd.DataFrame(rows).to_csv(
            f"{MODEL_PKL_PATH}/{model_name}/metrics_{model}_{mode}{eval_suffix}.csv",
            index=False
        )
        logging.info(f"[EVAL_ONLY] Saved metrics CSV -> metrics_{model}_{mode}{eval_suffix}.csv")
    
        return {"val": m_val, "test": m_test}

    X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    X_val   = X_val.loc[:,   ~X_val.columns.duplicated()]
    X_test  = X_test.loc[:,  ~X_test.columns.duplicated()]

    # Apply oversampling if requested
    if use_oversampling:
        logging.info(f"Applying oversampling method: {oversample_method}")
        logging.info(f"Class distribution before oversampling: {np.bincount(y_train)}")
        
        # Use conservative sampling_strategy to avoid extreme oversampling
        # Instead of 1:1 balance, aim for minority:majority = 1:3 ratio
        minority_count = np.bincount(y_train).min()
        majority_count = np.bincount(y_train).max()
        target_ratio = 0.33  # 1:3 ratio (minority:majority)
        
        if oversample_method == "smote":
            sampler = SMOTE(
                sampling_strategy=target_ratio,
                random_state=42,
                k_neighbors=min(5, minority_count - 1)
            )
        elif oversample_method == "adasyn":
            sampler = ADASYN(
                sampling_strategy=target_ratio,
                random_state=42,
                n_neighbors=min(5, minority_count - 1)
            )
        elif oversample_method == "borderline":
            sampler = BorderlineSMOTE(
                sampling_strategy=target_ratio,
                random_state=42,
                k_neighbors=min(5, minority_count - 1)
            )
        elif oversample_method == "smote_tomek":
            # SMOTE + Tomek Links: oversample then clean boundary pairs
            # Note: SMOTETomek requires sampling_strategy as ratio (0-1) or dict, not count
            sampler = SMOTETomek(
                sampling_strategy=target_ratio,
                random_state=42,
                n_jobs=1,
                smote=SMOTE(
                    random_state=42,
                    k_neighbors=min(5, minority_count - 1)
                )
            )
            logging.info("Using SMOTE + Tomek Links (boundary cleaning)")
        elif oversample_method == "smote_enn":
            # SMOTE + ENN: oversample then clean noisy samples
            # Note: SMOTEENN requires sampling_strategy as ratio (0-1) or dict, not count
            sampler = SMOTEENN(
                sampling_strategy=target_ratio,
                random_state=42,
                n_jobs=1,
                smote=SMOTE(
                    random_state=42,
                    k_neighbors=min(5, minority_count - 1)
                )
            )
            logging.info("Using SMOTE + ENN (aggressive noise cleaning)")
        elif oversample_method == "smote_rus":
            # SMOTE + RandomUnderSampler: hybrid approach
            # First oversample minority to 50%, then undersample majority to balance
            smote = SMOTE(
                sampling_strategy=0.5,  # Minority becomes 50% of majority
                random_state=42,
                k_neighbors=min(5, minority_count - 1)
            )
            rus = RandomUnderSampler(
                sampling_strategy=0.8,  # Final ratio: minority = 80% of majority
                random_state=42
            )
            sampler = ImbPipeline([
                ('smote', smote),
                ('rus', rus)
            ])
            logging.info("Using SMOTE + RandomUnderSampler (hybrid sampling)")
        elif oversample_method == "undersample_rus":
            # Random Under-Sampling only (no oversampling)
            # Reduces majority class to achieve target_ratio
            sampler = RandomUnderSampler(
                sampling_strategy=target_ratio,  # minority/majority ratio after sampling
                random_state=42
            )
            logging.info(f"Using Random Under-Sampling only (target ratio: {target_ratio})")
        elif oversample_method == "undersample_tomek":
            # Tomek Links only (boundary cleaning)
            # Removes majority samples that form Tomek links with minority
            sampler = TomekLinks(
                sampling_strategy='majority',  # Only remove majority class samples
                n_jobs=1
            )
            logging.info("Using Tomek Links only (boundary cleaning)")
        elif oversample_method in ["jitter", "scale", "jitter_scale"]:
            # Time-series inspired augmentation (does not use imblearn sampler)
            logging.info(f"Using time-series augmentation: {oversample_method}")
            X_train, y_train = augment_minority_class(
                X_train, y_train,
                method=oversample_method,
                target_ratio=target_ratio,
                adaptive_sigma="interclass",  # Use data-driven sigma estimation
                random_state=42,
            )
            # Skip the fit_resample step below
            sampler = None
        else:
            raise ValueError(f"Unknown oversample_method: {oversample_method}")
        
        # Apply imblearn sampler (skip if using jitter/scale augmentation)
        if sampler is not None:
            X_train, y_train = sampler.fit_resample(X_train, y_train)
        logging.info(f"Class distribution after oversampling: {np.bincount(y_train)}")
        logging.info(f"Oversampling ratio: minority increased from {minority_count} to {np.bincount(y_train).min()}")

    def objective(trial):
        logging.debug(f'X_train.shape: {X_train.shape}')
        logging.debug(f'X_train nan: {np.isnan(X_train.values).sum()} inf: {np.isinf(X_train.values).sum()}')
        logging.debug(f'X_train col nan count: {np.isnan(X_train.values).sum(axis=0)}')
        logging.debug(f'X_train all const col: {[c for c in X_train.columns if X_train[c].nunique() == 1]}')
        logging.debug(f'selected_features: {selected_features}')
        logging.debug(f'y_train shape: {y_train.shape}')
        logging.debug(f'y_train nan: {np.isnan(y_train).sum()} unique: {np.unique(y_train)}')
        logging.debug(f'X_train[selected_features] shape: {X_train[selected_features].shape}')
        logging.debug(f'X_train[selected_features] col nan: {np.isnan(X_train[selected_features].values).sum(axis=0)}')
 
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
                "n_jobs": 1,  
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
                "n_jobs": 1, 
                "tree_method": "hist",  
            }
            clf = XGBClassifier(**params)

        elif model == "RF":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 500),
                "max_depth": trial.suggest_int("max_depth", 6, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
                "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.1),
                "random_state": 42,
                "class_weight": "balanced_subsample",
                "n_jobs": 1,  
                "max_samples": None,
                "warm_start": False,
                "bootstrap": True,
                "oob_score": False,
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

        elif model == "EasyEnsemble":
            # EasyEnsemble: ensemble of balanced subsets with AdaBoost
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 50),
                "sampling_strategy": trial.suggest_categorical(
                    "sampling_strategy", ["auto", "majority", "not majority", "all"]
                ),
                "replacement": trial.suggest_categorical("replacement", [False, True]),
                "random_state": 42,
                "n_jobs": 1,
            }
            clf = EasyEnsembleClassifier(**params)

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

        elif model in ["SVM", "SvmW"]:
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

        try:
            if data_leak:
                X_all = np.vstack([X_train[selected_features], X_test[selected_features]])
                y_all = np.concatenate([y_train, y_test])
                X_all_scaled = scaler.transform(X_all)
                cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
                for i, (_, va_idx) in enumerate(cv.split(X_all_scaled, y_all)):
                    bincounts = np.bincount(y_all[va_idx].astype(int))
                    logging.debug(f"[CV-Leak] Fold {i} y_val bincount: {bincounts}")
                    if len(bincounts) < 2 or np.any(bincounts == 0):
                        logging.debug(f"[CV-Leak] Fold {i} has only one class! Skipping trial.")
                        return 0.0
                # Manual CV with AP scoring and (if possible) sample_weight
                scores = []
                for tr_idx, va_idx in cv.split(X_all_scaled, y_all):
                    try:
                        clf.fit(X_all_scaled[tr_idx], y_all[tr_idx])
                    except TypeError:
                        clf.fit(X_all_scaled[tr_idx], y_all[tr_idx])
                    if hasattr(clf, "predict_proba"):
                        p = clf.predict_proba(X_all_scaled[va_idx])[:, 1]
                    else:
                        p = clf.decision_function(X_all_scaled[va_idx])
                    # Use Precision@Recall≥70% for Precision-focused optimization
                    scores.append(precision_at_min_recall(y_all[va_idx], p, min_recall=0.70))
                score = float(np.nanmean(scores)) if len(scores) else 0.0
            else:
                # Use pre-trained scaler even inside the objective function
                X_train_scaled = scaler.transform(X_train[selected_features])
                # Use 3-fold CV for faster training while maintaining reasonable evaluation
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

                # ---- Manual CV with sample_weight & AP scoring ----
                # Build class-balanced sample weights for each fold
                def _make_sw(y):
                    y = np.asarray(y)
                    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
                    n_all = len(y)
                    if n_pos == 0 or n_neg == 0 or n_all == 0:
                        return np.ones_like(y, dtype=float)
                    return np.where(y == 1, n_all/(2.0*n_pos), n_all/(2.0*n_neg)).astype(float)

                scores = []
                for i, (tr_idx, va_idx) in enumerate(cv.split(X_train_scaled, y_train)):
                    y_tr = y_train.to_numpy()[tr_idx]
                    sw_tr = _make_sw(y_tr)
                    try:
                        clf.fit(X_train_scaled[tr_idx], y_tr, sample_weight=sw_tr)
                    except TypeError:
                        clf.fit(X_train_scaled[tr_idx], y_tr)
                    y_va = y_train.to_numpy()[va_idx]
                    if hasattr(clf, "predict_proba"):
                        p = clf.predict_proba(X_train_scaled[va_idx])[:, 1]
                    else:
                        p = clf.decision_function(X_train_scaled[va_idx])
                    # Use F2 score for Recall-focused optimization (better for imbalanced data)
                    y_pred = (p >= 0.5).astype(int)
                    f2 = fbeta_score(y_va, y_pred, beta=2, zero_division=0)
                    # Avoid flooding stdout; keep detailed per-fold in debug only
                    logging.debug(f"[CV] Fold {i}: F2 = {f2:.6f}")
                    scores.append(f2)
                if not scores or np.any(np.isnan(scores)):
                    return 0.0
                score = float(np.nanmean(scores))

        except Exception as e:
            msg = str(e)
            # --- Suppress known benign scoring errors silently ---
            if "not in index" in msg or "Scoring failed" in msg or "only one class" in msg:
                return 0.0
            else:
                logging.debug(f"[cross_val_score outer exception suppressed] {msg}")
                return 0.0
 
        return score

    # Run Optuna only for supported models
    optuna_supported = [
        "LightGBM", "XGBoost", "CatBoost",
        "RF", "BalancedRF", "EasyEnsemble",
        "LogisticRegression", "SVM",
        "DecisionTree", "AdaBoost", "GradientBoosting",
        "K-Nearest Neighbors", "MLP"
    ]

    if model in optuna_supported:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=OPTUNA_N_STARTUP_TRIALS,
                n_warmup_steps=OPTUNA_N_WARMUP_STEPS,
                interval_steps=OPTUNA_INTERVAL_STEPS
            )
        )
        # === Safe Optuna execution (no parallel trials, forced GC after each) ===
        optimize_kwargs = dict(
            n_trials=N_TRIALS,
            n_jobs=1,
            gc_after_trial=True,
        )
        if OPTUNA_TIMEOUT_SEC > 0:
            optimize_kwargs["timeout"] = OPTUNA_TIMEOUT_SEC
        study.optimize(objective, **optimize_kwargs)
        best_params = study.best_params
        logging.info(f"Best hyperparameters: {best_params}")
    else:
        logging.warning(f"Optuna tuning skipped: not implemented for model={model}")
        best_params = {}

    logging.info(f"Selected features (from input): {selected_features}")

    if scaler is None:
        raise ValueError("Scaler must be provided (pre-fitted in pipeline).")

    X_train_scaled = scaler.transform(X_train[selected_features])
    X_val_scaled   = scaler.transform(X_val[selected_features])
    X_test_scaled  = scaler.transform(X_test[selected_features])

    # ========== class-balanced sample_weight helpers ==========
    def _make_sample_weight(y: np.ndarray) -> np.ndarray:
        """Return class-balanced sample_weight (no-ops if any class count is 0)."""
        y = np.asarray(y)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        n_all = len(y)
        if n_pos == 0 or n_neg == 0 or n_all == 0:
            return np.ones_like(y, dtype=float)
        w_pos = n_all / (2.0 * n_pos)
        w_neg = n_all / (2.0 * n_neg)
        sw = np.where(y == 1, w_pos, w_neg).astype(float)
        return sw

    sw_train = _make_sample_weight(y_train)
    sw_val   = _make_sample_weight(y_val)
    sw_test  = _make_sample_weight(y_test)
    # ================================================================

    # Track whether the classifier has already been fully fitted (e.g., after calibration)
    already_fitted = False

    if model == "LightGBM":
        best_clf = LGBMClassifier(**best_params)

    elif model == "XGBoost":
        best_clf = XGBClassifier(**best_params)

    elif model == "CatBoost":
        best_clf = CatBoostClassifier(**best_params)

    elif model == "RF":
        if "class_weight" in best_params:
            best_params.pop("class_weight")
        # Precision-focused: reduced minority weight from 10.0 to 3.0
        # Ensure random_state=42 for reproducibility (not included in Optuna best_params)
        best_clf = RandomForestClassifier(
            **best_params,
            class_weight={0:1.0, 1:3.0},
            n_jobs=1,
            random_state=42
        )

        # --- Simplified calibration using train+val together (more stable) ---
        from sklearn.calibration import CalibratedClassifierCV

        logging.info("[CALIBRATION] Performing single-step calibration (Sigmoid, 5-fold CV) using train+val combined...")

        # Concatenate train and validation sets for calibration stability
        X_combined = np.vstack([X_train_scaled, X_val_scaled])
        y_combined = np.concatenate([y_train, y_val])
        sw_combined = np.concatenate([sw_train, sw_val]).astype(float)

        # Train the base RF
        best_clf.fit(X_combined, y_combined, sample_weight=sw_combined)

        # Apply sigmoid calibration over 5-fold CV with fixed seed for reproducibility
        cv_calib = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        calib = CalibratedClassifierCV(best_clf, cv=cv_calib, method='sigmoid')
        try:
            calib.fit(X_combined, y_combined, sample_weight=sw_combined)
        except TypeError:
            logging.warning("[CALIBRATION] sample_weight not supported by this sklearn version. Fitting without weights.")
            calib.fit(X_combined, y_combined)

        best_clf = calib
        already_fitted = True

        logging.info("[CALIBRATION] Completed single-step sigmoid calibration successfully.")

        # Check the number of trees after fitting (defensive across sklearn versions)
        try:
            n_trees = None
            # sklearn >= 1.0: calibrated_classifiers_ -> .estimator for each fold
            if hasattr(best_clf, "calibrated_classifiers_") and best_clf.calibrated_classifiers_:
                base = best_clf.calibrated_classifiers_[0].estimator
                if hasattr(base, "estimators_"):
                    n_trees = len(base.estimators_)
            # fallback: some versions expose base_estimator_ on CV wrapper
            elif hasattr(best_clf, "base_estimator_") and hasattr(best_clf.base_estimator_, "estimators_"):
                n_trees = len(best_clf.base_estimator_.estimators_)
            if n_trees is not None:
                logging.info(f"Number of trees in the forest: {n_trees}")
        except Exception as _e:
            logging.debug(f"[CALIBRATION] Could not read n_trees: {_e}")

    elif model == "BalancedRF":
        best_clf = BalancedRandomForestClassifier(**best_params, random_state=42)

    elif model == "EasyEnsemble":
        best_clf = EasyEnsembleClassifier(**best_params, random_state=42)

    elif model == "LogisticRegression":
        if "class_weight" in best_params:
                best_params.pop("class_weight")
        best_clf = LogisticRegression(**best_params, class_weight="balanced", random_state=42)

    elif model in ["SVM", "SvmW"]:
        best_clf = SVC(**best_params, probability=True, class_weight="balanced", random_state=42)

    elif model == "DecisionTree":
        if "class_weight" in best_params:
            best_params.pop("class_weight")
        best_clf = DecisionTreeClassifier(**best_params, class_weight="balanced", random_state=42)

    elif model == "AdaBoost":
        best_clf = AdaBoostClassifier(**best_params, random_state=42)

    elif model == "GradientBoosting":
        best_clf = GradientBoostingClassifier(**best_params, random_state=42)

    elif model == "K-Nearest Neighbors":
        best_clf = KNeighborsClassifier(**best_params)

    elif model == "MLP":
        best_clf = MLPClassifier(**best_params, random_state=42)

    else:
        raise ValueError(f"Unknown model: {model}")

    # ---- Final fit with sample_weight (skip if already fitted, e.g., calibrated RF) ----
    if not already_fitted:
        try:
            if data_leak:
                X_all = np.vstack([X_train_scaled, X_val_scaled, X_test_scaled])
                y_all = np.concatenate([y_train, y_val, y_test])
                sw_all = np.concatenate([sw_train, sw_val, sw_test]).astype(float)
                best_clf.fit(X_all, y_all, sample_weight=sw_all)
            else:
                best_clf.fit(X_train_scaled, y_train, sample_weight=sw_train)
        except TypeError:
            # Some estimators (e.g., KNeighborsClassifier) do not accept sample_weight in fit()
            if data_leak:
                X_all = np.vstack([X_train_scaled, X_val_scaled, X_test_scaled])
                y_all = np.concatenate([y_train, y_val, y_test])
                best_clf.fit(X_all, y_all)
            else:
                best_clf.fit(X_train_scaled, y_train)

    # ---------- Prepare feature metadata ----------
    feature_meta = {
        "selected_features": selected_features,
        "feature_source": model_name
    }

    if train_only:
        logging.info(f"[TRAIN_ONLY] Model trained, skipping evaluation/return.")
        return best_clf, scaler, None, feature_meta, {}

    # ---------- Evaluate & Save per-split metrics ----------
    def _eval_split(Xs, ys):
        yhat = best_clf.predict(Xs)
        # Get scores for ROC/PR curves
        y_score = None
        if hasattr(best_clf, "predict_proba"):
            y_score = best_clf.predict_proba(Xs)[:, 1]
        elif hasattr(best_clf, "decision_function"):
            y_score = best_clf.decision_function(Xs)
        
        # Use unified metrics calculation
        out = calculate_extended_metrics(ys, yhat, y_score, zero_division=0)
        # Preserve internal keys for downstream threshold search
        out["_y_true"] = ys
        out["_y_pred"] = yhat
        out["_proba"] = y_score
        return out

    m_train = _eval_split(X_train_scaled, y_train)
    m_val   = _eval_split(X_val_scaled,   y_val)
    m_test  = _eval_split(X_test_scaled,  y_test)
    logging.info(f"{model} (Optuna) metrics: "
                 f"train acc={m_train['accuracy']:.3f}, val acc={m_val['accuracy']:.3f}, test acc={m_test['accuracy']:.3f}")
    # save log of Test classification report
    y_pred_test = best_clf.predict(X_test_scaled)
    logging.info("Test classification report:\n" + classification_report(y_test, y_pred_test))

    # ---------- Post-process artifacts ----------
    results = {
        "train": {k:v for k,v in m_train.items() if not k.startswith("_")},
        "val":   {k:v for k,v in m_val.items()   if not k.startswith("_")},
        "test":  {k:v for k,v in m_test.items()  if not k.startswith("_")},
    }

    # ---------- Save artifacts (RF only: organized directories) ----------

    # --- robust model_name inference and unified save ---
    try:
        _model_name = model if isinstance(model, str) else getattr(model, "__name__", "unknown")
    except Exception:
        _model_name = "unknown"

    # Handle accidental object in `mode` (guard against estimators passed by mistake)
    if not isinstance(mode, str):
        logging.warning("[common_train] `mode` is not a string. Forcing to 'default'.")
        mode = "default"
    else:
        mode = mode.strip()
        if not mode:
            logging.info("[common_train] empty `mode` string detected, proceeding without suffix.")

    if isinstance(_model_name, dict):
        logging.warning(f"[common_train] model_name was dict: {_model_name}")
        _model_name = _model_name.get("name", "unknown")

    save_artifacts(
        model_name=str(_model_name),
        suffix=suffix,
        best_clf=best_clf,
        scaler=scaler,
        selected_features=selected_features,
        feature_meta=feature_meta,
    )

    # ---------- Threshold optimization on validation (maximize F2) ----------
    # Initialize threshold in case model lacks proba/decision_function.
    best_threshold = None
    if m_val["_proba"] is not None:

        # Use unified threshold optimization (F0.5 emphasizes precision)
        best_threshold, best_f05 = find_optimal_threshold(y_val, m_val["_proba"], beta=0.5)
        logging.info(f"Optimal threshold for F0.5 (β=0.5): {best_threshold:.3f} (F0.5={best_f05:.3f})")
        
        def _apply_thr(proba, y_true):
            yhat = apply_threshold(proba, best_threshold)
            metrics = calculate_extended_metrics(y_true, yhat, proba, zero_division=0, include_roc=False, include_pr=False)
            # Add F0.5 score (Precision-focused)
            metrics["f05"] = float(fbeta_score(y_true, yhat, beta=0.5, zero_division=0))
            return metrics

        thr_val  = _apply_thr(m_val["_proba"],  y_val)
        thr_test = _apply_thr(m_test["_proba"], y_test) if m_test["_proba"] is not None else None

        logging.info("Validation (F0.5-opt threshold) metrics: " + json.dumps(thr_val))
        if thr_test:
            logging.info("Test (F0.5-opt threshold from Val) metrics: " + json.dumps(thr_test))

        # NOTE: Actual threshold saving is unified in savers.save_artifacts()
        # via train_pipeline finally-block. We just return the value here.

    else:
        logging.warning("Threshold optimization skipped: model does not support probability estimation.")

    # ---------- Return all artifacts ----------
    return best_clf, scaler, best_threshold, feature_meta, results
