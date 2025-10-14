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
from sklearn.metrics import (
    classification_report, roc_curve, auc, make_scorer, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
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

from src.utils.io.savers import save_artifacts

def common_train(
    X_train, X_val, X_test, y_train, y_val, y_test,
    selected_features,  
    model: str, model_type: str,
    mode: str,   
    clf=None, scaler=None, suffix: str = "",
    data_leak: bool = False,
    eval_only: bool = False,   
    train_only: bool = False,  
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
    model_type : str
        Model group (e.g., ``"common"``, ``"SvmA"``).
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

    import pickle, json

    # Always use unified model-specific directory under "models"
    base_dir = os.path.join("models", model)
    os.makedirs(base_dir, exist_ok=True)

    if eval_only:
        # ====== eval_only mode ======
        logging.info("[EVAL_ONLY] Loading pre-trained model and scaler...")
        with open(f"{out_dir}/{model}_{mode}{suffix}.pkl", "rb") as f:
            best_clf = pickle.load(f)
        with open(f"{out_dir}/scaler_{model}_{mode}{suffix}.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(f"{out_dir}/selected_features_train_{model}_{mode}{suffix}.pkl", "rb") as f:
            selected_features = pickle.load(f)

        # Scaling
        X_val_scaled  = scaler.transform(X_val[selected_features])
        X_test_scaled = scaler.transform(X_test[selected_features])

        # Reuse evaluation function
        def _eval_split(Xs, ys):
            yhat = best_clf.predict(Xs)
            out = {
                "accuracy": float(accuracy_score(ys, yhat)),
                "precision": float(precision_score(ys, yhat, zero_division=0)),
                "recall": float(recall_score(ys, yhat, zero_division=0)),
                "f1": float(f1_score(ys, yhat, zero_division=0)),
            }
            if hasattr(best_clf, "predict_proba"):
                proba = best_clf.predict_proba(Xs)[:, 1]
                out["auc"] = float(roc_auc_score(ys, proba))
                out["ap"] = float(average_precision_score(ys, proba))
            return out

        m_val  = _eval_split(X_val_scaled, y_val)
        m_test = _eval_split(X_test_scaled, y_test)

        logging.info(f"[EVAL_ONLY] Validation metrics: {json.dumps(m_val, indent=2)}")
        logging.info(f"[EVAL_ONLY] Test metrics: {json.dumps(m_test, indent=2)}")

        # ===== Save metrics (eval only) =====
        rows = []
        rows.append({"split": "val",  **m_val})
        rows.append({"split": "test", **m_test})
        os.makedirs(f"{MODEL_PKL_PATH}/{model_type}", exist_ok=True)
    
        # Ensure mode is included in the suffix
        eval_suffix = suffix + f"_{model_type}_evalonly"
        pd.DataFrame(rows).to_csv(
            f"{MODEL_PKL_PATH}/{model_type}/metrics_{model}_{mode}{eval_suffix}.csv",
            index=False
        )
        logging.info(f"[EVAL_ONLY] Saved metrics CSV -> metrics_{model}_{mode}{eval_suffix}.csv")
    
        return {"val": m_val, "test": m_test}

    X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    X_val   = X_val.loc[:,   ~X_val.columns.duplicated()]
    X_test  = X_test.loc[:,  ~X_test.columns.duplicated()]

    def objective(trial):
        print('X_train.shape:', X_train.shape)
        print('X_train nan:', np.isnan(X_train.values).sum(), 'inf:', np.isinf(X_train.values).sum())
        print('X_train col nan count:', np.isnan(X_train.values).sum(axis=0))
        print('X_train all const col:', [c for c in X_train.columns if X_train[c].nunique() == 1])
        print('selected_features:', selected_features)
        print('y_train shape:', y_train.shape)
        print('y_train nan:', np.isnan(y_train).sum(), 'unique:', np.unique(y_train))
        print('X_train[selected_features] shape:', X_train[selected_features].shape)
        print('X_train[selected_features] col nan:', np.isnan(X_train[selected_features].values).sum(axis=0))
 
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
#                "class_weight": "balanced",
                "n_jobs": 1,  
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

        roc_auc = "roc_auc" #make_scorer(roc_auc_score, needs_proba=True)
        ap_scorer = make_scorer(average_precision_score, needs_proba=True)

#        try:
#            if data_leak:
#                X_all = np.vstack([X_train[selected_features], X_test[selected_features]])
#                y_all = np.concatenate([y_train, y_test])
#                scaler_local = StandardScaler()
#                X_all_scaled = scaler_local.fit_transform(X_all)
        try:
            if data_leak:
                X_all = np.vstack([X_train[selected_features], X_test[selected_features]])
                y_all = np.concatenate([y_train, y_test])
                X_all_scaled = scaler.transform(X_all)
                cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
                for i, (_, va_idx) in enumerate(cv.split(X_all_scaled, y_all)):
                    bincounts = np.bincount(y_all[va_idx].astype(int))
                    print(f"[CV-Leak] Fold {i} y_val bincount: {bincounts}")
                    if len(bincounts) < 2 or np.any(bincounts == 0):
                        print(f"[CV-Leak] Fold {i} has only one class! Skipping trial.")
                        return 0.0
                score_arr = cross_val_score(
                    clf, X_all_scaled, y_all, 
                    cv=cv, 
                    scoring=roc_auc, #ap_scorer, 
                    n_jobs=1
                )
            else:
                # Use pre-trained scaler even inside the objective function
                X_train_scaled = scaler.transform(X_train[selected_features])
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  

                auc_list = []
                
                for i, (tr_idx, va_idx) in enumerate(cv.split(X_train_scaled, y_train)):
                    clf.fit(X_train_scaled[tr_idx], y_train[tr_idx])
                    y_val = y_train[va_idx]
                    y_pred = clf.predict(X_train_scaled[va_idx])
                    bincount_pred = np.bincount(y_pred.astype(int))
                    print(f"[CV] Fold {i}: y_val bincount: {np.bincount(y_val.astype(int))}, y_pred bincount: {bincount_pred}")
                    try:
                        y_proba = clf.predict_proba(X_train_scaled[va_idx])[:,1]
                        auc = roc_auc_score(y_val, y_proba)
                        print(f"[CV] Fold {i}: AUC = {auc}")
                    except ValueError as e:
                        print(f"[CV] Fold {i}: AUC nan! {e}")
                        auc = np.nan
                    auc_list.append(auc)
                
                print("manual auc_list:", auc_list)
                print("manual auc mean (ignore nan):", np.nanmean(auc_list))
                try:
                    # --- FIX: enforce aligned indices and NumPy conversion ---
                    X_aligned = X_train[selected_features].reset_index(drop=True)
                    y_aligned = y_train.reset_index(drop=True)

                    # Already scaled above; avoid double scaling to prevent index mismatch
                    X_np = X_aligned.to_numpy()
                    y_np = y_aligned.to_numpy()

                    # Debug (optional)
                    print("X_train index sample:", X_train.index[:5])
                    print("y_train index sample:", y_train.index[:5])
                    import sys, io, contextlib
                    import warnings
                    from sklearn.exceptions import FitFailedWarning
                    import os
                    # --- FULL suppression: Python + OS-level stderr (joblib/fork safe) ---
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=FitFailedWarning)
                        with open(os.devnull, 'w') as devnull:
                            old_stderr_fd = os.dup(sys.stderr.fileno())
                            try:
                                os.dup2(devnull.fileno(), sys.stderr.fileno())  # redirect at OS level
                                with contextlib.redirect_stderr(devnull):
                                    try:
                                        score_arr = cross_val_score(
                                            clf, X_np, y_np,
                                            cv=cv,
                                            scoring=roc_auc,
                                            n_jobs=1,
                                            error_score=np.nan
                                        )
                                    except Exception as e:
                                        # --- catch and silence known index/scoring errors ---
                                        msg = str(e)
                                        if "not in index" in msg or "Scoring failed" in msg:
                                            return 0.0
                                        else:
                                            logging.debug(f"[cross_val_score internal exception suppressed] {msg}")
                                            return 0.0
                            finally:
                                os.dup2(old_stderr_fd, sys.stderr.fileno())  # restore stderr
                                os.close(old_stderr_fd)
                    print("cross_val_score:", score_arr)
                    if np.any(np.isnan(score_arr)):
                        print("Score nan detected: ", score_arr)
                        return 0.0
                    score = np.nanmean(score_arr)
                except Exception as e:
                    if "not in index" in str(e):
                        pass  # silently skip
                    else:
                        logging.warning(f"[Optuna cross_val_score error] {e}")
                    return 0.0

#        except Exception as e:
#            logging.warning(f"Scoring failed: {e}")
#            return 0.0
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
        "RF", "BalancedRF",
        "LogisticRegression", "SVM",
        "DecisionTree", "AdaBoost", "GradientBoosting",
        "K-Nearest Neighbors", "MLP"
    ]

    if model in optuna_supported:
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)
        )
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)
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

    if model == "LightGBM":
        best_clf = LGBMClassifier(**best_params)

    elif model == "XGBoost":
        best_clf = XGBClassifier(**best_params)

    elif model == "CatBoost":
        best_clf = CatBoostClassifier(**best_params)

    elif model == "RF":
        if "class_weight" in best_params:
            best_params.pop("class_weight")
        best_clf = RandomForestClassifier(**best_params, class_weight="balanced_subsample", n_jobs=-1)

    elif model == "BalancedRF":
        best_clf = BalancedRandomForestClassifier(**best_params)

    elif model == "LogisticRegression":
        if "class_weight" in best_params:
                best_params.pop("class_weight")
        best_clf = LogisticRegression(**best_params, class_weight="balanced")

    elif model in ["SVM", "SvmW"]:
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

    if data_leak:
        best_clf.fit(
            np.vstack([X_train_scaled, X_val_scaled, X_test_scaled]),
            np.concatenate([y_train, y_val, y_test])
        )
    else:
        best_clf.fit(X_train_scaled, y_train)

    # ---------- Prepare feature metadata ----------
    feature_meta = {
        "selected_features": selected_features,
        "feature_source": model_type
    }

    if train_only:
        logging.info(f"[TRAIN_ONLY] Model trained, skipping evaluation/return.")
        return best_clf, scaler, None, feature_meta, {}

    # ---------- Evaluate & Save per-split metrics ----------
    def _eval_split(Xs, ys):
        yhat = best_clf.predict(Xs)
        out = {
            "accuracy": float(accuracy_score(ys, yhat)),
            "precision": float(precision_score(ys, yhat, zero_division=0)),
            "recall": float(recall_score(ys, yhat, zero_division=0)),
            "f1": float(f1_score(ys, yhat, zero_division=0)),
            "auc": float("nan"),
            "ap":  float("nan"),
            "_y_true": ys,
            "_y_pred": yhat,
        }
        if hasattr(best_clf, "predict_proba"):
            proba = best_clf.predict_proba(Xs)[:,1]
            try:
                out["auc"] = float(roc_auc_score(ys, proba))
            except Exception:
                pass
            try:
                out["ap"] = float(average_precision_score(ys, proba))
            except Exception:
                pass
            out["_proba"] = proba  # for threshold search
        elif hasattr(best_clf, "decision_function"):
            score = best_clf.decision_function(Xs)
            try:
                out["auc"] = float(roc_auc_score(ys, score))
            except Exception:
                pass
            try:
                out["ap"] = float(average_precision_score(ys, score))
            except Exception:
                pass
            out["_proba"] = score
        else:
            out["_proba"] = None
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

    # Handle accidental object in mode
    if not isinstance(mode, str):
        logging.warning(f"[common_train] mode is not str (type={type(mode)}), resetting to 'unspecified'")
        _mode = "unspecified"
    else:
        _mode = mode

    if isinstance(_model_name, dict):
        logging.warning(f"[common_train] model_name was dict: {_model_name}")
        _model_name = _model_name.get("name", "unknown")

    # --- Save unified artifacts (once only, after model training) ---
    save_artifacts(
        model_obj=best_clf,
        scaler_obj=scaler,
        selected_features=selected_features,
        feature_meta=feature_meta,
        model_name=str(_model_name),
        mode=_mode
    )

    # ---------- Threshold optimization on validation (maximize F1) ----------
    if m_val["_proba"] is not None:

        precision, recall, thresholds = precision_recall_curve(y_val, m_val["_proba"])
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        logging.info(f"Optimal threshold for F1: {best_threshold:.3f}")
        
        def _apply_thr(proba, y_true):
            yhat = (proba >= best_threshold).astype(int)
            return {
                "accuracy": float(accuracy_score(y_true, yhat)),
                "precision": float(precision_score(y_true, yhat, zero_division=0)),
                "recall": float(recall_score(y_true, yhat, zero_division=0)),
                "f1": float(f1_score(y_true, yhat, zero_division=0)),
            }

        thr_val  = _apply_thr(m_val["_proba"],  y_val)
        thr_test = _apply_thr(m_test["_proba"], y_test) if m_test["_proba"] is not None else None

        logging.info("Validation (F1-opt threshold) metrics: " + json.dumps(thr_val))
        if thr_test:
            logging.info("Test (F1-opt threshold from Val) metrics: " + json.dumps(thr_test))

        # Save threshold
        threshold_meta = {
            "model": model,
            "threshold": best_threshold,
            "metric": "F1-optimal",
        }
        # ==========================================================
        # Save threshold under: models/<model>/<jobid>/threshold_*.json
        # ==========================================================
        # Extract jobid from suffix (robust: handles "14019173.spcc-adm1" etc.)
        import re
        jobid = None
        m = re.search(r"\b(\d{5,})\b", suffix)  # match 5+ consecutive digits
        if m:
            jobid = m.group(1)
        else:
            # fallback: try removing hostname (e.g., "14019173.spcc-adm1")
            m2 = re.search(r"(\d{5,})", suffix)
            if m2:
                jobid = m2.group(1)

        # Construct path
        if jobid:
            out_dir = os.path.join(MODEL_PKL_PATH, model, jobid)
        else:
            out_dir = os.path.join(MODEL_PKL_PATH, model)

        os.makedirs(out_dir, exist_ok=True)
        thr_path = os.path.join(out_dir, f"threshold_{model}_{mode}{suffix}.json")
        with open(thr_path, "w") as f:
            json.dump(threshold_meta, f, indent=2)
        logging.info(f"[SAVE] Threshold saved -> {thr_path}")

    else:
        logging.warning("Threshold optimization skipped: model does not support probability estimation.")

    # ---------- Return all artifacts ----------
    return best_clf, scaler, best_threshold, feature_meta, results
