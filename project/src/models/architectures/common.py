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

def common_train(
    X_train, X_val, X_test, y_train, y_val, y_test,
    selected_features,  
    model: str, model_type: str,
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

    out_dir = f"{MODEL_PKL_PATH}/{model_type}"
    os.makedirs(out_dir, exist_ok=True)

    if eval_only:
        # ====== eval_onlyモード ======
        logging.info("[EVAL_ONLY] Loading pre-trained model and scaler...")
        with open(f"{out_dir}/{model}{suffix}.pkl", "rb") as f:
            best_clf = pickle.load(f)
        with open(f"{out_dir}/selected_features_train_{model}{suffix}.pkl", "rb") as f:
            selected_features = pickle.load(f)
        with open(f"{out_dir}/scaler_{model}{suffix}.pkl", "rb") as f:
            scaler = pickle.load(f)

        # スケーリング
        X_val_scaled  = scaler.transform(X_val[selected_features])
        X_test_scaled = scaler.transform(X_test[selected_features])

        # 評価関数の再利用
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
    
        # suffix に mode を含める
        eval_suffix = suffix + f"_{model_type}_evalonly"
        pd.DataFrame(rows).to_csv(
            f"{MODEL_PKL_PATH}/{model_type}/metrics_{model}{eval_suffix}.csv",
            index=False
        )
        logging.info(f"[EVAL_ONLY] Saved metrics CSV -> metrics_{model}{eval_suffix}.csv")
    
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

        roc_auc = "roc_auc" #make_scorer(roc_auc_score, needs_proba=True)

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
                score_arr = cross_val_score(clf, X_all_scaled, y_all, cv=cv, scoring=roc_auc, n_jobs=1)
#            else:
#                scaler_local = StandardScaler()
#                X_train_scaled = scaler_local.fit_transform(X_train[selected_features])
            else:
                # 目的関数内でも pretrain 済みスケーラを使用
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
                    score_arr = cross_val_score(
                        clf, X_train_scaled, y_train, 
                        cv=cv, 
                        scoring=roc_auc,
                        n_jobs=1,
                        error_score='raise'  
                    )
                    print("cross_val_score:", score_arr)
                    if np.any(np.isnan(score_arr)):
                        print("Score nan detected: ", score_arr)
                        return 0.0
                    score = np.nanmean(score_arr)
                except Exception as e:
                    print(f"[cross_val_score error] {e}")
                    import traceback
                    traceback.print_exc()
                    return 0.0

        except Exception as e:
            logging.warning(f"Scoring failed: {e}")
            return 0.0
 
        return score

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)
    )
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1)

    best_params = study.best_params

    logging.info(f"Best hyperparameters: {best_params}")
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

    if data_leak:
        best_clf.fit(
            np.vstack([X_train_scaled, X_val_scaled, X_test_scaled]),
            np.concatenate([y_train, y_val, y_test])
        )
    else:
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

    if train_only:
        logging.info(f"[TRAIN_ONLY] Saved model, features, scaler -> {suffix}. Skipping evaluation.")
        return

    # Save metadata (feature source info)
    feature_meta = {
        "selected_features": selected_features,
        "feature_source": model_type  # e.g., "common"
    }
    with open(f"{MODEL_PKL_PATH}/{model_type}/feature_meta_{model}{suffix}.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

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

    # ---------- Save Confusion Matrices (default decision) ----------
    out_dir = f"{MODEL_PKL_PATH}/{model_type}"
    os.makedirs(out_dir, exist_ok=True)

    def _save_cm(y_true, y_pred, split, tag_suffix):
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        # CSV (2x2)
        pd.DataFrame(cm, index=["true_0","true_1"], columns=["pred_0","pred_1"]).to_csv(
            f"{out_dir}/cm_{split}_{model}{tag_suffix}.csv"
        )
        # PNG
        import matplotlib.pyplot as plt
        plt.figure(figsize=(3.6,3.2))
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"CM ({split})")
        plt.xlabel("Predicted"); plt.ylabel("True")
        for (i,j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/cm_{split}_{model}{tag_suffix}.png", dpi=160)
        plt.close()

    _save_cm(m_train["_y_true"], m_train["_y_pred"], "train", suffix)
    _save_cm(m_val["_y_true"],   m_val["_y_pred"],   "val",   suffix)
    _save_cm(m_test["_y_true"],  m_test["_y_pred"],  "test",  suffix)

    # ---------- Save ROC curves (image + raw data) ----------
    def _save_roc(y_true, scores, split, tag_suffix):
        if scores is None:
            return
        fpr, tpr, thr = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        # data
        pd.DataFrame({"fpr":fpr, "tpr":tpr, "threshold":thr}).to_csv(
            f"{out_dir}/roc_{split}_{model}{tag_suffix}.csv", index=False
        )
        # plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4.0,3.6))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0,1],[0,1], linestyle="--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC ({split})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/roc_{split}_{model}{tag_suffix}.png", dpi=160)
        plt.close()

    _save_roc(m_train["_y_true"], m_train.get("_proba"), "train", suffix)
    _save_roc(m_val["_y_true"],   m_val.get("_proba"),   "val",   suffix)
    _save_roc(m_test["_y_true"],  m_test.get("_proba"),  "test",  suffix)

    # ---------- Save PR curves (image + raw data) ----------
    def _save_pr(y_true, scores, split, tag_suffix):
        if scores is None:
            return
        # precision_recall_curve returns precision, recall, thresholds (len(thr)=len(prec)-1)
        prec, rec, thr = precision_recall_curve(y_true, scores)
        import numpy as np, pandas as pd
        thr_aligned = np.concatenate([thr, [np.nan]])
        pd.DataFrame({"precision":prec, "recall":rec, "threshold":thr_aligned}).to_csv(
            f"{out_dir}/pr_{split}_{model}{tag_suffix}.csv", index=False
        )
        # plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4.0,3.6))
        plt.plot(rec, prec)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR ({split})")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/pr_{split}_{model}{tag_suffix}.png", dpi=160)
        plt.close()

    _save_pr(m_train["_y_true"], m_train.get("_proba"), "train", suffix)
    _save_pr(m_val["_y_true"],   m_val.get("_proba"),   "val",   suffix)
    _save_pr(m_test["_y_true"],  m_test.get("_proba"),  "test",  suffix)

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
        with open(f"{MODEL_PKL_PATH}/{model_type}/threshold_{model}{suffix}.json", "w") as f:
            json.dump(threshold_meta, f, indent=2)

    else:
        logging.warning("Threshold optimization skipped: model does not support probability estimation.")


    # ---------- Save metrics CSV ----------
    rows = []
    rows.append({"split":"train", **{k:v for k,v in m_train.items() if not k.startswith("_")}})
    rows.append({"split":"val",   **{k:v for k,v in m_val.items()   if not k.startswith("_")}})
    rows.append({"split":"test",  **{k:v for k,v in m_test.items()  if not k.startswith("_")}})
    os.makedirs(f"{MODEL_PKL_PATH}/{model_type}", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"{MODEL_PKL_PATH}/{model_type}/metrics_{model}{suffix}.csv", index=False)
    logging.info(f"Saved metrics CSV -> {MODEL_PKL_PATH}/{model_type}/metrics_{model}{suffix}.csv")
