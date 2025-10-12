"""Model training and artifact-saving helpers.

This module centralizes model-specific training dispatch
and artifact persistence used by `model_pipeline.py`.
"""

import os
import json
import pickle
import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler

from src.models.architectures.helpers import get_classifier
from src.models.architectures.SvmA import SvmA_train
from src.models.architectures.lstm import lstm_train
from src.models.architectures.common import common_train


__all__ = ["train_model", "save_artifacts"]


def train_model(
    model_name: str,
    model_type: str,
    X_train_fs: pd.DataFrame,
    X_val_fs: pd.DataFrame,
    X_test_fs: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    selected_features: List[str],
    scaler: Optional[StandardScaler],
    suffix: str,
) -> Tuple[Optional[object], Optional[StandardScaler], Optional[float], Dict, Dict]:
    """Dispatch to the appropriate training routine and return artifacts."""

    if model_name == "Lstm":
        lstm_train(X_train_fs, y_train, model_name)
        logging.info("LSTM training completed.")
        return None, None, None, {}, {}

    if model_name == "SvmA":
        SvmA_train(X_train_fs, X_val_fs, y_train, y_val, selected_features, model_name)
        logging.info("SvmA training completed.")
        return None, None, None, {}, {}

    # Tree-based / linear models
    clf = get_classifier(model_name)
    best_clf, scaler, best_threshold, feature_meta, results = common_train(
        X_train_fs, X_val_fs, X_test_fs,
        y_train, y_val, y_test,
        selected_features,
        model_name, model_type, clf,
        scaler=scaler,
        suffix=suffix,
        data_leak=False,
    )
    return best_clf, scaler, best_threshold, feature_meta, results


def save_artifacts(
    model_name: str,
    model_type: str,
    suffix: str,
    best_clf: Optional[object],
    scaler: Optional[StandardScaler],
    selected_features: Optional[List[str]],
    feature_meta: Optional[Dict],
    results: Optional[Dict],
    best_threshold: Optional[float],
) -> None:
    """Persist models, scalers, features, and training-time metrics to disk."""

    os.makedirs(f"models/{model_type}", exist_ok=True)
    os.makedirs(f"results/train/{model_name}", exist_ok=True)

    if best_clf is not None:
        with open(f"models/{model_type}/{model_name}{suffix}.pkl", "wb") as f:
            pickle.dump(best_clf, f)
    if scaler is not None:
        with open(f"models/{model_type}/scaler_{model_name}{suffix}.pkl", "wb") as f:
            pickle.dump(scaler, f)
    if selected_features is not None:
        with open(f"models/{model_type}/selected_features_{model_name}{suffix}.pkl", "wb") as f:
            pickle.dump(selected_features, f)

    if feature_meta:
        with open(f"models/{model_type}/feature_meta_{model_name}{suffix}.json", "w") as f:
            json.dump(feature_meta, f, indent=2)

    if best_threshold is not None:
        thr_meta = {"model": model_name, "threshold": float(best_threshold), "metric": "F1-optimal"}
        with open(f"results/train/{model_name}/threshold_{model_name}{suffix}.json", "w") as f:
            json.dump(thr_meta, f, indent=2)

    if results:
        rows = [{"phase": "training", "split": split, **metrics} for split, metrics in results.items()]
        df_results = pd.DataFrame(rows)
        df_results.to_csv(f"results/train/{model_name}/trainmetrics_{model_name}{suffix}.csv", index=False)
        with open(f"results/train/{model_name}/trainmetrics_{model_name}{suffix}.json", "w") as f:
            json.dump(rows, f, indent=2)

    logging.info("Artifacts saved under models/%s and results/train/%s", model_type, model_name)
