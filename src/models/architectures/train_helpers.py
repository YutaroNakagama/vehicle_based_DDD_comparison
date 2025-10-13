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

from src.utils.io.savers import save_artifacts


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


