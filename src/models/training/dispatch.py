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

from src.models.training.model_factory import get_classifier
from src.models.architectures.SvmA import SvmA_train, compute_feature_indices
from src.models.architectures.lstm import lstm_train
from src.models.training.pipeline import common_train

from src.utils.io.savers import save_artifacts


__all__ = ["train_model", "save_artifacts"]


def train_model(
    model_name: str,
    X_train_fs: pd.DataFrame,
    X_val_fs: pd.DataFrame,
    X_test_fs: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    selected_features: List[str],
    scaler: Optional[StandardScaler],
    suffix: str,
    mode: str = "pooled",
    use_oversampling: bool = False,
    oversample_method: str = "smote",
    target_ratio: float = 0.33,
    seed: int = 42,
) -> Tuple[Optional[object], Optional[StandardScaler], Optional[float], Dict, Dict]:
    """Dispatch to the appropriate training routine and return artifacts.

    Unified naming: removed obsolete `model_type` parameter. All downstream
    logic uses `model_name` for classifier resolution and artifact metadata.
    """

    if model_name == "Lstm":
        model_obj, scaler_obj, selected_feats, results = lstm_train(
            X_train_fs, y_train, model_name,
            X_val=X_val_fs, y_val=y_val,
            X_test=X_test_fs, y_test=y_test,
        )
        logging.info("LSTM training completed.")
        return model_obj, scaler_obj, None, {"selected_features": selected_feats}, results

    elif model_name == "SvmA":
        # Compute ANFIS feature indices from training data
        # indices_df contains Fisher, Correlation, T-test, and MI scores per feature
        indices_df = compute_feature_indices(X_train_fs, y_train)
        logging.info(f"[SvmA] Computed feature indices with shape {indices_df.shape}")
        
        model_obj, scaler_obj, selected_feats, results = SvmA_train(
            X_train_fs, X_val_fs, y_train, y_val, indices_df, model_name,
            X_test=X_test_fs, y_test=y_test,
        )
        logging.info("SvmA training completed.")
        return model_obj, scaler_obj, None, {"selected_features": selected_feats}, results

    else:
        # Tree-based / linear models (RF, XGBoost, LightGBM, etc.)
        clf = get_classifier(model_name, seed=seed)
        best_clf, scaler, best_threshold, feature_meta, results = common_train(
            X_train_fs, X_val_fs, X_test_fs,
            y_train, y_val, y_test,
            selected_features,
            model_name, model_name, mode,
            clf=clf,
            scaler=scaler,
            suffix=suffix,
            data_leak=False,
            use_oversampling=use_oversampling,
            oversample_method=oversample_method,
            target_ratio=target_ratio,
            seed=seed,
        )
        return best_clf, scaler, best_threshold, feature_meta, results
