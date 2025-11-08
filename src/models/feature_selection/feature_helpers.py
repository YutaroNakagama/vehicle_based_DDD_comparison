"""Feature selection and scaling helpers for model_pipeline.

This module isolates feature-selection logic (RF, MI, ANOVA) and
scaler fitting from the main training pipeline, to simplify
`model_pipeline.py` and allow reuse across evaluation scripts.
"""

import logging
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

from src.config import TOP_K_FEATURES
from src.utils.io.split import _check_nonfinite
from src.models.feature_selection.rf_importance import select_top_features_by_importance

__all__ = ["select_features_and_scale"]

def select_features_and_scale(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    feature_selection_method: str = "rf",
    top_k: int = TOP_K_FEATURES,
    data_leak: bool = False,
) -> Tuple[List[str], StandardScaler, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select features and fit a scaler.

    Parameters
    ----------
    X_train, X_val, X_test : pandas.DataFrame
        Input features for train, validation, and test splits.
    y_train : pandas.Series
        Training labels.
    feature_selection_method : {"rf", "mi", "anova"}, default="rf"
        Method to use for feature selection.
    top_k : int, default=TOP_K_FEATURES
        Number of top features to retain.
    data_leak : bool, default=False
        If True, scaler is fit on train+val (legacy mode).

    Returns
    -------
    selected_features : list of str
        Names of selected feature columns.
    scaler : StandardScaler
        Fitted scaler instance.
    X_train_fs, X_val_fs, X_test_fs : pandas.DataFrame
        Preprocessed feature matrices (not yet scaled).
    """

    # --- (NEW) Drop EEG-related columns as the very first step to avoid huge-value warnings ---
    eeg_keywords = ["Channel_", "EEG", "Theta", "Alpha", "Beta", "Gamma", "Delta"]

    def _drop_eeg(df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            return None
        drop_cols = [c for c in df.columns if any(k in c for k in eeg_keywords)]
        if drop_cols:
            logging.info(f"[FS] Dropping {len(drop_cols)} EEG-related columns "
                         f"(e.g., {drop_cols[:5]}) in feature selection stage")
            df = df.drop(columns=drop_cols, errors="ignore")
        return df

    X_train = _drop_eeg(X_train)
    X_val   = _drop_eeg(X_val)
    X_test  = _drop_eeg(X_test)

    # ---------------------------------------------------------------------

    # Drop subject_id if exists and sanitize values
    X_train = _check_nonfinite(X_train.drop(columns=["subject_id"], errors="ignore"), "X_train")
    X_val   = _check_nonfinite(X_val.drop(columns=["subject_id"], errors="ignore"), "X_val")
    X_test  = _check_nonfinite(X_test.drop(columns=["subject_id"], errors="ignore"), "X_test")

    # --- Feature selection ---
    if feature_selection_method == "mi":
        selector = SelectKBest(score_func=mutual_info_classif, k=top_k)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()].tolist()

    elif feature_selection_method == "anova":
        selector = SelectKBest(score_func=f_classif, k=top_k)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()].tolist()

    elif feature_selection_method == "rf":
        selected_features = select_top_features_by_importance(X_train, y_train, top_k=top_k)
        # Double-check after selection
        X_train = _check_nonfinite(X_train[selected_features], "X_train(post-rf)")
        X_val   = _check_nonfinite(X_val[selected_features], "X_val(post-rf)")
        X_test  = _check_nonfinite(X_test[selected_features], "X_test(post-rf)")

    else:
        raise ValueError(f"Unknown feature_selection_method: {feature_selection_method}")

    # --- Scaling ---
    scaler = StandardScaler()
    if data_leak:
        scaler.fit(
            pd.concat([
                X_train[selected_features], 
                X_val[selected_features]], 
                axis=0
            )
        )
    else:
        scaler.fit(X_train[selected_features])

    logging.info("Selected %d features using '%s'.", len(selected_features), feature_selection_method)
    return selected_features, scaler, X_train, X_val, X_test
