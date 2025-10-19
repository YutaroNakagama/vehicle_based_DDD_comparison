# src/utils/io/feature_utils.py
"""Utility functions for feature name normalization and consistency checks."""

import logging
import pandas as pd
import numpy as np
from typing import List, Tuple


def normalize_feature_names(cols: List[str]) -> List[str]:
    """Normalize feature names to consistent format (for EEG & numeric features).

    This function cleans and standardizes feature names by removing whitespace,
    replacing symbols, and unifying EEG band naming conventions.

    Parameters
    ----------
    cols : list of str
        Original column names.

    Returns
    -------
    list of str
        Normalized column names.
    """
    normalized = []
    for c in cols:
        n = c.strip().replace(" ", "_").replace("(", "_").replace(")", "_").replace("-", "_")
        n = n.replace("__", "_")

        # Normalize EEG band notation
        n = n.replace("_30_100_Hz_", "(30-100_Hz)")
        n = n.replace("_13_30_Hz_", "(13-30_Hz)")
        n = n.replace("_4_8_Hz_", "(4-8_Hz)")
        n = n.replace("_05_4_Hz_", "(0.5-4_Hz)")
        n = n.replace("(30_100_Hz)", "(30-100_Hz)")
        n = n.replace("(13_30_Hz)", "(13-30_Hz)")
        n = n.replace("(4_8_Hz)", "(4-8_Hz)")
        n = n.replace("(05_4_Hz)", "(0.5-4_Hz)")

        # Ensure clean underscores
        n = n.replace("_(", "(").replace(")_", ")")
        normalized.append(n)

    return normalized


def normalize_and_align_features(X_test: pd.DataFrame, selected_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Normalize and align feature names with training pipeline convention.

    Parameters
    ----------
    X_test : pandas.DataFrame
        Test feature DataFrame.
    selected_features : list of str
        Features used during training.

    Returns
    -------
    tuple
        (X_test_aligned, selected_features_normalized)
    """
    X_test.columns = normalize_feature_names(X_test.columns)
    selected_features = normalize_feature_names(selected_features)

    # Remove duplicate columns and non-feature identifiers
    X_test = X_test.loc[:, ~X_test.columns.duplicated()]
    X_test = X_test.drop(columns=["subject_id"], errors="ignore")


    # --- Detect missing or extra features ---
    missing = [f for f in selected_features if f not in X_test.columns]
    extra   = [f for f in X_test.columns if f not in selected_features]

    # --- Add missing columns (filled with 0.0) ---
    if missing:
        logging.warning(f"[EVAL] Missing features ({len(missing)}): {missing[:10]} ... (total {len(missing)})")
        for f in missing:
            X_test[f] = 0.0

    # --- Drop extra columns ---
    if extra:
        logging.info(f"[EVAL] Dropping extra features ({len(extra)}): {extra[:10]} ... (total {len(extra)})")
        X_test = X_test.drop(columns=extra)

    # --- Align columns safely (skip missing EEG columns etc.) ---
    existing_features = [f for f in selected_features if f in X_test.columns]
    missing_features = [f for f in selected_features if f not in X_test.columns]

    if missing_features:
        logging.warning(f"[EVAL] Dropping {len(missing_features)} missing features from selected_features (e.g., EEG cols).")

    # --- Align columns safely (skip missing EEG columns etc.) ---
    existing_features = [f for f in selected_features if f in X_test.columns]
    missing_features = [f for f in selected_features if f not in X_test.columns]

    if missing_features:
        logging.warning(
            f"[EVAL] Dropping {len(missing_features)} missing features from selected_features (e.g., EEG cols)."
        )

    # --- Reorder only existing features ---
    X_test = X_test[existing_features]

    # --- Replace any NaN or Inf values with 0 ---
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X_test, existing_features

def align_and_normalize_features(X_test: pd.DataFrame, selected_features: List[str]):
    """Alias wrapper for normalize_and_align_features (directly call updated logic)."""
    return normalize_and_align_features(X_test, selected_features)

