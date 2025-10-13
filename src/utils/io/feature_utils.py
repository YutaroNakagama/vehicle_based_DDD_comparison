# src/utils/io/feature_utils.py
"""Utility functions for feature name normalization and consistency checks."""

import logging
import pandas as pd
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

    missing = [f for f in selected_features if f not in X_test.columns]
    if missing:
        logging.warning(f"[EVAL] Missing normalized features: {missing}")

    return X_test[selected_features], selected_features

