"""Feature preprocessing utilities for DDD pipelines.

This module provides shared preprocessing functions used by both
training and evaluation pipelines, ensuring consistent feature
transformation across the project.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional

# EEG-related column keywords (excluded in vehicle-based DDD)
EEG_KEYWORDS = ["Channel_", "EEG", "Theta", "Alpha", "Beta", "Gamma", "Delta"]


def drop_eeg_columns(df: pd.DataFrame, keywords: Optional[List[str]] = None) -> pd.DataFrame:
    """Remove EEG-related columns from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature DataFrame.
    keywords : list of str, optional
        List of substrings to match for EEG columns.
        Defaults to standard EEG keywords if not provided.

    Returns
    -------
    pd.DataFrame
        DataFrame with EEG columns removed.
    """
    if keywords is None:
        keywords = EEG_KEYWORDS
    
    drop_cols = [c for c in df.columns if any(k in c for k in keywords)]
    if drop_cols:
        logging.info(f"[PREPROCESS] Dropping {len(drop_cols)} EEG columns (e.g., {drop_cols[:5]})")
        df = df.drop(columns=drop_cols)
    return df


def clean_feature_dataframe(
    df: pd.DataFrame,
    drop_subject_id: bool = True,
    drop_eeg: bool = True,
    numeric_only: bool = True,
) -> pd.DataFrame:
    """Apply standard cleaning to a feature DataFrame.

    This function applies common preprocessing steps:
    - Remove duplicate columns
    - Optionally drop subject_id
    - Optionally drop EEG-related columns
    - Optionally keep only numeric columns

    Parameters
    ----------
    df : pd.DataFrame
        Input feature DataFrame.
    drop_subject_id : bool, default=True
        Whether to drop the subject_id column.
    drop_eeg : bool, default=True
        Whether to drop EEG-related columns.
    numeric_only : bool, default=True
        Whether to keep only numeric columns.

    Returns
    -------
    pd.DataFrame
        Cleaned feature DataFrame.
    """
    # Remove duplicated columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Drop subject_id if requested
    if drop_subject_id:
        df = df.drop(columns=["subject_id"], errors="ignore")
    
    # Drop EEG columns if requested
    if drop_eeg:
        df = drop_eeg_columns(df)
    
    # Keep only numeric columns if requested
    if numeric_only:
        df = df.select_dtypes(include=[np.number])
    
    return df


def align_feature_columns(
    df: pd.DataFrame,
    expected_columns: List[str],
    fill_missing: float = 0.0,
    drop_extra: bool = True,
) -> pd.DataFrame:
    """Align DataFrame columns to match expected features.

    This function ensures that the DataFrame has exactly the expected columns,
    adding missing ones (filled with a constant) and optionally dropping extra ones.

    Parameters
    ----------
    df : pd.DataFrame
        Input feature DataFrame.
    expected_columns : list of str
        List of expected column names.
    fill_missing : float, default=0.0
        Value to fill for missing columns.
    drop_extra : bool, default=True
        Whether to drop columns not in expected_columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with aligned columns in the order of expected_columns.
    """
    extra_cols = [c for c in df.columns if c not in expected_columns]
    missing_cols = [c for c in expected_columns if c not in df.columns]
    
    if extra_cols and drop_extra:
        logging.info(f"[PREPROCESS] Dropping {len(extra_cols)} extra columns (e.g., {extra_cols[:5]})")
        df = df.drop(columns=extra_cols, errors="ignore")
    
    if missing_cols:
        logging.warning(f"[PREPROCESS] {len(missing_cols)} missing columns filled with {fill_missing} (e.g., {missing_cols[:5]})")
        for c in missing_cols:
            df[c] = fill_missing
    
    return df.reindex(columns=expected_columns)


def prepare_evaluation_features(
    df: pd.DataFrame,
    scaler,
    selected_features: Optional[List[str]] = None,
    clip_range: tuple = (-1_000_000.0, 1_000_000.0),
) -> pd.DataFrame:
    """Prepare and transform features for evaluation.

    This function applies the complete preprocessing pipeline used in evaluation:
    - Clean DataFrame (drop duplicates, EEG, subject_id)
    - Align columns to match training features
    - Clip outliers
    - Apply scaler transformation

    Parameters
    ----------
    df : pd.DataFrame
        Input feature DataFrame.
    scaler : sklearn transformer
        Fitted scaler from training.
    selected_features : list of str, optional
        List of selected feature names from training.
        If None, uses scaler.feature_names_in_ if available.
    clip_range : tuple of (float, float), default=(-1e6, 1e6)
        Range for clipping outliers.

    Returns
    -------
    pd.DataFrame
        Transformed feature DataFrame ready for model prediction.
    """
    # Step 1: Clean the DataFrame
    df = clean_feature_dataframe(
        df, 
        drop_subject_id=True, 
        drop_eeg=True, 
        numeric_only=True
    )
    
    # Replace inf with NaN and fill with 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Step 2: Align columns to selected features or scaler features
    expected_cols = None
    if selected_features is not None and len(selected_features) > 0:
        expected_cols = selected_features
    elif hasattr(scaler, "feature_names_in_"):
        expected_cols = list(scaler.feature_names_in_)
    
    if expected_cols:
        df = align_feature_columns(df, expected_cols, fill_missing=0.0, drop_extra=True)
    
    # Step 3: Clip outliers
    df = df.clip(lower=clip_range[0], upper=clip_range[1], axis=1)
    
    # Step 4: Ensure exact alignment with scaler's expected features
    if hasattr(scaler, "feature_names_in_"):
        scaler_cols = list(scaler.feature_names_in_)
        df = align_feature_columns(df, scaler_cols, fill_missing=0.0, drop_extra=True)
    
    # Step 5: Transform using scaler
    transformed = scaler.transform(df)
    
    return pd.DataFrame(transformed, index=df.index)


def align_train_val_test_columns(*dataframes: pd.DataFrame) -> List[pd.DataFrame]:
    """Align columns across train/val/test DataFrames to common subset.

    Parameters
    ----------
    *dataframes : pd.DataFrame
        Variable number of DataFrames to align (typically X_train, X_val, X_test).

    Returns
    -------
    list of pd.DataFrame
        List of aligned DataFrames with only common columns.
    """
    if not dataframes:
        return []
    
    # Find common columns across all DataFrames
    common_cols = dataframes[0].columns
    for df in dataframes[1:]:
        if df is not None:
            common_cols = common_cols.intersection(df.columns)
    
    # Apply common columns to all DataFrames
    aligned = []
    for df in dataframes:
        if df is not None:
            aligned.append(df[common_cols])
        else:
            aligned.append(None)
    
    logging.info(f"[PREPROCESS] Aligned {len(dataframes)} DataFrames to {len(common_cols)} common columns")
    return aligned
