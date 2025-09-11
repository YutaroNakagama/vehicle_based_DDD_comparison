"""Data Splitting Utilities for Driver Drowsiness Detection.

This module provides functions for splitting datasets into training, validation,
and test sets, with a focus on handling KSS (Karolinska Sleepiness Scale)-based
binary classification. It includes functionalities for filtering data based on KSS scores,
mapping them to binary labels, and performing both random and subject-wise data splits
to ensure proper evaluation and prevent data leakage.
"""

import numpy as np
import pandas as pd
import logging
from src.config import KSS_BIN_LABELS, KSS_LABEL_MAP
from typing import List, Tuple
from sklearn.model_selection import train_test_split


def data_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
):
    """
    Split the dataset into train, validation, and test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing features and a ``KSS_Theta_Alpha_Beta`` column.
    train_ratio : float, default=0.8
        Proportion of the dataset to allocate for training.
    val_ratio : float, default=0.1
        Proportion of the dataset to allocate for validation.
    test_ratio : float, default=0.1
        Proportion of the dataset to allocate for testing.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame,
              pandas.Series, pandas.Series, pandas.Series)
        - ``X_train`` : Training features.
        - ``X_val`` : Validation features.
        - ``X_test`` : Test features.
        - ``y_train`` : Training labels.
        - ``y_val`` : Validation labels.
        - ``y_test`` : Test labels.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    df = df[df["KSS_Theta_Alpha_Beta"].isin(KSS_BIN_LABELS)]

    start_col = "Steering_Range"
    end_col = "LaneOffset_AAA"
    feature_columns = df.loc[:, start_col:end_col].columns.tolist()

    X = df[feature_columns].dropna()
    y = df.loc[X.index, "KSS_Theta_Alpha_Beta"].replace(KSS_LABEL_MAP)

    # Calculate sizes
    total_size = len(X)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size # Ensure all samples are used

    # Split into train and temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size) / total_size, random_state=random_state, stratify=y
    )

    # Split temp into val and test
    if val_size + test_size > 0:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size / (val_size + test_size), random_state=random_state, stratify=y_temp
        )
    else:
        X_val, y_val = pd.DataFrame(), pd.Series(dtype=int)
        X_test, y_test = pd.DataFrame(), pd.Series(dtype=int)

    return X_train, X_val, X_test, y_train, y_val, y_test



def data_split_by_subject(
    df: pd.DataFrame,
    train_subjects: list,
    seed: int = 42,
    val_subjects: list = None,
    test_subjects: list = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split the dataset by subject IDs into train, validation, and test sets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing features, labels, and ``subject_id`` column.
    train_subjects : list of str
        Subject IDs used for training.
    seed : int, default=42
        Random seed for reproducibility.
    val_subjects : list of str, optional
        Subject IDs used for validation. If ``None``, no validation set is created.
    test_subjects : list of str, optional
        Subject IDs used for testing. If ``None``, no test set is created.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame,
              pandas.Series, pandas.Series, pandas.Series)
        - ``X_train`` : Training features.
        - ``X_val`` : Validation features.
        - ``X_test`` : Test features.
        - ``y_train`` : Training labels.
        - ``y_val`` : Validation labels.
        - ``y_test`` : Test labels.
    """

    logging.info(f"Splitting data by subject.")
    logging.info(f"  Train subjects ({len(train_subjects)}): {train_subjects}")
    logging.info(f"  Validation subjects ({len(val_subjects) if val_subjects else 0}): {val_subjects}")
    logging.info(f"  Test subjects ({len(test_subjects) if test_subjects else 0}): {test_subjects}")

    # Step 1: Filter rows by KSS labels and create 'label' column
    df_filtered = df[df["KSS_Theta_Alpha_Beta"].isin(KSS_BIN_LABELS)].copy()
    df_filtered["label"] = df_filtered["KSS_Theta_Alpha_Beta"].replace(KSS_LABEL_MAP)
    logging.info(f"Filtered data from {df.shape[0]} to {df_filtered.shape[0]} rows based on KSS labels.")

    # Step 2: Define feature columns
    start_col = "Steering_Range"
    end_col = "LaneOffset_AAA"
    feature_columns = df_filtered.loc[:, start_col:end_col].columns.tolist()
    if 'subject_id' in df_filtered.columns:
        feature_columns.append('subject_id')

    # Step 3: Create datasets based on subject lists
    X_train = df_filtered[df_filtered["subject_id"].isin(train_subjects)][feature_columns].dropna()
    y_train = df_filtered.loc[X_train.index, "label"]

    X_val = pd.DataFrame()
    y_val = pd.Series(dtype=int)
    if val_subjects:
        X_val = df_filtered[df_filtered["subject_id"].isin(val_subjects)][feature_columns].dropna()
        y_val = df_filtered.loc[X_val.index, "label"]

    X_test = pd.DataFrame()
    y_test = pd.Series(dtype=int)
    if test_subjects:
        X_test = df_filtered[df_filtered["subject_id"].isin(test_subjects)][feature_columns].dropna()
        y_test = df_filtered.loc[X_test.index, "label"]

    logging.info("Data splitting summary:")
    logging.info(f"  X_train shape: {X_train.shape}, y_train distribution: {y_train.value_counts().to_dict()}")
    logging.info(f"  X_val shape: {X_val.shape}, y_val distribution: {y_val.value_counts().to_dict()}")
    logging.info(f"  X_test shape: {X_test.shape}, y_test distribution: {y_test.value_counts().to_dict()}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def data_time_split_by_subject(
    df,
    subject_col="subject_id",
    time_col="timestamp",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
):
    """
    Split data by subject while preserving temporal order within each subject.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing features, labels, and subject identifiers.
    subject_col : str, default="subject_id"
        Column name for subject identifiers.
    time_col : str, default="timestamp"
        Column name for timestamps used to order samples.
    train_ratio : float, default=0.8
        Proportion of samples per subject used for training.
    val_ratio : float, default=0.1
        Proportion of samples per subject used for validation.
    test_ratio : float, default=0.1
        Proportion of samples per subject used for testing.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame,
              pandas.Series, pandas.Series, pandas.Series)
        - ``X_train`` : Training features.
        - ``X_val`` : Validation features.
        - ``X_test`` : Test features.
        - ``y_train`` : Training labels.
        - ``y_val`` : Validation labels.
        - ``y_test`` : Test labels.
    """
    from src.config import KSS_BIN_LABELS, KSS_LABEL_MAP

    # 1. KSS fileter & Binary convert
    if "KSS_Theta_Alpha_Beta" in df.columns:
        df = df[df["KSS_Theta_Alpha_Beta"].isin(KSS_BIN_LABELS)].copy()
        df["label"] = df["KSS_Theta_Alpha_Beta"].replace(KSS_LABEL_MAP)
    elif "KSS" in df.columns:
        df["label"] = df["KSS"]
    else:
        raise ValueError("KSS or KSS_Theta_Alpha_Beta column not found")

    # 2. Column define （e.g. Steering_Range ～ LaneOffset_AAA, subject_id）
    start_col = "Steering_Range"
    end_col = "LaneOffset_AAA"
    feature_columns = df.loc[:, start_col:end_col].columns.tolist()
    if subject_col in df.columns:
        feature_columns.append(subject_col)

    dfs_train, dfs_val, dfs_test = [], [], []

    for subj, df_sub in df.groupby(subject_col):
        df_sub = df_sub.sort_values(time_col)
        n = len(df_sub)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        dfs_train.append(df_sub.iloc[:n_train])
        dfs_val.append(df_sub.iloc[n_train:n_train+n_val])
        dfs_test.append(df_sub.iloc[n_train+n_val:])

    train = pd.concat(dfs_train).reset_index(drop=True)
    val = pd.concat(dfs_val).reset_index(drop=True)
    test = pd.concat(dfs_test).reset_index(drop=True)

    X_train = train[feature_columns].drop(columns=[subject_col], errors="ignore")
    X_val = val[feature_columns].drop(columns=[subject_col], errors="ignore")
    X_test = test[feature_columns].drop(columns=[subject_col], errors="ignore")
    y_train = train["label"]
    y_val = val["label"]
    y_test = test["label"]

    return X_train, X_val, X_test, y_train, y_val, y_test

def time_stratified_three_way_split(
    df: pd.DataFrame,
    label_col: str,
    sort_keys=("Timestamp",),          # or ("subject_id","Timestamp") if already concatenated per subject
    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
    tolerance=0.02,
    window_prop=0.10,
    min_chunk=100,
):
    """
    Perform stratified time-based split into train, validation, and test sets.

    Ensures class balance is approximately preserved in each split by
    adjusting cut points around nominal ratios.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing features and labels.
    label_col : str
        Column name for labels.
    sort_keys : tuple of str, default=("Timestamp",)
        Columns to sort by before splitting (e.g., ``("subject_id","Timestamp")``).
    train_ratio : float, default=0.8
        Proportion of data for training.
    val_ratio : float, default=0.1
        Proportion of data for validation.
    test_ratio : float, default=0.1
        Proportion of data for testing.
    tolerance : float, default=0.02
        Allowed deviation in class distribution ratios.
    window_prop : float, default=0.10
        Proportion of dataset considered for searching optimal cut points.
    min_chunk : int, default=100
        Minimum number of samples per split.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        Index arrays for training, validation, and test sets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "ratios must sum to 1"

    # 1) Sort by time (and optionally by subject_id then time)
    df_sorted = df.sort_values(list(sort_keys)).reset_index(drop=False)  # keep original index
    y = df_sorted[label_col].astype(int).to_numpy()
    n = len(y)
    if n < (3 * min_chunk):
        # Too small, fall back to nominal cuts
        i0 = int(round(train_ratio * n))
        j0 = int(round((train_ratio + val_ratio) * n))
        return df_sorted.loc[:i0-1, "index"].to_numpy(), \
               df_sorted.loc[i0:j0-1, "index"].to_numpy(), \
               df_sorted.loc[j0:, "index"].to_numpy()

    # 2) Global target ratio
    total_pos = int(y.sum())
    r_target = total_pos / n

    # 3) Cumulative positives
    cum_pos = np.cumsum(y)                   # cum_pos[k] = #pos in [0..k]
    # cum_count is simply np.arange(1, n+1)

    # 4) Nominal cuts & search windows
    i0 = int(round(train_ratio * n))
    j0 = int(round((train_ratio + val_ratio) * n))
    w  = max(1, int(round(window_prop * n)))

    i_min = max(min_chunk, i0 - w)
    i_max = min(i0 + w, n - 2*min_chunk)
    best = (1e9, i0, j0)  # (loss, i, j)

    # weights: emphasize ratio matching but also keep sizes near nominal
    w_ratio = 2.0
    w_size  = 1.0

    for i in range(i_min, i_max + 1):
        j_min = max(i + min_chunk, j0 - w)
        j_max = min(j0 + w, n - min_chunk)
        for j in range(j_min, j_max + 1):
            n1 = i
            n2 = j - i
            n3 = n - j

            # segment pos counts
            p1 = cum_pos[i-1] if i > 0 else 0
            p2 = cum_pos[j-1] - cum_pos[i-1]
            p3 = cum_pos[n-1] - cum_pos[j-1]

            r1 = p1 / max(1, n1)
            r2 = p2 / max(1, n2)
            r3 = p3 / max(1, n3)

            # ratio loss + size loss (L1)
            loss_ratio = abs(r1 - r_target) + abs(r2 - r_target) + abs(r3 - r_target)
            loss_size  = abs(n1/n - train_ratio) + abs(n2/n - val_ratio) + abs(n3/n - test_ratio)
            loss = w_ratio * loss_ratio + w_size * loss_size

            if loss < best[0]:
                best = (loss, i, j)

    _, i_star, j_star = best

    # 5) If still far from tolerance, we accept the best (cannot do better without resampling)
    # Caller can check/ log actual deviations if desired.

    idx_train = df_sorted.loc[:i_star-1, "index"].to_numpy()
    idx_val   = df_sorted.loc[i_star:j_star-1, "index"].to_numpy()
    idx_test  = df_sorted.loc[j_star:, "index"].to_numpy()
    return idx_train, idx_val, idx_test

