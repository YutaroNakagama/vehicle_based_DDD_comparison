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
from src.config import KSS_BIN_LABELS, KSS_LABEL_MAP, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
from typing import List, Tuple
from sklearn.model_selection import train_test_split


# --- Model-aware feature column ranges (mirrors loaders.py) ---------------
_MODEL_FEATURE_RANGES = {
    "Lstm": ("steering_std_dev", "lane_offset_mean"),
    "SvmA": ("Steering_Range", "LongAcc_SampleEntropy"),
    "RF":   ("Steering_Range", "LongAcc_SampleEntropy"),
    "SvmW": ("SteeringWheel_DDD", "SteeringWheel_AAA"),
    # common / BalancedRF data uses this range:
    "common": ("Steering_Range", "LaneOffset_AAA"),
}


def _resolve_feature_columns(df: pd.DataFrame, include_subject_id: bool = False) -> list:
    """Resolve feature columns from *df* using model-aware range detection.

    Tries each known (start_col, end_col) pair from ``_MODEL_FEATURE_RANGES``.
    If a matching pair is found in *df*.columns, returns the slice.
    Otherwise falls back to an exclude-list approach (excludes Timestamp,
    KSS labels, etc.).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns should be inspected.
    include_subject_id : bool
        If True, append ``"subject_id"`` when present.

    Returns
    -------
    list[str]
        Ordered list of feature column names.
    """
    for _model, (start_col, end_col) in _MODEL_FEATURE_RANGES.items():
        if start_col in df.columns and end_col in df.columns:
            cols = df.loc[:, start_col:end_col].columns.tolist()
            if include_subject_id and "subject_id" in df.columns:
                cols.append("subject_id")
            return cols

    # Fallback: exclude non-feature columns
    exclude = {
        "Timestamp", "subject_id",
        "KSS_Theta_Alpha_Beta", "KSS_Theta_Alpha_Beta_percent",
        "theta_alpha_over_beta", "theta_alpha_over_beta_label",
        "label",
    }
    if include_subject_id:
        exclude.discard("subject_id")
    cols = [c for c in df.columns if c not in exclude]
    logging.info("[_resolve_feature_columns] Fallback: using %d columns (exclude-list).", len(cols))
    return cols


def _check_nonfinite(X: pd.DataFrame, name: str, preserve_cols: list = None) -> pd.DataFrame:
    """Check for NaN or infinite values in numeric columns and replace safely.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input dataframe to check.
    name : str
        Name for logging purposes.
    preserve_cols : list, optional
        Non-numeric columns to preserve (e.g., ['subject_id']).
    """
    CLIP_THRESHOLD = 1e6  # Avoid float overflow during sklearn fit
    preserve_cols = preserve_cols or []
    
    # Convert only numeric columns to numpy
    X_num = X.select_dtypes(include=[np.number])
    non_numeric_cols = X.columns.difference(X_num.columns)
    
    # Identify which non-numeric columns should be dropped vs preserved
    cols_to_preserve = [c for c in non_numeric_cols if c in preserve_cols]
    cols_to_drop = [c for c in non_numeric_cols if c not in preserve_cols]

    if len(cols_to_drop) > 0:
        logging.warning(f"{name} contains non-numeric columns: {list(cols_to_drop)}")
        X = X.drop(columns=cols_to_drop)

    # Detect NaN / Inf
    mask_nan = X_num.isna()
    mask_inf = np.isinf(X_num.to_numpy())

    if mask_nan.values.any() or mask_inf.any():
        logging.warning(f"{name} detected NaN/Inf values → replacing with column means.")
        # Replace Inf with NaN first, then fill by mean
        X_num = X_num.replace([np.inf, -np.inf], np.nan)
        X_num = X_num.fillna(X_num.mean())
        X[X_num.columns] = X_num

        # Log which columns were affected
        bad_cols = X_num.columns[(mask_nan.any(axis=0) | mask_inf.any(axis=0))]
        if len(bad_cols) > 0:
            logging.info(f"{name} fixed columns: {list(bad_cols)}")
    else:
        logging.debug(f"{name} has no NaN/Inf values.")

    # Clip extremely large or small values
    max_abs = X_num.abs().max()
    large_cols = max_abs[max_abs > CLIP_THRESHOLD].index.tolist()
    if len(large_cols) > 0:
        logging.warning(f"{name} has extremely large values in: {large_cols} → clipping to ±{CLIP_THRESHOLD}")
        X[large_cols] = X[large_cols].clip(-CLIP_THRESHOLD, CLIP_THRESHOLD)

    return X


def data_split(
    df: pd.DataFrame,
    train_ratio: float = None,
    val_ratio: float = None,
    test_ratio: float = None,
    random_state: int = 42,
    keep_subject_id: bool = False,
    kss_bin_labels=None,
    kss_label_map=None,
):
    """
    Split the dataset into train, validation, and test sets.
    Automatically detects feature columns by excluding label-related ones.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing features and label columns.
    train_ratio : float, optional
        Proportion of the dataset to allocate for training. If None, uses config.TRAIN_RATIO.
    val_ratio : float, optional
        Proportion of the dataset to allocate for validation. If None, uses config.VAL_RATIO.
    test_ratio : float, optional
        Proportion of the dataset to allocate for testing. If None, uses config.TEST_RATIO.
    random_state : int, default=42
        Random seed for reproducibility.
    keep_subject_id : bool, default=False
        If True, retain subject_id column in X_train for subject-wise oversampling.
    kss_bin_labels : list, optional
        Model-specific KSS bin labels. If None, uses config.KSS_BIN_LABELS.
    kss_label_map : dict, optional
        Model-specific KSS label mapping. If None, uses config.KSS_LABEL_MAP.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame, pandas.DataFrame,
              pandas.Series, pandas.Series, pandas.Series)
    """
    # Use config defaults if not provided
    if train_ratio is None:
        train_ratio = TRAIN_RATIO
    if val_ratio is None:
        val_ratio = VAL_RATIO
    if test_ratio is None:
        test_ratio = TEST_RATIO

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    # --- Normalize column names to prevent KeyError ---
    def _normalize_col(c):
        if c is None:
            return ""
        s = str(c)
        for bad, rep in [('\ufeff', ''), ('\r', ''), ('\n', ''), (' ', '_')]:
            s = s.replace(bad, rep)
        return s.strip()

    df.columns = [_normalize_col(c) for c in df.columns]

    # --- Step 1: Filter valid KSS labels ---
    active_kss_bins = kss_bin_labels if kss_bin_labels is not None else KSS_BIN_LABELS
    active_kss_map = kss_label_map if kss_label_map is not None else KSS_LABEL_MAP
    df = df[df["KSS_Theta_Alpha_Beta"].isin(active_kss_bins)].copy()

    # --- Step 2: Define columns ---
    exclude_cols = {
        "Timestamp",
        "KSS_Theta_Alpha_Beta",
        "KSS_Theta_Alpha_Beta_percent",
        "theta_alpha_over_beta",
        "theta_alpha_over_beta_label",
    }
    # Conditionally exclude subject_id based on keep_subject_id flag
    if not keep_subject_id:
        exclude_cols.add("subject_id")
    feature_columns = [c for c in df.columns if c not in exclude_cols]

    # --- Step 3: Define X, y ---
    X = df[feature_columns].dropna()
    y = df.loc[X.index, "KSS_Theta_Alpha_Beta"].replace(active_kss_map)

    # --- Step 4: Calculate split sizes ---
    total_size = len(X)
    train_size = int(total_size * train_ratio)
    val_size   = int(total_size * val_ratio)
    test_size  = total_size - train_size - val_size

    # --- Step 5: Split into train / temp ---
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size) / total_size,
        random_state=random_state, stratify=y
    )

    # --- Step 6: Split temp into val / test ---
    if val_size + test_size > 0:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_size / (val_size + test_size),
            random_state=random_state, stratify=y_temp
        )
    else:
        X_val, y_val = pd.DataFrame(), pd.Series(dtype=int)
        X_test, y_test = pd.DataFrame(), pd.Series(dtype=int)

    # Determine which columns to preserve
    preserve_cols = ["subject_id"] if keep_subject_id else []
    X_train = _check_nonfinite(X_train, "X_train", preserve_cols=preserve_cols)
    X_val   = _check_nonfinite(X_val, "X_val")
    X_test  = _check_nonfinite(X_test, "X_test")

    return X_train, X_val, X_test, y_train, y_val, y_test


def data_split_by_subject(
    df: pd.DataFrame,
    train_subjects: list,
    seed: int = 42,
    val_subjects: list = None,
    test_subjects: list = None,
    kss_bin_labels=None,
    kss_label_map=None,
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
    kss_bin_labels : list, optional
        Model-specific KSS labels to keep. Falls back to config default.
    kss_label_map : dict, optional
        Model-specific KSS→binary mapping. Falls back to config default.

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

    active_kss_bins = kss_bin_labels if kss_bin_labels is not None else KSS_BIN_LABELS
    active_kss_map = kss_label_map if kss_label_map is not None else KSS_LABEL_MAP

    logging.info(f"Splitting data by subject.")
    logging.info(f"  Train subjects ({len(train_subjects)}): {train_subjects}")
    logging.info(f"  Validation subjects ({len(val_subjects) if val_subjects else 0}): {val_subjects}")
    logging.info(f"  Test subjects ({len(test_subjects) if test_subjects else 0}): {test_subjects}")

    # --- Normalize column names safely (no .str accessor) ---
    def _normalize_col(c):
        if c is None:
            return ""
        s = str(c)
        for bad, rep in [('\ufeff', ''), ('\r', ''), ('\n', ''), (' ', '_')]:
            s = s.replace(bad, rep)
        return s.strip()

    df.columns = [_normalize_col(c) for c in df.columns]

    # Step 1: Filter rows by KSS labels and create 'label' column
    df_filtered = df[df["KSS_Theta_Alpha_Beta"].isin(active_kss_bins)].copy()
    df_filtered["label"] = df_filtered["KSS_Theta_Alpha_Beta"].replace(active_kss_map)
    logging.info(f"Filtered data from {df.shape[0]} to {df_filtered.shape[0]} rows based on KSS labels.")

    # Step 2: Define feature columns (model-aware)
    feature_columns = _resolve_feature_columns(df_filtered, include_subject_id=True)

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

    X_train = _check_nonfinite(X_train, "X_train")
    X_val   = _check_nonfinite(X_val, "X_val")
    X_test  = _check_nonfinite(X_test, "X_test")

    logging.info("Data splitting summary:")
    logging.info(f"  X_train shape: {X_train.shape}, y_train distribution: {y_train.value_counts().to_dict()}")
    logging.info(f"  X_val shape: {X_val.shape}, y_val distribution: {y_val.value_counts().to_dict()}")
    logging.info(f"  X_test shape: {X_test.shape}, y_test distribution: {y_test.value_counts().to_dict()}")

    return X_train, X_val, X_test, y_train, y_val, y_test

def data_time_split_by_subject(
    df,
    subject_col="subject_id",
    time_col="timestamp",
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    kss_bin_labels=None,
    kss_label_map=None,
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
    train_ratio : float, default=0.7
        Proportion of samples per subject used for training.
    val_ratio : float, default=0.2
        Proportion of samples per subject used for validation.
    test_ratio : float, default=0.1
        Proportion of samples per subject used for testing.
    kss_bin_labels : list, optional
        Model-specific KSS labels to keep (e.g. SvmA includes KSS 6).
        Falls back to ``config.KSS_BIN_LABELS`` when *None*.
    kss_label_map : dict, optional
        Model-specific KSS→binary mapping.
        Falls back to ``config.KSS_LABEL_MAP`` when *None*.

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

    active_kss_bins = kss_bin_labels if kss_bin_labels is not None else KSS_BIN_LABELS
    active_kss_map = kss_label_map if kss_label_map is not None else KSS_LABEL_MAP

    # 1. KSS fileter & Binary convert
    if "KSS_Theta_Alpha_Beta" in df.columns:
        df = df[df["KSS_Theta_Alpha_Beta"].isin(active_kss_bins)].copy()
        df["label"] = df["KSS_Theta_Alpha_Beta"].replace(active_kss_map)
    elif "KSS" in df.columns:
        df["label"] = df["KSS"]
    else:
        raise ValueError("KSS or KSS_Theta_Alpha_Beta column not found")

    # 2. Column define (model-aware: auto-detects feature range)
    feature_columns = _resolve_feature_columns(df, include_subject_id=(subject_col in df.columns))

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

    X_train = _check_nonfinite(X_train, "X_train")
    X_val   = _check_nonfinite(X_val, "X_val")
    X_test  = _check_nonfinite(X_test, "X_test")

    return X_train, X_val, X_test, y_train, y_val, y_test

def time_stratified_three_way_split(
    df: pd.DataFrame,
    label_col: str,
    sort_keys=("Timestamp",),          # or ("subject_id","Timestamp") if already concatenated per subject
    train_ratio=0.6, 
    val_ratio=0.2, 
    test_ratio=0.2,
    tolerance=0.02,  # noqa: ARG001 - reserved for future tolerance checking
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
    train_ratio : float, default=0.7
        Proportion of data for training.
    val_ratio : float, default=0.2
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

