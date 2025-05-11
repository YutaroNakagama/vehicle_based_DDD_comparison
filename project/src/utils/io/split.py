"""Data splitting utility for KSS-based binary classification.

This module filters KSS scores and splits the data into train/validation/test
sets for use in supervised learning pipelines.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split


def data_split(df: pd.DataFrame):
    """Split dataset into train/validation/test sets after KSS-based filtering.

    This function:
    - Filters only rows where KSS_Theta_Alpha_Beta is in {1, 2, 8, 9}
    - Maps labels: 1/2 → 0 (alert), 8/9 → 1 (drowsy)
    - Splits into 60% train, 20% val, 20% test
    - Retains 'subject_id' column if available

    Args:
        df (pd.DataFrame): Input DataFrame including features and KSS labels.

    Returns:
        tuple:
            - X_train (pd.DataFrame)
            - X_val (pd.DataFrame)
            - X_test (pd.DataFrame)
            - y_train (pd.Series)
            - y_val (pd.Series)
            - y_test (pd.Series)
    """
    df = df[df["KSS_Theta_Alpha_Beta"].isin([1, 2, 6, 7, 8, 9])]

    start_col = "Steering_Range"
    end_col = "LaneOffset_AAA"
    feature_columns = df.loc[:, start_col:end_col].columns.tolist()

    if 'subject_id' in df.columns:
        feature_columns.append('subject_id')

    X = df[feature_columns].dropna()
    y = df.loc[X.index, "KSS_Theta_Alpha_Beta"].replace({1: 0, 2: 0, 6: 1, 7: 1, 8: 1, 9: 1})

    # Train/val/test split: 60% / 20% / 20%
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test



def data_split_by_subject(df: pd.DataFrame, subject_list: List[str], seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data by subject IDs to avoid data leakage across train/val/test sets.

    Args:
        df (pd.DataFrame): Combined dataframe containing features and subject_id.
        subject_list (List[str]): List of subject IDs to split.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple of:
            - X_train (pd.DataFrame)
            - X_val (pd.DataFrame)
            - X_test (pd.DataFrame)
            - y_train (pd.Series)
            - y_val (pd.Series)
            - y_test (pd.Series)
    """

    # Step 1: Filter rows by KSS labels (1, 2 → 0 [alert], 8, 9 → 1 [drowsy])
    df = df[df["KSS_Theta_Alpha_Beta"].isin([1, 2, 8, 9])].copy()
    df["label"] = df["KSS_Theta_Alpha_Beta"].replace({1: 0, 2: 0, 8: 1, 9: 1})

    # Step 2: Split subject_id list into train/val/test (60/20/20)
    unique_subjects = list(set(subject_list))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_subjects)

    n_total = len(unique_subjects)
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.2)
    n_test = n_total - n_train - n_val  # 端数の調整

    subjects_train = unique_subjects[:n_train]
    subjects_val = unique_subjects[n_train:n_train + n_val]
    subjects_test = unique_subjects[n_train + n_val:]

    # Step 3: Extract subsets by subject group
    df_train = df[df["subject_id"].isin(subjects_train)].copy()
    df_val = df[df["subject_id"].isin(subjects_val)].copy()
    df_test = df[df["subject_id"].isin(subjects_test)].copy()

    # Step 4: Define feature columns based on known range
    start_col = "Steering_Range"
    end_col = "LaneOffset_AAA"
    feature_columns = df.loc[:, start_col:end_col].columns.tolist()

    if 'subject_id' in df.columns:
        feature_columns.append('subject_id')

    X_train = df_train[feature_columns].dropna()
    y_train = df_train.loc[X_train.index, "label"]

    X_val = df_val[feature_columns].dropna()
    y_val = df_val.loc[X_val.index, "label"]

    X_test = df_test[feature_columns].dropna()
    y_test = df_test.loc[X_test.index, "label"]

    return X_train, X_val, X_test, y_train, y_val, y_test

