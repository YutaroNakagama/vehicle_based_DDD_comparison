"""Data splitting utility for KSS-based binary classification.

This module filters KSS scores and splits the data into train/validation/test
sets for use in supervised learning pipelines.
"""

import numpy as np
import pandas as pd
import logging
from src.config import KSS_BIN_LABELS, KSS_LABEL_MAP
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
    df = df[df["KSS_Theta_Alpha_Beta"].isin(KSS_BIN_LABELS)]

    start_col = "Steering_Range"
    end_col = "LaneOffset_AAA"
    feature_columns = df.loc[:, start_col:end_col].columns.tolist()

    if 'subject_id' in df.columns:
        feature_columns.append('subject_id')

    X = df[feature_columns].dropna()
    y = df.loc[X.index, "KSS_Theta_Alpha_Beta"].replace(KSS_LABEL_MAP)

    # Train/val/test split: 60% / 20% / 20%
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test



def data_split_by_subject(
    df: pd.DataFrame,
    train_subjects: list,
    seed: int = 42,
    val_subjects: list = None,
    test_subjects: list = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
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

    logging.info(f"df shape before label filter: {df.shape}")

    # Step 1: Filter rows by KSS labels (1, 2 → 0 [alert], 8, 9 → 1 [drowsy])
    df = df[df["KSS_Theta_Alpha_Beta"].isin(KSS_BIN_LABELS)].copy()
    df["label"] = df["KSS_Theta_Alpha_Beta"].replace(KSS_LABEL_MAP)

    logging.info(f"df shape after label filter: {df.shape}")
    logging.info(f"train_subjects: {train_subjects}")
    logging.info(f"val_subjects: {val_subjects}")
    logging.info(f"test_subjects: {test_subjects}")
    logging.info(f"df['subject_id'].unique(): {df['subject_id'].unique()}")

    # Step 4: Define feature columns based on known range
    start_col = "Steering_Range"
    end_col = "LaneOffset_AAA"
    feature_columns = df.loc[:, start_col:end_col].columns.tolist()

    if 'subject_id' in df.columns:
        feature_columns.append('subject_id')

    X_train = df[df["subject_id"].isin(train_subjects)][feature_columns].dropna()
    y_train = df[df["subject_id"].isin(train_subjects)].loc[X_train.index, "label"]

    X_val = pd.DataFrame(); y_val = pd.Series(dtype=int)
    X_test = pd.DataFrame(); y_test = pd.Series(dtype=int)

    if val_subjects is not None:
        X_val = df[df["subject_id"].isin(val_subjects)][feature_columns].dropna()
        y_val = df[df["subject_id"].isin(val_subjects)].loc[X_val.index, "label"]

    if test_subjects is not None:
        X_test = df[df["subject_id"].isin(test_subjects)][feature_columns].dropna()
        y_test = df[df["subject_id"].isin(test_subjects)].loc[X_test.index, "label"]

    logging.info(f"X_train shape: {X_train.shape}, y_train.value_counts: {y_train.value_counts().to_dict()}")
    logging.info(f"X_val shape: {X_val.shape}, y_val.value_counts: {y_val.value_counts().to_dict()}")
    logging.info(f"X_test shape: {X_test.shape}, y_test.value_counts: {y_test.value_counts().to_dict()}")

    return X_train, X_val, X_test, y_train, y_val, y_test

