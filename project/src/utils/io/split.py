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
    random_state: int = 42,
):
    """Splits the input DataFrame into training, validation, and test sets.

    This function first filters the DataFrame to include only rows with specific
    KSS_Theta_Alpha_Beta values, maps these to binary labels (0 for alert, 1 for drowsy),
    and then divides the data into 60% training, 20% validation, and 20% test sets.
    It ensures that the 'subject_id' column is retained if present.

    Args:
        df (pd.DataFrame): The input DataFrame, expected to include feature columns
                           and a 'KSS_Theta_Alpha_Beta' column.
        random_state (int): Seed for the random number generator to ensure reproducibility
                            of the data split. Defaults to 42.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
            A tuple containing:
            - X_train (pd.DataFrame): Features for the training set.
            - X_val (pd.DataFrame): Features for the validation set.
            - X_test (pd.DataFrame): Features for the test set.
            - y_train (pd.Series): Labels for the training set.
            - y_val (pd.Series): Labels for the validation set.
            - y_test (pd.Series): Labels for the test set.
    """
    df = df[df["KSS_Theta_Alpha_Beta"].isin(KSS_BIN_LABELS)]

    start_col = "Steering_Range"
    end_col = "LaneOffset_AAA"
    feature_columns = df.loc[:, start_col:end_col].columns.tolist()

#    if 'subject_id' in df.columns:
#        feature_columns.append('subject_id')

    X = df[feature_columns].dropna()
    y = df.loc[X.index, "KSS_Theta_Alpha_Beta"].replace(KSS_LABEL_MAP)

    # Train/val/test split: 60% / 20% / 20%
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test



def data_split_by_subject(
    df: pd.DataFrame,
    train_subjects: list,
    seed: int = 42,
    val_subjects: list = None,
    test_subjects: list = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Splits data into training, validation, and test sets based on provided subject IDs.

    This function is crucial for maintaining subject independence across datasets,
    preventing data leakage when evaluating models. It filters the input DataFrame
    by KSS labels, maps them to binary drowsiness states, and then assigns data
    rows to train, validation, or test sets based on their 'subject_id'.

    Args:
        df (pd.DataFrame): The combined DataFrame containing features and a 'subject_id' column.
        train_subjects (list[str]): A list of subject IDs designated for the training set.
        seed (int): Random seed for reproducibility, particularly for internal operations
                    like label mapping if not explicitly handled. Defaults to 42.
        val_subjects (list[str], optional): A list of subject IDs for the validation set.
                                            If None, the validation set will be empty.
        test_subjects (list[str], optional): A list of subject IDs for the test set.
                                             If None, the test set will be empty.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
            A tuple containing:
            - X_train (pd.DataFrame): Features for the training set.
            - X_val (pd.DataFrame): Features for the validation set.
            - X_test (pd.DataFrame): Features for the test set.
            - y_train (pd.Series): Labels for the training set.
            - y_val (pd.Series): Labels for the validation set.
            - y_test (pd.Series): Labels for the test set.
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

