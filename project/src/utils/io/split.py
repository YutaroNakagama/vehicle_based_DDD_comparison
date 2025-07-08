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
    """Splits the input DataFrame into training, validation, and test sets based on specified ratios.

    This function first filters the DataFrame to include only rows with specific
    KSS_Theta_Alpha_Beta values, maps these to binary labels (0 for alert, 1 for drowsy),
    and then divides the data into training, validation, and test sets according to the
    provided ratios. It ensures that the 'subject_id' column is retained if present.

    Args:
        df (pd.DataFrame): The input DataFrame, expected to include feature columns
                           and a 'KSS_Theta_Alpha_Beta' column.
        train_ratio (float): The proportion of the dataset to include in the train split.
        val_ratio (float): The proportion of the dataset to include in the validation split.
        test_ratio (float): The proportion of the dataset to include in the test split.
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

