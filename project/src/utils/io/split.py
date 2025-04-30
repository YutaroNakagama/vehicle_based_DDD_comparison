"""Data splitting utility for KSS-based binary classification.

This module filters KSS scores and splits the data into train/validation/test
sets for use in supervised learning pipelines.
"""

from sklearn.model_selection import train_test_split
import pandas as pd


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
    df = df[df["KSS_Theta_Alpha_Beta"].isin([1, 2, 8, 9])]

    feature_columns = df.columns[1:46].tolist()
    if 'subject_id' in df.columns:
        feature_columns.append('subject_id')

    X = df[feature_columns].dropna()
    y = df.loc[X.index, "KSS_Theta_Alpha_Beta"].replace({1: 0, 2: 0, 8: 1, 9: 1})

    # Train/val/test split: 60% / 20% / 20%
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

