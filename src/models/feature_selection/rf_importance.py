"""Feature selection using RandomForest-based importance ranking.

This module provides a function to select the most important features
according to feature importance scores from a RandomForest classifier.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.utils.io.split import _check_nonfinite
import logging

def select_top_features_by_importance(X: pd.DataFrame, y: pd.Series, top_k: int = 10) -> list:
    """
    Select the top-k most important features using a RandomForest classifier.

    The function fits a RandomForest model to the input data and ranks
    features by their importance scores. It then returns the names of the
    top-k features.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix of shape ``(n_samples, n_features)``.
    y : pandas.Series
        Target labels corresponding to ``X``.
    top_k : int, default=10
        Number of top features to select.

    Returns
    -------
    list of str
        List containing the names of the top-k features ranked by importance.
    """
    # Safety check: ensure no NaN/inf in the input
    X = _check_nonfinite(X, "rf_importance.X")

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X, y)

    importances = clf.feature_importances_
    feature_ranking = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
    selected = [name for name, _ in feature_ranking[:top_k]]

    logging.info(f"[RF Importance] Selected top-{top_k} features: {selected}")
    return selected

