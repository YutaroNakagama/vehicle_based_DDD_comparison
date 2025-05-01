"""Feature selection metrics for evaluating the relevance of input features.

This module defines functions to calculate multiple feature importance indices:
- Fisher Index
- Correlation Index
- T-test Index
- Mutual Information

Used for ranking features in classification models (e.g., SVM, RF).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score


def calculate_feature_indices(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Calculate feature importance indices for each feature.

    For each feature (column in X), the following indices are computed:
    - Fisher Index: difference in means normalized by variance.
    - Correlation Index: Pearson-like correlation to target.
    - T-test Index: standard two-class t-test statistic.
    - Mutual Information Index: discrete mutual information.

    Args:
        X (pd.DataFrame): Feature matrix of shape (n_samples, n_features).
        y (pd.Series): Target labels (assumed binary: 0 and 1).

    Returns:
        pd.DataFrame: Feature importance scores for all features,
                      indexed by feature names and containing one row per feature.
    """

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    assert len(X) == len(y), f"Mismatch in X and y: {len(X)} != {len(y)}"
    assert set(y.unique()) == {0, 1}, "y must contain only binary labels: 0 and 1"

    indices = {
        "Fisher_Index": [],
        "Correlation_Index": [],
        "T-test_Index": [],
        "Mutual_Information_Index": []
    }

    y_classes = [0, 1]

    for i in range(X.shape[1]):
        xi = X.iloc[:, i].reset_index(drop=True)
    
        mask0 = (y == y_classes[0]).values
        mask1 = (y == y_classes[1]).values
    
        mu0 = xi.iloc[mask0].mean()
        mu1 = xi.iloc[mask1].mean()
        sigma0 = xi.iloc[mask0].std()
        sigma1 = xi.iloc[mask1].std()
        n0 = mask0.sum()
        n1 = mask1.sum()
    
        fisher_index = abs(mu1 - mu0) / (sigma1**2 + sigma0**2 + 1e-6)
        indices["Fisher_Index"].append(fisher_index)
    
        correlation_index = np.cov(xi, y)[0, 1] / (np.std(xi) * np.std(y) + 1e-6)
        indices["Correlation_Index"].append(correlation_index)
    
        t_test_index = abs(mu1 - mu0) / np.sqrt((sigma1**2 / n1) + (sigma0**2 / n0) + 1e-6)
        indices["T-test_Index"].append(t_test_index)
    
        mutual_info = mutual_info_score(xi, y)
        indices["Mutual_Information_Index"].append(mutual_info)

    return pd.DataFrame(indices, index=X.columns)

