"""Domain Mixup data augmentation for domain generalization.

This module implements Domain Mixup, a data augmentation technique
that interpolates samples across different domains to improve
generalization under domain shift.
"""

import numpy as np
import pandas as pd


def generate_domain_labels(subject_list, X: pd.DataFrame) -> np.ndarray:
    """
    Generate domain labels from the ``subject_id`` column.

    Parameters
    ----------
    subject_list : list
        List of subject identifiers. Currently unused.
    X : pandas.DataFrame
        Feature matrix containing a ``subject_id`` column.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Domain label array aligned with rows of ``X``.
    """
    return X['subject_id'].values

def domain_mixup(
    X: pd.DataFrame,
    y: pd.Series,
    domain_labels: np.ndarray,
    alpha: float = 0.2,
    augment_ratio: float = 0.3,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Perform Domain Mixup augmentation by interpolating samples from different domains.

    Synthetic samples are created by linearly interpolating numeric features
    between pairs of samples from distinct domains. Labels are inherited
    from one of the source samples, weighted by interpolation.

    Parameters
    ----------
    X : pandas.DataFrame
        Original feature matrix. May contain both numeric and non-numeric columns.
    y : pandas.Series
        Labels corresponding to ``X``.
    domain_labels : np.ndarray of shape (n_samples,)
        Domain identifiers for each row in ``X``.
    alpha : float, default=0.2
        Beta distribution parameter controlling the interpolation weight ``λ``.
    augment_ratio : float, default=0.3
        Proportion of synthetic samples to generate relative to ``X``.

    Returns
    -------
    tuple
        A tuple containing:

        - **X_combined** : pandas.DataFrame  
          Original and augmented feature matrix, with numeric and non-numeric columns.
        - **y_combined** : pandas.Series  
          Corresponding labels for original and augmented samples.
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    domain_labels = pd.Series(domain_labels).reset_index(drop=True).values

    unique_domains = np.unique(domain_labels)
    augmented_X = []
    augmented_y = []

    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    num_augment = int(len(X) * augment_ratio)

    for _ in range(num_augment):
        dom1, dom2 = np.random.choice(unique_domains, 2, replace=False)

        dom1_indices = np.where(domain_labels == dom1)[0]
        dom2_indices = np.where(domain_labels == dom2)[0]

        if len(dom1_indices) == 0 or len(dom2_indices) == 0:
            continue

        idx1 = np.random.choice(dom1_indices)
        idx2 = np.random.choice(dom2_indices)

        lam = np.random.beta(alpha, alpha)

        new_numeric = lam * X.iloc[idx1][numeric_columns].values + \
                      (1 - lam) * X.iloc[idx2][numeric_columns].values

        new_y = y.iloc[idx1] if np.random.rand() < lam else y.iloc[idx2]

        new_X = pd.DataFrame([new_numeric], columns=numeric_columns)
        augmented_X.append(new_X)
        augmented_y.append(new_y)

    X_aug_numeric = pd.concat(augmented_X, ignore_index=True)
    y_aug = pd.Series(augmented_y)

    # Separate numeric and non-numeric
    non_numeric = X.drop(columns=numeric_columns).reset_index(drop=True)
    X_numeric = X[numeric_columns].reset_index(drop=True)

    # Concatenate original + augmented
    X_combined = pd.concat([X_numeric, X_aug_numeric], ignore_index=True)
    y_combined = pd.concat([y.reset_index(drop=True), y_aug], ignore_index=True)

    # Handle non-numeric columns (copied randomly from original data)
    if not non_numeric.empty:
        non_numeric_aug = non_numeric.sample(n=len(X_aug_numeric), replace=True).reset_index(drop=True)
        non_numeric_combined = pd.concat([non_numeric, non_numeric_aug], ignore_index=True)
        X_combined = pd.concat([X_combined, non_numeric_combined], axis=1)

    return X_combined, y_combined

