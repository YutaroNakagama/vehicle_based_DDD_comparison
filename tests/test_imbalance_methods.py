"""Tests for imbalanced data handling methods.

These tests verify that all implemented imbalance handling methods work correctly,
ensuring the KNN + Imbalance experiments will run without implementation issues.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================================
# Fixtures for imbalanced data
# ============================================================================

@pytest.fixture
def imbalanced_feature_matrix():
    """Generate an imbalanced feature matrix (10:1 ratio).
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X, y) where X is feature matrix and y has imbalanced labels
    """
    np.random.seed(42)
    n_majority = 500
    n_minority = 50  # 10:1 imbalance ratio
    n_features = 20
    
    # Majority class
    X_majority = np.random.randn(n_majority, n_features)
    y_majority = np.zeros(n_majority, dtype=int)
    
    # Minority class (slightly shifted distribution)
    X_minority = np.random.randn(n_minority, n_features) + 0.5
    y_minority = np.ones(n_minority, dtype=int)
    
    X = np.vstack([X_majority, X_minority])
    y = np.hstack([y_majority, y_minority])
    
    # Shuffle
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]
    
    return X, y


@pytest.fixture
def imbalanced_dataframe(imbalanced_feature_matrix):
    """Convert imbalanced feature matrix to DataFrame format."""
    X, y = imbalanced_feature_matrix
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="label")
    return X_df, y_series


# ============================================================================
# Test SMOTE variants (Oversampling)
# ============================================================================

@pytest.mark.smoke
@pytest.mark.fast
def test_smote_basic(imbalanced_feature_matrix):
    """Test that basic SMOTE works on imbalanced data."""
    from imblearn.over_sampling import SMOTE
    
    X, y = imbalanced_feature_matrix
    
    minority_count_before = np.sum(y == 1)
    majority_count_before = np.sum(y == 0)
    
    sampler = SMOTE(
        sampling_strategy=0.33,  # Target ratio same as implementation
        random_state=42,
        k_neighbors=min(5, minority_count_before - 1)
    )
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    minority_count_after = np.sum(y_resampled == 1)
    
    # Check that minority class was increased
    assert minority_count_after > minority_count_before
    # Check that we have both classes
    assert len(np.unique(y_resampled)) == 2
    # Check features maintained
    assert X_resampled.shape[1] == X.shape[1]


@pytest.mark.smoke
def test_smote_tomek(imbalanced_feature_matrix):
    """Test SMOTE + Tomek Links combination."""
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    
    X, y = imbalanced_feature_matrix
    
    minority_count = np.sum(y == 1)
    
    sampler = SMOTETomek(
        sampling_strategy=0.33,
        random_state=42,
        n_jobs=1,
        smote=SMOTE(
            random_state=42,
            k_neighbors=min(5, minority_count - 1)
        )
    )
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    # Check that we have both classes
    assert len(np.unique(y_resampled)) == 2
    # Check that minority increased
    assert np.sum(y_resampled == 1) > minority_count


@pytest.mark.smoke
def test_smote_enn(imbalanced_feature_matrix):
    """Test SMOTE + ENN combination."""
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import SMOTE
    
    X, y = imbalanced_feature_matrix
    
    minority_count = np.sum(y == 1)
    
    sampler = SMOTEENN(
        sampling_strategy=0.33,
        random_state=42,
        n_jobs=1,
        smote=SMOTE(
            random_state=42,
            k_neighbors=min(5, minority_count - 1)
        )
    )
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    # Check that we have both classes
    assert len(np.unique(y_resampled)) == 2


@pytest.mark.smoke
def test_smote_rus_pipeline(imbalanced_feature_matrix):
    """Test SMOTE + RandomUnderSampler hybrid pipeline."""
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    
    X, y = imbalanced_feature_matrix
    
    minority_count = np.sum(y == 1)
    
    smote = SMOTE(
        sampling_strategy=0.5,  # Minority becomes 50% of majority
        random_state=42,
        k_neighbors=min(5, minority_count - 1)
    )
    rus = RandomUnderSampler(
        sampling_strategy=0.8,  # Final ratio: minority = 80% of majority
        random_state=42
    )
    sampler = ImbPipeline([
        ('smote', smote),
        ('rus', rus)
    ])
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    # Check that we have both classes
    assert len(np.unique(y_resampled)) == 2
    # Check that data size changed
    assert len(y_resampled) != len(y)


# ============================================================================
# Test Undersampling methods
# ============================================================================

@pytest.mark.smoke
@pytest.mark.fast
def test_random_undersampler(imbalanced_feature_matrix):
    """Test Random Under-Sampling."""
    from imblearn.under_sampling import RandomUnderSampler
    
    X, y = imbalanced_feature_matrix
    
    majority_count_before = np.sum(y == 0)
    
    sampler = RandomUnderSampler(
        sampling_strategy=0.33,  # minority/majority ratio after sampling
        random_state=42
    )
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    majority_count_after = np.sum(y_resampled == 0)
    
    # Check that majority class was reduced
    assert majority_count_after < majority_count_before
    # Check that minority class unchanged
    assert np.sum(y_resampled == 1) == np.sum(y == 1)
    # Check that we have both classes
    assert len(np.unique(y_resampled)) == 2


@pytest.mark.smoke
@pytest.mark.fast
def test_tomek_links(imbalanced_feature_matrix):
    """Test Tomek Links undersampling."""
    from imblearn.under_sampling import TomekLinks
    
    X, y = imbalanced_feature_matrix
    
    original_size = len(y)
    
    sampler = TomekLinks(
        sampling_strategy='majority',  # Only remove majority class samples
        n_jobs=1
    )
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    # Tomek Links may or may not remove samples depending on boundary pairs
    # Just check that it doesn't crash and maintains both classes
    assert len(np.unique(y_resampled)) == 2
    # Size should be <= original (Tomek removes boundary samples)
    assert len(y_resampled) <= original_size


# ============================================================================
# Test Ensemble methods (for reference)
# ============================================================================

@pytest.mark.smoke
def test_balanced_random_forest(imbalanced_feature_matrix):
    """Test BalancedRandomForestClassifier."""
    from imblearn.ensemble import BalancedRandomForestClassifier
    
    X, y = imbalanced_feature_matrix
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    clf = BalancedRandomForestClassifier(
        n_estimators=10,  # Small for testing
        random_state=42,
        n_jobs=1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Check outputs
    assert len(y_pred) == len(y_test)
    assert set(y_pred).issubset({0, 1})


@pytest.mark.smoke
def test_easy_ensemble(imbalanced_feature_matrix):
    """Test EasyEnsembleClassifier."""
    from imblearn.ensemble import EasyEnsembleClassifier
    
    X, y = imbalanced_feature_matrix
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    clf = EasyEnsembleClassifier(
        n_estimators=5,  # Small for testing
        random_state=42,
        n_jobs=1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Check outputs
    assert len(y_pred) == len(y_test)
    assert set(y_pred).issubset({0, 1})


# ============================================================================
# Test full training flow with imbalance methods
# ============================================================================

@pytest.mark.smoke
def test_rf_training_with_smote(imbalanced_dataframe):
    """Test RF training pipeline with SMOTE."""
    from imblearn.over_sampling import SMOTE
    
    X_df, y = imbalanced_dataframe
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE
    minority_count = np.sum(y_train == 1)
    sampler = SMOTE(
        sampling_strategy=0.33,
        random_state=42,
        k_neighbors=min(5, minority_count - 1)
    )
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
    
    # Train RF
    clf = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=1
    )
    clf.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    
    # Check predictions
    assert len(y_pred) == len(y_test)
    assert all(0 <= p <= 1 for p in y_proba)


@pytest.mark.smoke
def test_rf_training_with_undersample_rus(imbalanced_dataframe):
    """Test RF training pipeline with Random Under-Sampling."""
    from imblearn.under_sampling import RandomUnderSampler
    
    X_df, y = imbalanced_dataframe
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply RUS
    sampler = RandomUnderSampler(
        sampling_strategy=0.33,
        random_state=42
    )
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
    
    # Train RF
    clf = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=1
    )
    clf.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    
    # Check predictions
    assert len(y_pred) == len(y_test)


@pytest.mark.smoke
def test_rf_training_with_undersample_tomek(imbalanced_dataframe):
    """Test RF training pipeline with Tomek Links."""
    from imblearn.under_sampling import TomekLinks
    
    X_df, y = imbalanced_dataframe
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply Tomek Links
    sampler = TomekLinks(
        sampling_strategy='majority',
        n_jobs=1
    )
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
    
    # Train RF
    clf = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=1
    )
    clf.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    
    # Check predictions
    assert len(y_pred) == len(y_test)


@pytest.mark.smoke
def test_rf_training_with_smote_rus(imbalanced_dataframe):
    """Test RF training pipeline with SMOTE + RUS hybrid."""
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    
    X_df, y = imbalanced_dataframe
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE + RUS
    minority_count = np.sum(y_train == 1)
    smote = SMOTE(
        sampling_strategy=0.5,
        random_state=42,
        k_neighbors=min(5, minority_count - 1)
    )
    rus = RandomUnderSampler(
        sampling_strategy=0.8,
        random_state=42
    )
    sampler = ImbPipeline([
        ('smote', smote),
        ('rus', rus)
    ])
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
    
    # Train RF
    clf = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=1
    )
    clf.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    
    # Check predictions
    assert len(y_pred) == len(y_test)


@pytest.mark.smoke
def test_rf_training_with_smote_tomek(imbalanced_dataframe):
    """Test RF training pipeline with SMOTE + Tomek Links."""
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    
    X_df, y = imbalanced_dataframe
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE + Tomek
    minority_count = np.sum(y_train == 1)
    sampler = SMOTETomek(
        sampling_strategy=0.33,
        random_state=42,
        n_jobs=1,
        smote=SMOTE(
            random_state=42,
            k_neighbors=min(5, minority_count - 1)
        )
    )
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
    
    # Train RF
    clf = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        class_weight='balanced_subsample',
        n_jobs=1
    )
    clf.fit(X_train_resampled, y_train_resampled)
    
    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    
    # Check predictions
    assert len(y_pred) == len(y_test)


# ============================================================================
# Test evaluation metrics for imbalanced data
# ============================================================================

@pytest.mark.smoke
@pytest.mark.fast
def test_imbalanced_metrics(imbalanced_feature_matrix):
    """Test that appropriate metrics for imbalanced data can be computed."""
    from sklearn.metrics import (
        precision_recall_curve,
        average_precision_score,
        fbeta_score,
        recall_score,
        precision_score
    )
    from imblearn.over_sampling import SMOTE
    
    X, y = imbalanced_feature_matrix
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Apply SMOTE and train
    minority_count = np.sum(y_train == 1)
    sampler = SMOTE(
        sampling_strategy=0.33,
        random_state=42,
        k_neighbors=min(5, minority_count - 1)
    )
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    clf.fit(X_train_resampled, y_train_resampled)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Compute imbalanced-appropriate metrics
    auprc = average_precision_score(y_test, y_proba)
    f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    
    # All metrics should be in [0, 1]
    assert 0 <= auprc <= 1
    assert 0 <= f2 <= 1
    assert 0 <= recall <= 1
    assert 0 <= precision <= 1


# ============================================================================
# Test baseline (no imbalance handling)
# ============================================================================

@pytest.mark.smoke
@pytest.mark.fast
def test_baseline_no_sampling(imbalanced_dataframe):
    """Test baseline training without any imbalance handling."""
    X_df, y = imbalanced_dataframe
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train RF without any sampling (baseline)
    clf = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        class_weight='balanced_subsample',  # Still use class_weight
        n_jobs=1
    )
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    
    # Check predictions
    assert len(y_pred) == len(y_test)
    assert set(y_pred).issubset({0, 1})
