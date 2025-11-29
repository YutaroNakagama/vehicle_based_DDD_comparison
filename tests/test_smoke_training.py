"""Smoke tests for model training pipeline.

These tests verify that basic ML model training can run end-to-end
with minimal data, without extensive hyperparameter tuning.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC


@pytest.mark.smoke
@pytest.mark.fast
def test_logistic_regression_training(sample_feature_matrix):
    """Test that logistic regression can train on sample data."""
    from sklearn.model_selection import train_test_split
    from src import config
    
    X, y = sample_feature_matrix
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=config.DEFAULT_RANDOM_SEED, stratify=y
    )
    
    # Train model
    model = LogisticRegression(random_state=config.DEFAULT_RANDOM_SEED, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Check outputs
    assert len(y_pred) == len(y_test)
    assert set(y_pred).issubset({0, 1})
    
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    assert 0 <= acc <= 1


@pytest.mark.smoke
@pytest.mark.fast
def test_random_forest_training(sample_feature_matrix):
    """Test that random forest can train on sample data."""
    from sklearn.model_selection import train_test_split
    from src import config
    
    X, y = sample_feature_matrix
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=config.DEFAULT_RANDOM_SEED, stratify=y
    )
    
    # Train model with minimal trees for speed
    model = RandomForestClassifier(
        n_estimators=10,  # Small number for testing
        random_state=config.DEFAULT_RANDOM_SEED,
        max_depth=5,
        n_jobs=1
    )
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Check outputs
    assert len(y_pred) == len(y_test)
    assert set(y_pred).issubset({0, 1})


@pytest.mark.smoke
def test_svm_training(sample_feature_matrix):
    """Test that SVM can train on sample data."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from src import config
    
    X, y = sample_feature_matrix
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=config.DEFAULT_RANDOM_SEED, stratify=y
    )
    
    # Scale features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with linear kernel for speed
    model = SVC(kernel='linear', random_state=config.DEFAULT_RANDOM_SEED)
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Check outputs
    assert len(y_pred) == len(y_test)
    assert set(y_pred).issubset({0, 1})


@pytest.mark.smoke
@pytest.mark.fast
def test_all_classification_metrics(sample_feature_matrix):
    """Test that all classification metrics can be computed."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score
    from src import config
    
    X, y = sample_feature_matrix
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=config.DEFAULT_RANDOM_SEED, stratify=y
    )
    
    # Train simple model
    model = LogisticRegression(random_state=config.DEFAULT_RANDOM_SEED, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Compute all metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba),
        "auc_pr": average_precision_score(y_test, y_proba),
    }
    
    # Check all metrics are in valid range
    for metric_name, value in metrics.items():
        assert 0 <= value <= 1, f"{metric_name} = {value} is out of range [0, 1]"


@pytest.mark.smoke
def test_model_serialization(sample_feature_matrix, temp_dir):
    """Test that trained models can be saved and loaded."""
    import joblib
    from sklearn.model_selection import train_test_split
    from src import config
    
    X, y = sample_feature_matrix
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=config.DEFAULT_RANDOM_SEED, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=10,
        random_state=config.DEFAULT_RANDOM_SEED,
        n_jobs=1
    )
    model.fit(X_train, y_train)
    
    # Get prediction before saving
    y_pred_before = model.predict(X_test)
    
    # Save model
    model_path = temp_dir / "test_model.pkl"
    joblib.dump(model, model_path)
    
    # Load model
    loaded_model = joblib.load(model_path)
    
    # Get prediction after loading
    y_pred_after = loaded_model.predict(X_test)
    
    # Check that predictions are identical
    assert np.array_equal(y_pred_before, y_pred_after)


@pytest.mark.smoke
@pytest.mark.fast
def test_cross_validation_smoke(sample_feature_matrix):
    """Test that cross-validation can run on sample data."""
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from src import config
    
    X, y = sample_feature_matrix
    
    # Simple logistic regression
    model = LogisticRegression(random_state=config.DEFAULT_RANDOM_SEED, max_iter=1000)
    
    # 3-fold CV (small number for speed)
    scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
    
    # Check that we got 3 scores
    assert len(scores) == 3
    
    # Check that all scores are valid
    assert all(0 <= score <= 1 for score in scores)


@pytest.mark.smoke
def test_optuna_minimal_optimization(sample_feature_matrix):
    """Test that Optuna hyperparameter optimization can run."""
    import optuna
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier
    from src import config
    
    X, y = sample_feature_matrix
    
    def objective(trial):
        # Suggest hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 5, 20)
        max_depth = trial.suggest_int("max_depth", 2, 10)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=config.DEFAULT_RANDOM_SEED,
            n_jobs=1
        )
        
        # Evaluate with CV
        score = cross_val_score(model, X, y, cv=2, scoring='accuracy').mean()
        return score
    
    # Run optimization with minimal trials
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3, show_progress_bar=False)
    
    # Check that we got a best trial
    assert study.best_trial is not None
    assert 0 <= study.best_value <= 1


@pytest.mark.smoke
@pytest.mark.fast
def test_feature_importance_extraction(sample_feature_matrix):
    """Test that feature importances can be extracted from tree-based models."""
    from sklearn.ensemble import RandomForestClassifier
    from src import config
    
    X, y = sample_feature_matrix
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=10,
        random_state=config.DEFAULT_RANDOM_SEED,
        n_jobs=1
    )
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Check shape
    assert len(importances) == X.shape[1]
    
    # Check that importances sum to 1
    assert abs(np.sum(importances) - 1.0) < 1e-6
    
    # Check all importances are non-negative
    assert np.all(importances >= 0)
