"""Smoke tests for data pipeline.

These tests verify that the data processing pipeline can run end-to-end
with minimal data, without checking correctness in detail.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.smoke
@pytest.mark.fast
def test_sample_data_fixtures(sample_eeg_data, sample_simlsl_data):
    """Test that sample data fixtures are generated correctly."""
    # Check EEG data
    assert isinstance(sample_eeg_data, pd.DataFrame)
    assert len(sample_eeg_data) > 0
    assert "subject_id" in sample_eeg_data.columns
    assert "KSS" in sample_eeg_data.columns
    
    # Check SIMLSL data
    assert isinstance(sample_simlsl_data, pd.DataFrame)
    assert len(sample_simlsl_data) > 0
    assert "subject_id" in sample_simlsl_data.columns
    assert "KSS" in sample_simlsl_data.columns


@pytest.mark.smoke
@pytest.mark.fast
def test_kss_label_conversion(sample_eeg_data):
    """Test that KSS labels can be converted to binary."""
    from src import config
    
    # Map KSS to binary
    kss_values = sample_eeg_data["KSS"].values
    binary_labels = np.array([config.KSS_LABEL_MAP[k] for k in kss_values])
    
    # Check that we have both classes
    assert set(binary_labels).issubset({0, 1})
    assert len(binary_labels) == len(kss_values)


@pytest.mark.smoke
@pytest.mark.fast
def test_feature_matrix_fixture(sample_feature_matrix):
    """Test that feature matrix fixture is generated correctly."""
    X, y = sample_feature_matrix
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.ndim == 2
    assert y.ndim == 1
    assert len(X) == len(y)
    assert set(y).issubset({0, 1})


@pytest.mark.smoke
def test_data_windowing_logic():
    """Test basic windowing logic for time series data."""
    from src import config
    
    # Simulate time series data
    np.random.seed(42)
    n_samples = 1000
    data = np.random.randn(n_samples, 5)  # 5 channels
    
    # Get window parameters
    window_sec = config.MODEL_WINDOW_CONFIG["common"]["window_sec"]
    step_sec = config.MODEL_WINDOW_CONFIG["common"]["step_sec"]
    sample_rate = config.SAMPLE_RATE_SIMLSL
    
    window_size = int(window_sec * sample_rate)
    step_size = int(step_sec * sample_rate)
    
    # Create windows
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data[start:start + window_size]
        windows.append(window)
    
    # Check that we got some windows
    assert len(windows) > 0
    
    # Check window shape
    assert windows[0].shape == (window_size, 5)


@pytest.mark.smoke
@pytest.mark.fast
def test_train_test_split_with_sample_data(sample_feature_matrix):
    """Test that sklearn train_test_split works with sample data."""
    from sklearn.model_selection import train_test_split
    from src import config
    
    X, y = sample_feature_matrix
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.DEFAULT_RANDOM_SEED, stratify=y
    )
    
    val_test_ratio = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_test_ratio),
        random_state=config.DEFAULT_RANDOM_SEED, stratify=y_temp
    )
    
    # Check shapes
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0
    assert len(X_train) + len(X_val) + len(X_test) == len(X)


@pytest.mark.smoke
def test_basic_feature_extraction(sample_eeg_data):
    """Test basic feature extraction (mean, std) from EEG data."""
    # Select numeric columns (exclude subject_id, timestamp, KSS)
    eeg_columns = [col for col in sample_eeg_data.columns 
                   if col not in ["subject_id", "timestamp", "KSS"]]
    
    eeg_data = sample_eeg_data[eeg_columns].values
    
    # Compute basic statistics
    means = np.mean(eeg_data, axis=0)
    stds = np.std(eeg_data, axis=0)
    
    # Check shapes
    assert len(means) == len(eeg_columns)
    assert len(stds) == len(eeg_columns)
    
    # Check that we got reasonable values
    assert np.all(np.isfinite(means))
    assert np.all(np.isfinite(stds))
    assert np.all(stds >= 0)


@pytest.mark.smoke
@pytest.mark.fast
def test_imbalanced_data_handling(sample_feature_matrix):
    """Test that SMOTE can be applied to handle imbalanced data."""
    from imblearn.over_sampling import SMOTE
    
    X, y = sample_feature_matrix
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Check that data was resampled
    assert len(X_resampled) >= len(X)
    
    # Check that classes are now balanced
    unique, counts = np.unique(y_resampled, return_counts=True)
    assert len(unique) == 2
    assert counts[0] == counts[1]  # Should be balanced


@pytest.mark.smoke
def test_wavelet_filters_are_valid():
    """Test that wavelet filter coefficients are properly defined."""
    from src import config
    
    # Check filter shapes
    assert len(config.SCALING_FILTER) == 4
    assert len(config.WAVELET_FILTER) == 4
    
    # Check that they are numpy arrays
    assert isinstance(config.SCALING_FILTER, np.ndarray)
    assert isinstance(config.WAVELET_FILTER, np.ndarray)
    
    # Check that wavelet level is positive
    assert config.WAVELET_LEV > 0
