"""Tests for configuration module.

This module tests that all configuration values are properly loaded
and that paths are correctly configured.
"""

import os
from pathlib import Path

import pytest


def test_config_import():
    """Test that config module can be imported."""
    from src import config
    assert config is not None


def test_config_constants():
    """Test that basic configuration constants are defined."""
    from src import config
    
    # Check that essential constants exist
    assert hasattr(config, "SAMPLE_RATE_SIMLSL")
    assert hasattr(config, "SAMPLE_RATE_EEG")
    assert hasattr(config, "KSS_BIN_LABELS")
    assert hasattr(config, "KSS_LABEL_MAP")
    assert hasattr(config, "WAVELET_LEV")
    assert hasattr(config, "DEFAULT_RANDOM_SEED")


def test_sample_rates():
    """Test that sample rates are sensible."""
    from src import config
    
    assert config.SAMPLE_RATE_SIMLSL > 0
    assert config.SAMPLE_RATE_EEG > 0
    assert config.SAMPLE_RATE_EEG >= config.SAMPLE_RATE_SIMLSL  # EEG usually higher


def test_kss_labels_binary_mapping():
    """Test that KSS labels are properly mapped to binary classes."""
    from src import config
    
    # Check that all KSS_BIN_LABELS have a mapping
    for label in config.KSS_BIN_LABELS:
        assert label in config.KSS_LABEL_MAP
    
    # Check that mapping only contains 0 (alert) and 1 (drowsy)
    mapped_values = set(config.KSS_LABEL_MAP.values())
    assert mapped_values.issubset({0, 1})
    
    # Check that both classes are represented
    assert 0 in mapped_values
    assert 1 in mapped_values


def test_data_split_ratios():
    """Test that train/val/test ratios sum to 1.0."""
    from src import config
    
    total = config.TRAIN_RATIO + config.VAL_RATIO + config.TEST_RATIO
    assert abs(total - 1.0) < 1e-6, f"Split ratios sum to {total}, not 1.0"
    
    # Check all ratios are positive
    assert config.TRAIN_RATIO > 0
    assert config.VAL_RATIO > 0
    assert config.TEST_RATIO > 0


def test_model_choices_not_empty():
    """Test that model choices list is not empty."""
    from src import config
    
    assert len(config.MODEL_CHOICES) > 0
    assert len(config.DATA_PROCESS_CHOICES) > 0


def test_window_config_structure():
    """Test that window configuration has expected structure."""
    from src import config
    
    assert isinstance(config.MODEL_WINDOW_CONFIG, dict)
    
    for model_name, window_params in config.MODEL_WINDOW_CONFIG.items():
        assert "window_sec" in window_params
        assert "step_sec" in window_params
        assert window_params["window_sec"] > 0
        assert window_params["step_sec"] > 0
        assert window_params["step_sec"] <= window_params["window_sec"]


def test_distance_metrics():
    """Test that distance metrics are defined."""
    from src import config
    
    assert len(config.DISTANCE_METRICS) > 0
    assert "mmd" in config.DISTANCE_METRICS
    assert "wasserstein" in config.DISTANCE_METRICS


def test_training_modes():
    """Test that training modes are properly configured."""
    from src import config
    
    assert len(config.TRAINING_MODES) > 0
    assert "source_only" in config.TRAINING_MODES
    assert "target_only" in config.TRAINING_MODES
    assert "finetune" in config.TRAINING_MODES


def test_configure_blas_threads():
    """Test BLAS thread configuration function."""
    from src import config
    
    # Test with 1 thread
    config.configure_blas_threads(1)
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
    
    # Test with 4 threads
    config.configure_blas_threads(4)
    assert os.environ.get("OMP_NUM_THREADS") == "4"
    assert os.environ.get("OPENBLAS_NUM_THREADS") == "4"


@pytest.mark.fast
def test_classification_metrics():
    """Test that classification metrics are defined."""
    from src import config
    
    assert len(config.CLASSIFICATION_METRICS) > 0
    assert "accuracy" in config.CLASSIFICATION_METRICS
    assert "precision" in config.CLASSIFICATION_METRICS
    assert "recall" in config.CLASSIFICATION_METRICS
    assert "f1" in config.CLASSIFICATION_METRICS


@pytest.mark.fast
def test_optuna_config():
    """Test that Optuna hyperparameter tuning config is reasonable."""
    from src import config
    
    assert config.N_TRIALS > 0
    assert config.OPTUNA_N_STARTUP_TRIALS >= 0
    assert config.OPTUNA_N_WARMUP_STEPS >= 0
    assert config.OPTUNA_INTERVAL_STEPS > 0
