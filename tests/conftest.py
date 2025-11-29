"""Pytest configuration and shared fixtures for all tests.

This module provides common fixtures and test configuration that are
automatically available to all test modules.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# Session-level fixtures (run once per test session)
# ============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def src_dir(project_root: Path) -> Path:
    """Return the src directory."""
    return project_root / "src"


@pytest.fixture(scope="session")
def test_dir(project_root: Path) -> Path:
    """Return the tests directory."""
    return project_root / "tests"


@pytest.fixture(scope="session")
def fixtures_dir(test_dir: Path) -> Path:
    """Return the test fixtures directory."""
    fixtures = test_dir / "fixtures"
    fixtures.mkdir(exist_ok=True)
    return fixtures


# ============================================================================
# Function-level fixtures (run for each test function)
# ============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config_env(temp_dir: Path, monkeypatch) -> dict:
    """Set up a mock configuration environment with temporary paths.
    
    This fixture:
    - Creates temporary directories for data/models/results
    - Sets environment variables to point to these temp directories
    - Enables test mode flags (simplified KSS, reduced trials)
    
    Returns
    -------
    dict
        Dictionary containing all temporary paths
    """
    # Create temporary directory structure
    data_dir = temp_dir / "data"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"
    models_dir = temp_dir / "models"
    results_dir = temp_dir / "results"
    config_dir = temp_dir / "config"
    subjects_dir = config_dir / "subjects"
    
    for d in [interim_dir, processed_dir, models_dir, results_dir, subjects_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Create minimal subject list file
    subject_list_path = subjects_dir / "subject_list.txt"
    subject_list_path.write_text("S001\nS002\nS003\n")
    
    # Set environment variables for test mode
    env_vars = {
        "DDD_SUBJECT_LIST_PATH": str(subject_list_path),
        "DDD_MODEL_PKL_PATH": str(models_dir),
        "KSS_SIMPLIFIED": "1",  # Use simplified KSS labels (1-3, 8-9 only)
        "N_TRIALS_OVERRIDE": "2",  # Reduce Optuna trials for faster tests
        "TRAIN_RATIO": "0.4",  # Smaller ratios for faster tests
        "VAL_RATIO": "0.3",
        "TEST_RATIO": "0.3",
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return {
        "temp_dir": temp_dir,
        "data_dir": data_dir,
        "interim_dir": interim_dir,
        "processed_dir": processed_dir,
        "models_dir": models_dir,
        "results_dir": results_dir,
        "config_dir": config_dir,
        "subjects_dir": subjects_dir,
        "subject_list_path": subject_list_path,
    }


@pytest.fixture
def sample_eeg_data() -> pd.DataFrame:
    """Generate a small sample of EEG-like data for testing.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: subject_id, timestamp, Fp1, Fp2, F3, F4, KSS
    """
    np.random.seed(42)
    n_samples = 100
    
    data = {
        "subject_id": ["S001"] * n_samples,
        "timestamp": np.arange(n_samples),
        "Fp1": np.random.randn(n_samples) * 10,
        "Fp2": np.random.randn(n_samples) * 10,
        "F3": np.random.randn(n_samples) * 10,
        "F4": np.random.randn(n_samples) * 10,
        "C3": np.random.randn(n_samples) * 10,
        "C4": np.random.randn(n_samples) * 10,
        "KSS": np.random.choice([1, 2, 3, 8, 9], size=n_samples),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_simlsl_data() -> pd.DataFrame:
    """Generate a small sample of SIMLSL (vehicle) data for testing.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: subject_id, timestamp, lane_position, steering_angle, etc.
    """
    np.random.seed(42)
    n_samples = 100
    
    data = {
        "subject_id": ["S001"] * n_samples,
        "timestamp": np.arange(n_samples),
        "lane_position": np.random.randn(n_samples) * 0.5,
        "steering_angle": np.random.randn(n_samples) * 10,
        "speed": 60 + np.random.randn(n_samples) * 5,
        "KSS": np.random.choice([1, 2, 3, 8, 9], size=n_samples),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_feature_matrix() -> tuple[np.ndarray, np.ndarray]:
    """Generate a small feature matrix and labels for testing ML models.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X, y) where X is feature matrix (n_samples, n_features) and y is labels
    """
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    # Generate balanced labels
    y = np.array([0] * 100 + [1] * 100)
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]
    
    return X, y


# ============================================================================
# Skip conditions for conditional tests
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items based on markers and conditions."""
    # Skip GPU tests if no GPU available
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    
    for item in items:
        if "requires_gpu" in item.keywords:
            # Check if GPU is available (simplified check)
            try:
                import tensorflow as tf
                if not tf.config.list_physical_devices('GPU'):
                    item.add_marker(skip_gpu)
            except Exception:
                item.add_marker(skip_gpu)
