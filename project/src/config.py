"""Configuration settings for data paths, model parameters, and sampling rates.

This module defines constants used throughout the driver drowsiness detection pipeline,
including:
- file paths to datasets and models
- data processing configuration
- sampling rates for physiological signals
- filters for wavelet transformation
"""

import numpy as np

# File paths
SUBJECT_LIST_PATH = '../../dataset/mdapbe/subject_list.txt'
"""Path to the file containing the list of all subject IDs."""
SUBJECT_LIST_PATH_TRAIN = '../../dataset/mdapbe/subject_list_train.txt'
"""Path to the file containing the list of subject IDs for training."""
SUBJECT_LIST_PATH_FOLD = '../../dataset/mdapbe/subject_folds'
"""Base path for subject fold definitions, used in cross-validation."""
DATASET_PATH = '../../dataset/mdapbe/physio'
"""Base path to the raw physiological data (e.g., SIMlsl, EEG)."""
INTRIM_CSV_PATH = './data/interim'
"""Path to store interim processed CSV files."""
PROCESS_CSV_PATH = './data/processed'
"""Path to store final processed CSV files ready for model training."""
MODEL_PKL_PATH = './model/'
"""Path to save trained machine learning models (e.g., .pkl files)."""
OUTPUT_SVG_PATH = './output/svg'
"""Path to save generated SVG plots and visualizations."""

# Choices for processing and modeling
DATA_PROCESS_CHOICES = ["SvmA", "SvmW", "Lstm", "common"]
"""List of available data processing strategies/models for preprocessing."""
MODEL_CHOICES = [
    "SvmA", "SvmW", "Lstm", "RF", "BalancedRF", "DecisionTree", "AdaBoost", "GradientBoosting",
    "XGBoost", "LightGBM", "CatBoost", "LogisticRegression", "SVM", "K-Nearest Neighbors", "MLP"
]
"""List of available machine learning models for training and evaluation."""


# Data process parameters
MODEL_WINDOW_CONFIG = {
    "common": {"window_sec": 3, "step_sec": 1.5},
    "SvmA":   {"window_sec": 3, "step_sec": 1.5},
    "SvmW":   {"window_sec": 3, "step_sec": 1.5},
    "Lstm":   {"window_sec": 3, "step_sec": 1.5},
}
"""
Configuration for data windowing, specifying window size and step size in seconds
for different processing models.
- 'window_sec': The duration of each data window.
- 'step_sec': The step size to move the window for overlapping windows.
"""

# Sampling rates
SAMPLE_RATE_SIMLSL = 60  # sample rate for simlsl
"""Sampling rate for SIMLSL (Simulated Lane-keeping System) data in Hz."""
SAMPLE_RATE_EEG = 500
"""Sampling rate for EEG (Electroencephalography) data in Hz."""

# KSS label
KSS_BIN_LABELS = [1,2,3,4,5,7,8,9]
"""Original Karolinska Sleepiness Scale (KSS) labels used in the dataset."""
KSS_LABEL_MAP = {1:0, 2:0, 3:0, 4:0, 5:1, 6:1, 7:1, 8:1, 9:1}
"""
Mapping of original KSS labels to binary drowsiness states:
- 0: Alert/Non-drowsy (KSS 1-5)
- 1: Drowsy (KSS 8-9)
"""

# Filters for EEG wavelet decomposition
SCALING_FILTER = np.array([0.48296, 0.83652, 0.22414, -0.12941])
"""Scaling filter coefficients for Discrete Wavelet Transform (DWT) of EEG signals."""
WAVELET_FILTER = np.array([-0.12941, -0.22414, 0.83652, -0.48296])
"""Wavelet filter coefficients for Discrete Wavelet Transform (DWT) of EEG signals."""
WAVELET_LEV = 3
"""Decomposition level for wavelet transformation of EEG signals."""

# Feature selection
TOP_K_FEATURES = 10
"""Number of top features to select during feature selection processes."""

# Optuna
N_TRIALS = 10
"""Number of trials for hyperparameter optimization using Optuna."""
