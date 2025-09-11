"""Configuration settings for the DDD pipeline.

This module centralizes configuration parameters for driver drowsiness detection,
including dataset paths, model options, sampling rates, wavelet filters, and
hyperparameter tuning.

Notes
-----
All variables are defined as module-level constants. They are imported directly
by other modules in the pipeline (e.g., preprocessing, training, evaluation).
"""

import numpy as np

# ---------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------
SUBJECT_LIST_PATH = '../../dataset/mdapbe/subject_list.txt'
"""str : Path to the file containing the list of all subject IDs."""

SUBJECT_LIST_PATH_TRAIN = '../../dataset/mdapbe/subject_list_train.txt'
"""str : Path to the file containing the list of subject IDs for training."""

SUBJECT_LIST_PATH_FOLD = '../../dataset/mdapbe/subject_folds'
"""str : Base path for subject fold definitions, used in cross-validation."""

DATASET_PATH = '../../dataset/mdapbe/physio'
"""str : Base path to the raw physiological data (e.g., SIMlsl, EEG)."""

INTRIM_CSV_PATH = './data/interim'
"""str : Directory to store interim processed CSV files."""

PROCESS_CSV_PATH = './data/processed'
"""str : Directory to store final processed CSV files ready for model training."""

MODEL_PKL_PATH = './model/'
"""str : Directory to save trained machine learning models (e.g., `.pkl`)."""

OUTPUT_SVG_PATH = './output/svg'
"""str : Directory to save generated SVG plots and visualizations."""

# ---------------------------------------------------------------------
# Choices for processing and modeling
# ---------------------------------------------------------------------
DATA_PROCESS_CHOICES = ["SvmA", "SvmW", "Lstm", "common"]
"""list of str : Supported data processing strategies/models for preprocessing."""

MODEL_CHOICES = [
    "SvmA", "SvmW", "Lstm", "RF", "BalancedRF", "DecisionTree", "AdaBoost", "GradientBoosting",
    "XGBoost", "LightGBM", "CatBoost", "LogisticRegression", "SVM", "K-Nearest Neighbors", "MLP"
]
"""list of str : Supported machine learning models for training and evaluation."""

# ---------------------------------------------------------------------
# Data process parameters
# ---------------------------------------------------------------------
MODEL_WINDOW_CONFIG = {
    "common": {"window_sec": 3, "step_sec": 1.5},
    "SvmA":   {"window_sec": 3, "step_sec": 1.5},
    "SvmW":   {"window_sec": 3, "step_sec": 1.5},
    "Lstm":   {"window_sec": 3, "step_sec": 1.5},
}
"""dict : Windowing configuration (window length and step size in seconds)."""

# ---------------------------------------------------------------------
# Sampling rates
# ---------------------------------------------------------------------
SAMPLE_RATE_SIMLSL = 60
"""int : Sampling rate for SIMLSL (Simulated Lane-keeping System) data [Hz]."""

SAMPLE_RATE_EEG = 500
"""int : Sampling rate for EEG (Electroencephalography) data [Hz]."""

# ---------------------------------------------------------------------
# KSS labels
# ---------------------------------------------------------------------
KSS_BIN_LABELS = [1, 2, 3, 4, 5, 7, 8, 9]
"""list of int : Original Karolinska Sleepiness Scale (KSS) labels in dataset."""

KSS_LABEL_MAP = {1:0, 2:0, 3:0, 4:0, 5:1, 6:1, 7:1, 8:1, 9:1}
"""dict : Mapping of KSS labels to binary states (0=Alert, 1=Drowsy)."""

# ---------------------------------------------------------------------
# Filters for EEG wavelet decomposition
# ---------------------------------------------------------------------
SCALING_FILTER = np.array([0.48296, 0.83652, 0.22414, -0.12941])
"""np.ndarray : Scaling filter coefficients for Discrete Wavelet Transform (DWT)."""

WAVELET_FILTER = np.array([-0.12941, -0.22414, 0.83652, -0.48296])
"""np.ndarray : Wavelet filter coefficients for Discrete Wavelet Transform (DWT)."""

WAVELET_LEV = 3
"""int : Decomposition level for wavelet transformation of EEG signals."""

# ---------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------
TOP_K_FEATURES = 10
"""int : Number of top features to select during feature selection."""

# ---------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------
N_TRIALS = 100
"""int : Number of trials for hyperparameter optimization using Optuna."""

