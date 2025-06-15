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
SUBJECT_LIST_PATH       = '../../dataset/mdapbe/subject_list.txt'
SUBJECT_LIST_PATH_TRAIN = '../../dataset/mdapbe/subject_list_train.txt'
SUBJECT_LIST_PATH_FOLD  = '../../dataset/mdapbe/subject_folds'
DATASET_PATH            = '../../dataset/mdapbe/physio'
INTRIM_CSV_PATH         = './data/interim'
PROCESS_CSV_PATH        = './data/processed'
MODEL_PKL_PATH          = './model/'
OUTPUT_SVG_PATH         = './output/svg'

# Choices for processing and modeling
DATA_PROCESS_CHOICES = ["SvmA", "SvmW", "Lstm", "common"]
MODEL_CHOICES = [
    "SvmA", "SvmW", "Lstm", "RF", "BalancedRF", "DecisionTree", "AdaBoost", "GradientBoosting",
    "XGBoost", "LightGBM", "CatBoost", "LogisticRegression", "SVM", "K-Nearest Neighbors", "MLP"
]


# Data process parameters
MODEL_WINDOW_CONFIG = {
    "common": {"window_sec": 3, "step_sec": 1.5},
    "SvmA":   {"window_sec": 3, "step_sec": 1.5},
    "SvmW":   {"window_sec": 3, "step_sec": 1.5},
    "Lstm":   {"window_sec": 3, "step_sec": 1.5},
}

# Sampling rates
SAMPLE_RATE_SIMLSL = 60  # sample rate for simlsl
SAMPLE_RATE_EEG = 500

# KSS label
KSS_BIN_LABELS = [1, 8, 9]
KSS_LABEL_MAP = {1: 0, 8: 1, 9: 1}

# Filters for EEG wavelet decomposition
SCALING_FILTER = np.array([0.48296, 0.83652, 0.22414, -0.12941])
WAVELET_FILTER = np.array([-0.12941, -0.22414, 0.83652, -0.48296])
WAVELET_LEV = 3

# Feature selection
TOP_K_FEATURES = 10

# Optuna 
N_TRIALS = 20
