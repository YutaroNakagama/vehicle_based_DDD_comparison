"""Configuration settings for the DDD pipeline.

This module centralizes configuration parameters for driver drowsiness detection,
including dataset paths, model options, sampling rates, wavelet filters, and
hyperparameter tuning.

Notes
-----
All variables are defined as module-level constants. They are imported directly
by other modules in the pipeline (e.g., preprocessing, training, evaluation).
"""

import os
import numpy as np

# ---------------------------------------------------------------------
# File paths (from environment variables, fallback to defaults)
# ---------------------------------------------------------------------
SUBJECT_LIST_PATH = os.environ.get(
    "DDD_SUBJECT_LIST_PATH", "config/subjects/subject_list.txt"
)
"""str : Path to the file containing the list of all subject IDs."""

SUBJECT_LIST_PATH_TRAIN = os.environ.get(
    "DDD_SUBJECT_LIST_PATH_TRAIN", "../dataset/mdapbe/subject_list_train.txt"
)
"""str : Path to the file containing the list of subject IDs for training."""

SUBJECT_LIST_PATH_FOLD = os.environ.get(
    "DDD_SUBJECT_LIST_PATH_FOLD", "../dataset/mdapbe/subject_folds"
)
"""str : Base path for subject fold definitions, used in cross-validation."""

DATASET_PATH = os.environ.get(
    "DDD_DATASET_PATH", "../dataset/mdapbe/physio"
)
"""str : Base path to the raw physiological data (e.g., SIMlsl, EEG)."""

INTRIM_CSV_PATH = './data/interim'
"""str : Directory to store interim processed CSV files."""

PROCESS_CSV_PATH = './data/processed'
"""str : Directory to store final processed CSV files ready for model training."""

PROCESS_CSV_COMMON_PATH = './data/processed/common'
"""str : Directory for processed common feature CSVs (used across models)."""

MODEL_PKL_PATH = os.environ.get("DDD_MODEL_PKL_PATH", "./models/")
"""str : Directory to save trained machine learning models (e.g., `.pkl`)."""

RESULTS_PATH = './results'
"""str : Root directory for storing analysis results and evaluation metrics."""

RESULTS_EVALUATION_PATH = './results/evaluation'
"""str : Directory for evaluation metrics (JSON/CSV outputs)."""

RESULTS_DOMAIN_ANALYSIS_PATH = './results/domain_analysis'
"""str : Directory for domain analysis results (distances, rankings)."""

RESULTS_DOMAIN_GENERALIZATION_PATH = './results/domain_generalization'
"""str : Directory for domain generalization experiments and metrics."""

CONFIG_PATH = './config'
"""str : Directory containing configuration files (subject lists, target groups)."""

OUTPUT_SVG_PATH = './output/svg'
"""str : Directory to save generated SVG plots and visualizations."""

# ---------------------------------------------------------------------
# Common file name patterns
# ---------------------------------------------------------------------
LATEST_JOB_FILENAME = "latest_job.txt"
"""str : Filename for tracking the latest training job ID."""

RANK_NAMES_FILENAME = "rank_names.txt"
"""str : Filename for mapping tags to target subject groups."""

TARGET_GROUPS_FILENAME = "target_groups.txt"
"""str : Default filename for target subject group definitions."""

SUBJECT_LIST_FILENAME = "subject_list.txt"
"""str : Default filename for subject ID lists."""

# ---------------------------------------------------------------------
# File extensions
# ---------------------------------------------------------------------
MODEL_FILE_EXTENSION = ".pkl"
"""str : File extension for saved model artifacts."""

METRICS_FILE_EXTENSION_JSON = ".json"
"""str : File extension for JSON metric outputs."""

METRICS_FILE_EXTENSION_CSV = ".csv"
"""str : File extension for CSV metric outputs."""

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
KSS_BIN_LABELS = [1, 2, 3, 4, 5, 8, 9]
"""list of int : Original Karolinska Sleepiness Scale (KSS) labels in dataset."""

# Allow test mode override: simplified KSS mapping (1-3=0, 8-9=1)
if os.environ.get("KSS_SIMPLIFIED") == "1":
    KSS_LABEL_MAP = {1:0, 2:0, 3:0, 8:1, 9:1}
    """dict : TEST MODE - Simplified KSS mapping (1-3=Alert, 8-9=Drowsy)."""
else:
    KSS_LABEL_MAP = {1:0, 2:0, 3:0, 4:0, 5:0, 8:1, 9:1}
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
# Allow test mode override for faster training
N_TRIALS = int(os.environ.get("N_TRIALS_OVERRIDE", 50))
"""int : Number of trials for hyperparameter optimization using Optuna."""

OPTUNA_N_STARTUP_TRIALS = 5
"""int : Number of startup trials for Optuna's MedianPruner."""

OPTUNA_N_WARMUP_STEPS = 1
"""int : Number of warmup steps for Optuna's MedianPruner."""

OPTUNA_INTERVAL_STEPS = 1
"""int : Interval steps for Optuna's MedianPruner."""

# ---------------------------------------------------------------------
# Distance metrics and training modes
# ---------------------------------------------------------------------
DISTANCE_METRICS = ["mmd", "wasserstein", "dtw"]
"""list of str : Supported distance metrics for domain analysis."""

# Note: TRAINING_MODES includes dynamic scheme names based on TARGET_GROUP_SIZE
# These are computed after TARGET_GROUP_SIZE is defined below
TRAINING_MODES_BASE = ["source_only", "target_only", "finetune", "only_general"]
"""list of str : Base training modes (before adding target size-specific modes)."""

RANKING_LEVELS = ["high", "middle", "low"]
"""list of str : Ranking levels for subject group stratification."""

# ---------------------------------------------------------------------
# Target group sizes for domain generalization experiments
# ---------------------------------------------------------------------
TARGET_GROUP_SIZE = 29
"""int : Number of subjects in target domain groups (used in training schemes like 'only29')."""

SOURCE_GROUP_SIZE = 59
"""int : Number of subjects in source/general domain groups (complement of target group)."""

# Scheme naming based on target group size
TARGET_SCHEME_NAME = f"only{TARGET_GROUP_SIZE}"
"""str : Training scheme name for target-only experiments (e.g., 'only29')."""

SOURCE_SCHEME_NAME = f"only{SOURCE_GROUP_SIZE}"
"""str : Training scheme name for source-only experiments (e.g., 'only59')."""

# Construct complete training modes list
TRAINING_MODES = TRAINING_MODES_BASE + [TARGET_SCHEME_NAME]
"""list of str : Supported training modes for domain generalization experiments."""

# ---------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------
CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1", "auc", "auc_pr"]
"""list of str : Standard classification metrics tracked across experiments."""

# ---------------------------------------------------------------------
# Random seed for reproducibility
# ---------------------------------------------------------------------
DEFAULT_RANDOM_SEED = 42
"""int : Default random seed used across training, evaluation, and analysis pipelines."""

# ---------------------------------------------------------------------
# Data split ratios (train:val:test)
# ---------------------------------------------------------------------
# Allow test mode override for faster training (e.g., 0.4:0.3:0.3 instead of 0.6:0.2:0.2)
TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", 0.6))
"""float : Training set ratio (default: 0.6)."""

VAL_RATIO = float(os.environ.get("VAL_RATIO", 0.2))
"""float : Validation set ratio (default: 0.2)."""

TEST_RATIO = float(os.environ.get("TEST_RATIO", 0.2))
"""float : Test set ratio (default: 0.2)."""

# ---------------------------------------------------------------------
# Time-stratified splitting configuration
# ---------------------------------------------------------------------
TIME_STRATIFY_TOLERANCE = 0.02
"""float : Tolerance for positive class ratio in time-stratified split."""

TIME_STRATIFY_WINDOW = 0.10
"""float : Boundary search window fraction for time-stratified split."""

TIME_STRATIFY_MIN_CHUNK = 100
"""int : Minimum rows per split in time-stratified split."""

# ---------------------------------------------------------------------
# Logging format
# ---------------------------------------------------------------------
LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
"""str : Standard logging format string for all pipeline modules."""

# ---------------------------------------------------------------------
# Environment setup utilities
# ---------------------------------------------------------------------
def configure_blas_threads(n_threads: int = 1) -> None:
    """Configure BLAS/OpenMP thread limits for HPC environments.
    
    This prevents thread oversubscription in PBS/SLURM jobs and ensures
    reproducible single-threaded execution for CPU-bound operations.
    
    Parameters
    ----------
    n_threads : int, default=1
        Number of threads to use for BLAS operations.
    
    Notes
    -----
    Call this function early in your script (before importing numpy/sklearn).
    """
    thread_str = str(n_threads)
    os.environ["OMP_NUM_THREADS"] = thread_str
    os.environ["OPENBLAS_NUM_THREADS"] = thread_str
    os.environ["MKL_NUM_THREADS"] = thread_str
    os.environ["VECLIB_MAXIMUM_THREADS"] = thread_str
    os.environ["NUMEXPR_NUM_THREADS"] = thread_str
    os.environ["NUMBA_NUM_THREADS"] = thread_str
    os.environ["LIGHTGBM_NUM_THREADS"] = thread_str
    os.environ["XGBOOST_NUM_THREADS"] = thread_str
    os.environ["SKLEARN_NO_OPENMP"] = "1"

