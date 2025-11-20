"""I/O Utility Functions for Data Loading and Saving in the DDD Project.

This module provides a collection of utility functions designed to handle data input
and output operations within the Driver Drowsiness Detection (DDD) project. It includes
functionalities for safely loading MATLAB (.mat) files, reading subject lists from text files,
saving processed data to CSV files, and batch loading of processed CSV data for multiple subjects.

Functions:
- `safe_load_mat()`: Safely loads MATLAB .mat files.
- `read_subject_list()`: Reads a list of all subject IDs.
- `read_subject_list_fold()`: Reads a list of subject IDs for a specific cross-validation fold.
- `read_train_subject_list()`: Reads a list of subject IDs designated for training.
- `read_train_subject_list_fold()`: Reads a list of subject IDs for training within a specific cross-validation fold.
- `save_csv()`: Saves a Pandas DataFrame to a CSV file in a structured directory.
- `load_subject_csvs()`: Loads processed CSV data for multiple subjects.
- `load_subjects_and_data()`: Loads subject list and combined dataset for evaluation.
- `load_model_and_scaler()`: Loads trained model, scaler, and feature selection files.
"""

from src.config import (
    SUBJECT_LIST_PATH,
    SUBJECT_LIST_PATH_TRAIN,
    SUBJECT_LIST_PATH_FOLD,
    DATASET_PATH,
    INTRIM_CSV_PATH,
    PROCESS_CSV_PATH,
    SCALING_FILTER,
    WAVELET_FILTER,
)

import os
import scipy.io
import logging
import pandas as pd
from typing import Tuple

def safe_load_mat(file_path: str):
    """
    Safely load a MATLAB ``.mat`` file.

    Attempts to load a MATLAB file and returns its contents. If the file is not found
    or another error occurs, logs the error and returns ``None``.

    Parameters
    ----------
    file_path : str
        Absolute path to the ``.mat`` file.

    Returns
    -------
    dict or None
        Parsed content of the ``.mat`` file if successful, otherwise ``None``.
    """
    try:
        return scipy.io.loadmat(file_path)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        return None

def read_subject_list() -> list:
    """
    Read the list of all subject IDs.

    Reads subject IDs from a predefined text file. Each line contains one subject ID.
    Empty lines are ignored.

    Returns
    -------
    list of str
        List of subject IDs, e.g. ``["S0210_1", "S0211_2"]``.
    """
    with open(SUBJECT_LIST_PATH, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def read_subject_list_fold(fold: int) -> list:
    """
    Read subject IDs for a specific cross-validation fold.

    Parameters
    ----------
    fold : int
        Fold index.

    Returns
    -------
    list of str
        List of subject IDs for the fold.
    """
    input_fp = os.path.join(SUBJECT_LIST_PATH_FOLD, f'subject_list_{fold}.txt')
    with open(input_fp, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def read_train_subject_list() -> list:
    """
    Read the list of subject IDs for training.

    Returns
    -------
    list of str
        List of training subject IDs.
    """
    with open(SUBJECT_LIST_PATH_TRAIN, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def read_train_subject_list_fold(fold: int) -> list:
    """
    Read subject IDs for training in a specific cross-validation fold.

    Parameters
    ----------
    fold : int
        Fold index.

    Returns
    -------
    list of str
        List of training subject IDs for the fold.
    """
    input_fp = os.path.join(SUBJECT_LIST_PATH_FOLD, f'subject_list_train_{fold}.txt')
    with open(input_fp, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def save_csv(df: pd.DataFrame, subject_id: str, version: str, feat: str, model_name: str) -> None:
    """
    Save a DataFrame as a CSV file in a structured directory hierarchy.

    Parameters
    ----------
    df : pandas.DataFrame
        Data to save.
    subject_id : str
        Subject identifier (e.g. ``"S0210"``).
    version : str
        Session/version identifier (e.g. ``"1"``).
    feat : str
        Feature type or processing stage (e.g. ``"eeg"``, ``"wavelet"``, ``"processed"``).
        If ``"processed"``, saved under ``PROCESS_CSV_PATH``; otherwise under ``INTRIM_CSV_PATH``.
    model_name : str
        Model architecture subdirectory (e.g. ``"common"``, ``"SvmA"``).

    Returns
    -------
    None
    """
    if feat == 'processed':
        output_fp = os.path.join(PROCESS_CSV_PATH, model_name, f'processed_{subject_id}_{version}.csv')
    else:
        output_fp = os.path.join(INTRIM_CSV_PATH, feat, model_name, f'{feat}_{subject_id}_{version}.csv')

    os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    df.to_csv(output_fp, index=False)
    logging.info(f"CSV file has been saved at: {output_fp.replace(os.sep, '/')}")

def load_subject_csvs(
    subject_list: list,
    model_name: str,
    add_subject_id: bool = False,
    base_path: str = None
) -> Tuple[pd.DataFrame, list]:
    """
    Load processed CSV files for a list of subjects.

    Parameters
    ----------
    subject_list : list of str
        List of subject IDs (e.g. ["S0120_2"]).
    model_name : str
        Model architecture directory (e.g. "Lstm", "common").
    add_subject_id : bool, default=False
        If True, add a subject_id column to each row.

    Returns
    -------
    tuple
        - pandas.DataFrame : Concatenated subject data (empty if none loaded).
        - list of str : Feature column names (empty if no data loaded).
    """
    dfs = []
    for subject in subject_list:
        parts = subject.split('_')
        if len(parts) != 2:
            logging.warning(f"Unexpected subject format: {subject}")
            continue
        subject_id, version = parts
        file_name = f'processed_{subject_id}_{version}.csv'
        if base_path is not None:
            file_path = os.path.join(base_path, file_name)
        elif model_name is not None:
            file_path = os.path.join(PROCESS_CSV_PATH, model_name, file_name)
        else:
            file_path = os.path.join(PROCESS_CSV_PATH, "common", file_name)
        try:
            df = pd.read_csv(file_path)
            if add_subject_id:
                df['subject_id'] = f"{subject_id}_{version}"
            dfs.append(df)
            logging.info(f"Loaded: {file_path}")
        except FileNotFoundError:
            logging.warning(f"File not found: {file_name}")

    df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    feature_columns = []
    if not df_all.empty:
        ranges = {
            "Lstm": ("steering_std_dev", "lane_offset_gaussian_smooth"),
            "SvmA": ("Steering_Range", "LongAcc_SampleEntropy"),
            "RF":   ("Steering_Range", "LongAcc_SampleEntropy"),
            "SvmW": ("SteeringWheel_DDD", "LaneOffset_AAA"),
        }
        exclude_cols = {"Timestamp", "subject_id"}

        if model_name in ranges:
            start_col, end_col = ranges[model_name]
            if start_col in df_all.columns and end_col in df_all.columns:
                feature_columns = df_all.loc[:, start_col:end_col].columns.tolist()
            else:
                logging.warning(
                    f"[{model_name}] Expected columns {start_col}–{end_col} not found. "
                    "Using all non-excluded columns."
                )
                feature_columns = [c for c in df_all.columns if c not in exclude_cols]
        else:
            feature_columns = [c for c in df_all.columns if c not in exclude_cols]

    return df_all, feature_columns


# ==========================================================
#  Evaluation-specific utility functions
# ==========================================================

import joblib
import pickle
from tensorflow.keras.models import load_model
from src.evaluation.models.lstm import AttentionLayer


def load_subjects_and_data(
    model_name: str,
    fold: int = 0,
    sample_size: int = None,
    seed: int = 42,
    subject_wise_split: bool = False
):
    """Load subject list and combined dataset for evaluation.

    Parameters
    ----------
    model_name : str
        Model name (e.g. "Lstm", "SvmA").
    fold : int
        Fold number for cross-validation.
    sample_size : int
        Number of subjects to evaluate (subset if provided).
    seed : int
        Random seed for reproducibility.
    subject_wise_split : bool
        Whether to include subject_id column for subject-wise split.

    Returns
    -------
    tuple
        (subjects, model_name, combined_data)
    """
    from src.utils.io.loaders import read_subject_list, read_subject_list_fold, load_subject_csvs
    import numpy as np

    if fold == 0:
        subjects = read_subject_list()
    else:
        subjects = read_subject_list_fold(fold)
    if sample_size:
        rng = np.random.default_rng(seed)
        subjects = rng.choice(subjects, size=min(sample_size, len(subjects)), replace=False).tolist()
        logging.info(f"Evaluating {len(subjects)} subjects: {subjects}")

    combined_data, feature_columns = load_subject_csvs(subjects, model_name, add_subject_id=subject_wise_split)
    return subjects, model_name, combined_data


def load_model_and_scaler(model_name: str, mode: str, tag: str, fold: int, jobid: str = None):
    """Load trained model, scaler, and feature selection files.

    Parameters
    ----------
    model_name : str
        Model name ("Lstm", "SvmA", etc.).
    mode : str
        Experiment mode ("pooled", "only_target", etc.).
    tag : str
        Optional tag for variant models.
    fold : int
        Fold index for CV evaluation.
    jobid : str, optional
        Job ID directory name for artifact lookup.

    Returns
    -------
    tuple
        (model_instance, scaler, selected_features)
    """
    import joblib
    import pickle
    import os
    import logging
    from tensorflow.keras.models import load_model
    from src.evaluation.models.lstm import AttentionLayer

    # --- Normalize PBS jobid ---
    if jobid is None:
        jobid = os.environ.get("PBS_JOBID", "local")
    if "." in jobid:
        jobid = jobid.split(".")[0]

    # --- Define model directory ---
    # Strip possible "[n]" suffix (some PBS_JOBID may propagate fold info)
    import re
    pure_jobid = re.sub(r"\[\d+\]$", "", jobid)
    model_dir = os.path.join("models", model_name, pure_jobid)

    # Fallback: use latest_job.txt if no explicit jobid folder exists
    if not os.path.exists(model_dir):
        latest_marker = os.path.join("models", model_name, "latest_job.txt")
        if os.path.exists(latest_marker):
            with open(latest_marker) as f:
                jobid = f.read().strip()
            model_dir = os.path.join("models", model_name, jobid)

    # --- Define file paths (new unified structure) ---
    if model_name == "Lstm":
        model_file = f"Lstm.pkl"
        scaler_file = f"scaler_Lstm.pkl"
        feature_file = f"selected_features_Lstm.pkl"
    else:
        model_file = f"{model_name}.pkl"
        scaler_file = f"scaler_{model_name}.pkl"
        feature_file = f"selected_features_{model_name}.pkl"

    # ============================================================
    # Improved hierarchical model loading (handles jobid[fold] dirs)
    # ============================================================
    import glob
    import re

    # base_dir = models/<model_name>/<jobid>
    base_dir = os.path.join("models", model_name, pure_jobid)

    # fold_dir = models/<model_name>/<jobid>/<jobid>[fold]
    fold_dir = os.path.join(base_dir, f"{jobid}[{fold}]") if fold else base_dir

    # --- Search patterns (recursive) ---
    # --- Search patterns (mode-aware) ---
    # Prefer exact mode match (e.g., RF_target_only_rank_dtw_mean_high_*.pkl)
    tag_key = tag.replace("rank_", "") if tag else ""

    exact_pattern = os.path.join(base_dir, "**", f"{model_name}_{mode}_rank_*{tag_key}*.pkl")
    model_matches = glob.glob(exact_pattern, recursive=True)

    # If no exact match found, fallback to any file that includes the same mode
    if not model_matches:
        fallback_pattern = os.path.join(base_dir, "**", f"{model_name}_{mode}_*.pkl")
        model_matches = glob.glob(fallback_pattern, recursive=True)

    if not model_matches:
        logging.error(f"[EVAL] No model file found for mode={mode}, tag={tag} in {base_dir}")
        return None, None, None

    model_matches.sort(key=os.path.getmtime, reverse=True)
    model_path = model_matches[0]
    logging.info(f"[EVAL] Found model file (exact mode match): {model_path}")

    # --- Locate corresponding scaler/feature files (if exist) ---
    scaler_pattern = os.path.join(base_dir, "**", f"scaler_{model_name}_*.pkl")
    feature_pattern = os.path.join(base_dir, "**", f"selected_features_{model_name}_*.pkl")
    scaler_matches = glob.glob(scaler_pattern, recursive=True)
    feature_matches = glob.glob(feature_pattern, recursive=True)

    scaler_path = scaler_matches[0] if scaler_matches else None
    feature_path = feature_matches[0] if feature_matches else None

    # --- Legacy fallback (for pre-refactor models/common/) ---
    if not os.path.exists(model_path):
        legacy_dir = os.path.join("models", "common")
        legacy_model_path = os.path.join(legacy_dir, f"{model_name}.pkl")
        if os.path.exists(legacy_model_path):
            model_path = legacy_model_path
            scaler_path = os.path.join(legacy_dir, f"scaler_{model_name}.pkl")
            feature_path = os.path.join(legacy_dir, f"selected_features_{model_name}.pkl")
            logging.warning(f"[EVAL] Fallback to legacy model path: {legacy_dir}")

    try:
        # --- Keras 3.x Compatibility: auto-detect .keras / .h5 ---
        if model_name == "Lstm":
            keras_path = os.path.join(model_dir, f"{model_name}.keras")
            h5_path    = os.path.join(model_dir, f"{model_name}.h5")

            if os.path.exists(keras_path):
                clf = load_model(keras_path, custom_objects={"AttentionLayer": AttentionLayer})
                logging.info(f"[EVAL] Loaded Keras model (.keras): {keras_path}")
            elif os.path.exists(h5_path):
                clf = load_model(h5_path, custom_objects={"AttentionLayer": AttentionLayer})
                logging.info(f"[EVAL] Loaded Keras model (.h5): {h5_path}")
            elif os.path.exists(model_path):
                # Fallback (legacy): allow .pkl if older version
                clf = joblib.load(model_path)
                logging.warning(f"[EVAL] Fallback to legacy .pkl model: {model_path}")
            else:
                logging.error(f"[EVAL] No valid model file found in {model_dir}")
                return None, None, None

        else:
            # Standard sklearn models
            clf = joblib.load(model_path)

    except Exception as e:
        logging.error(f"[EVAL] Failed to load model: {e}")
        return None, None, None

    # --- Load scaler (optional) ---
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            logging.warning(f"[EVAL] Failed to load scaler: {e}")

    # --- Load selected features (optional) ---
    features = []
    if feature_path and os.path.exists(feature_path):
        try:
            with open(feature_path, "rb") as f:
                features = pickle.load(f)
        except Exception as e:
            logging.warning(f"[EVAL] Failed to load feature list: {e}")

    logging.info(f"[EVAL] Loaded model/scaler/features from: {model_dir}")
    return clf, scaler, features
