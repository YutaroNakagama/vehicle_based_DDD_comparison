"""I/O Utility Functions for Data Loading and Saving in the DDD Project.

This module provides a collection of utility functions designed to handle data input
and output operations within the Driver Drowsiness Detection (DDD) project. It includes
functionalities for safely loading MATLAB (.mat) files, reading subject lists from text files,
saving processed data to CSV files, determining model-specific directory names, and
batch loading of processed CSV data for multiple subjects.

Functions:
- `safe_load_mat()`: Safely loads MATLAB .mat files.
- `read_subject_list()`: Reads a list of all subject IDs.
- `read_subject_list_fold()`: Reads a list of subject IDs for a specific cross-validation fold.
- `read_train_subject_list()`: Reads a list of subject IDs designated for training.
- `read_train_subject_list_fold()`: Reads a list of subject IDs for training within a specific cross-validation fold.
- `save_csv()`: Saves a Pandas DataFrame to a CSV file in a structured directory.
- `get_model_type()`: Determines the appropriate model type directory name.
- `load_subject_csvs()`: Loads processed CSV data for multiple subjects.
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

def save_csv(df: pd.DataFrame, subject_id: str, version: str, feat: str, model: str) -> None:
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
    model : str
        Model type subdirectory (e.g. ``"common"``, ``"SvmA"``).

    Returns
    -------
    None
    """
    if feat == 'processed':
        output_fp = os.path.join(PROCESS_CSV_PATH, model, f'processed_{subject_id}_{version}.csv')
    else:
        output_fp = os.path.join(INTRIM_CSV_PATH, feat, model, f'{feat}_{subject_id}_{version}.csv')

    os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    df.to_csv(output_fp, index=False)
    logging.info(f"CSV file has been saved at: {output_fp.replace(os.sep, '/')}")


def get_model_type(model_name: str) -> str:
    """
    Map model name to its directory type.

    Parameters
    ----------
    model_name : str
        Model name (e.g. ``"Lstm"``, ``"SvmA"``, ``"RF"``).

    Returns
    -------
    str
        Directory name for the model type.
    """
    return model_name if model_name in {"SvmW", "SvmA", "Lstm"} else "common"

def load_subject_csvs(
    subject_list: list,
    model_type: str,
    add_subject_id: bool = False,
    base_path: str = None
) -> Tuple[pd.DataFrame, list]:
    """
    Load processed CSV files for a list of subjects.

    Parameters
    ----------
    subject_list : list of str
        List of subjects (e.g. ["S0120_2"]).
    model_type : str
        Model type directory (e.g. "Lstm", "common").
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
        # --- Determine file path with override ---
        if base_path is not None:
            # Explicitly use base_path (e.g., data/processed/common)
            file_path = os.path.join(base_path, file_name)
        elif model_type is not None:
            # Model-specific path (e.g., data/processed/Lstm)
            file_path = os.path.join(PROCESS_CSV_PATH, model_type, file_name)
        else:
            # Fallback to shared "common"
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

        if model_type in ranges:
            start_col, end_col = ranges[model_type]
            if start_col in df_all.columns and end_col in df_all.columns:
                feature_columns = df_all.loc[:, start_col:end_col].columns.tolist()
            else:
                logging.warning(
                    f"[{model_type}] Expected columns {start_col}–{end_col} not found. "
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
    model: str,
    fold: int = 0,
    sample_size: int = None,
    seed: int = 42,
    subject_wise_split: bool = False
):
    """Load subject list and combined dataset for evaluation.

    Parameters
    ----------
    model : str
        Model name (e.g. "Lstm", "SvmA").
    fold : int
        Fold number for cross-validation.
    sample_size : int
        Number of subjects to evaluate (subset if provided).
    seed : int
        Random seed for reproducibility.
    subject_wise_split : bool
        Whether to include subject_id column for subject-wise split.

    Notes
    -----
    This function is called by ``eval_pipeline``. All parameters have defaults.

    Returns
    -------
    tuple
        (subjects, model_type, combined_data)
    """
    from src.utils.io.loaders import read_subject_list, read_subject_list_fold, get_model_type, load_subject_csvs
    import numpy as np

    if fold == 0:
        subjects = read_subject_list()
    else:
        subjects = read_subject_list_fold(fold)

    if sample_size:
        rng = np.random.default_rng(seed)
        subjects = rng.choice(subjects, size=min(sample_size, len(subjects)), replace=False).tolist()
        logging.info(f"Evaluating {len(subjects)} subjects: {subjects}")

    model_type = get_model_type(model)
    combined_data, feature_columns = load_subject_csvs(subjects, model_type, add_subject_id=subject_wise_split)
    return subjects, model_type, combined_data


def load_model_and_scaler(model: str, model_type: str, mode: str, tag: str, fold: int, jobid: str = None):
    """Load trained model, scaler, and feature selection files.

    Parameters
    ----------
    model : str
        Model name ("Lstm", "SvmA", etc.).
    model_type : str
        Corresponding model directory type.
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
    from src.config import MODEL_PKL_PATH

    suffix = f"_{mode}" if mode else ""
    if tag:
        suffix += f"_{tag}"

    model_base = os.path.join(MODEL_PKL_PATH, model_type)
    if jobid:
        model_base = os.path.join(model_base, jobid)
        logging.info(f"[EVAL] Using model artifacts from job {jobid}: {model_base}")

    if model == "Lstm":
        model_file = f"lstm_model_fold{fold or 1}.keras"
        scaler_file = f"scaler_fold{fold or 1}.pkl"
        feature_file = "selected_features_Lstm.pkl"
    else:
        model_file = f"{model}.pkl"
        scaler_file = f"scaler_{model}.pkl"
        feature_file = f"selected_features_{model}.pkl"

    model_path = os.path.join(model_base, model_file)
    scaler_path = os.path.join(model_base, scaler_file)
    feature_path = os.path.join(model_base, feature_file)

    # --- Load model ---
    if not os.path.exists(model_path):
        logging.error(f"[EVAL] Model file not found: {model_path}")
        return None, None, None

    try:
        if model == "Lstm":
            clf = load_model(model_path, custom_objects={"AttentionLayer": AttentionLayer})
        else:
            clf = joblib.load(model_path)
    except Exception as e:
        logging.error(f"[EVAL] Failed to load model: {e}")
        return None, None, None

    # --- Load scaler (optional) ---
    scaler = None
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            logging.warning(f"[EVAL] Failed to load scaler: {e}")

    # --- Load selected features (optional) ---
    features = []
    if os.path.exists(feature_path):
        try:
            with open(feature_path, "rb") as f:
                features = pickle.load(f)
        except Exception as e:
            logging.warning(f"[EVAL] Failed to load feature list: {e}")

    logging.info(f"[EVAL] Loaded model from {model_path}")
    return clf, scaler, features
