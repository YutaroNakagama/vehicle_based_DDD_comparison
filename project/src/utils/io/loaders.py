"""I/O utility functions for data loading and saving in the DDD project.

Includes:
- Safe .mat file loading
- Subject list loading
- CSV saving for feature files
- Model type resolution
- Batch CSV loading for subject list
"""

from src.config import (
    SUBJECT_LIST_PATH,
    SUBJECT_LIST_PATH_TRAIN,
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


def safe_load_mat(file_path: str):
    """Safely load a MATLAB .mat file.

    Args:
        file_path (str): Path to the .mat file.

    Returns:
        dict or None: Parsed content if successful, else None.
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
    """Read the list of subject IDs from a text file.

    Returns:
        list: List of subject strings, e.g., ["S0210_1", "S0211_2"]
    """
    with open(SUBJECT_LIST_PATH, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def read_train_subject_list() -> list:
    """Read the list of subject IDs from a text file.

    Returns:
        list: List of subject strings, e.g., ["S0210_1", "S0211_2"]
    """
    with open(SUBJECT_LIST_PATH_TRAIN, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def save_csv(df: pd.DataFrame, subject_id: str, version: str, feat: str, model: str) -> None:
    """Save a DataFrame to CSV in the interim or processed directory.

    Args:
        df (pd.DataFrame): Data to save.
        subject_id (str): Subject ID (e.g., "S0210").
        version (str): Session/version ID (e.g., "1").
        feat (str): Feature name ('eeg', 'wavelet', 'processed', etc.).
        model (str): Model type ('common', 'SvmA', etc.).

    Returns:
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
    """Determine the model type directory name based on the model name.

    Args:
        model_name (str): Name of the model (e.g., 'Lstm', 'SvmA', 'RF').

    Returns:
        str: Corresponding type name ('Lstm', 'SvmA', 'SvmW', or 'common').
    """
    return model_name if model_name in {"SvmW", "SvmA", "Lstm"} else "common"

def load_subject_csvs(subject_list: list, model_type: str, add_subject_id: bool = False) -> pd.DataFrame:
    """Load processed CSV files for all subjects in the given list.

    Args:
        subject_list (list): List of subject strings (e.g., ["S0120_2"]).
        model_type (str): Model type directory (e.g., 'Lstm', 'common').
        add_subject_id (bool): If True, adds a 'subject_id' column to each row.

    Returns:
        pd.DataFrame: Concatenated DataFrame of all loaded subjects.
    """
    dfs = []
    for subject in subject_list:
        # subject: e.g., 'S0120_2'
        parts = subject.split('_')
        if len(parts) != 2:
            logging.warning(f"Unexpected subject format: {subject}")
            continue
        subject_id, version = parts
        file_name = f'processed_{subject_id}_{version}.csv'
        file_path = os.path.join(PROCESS_CSV_PATH, model_type, file_name)
        try:
            df = pd.read_csv(file_path)
            if add_subject_id:
                df['subject_id'] = subject_id
            dfs.append(df)
            logging.info(f"Loaded: {file_path}")
        except FileNotFoundError:
            logging.warning(f"File not found: {file_name}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

