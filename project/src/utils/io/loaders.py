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
    """Safely loads a MATLAB .mat file, handling potential FileNotFoundError or other exceptions.

    This function attempts to load a .mat file and logs an error if the file is not found
    or if any other exception occurs during loading, returning None in such cases.

    Args:
        file_path (str): The absolute path to the .mat file to be loaded.

    Returns:
        dict | None: The parsed content of the .mat file as a dictionary if successful,
                     otherwise returns None.
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
    """Reads the list of all subject IDs from a predefined text file.

    Each line in the text file is expected to contain a single subject ID.
    Empty lines are ignored.

    Returns:
        list[str]: A list of strings, where each string is a subject ID
                   (e.g., ["S0210_1", "S0211_2"]).
    """
    with open(SUBJECT_LIST_PATH, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def read_subject_list_fold(fold: int) -> list:
    """Reads the list of subject IDs for a specific cross-validation fold from a text file.

    The function constructs the file path based on the provided fold number.
    Each line in the text file is expected to contain a single subject ID.
    Empty lines are ignored.

    Args:
        fold (int): The fold number for which to read the subject list.

    Returns:
        list[str]: A list of strings, where each string is a subject ID
                   (e.g., ["S0210_1", "S0211_2"]).
    """
    input_fp = os.path.join(SUBJECT_LIST_PATH_FOLD, f'subject_list_{fold}.txt')
    with open(input_fp, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def read_train_subject_list() -> list:
    """Reads the list of subject IDs designated for training from a predefined text file.

    Each line in the text file is expected to contain a single subject ID.
    Empty lines are ignored.

    Returns:
        list[str]: A list of strings, where each string is a subject ID
                   (e.g., ["S0210_1", "S0211_2"]).
    """
    with open(SUBJECT_LIST_PATH_TRAIN, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def read_train_subject_list_fold(fold: int) -> list:
    """Reads the list of subject IDs for training within a specific cross-validation fold from a text file.

    The function constructs the file path based on the provided fold number.
    Each line in the text file is expected to contain a single subject ID.
    Empty lines are ignored.

    Args:
        fold (int): The fold number for which to read the training subject list.

    Returns:
        list[str]: A list of strings, where each string is a subject ID
                   (e.g., ["S0210_1", "S0211_2"]).
    """
    input_fp = os.path.join(SUBJECT_LIST_PATH_FOLD, f'subject_list_train_{fold}.txt')
    with open(input_fp, 'r') as file:
        return [line.strip() for line in file if line.strip()]

def save_csv(df: pd.DataFrame, subject_id: str, version: str, feat: str, model: str) -> None:
    """Saves a Pandas DataFrame to a CSV file within a structured directory hierarchy.

    The output path is determined by whether the data is 'processed' or an 'interim' feature,
    along with the model type, subject ID, and version. Directories are created if they don't exist.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be saved.
        subject_id (str): The subject's identifier (e.g., "S0210").
        version (str): The session or version identifier for the subject's data (e.g., "1").
        feat (str): The feature name or processing stage (e.g., 'eeg', 'wavelet', 'processed').
                    If 'processed', the file is saved in the `PROCESS_CSV_PATH`; otherwise, in `INTRIM_CSV_PATH`.
        model (str): The model type, used as a subdirectory name to organize saved files
                     (e.g., 'common', 'SvmA').

    Returns:
        None: The function saves the DataFrame to a CSV file and does not return any value.
    """
    if feat == 'processed':
        output_fp = os.path.join(PROCESS_CSV_PATH, model, f'processed_{subject_id}_{version}.csv')
    else:
        output_fp = os.path.join(INTRIM_CSV_PATH, feat, model, f'{feat}_{subject_id}_{version}.csv')

    os.makedirs(os.path.dirname(output_fp), exist_ok=True)
    df.to_csv(output_fp, index=False)
    logging.info(f"CSV file has been saved at: {output_fp.replace(os.sep, '/')}")


def get_model_type(model_name: str) -> str:
    """Determines the appropriate model type directory name based on a given model name.

    This function maps specific model names (e.g., 'Lstm', 'SvmA', 'SvmW') to their
    corresponding directory names used for organizing files. If the model name does
    not match a specific type, it defaults to 'common'.

    Args:
        model_name (str): The name of the model (e.g., 'Lstm', 'SvmA', 'RF').

    Returns:
        str: The corresponding model type directory name (e.g., 'Lstm', 'SvmA',
             'SvmW', or 'common').
    """
    return model_name if model_name in {"SvmW", "SvmA", "Lstm"} else "common"

def load_subject_csvs(subject_list: list, model_type: str, add_subject_id: bool = False) -> Tuple[pd.DataFrame, list]:
    """Loads processed CSV files for all subjects in the given list and concatenates them into a single DataFrame.

    This function iterates through a list of subject IDs, constructs the file path
    for their processed CSV data based on the model type, loads each CSV, and
    optionally adds a 'subject_id' column. It then concatenates all loaded DataFrames
    and extracts the names of the feature columns.

    Args:
        subject_list (list): A list of subject strings (e.g., ["S0120_2"]).
        model_type (str): The model type directory name (e.g., 'Lstm', 'common'),
                          indicating where the processed CSVs are located.
        add_subject_id (bool): If True, a 'subject_id' column (format: 'subjectID_version')
                               is added to each row of the loaded DataFrames. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, list]: A tuple containing:
            - pd.DataFrame: A concatenated DataFrame of all loaded subject data.
                            Returns an empty DataFrame if no files are loaded.
            - list: A list of strings representing the names of the feature columns
                    (from 'Steering_Range' to 'LaneOffset_AAA'). Returns an empty list
                    if the concatenated DataFrame is empty.
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
                df['subject_id'] = f"{subject_id}_{version}"
            dfs.append(df)
            logging.info(f"Loaded: {file_path}")
        except FileNotFoundError:
            logging.warning(f"File not found: {file_name}")
    
    df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if not df_all.empty:
        start_col = "Steering_Range"
        end_col = "LaneOffset_AAA"
        feature_columns = df_all.loc[:, start_col:end_col].columns.tolist()
    else:
        feature_columns = []
    return df_all, feature_columns

