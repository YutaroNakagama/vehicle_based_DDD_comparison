"""Feature Merging Utilities for Driver Drowsiness Detection.

This module provides functions for merging various intermediate feature CSVs
(e.g., EEG, wavelet, time-frequency features) into a single, time-aligned dataset.
It handles the loading of individual feature files, performs time-based merging,
and saves the combined data to the processed dataset directory. This is crucial
for creating a unified feature set for machine learning models.
"""

import os
import pandas as pd
import logging

from src.utils.io.loaders import save_csv
from src.config import INTRIM_CSV_PATH, PROCESS_CSV_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FEATURES_BY_MODEL = {
    'common': {
        "time_freq_domain": "Time (seconds)",
        "smooth_std_pe": "Timestamp",
        "wavelet": "Timestamp",
        # "perclos": "Timestamp_x",
        # "pupil": "Timestamp_2D",
        "eeg": "Timestamp"
    },
    'SvmA': {
        "time_freq_domain": "Time (seconds)",
        "eeg": "Timestamp"
    },
    'SvmW': {
        "wavelet": "Timestamp",
        "eeg": "Timestamp"
    },
    'Lstm': {
        "smooth_std_pe": "Timestamp",
        "eeg": "Timestamp"
    }
}


def load_feature_csv(feature: str, timestamp_col: str, model: str, subject_id: str, version: str) -> pd.DataFrame | None:
    """Loads a specific feature CSV file and standardizes its timestamp column to 'Timestamp'.

    This function constructs the expected file path for an interim feature CSV,
    loads it into a Pandas DataFrame, and renames the specified `timestamp_col`
    to a generic 'Timestamp' for consistent merging. It handles cases where the
    file might not exist.

    Args:
        feature (str): The name of the feature (e.g., 'eeg', 'wavelet', 'smooth_std_pe').
        timestamp_col (str): The original name of the timestamp column within the CSV file.
        model (str): The model type (e.g., 'common', 'SvmA', 'Lstm'), used to locate the file.
        subject_id (str): The subject's identifier (e.g., "S0210").
        version (str): The session or version identifier (e.g., "1").

    Returns:
        pd.DataFrame | None: The loaded DataFrame with a standardized 'Timestamp' column
                             if the file is found and loaded successfully; otherwise, None.
    """
    file_path = os.path.join(INTRIM_CSV_PATH, feature, model, f"{feature}_{subject_id}_{version}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df.rename(columns={timestamp_col: "Timestamp"})
    else:
        logging.warning(f"File not found: {file_path}")
        return None


def merge_features(features: dict, model: str, subject_id: str, version: str) -> pd.DataFrame:
    """Merges multiple feature DataFrames for a given subject based on timestamp alignment.

    This function iterates through a dictionary of features, loads each feature's
    CSV file, and performs a time-series merge (`pd.merge_asof`) to align them
    by their timestamps. The merging is done with a 'nearest' direction to find
    the closest timestamp match.

    Args:
        features (dict): A dictionary where keys are feature names (str) and values
                         are the original timestamp column names (str) within those feature CSVs.
        model (str): The model type (e.g., 'common', 'SvmA', etc.), used to locate feature files.
        subject_id (str): The subject's identifier (e.g., "S0210").
        version (str): The session or version identifier (e.g., "1").

    Returns:
        pd.DataFrame: A single, merged DataFrame containing all specified features,
                      aligned and sorted by the 'Timestamp' column. Returns an empty
                      DataFrame if no features could be loaded or merged.
    """
    merged_df = pd.DataFrame()
    for feature, timestamp_col in features.items():
        df = load_feature_csv(feature, timestamp_col, model, subject_id, version)
        if df is not None:
            if merged_df.empty:
                merged_df = df
            else:
                merged_df = pd.merge_asof(
                    merged_df.sort_values("Timestamp"),
                    df.sort_values("Timestamp"),
                    on="Timestamp",
                    direction="nearest"
                )
    return merged_df


def merge_process(subject: str, model: str) -> None:
    """Merges selected features for a given subject and model, and saves the combined data to disk.

    This is the main entry point for the merging process. It identifies the relevant
    features based on the specified model, loads and merges them, and then saves
    the resulting comprehensive dataset as a 'merged' CSV file in the processed data directory.

    Args:
        subject (str): The subject string in 'subjectID_version' format (e.g., 'S0120_2').
        model (str): The model type, used to select the specific set of features to merge.

    Returns:
        None: The function saves the merged data to a CSV file and does not return any value.
    """
    parts = subject.split('_')
    if len(parts) != 2:
        logging.error(f"Unexpected subject format: {subject}")
        return

    subject_id, version = parts
    features = FEATURES_BY_MODEL.get(model, {})

    merged_df = merge_features(features, model, subject_id, version)

    if not merged_df.empty:
        save_csv(merged_df, subject_id, version, 'merged', model)
        logging.info(f"Merged data saved for {subject_id}_{version} [{model}].")
    else:
        logging.warning(f"No data merged for {subject_id}_{version} [{model}].")


def combine_file(subject: str) -> list[pd.DataFrame] | None:
    """Loads a processed CSV file for a given subject (legacy function).

    This function is intended for loading previously processed and saved CSV files
    for a specific subject. It is marked as legacy, suggesting newer approaches
    might be preferred for data loading.

    Args:
        subject (str): The subject string in 'subjectID_version' format (e.g., 'S0120_2').

    Returns:
        list[pd.DataFrame] | None: A list containing a single Pandas DataFrame if the file
                                   is successfully loaded; otherwise, None if the file is
                                   not found or the subject format is incorrect.
    """
    dfs = []
    parts = subject.split('_')
    if len(parts) != 2:
        print(f"Unexpected subject format: {subject}")
        return None

    subject_id, version = parts
    file_name = f'processed_{subject_id}_{version}.csv'

    try:
        df = pd.read_csv(f'{PROCESS_CSV_PATH}/{file_name}')
        dfs.append(df)
        return dfs
    except FileNotFoundError:
        print(f"File not found: {file_name}")
        return None

