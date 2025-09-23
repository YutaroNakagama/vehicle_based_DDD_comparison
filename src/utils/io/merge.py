"""Feature Merging Utilities for Driver Drowsiness Detection.

This module provides functions for merging various intermediate feature CSVs
(e.g., EEG, wavelet, time-frequency features) into a single, time-aligned dataset.
It handles the loading of individual feature files, performs time-based merging,
and saves the combined data to the processed dataset directory. This is crucial
for creating a unified feature set for machine learning models.
"""

from __future__ import annotations   

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
    """
    Load a feature CSV and standardize its timestamp column.

    Parameters
    ----------
    feature : str
        Feature name (e.g., ``"eeg"``, ``"wavelet"``, ``"smooth_std_pe"``).
    timestamp_col : str
        Name of the timestamp column in the original CSV.
    model : str
        Model type (e.g., ``"common"``, ``"SvmA"``, ``"Lstm"``).
    subject_id : str
        Subject identifier (e.g., ``"S0210"``).
    version : str
        Session or version identifier (e.g., ``"1"``).

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with standardized ``Timestamp`` column if the file is found,
        otherwise ``None``.
    """
    file_path = os.path.join(INTRIM_CSV_PATH, feature, model, f"{feature}_{subject_id}_{version}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df.rename(columns={timestamp_col: "Timestamp"})
    else:
        logging.warning(f"File not found: {file_path}")
        return None


def merge_features(features: dict, model: str, subject_id: str, version: str) -> pd.DataFrame:
    """
    Merge multiple feature DataFrames for a subject based on timestamp alignment.

    Parameters
    ----------
    features : dict
        Mapping of feature names to their timestamp column names.
    model : str
        Model type (e.g., ``"common"``, ``"SvmA"``).
    subject_id : str
        Subject identifier (e.g., ``"S0210"``).
    version : str
        Session or version identifier (e.g., ``"1"``).

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame aligned by ``Timestamp``.  
        Returns an empty DataFrame if no features are loaded.
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
    """
    Merge and save features for a given subject and model.

    Parameters
    ----------
    subject : str
        Subject string in the format ``"subjectID_version"`` (e.g., ``"S0120_2"``).
    model : str
        Model type determining which features to merge.

    Returns
    -------
    None
        The merged dataset is saved as CSV.
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
    """
    Load a processed CSV for a subject (legacy function).

    Parameters
    ----------
    subject : str
        Subject string in the format ``"subjectID_version"``.

    Returns
    -------
    list of pandas.DataFrame or None
        List containing the loaded DataFrame if found, otherwise ``None``.
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

