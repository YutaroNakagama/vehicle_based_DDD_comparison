import os
import pandas as pd
import logging

from src.utils.io.loaders import save_csv

from src.config import (
    INTRIM_CSV_PATH, 
    PROCESS_CSV_PATH, 
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FEATURES_BY_MODEL = {
    'common': {
        "time_freq_domain": "Time (seconds)",
        "smooth_std_pe": "Timestamp",
        "wavelet": "Timestamp",
        #"perclos": "Timestamp_x",
        #"pupil": "Timestamp_2D",
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

def load_feature_csv(feature, timestamp_col, model, subject_id, version):
    """Helper function to load feature CSV and standardize timestamp."""
    file_path = os.path.join(INTRIM_CSV_PATH, feature, model, f"{feature}_{subject_id}_{version}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df.rename(columns={timestamp_col: "Timestamp"})
    else:
        logging.warning(f"File not found: {file_path}")
        return None

def merge_features(features, model, subject_id, version):
    """Merge multiple feature DataFrames on Timestamp."""
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

def merge_process(subject, model):
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
    features = FEATURES_BY_MODEL.get(model, {})

    merged_df = merge_features(features, model, subject_id, version)

    if not merged_df.empty:
        save_csv(merged_df, subject_id, version, 'merged', model)
        logging.info(f"Merged data saved for {subject_id}_{version} [{model}].")
    else:
        logging.warning(f"No data merged for {subject_id}_{version} [{model}].")

def combine_file(subject):
    dfs = []

    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
    file_name = f'processed_{subject_id}_{version}.csv'

    try:
        df = pd.read_csv(f'{PROCESS_CSV_PATH}/{file_name}')
        dfs.append(df)
        return dfs
    except FileNotFoundError:
        print(f"File not found: {file_name}")

