from src.config import (
    SUBJECT_LIST_PATH, 
    DATASET_PATH, 
    INTRIM_CSV_PATH, 
    PROCESS_CSV_PATH, 
    SCALING_FILTER,
    WAVELET_FILTER,
)

import scipy.io
import logging
import pandas as pd

def safe_load_mat(file_path):
    try:
        return scipy.io.loadmat(file_path)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        return None

def read_subject_list():
    with open(SUBJECT_LIST_PATH, 'r') as file:
        return file.read().splitlines()

def save_csv(df, subject_id, version, feat, model):
    if feat == 'processed':
        output_fp = f'{PROCESS_CSV_PATH}/{model}/processed_{subject_id}_{version}.csv'
    else:
        output_fp = f'{INTRIM_CSV_PATH}/{feat}/{model}/{feat}_{subject_id}_{version}.csv'

    df.to_csv(output_fp, index=False)
    logging.info(f"CSV file has been saved at: {output_fp}")


def get_model_type(model_name):
    return model_name if model_name in {"SvmW", "SvmA", "Lstm"} else "common"


def load_subject_csvs(subject_list, model_type, add_subject_id=False):
    dfs = []
    for subject in subject_list:
        subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
        file_name = f'processed_{subject_id}_{version}.csv'
        file_path = f'{PROCESS_CSV_PATH}/{model_type}/{file_name}'
        try:
            df = pd.read_csv(file_path)
            if add_subject_id:
                df['subject_id'] = subject_id
            dfs.append(df)
            logging.info(f"Loaded: {file_name}")
        except FileNotFoundError:
            logging.warning(f"File not found: {file_name}")
    return pd.concat(dfs, ignore_index=True)

