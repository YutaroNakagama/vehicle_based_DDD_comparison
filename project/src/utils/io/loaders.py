from src.config import (
    SUBJECT_LIST_PATH, 
    DATASET_PATH, 
    INTRIM_CSV_PATH, 
    PROCESS_CSV_PATH, 
    #WINDOW_SIZE_SEC, 
    #STEP_SIZE_SEC,
    SCALING_FILTER,
    WAVELET_FILTER,
)

import scipy.io
import logging

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
