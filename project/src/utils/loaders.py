# mat_loader.py: MATファイルの読み込みとエラーハンドリング

# Local application imports
from config import (
    SUBJECT_LIST_PATH, 
    DATASET_PATH, 
    INTRIM_CSV_PATH, 
    PROCESS_CSV_PATH, 
    WINDOW_SIZE_SEC, 
    STEP_SIZE_SEC,
    SCALING_FILTER,
    WAVELET_FILTER,
)

import scipy.io
import logging

def load_mat_file(file_path, key=None):
    """
    Safely loads a MAT file and optionally extracts a specific key.
    
    Args:
        file_path (str): Path to the MAT file.
        key (str, optional): Key to extract specific data. Defaults to None.

    Returns:
        dict or ndarray: Loaded data or specific key data.
    """
    try:
        data = scipy.io.loadmat(file_path)
        return data[key] if key else data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
    return None

def safe_load_mat(file_path):
    """
    Safely loads a MAT file and handles errors.

    Args:
        file_path (str): Path to the MAT file.

    Returns:
        dict or None: Loaded data if successful, None otherwise.
    """
    try:
        return scipy.io.loadmat(file_path)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        return None

def read_subject_list():
    # Read the list of files from the text file
    with open(SUBJECT_LIST_PATH, 'r') as file:
        return file.read().splitlines()

def save_csv(df, subject_id, version, feat):
    output_fp = f'{INTRIM_CSV_PATH}/{feat}/{feat}_{subject_id}_{version}.csv'
    df.to_csv(output_fp, index=False)
    logging.info(f"CSV file has been saved at: {output_fp}")
