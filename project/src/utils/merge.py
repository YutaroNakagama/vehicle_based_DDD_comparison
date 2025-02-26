from src.utils.loaders import save_csv

import os
import pandas as pd
import numpy as np
import logging
from scipy.signal import lfilter

# Local application imports
from src.config import (
    SUBJECT_LIST_PATH, 
    DATASET_PATH, 
    INTRIM_CSV_PATH, 
    PROCESS_CSV_PATH, 
    SCALING_FILTER,
    WAVELET_FILTER,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

dfs = []

def merge_process(subject):
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
    features = {
        "time_freq_domain": "Time (seconds)",
        "smooth_std_pe": "Timestamp",
        "wavelet": "Timestamp",
        #"perclos": "Timestamp_x",
        #"pupil": "Timestamp_2D",
        "eeg": "Timestamp"
    }

    # Initialize an empty DataFrame for merging results
    merged_df = pd.DataFrame()
    
    # Process each feature for the current subject
    for feature, timestamp_col in features.items():
        # Construct the file path for each feature CSV
        file_path = f"{INTRIM_CSV_PATH}/{feature}/{feature}_{subject_id}_{version}.csv"
        
        # Check if the file exists
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Rename the timestamp column to "Timestamp" for consistent merging
            df = df.rename(columns={timestamp_col: "Timestamp"})
            
            # Merge on "Timestamp" with nearest matching
            if merged_df.empty:
                merged_df = df
            else:
                merged_df = pd.merge_asof(merged_df.sort_values("Timestamp"),
                                          df.sort_values("Timestamp"),
                                          on="Timestamp",
                                          direction="nearest")
        else:
            #print(f"File not found: {file_path}")
            logging.warning(f"File not found: {file_path}")
    
    save_csv(merged_df, subject_id, version, 'merged') 
#    # Save the merged result for the current subject
#    output_file = f"{INTRIM_CSV_PATH}/{subject_id}_{version}_merged_data.csv"
#    merged_df.to_csv(output_file, index=False)
#    #print(f"Saved merged file for {subject_id}_{version} to {output_file}")
#    logging.info(f"Saved merged file for {subject_id}_{version} to {output_file}")

def combine_file(subject):
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
    #subject_id, version = subject
    file_name = f'processed_{subject_id}_{version}.csv'
    try:
        df = pd.read_csv(f'{PROCESS_CSV_PATH}/{file_name}')
        dfs.append(df)
        return dfs
    except FileNotFoundError:
        print(f"File not found: {file_name}")

