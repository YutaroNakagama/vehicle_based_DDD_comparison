from src.utils.io.loaders import safe_load_mat, save_csv

# Third-party imports
import pandas as pd
import numpy as np
import logging
from scipy.interpolate import interp1d

# Local application imports
from src.config import (
    SUBJECT_LIST_PATH, 
    DATASET_PATH, 
    INTRIM_CSV_PATH, 
    PROCESS_CSV_PATH, 
    WINDOW_SIZE_SEC, 
    STEP_SIZE_SEC,
)

def calculate_and_save_perclos_physio_combined(blink_data_path, physio_data_path):
    """
    Load blink and physiological data from .mat files, calculate PERCLOS, resample physiological data,
    and save the combined result as a CSV file.
    """
    
    # Load and process PERCLOS data
    blink_data = safe_load_mat(blink_data_path)
    if blink_data is None:
        return
    blinks = blink_data['Blinks']
    blink_starts = blinks[:, 10]  # Start times
    blink_durations = blinks[:, 2]  # Duration (seconds)

    # Determine the total duration and number of windows
    total_duration = blink_starts[-1] - blink_starts[0]
    num_windows = int(total_duration // WINDOW_SIZE_SEC)

    # Calculate PERCLOS per window
    perclos_values = []
    timestamps = []
    start_time = blink_starts[0]
    
    for i in range(num_windows):
        window_start = start_time + i * WINDOW_SIZE_SEC
        mask = (blink_starts >= window_start) & (blink_starts < window_start + WINDOW_SIZE_SEC)
        total_blink_duration = blink_durations[mask].sum()
        perclos = total_blink_duration / WINDOW_SIZE_SEC
        perclos_values.append(perclos)
        timestamps.append(window_start)
    
    perclos_df = pd.DataFrame({"Timestamp": timestamps, "PERCLOS": perclos_values})
    perclos_df['Normalized_Timestamp'] = perclos_df['Timestamp'] - perclos_df['Timestamp'].min()
    perclos_df['Window'] = (perclos_df['Normalized_Timestamp'] // WINDOW_SIZE_SEC).astype(int)

    # Load and process Physio data
    physio_data = safe_load_mat(physio_data_path)
    if physio_data is None:
        return  # エラー発生時にスキップ
    physio_content = physio_data['PhysioData']
    physio_df = pd.DataFrame({
        "Timestamp": physio_content[0],
        "Blood_Pressure": physio_content[1],
        "Skin_Conductance": physio_content[2],
        "Respiration": physio_content[3],
        "Oxygen_Saturation": physio_content[4]
    })
    
    # Normalize timestamps for alignment and create time windows for resampling
    physio_df['Normalized_Timestamp'] = physio_df['Timestamp'] - physio_df['Timestamp'].min()
    physio_df['Window'] = (physio_df['Normalized_Timestamp'] // WINDOW_SIZE_SEC).astype(int)
    physio_resampled = physio_df.groupby('Window').mean().reset_index()

    # Merge PERCLOS and resampled Physio data on the 'Window' key
    merged_df = pd.merge(physio_resampled, perclos_df, on="Window", how="inner")

    # Save the combined data to CSV
    return merged_df

# Update the function to use the correct file naming convention for Blink and Physio files
def process_all_subjects(subject_list_path, base_path, output_base_path):
    """
    Process all subjects' Blink and Physio files listed in the subject list.
    For each subject, calculate PERCLOS, resample physiological data, and save the results as CSV.
    """
    # Read the subject list file
    with open(subject_list_path, 'r') as f:
        subject_list = f.read().splitlines()

    for subject in subject_list:
        # Extract subject-specific identifiers
        subject_name, version = subject.split('/')
        
        # Use corrected file naming convention for Blink and Physio files
        blink_data_file = f"{base_path}/{subject_name}/Blinks_{version}.mat"
        physio_data_file = f"{base_path}/{subject_name}/Physio_{version}.mat"
        output_csv_combined_file = f"{output_base_path}/perclos_physio_{version}.csv"

        # Check if both required files exist for the subject, proceed if they do
        try:
            # Calculate and save PERCLOS and Physio combined data for each subject
            calculate_and_save_perclos_physio_combined(blink_data_file, physio_data_file, output_csv_combined_file)
            #print(f"Processed and saved data for {subject_name} {version}")
            logging.info(f"Processed and saved data for {subject_name} {version}")
        except FileNotFoundError:
            #print(f"Missing files for {subject_name} {version}, skipping...")
            logging.warning(f"Missing files for {subject_name} {version}, skipping...")
        except Exception as e:
            #print(f"Error processing {subject_name} {version}: {e}")
            logging.error(f"Error processing {subject_name} {version}: {e}")

# Update the function to use the correct file naming convention for Blink and Physio files
def perclos_process(subject):
    """
    Process all subjects' Blink and Physio files listed in the subject list.
    For each subject, calculate PERCLOS, resample physiological data, and save the results as CSV.
    """
    # Extract subject-specific identifiers
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
    #subject_name, version = subject.split('/')
    
    # Use corrected file naming convention for Blink and Physio files
    blink_data_file = f"{DATASET_PATH}/{subject_id}/Blinks_{subject_id}_{version}.mat"
    physio_data_file = f"{DATASET_PATH}/{subject_id}/Physio_{subject_id}_{version}.mat"
    #output_csv_combined_file = f"{INTRIM_CSV_PATH}/perclos_physio_{subject_id}_{version}.csv"

    df = calculate_and_save_perclos_physio_combined(blink_data_file, physio_data_file)
    save_csv(df, subject_id, version, 'perclos')

# --------------------------------------------------------------------------- #
#                               pupil size                                    #
# --------------------------------------------------------------------------- #

# Define helper functions for outlier replacement and non-overlapping average calculation
def replace_outliers_with_interpolation(data, threshold=3):
    mean = np.nanmean(data)
    std_dev = np.nanstd(data)
    lower_bound = mean - threshold * std_dev
    upper_bound = mean + threshold * std_dev
    outliers = (data < lower_bound) | (data > upper_bound)
    data[outliers] = np.nan
    not_nan_indices = np.where(~np.isnan(data))[0]
    interp_func = interp1d(not_nan_indices, data[not_nan_indices], bounds_error=False, fill_value="extrapolate")
    return interp_func(np.arange(len(data)))

def non_overlapping_average(data, window_size):
    num_windows = len(data) // window_size
    avg_data = [np.mean(data[i*window_size:(i+1)*window_size]) for i in range(num_windows)]
    return np.array(avg_data)

# Main processing function for each subject's MAT file
def process_pupil_data(subject_id, version, threshold=3):
    #base_path = '../data/Aygun2024/physio'
    file_path = f'{DATASET_PATH}/{subject_id}/PupilData_{subject_id}_{version}.mat'
    data = safe_load_mat(file_path)
    if data is None:
        return  # エラー発生時にスキップ

    timestamps_2d = data['Pupil2D'][0]
    left_pupil_2d = data['Pupil2D'][1]
    right_pupil_2d = data['Pupil2D'][2]
    timestamps_3d = data['Pupil3D'][0]
    left_pupil_3d = data['Pupil3D'][1]
    right_pupil_3d = data['Pupil3D'][2]

    sampling_rate_2d = len(timestamps_2d) / (timestamps_2d[-1] - timestamps_2d[0])
    sampling_rate_3d = len(timestamps_3d) / (timestamps_3d[-1] - timestamps_3d[0])
    
    # Calculate non-overlapping averages for 2D pupil data
    window_samples_2d = int(WINDOW_SIZE_SEC * sampling_rate_2d)
    left_pupil_2d_cleaned = replace_outliers_with_interpolation(left_pupil_2d, threshold)
    right_pupil_2d_cleaned = replace_outliers_with_interpolation(right_pupil_2d, threshold)
    left_pupil_2d_avg = non_overlapping_average(left_pupil_2d_cleaned, window_samples_2d)
    right_pupil_2d_avg = non_overlapping_average(right_pupil_2d_cleaned, window_samples_2d)
    timestamps_2d_avg = timestamps_2d[::window_samples_2d][:len(left_pupil_2d_avg)]
    
    # Calculate non-overlapping averages for 3D pupil data
    window_samples_3d = int(WINDOW_SIZE_SEC * sampling_rate_3d)
    left_pupil_3d_cleaned = replace_outliers_with_interpolation(left_pupil_3d, threshold)
    right_pupil_3d_cleaned = replace_outliers_with_interpolation(right_pupil_3d, threshold)
    left_pupil_3d_avg = non_overlapping_average(left_pupil_3d_cleaned, window_samples_3d)
    right_pupil_3d_avg = non_overlapping_average(right_pupil_3d_cleaned, window_samples_3d)
    timestamps_3d_avg = timestamps_3d[::window_samples_3d][:len(left_pupil_3d_avg)]
    
    # Create DataFrame for the subject's processed data
    return pd.DataFrame({
        "Timestamp_2D": timestamps_2d_avg,
        "Left_Pupil_2D_Avg": left_pupil_2d_avg,
        "Right_Pupil_2D_Avg": right_pupil_2d_avg,
        "Timestamp_3D": timestamps_3d_avg,
        "Left_Pupil_3D_Avg": left_pupil_3d_avg,
        "Right_Pupil_3D_Avg": right_pupil_3d_avg
    })


def pupil_process(subject):
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
    
    # Process each subject's data and save to an individual CSV file
    subject_df = process_pupil_data(subject_id, version)
    save_csv(subject_df, subject_id, version, 'pupil')
    
