import scipy.io
import numpy as np
import pandas as pd
import os

# Parameters
window_size_seconds = 60  # Time window size in seconds
step_size_seconds = 60    # Step size in seconds

# Function to extract features
def extract_features(data, window_size, step_size):
    features = {'std_dev': [], 'pred_error': [], 'gaussian_smooth': []}
    
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data[start:start + window_size]

        # Standard Deviation
        std_dev = np.std(window)
        features['std_dev'].append(std_dev)

        # Predicted Error (using second-order Taylor approximation)
        if len(window) > 3:
            pred_value = window[-3] + 2 * (window[-2] - window[-3])
            pred_error = abs(pred_value - window[-1])
            features['pred_error'].append(pred_error)
        else:
            features['pred_error'].append(np.nan)

        # Gaussian Smoothed Value
        gaussian_weights = np.exp(-0.5 * (np.linspace(-1, 1, len(window)) ** 2))
        gaussian_weights /= gaussian_weights.sum()
        gaussian_smooth = np.sum(window * gaussian_weights)
        features['gaussian_smooth'].append(gaussian_smooth)
        
    return features

# Main function to process each MAT file and save results to CSV
def process_mat_file(file_path):
    # Load the MAT file
    data = scipy.io.loadmat(file_path)

    # Extract the relevant data from the loaded MAT file
    sim_lsl_data = data.get('SIM_lsl', np.array([]))

    # Check if the data structure is valid and contains the required rows
    if sim_lsl_data.size > 0 and sim_lsl_data.shape[0] >= 31:
        # Extract the time data and other relevant series
        time_data = sim_lsl_data[0]             # LSL Time
        steering_position = sim_lsl_data[29]    # Steering Wheel Position (rad)
        lat_acceleration = sim_lsl_data[19]     # Lateral Acceleration (m/s²)
        long_acceleration = sim_lsl_data[18]    # Longitudinal Acceleration (m/s²)
        lane_offset = sim_lsl_data[27]          # Lane Offset (m)

        # Calculate sample rate
        sample_rate = len(time_data) / (time_data[-1] - time_data[0])  # samples per second
        window_size = int(window_size_seconds * sample_rate)  # Window size in samples
        step_size = int(step_size_seconds * sample_rate)      # Step size in samples

        # Extract features for each data series
        steering_features = extract_features(steering_position, window_size, step_size)
        lat_acc_features = extract_features(lat_acceleration, window_size, step_size)
        long_acc_features = extract_features(long_acceleration, window_size, step_size)
        lane_offset_features = extract_features(lane_offset, window_size, step_size)

        # Combine all features into a DataFrame
        all_features = {
            'Timestamp': time_data[::step_size][:len(steering_features['std_dev'])],  # Resampled timestamps
            'steering_std_dev': steering_features['std_dev'],
            'steering_pred_error': steering_features['pred_error'],
            'steering_gaussian_smooth': steering_features['gaussian_smooth'],
            'lat_acc_std_dev': lat_acc_features['std_dev'],
            'lat_acc_pred_error': lat_acc_features['pred_error'],
            'lat_acc_gaussian_smooth': lat_acc_features['gaussian_smooth'],
            'long_acc_std_dev': long_acc_features['std_dev'],
            'long_acc_pred_error': long_acc_features['pred_error'],
            'long_acc_gaussian_smooth': long_acc_features['gaussian_smooth'],
            'lane_offset_std_dev': lane_offset_features['std_dev'],
            'lane_offset_pred_error': lane_offset_features['pred_error'],
            'lane_offset_gaussian_smooth': lane_offset_features['gaussian_smooth']
        }

        # Convert to DataFrame
        all_features_df = pd.DataFrame(all_features)

        # Define output CSV path based on input file name
        output_filename = f"./extracted_features_{os.path.basename(file_path).replace('.mat', '_with_60s_step.csv')}"
        all_features_df.to_csv(output_filename, index=False)
        print(f"Features saved to {output_filename}")
    else:
        print("The data structure is invalid or does not contain the required rows.")

# Example usage for two files
process_mat_file('../../../../dataset/Aygun2024/physio/S0120/SIMlsl_S0120_1.mat')
process_mat_file('../../../../dataset/Aygun2024/physio/S0139/SIMlsl_S0139_1.mat')

