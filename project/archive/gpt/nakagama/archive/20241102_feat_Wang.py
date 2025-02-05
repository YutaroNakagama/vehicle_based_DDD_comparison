import scipy.io
import numpy as np
import pandas as pd

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

# Load the MAT file
file_path = '../../../../dataset/Aygun2024/physio/S0120/SIMlsl_S0120_1.mat'
data_new = scipy.io.loadmat(file_path)

# Extract the relevant data from the loaded MAT file
sim_lsl_data_new = data_new.get('SIM_lsl', np.array([]))

# Check if the data structure is valid and contains the required rows
if sim_lsl_data_new.size > 0 and sim_lsl_data_new.shape[0] >= 31:
    # Extract the time data and other relevant series
    time_data_new = sim_lsl_data_new[0]            # LSL Time
    steering_position_new = sim_lsl_data_new[29]   # Steering Wheel Position (rad)
    lat_acceleration_new = sim_lsl_data_new[19]    # Lateral Acceleration (m/s²)
    long_acceleration_new = sim_lsl_data_new[18]   # Longitudinal Acceleration (m/s²)
    lane_offset_new = sim_lsl_data_new[27]         # Lane Offset (m)

    # Calculate sample rate
    sample_rate_new = len(time_data_new) / (time_data_new[-1] - time_data_new[0])  # samples per second
    window_size_new = int(window_size_seconds * sample_rate_new)  # Window size in samples
    step_size_new = int(step_size_seconds * sample_rate_new)      # Step size in samples

    # Extract features for each data series
    steering_features_new = extract_features(steering_position_new, window_size_new, step_size_new)
    lat_acc_features_new = extract_features(lat_acceleration_new, window_size_new, step_size_new)
    long_acc_features_new = extract_features(long_acceleration_new, window_size_new, step_size_new)
    lane_offset_features_new = extract_features(lane_offset_new, window_size_new, step_size_new)

    # Combine all features into a DataFrame
    all_features_new = {
        'Timestamp': time_data_new[::step_size_new][:len(steering_features_new['std_dev'])],  # Resampled timestamps
        'steering_std_dev': steering_features_new['std_dev'],
        'steering_pred_error': steering_features_new['pred_error'],
        'steering_gaussian_smooth': steering_features_new['gaussian_smooth'],
        'lat_acc_std_dev': lat_acc_features_new['std_dev'],
        'lat_acc_pred_error': lat_acc_features_new['pred_error'],
        'lat_acc_gaussian_smooth': lat_acc_features_new['gaussian_smooth'],
        'long_acc_std_dev': long_acc_features_new['std_dev'],
        'long_acc_pred_error': long_acc_features_new['pred_error'],
        'long_acc_gaussian_smooth': long_acc_features_new['gaussian_smooth'],
        'lane_offset_std_dev': lane_offset_features_new['std_dev'],
        'lane_offset_pred_error': lane_offset_features_new['pred_error'],
        'lane_offset_gaussian_smooth': lane_offset_features_new['gaussian_smooth']
    }

    # Convert to DataFrame
    all_features_df_new = pd.DataFrame(all_features_new)

    # Save to CSV
    output_path_new = './extracted_features_SIMlsl_S0120_1_with_60s_step.csv'
    all_features_df_new.to_csv(output_path_new, index=False)
    print(f"Features saved to {output_path_new}")
else:
    print("The data structure is invalid or does not contain the required rows.")

