import pandas as pd
import os

# Path to the subject list
subject_list_path = '../../../../../dataset/Aygun2024/subject_list_temp.txt'

# Read the subject list
with open(subject_list_path, 'r') as f:
    subjects = [line.strip() for line in f.readlines()]

# Iterate over each subject in the list
for subject in subjects:
    # Define paths for the three CSV files for each subject
    base_path = subject.split('/')[0]  # e.g., "S0113"
    trial_id = subject.split('/')[1]   # e.g., "S0113_1"
    
    file_path_1 = f'./wang/extracted_features_SIMlsl_{trial_id}_with_60s_step.csv'
    file_path_2 = f'./ghm/32_Decomposed_Signals_with_Timestamps_SIMlsl_{trial_id}.csv'
    file_path_3 = f'./aref/{trial_id}_Combined_Features.csv'
    
    # Check if all files exist before proceeding
    if os.path.exists(file_path_1) and os.path.exists(file_path_2) and os.path.exists(file_path_3):
        # Load data from the CSV files
        extracted_features_df = pd.read_csv(file_path_1)
        decomposed_signals_df = pd.read_csv(file_path_2)
        combined_features_df = pd.read_csv(file_path_3)

        # Rename 'Time (seconds)' to 'Timestamp' in combined_features_df
        combined_features_df = combined_features_df.rename(columns={'Time (seconds)': 'Timestamp'})

        # Convert 'Timestamp' columns to integer values by truncating the decimal part
        extracted_features_df['Timestamp'] = extracted_features_df['Timestamp'].astype(int)
        decomposed_signals_df['Timestamp'] = decomposed_signals_df['Timestamp'].astype(int)
        combined_features_df['Timestamp'] = combined_features_df['Timestamp'].astype(int)

        # Sort data by 'Timestamp' for asof merge to work correctly
        extracted_features_df = extracted_features_df.sort_values(by='Timestamp')
        decomposed_signals_df = decomposed_signals_df.sort_values(by='Timestamp')
        combined_features_df = combined_features_df.sort_values(by='Timestamp')

        # Merge extracted_features_df and decomposed_signals_df within a 2-second tolerance
        merged_df = pd.merge_asof(extracted_features_df, decomposed_signals_df, on='Timestamp', tolerance=2, direction='nearest')

        # Merge the result with combined_features_df within a 2-second tolerance
        merged_df = pd.merge_asof(merged_df, combined_features_df, on='Timestamp', tolerance=2, direction='nearest')

        # Define output path and save merged data
        output_dir = './all_feat'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/merged_data_table_{trial_id}.csv'
        merged_df.to_csv(output_path, index=False)
        
        print(f"CSV file saved for {trial_id} at: {output_path}")
    else:
        print(f"Missing files for {trial_id}, skipping...")

