import pandas as pd

# Load the provided CSV files with updated paths
file_path_1 = './wang/extracted_features_SIMlsl_S0113_1_with_60s_step.csv'
file_path_2 = './ghm/32_Decomposed_Signals_with_Timestamps_SIMlsl_S0113_1.csv'
file_path_3 = './aref/S0113_1_Combined_Features.csv'

# Read the data from each file
extracted_features_df = pd.read_csv(file_path_1)
decomposed_signals_df = pd.read_csv(file_path_2)
combined_features_df = pd.read_csv(file_path_3)

# Renaming the time columns to have a consistent name for merging
combined_features_df = combined_features_df.rename(columns={'Time (seconds)': 'Timestamp'})

# Convert the 'Timestamp' columns in each dataframe to integer values by truncating the decimal part
extracted_features_df['Timestamp'] = extracted_features_df['Timestamp'].astype(int)
decomposed_signals_df['Timestamp'] = decomposed_signals_df['Timestamp'].astype(int)
combined_features_df['Timestamp'] = combined_features_df['Timestamp'].astype(int)

# Merge the dataframes again on the modified 'Timestamp' column
merged_df = extracted_features_df.merge(decomposed_signals_df, on='Timestamp', how='inner')
merged_df = merged_df.merge(combined_features_df, on='Timestamp', how='inner')

# Save the merged DataFrame as a CSV file
output_path = './all_feat/merged_data_table.csv'
merged_df.to_csv(output_path, index=False)

# Output path for reference
print(f"CSV file saved at: {output_path}")

