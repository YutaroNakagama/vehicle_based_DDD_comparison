import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import ttest_ind, pearsonr

# テキストファイルから被験者リストを読み込む
with open('subject_list_temp.txt', 'r') as f:
    lines = f.read().splitlines()

# 被験者IDとバージョンからCSVファイルパスを生成
file_paths = [
    f"{subject_id}_{version}_merged_data_with_KSS.csv"
    for line in lines
    for subject_id, version in [line.split('/')]
]

# 生体情報と時間の情報に関する列
biosignal_and_time_columns = [
    'Blood_Pressure', 'Skin_Conductance', 'Respiration', 'Oxygen_Saturation',
    'PERCLOS', 'Left_Pupil_2D_Avg', 'Right_Pupil_2D_Avg', 'Left_Pupil_3D_Avg', 
    'Right_Pupil_3D_Avg', 'Channel_1_Delta (0.5-4 Hz)', 'Channel_1_Theta (4-8 Hz)',
    'Channel_1_Alpha (8-13 Hz)', 'Channel_1_Beta (13-30 Hz)', 'Channel_1_Gamma (30-100 Hz)',
    'Channel_2_Delta (0.5-4 Hz)', 'Channel_2_Theta (4-8 Hz)', 'Channel_2_Alpha (8-13 Hz)',
    'Channel_2_Beta (13-30 Hz)', 'Channel_2_Gamma (30-100 Hz)', 'Channel_3_Delta (0.5-4 Hz)',
    'Channel_3_Theta (4-8 Hz)', 'Channel_3_Alpha (8-13 Hz)', 'Channel_3_Beta (13-30 Hz)',
    'Channel_3_Gamma (30-100 Hz)', 'Channel_4_Delta (0.5-4 Hz)', 'Channel_4_Theta (4-8 Hz)',
    'Channel_4_Alpha (8-13 Hz)', 'Channel_4_Beta (13-30 Hz)', 'Channel_4_Gamma (30-100 Hz)',
    'Channel_5_Delta (0.5-4 Hz)', 'Channel_5_Theta (4-8 Hz)', 'Channel_5_Alpha (8-13 Hz)',
    'Channel_5_Beta (13-30 Hz)', 'Channel_5_Gamma (30-100 Hz)', 'Channel_6_Delta (0.5-4 Hz)',
    'Channel_6_Theta (4-8 Hz)', 'Channel_6_Alpha (8-13 Hz)', 'Channel_6_Beta (13-30 Hz)',
    'Channel_6_Gamma (30-100 Hz)', 'Channel_7_Delta (0.5-4 Hz)', 'Channel_7_Theta (4-8 Hz)',
    'Channel_7_Alpha (8-13 Hz)', 'Channel_7_Beta (13-30 Hz)', 'Channel_7_Gamma (30-100 Hz)',
    'Channel_8_Delta (0.5-4 Hz)', 'Channel_8_Theta (4-8 Hz)', 'Channel_8_Alpha (8-13 Hz)',
    'Channel_8_Beta (13-30 Hz)', 'Channel_8_Gamma (30-100 Hz)', 'Timestamp', 
    'Normalized_Timestamp_x', 'Timestamp_y', 'Normalized_Timestamp_y', 'Timestamp_3D'
]

# 覚醒度のラベルを追加する関数
def label_awareness_level(class_value):
    if class_value == 2:
        return 1  # 覚醒中（ラベル1）
    elif class_value in [4, 6]:
        return 0  # 眠気有（ラベル0）
    else:
        return None  # 未定義クラス

# すべての被験者のデータを統合
all_features = []
all_labels = []

for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Please check if the subject's CSV is available in another location.")
        continue
    
    try:
        # データを読み込む
        data = pd.read_csv(file_path)
        
        # 生体情報と時間の情報の列を除く119列を特徴量として抽出
        feature_columns = [col for col in data.columns if col not in biosignal_and_time_columns]
        
        # 必要な特徴量がすべて揃っているかを確認
        if len(feature_columns) != 119:
            print(f"Skipping {file_path} because it does not contain the required 119 features.")
            continue
        
        features = data[feature_columns]
        
        # KSS_Theta_Alphaに基づいて覚醒度ラベルを追加
        data['Awareness_Label'] = data['KSS_Theta_Alpha'].apply(label_awareness_level)
        labels = data['Awareness_Label']
        
        # 有効なラベルの行のみ抽出し、リストに追加
        all_features.append(features[labels.notna()])
        all_labels.append(labels[labels.notna()])
    
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        continue

# データを統合
all_features_df = pd.concat(all_features, axis=0, ignore_index=True)
all_labels_series = pd.concat(all_labels, axis=0, ignore_index=True)

# 統計インデックスの計算
index_results = {
    'Feature': [],
    'Fisher_Index': [],
    'Correlation_Index': [],
    'T_test_Index': [],
    'Mutual_Information_Index': []
}

for feature_name in all_features_df.columns:
    x = all_features_df[feature_name]
    y = all_labels_series
    
    # Fisher Index
    x0, x1 = x[y == 0], x[y == 1]
    mean_0, mean_1 = np.mean(x0), np.mean(x1)
    std_0, std_1 = np.std(x0, ddof=1), np.std(x1, ddof=1)
    fisher_index = abs(mean_1 - mean_0) / (std_1**2 + std_0**2) if (std_1**2 + std_0**2) != 0 else np.nan
    
    # Correlation Index
    correlation_index, _ = pearsonr(x, y)
    
    # T-test Index
    t_stat, _ = ttest_ind(x0, x1, equal_var=False)
    t_test_index = abs(t_stat)
    
    # Mutual Information Index
    mutual_info = mutual_info_classif(x.values.reshape(-1, 1), y, discrete_features=True)
    mutual_information_index = mutual_info[0]
    
    # 結果を保存
    index_results['Feature'].append(feature_name)
    index_results['Fisher_Index'].append(fisher_index)
    index_results['Correlation_Index'].append(correlation_index)
    index_results['T_test_Index'].append(t_test_index)
    index_results['Mutual_Information_Index'].append(mutual_information_index)

# DataFrameを作成しCSVに保存
index_df = pd.DataFrame(index_results)
index_csv_path = 'feature_indices.csv'
index_df.to_csv(index_csv_path, index=False)

print(f"Index calculations saved to {index_csv_path}")

