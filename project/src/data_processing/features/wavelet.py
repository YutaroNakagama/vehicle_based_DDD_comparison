
from src.utils.loaders import safe_load_mat, save_csv

import os
import pandas as pd
import numpy as np
import logging
from scipy.signal import lfilter

# Local application imports
from config import (
    SUBJECT_LIST_PATH, 
    DATASET_PATH, 
    INTRIM_CSV_PATH, 
    PROCESS_CSV_PATH, 
    WINDOW_SIZE_SEC, 
    WINDOW_SIZE_SAMPLE_SIMLSL,
    STEP_SIZE_SEC,
    STEP_SIZE_SAMPLE_SIMLSL,
    SCALING_FILTER,
    WAVELET_FILTER,
    WAVELET_LEV,
)

# ウェーブレット変換を実行する関数
def ghm_wavelet_transform(signal):
    coeffs = []
    approx = signal
    for level in range(WAVELET_LEV):
        # スケーリング係数とウェーブレット係数の計算
        scaling_coeffs = lfilter(SCALING_FILTER, [1], approx)[::2]  # ダウンサンプリング
        wavelet_coeffs = lfilter(WAVELET_FILTER, [1], approx)[::2]  # ダウンサンプリング
        
        approx = scaling_coeffs
        coeffs.append((scaling_coeffs, wavelet_coeffs))
    
    return coeffs

# adjust_and_add 関数を再定義
def adjust_and_add(coeff1, coeff2):
    min_len = min(len(coeff1), len(coeff2))
    return coeff1[:min_len] + coeff2[:min_len]

# 各信号の8つのDecomposition信号を作成
def generate_decomposition_signals(coeffs):
    return [
        coeffs[0][1],  # DDD
        adjust_and_add(coeffs[0][1], coeffs[1][0]),  # DDA
        adjust_and_add(coeffs[0][1], coeffs[1][1]),  # DAD
        adjust_and_add(coeffs[0][1], coeffs[2][0]),  # DAA
        adjust_and_add(coeffs[1][1], coeffs[2][1]),  # ADD
        adjust_and_add(coeffs[1][0], coeffs[2][1]),  # ADA
        adjust_and_add(coeffs[1][0], coeffs[2][0]),  # AAD
        coeffs[2][0]   # AAA
    ]

def calculate_power(signal):
    return np.sum(np.square(signal)) / len(signal)

def wavelet_process(subject):
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
    # 処理を行うファイルのリストを作成
    #mat_file_path = [os.path.join(DATASET_PATH + '/', subject.replace('/', '/SIMlsl_') + '.mat') for subject in subjects]
    mat_file_path = os.path.join(DATASET_PATH + '/', subject.replace('/', '/SIMlsl_') + '.mat')
    # ファイルを読み込む
    mat_data = safe_load_mat(mat_file_path)
    if mat_data is None:
        return  # エラー発生時にスキップ
    
    # データを取得
    steering_wheel_position = mat_data['SIM_lsl'][29, :]  # ステアリング位置 (30行目)
    long_acceleration = mat_data['SIM_lsl'][18, :]        # 縦加速度 (19行目)
    lat_acceleration = mat_data['SIM_lsl'][19, :]         # 横加速度 (20行目)
    lane_offset = mat_data['SIM_lsl'][27, :]              # Lane offset (28行目)
    sim_time = mat_data['SIM_lsl'][0, :]                  # タイムスタンプ (1行目)
    
    # NaNの値を0に置き換える
    steering_wheel_position_trimmed = np.nan_to_num(steering_wheel_position)
    long_acceleration_trimmed = np.nan_to_num(long_acceleration)
    lat_acceleration_trimmed = np.nan_to_num(lat_acceleration)
    lane_offset_trimmed = np.nan_to_num(lane_offset)
    
    # サンプリング周波数を推測（タイムスタンプの差から計算）
    time_diff = np.diff(sim_time)
    time_diff = time_diff[time_diff > 0]  # ゼロまたは負の差を除外
    if len(time_diff) == 0:
        raise ValueError("タイムスタンプの差に有効な値がありません。タイムスタンプデータを確認してください。")
    
    # データを時間窓で切り取る
    num_windows = int(len(sim_time) / WINDOW_SIZE_SEC)
    
    # 各信号に対してGHMウェーブレット変換を実行し、時間窓ごとに分割して処理
    all_powers = []
    all_timestamps = []
    
    for start in range(0, len(sim_time) - WINDOW_SIZE_SAMPLE_SIMLSL + 1, STEP_SIZE_SAMPLE_SIMLSL):
        #start_idx = i * WINDOW_SIZE_SEC
        #end_idx = start_idx + WINDOW_SIZE_SEC
        
        # 各時間窓でのデータを取得
        steering_window = steering_wheel_position_trimmed[start:start + WINDOW_SIZE_SAMPLE_SIMLSL]
        long_accel_window = long_acceleration_trimmed[start:start + WINDOW_SIZE_SAMPLE_SIMLSL]
        lat_accel_window = lat_acceleration_trimmed[start:start + WINDOW_SIZE_SAMPLE_SIMLSL]
        lane_offset_window = lane_offset_trimmed[start:start + WINDOW_SIZE_SAMPLE_SIMLSL]
        timestamp_window = sim_time[start:start + WINDOW_SIZE_SAMPLE_SIMLSL]
        
        # GHMウェーブレット変換を実行
        steering_coeffs = ghm_wavelet_transform(steering_window)
        long_accel_coeffs = ghm_wavelet_transform(long_accel_window)
        lat_accel_coeffs = ghm_wavelet_transform(lat_accel_window)
        lane_offset_coeffs = ghm_wavelet_transform(lane_offset_window)
        
        # 8つのDecomposition信号を生成
        steering_signals = generate_decomposition_signals(steering_coeffs)
        long_accel_signals = generate_decomposition_signals(long_accel_coeffs)
        lat_accel_signals = generate_decomposition_signals(lat_accel_coeffs)
        lane_offset_signals = generate_decomposition_signals(lane_offset_coeffs)
        
        # 信号パワーを計算
        power_values = [
            calculate_power(signal) for signal in (steering_signals + long_accel_signals + lat_accel_signals + lane_offset_signals)
        ]

        all_powers.append(power_values)
        # 各時間窓のタイムスタンプを代表する値を使用（窓の最初のタイムスタンプを使用）
        all_timestamps.append(timestamp_window[0])
        
    # シグナル名を工夫して、どの信号から生成されたどのdecompositionか分かるようにする
    signal_names = ['SteeringWheel', 'LongitudinalAccel', 'LateralAccel', 'LaneOffset']
    decomposition_labels = ['DDD', 'DDA', 'DAD', 'DAA', 'ADD', 'ADA', 'AAD', 'AAA']
    
    column_names = [f'{base_signal}_{label}' for base_signal in signal_names for label in decomposition_labels]

    # データフレームを作成
    df = pd.DataFrame(all_powers, columns=column_names)
    df.insert(0, 'Timestamp', all_timestamps)

    save_csv(df, subject_id, version, 'wavelet') 

