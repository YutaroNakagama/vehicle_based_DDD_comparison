import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import pandas as pd
import scipy.io
import os

# 処理を行うファイルのリスト
file_paths = [
    '../../../../dataset/Aygun2024/physio/S0116/SIMlsl_S0116_1.mat',
    '../../../../dataset/Aygun2024/physio/S0113/SIMlsl_S0113_1.mat'
]

# 各ファイルに対して処理を実行
for mat_file_path in file_paths:
    # ファイルを読み込む
    mat_data = scipy.io.loadmat(mat_file_path)

    # 厳密なGHMウェーブレットのスケーリング関数とウェーブレット関数の定義
    def ghm_scaling_function(t):
        # GHMスケーリング関数の近似をフラクタル的に定義
        return np.piecewise(t, [t < 0, (t >= 0) & (t < 0.25), (t >= 0.25) & (t < 0.5), (t >= 0.5) & (t < 0.75), t >= 0.75],
                            [0, lambda t: 4 * t, lambda t: 1 - 4 * (t - 0.25), lambda t: 4 * (t - 0.5), lambda t: 1 - 4 * (t - 0.75)])

    def ghm_wavelet_function(t):
        # GHMウェーブレット関数の近似をフラクタル的に定義
        return np.piecewise(t, [t < 0, (t >= 0) & (t < 0.5), (t >= 0.5) & (t < 1), t >= 1],
                            [0, lambda t: 2 * t, lambda t: -2 * t + 2, 0])

    # ウェーブレット変換を実行する関数
    def ghm_wavelet_transform(signal, levels):
        coeffs = []
        approx = signal
        for level in range(levels):
            # スケーリング係数とウェーブレット係数の計算
            t = np.linspace(-1, 1, len(approx))
            scaling_filter = ghm_scaling_function(t)
            wavelet_filter = ghm_wavelet_function(t)
            
            scaling_coeffs = convolve(approx, scaling_filter, mode='same')
            wavelet_coeffs = convolve(approx, wavelet_filter, mode='same')
            
            approx = scaling_coeffs[::2]
            coeffs.append((scaling_coeffs, wavelet_coeffs))
        
        return coeffs

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

    # adjust_and_add 関数を再定義
    def adjust_and_add(coeff1, coeff2):
        min_len = min(len(coeff1), len(coeff2))
        return coeff1[:min_len] + coeff2[:min_len]

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
    sampling_frequency = 1 / np.mean(time_diff)  # 平均時間差の逆数

    # 60秒の時間窓のサンプル数を計算
    window_size = int(60 * sampling_frequency)

    # データを時間窓で切り取る
    num_windows = int(len(sim_time) / window_size)

    # ウェーブレット変換のレベルを定義
    levels = 3

    # 各信号に対してGHMウェーブレット変換を実行し、時間窓ごとに分割して処理
    all_signals = []
    all_timestamps = []

    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        
        # 各時間窓でのデータを取得
        steering_window = steering_wheel_position_trimmed[start_idx:end_idx]
        long_accel_window = long_acceleration_trimmed[start_idx:end_idx]
        lat_accel_window = lat_acceleration_trimmed[start_idx:end_idx]
        lane_offset_window = lane_offset_trimmed[start_idx:end_idx]
        timestamp_window = sim_time[start_idx:end_idx]
        
        # GHMウェーブレット変換を実行
        steering_coeffs = ghm_wavelet_transform(steering_window, levels)
        long_accel_coeffs = ghm_wavelet_transform(long_accel_window, levels)
        lat_accel_coeffs = ghm_wavelet_transform(lat_accel_window, levels)
        lane_offset_coeffs = ghm_wavelet_transform(lane_offset_window, levels)
        
        # 8つのDecomposition信号を生成
        steering_signals = generate_decomposition_signals(steering_coeffs)
        long_accel_signals = generate_decomposition_signals(long_accel_coeffs)
        lat_accel_signals = generate_decomposition_signals(lat_accel_coeffs)
        lane_offset_signals = generate_decomposition_signals(lane_offset_coeffs)
        
        # 各時間窓のタイムスタンプを代表する値を使用（窓の最初のタイムスタンプを使用）
        all_timestamps.append(timestamp_window[0])
        
        # 全ての信号をまとめる
        all_signals.extend(steering_signals + long_accel_signals + lat_accel_signals + lane_offset_signals)

    # シグナル名を工夫して、どの信号から生成されたどのdecompositionか分かるようにする
    signal_names = ['SteeringWheel', 'LongitudinalAccel', 'LateralAccel', 'LaneOffset']
    decomposition_labels = ['DDD', 'DDA', 'DAD', 'DAA', 'ADD', 'ADA', 'AAD', 'AAA']

    # 各信号の長さを揃える
    min_length = min([len(signal) for signal in all_signals] + [len(all_timestamps)])
    all_signals = [signal[:min_length] for signal in all_signals]
    sim_time_trimmed = all_timestamps[:min_length]

    # データフレーム用の辞書を作成
    data = {'Timestamp': sim_time_trimmed}
    for i, base_signal in enumerate(signal_names):
        for j, label in enumerate(decomposition_labels):
            data[f'{base_signal}_{label}'] = all_signals[i * 8 + j]

    # データフレームを作成
    df = pd.DataFrame(data)

    # 出力ファイル名を生成
    output_filename = f'./32_Decomposed_Signals_with_Timestamps_{os.path.basename(mat_file_path).replace(".mat", ".csv")}'

    # データフレームをCSVファイルに保存
    df.to_csv(output_filename, index=False)

    # CSVファイルのパスを表示
    print("CSV file has been saved at:", output_filename)

