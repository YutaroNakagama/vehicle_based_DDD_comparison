
import os
import pandas as pd
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, welch, lfilter
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d
from sklearn.decomposition import FastICA

# Parameters
window_size_seconds = 60 # Time window size in seconds
step_size_seconds = 60   # Step size in seconds

# --------------------------------------------------------------------------- #
#                               LSTM                                          #
# --------------------------------------------------------------------------- #

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

        # Check if time data is valid to calculate sample rate
        if time_data[-1] - time_data[0] == 0:
            print(f"Warning: Invalid time data in {file_path}. Skipping this file.")
            return

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

        # Define output CSV path with the new directory structure
        #output_dir = "./csv/wang/"
        output_dir = "./"
        os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
        output_filename = os.path.join(output_dir, f"feat_wang_{os.path.basename(file_path).replace('.mat', '.csv')}")
        all_features_df.to_csv(output_filename, index=False)
        print(f"Features saved to {output_filename}")
    else:
        print(f"The data structure in {file_path} is invalid or does not contain the required rows.")

# Read the list of files from the text file
with open('../../../../dataset/Aygun2024/subject_list_temp.txt', 'r') as file:
    subject_files = file.read().splitlines()

# Process each file listed in the subject list
base_path = '../../../../dataset/Aygun2024/physio/'
for subject_file in subject_files:
    # Correct file path by adding "SIMlsl_" only to the file name
    dir_name, file_name = os.path.split(subject_file)
    file_path = os.path.join(base_path, dir_name, f"SIMlsl_{file_name}.mat")
    process_mat_file(file_path)

# --------------------------------------------------------------------------- #
#                               GHM                                           #
# --------------------------------------------------------------------------- #

#def feat_ext_zhao():
# 被験者リストファイルを読み込み、ファイルパスを生成
subject_list_path = '../../../../dataset/Aygun2024/subject_list_temp.txt'
with open(subject_list_path, 'r') as f:
    subjects = f.read().splitlines()

# 処理を行うファイルのリストを作成
base_path = '../../../../dataset/Aygun2024/physio/'
file_paths = [os.path.join(base_path, subject.replace('/', '/SIMlsl_') + '.mat') for subject in subjects]

# GHMスケーリングフィルタとウェーブレットフィルタの係数（厳密な係数を使用）
scaling_filter = np.array([0.48296, 0.83652, 0.22414, -0.12941])
wavelet_filter = np.array([-0.12941, -0.22414, 0.83652, -0.48296])

# 各ファイルに対して処理を実行
for mat_file_path in file_paths:
    # ファイルを読み込む
    mat_data = scipy.io.loadmat(mat_file_path)

    # ウェーブレット変換を実行する関数
    def ghm_wavelet_transform(signal, levels):
        coeffs = []
        approx = signal
        for level in range(levels):
            # スケーリング係数とウェーブレット係数の計算
            scaling_coeffs = lfilter(scaling_filter, [1], approx)[::2]  # ダウンサンプリング
            wavelet_coeffs = lfilter(wavelet_filter, [1], approx)[::2]  # ダウンサンプリング
            
            approx = scaling_coeffs
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
    time_diff = time_diff[time_diff > 0]  # ゼロまたは負の差を除外
    if len(time_diff) == 0:
        raise ValueError("タイムスタンプの差に有効な値がありません。タイムスタンプデータを確認してください。")

    sampling_frequency = 1 / np.mean(time_diff)  # 平均時間差の逆数

    # 60秒の時間窓のサンプル数を計算
    window_size = int(window_size_seconds * sampling_frequency)

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
    #output_filename = f'./csv/ghm/32_Decomposed_Signals_with_Timestamps_{os.path.basename(mat_file_path).replace(".mat", ".csv")}'
    output_filename = f'./feat_ghm_{os.path.basename(mat_file_path).replace(".mat", ".csv")}'

    # データフレームをCSVファイルに保存
    df.to_csv(output_filename, index=False)

    # CSVファイルのパスを表示
    print("CSV file has been saved at:", output_filename)
    
# --------------------------------------------------------------------------- #
#                               Arefnezhad                                    #
# --------------------------------------------------------------------------- #

# ファイルの読み込み関数
def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    return data

# バンドパワーを計算
def bandpower(data, band, fs):
    fmin, fmax = band
    freqs, psd = np.fft.rfftfreq(len(data), 1/fs), np.abs(np.fft.rfft(data))**2
    band_power = np.sum(psd[(freqs >= fmin) & (freqs <= fmax)])
    return band_power

# Theta/Beta比を計算
def theta_beta_ratio(signal, fs):
    theta_band = (4, 8)
    beta_band = (13, 30)
    theta_power = bandpower(signal, theta_band, fs)
    beta_power = bandpower(signal, beta_band, fs)
    return theta_power / beta_power if beta_power != 0 else 0

# サンプルエントロピーを計算
def sample_entropy(signal, m=2, r=None):
    if r is None:
        r = 0.2 * np.std(signal)
    N = len(signal)
    def _phi(m):
        X = np.array([signal[i : i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(X[:, None] - X[None, :]), axis=2) <= r, axis=0) - 1
        return np.sum(C) / (N - m + 1)
    return -np.log(_phi(m+1) / _phi(m))

# 特徴量抽出関数
def extract_features(signal, fs, prefix=""):
    features = {}
    features[f'{prefix}Range'] = np.max(signal) - np.min(signal)
    features[f'{prefix}Standard Deviation'] = np.std(signal)
    features[f'{prefix}Energy'] = np.sum(signal ** 2)
    features[f'{prefix}Zero Crossing Rate'] = ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)
    features[f'{prefix}First Quartile'] = np.percentile(signal, 25)
    features[f'{prefix}Second Quartile'] = np.median(signal)
    features[f'{prefix}Third Quartile'] = np.percentile(signal, 75)
    L = np.sum(np.sqrt(1 + np.diff(signal)**2))
    d = np.max(np.abs(signal - np.mean(signal)))
    features[f'{prefix}Katz Fractal Dimension'] = np.log10(len(signal)) / (np.log10(len(signal)) + np.log10(d/L) if d != 0 else 1)
    features[f'{prefix}Skewness'] = skew(signal)
    features[f'{prefix}Kurtosis'] = kurtosis(signal)
    hist, bin_edges = np.histogram(signal, bins=10, density=True)
    prob = hist / np.sum(hist)
    features[f'{prefix}Shannon Entropy'] = -np.sum(prob * np.log2(prob + np.finfo(float).eps))
    freqs = fftfreq(len(signal), 1/fs)
    spectrum = np.abs(fft(signal))**2
    freq_band = (0.5, 30)
    band_indices = np.where((freqs >= freq_band[0]) & (freqs <= freq_band[1]))
    features[f'{prefix}Frequency Variability'] = np.var(freqs[band_indices])
    spectral_prob = spectrum[band_indices] / np.sum(spectrum[band_indices])
    features[f'{prefix}Spectral Entropy'] = -np.sum(spectral_prob * np.log2(spectral_prob + np.finfo(float).eps))
    features[f'{prefix}Spectral Flux'] = np.sqrt(np.sum((np.diff(spectrum) ** 2)))
    features[f'{prefix}Center of Gravity of Frequency'] = np.sum(freqs[band_indices] * spectrum[band_indices]) / np.sum(spectrum[band_indices])
    features[f'{prefix}Dominant Frequency'] = freqs[np.argmax(spectrum)]
    features[f'{prefix}Average Value of PSD'] = np.mean(spectrum[band_indices])
    features[f'{prefix}Sample Entropy'] = 0#sample_entropy(signal) # for time save
    return features

# EEGデータを処理
def process_eeg_data(eeg_data, fs, window_size_sec=window_size_seconds):
    window_size_samples = int(window_size_sec * fs)
    step_size_samples = window_size_samples
    num_windows = (eeg_data.shape[1] - window_size_samples) // step_size_samples + 1
    ica = FastICA(n_components=eeg_data.shape[0], random_state=42)
    ica_components = ica.fit_transform(eeg_data.T).T
    theta_beta_ratios = {f"IC_{i+1}": [] for i in range(ica_components.shape[0])}
    for i in range(num_windows):
        start = i * step_size_samples
        end = start + window_size_samples
        for j, component in enumerate(ica_components):
            ratio = theta_beta_ratio(component[start:end], fs)
            theta_beta_ratios[f"IC_{j+1}"].append(ratio)
    mean_theta_beta_ratio = np.mean(list(theta_beta_ratios.values()), axis=0)
    time_values = np.arange(num_windows) * (step_size_samples / fs)
    return time_values, mean_theta_beta_ratio

# SIMlslデータを処理
def process_simlsl_data(signal1, signal2, signal3, signal4, fs, window_size_sec=window_size_seconds):
    window_size_samples = int(window_size_sec * fs)
    step_size_samples = window_size_samples
    num_windows = (signal1.shape[0] - window_size_samples) // step_size_samples + 1
    steering_features, lat_accel_features, lane_offset_features, long_accel_features = [], [], [], []
    for i in range(num_windows):
        start = i * step_size_samples
        end = start + window_size_samples
        segment1 = signal1[start:end]
        segment2 = signal2[start:end]
        segment3 = signal3[start:end]
        segment4 = signal4[start:end]
        steering_features.append(extract_features(segment1, fs, prefix="Steering_"))
        lat_accel_features.append(extract_features(segment2, fs, prefix="Lateral_"))
        lane_offset_features.append(extract_features(segment3, fs, prefix="LaneOffset_"))
        long_accel_features.append(extract_features(segment4, fs, prefix="LongAcc_"))
    return (pd.DataFrame(steering_features), pd.DataFrame(lat_accel_features),
            pd.DataFrame(lane_offset_features), pd.DataFrame(long_accel_features))

#def feat_ext_aref():
# 共通のファイルパス部分を定義
base_path = '../../../../dataset/Aygun2024/physio/'

# 被験者リストのファイルパスを設定
subject_list_path = '../../../../dataset/Aygun2024/subject_list_temp.txt'

# 被験者リストからデータを読み込む
with open(subject_list_path, 'r') as f:
    subject_list = [line.strip() for line in f]

# 各被験者のデータを処理
for entry in subject_list:
    subject_id, version = entry.split('/')[0], entry.split('/')[1].split('_')[-1]
    eeg_file = os.path.join(base_path, f"{subject_id}/EEG_{subject_id}_{version}.mat")
    simlsl_file = os.path.join(base_path, f"{subject_id}/SIMlsl_{subject_id}_{version}.mat")

    # EEGデータとSIMlslデータを読み込む
    try:
        print(f"processing for {subject_id}_{version}")
        eeg_data = load_data(eeg_file)['rawEEG'][1:, :]
        simlsl_data = load_data(simlsl_file)['SIM_lsl']
    except FileNotFoundError:
        print(f"File not found for {subject_id}_{version}, skipping...")
        continue
    
    # サンプリング周波数の計算
    eeg_timestamps = load_data(eeg_file)['rawEEG'][0, :]
    eeg_fs = 1 / np.mean(np.diff(eeg_timestamps))
    simlsl_timestamps = simlsl_data[0, :]
    simlsl_fs = 1 / np.mean(np.diff(simlsl_timestamps))
    
    # EEGデータを処理
    print("processing EEG")
    time_values, eeg_theta_beta_ratio = process_eeg_data(eeg_data, eeg_fs)
    #eeg_df = pd.DataFrame({"Time (seconds)": time_values, "Theta/Beta Ratio": eeg_theta_beta_ratio})
    eeg_df = pd.DataFrame({"Time (seconds)": time_values})
    
    # SIMlslデータから各信号を抽出し、特徴量を計算
    steering_wheel_position = simlsl_data[29, :]
    lateral_acceleration = simlsl_data[19, :]
    lane_offset = simlsl_data[27, :]
    long_acceleration = simlsl_data[18, :]
    print("processing SIMlsl")
    steering_features_df, lat_accel_features_df, lane_offset_features_df, long_accel_features_df = process_simlsl_data(
        steering_wheel_position, lateral_acceleration, lane_offset, long_acceleration, simlsl_fs)
    
    # サンプル数の一致を確認し、CSVファイルに出力
    if len(eeg_df) == len(steering_features_df) == len(lat_accel_features_df) == len(lane_offset_features_df) == len(long_accel_features_df):
        # 全データを1つのデータフレームに結合
        combined_df = pd.concat([eeg_df, steering_features_df, lat_accel_features_df, lane_offset_features_df, long_accel_features_df], axis=1)
        # CSVファイルとして保存
        combined_df.to_csv(f"feat_aref_{subject_id}_{version}.csv", index=False)
        print(f"Data for {subject_id}_{version} saved as a single CSV file: {subject_id}_{version}_Combined_Features.csv")
    else:
        # サンプル数が一致しない場合は個別のCSVファイルとして保存
        #eeg_df.to_csv(f"{subject_id}_{version}_EEG_Theta_Beta_Ratio.csv", index=False)
        steering_features_df.to_csv(f"{subject_id}_{version}_Steering_Wheel_Features.csv", index=False)
        lat_accel_features_df.to_csv(f"{subject_id}_{version}_Lateral_Acceleration_Features.csv", index=False)
        lane_offset_features_df.to_csv(f"{subject_id}_{version}_Lane_Offset_Features.csv", index=False)
        long_accel_features_df.to_csv(f"{subject_id}_{version}_Longitudinal_Acceleration_Features.csv", index=False)
        print(f"Data for {subject_id}_{version} saved as separate CSV files due to sample count mismatch.")

# --------------------------------------------------------------------------- #
#                        physio PERCLOS                                       #
# --------------------------------------------------------------------------- #

def calculate_and_save_perclos_physio_combined(blink_data_path, physio_data_path, output_csv_path, window_size_sec=window_size_seconds):
    """
    Load blink and physiological data from .mat files, calculate PERCLOS, resample physiological data,
    and save the combined result as a CSV file.
    """
    
    # Load and process PERCLOS data
    blink_data = scipy.io.loadmat(blink_data_path)
    blinks = blink_data['Blinks']
    blink_starts = blinks[:, 10]  # Start times
    blink_durations = blinks[:, 2]  # Duration (seconds)

    # Determine the total duration and number of windows
    total_duration = blink_starts[-1] - blink_starts[0]
    num_windows = int(total_duration // window_size_sec)

    # Calculate PERCLOS per window
    perclos_values = []
    timestamps = []
    start_time = blink_starts[0]
    
    for i in range(num_windows):
        window_start = start_time + i * window_size_sec
        mask = (blink_starts >= window_start) & (blink_starts < window_start + window_size_sec)
        total_blink_duration = blink_durations[mask].sum()
        perclos = total_blink_duration / window_size_sec
        perclos_values.append(perclos)
        timestamps.append(window_start)
    
    perclos_df = pd.DataFrame({"Timestamp": timestamps, "PERCLOS": perclos_values})
    perclos_df['Normalized_Timestamp'] = perclos_df['Timestamp'] - perclos_df['Timestamp'].min()
    perclos_df['Window'] = (perclos_df['Normalized_Timestamp'] // window_size_sec).astype(int)

    # Load and process Physio data
    physio_data = scipy.io.loadmat(physio_data_path)
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
    physio_df['Window'] = (physio_df['Normalized_Timestamp'] // window_size_sec).astype(int)
    physio_resampled = physio_df.groupby('Window').mean().reset_index()

    # Merge PERCLOS and resampled Physio data on the 'Window' key
    merged_df = pd.merge(physio_resampled, perclos_df, on="Window", how="inner")

    # Save the combined data to CSV
    merged_df.to_csv(output_csv_path, index=False)

# Update the function to use the correct file naming convention for Blink and Physio files
def process_all_subjects(subject_list_path, base_path, output_base_path, window_size_sec=window_size_seconds):
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
        output_csv_combined_file = f"{output_base_path}perclos_physio_{version}.csv"

        # Check if both required files exist for the subject, proceed if they do
        try:
            # Calculate and save PERCLOS and Physio combined data for each subject
            calculate_and_save_perclos_physio_combined(blink_data_file, physio_data_file, output_csv_combined_file, window_size_sec)
            print(f"Processed and saved data for {subject_name} {version}")
        except FileNotFoundError:
            print(f"Missing files for {subject_name} {version}, skipping...")
        except Exception as e:
            print(f"Error processing {subject_name} {version}: {e}")

#def get_physio_perclos():
base_path = '../../../../dataset/Aygun2024/physio/'
list_base_path = '../../../../dataset/Aygun2024/'
subject_list_path = f'{list_base_path}subject_list_temp.txt'
output_base_path = './'  # Define a base output path for all processed CSVs

# Execute the processing function for all subjects
process_all_subjects(subject_list_path, base_path, output_base_path)
    
# --------------------------------------------------------------------------- #
#                        EEG                                                  #
# --------------------------------------------------------------------------- #

# Functions for bandpass filtering and power calculation
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def calculate_band_power(signal, band, fs):
    f, Pxx = welch(signal, fs, nperseg=1024)
    band_power = np.sum(Pxx[(f >= band[0]) & (f <= band[1])])
    return band_power

#def get_eeg():
# Define frequency bands and sampling parameters
frequency_bands = {
    "Delta (0.5-4 Hz)": (0.5, 4),
    "Theta (4-8 Hz)": (4, 8),
    "Alpha (8-13 Hz)": (8, 13),
    "Beta (13-30 Hz)": (13, 30),
    "Gamma (30-100 Hz)": (30, 100)
}
fs = 500  # Sampling frequency (assumed consistent across datasets)
window_size = window_size_seconds  # 1-minute window in seconds
window_samples = window_size * fs  # Convert window size to samples

# Read subject list file
base_path = '../../../../dataset/Aygun2024/physio'
list_base_path = '../../../../dataset/Aygun2024/'
with open(f'{list_base_path}subject_list_temp.txt', 'r') as file:
    subjects = [line.strip() for line in file]

# Process each subject's EEG data
for subject_path in subjects:
    try:
        # Extract subject name and version from the path (e.g., S0210/S0210_2 -> S0210_2)
        subject_name_version = subject_path.split('/')[-1]
        subject_name = subject_path.split('/')[-2]
        mat_file_name = f'EEG_{subject_name_version}.mat'
        
        # Load EEG data file
        #mat_data = sio.loadmat(f'{base_path}/{subject_name}/{mat_file_name}')
        mat_data = scipy.io.loadmat(f'{base_path}/{subject_name}/{mat_file_name}')
        eeg_data = mat_data.get('rawEEG')

        # Extract timestamps (first row)
        timestamps = eeg_data[0, :]

        # Initialize storage for each channel
        channel_band_powers = {ch: {band: [] for band in frequency_bands.keys()} for ch in range(1, eeg_data.shape[0])}
        timestamp_windows = []

        # Calculate band powers for each 1-minute window
        num_windows = int(eeg_data.shape[1] / window_samples)
        for w in range(num_windows):
            window_start_idx = w * window_samples
            window_end_idx = (w + 1) * window_samples
            window_data = eeg_data[:, window_start_idx:window_end_idx]

            # Record start timestamp for each window
            timestamp_windows.append(timestamps[window_start_idx])
            for ch in range(1, eeg_data.shape[0]):  # For each EEG channel (excluding timestamps)
                for band_name, (low, high) in frequency_bands.items():
                    band_power = calculate_band_power(bandpass_filter(window_data[ch, :], low, high, fs), (low, high), fs)
                    channel_band_powers[ch][band_name].append(band_power)

        # Convert results to a DataFrame, including timestamps
        data_for_csv = {'Timestamp': timestamp_windows}
        for ch in range(1, eeg_data.shape[0]):
            for band_name in frequency_bands.keys():
                column_name = f"Channel_{ch}_{band_name}"
                data_for_csv[column_name] = channel_band_powers[ch][band_name]
        df_results = pd.DataFrame(data_for_csv)
        
        # Save the results to a CSV file
        output_path = f'./eeg_{subject_name_version}.csv'
        df_results.to_csv(output_path, index=False)
        
        print(f"Processed and saved for {subject_name_version}")
        
    except Exception as e:
        print(f"Failed to process {subject_name_version}: {e}")
    
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
def process_pupil_data(subject_id, version, window_size_sec=window_size_seconds, threshold=3):
    base_path = '../../../../dataset/Aygun2024/physio'
    file_path = f'{base_path}/{subject_id}/PupilData_{subject_id}_{version}.mat'
    data = scipy.io.loadmat(file_path)

    timestamps_2d = data['Pupil2D'][0]
    left_pupil_2d = data['Pupil2D'][1]
    right_pupil_2d = data['Pupil2D'][2]
    timestamps_3d = data['Pupil3D'][0]
    left_pupil_3d = data['Pupil3D'][1]
    right_pupil_3d = data['Pupil3D'][2]

    sampling_rate_2d = len(timestamps_2d) / (timestamps_2d[-1] - timestamps_2d[0])
    sampling_rate_3d = len(timestamps_3d) / (timestamps_3d[-1] - timestamps_3d[0])
    
    # Calculate non-overlapping averages for 2D pupil data
    window_samples_2d = int(window_size_sec * sampling_rate_2d)
    left_pupil_2d_cleaned = replace_outliers_with_interpolation(left_pupil_2d, threshold)
    right_pupil_2d_cleaned = replace_outliers_with_interpolation(right_pupil_2d, threshold)
    left_pupil_2d_avg = non_overlapping_average(left_pupil_2d_cleaned, window_samples_2d)
    right_pupil_2d_avg = non_overlapping_average(right_pupil_2d_cleaned, window_samples_2d)
    timestamps_2d_avg = timestamps_2d[::window_samples_2d][:len(left_pupil_2d_avg)]
    
    # Calculate non-overlapping averages for 3D pupil data
    window_samples_3d = int(window_size_sec * sampling_rate_3d)
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

#def get_pupil():
# Load subject list and process each file
list_base_path = '../../../../dataset/Aygun2024'
with open(f'{list_base_path}/subject_list_temp.txt', 'r') as file:
    for line in file:
        subject_info = line.strip().split('/')
        subject_id = subject_info[0]
        version = subject_info[1].split('_')[1]
        
        # Process each subject's data and save to an individual CSV file
        subject_df = process_pupil_data(subject_id, version)
        subject_df.to_csv(f'pupil_{subject_id}_{version}.csv', index=False)


# --------------------------------------------------------------------------- #
#                               merge                                         #
# --------------------------------------------------------------------------- #

#def merge():
# Define the features and corresponding timestamp columns
features = {
    "feat_wang_SIMlsl": "Timestamp",
    "feat_ghm_SIMlsl": "Timestamp",
    "perclos_physio": "Timestamp_x",
    "feat_aref": "Time (seconds)",
    "pupil": "Timestamp_2D",
    "eeg": "Timestamp"
}

# Read the subject list
list_base_path = '../../../../dataset/Aygun2024/'
with open(f'{list_base_path}subject_list_temp.txt', 'r') as file:
    subjects = file.readlines()

# Process each subject
for subject in subjects:
    subject = subject.strip()
    subject_id, version = subject.split('/')
    version = version.split('_')[1]

    # Initialize an empty DataFrame for merging results
    merged_df = pd.DataFrame()

    # Process each feature for the current subject
    for feature, timestamp_col in features.items():
        # Construct the file path for each feature CSV
        file_path = f"{feature}_{subject_id}_{version}.csv"
        
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
            print(f"File not found: {file_path}")

    # Save the merged result for the current subject
    output_file = f"{subject_id}_{version}_merged_data.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"Saved merged file for {subject_id}_{version} to {output_file}")

# --------------------------------------------------------------------------- #
#                               estimate KSS                                  #
# --------------------------------------------------------------------------- #

#def kss_estimate():
# KSS変換関数を定義
def convert_perclos_to_kss(perclos):
    if perclos < 10:
        return 2  # KSS 1-3 (Awake state)
    elif 10 <= perclos < 20:
        return 4  # KSS 4-5 (Mild sleepiness)
    elif 20 <= perclos < 30:
        return 6  # KSS 6-7 (Moderate sleepiness)
    else:
        return 8  # KSS 8-9 (Strong sleepiness)

def convert_theta_alpha_to_kss(theta_alpha_ratio):
    if theta_alpha_ratio < 1.0:
        return 2  # KSS 1-3 (Awake state)
    elif 1.0 <= theta_alpha_ratio < 1.5:
        return 4  # KSS 4-5 (Mild sleepiness)
    elif 1.5 <= theta_alpha_ratio < 2.0:
        return 6  # KSS 6-7 (Moderate sleepiness)
    else:
        return 8  # KSS 8-9 (Strong sleepiness)

# テキストファイルから被験者リストを読み込み
list_base_path = '../../../../dataset/Aygun2024/'
with open(f'{list_base_path}subject_list_temp.txt', 'r') as file:
    subjects = file.read().splitlines()

# 各被験者ファイルに対して処理を行う
for subject in subjects:
    subject_id, version = subject.split('/')
    file_path = f'./{version}_merged_data.csv'
    
    # CSVデータを読み込み
    try:
        data = pd.read_csv(file_path)
        
        # PERCLOSおよびEEGデータが存在する場合のみ処理
        if 'PERCLOS' in data.columns and not data.filter(regex='Theta \(4-8 Hz\)').empty and not data.filter(regex='Alpha \(8-13 Hz\)').empty:
            # PERCLOSに基づくKSSを計算
            data['KSS_PERCLOS'] = data['PERCLOS'].apply(convert_perclos_to_kss)
            
            # Theta/Alpha比に基づくKSSを計算
            theta_mean = data.filter(regex='Theta \(4-8 Hz\)').mean(axis=1)
            alpha_mean = data.filter(regex='Alpha \(8-13 Hz\)').mean(axis=1).replace(0, 1e-10)  # ゼロ除算を避ける
            theta_alpha_ratio = theta_mean / alpha_mean
            data['KSS_Theta_Alpha'] = theta_alpha_ratio.apply(convert_theta_alpha_to_kss)
            
            # 新しいファイルとして保存
            output_file_path = f'{subject_id}_{version}_merged_data_with_KSS.csv'
            data.to_csv(output_file_path, index=False)
            print(f"Processed and saved: {output_file_path}")
            
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    
#def main():
#    feat_ext_wang()
#    feat_ext_zhao()
#    feat_ext_aref()
#    get_physio_perclos()
#    get_physio_perclos()
#    get_eeg()
#    merge()
#    kss_estimate()
#
#if __name__ == '__main__':
#    main()
