from src.utils.loaders import save_csv
from src.utils.visualization import plot_custom_colored_distribution

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
    STEP_SIZE_SEC,
    SCALING_FILTER,
    WAVELET_FILTER,
)

def convert_theta_alpha_to_kss_dynamic_from_data(theta_alpha_ratios):
    """
    Theta/Alpha比の最小値と最大値をデータ全体から算出してKSSスコアを割り当てる関数

    Parameters:
        theta_alpha_ratios (np.ndarray or list): Theta/Alpha比の全データ

    Returns:
        list: 割り当てられたKSSスコア（1～9）のリスト
    """
    # データ全体の最小値と最大値を取得
    min_value = np.min(theta_alpha_ratios)
    max_value = np.max(theta_alpha_ratios)

    # 最小値と最大値を9等分する閾値を計算
    thresholds = np.linspace(min_value, max_value, 10)  # 境界は9個の範囲を作る10個の値

    # 各Theta/Alpha比を対応するKSSスコアに割り当て
    kss_scores = []
    for ratio in theta_alpha_ratios:
        for kss_level in range(1, 10):  # KSSスコアは1～9
            if thresholds[kss_level - 1] <= ratio < thresholds[kss_level]:
                kss_scores.append(kss_level)
                break
        else:
            # 最大値以上の場合はKSS=9
            kss_scores.append(9)

    return kss_scores


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

def remove_outliers(data, threshold=3):
    """
    外れ値を除外する関数

    Parameters:
        data (np.ndarray or list): データ配列
        threshold (float): 標準偏差の閾値

    Returns:
        np.ndarray: 外れ値を除外したデータ
    """
    data = np.array(data)
    mean = np.mean(data)
    std_dev = np.std(data)
    lower_bound = mean - threshold * std_dev
    upper_bound = mean + threshold * std_dev
    return data[(data >= lower_bound) & (data <= upper_bound)]

def kss_process(subject):
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
    file_path = f'{INTRIM_CSV_PATH}/merged/merged_{subject_id}_{version}.csv'
    
    # CSVデータを読み込み
    try:
        data = pd.read_csv(file_path)
        
        # PERCLOSおよびEEGデータが存在する場合のみ処理
        if not data.filter(regex='Theta \(4-8 Hz\)').empty and not data.filter(regex='Alpha \(8-13 Hz\)').empty:
            # PERCLOSに基づくKSSを計算
            #data['KSS_PERCLOS'] = data['PERCLOS'].apply(convert_perclos_to_kss)
            
            # Theta/Alpha比に基づくKSSを計算
            theta_mean = data.filter(regex='Theta \(4-8 Hz\)').mean(axis=1)
            alpha_mean = data.filter(regex='Alpha \(8-13 Hz\)').mean(axis=1).replace(0, 1e-10)  # ゼロ除算を避ける
            beta_mean  = data.filter(regex='Beta \(13-30 Hz\)').mean(axis=1).replace(0, 1e-10)  # ゼロ除算を避ける
            theta_alpha_ratio = theta_mean / alpha_mean
            theta_alpha_beta_ratio = (theta_mean + alpha_mean) / beta_mean
            theta_alpha_beta_ratios = remove_outliers(theta_alpha_beta_ratio)
            #theta_alpha_beta_ratio = pd.Series(theta_alpha_beta_ratio)
            kss_scores = convert_theta_alpha_to_kss_dynamic_from_data(theta_alpha_beta_ratios)

            # 長さの調整
            if len(kss_scores) < len(data):
                # スコアが少ない場合、NaNで埋める
                kss_scores.extend([np.nan] * (len(data) - len(kss_scores)))
            elif len(kss_scores) > len(data):
                # スコアが多い場合、切り詰める
                kss_scores = kss_scores[:len(data)]

            data['KSS_Theta_Alpha_Beta'] = kss_scores
            plot_custom_colored_distribution(theta_alpha_beta_ratios)
            
            
            # 新しいファイルとして保存
            output_file_path = f'{PROCESS_CSV_PATH}/processed_{subject_id}_{version}.csv'
            data.to_csv(output_file_path, index=False)
            print(f"Processed and saved: {output_file_path}")
            
    except FileNotFoundError:
        print(f"File not found: {file_path}")
