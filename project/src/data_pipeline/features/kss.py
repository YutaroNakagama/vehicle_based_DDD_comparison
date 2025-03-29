import os
import pandas as pd
import numpy as np
import logging

from src.utils.io.loaders import save_csv
from src.utils.visualization.visualization import plot_custom_colored_distribution

from src.config import INTRIM_CSV_PATH, PROCESS_CSV_PATH, MODEL_WINDOW_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def convert_theta_alpha_to_kss(theta_alpha_ratios):
    """
    Assign KSS scores (1-9) based on the dynamic range of Theta/Alpha ratios.

    Parameters:
        theta_alpha_ratios (np.ndarray): Theta/Alpha ratio data.

    Returns:
        list: Assigned KSS scores.
    """
    min_value, max_value = np.min(theta_alpha_ratios), np.max(theta_alpha_ratios)
    thresholds = np.linspace(min_value, max_value, 10)

    kss_scores = []
    for ratio in theta_alpha_ratios:
        for kss_level in range(1, 10):
            if thresholds[kss_level - 1] <= ratio < thresholds[kss_level]:
                kss_scores.append(kss_level)
                break
        else:
            kss_scores.append(9)

    return kss_scores


def remove_outliers(data, threshold=3):
    """
    Remove outliers based on the specified standard deviation threshold.

    Parameters:
        data (np.ndarray): Data array.
        threshold (float): Standard deviation threshold.

    Returns:
        np.ndarray: Data without outliers.
    """
    mean, std_dev = np.mean(data), np.std(data)
    lower_bound, upper_bound = mean - threshold * std_dev, mean + threshold * std_dev
    return data[(data >= lower_bound) & (data <= upper_bound)]


def adjust_scores_length(scores, target_length):
    """
    Adjust the length of scores to match the target length.

    Parameters:
        scores (list): List of scores.
        target_length (int): Desired length.

    Returns:
        list: Adjusted scores list.
    """
    if len(scores) < target_length:
        scores.extend([np.nan] * (target_length - len(scores)))
    elif len(scores) > target_length:
        scores = scores[:target_length]
    return scores


def kss_process(subject, model):
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
    file_path = os.path.join(INTRIM_CSV_PATH, 'merged', model, f'merged_{subject_id}_{version}.csv')

    try:
        data = pd.read_csv(file_path)

        theta_columns = data.filter(regex='Theta \(4-8 Hz\)')
        alpha_columns = data.filter(regex='Alpha \(8-13 Hz\)')
        beta_columns = data.filter(regex='Beta \(13-30 Hz\)')

        if not theta_columns.empty and not alpha_columns.empty and not beta_columns.empty:
            theta_mean = theta_columns.mean(axis=1)
            alpha_mean = alpha_columns.mean(axis=1).replace(0, 1e-10)
            beta_mean = beta_columns.mean(axis=1).replace(0, 1e-10)

            theta_alpha_beta_ratio = (theta_mean + alpha_mean) / beta_mean
            clean_ratios = remove_outliers(theta_alpha_beta_ratio)
            kss_scores = convert_theta_alpha_to_kss(clean_ratios)

            kss_scores = adjust_scores_length(kss_scores, len(data))

            data['KSS_Theta_Alpha_Beta'] = kss_scores
            # Optional visualization
            # plot_custom_colored_distribution(clean_ratios)

            save_csv(data, subject_id, version, 'processed', model)

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")

