"""Karolinska Sleepiness Scale (KSS) Estimation from EEG-Derived Features.

This module focuses on estimating Karolinska Sleepiness Scale (KSS) scores, a subjective measure
of sleepiness, based on objective EEG-derived frequency band power ratios (specifically Theta, Alpha, and Beta).
It provides functions for processing EEG features, converting band power ratios into KSS scores
using various methods (e.g., direct binning, percentile-based), handling outliers, and saving the results.

This module is an integral part of the driver drowsiness detection preprocessing pipeline,
linking physiological EEG data to a recognized drowsiness scale.
"""

import os
import pandas as pd
import numpy as np
import logging

from src.utils.io.loaders import save_csv
from src.utils.visualization.visualization import plot_custom_colored_distribution

from src.config import INTRIM_CSV_PATH, PROCESS_CSV_PATH, MODEL_WINDOW_CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def convert_theta_alpha_to_kss(theta_alpha_ratios: np.ndarray) -> np.ndarray:
    """
    Assigns KSS scores (1-9) based on Theta/Alpha ratio thresholds using a 9-bin linear distribution.

    This function takes an array of Theta/Alpha ratios and maps them to KSS scores
    by dividing the range of ratios into 9 equal bins. This provides a quantized
    estimation of drowsiness based on EEG spectral power.

    Args:
        theta_alpha_ratios (np.ndarray): A NumPy array of Theta/Alpha ratio values.

    Returns:
        np.ndarray: A NumPy array of estimated KSS scores (integers from 1 to 9)
                    corresponding to each input ratio.
    """
    min_value, max_value = np.min(theta_alpha_ratios), np.max(theta_alpha_ratios)
    # 9 bins = 8 thresholds (score=1~9), np.linspaceで10個→bins=1~9
    thresholds = np.linspace(min_value, max_value, 10)[1:-1]  # exclude min, max
    kss_scores = np.digitize(theta_alpha_ratios, bins=thresholds) + 1
    return kss_scores.tolist()

def convert_theta_alpha_to_kss_percentile(theta_alpha_ratios: np.ndarray) -> np.ndarray:
    """Assigns KSS scores (1-9) based on Theta/Alpha ratio percentiles.

    This method provides a percentile-based approach to KSS scoring, where
    the range of Theta/Alpha ratios is divided into 9 bins based on their
    distribution percentiles. This can be useful for relative scoring within
    a dataset.

    Args:
        theta_alpha_ratios (np.ndarray): A NumPy array of Theta/Alpha ratio values.

    Returns:
        np.ndarray: A NumPy array of estimated KSS scores (integers from 1 to 9)
                    corresponding to each input ratio, based on percentiles.
    """
    percentiles = np.percentile(theta_alpha_ratios, np.arange(0, 100, 100/9))[1:-1]
    kss_scores = np.digitize(theta_alpha_ratios, bins=percentiles) + 1
    return kss_scores.tolist()

def remove_outliers(data: np.ndarray, threshold: float = 3) -> np.ndarray:
    """Removes outliers from a 1D NumPy array based on a standard deviation threshold.

    Outliers are defined as data points that fall outside a specified number of
    standard deviations from the mean. This function helps in cleaning data
    before further processing or analysis.

    Args:
        data (np.ndarray): The input 1D NumPy array.
        threshold (float): The number of standard deviations from the mean to define
                           the outlier boundaries. Defaults to 3.

    Returns:
        np.ndarray: A new NumPy array with outliers removed.
    """
    mean, std_dev = np.mean(data), np.std(data)
    lower_bound, upper_bound = mean - threshold * std_dev, mean + threshold * std_dev
    return data[(data >= lower_bound) & (data <= upper_bound)]

def adjust_scores_length(scores: list, target_length: int) -> list:
    """Adjusts the length of a list of scores to match a target length.

    If the input `scores` list is shorter than `target_length`, it is padded
    with NaN values. If it is longer, it is truncated. This ensures consistency
    in data dimensions for subsequent processing or saving.

    Args:
        scores (list): The list of scores (e.g., KSS scores) to be adjusted.
        target_length (int): The desired length of the scores list.

    Returns:
        list: The adjusted list of scores.
    """
    arr = np.asarray(scores)
    if len(arr) < target_length:
        arr = np.concatenate([arr, np.full(target_length - len(arr), np.nan)])
    elif len(arr) > target_length:
        arr = arr[:target_length]
    return arr.tolist()

def kss_process(subject: str, model: str) -> None:
    """Computes and saves KSS (Karolinska Sleepiness Scale) scores based on EEG band powers.

    This function loads preprocessed EEG data, calculates two types of KSS scores:
    one based on the global (Theta + Alpha) / Beta ratio across all channels,
    and another based on the (Theta + Alpha) / Beta ratio from specific FC1 and FC2 channels.
    Outliers are removed before KSS score conversion. The estimated KSS scores
    are then appended to the processed data and saved to a CSV file.

    Args:
        subject (str): The subject identifier in 'subjectID_version' format (e.g., 'S0120_2').
        model (str): The model name, used for resolving file paths to the preprocessed data.

    Returns:
        None: The function saves the processed data with KSS scores to a CSV file and does not return any value.
    """
    parts = subject.split('_')
    if len(parts) != 2:
        logging.error(f"Unexpected subject format: {subject}")
        return

    subject_id, version = parts
    file_path = os.path.join(INTRIM_CSV_PATH, 'merged', model, f'merged_{subject_id}_{version}.csv')

    try:
        data = pd.read_csv(file_path)

        # ---- Global band power mean (all channels) for KSS score estimation ----
        theta_columns = data.filter(regex=r'Theta \(4-8 Hz\)')
        alpha_columns = data.filter(regex=r'Alpha \(8-13 Hz\)')
        beta_columns = data.filter(regex=r'Beta \(13-30 Hz\)')

        if not theta_columns.empty and not alpha_columns.empty and not beta_columns.empty:
            theta_mean = theta_columns.mean(axis=1)
            alpha_mean = alpha_columns.mean(axis=1).replace(0, 1e-10)
            beta_mean = beta_columns.mean(axis=1).replace(0, 1e-10)

            theta_alpha_beta_ratio = (theta_mean + alpha_mean) / beta_mean
            clean_ratios = remove_outliers(theta_alpha_beta_ratio)
            kss_scores = convert_theta_alpha_to_kss(clean_ratios)
            kss_scores = adjust_scores_length(kss_scores, len(data))
            kss_scores_percent = convert_theta_alpha_to_kss_percentile(clean_ratios)
            kss_scores_percent = adjust_scores_length(kss_scores_percent, len(data))

            # Save KSS score based on full-channel (θ+α)/β
            data['KSS_Theta_Alpha_Beta'] = kss_scores
            data['KSS_Theta_Alpha_Beta_percent'] = kss_scores_percent

        # ---- FC1/FC2 based (θ + α) / β index calculation ----
        theta_fc1 = data.get("Channel_2_Theta (4-8 Hz)")
        theta_fc2 = data.get("Channel_7_Theta (4-8 Hz)")
        alpha_fc1 = data.get("Channel_2_Alpha (8-13 Hz)")
        alpha_fc2 = data.get("Channel_7_Alpha (8-13 Hz)")
        beta_fc1 = data.get("Channel_2_Beta (13-30 Hz)")
        beta_fc2 = data.get("Channel_7_Beta (13-30 Hz)")

        if all(v is not None for v in [theta_fc1, theta_fc2, alpha_fc1, alpha_fc2, beta_fc1, beta_fc2]):
            theta_avg = (theta_fc1 + theta_fc2) / 2
            alpha_avg = (alpha_fc1 + alpha_fc2) / 2
            beta_avg = (beta_fc1 + beta_fc2).replace(0, 1e-10) / 2
            over_beta = (theta_avg + alpha_avg) / beta_avg
            data["theta_alpha_over_beta"] = over_beta

            # 9-level label derived from the index
            clean_over_beta = remove_outliers(over_beta)
            label_over_beta = convert_theta_alpha_to_kss(clean_over_beta)
            label_over_beta = adjust_scores_length(label_over_beta, len(data))
            data["theta_alpha_over_beta_label"] = label_over_beta
        else:
            logging.warning(f"Missing FC1/FC2 EEG band columns for {subject}")

        # ---- Save result ----
        save_csv(data, subject_id, version, 'processed', model)

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")

