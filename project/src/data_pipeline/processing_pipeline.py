"""
Data Processing Pipeline for Driver Drowsiness Detection (DDD).

This module orchestrates the entire preprocessing workflow, including
feature extraction, transformation, and labeling of raw subject data.
It provides the main entry point :func:`main_pipeline` to prepare data
for model training and evaluation.

Notes
-----
- The pipeline dynamically selects preprocessing steps based on the
  chosen model type (e.g., common, SvmW, Lstm, LstmA).
- Processed subject data are first saved in an interim directory,
  then merged and labeled for downstream tasks.

Modules
-------
features.simlsl : Time-frequency domain and smoothing/STD feature extraction.
features.wavelet : Wavelet decomposition for EEG and related signals.
features.physio : Physiological features such as pupil size and PERCLOS.
features.eeg : EEG feature extraction.
features.kss : KSS (Karolinska Sleepiness Scale) label processing.

Functions
---------
main_pipeline(model: str, use_jittering: bool = False) -> None
    Run the full preprocessing pipeline for a given model.
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from src.data_pipeline.features.simlsl import time_freq_domain_process, smooth_std_pe_process
from src.data_pipeline.features.wavelet import wavelet_process
from src.data_pipeline.features.physio import perclos_process, pupil_process
from src.data_pipeline.features.eeg import eeg_process
from src.data_pipeline.features.kss import kss_process

from src.utils.io.loaders import read_subject_list
from src.utils.io.merge import merge_process

def main_pipeline(model: str, use_jittering: bool = False) -> None:
    """Run the full preprocessing pipeline for a given model.

    For each subject, this function applies model-specific feature extraction
    methods such as time-frequency domain analysis, wavelet transforms,
    EEG feature extraction, and merges the data for training.

    Parameters
    ----------
    model : str
        Name of the model to determine which preprocessing steps to run.
        Must be one of ``["common", "SvmW", "Lstm", "LstmA"]``.
    use_jittering : bool, default=False
        Whether to apply jittering data augmentation.

    Returns
    -------
    None
        This function performs preprocessing and saves the processed files,
        but does not return any value.
    """
    # Read the list of subjects to be processed
    subject_list = read_subject_list()

    # Iterate through each subject and apply the preprocessing steps
    for i, subject in enumerate(subject_list):
        logging.info(f"Processing subject {i+1}/{len(subject_list)}: {subject}")

        # Apply time-frequency domain feature extraction for SvmA and common models
        if model in ["common", "SvmA"]:
            time_freq_domain_process(subject, model, use_jittering=use_jittering)

        # Apply wavelet transformation for SvmW and common models
        if model in ["common", "SvmW"]:
            wavelet_process(subject, model, use_jittering=use_jittering)

        # Apply smoothing, standard deviation, and prediction error processing for Lstm and common models
        if model in ["common", "Lstm"]:
            smooth_std_pe_process(subject, model, use_jittering=use_jittering)

        # Process EEG signals to extract relevant features
        eeg_process(subject, model)

        # Process pupil data (currently commented out, but available for future use)
        #pupil_process(subject, model)

        # Process PERCLOS data (currently commented out, but available for future use)
        #perclos_process(subject, model)
        
        # Merge all processed features for the current subject
        merge_process(subject, model)

        # Process KSS (Karolinska Sleepiness Scale) labels and align with features
        kss_process(subject, model)

