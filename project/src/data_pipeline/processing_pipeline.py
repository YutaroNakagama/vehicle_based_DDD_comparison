"""Main processing pipeline for data preprocessing in DDD system.

This module defines a single entry point `main_pipeline()` that coordinates multiple
feature extraction steps (e.g., time-frequency, wavelet, EEG-based) for each subject
based on the selected model.

The output is stored in the interim directory and later merged and labeled for modeling.
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

    Args:
        model (str): Name of the model to determine which preprocessing steps to run.
                     Must be one of ['common', 'SvmW', 'Lstm', 'LstmA'].
        use_jittering (bool): Whether to apply jittering data augmentation.

    Returns:
        None
    """
    subject_list = read_subject_list()

    for i, subject in enumerate(subject_list):

        logging.info(f"Processing subject {i+1}/{len(subject_list)}: {subject}")

        if model in ["common", "SvmA"]:
            time_freq_domain_process(subject, model, use_jittering=use_jittering)
        if model in ["common", "SvmW"]:
            wavelet_process(subject, model, use_jittering=use_jittering)
        if model in ["common", "Lstm"]:
            smooth_std_pe_process(subject, model, use_jittering=use_jittering)

        # EEG process
        eeg_process(subject, model)

        # pupil process
        #pupil_process(subject, model)

        # perclos process
        #perclos_process(subject, model)
        
        # merge process 
        merge_process(subject, model)

        # kss process
        kss_process(subject, model)

