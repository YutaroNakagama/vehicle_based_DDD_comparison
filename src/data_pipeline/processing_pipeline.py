"""
Data Processing Pipeline for Driver Drowsiness Detection (DDD).

This module orchestrates the preprocessing workflow (feature extraction,
transformation, labeling). It exposes :func:`main_pipeline` to prepare
data for model training and evaluation.

Notes
-----
- Dynamically selects preprocessing steps based on the model architecture
  (supported: common, SvmA, SvmW, Lstm).
- Subject data are saved in interim directories then merged/labeled.
- Physiological (PERCLOS / pupil) steps are currently disabled (commented).

Modules
-------
features.simlsl          : Time-frequency + smooth/std/pred-error features.
features.wavelet         : Wavelet decomposition of steering & accel signals.
features.eeg             : EEG band power extraction.
features.kss             : KSS label generation from EEG features.

Functions
---------
main_pipeline(model_name: str, use_jittering: bool = False) -> None
    Run preprocessing steps for all subjects for a given model.
"""

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from src.data_pipeline.features.simlsl import time_freq_domain_process, smooth_std_pe_process
from src.data_pipeline.features.wavelet import wavelet_process
from src.data_pipeline.features.physio import perclos_process, pupil_process  # currently not invoked
from src.data_pipeline.features.eeg import eeg_process
from src.data_pipeline.features.kss import kss_process
from src.data_pipeline.features.event_labels import event_label_process

from src.utils.io.loaders import read_subject_list
from src.utils.io.merge import merge_process

def main_pipeline(model_name: str, use_jittering: bool = False) -> None:
    """Run preprocessing for all subjects for a given model.

    Parameters
    ----------
    model_name : {"common", "SvmA", "SvmW", "Lstm"}
        Model architecture used to choose feature subsets.
    use_jittering : bool, default=False
        Apply jittering augmentation where supported.
    """
    subject_list = read_subject_list()
    for i, subject in enumerate(subject_list):
        logging.info(f"Processing subject {i+1}/{len(subject_list)}: {subject}")
        if model_name in ["common", "SvmA"]:
            time_freq_domain_process(subject, model_name, use_jittering=use_jittering)
        if model_name in ["common", "SvmW"]:
            wavelet_process(subject, model_name, use_jittering=use_jittering)
        if model_name in ["common", "Lstm"]:
            smooth_std_pe_process(subject, model_name, use_jittering=use_jittering)
        # EEG is needed for KSS labeling (non-Lstm models) but not for
        # Lstm which uses event-based labels from EventTimes.
        if model_name != "Lstm":
            eeg_process(subject, model_name)
        # Optional future steps:
        # pupil_process(subject, model_name)
        # perclos_process(subject, model_name)
        merge_process(subject, model_name)
        if model_name == "Lstm":
            # Wang et al. 2022: event-window-based binary labeling
            event_label_process(subject, model_name)
        else:
            kss_process(subject, model_name)

