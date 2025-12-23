"""
Parallel Data Processing Pipeline for Driver Drowsiness Detection (DDD).

Multiprocessing variant of the preprocessing workflow. Distributes
subject-level feature extraction across CPU processes.

Notes
-----
- Reuses feature extraction components from :mod:`src.data_pipeline.processing_pipeline`.
- Number of workers: env var ``N_PROC`` (default = min(cpu_count, n_subjects, 16)).
- Each subject processed independently; errors logged without halting.

Functions
---------
process_one_subject(args: tuple) -> None
    Worker: preprocess a single subject (safe-logs errors).
main_pipeline(model_name: str, use_jittering: bool = False) -> None
    Run full preprocessing in parallel.
"""

import os
import logging
import multiprocessing as mp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from src.data_pipeline.features.simlsl import time_freq_domain_process, smooth_std_pe_process
from src.data_pipeline.features.wavelet import wavelet_process
from src.data_pipeline.features.physio import perclos_process, pupil_process
from src.data_pipeline.features.eeg import eeg_process
from src.data_pipeline.features.kss import kss_process

from src.utils.io.loaders import read_subject_list
from src.utils.io.merge import merge_process


def process_one_subject(args: tuple) -> None:
    """Process one subject’s features and save results.

    Parameters
    ----------
    args : tuple
        (subject, model_name, use_jittering, idx, total)
    """
    subject, model_name, use_jittering, idx, total = args
    try:
        logging.info(f"Processing subject {idx+1}/{total}: {subject}")
        if model_name in ["common", "SvmA"]:
            time_freq_domain_process(subject, model_name, use_jittering=use_jittering)
        if model_name in ["common", "SvmW"]:
            wavelet_process(subject, model_name, use_jittering=use_jittering)
        if model_name in ["common", "Lstm"]:
            smooth_std_pe_process(subject, model_name, use_jittering=use_jittering)
        eeg_process(subject, model_name)
        # pupil_process(subject, model_name)
        # perclos_process(subject, model_name)
        merge_process(subject, model_name)
        kss_process(subject, model_name)
    except Exception as e:
        logging.error(f"Error processing subject {subject}: {e}")


def main_pipeline(model_name: str, use_jittering: bool = False) -> None:
    """Run the preprocessing pipeline in parallel.

    Parameters
    ----------
    model_name : {"common", "SvmA", "SvmW", "Lstm"}
        Model architecture controlling which feature blocks run.
    use_jittering : bool, default=False
        Enable jittering augmentation in supported steps.
    """
    subject_list = read_subject_list()
    total = len(subject_list)
    n_proc = int(os.environ.get("N_PROC", min(mp.cpu_count(), total, 16)))
    logging.info(f"Parallel processing with {n_proc} processes.")
    with mp.Pool(processes=n_proc) as pool:
        args_list = [(subject, model_name, use_jittering, idx, total) for idx, subject in enumerate(subject_list)]
        pool.map(process_one_subject, args_list)
    logging.info("All subjects processed.")

