"""
Parallel Data Processing Pipeline for Driver Drowsiness Detection (DDD).

This module provides a multiprocessing-based implementation of the
preprocessing workflow. It distributes subject-level preprocessing
tasks across multiple CPU cores to accelerate feature extraction,
transformation, and labeling.

Notes
-----
- Uses the same feature extraction components as
  :mod:`src.data_pipeline.processing_pipeline`.
- The number of processes is controlled by the environment variable
  ``N_PROC``. If unset, defaults to ``min(cpu_count, n_subjects, 16)``.
- Each subject is processed independently, so failures are logged
  without stopping the entire pipeline.

Functions
---------
process_one_subject(args: tuple) -> None
    Run preprocessing for a single subject (worker function).
main_pipeline(model: str, use_jittering: bool = False) -> None
    Run the full preprocessing pipeline in parallel using multiprocessing.
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

def process_one_subject(args):
    """Process one subject’s features and save results.

    Parameters
    ----------
    args : tuple
        Tuple containing subject ID, model name, jittering flag, index, and total count.

    Returns
    -------
    None
        Processed data is written to disk.
    """
    subject, model, use_jittering, idx, total = args
    try:
        logging.info(f"Processing subject {idx+1}/{total}: {subject}")

        if model in ["common", "SvmA"]:
            time_freq_domain_process(subject, model, use_jittering=use_jittering)
        if model in ["common", "SvmW"]:
            wavelet_process(subject, model, use_jittering=use_jittering)
        if model in ["common", "Lstm"]:
            smooth_std_pe_process(subject, model, use_jittering=use_jittering)
        eeg_process(subject, model)
        # pupil_process(subject, model)
        # perclos_process(subject, model)
        merge_process(subject, model)
        kss_process(subject, model)
    except Exception as e:
        logging.error(f"Error processing subject {subject}: {e}")

def main_pipeline(model: str, use_jittering: bool = False) -> None:
    """Run the full preprocessing pipeline in parallel.

    Parameters
    ----------
    model : str
        Model name to determine preprocessing steps.
        Must be one of ``["common", "SvmA", "SvmW", "Lstm"]``.
    use_jittering : bool, default=False
        Whether to apply jittering data augmentation.

    Returns
    -------
    None
        Preprocessing outputs are written to disk; nothing is returned.

    Notes
    -----
    - Parallelism is controlled by ``N_PROC`` environment variable.
      Defaults to ``min(cpu_count, n_subjects, 16)``.
    - Logs progress and completion status for each subject.
    """
    subject_list = read_subject_list()
    total = len(subject_list)

    # プロセス数を環境変数から取得（なければmin(mp.cpu_count(), total)）
    n_proc = int(os.environ.get("N_PROC", min(mp.cpu_count(), total, 16)))   # 16などのデフォルト上限
    logging.info(f"Parallel processing with {n_proc} processes.")

    # 並列実行
    with mp.Pool(processes=n_proc) as pool:
        args_list = [(subject, model, use_jittering, idx, total) for idx, subject in enumerate(subject_list)]
        pool.map(process_one_subject, args_list)

    logging.info("All subjects processed.")

