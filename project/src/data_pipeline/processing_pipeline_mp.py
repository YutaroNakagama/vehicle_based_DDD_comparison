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

