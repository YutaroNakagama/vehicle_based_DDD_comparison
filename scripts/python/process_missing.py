"""Process only missing subjects."""
import os
import logging
import multiprocessing as mp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.data_pipeline.features.simlsl import time_freq_domain_process, smooth_std_pe_process
from src.data_pipeline.features.wavelet import wavelet_process
from src.data_pipeline.features.eeg import eeg_process
from src.data_pipeline.features.kss import kss_process
from src.utils.io.loaders import read_subject_list
from src.utils.io.merge import merge_process


def process_one_subject(args):
    """Process one subject's features."""
    subject, model_name, idx, total = args
    try:
        logging.info(f'Processing subject {idx+1}/{total}: {subject}')
        time_freq_domain_process(subject, model_name, use_jittering=False)
        wavelet_process(subject, model_name, use_jittering=False)
        smooth_std_pe_process(subject, model_name, use_jittering=False)
        eeg_process(subject, model_name)
        merge_process(subject, model_name)
        kss_process(subject, model_name)
        logging.info(f'Completed subject {idx+1}/{total}: {subject}')
    except Exception as e:
        logging.error(f'Error processing subject {subject}: {e}')


def main():
    # Get missing subjects
    subject_list = read_subject_list()
    processed_dir = './data/processed/common'
    processed = set(
        f.replace('processed_', '').replace('.csv', '') 
        for f in os.listdir(processed_dir) if f.endswith('.csv')
    )
    missing = [s for s in subject_list if s not in processed]
    
    total = len(missing)
    if total == 0:
        logging.info('All subjects already processed!')
        return
    
    logging.info(f'Found {total} missing subjects: {missing}')
    
    n_proc = min(mp.cpu_count(), total, 16)
    logging.info(f'Processing with {n_proc} processes...')
    
    with mp.Pool(processes=n_proc) as pool:
        args_list = [(s, 'common', idx, total) for idx, s in enumerate(missing)]
        pool.map(process_one_subject, args_list)
    
    logging.info('All missing subjects processed.')


if __name__ == '__main__':
    main()
