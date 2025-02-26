from src.data_processing.features.simlsl import time_freq_domain_process, smooth_std_pe_process
from src.data_processing.features.wavelet import wavelet_process 
from src.data_processing.features.physio import perclos_process, pupil_process
from src.data_processing.features.eeg import eeg_process
from src.data_processing.features.kss import kss_process 

from src.utils.loaders import read_subject_list
from src.utils.merge import merge_process

def main_pipeline(model):

    # load subject list
    subject_list = read_subject_list()
    
    for subject in subject_list:

        # wavelet process
        wavelet_process(subject)

        # smoothing std deviation PE process
        time_freq_domain_process(subject)

        # smoothing std deviation PE process
        smooth_std_pe_process(subject)

        # EEG process
        eeg_process(subject)

        # pupil process
        #pupil_process(subject)

        # perclos process
        #perclos_process(subject)
        
        # merge process 
        merge_process(subject)

        # kss process
        kss_process(subject)

