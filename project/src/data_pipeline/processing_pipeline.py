from src.data_pipeline.features.simlsl import time_freq_domain_process, smooth_std_pe_process
from src.data_pipeline.features.wavelet import wavelet_process 
from src.data_pipeline.features.physio import perclos_process, pupil_process
from src.data_pipeline.features.eeg import eeg_process
from src.data_pipeline.features.kss import kss_process 

from src.utils.io.loaders import read_subject_list
from src.utils.io.merge import merge_process

def main_pipeline(model, use_jittering=False):

    subject_list = read_subject_list()
    
    for subject in subject_list:

        if model in ["common", "LstmA"]:
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

