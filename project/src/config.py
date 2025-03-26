import numpy as np

# File paths
SUBJECT_LIST_PATH   = '../../dataset/mdapbe/subject_list.txt'
DATASET_PATH        = '../../dataset/mdapbe/physio'
INTRIM_CSV_PATH     = './data/interim'
PROCESS_CSV_PATH    = './data/processed'
MODEL_PKL_PATH    = './model/'
OUTPUT_SVG_PATH     = './output/svg'

DATA_PROCESS_CHOICES = ["SvmA", "SvmW", "Lstm", "common"] 
MODEL_CHOICES = ["SvmA", "SvmW", "Lstm", "RF"]  

# Data process parameters
MODEL_WINDOW_CONFIG = {
    "common": {"window_sec": 3, "step_sec": 0.5},
    "SvmA":   {"window_sec": 3, "step_sec": 0.5},
    "SvmW":   {"window_sec": 3, "step_sec": 0.5},
    "Lstm":   {"window_sec": 3, "step_sec": 0.5},
}

SAMPLE_RATE_SIMLSL = 60 # sample rate for simlsl

SAMPLE_RATE_EEG = 500

SCALING_FILTER = np.array([0.48296, 0.83652, 0.22414, -0.12941])
WAVELET_FILTER = np.array([-0.12941, -0.22414, 0.83652, -0.48296])
WAVELET_LEV = 3
