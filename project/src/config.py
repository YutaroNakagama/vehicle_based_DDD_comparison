import numpy as np

# File paths
#SUBJECT_LIST_PATH   = './data/external/Aygun2024/subject_list_temp.txt'
SUBJECT_LIST_PATH   = '../../dataset/mdapbe/subject_list_temp.txt'
#SUBJECT_LIST_PATH   = './data/Aygun2024/subject_list_temp_ori.txt'
#DATASET_PATH        = './data/external/Aygun2024/physio'
DATASET_PATH        = '../../dataset/mdapbe/physio'
#OUTPUT_CSV_PATH     = './output/csv'
INTRIM_CSV_PATH     = './data/interim'
PROCESS_CSV_PATH    = './data/processed'
OUTPUT_SVG_PATH     = './output/svg'

DATA_PROCESS_CHOICES = ["SvmA", "SvmW", "Lstm", "common"]  # ここでモデルの候補を定義
MODEL_CHOICES = ["SvmA", "SvmW", "Lstm", "RF"]  # ここでモデルの候補を定義

# Data process parameters
WINDOW_SIZE_SEC = 3   # Time window size in seconds
STEP_SIZE_SEC = 0.5     # Step size in seconds

SAMPLE_RATE_SIMLSL = 60 # sample rate for simlsl
WINDOW_SIZE_SAMPLE_SIMLSL = WINDOW_SIZE_SEC * SAMPLE_RATE_SIMLSL 
STEP_SIZE_SAMPLE_SIMLSL = int(STEP_SIZE_SEC * SAMPLE_RATE_SIMLSL) 

SAMPLE_RATE_EEG = 500
WINDOW_SIZE_SAMPLE_EEG = WINDOW_SIZE_SEC * SAMPLE_RATE_EEG 
STEP_SIZE_SAMPLE_EEG = int(STEP_SIZE_SEC * SAMPLE_RATE_EEG) 

# GHMスケーリングフィルタとウェーブレットフィルタの係数（厳密な係数を使用）
SCALING_FILTER = np.array([0.48296, 0.83652, 0.22414, -0.12941])
WAVELET_FILTER = np.array([-0.12941, -0.22414, 0.83652, -0.48296])
WAVELET_LEV = 3
