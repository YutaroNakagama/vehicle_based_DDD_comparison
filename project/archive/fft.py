# https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html

import warnings
warnings.filterwarnings("ignore")

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq

from tqdm import tqdm_notebook as tqdm

# class in other files
from module.preprocess import Preprocess
from module.dataset import Datasets
from module.train import PredictSimpleFormulaNet, Train 

if __name__ == '__main__':
    start = time.time()

    # load dataset
    dataset = Datasets()
    df_filt, curv_time = dataset.load_veh(output_type="curve", filt=False)
    #df_eeg, curv_time_drow = dataset.load_eeg(curv_time = curv_time)

    x = df_filt[df_filt["curv_cnt"]==0]["yaw_rate_a"].values
    t = df_filt[df_filt["curv_cnt"]==0]["time"].values
    d = df_filt[df_filt["curv_cnt"]==0]["time"].diff().mean()

    # FFT the signal
    sig_fft = fft(x)
    # copy the FFT results
    sig_fft_filtered = sig_fft.copy()

    # obtain the frequencies using scipy function
    freq = fftfreq(len(x), d=d)

    # define the cut-off frequency
    cut_off = 2

    # high-pass filter by assign zeros to the 
    # FFT amplitudes where the absolute 
    # frequencies smaller than the cut-off 
    sig_fft_filtered[np.abs(freq) > cut_off] = 0

    # get the filtered signal in time domain
    filtered = ifft(sig_fft_filtered)

    # plot the filtered signal
    fig = plt.figure(figsize=(12, 5))

    ax_1 = fig.add_subplot(3, 1, 1)
    ax_2 = fig.add_subplot(3, 2, (3,5))
    ax_3 = fig.add_subplot(3, 2, (4,6))

    ax_1.plot(t, x, label="before filtering")
    ax_1.plot(t, filtered, label="after filtering")
    ax_1.legend()
    ax_1.set_xlabel('Time (s)')
    ax_1.set_ylabel('Amplitude')

    # plot the FFT amplitude before and after
    ax_2.stem(freq, np.abs(sig_fft), 'b', \
          markerfmt=" ", basefmt="-b")
    ax_2.set_title('Before filtering')
    ax_2.set_xlim(0, .5/d)
    ax_2.set_xlabel('Frequency (Hz)')
    ax_2.set_ylabel('FFT Amplitude')

    ax_3.stem(freq, np.abs(sig_fft_filtered), 'b', markerfmt=" ", basefmt="-b")
    ax_3.set_title('After filtering')
    ax_3.set_xlim(0, .5/d)
    ax_3.set_xlabel('Frequency (Hz)')
    ax_3.set_ylabel('FFT Amplitude')

    plt.tight_layout()
    plt.savefig("../result/visualisation/fft/fft.pdf")

    plt.show()

    print("elapse time", time.time()-start)
