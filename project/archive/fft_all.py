# https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html

import warnings
warnings.filterwarnings("ignore")

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq

from tqdm import tqdm

# class in other files
from module.preprocess import Preprocess
from module.dataset import Datasets
from module.train import PredictSimpleFormulaNet, Train 

if __name__ == '__main__':
    start = time.time()

    # load dataset
    dataset = Datasets()
    df_filt, curv_time = dataset.load_veh(output_type="curve", filt=False)
    df_eeg = dataset.load_eeg(data_type="curve",curv_time = curv_time)

    print(df_eeg.head())
    plt.hist(df_eeg["alpha_af7"], alpha=0.5, label="alpha_af7")
    plt.hist(df_eeg["alpha_af8"], alpha=0.5, label="alpha_af8")
    plt.hist(df_eeg["alpha_tp9"], alpha=0.5, label="alpha_tp9")
    plt.hist(df_eeg["alpha_tp10"],alpha=0.5, label="alpha_tp10")
    plt.legend()
    plt.show()

    plt.hist(df_eeg["beta_af7"], alpha=0.5, label="beta_af7")
    plt.hist(df_eeg["beta_af8"], alpha=0.5, label="beta_af8")
    plt.hist(df_eeg["beta_tp9"], alpha=0.5, label="beta_tp9")
    plt.hist(df_eeg["beta_tp10"],alpha=0.5, label="beta_tp10")
    plt.legend()
    plt.savefig("beta.pdf")
    plt.show()

    df_eeg["lane_dev"] = df_eeg["yaw_rate_a"] - df_eeg["yaw_rate_i"]

    cols = ["time","yaw_rate_i","yaw_rate_a","lane_dev","curv_cnt",\
            "alpha_af7","alpha_af8","alpha_tp9","alpha_tp10",\
            "beta_af7","beta_af8","beta_tp9","beta_tp10"] 
    df_curv = pd.DataFrame(columns=cols, dtype=object)
    #df_curv = pd.DataFrame()
    print("before",df_curv.head())

    for cnt in tqdm(range(max(df_filt["curv_cnt"]))):
        df_curv.loc[str(cnt)] = df_eeg[df_eeg["curv_cnt"]==cnt].mean()

    fig = plt.figure(figsize=(12, 5))
    ax_1 = fig.add_subplot(2, 4, 1)
    ax_2 = fig.add_subplot(2, 4, 2)
    ax_3 = fig.add_subplot(2, 4, 3)
    ax_4 = fig.add_subplot(2, 4, 4)
    ax_5 = fig.add_subplot(2, 4, 5)
    ax_6 = fig.add_subplot(2, 4, 6)
    ax_7 = fig.add_subplot(2, 4, 7)
    ax_8 = fig.add_subplot(2, 4, 8)

    ax_1.scatter(df_curv["lane_dev"],df_curv["alpha_af7"])
    ax_2.scatter(df_curv["lane_dev"],df_curv["alpha_af8"])
    ax_3.scatter(df_curv["lane_dev"],df_curv["alpha_tp9"])
    ax_4.scatter(df_curv["lane_dev"],df_curv["alpha_tp10"])
    ax_5.scatter(df_curv["lane_dev"],df_curv["beta_af7"])
    ax_6.scatter(df_curv["lane_dev"],df_curv["beta_af8"])
    ax_7.scatter(df_curv["lane_dev"],df_curv["beta_tp9"])
    ax_8.scatter(df_curv["lane_dev"],df_curv["beta_tp10"])

    plt.show()

    fig = plt.figure(figsize=(12, 5))
    ax_1 = fig.add_subplot(3, 1, 1)
    ax_2 = fig.add_subplot(3, 2, (3,5))
    ax_3 = fig.add_subplot(3, 2, (4,6))

    df_fft = pd.DataFrame()
    sig_fft_all_drow = []
    sig_fft_all_awak = []
    for cnt in tqdm(df_curv[df_curv["beta_af7"] > 0.9]["curv_cnt"].tolist()):
        x = df_filt[df_filt["curv_cnt"]==cnt]["yaw_rate_a"].values
        t = df_filt[df_filt["curv_cnt"]==cnt]["time"].values
        d = df_filt[df_filt["curv_cnt"]==cnt]["time"].diff().mean()
    
        # FFT the signal
        sig_fft = fft(x)
        sig_fft_all_awak.append(sig_fft)
        # copy the FFT results
        sig_fft_filtered = sig_fft.copy()
    
        # obtain the frequencies using scipy function
        freq = fftfreq(len(x), d=d)
    
        # define the cut-off frequency
        cut_off = 5
    
        # high-pass filter by assign zeros to the 
        # FFT amplitudes where the absolute 
        # frequencies smaller than the cut-off 
        sig_fft_filtered[np.abs(freq) > cut_off] = 0
    
        # get the filtered signal in time domain
        filtered = ifft(sig_fft_filtered)
    
        # plot the filtered signal
    
        # plot the FFT amplitude before and after
        ax_2.plot(freq[:int(len(freq)/2)], np.abs(sig_fft[:int(len(freq)/2)]), 'b')
        #ax_2.stem(freq, np.abs(sig_fft), 'b', markerfmt=" ", basefmt="-b")
        ax_2.set_title('Before filtering')
        ax_2.set_xlim(0, .5/d)
        ax_2.set_yscale("log")
        ax_2.set_xlabel('Frequency (Hz)')
        ax_2.set_ylabel('FFT Amplitude')
    
        ax_3.plot(freq[:int(len(freq)/2)], np.abs(sig_fft_filtered[:int(len(freq)/2)]), 'b')
        #ax_3.stem(freq, np.abs(sig_fft_filtered), 'b', markerfmt=" ", basefmt="-b")
        ax_3.set_title('After filtering')
        ax_3.set_xlim(0, .5/d)
        ax_3.set_yscale("log")
        ax_3.set_xlabel('Frequency (Hz)')
        ax_3.set_ylabel('FFT Amplitude')

    ax_1.plot(t, x, label="before filtering")
    ax_1.plot(t, filtered, label="after filtering")
    ax_1.legend()
    ax_1.set_xlabel('Time (s)')
    ax_1.set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig("../result/visualisation/fft/fft.pdf")

    plt.show()

    for cnt in tqdm(df_curv[df_curv["beta_af7"] <= 0.3]["curv_cnt"].tolist()):
        x = df_filt[df_filt["curv_cnt"]==cnt]["yaw_rate_a"].values
        t = df_filt[df_filt["curv_cnt"]==cnt]["time"].values
        d = df_filt[df_filt["curv_cnt"]==cnt]["time"].diff().mean()
    
        # FFT the signal
        sig_fft = fft(x)
        sig_fft_all_drow.append(sig_fft)

    fft_all_awake = pd.DataFrame(sig_fft_all_awak)
    fft_all_drow  = pd.DataFrame(sig_fft_all_drow)
    fft_all_awake.loc["ave"] = fft_all_awake.abs().mean()
    fft_all_awake.loc["std"] = fft_all_awake.abs().std()
    fft_all_drow.loc["ave"] = fft_all_drow.abs().mean()
    fft_all_drow.loc["std"] = fft_all_drow.abs().std()

    plt.plot(freq[:int(len(freq)/2)], fft_all_awake.loc["ave"][:int(len(freq)/2)], label="ave (awake)")

    plt.plot(freq[:int(len(freq)/2)], 
             fft_all_awake.loc["ave"][:int(len(freq)/2)] + fft_all_awake.loc["std"][:int(len(freq)/2)], 
             label="ave+std (awake)")

    plt.plot(freq[:int(len(freq)/2)],
             fft_all_awake.loc["ave"][:int(len(freq)/2)] - fft_all_awake.loc["std"][:int(len(freq)/2)], 
             label="ave-std (awake)")

    plt.plot(freq[:int(len(freq)/2)], fft_all_drow.loc["ave"][:int(len(freq)/2)], label="ave (drow)")

    plt.plot(freq[:int(len(freq)/2)], 
             fft_all_drow.loc["ave"][:int(len(freq)/2)] + fft_all_drow.loc["std"][:int(len(freq)/2)], 
             label="ave+std (drow)")

    plt.plot(freq[:int(len(freq)/2)],
             fft_all_drow.loc["ave"][:int(len(freq)/2)] - fft_all_drow.loc["std"][:int(len(freq)/2)], 
             label="ave-std (drow)")

    plt.xlim(0, .5/d)
    plt.yscale("log")
    plt.legend()
    plt.savefig("fft.pdf")
    plt.show()

    print("elapse time", time.time()-start)
