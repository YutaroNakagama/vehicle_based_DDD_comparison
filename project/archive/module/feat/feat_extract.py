import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import antropy as ant
import neurokit2 as nk
import collections
import librosa

from tqdm import tqdm
from darts import TimeSeries
from sklearn import preprocessing
from scipy import stats
from scipy.fftpack import fft, ifft, fftfreq

from module.preprocess import Preprocess

def feat_Arefnezhad2019(df_simlsl):
    # ---------------------------------------------------------- #
    #                  steering angle analysis                   #
    # ---------------------------------------------------------- #

    time_window_veh = 3 # sec
    sampling_freq_simlsl = 60  # Hz
    window_len = sampling_freq_simlsl * time_window_veh

    # preprocess = remove road curve 
    print(type(df_simlsl))
    df_simlsl["str_vel"] = df_simlsl["Steering_Wheel_Pos"].diff() / df_simlsl["row LSL time"].diff() 
    df_simlsl["road_curve"] =\
            df_simlsl["Steering_Wheel_Pos"].rolling(window_len).mean().shift(-window_len//2) 
    df_simlsl["str_no_road"] =\
            df_simlsl["Steering_Wheel_Pos"] - df_simlsl["road_curve"]

#    fig,ax = plt.subplots(nrows=2,ncols=1,figsize=(8,5),tight_layout=True)
#    df_simlsl.astype('float32').plot(ax=ax[0],x="row LSL time", y="Steering_Wheel_Pos")
#    df_simlsl.astype('float32').plot(ax=ax[1],x="row LSL time", y=["road_curve",'str_no_road'])
#    ax[0].set_xlim([500,550])
#    ax[1].set_xlim([500,550])
#    ax[0].set_ylim([-0.008,0.008])
#    ax[1].set_ylim([-0.008,0.008])
    #plt.show()

    # range = max - min
    df_simlsl["range_a"] =\
            df_simlsl["str_no_road"].rolling(window_len).max().shift(-window_len//2) -\
            df_simlsl["str_no_road"].rolling(window_len).min().shift(-window_len//2) 
    df_simlsl["range_v"] =\
            df_simlsl["str_vel"].rolling(window_len).max().shift(-window_len//2) -\
            df_simlsl["str_vel"].rolling(window_len).min().shift(-window_len//2) 

    # standard deviation
    df_simlsl["std_a"] = df_simlsl["str_no_road"].rolling(window_len).std().shift(-window_len//2)
    df_simlsl["std_v"] = df_simlsl["str_vel"].rolling(window_len).std().shift(-window_len//2)

    # energy 
    df_simlsl["power_str_no_road"] = df_simlsl["str_no_road"].abs()**2
    df_simlsl["power_str_vel"]     = df_simlsl["str_vel"].abs()**2
    df_simlsl["engy_a"] = df_simlsl["power_str_no_road"].rolling(window_len).mean().shift(-window_len//2)
    df_simlsl["engy_v"] = df_simlsl["power_str_vel"].rolling(window_len).mean().shift(-window_len//2)

    # zero crossing rate =  
    df_simlsl["str_dir"] = df_simlsl["str_no_road"].diff()/df_simlsl["str_no_road"].diff().abs() 
    df_simlsl["str_dir"] = df_simlsl["str_dir"].fillna(0) # replace nan to 0
    df_simlsl["str_dir_change"] = df_simlsl["str_dir"].diff().abs()/2
    df_simlsl["str_dir_change"] = df_simlsl["str_dir_change"].round()
    df_simlsl["zcr_a"] =\
            df_simlsl["str_dir_change"].rolling(window_len).sum().shift(-window_len//2) /\
            time_window_veh

    df_simlsl["str_vel_dir"] = df_simlsl["str_vel"].diff()/df_simlsl["str_vel"].diff().abs() 
    df_simlsl["str_vel_dir"] = df_simlsl["str_vel_dir"].fillna(0) # replace nan to 0
    df_simlsl["str_vel_dir_change"] = df_simlsl["str_vel_dir"].diff().abs()/2
    df_simlsl["str_vel_dir_change"] = df_simlsl["str_vel_dir_change"].round()
    df_simlsl["zcr_v"] =\
            df_simlsl["str_vel_dir_change"].rolling(window_len).sum().shift(-window_len//2) /\
            time_window_veh

    # First Quartile = (min + median)/2
    df_simlsl["fq_a"] = (\
            df_simlsl["str_no_road"].rolling(window_len).min().shift(-window_len//2) +\
            df_simlsl["str_no_road"].rolling(window_len).median().shift(-window_len//2)) / 2
    df_simlsl["fq_v"] = (\
            df_simlsl["str_vel"].rolling(window_len).min().shift(-window_len//2) +\
            df_simlsl["str_vel"].rolling(window_len).median().shift(-window_len//2)) / 2

    # Second Quartile = median
    df_simlsl["sq_a"] = df_simlsl["str_no_road"].rolling(window_len).median().shift(-window_len//2)
    df_simlsl["sq_v"] = df_simlsl["str_vel"].rolling(window_len).median().shift(-window_len//2)

    # Third Quartile = (median + max)/2
    df_simlsl["tq_a"] = (\
            df_simlsl["str_no_road"].rolling(window_len).max().shift(-window_len//2) +\
            df_simlsl["str_no_road"].rolling(window_len).median().shift(-window_len//2)) / 2
    df_simlsl["tq_v"] = (\
            df_simlsl["str_vel"].rolling(window_len).max().shift(-window_len//2) +\
            df_simlsl["str_vel"].rolling(window_len).median().shift(-window_len//2)) / 2

    # Katz Fractal Dimension
    df_simlsl["kfd_a"] = pd.DataFrame(np.zeros_like(df_simlsl["str_no_road"]))
    df_simlsl["kfd_v"] = pd.DataFrame(np.zeros_like(df_simlsl["str_vel"]))
    for i in tqdm(range(0, len(df_simlsl)-window_len, sampling_freq_simlsl)):
        df_simlsl["kfd_a"].iloc[i],_ =\
                nk.fractal_katz(df_simlsl["str_no_road"].iloc[i:i+window_len-1].to_numpy())
        df_simlsl["kfd_v"].iloc[i],_ =\
                nk.fractal_katz(df_simlsl["str_vel"].iloc[i:i+window_len-1].to_numpy())

    # Skewness
    df_simlsl["skw_a"] = pd.DataFrame(np.zeros_like(df_simlsl["str_no_road"]))
    df_simlsl["skw_v"] = pd.DataFrame(np.zeros_like(df_simlsl["str_vel"]))
    for i in tqdm(range(0, len(df_simlsl)-window_len, sampling_freq_simlsl)):
        df_simlsl["skw_a"].iloc[i] = stats.skew(df_simlsl["str_no_road"].iloc[i:i+window_len-1])
        df_simlsl["skw_v"].iloc[i] = stats.skew(df_simlsl["str_vel"].iloc[i:i+window_len-1])

    # Kurtosis
    df_simlsl["kurt_a"] = pd.DataFrame(np.zeros_like(df_simlsl["str_no_road"]))
    df_simlsl["kurt_v"] = pd.DataFrame(np.zeros_like(df_simlsl["str_vel"]))
    for i in tqdm(range(0, len(df_simlsl)-window_len, sampling_freq_simlsl)):
        df_simlsl["kurt_a"].iloc[i] = stats.kurtosis(df_simlsl["str_no_road"].iloc[i:i+window_len-1])
        df_simlsl["kurt_v"].iloc[i] = stats.kurtosis(df_simlsl["str_vel"].iloc[i:i+window_len-1])

    # sample entropy
    df_simlsl["samen_a"] = pd.DataFrame(np.zeros_like(df_simlsl["str_no_road"]))
    df_simlsl["samen_v"] = pd.DataFrame(np.zeros_like(df_simlsl["str_vel"]))
    for i in tqdm(range(0, len(df_simlsl)-window_len, sampling_freq_simlsl)):
        df_simlsl["samen_a"].iloc[i] = ant.sample_entropy(df_simlsl["str_no_road"].iloc[i:i+window_len-1])
        df_simlsl["samen_v"].iloc[i] = ant.sample_entropy(df_simlsl["str_vel"].iloc[i:i+window_len-1])
        #df_simlsl["samen_a"].iloc[i] =\
                #        nk.entropy_sample(df_simlsl["str_no_road"].fillna(0).iloc[i:i+window_len-1])
        #df_simlsl["samen_v"].iloc[i] =\
                #        nk.entropy_sample(df_simlsl["str_vel"].fillna(0).iloc[i:i+window_len-1])

    # shannon entropy
    df_simlsl["shen_a"] = pd.DataFrame(np.zeros_like(df_simlsl["str_no_road"]))
    df_simlsl["shen_v"] = pd.DataFrame(np.zeros_like(df_simlsl["str_vel"]))
    for i in tqdm(range(0, len(df_simlsl)-window_len, sampling_freq_simlsl)):
        _, freq_a = np.unique(df_simlsl["str_no_road"].iloc[i:i+window_len-1], return_counts=True)
        _, freq_v = np.unique(df_simlsl["str_vel"].iloc[i:i+window_len-1], return_counts=True)
        df_simlsl["shen_a"].iloc[i],_ = nk.entropy_shannon(freq_a)
        df_simlsl["shen_v"].iloc[i],_ = nk.entropy_shannon(freq_v)

    # Frequency variability 
    def spectral_variance(x, samplerate=sampling_freq_simlsl):
        magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
        length = len(x)
        freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1]) # positive frequencies
        avg = np.average(magnitudes, weights = freqs)
        dev = freqs * (magnitudes - avg) ** 2
        return dev.sum() / (freqs.sum() - 1) 

    df_simlsl["freqvar_a"] = pd.DataFrame(np.zeros_like(df_simlsl["str_no_road"]))
    df_simlsl["freqvar_v"] = pd.DataFrame(np.zeros_like(df_simlsl["str_vel"]))
    for i in tqdm(range(0, len(df_simlsl)-window_len, sampling_freq_simlsl)):
        df_simlsl["freqvar_a"].iloc[i] = spectral_variance(df_simlsl["str_no_road"].iloc[i:i+window_len-1])
        df_simlsl["freqvar_v"].iloc[i] = spectral_variance(df_simlsl["str_vel"].iloc[i:i+window_len-1])

    # spectral entropy
    df_simlsl["spen_a"] = pd.DataFrame(np.zeros_like(df_simlsl["str_no_road"]))
    df_simlsl["spen_v"] = pd.DataFrame(np.zeros_like(df_simlsl["str_vel"]))
    for i in tqdm(range(0, len(df_simlsl)-window_len, sampling_freq_simlsl)):
        df_simlsl["spen_a"].iloc[i] = ant.spectral_entropy(
                df_simlsl["str_no_road"].iloc[i:i+window_len-1],
                sampling_freq_simlsl,method='fft')
        df_simlsl["spen_v"].iloc[i] = ant.spectral_entropy(
                df_simlsl["str_vel"].iloc[i:i+window_len-1],
                sampling_freq_simlsl,method='fft')

                        # Spectral Flux
    def FeatureSpectralFlux(X, f_s=sampling_freq_simlsl):
        isSpectrum = X.ndim == 1
        if isSpectrum:
            X = np.expand_dims(X, axis=1)
        # difference spectrum (set first diff to zero)
        X = np.c_[X[:, 0], X]
        afDeltaX = np.diff(X, 1, axis=1)
        # flux
        vsf = np.sqrt((afDeltaX**2).sum(axis=0)) / X.shape[0]
        return np.squeeze(vsf) if isSpectrum else vsf

    df_simlsl["spfl_a"] = pd.DataFrame(np.zeros_like(df_simlsl["str_no_road"]))
    df_simlsl["spfl_v"] = pd.DataFrame(np.zeros_like(df_simlsl["str_vel"]))
    for i in tqdm(range(0, len(df_simlsl)-window_len, sampling_freq_simlsl)):
        fft_a = fft(df_simlsl["str_no_road"].iloc[i:i+window_len-1].values)
        fft_v = fft(df_simlsl["str_vel"].iloc[i:i+window_len-1].values)
        df_simlsl["spfl_a"].iloc[i] = FeatureSpectralFlux(fft_a)
        df_simlsl["spfl_v"].iloc[i] = FeatureSpectralFlux(fft_v)

    # Center of Gravity of Frequency
    def spectral_centroid(x, samplerate=sampling_freq_simlsl):
        magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
        length = len(x)
        freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1]) # positive frequencies
        return np.sum(magnitudes*freqs) / np.sum(magnitudes) # return weighted mean
    df_simlsl["cgf_a"] = pd.DataFrame(np.zeros_like(df_simlsl["str_no_road"]))
    df_simlsl["cgf_v"] = pd.DataFrame(np.zeros_like(df_simlsl["str_vel"]))
    for i in tqdm(range(0, len(df_simlsl)-window_len, sampling_freq_simlsl)):
        df_simlsl["cgf_a"].iloc[i] = spectral_centroid(df_simlsl["str_no_road"].iloc[i:i+window_len-1])
        df_simlsl["cgf_v"].iloc[i] = spectral_centroid(df_simlsl["str_vel"].iloc[i:i+window_len-1])

    # Dominant frequency
    def extract_peak_frequency(data, sampling_rate=sampling_freq_simlsl):
        fft_data = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data))
        peak_coefficient = np.argmax(np.abs(fft_data))
        peak_freq = freqs[peak_coefficient]
        return abs(peak_freq * sampling_rate)
    df_simlsl["domfreq_a"] = pd.DataFrame(np.zeros_like(df_simlsl["str_no_road"]))
    df_simlsl["domfreq_v"] = pd.DataFrame(np.zeros_like(df_simlsl["str_vel"]))
    for i in tqdm(range(0, len(df_simlsl)-window_len, sampling_freq_simlsl)):
        df_simlsl["domfreq_a"].iloc[i] =\
                extract_peak_frequency(df_simlsl["str_no_road"].iloc[i:i+window_len-1])
        df_simlsl["domfreq_v"].iloc[i] =\
                extract_peak_frequency(df_simlsl["str_vel"].iloc[i:i+window_len-1])

    # average of Frequency
    def spectral_average(x, samplerate=sampling_freq_simlsl):
        magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
        length = len(x)
        freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1]) # positive frequencies
        #print(freqs,magnitudes)
        return np.mean(freqs[magnitudes>0.1]) # return weighted mean
    df_simlsl["avepsd_a"] = pd.DataFrame(np.zeros_like(df_simlsl["str_no_road"]))
    df_simlsl["avepsd_v"] = pd.DataFrame(np.zeros_like(df_simlsl["str_vel"]))
    for i in tqdm(range(0, len(df_simlsl)-window_len, sampling_freq_simlsl)):
        df_simlsl["avepsd_a"].iloc[i] = spectral_average(df_simlsl["str_no_road"].iloc[i:i+window_len-1])
        df_simlsl["avepsd_v"].iloc[i] = spectral_average(df_simlsl["str_vel"].iloc[i:i+window_len-1])

    feat_list = [
    "range_a", "std_a", "engy_a", "zcr_a", "fq_a", "sq_a", "tq_a", 
    "range_v", "std_v", "engy_v", "zcr_v", "fq_v", "sq_v", "tq_v", 
    "kfd_a", "skw_a", "kurt_a", "samen_a", 
    "kfd_v", "skw_v", "kurt_v", "samen_v", 
    "freqvar_a", "spen_a",           "cgf_a",              "avepsd_a",
    "freqvar_v", "spen_v",           "cgf_v",              "avepsd_v",
    ]

    return df_simlsl, feat_list

def pe_2nd_taylor(df):
    df_pred =  df + df.diff() + 0.5 * df.diff().diff()
    return df_pred.shift(1) - df

def feat_Wang2022(simlsl_raw):
    dt_simlsl = simlsl_raw['row LSL time'].diff().mean() 

    # feature extraction
    win_size = 5
    simlsl_raw['speed_mean'] = \
            simlsl_raw['Velocity'].rolling(win_size, win_type='gaussian', center=True).mean(std=0.5) 
    simlsl_raw['lon_acc_mean'] = \
            simlsl_raw['Long_Accel'].rolling(win_size, win_type='gaussian', center=True).mean(std=0.5) 
    simlsl_raw['lat_acc_mean'] =\
            simlsl_raw['Lat_Accel'].rolling(win_size, win_type='gaussian', center=True).mean(std=0.5) 
    simlsl_raw['lane_off_mean'] = \
            simlsl_raw['Lane_Offset'].rolling(win_size, win_type='gaussian', center=True).mean(std=0.5) 
    simlsl_raw['str_rate_mean'] = \
            simlsl_raw['Steering_Wheel_Pos'].rolling(win_size, win_type='gaussian', center=True).mean(std=0.5) 

    print("std")
    simlsl_raw['speed_std']     = simlsl_raw['Velocity'].rolling(win_size, center=True).std() 
    simlsl_raw['lon_acc_std']   = simlsl_raw['Long_Accel'].rolling(win_size, center=True).std() 
    simlsl_raw['lat_acc_std']   = simlsl_raw['Lat_Accel'].rolling(win_size, center=True).std() 
    simlsl_raw['lane_off_std']  = simlsl_raw['Lane_Offset'].rolling(win_size, center=True).std() 
    simlsl_raw['str_rate_std']  = simlsl_raw['Steering_Wheel_Pos'].rolling(win_size, center=True).std() 

    print('pe')
    simlsl_raw['speed_pe']    = pe_2nd_taylor(simlsl_raw['Velocity'])
    simlsl_raw['lon_acc_pe']  = pe_2nd_taylor(simlsl_raw['Long_Accel'])
    simlsl_raw['lat_acc_pe']  = pe_2nd_taylor(simlsl_raw['Lat_Accel'])
    simlsl_raw['lane_off_pe'] = pe_2nd_taylor(simlsl_raw['Lane_Offset'])
    simlsl_raw['str_rate_pe'] = pe_2nd_taylor(simlsl_raw['Steering_Wheel_Pos'])

    print('DRT')
    # data clipping
    simlsl_raw['DRT_event'] = (simlsl_raw.DRT_Resp_Time/simlsl_raw.DRT_Resp_Time).fillna(0) 

    simlsl_raw['DRT_baseline1'] = simlsl_raw['DRT_event'].copy()  
    for i in tqdm(range(int(20//dt_simlsl))): 
        simlsl_raw['DRT_baseline1'] = simlsl_raw.DRT_baseline1 + simlsl_raw.DRT_baseline1.shift(-1)

    simlsl_raw['DRT_baseline2'] = simlsl_raw['DRT_event'].copy()     
    for i in tqdm(range(int(10//dt_simlsl))): 
        simlsl_raw['DRT_baseline2'] = simlsl_raw.DRT_baseline2 + simlsl_raw.DRT_baseline2.shift(1)

    simlsl_raw.DRT_baseline1 = simlsl_raw.DRT_baseline1.clip(0,1)
    simlsl_raw['DRT_baseline'] = simlsl_raw.DRT_baseline1 + simlsl_raw.DRT_baseline2
    simlsl_raw.DRT_baseline = simlsl_raw.DRT_baseline.clip(0,1) - simlsl_raw.DRT_event 

    simlsl_raw['speed'] = simlsl_raw.Route_Position.diff()/dt_simlsl

#    fig,ax = plt.subplots(5,1,tight_layout=True)
#    simlsl_raw.plot(ax=ax[0],x='row LSL time',y=['Long_Accel'])
#    simlsl_raw.plot(ax=ax[1],x='row LSL time',y=['Lat_Accel'])
#    #simlsl_raw.plot(ax=ax[0],x='row LSL time',y='speed')
#    #ax[0].set_ylim([-20,20])
#    simlsl_raw.plot(ax=ax[2],x='row LSL time',y=['Lane_Offset'])
#    simlsl_raw.plot(ax=ax[3],x='row LSL time',y=['Steering_Wheel_Pos'])
#    simlsl_raw.plot(ax=ax[4],x='row LSL time',y=['DRT_event','DRT_baseline'])
#    #ax[3].set_ylim([0,2])
#
#    for i in range(len(ax)):
#        ax[i].set_xlim([280,320])
#    plt.show()

    return simlsl_raw

def feat_Zhao2009(data_raw, veh_raw):

    import sys
    import time
    import itertools
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    # https://github.com/PyWavelets/pywt
    import pywt 
    # https://github.com/Timorleiderman/tensorflow-wavelets 
    import tensorflow_wavelets.Layers.DMWT as DMWT

    subj_all = (data_raw.subj.drop_duplicates().to_list())
    # get min size of each subject data len for db6
    size_min=80000    
    for s in subj_all:
        str_ang = veh_raw[veh_raw.subj==s].fillna(0).Steering_Wheel_Pos.to_numpy()
        size = len(str_ang) #2048
        if size_min > size:
            size_min = size
    print("size_min = ", size_min)

    wav_packet_energy_db6 = np.repeat(0, len(subj_all)*8).reshape(len(subj_all),8)
    wav_packet_energy_ghm = np.repeat(0, len(subj_all)*8*8).reshape(len(subj_all),8*8)
    eeg_gt = "alpha_p_beta"
    gt_df = []
    i_temp = 0
    for s in subj_all:
        # make grand truth 
        eeg_gt_temp = data_raw[data_raw.subj==s]
        gt_df.append(eeg_gt_temp[eeg_gt].mean())

        # --------- feature extraction from steering angle dataset --------
        # db6
        str_ang = veh_raw[veh_raw.subj==s].fillna(0).Steering_Wheel_Pos.to_numpy()
        str_ang_len = len(str_ang)//size_min
        resized_str_ang = np.reshape(str_ang[len(str_ang)%size_min:], (str_ang_len,size_min))
    
        wp = pywt.WaveletPacket(resized_str_ang[0,:], wavelet = "db6", maxlevel = 3) 
        packet_names = [node.path for node in wp.get_level(3, "natural")] # Packet node names.
        for j in range(8):
            new_wp = pywt.WaveletPacket(data = None, wavelet = "db6", maxlevel = 3)
            new_wp[packet_names[j]] = wp[packet_names[j]].data
            wav_packet_energy_db6[i_temp,j] = np.linalg.norm(wp[packet_names[j]].data)**2 

        # GHM: https://github.com/Timorleiderman/tensorflow-wavelets/tree/main
        if len(veh_raw[veh_raw.subj==s]) > 256**2:
            str_ang = veh_raw[veh_raw.subj==s].iloc[:256**2].fillna(0).Steering_Wheel_Pos.to_numpy()
            resized_str_ang = np.reshape(str_ang, (1,256,256,1))
            dmwt = DMWT.DMWT('ghm')(resized_str_ang)
            dmwt2 = DMWT.DMWT('ghm')(dmwt)
            dmwt3 = DMWT.DMWT('ghm')(dmwt2)
            print(dmwt3.shape)
            for i,j in itertools.product(range(8),range(8)):
                wav_packet_energy_ghm[i_temp,i*8+j] = \
                       np.linalg.norm(dmwt3[0,i*256:(i+1)*256, j*256:(j+1)*256,0].numpy().reshape(-1))**2
        i_temp += 1

    X_db6 = wav_packet_energy_db6 
    X_ghm = wav_packet_energy_ghm 

    Y_temp = pd.DataFrame(gt_df) 
    Y = Y_temp.where(Y_temp>1,0)
    Y = Y.where(Y_temp<1,1)

    return X_db6, X_ghm, Y
