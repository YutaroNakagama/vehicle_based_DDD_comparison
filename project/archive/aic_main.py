import numpy as np
#import pylab
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as st
import sys
import sklearn.metrics as mt
#import keras
import scipy as sp
import pandas as pd
import datetime

from tqdm import tqdm

from pandas import read_csv
from random import gauss

from statsmodels.tsa.ar_model import AR

from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler

from matplotlib import animation as ani

#from keras import backend as K
#from keras.models import Model, Sequential
#from keras.layers import Input, LSTM, GRU, RepeatVector, Activation
#from tensorflow.python.keras.layers.core import Dense, Lambda

#from tensorflow.keras.optimizers import Adam
#from keras import objectives

from copy import copy 

from dataclasses import dataclass

from typing import List, Optional

from scipy.interpolate import interp1d
from scipy.stats import spearmanr
from scipy import signal

from statistics import stdev, variance, median

#aic
from typing import List, Tuple

from numpy.typing import NDArray

from local_lib.preprocess import predict, butter_lowpass,butter_lowpass_filter,\
                                 execute_regression, calculate_mse, curve_check,\
                                 calculate_aic_and_bic,\
                                 save_aic_and_bic,\
                                 make_aic_and_bic#,\
                                 #load_dataset

plt.style.use('ggplot')

#plt.rcParams['font.family'] = 'Times New Roman' # font family
#plt.rcParams['mathtext.fontset'] = 'stix' # math font
#plt.rcParams["font.size"] = 12 
#plt.rcParams['xtick.labelsize'] = 9 
#plt.rcParams['ytick.labelsize'] = 24 
#plt.rcParams['xtick.direction'] = 'in' # x axis in
#plt.rcParams['ytick.direction'] = 'in' # y axis in 
#plt.rcParams['axes.linewidth'] = 1.0 # axis line width
plt.rcParams['axes.grid'] = True  # make grid
#plt.rcParams["legend.fancybox"] = False
#plt.rcParams["legend.framealpha"] = 1 #
#plt.rcParams["legend.edgecolor"] = 'black' # edge
#plt.rcParams["legend.handlelength"] = 1 
#plt.rcParams["legend.labelspacing"] = 5. 
#plt.rcParams["legend.handletextpad"] = 3. 
#plt.rcParams["legend.markerscale"] = 2 # 
#plt.rcParams["legend.borderaxespad"] = 0. #

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams['mathtext.fontset'] = 'stix'  # The setting of math font

plt.rcParams["legend.fancybox"] = False # 
plt.rcParams["legend.framealpha"] = 1 # 
plt.rcParams["legend.edgecolor"] = 'black' # edge
plt.rcParams["legend.handlelength"] = 2 #
plt.rcParams["legend.labelspacing"] = 0. # 
plt.rcParams["legend.handletextpad"] = 0.5 # 
plt.rcParams["legend.markerscale"] = 2 #e

plt.rcParams['xtick.direction'] = 'in'#x
plt.rcParams['ytick.direction'] = 'in'#
plt.rcParams['xtick.major.width'] = 1.0#
plt.rcParams['ytick.major.width'] = 1.0#
plt.rcParams['axes.linewidth'] = 1.0# 
plt.rcParams['lines.linewidth'] = 1.0#

str_today = str(datetime.datetime.utcnow().date())


# LPF  
csv_path_veh = ["" for x in range(8)]
csv_path_eeg = ["" for x in range(8)]

dataset_fp = "../dataset/bz2/"
dataset_format = "bz2" # "csv"

csv_path_veh[0] = dataset_fp + "dataset_pandas_2023-09-10_1404_20230930."   + dataset_format
csv_path_veh[1] = dataset_fp + "dataset_pandas_2023-11-14_0025_2023-11-14." + dataset_format
csv_path_veh[2] = dataset_fp + "dataset_pandas_2023-11-15_0115_2023-11-15." + dataset_format
csv_path_veh[3] = dataset_fp + "dataset_pandas_2023-11-19_2306_2023-11-25." + dataset_format
csv_path_veh[4] = dataset_fp + "dataset_pandas_2023-11-21_0004_2023-11-26." + dataset_format
csv_path_veh[5] = dataset_fp + "dataset_pandas_2023-11-26_0959_2023-11-26." + dataset_format
csv_path_veh[6] = dataset_fp + "dataset_pandas_2023-11-26_1346_2023-11-26." + dataset_format
csv_path_veh[7] = dataset_fp + "dataset_pandas_2023-12-02_1003_2023-12-02." + dataset_format

csv_path_eeg[0] = dataset_fp + "dataset_EEG_2023-09-10_1404_20230909."   + dataset_format
csv_path_eeg[1] = dataset_fp + "dataset_EEG_2023-11-14_0022_20231104."   + dataset_format
csv_path_eeg[2] = dataset_fp + "dataset_EEG_2023-11-15_0113_2023-11-15." + dataset_format
csv_path_eeg[3] = dataset_fp + "dataset_EEG_2023-11-19_2303_2023-11-25." + dataset_format
csv_path_eeg[4] = dataset_fp + "dataset_EEG_2023-11-21_0001_2023-11-26." + dataset_format
csv_path_eeg[5] = dataset_fp + "dataset_EEG_2023-11-26_0959_2023-11-26." + dataset_format
csv_path_eeg[6] = dataset_fp + "dataset_EEG_2023-11-26_1347_2023-11-26." + dataset_format
csv_path_eeg[7] = dataset_fp + "dataset_EEG_2023-12-02_1003_2023-12-02." + dataset_format

time_str         = []
yaw_rate_i       = []   
yaw_rate_a       = []   
angular_velocity = []
curvature        = []
steering_angle   = []
dev_center       = []

time_eeg  = [] 
alpha_af7 = [] 
alpha_af8 = [] 
beta_af7  = [] 
beta_af8  = [] 

for i in range(len(csv_path_veh)):
#for i in range(2):
    csv_df_veh = pd.read_pickle(csv_path_veh[i]).query('-5<yaw_rate_a<5').query('-5<yaw_rate_i<5')
    csv_df_eeg = pd.read_pickle(csv_path_eeg[i])
    
    time_str         = np.concatenate((time_str        ,csv_df_veh['time'].values      ), axis = 0) 
    yaw_rate_i       = np.concatenate((yaw_rate_i      ,csv_df_veh['yaw_rate_i'].values), axis = 0) 
    yaw_rate_a       = np.concatenate((yaw_rate_a      ,csv_df_veh['yaw_rate_a'].values), axis = 0) 
    angular_velocity = np.concatenate((angular_velocity,csv_df_veh['angular_vel']      ), axis = 0) 
    curvature        = np.concatenate((curvature       ,csv_df_veh['curvature']        ), axis = 0) 
    steering_angle   = np.concatenate((steering_angle  ,csv_df_veh['steering_angle']   ), axis = 0) 
    dev_center       = np.concatenate((dev_center      ,csv_df_veh['dev_center']       ), axis = 0) 

    time_eeg  = np.concatenate((time_str        ,csv_df_eeg['time'].values     ), axis = 0) 
    alpha_af7 = np.concatenate((yaw_rate_i      ,csv_df_eeg['alpha_af7'].values), axis = 0) 
    alpha_af8 = np.concatenate((yaw_rate_a      ,csv_df_eeg['alpha_af8'].values), axis = 0) 
    beta_af7  = np.concatenate((angular_velocity,csv_df_eeg['beta_af7'].values ), axis = 0) 
    beta_af8  = np.concatenate((curvature       ,csv_df_eeg['beta_af8'].values ), axis = 0) 

alpha_local      = (alpha_af7 + alpha_af8)/2
beta_local       = (beta_af7 + beta_af8)/2
beta_alpha_local = beta_local / alpha_local
alpha_ave        = []
beta_ave         = []
beta_alpha_ave   = []

N = len(time_str)               # number of sample 
N_curve = 2**9                  #len(time[i-int(2/dt):i+int(5/dt)]) 
dt = time_str[1]-time_str[0]    # sampling time [s]
dt_eeg = 10 * 0.0001            #time_eeg[1]-time_eeg[0]  # sampling time [s]
fs = 1/dt

save_fig_fft = 0
if save_fig_fft==1:
    fig, axes = plt.subplots(4, 1, tight_layout=True)

in_curve_flg = 0
in_curve_cnt = 0
curve_cnt = 0
Amp_i_all = []
Amp_a_all = []

for i in tqdm(range(len(time_str)+int(3/dt)-N_curve)):
    if in_curve_flg==1: 
        in_curve_cnt=in_curve_cnt+1
    if in_curve_cnt>3//dt:
        in_curve_flg,in_curve_cnt=0,0
    early_stop = False
    if time_str[i]>69 and early_stop==True:
        break
    if curve_check(
        yaw_rate_i[i          :i+int(1/dt)],
        yaw_rate_i[i+int(1/dt):i+int(4/dt)],
        yaw_rate_a[i          :i+int(fs*5)],
        in_curve_flg,
        dt
    ):
        in_curve_flg=1
        curve_cnt = curve_cnt + 1
        index_start_eeg = int(255.5*(1.0+time_str[i-int(3/dt)]))
        index_end_eeg   = int(255.5*(1.0+time_str[i-int(3/dt)+N_curve]))

        alpha_ave      = np.nanmean(alpha_local[     index_start_eeg:index_end_eeg] ,axis=0, dtype=np.float16)
        beta_ave       = np.nanmean(beta_local[      index_start_eeg:index_end_eeg] ,axis=0, dtype=np.float16)
        beta_alpha_ave = np.nanmean(beta_alpha_local[index_start_eeg:index_end_eeg] ,axis=0, dtype=np.float16)

        # FFT (original data)   
        F_i = np.fft.fft(yaw_rate_i[i-int(3/dt):i-int(3/dt)+N_curve]) 
        F_a = np.fft.fft(yaw_rate_a[i-int(3/dt):i-int(3/dt)+N_curve]) 
        freq = np.fft.fftfreq(N_curve, d=dt)
        
        F_i = F_i / (N_curve / 2)
        F_a = F_a / (N_curve / 2)

        if save_fig_fft==1:
            axes[0].plot(
                time_str[i-int(3/dt):i-int(3/dt)+N_curve], 
                yaw_rate_i[i-int(3/dt):i-int(3/dt)+N_curve], 
                label="ideal"
            )
            axes[0].plot(
                time_str[i-int(3/dt):i-int(3/dt)+N_curve], 
                yaw_rate_a[i-int(3/dt):i-int(3/dt)+N_curve], 
                label="actual(max=%.4f)"%max(abs(yaw_rate_a[i:i+int(fs*10)]))
            )
            axes[0].set_xlabel("time") 
            axes[0].set_ylabel("yaw rate")
                
        Amp_i = np.abs(F_i)
        Amp_a = np.abs(F_a)

        if save_fig_fft==1:
            axes[1].plot(freq[:N_curve//2], Amp_i[:N_curve//2], label="ideal")
            axes[1].plot(freq[:N_curve//2], Amp_a[:N_curve//2], label="actual")
            axes[1].set_xlabel("Frequency [Hz]")
            axes[1].set_ylabel("Amplitude")
            axes[1].set_xlim(0,1/dt/2)

        if curve_cnt==1:
#            Amp_i_all = np.hstack(["frequency",freq[:N_curve//2]])
#            Amp_a_all = np.hstack(["frequency",freq[:N_curve//2]])
            Amp_i_all = pd.DataFrame({'frequency':freq[:N_curve//2]})
            Amp_a_all = pd.DataFrame({'frequency':freq[:N_curve//2]})

        Amp_i_temp = pd.DataFrame({'curve_'+str(curve_cnt):Amp_i[:N_curve//2]})
        Amp_a_temp = pd.DataFrame({'curve_'+str(curve_cnt):Amp_a[:N_curve//2]})

        Amp_i_all = pd.concat([Amp_i_all,Amp_i_temp],axis=1)
        Amp_a_all = pd.concat([Amp_a_all,Amp_a_temp],axis=1)

        # LPF (original data) 
        #print("LPF")
        filt_type = "LPF"
        if filt_type == "LPF":
            # LPF
            freq_th = 1
            yaw_rate_a_filt = butter_lowpass_filter(
                yaw_rate_a[i-int(3/dt):i-int(3/dt)+N_curve], 
                freq_th, 
                fs, 
                order=4
            )
            yaw_rate_i_filt = butter_lowpass_filter(
                yaw_rate_i[i-int(3/dt):i-int(3/dt)+N_curve], 
                freq_th, 
                fs, 
                order=4
            )
        elif filt_type == "MA":
            # MA
            n_conv = 2
            b = np.ones(n_conv) / n_conv # >> [1/n_size, 1/n_size, ..., 1/n_size]
            yaw_rate_a_filt = np.convolve(
                yaw_rate_a[i-int(3/dt):i-int(3/dt)+N_curve], 
                b, 
                mode="same"
            )
        if save_fig_fft==1:
            axes[0].plot(
                time_str[i-int(3/dt):i-int(3/dt)+N_curve], 
                yaw_rate_a_filt, 
                label="actual_filt"
            )
            axes[0].plot(time_str[i-int(3/dt):i-int(3/dt)+N_curve], yaw_rate_i_filt, label="ideal_filt")

            axes[2].plot(freq[:N_curve//2], Amp_i_filt[:N_curve//2], label="ideal_filt") 
            axes[2].plot(freq[:N_curve//2], Amp_a_filt[:N_curve//2], label="actual_filt")
            axes[2].set_xlabel("Frequency [Hz]")
            axes[2].set_ylabel("Amplitude")
            axes[2].set_xlim(0,1/dt/2)
    
            axes[3].plot(time_eeg[   index_start_eeg:index_end_eeg], 
                         alpha_local[index_start_eeg:index_end_eeg], 
                         label="alpha") 
            axes[3].plot(time_eeg[  index_start_eeg:index_end_eeg], 
                         beta_local[index_start_eeg:index_end_eeg], 
                         label="beta") 
            axes[3].plot(time_eeg[        index_start_eeg:index_end_eeg], 
                         beta_alpha_local[index_start_eeg:index_end_eeg], 
                         label="beta/alpha") 
            axes[3].set_xlabel("time")
            axes[3].set_ylabel("Amplitude")

            axes[0].legend(loc='upper right')
            axes[1].legend()
            axes[2].legend()
            axes[3].legend()
            
            plt.savefig("fig/fft_fc_"+str(freq_th)+"_"\
                         + str(int(1000*time_str[i-int(3/dt)])) + "_"\
                         + str_today+".png",\
                         format="png", dpi=1600)
            #plt.show()
            axes[0].cla()
            axes[1].cla()
            axes[2].cla()
            axes[3].cla()
            #plt.close()

        MAX_DEGREE = 10
        make_aic_and_bic(curve_cnt, time_str[i-int(3/dt):i-int(3/dt)+N_curve], yaw_rate_a_filt, N_curve, MAX_DEGREE,"yawrate")

save_amp_all = False
if save_amp_all == True:
    Amp_i_all.to_csv('./csv/Amp_i_all_'+str(freq_th)+'_'+str_today+'.csv')
    Amp_a_all.to_csv('./csv/Amp_a_all_'+str(freq_th)+'_'+str_today+'.csv')

save_eeg = False
if save_eeg == True:
    csv_df = pd.DataFrame(data={
                          'alpha_ave': pd.Series(alpha_ave),     
                          'beta_ave': pd.Series(beta_ave),
                          'beta_alpha_ave': pd.Series(beta_alpha_ave)
                          })     
                                       
    csv_path = "./result/csv/aic/eeg_local_"+str_today+".csv"
    csv_df.to_csv(csv_path,mode='a', header=True) 
