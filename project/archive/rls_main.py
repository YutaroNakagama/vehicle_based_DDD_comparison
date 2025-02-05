
import datetime
import time
import sys

import control
import numpy as np
import pylab
import pandas as pd
import scipy as sp
import scipy.stats as st
import statsmodels.api as sm
import sklearn.metrics as mt
#import keras
import matplotlib.pyplot as plt

#from numpy.typing import NDArray
from control.matlab import * 
from pandas import read_csv, read_pickle
from random import gauss
from matplotlib import animation as ani

#from keras.models import Model, Sequential
#from keras.layers import Input, LSTM, GRU, RepeatVector, Activation
#from tensorflow.python.keras.layers.core import Dense, Lambda

#from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.ar_model import AR
from scipy.interpolate import interp1d
from scipy.stats import spearmanr
from scipy import signal
from statistics import stdev, variance, median
from sklearn.metrics import roc_curve, auc, mean_squared_error

from copy import copy 
from dataclasses import dataclass
#from typing import List, Optional, Tuple
from tqdm import tqdm

# local lib
from local_lib.preprocess import calc_fft, butter_lowpass,butter_lowpass_filter,\
                                 filtering_process,\
                                 curve_check, load_dataset, curve_detection,\
                                 save_dataset_csv, save_rmse_csv,\
                                 trim_mean, append_mean_trim, trim_idx

plt.style.use('ggplot')

plt.rcParams['axes.grid'] = True  # make grid
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams['mathtext.fontset'] = 'stix'  # The setting of math font
plt.rcParams["legend.fancybox"] = False 
plt.rcParams["legend.framealpha"] = 1 
plt.rcParams["legend.edgecolor"] = 'black' 
plt.rcParams["legend.handlelength"] = 2 
plt.rcParams["legend.labelspacing"] = 0. 
plt.rcParams["legend.handletextpad"] = 0.5 
plt.rcParams["legend.markerscale"] = 2 
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['lines.linewidth'] = 1.0

def rls(theta_, pn_, rn_, yn, zn): 
    trzn = np.reshape(zn, (1, 8))
    rls_num = rn_ + trzn @ pn_ @ zn
    pn = 1 / rn_ * (pn_ - (pn_ @ zn @ trzn @ pn_) / rls_num)
    ln = (pn_ @ zn) / rls_num
    en = yn - trzn @ theta_
    theta = theta_ + ln * en
    return theta, pn
 
def print_stdio(process_name:str):
    if process_name == "curve_detect":
        print("###############################")
        print("####    Curve detection    ####")
        print("###############################")
    if process_name == "train_focus":
        print("\n############################################")
        print("####   Training with focused datasets   ####")
        print("############################################")
    if process_name == "test_drow":
        print("###########################################")
        print("####   Test with drowsiness datasets   ####")
        print("###########################################")
        print("########################################")
        print("####   Test with focused datasets   ####")
        print("########################################")
    if process_name == "test_drow_focus":
        print("###############################################")
        print("####   Test with drow & focused datasets   ####")
        print("###############################################")

def save_system_csv(
    time_str_system,
    a1_system,
    a2_system,
    a3_system,
    a4_system,
    b1_system,
    b2_system,
    b3_system,
    b4_system,
    yH_system,
    y_train_system,
    curve_cnt_system
):
    csv_df = pd.DataFrame(
        data={
            'train_start': pd.Series(int(1000*time_str_system)), 
            'a1': pd.Series(a1_system),
            'a2': pd.Series(a2_system),
            'a3': pd.Series(a3_system),
            'a4': pd.Series(a4_system),
            'b1': pd.Series(b1_system),
            'b2': pd.Series(b2_system),
            'b3': pd.Series(b3_system),
            'b4': pd.Series(b4_system),
            'rmse': pd.Series(np.nanmean(np.abs(yH_system-y_train_system)))#,
        }
    )     
    csv_path ="./csv/system_manual_4th_" + str(datetime.datetime.utcnow().date()) + ".csv"
    header_flg = True if curve_cnt_system == 1 else False
    csv_df.to_csv(csv_path,mode='a', header=header_flg)#False)#True) 

def calc_rls(NUM_RLS, NUM_DATA, time_curve_rls, yawrate_i_filt_rls, yawrate_a_filt_rls):
    t,r,y = [],[],[]
    yH_local = []
    yH_rls = []
    dev_center_seg = []
    beta_alpha_output_vali = []
    delta_seg,theta_seg,alpha_seg = [],[],[]
    beta,gamma = [],[]
    
    alpha = NUM_RLS
    Pn    = alpha * np.eye(8)
    est   = np.zeros((NUM_DATA, 8))
    Theta = np.reshape(np.zeros(8),(8,1))
    
    a1,a2,a3,a4 = np.zeros(NUM_RLS),np.zeros(NUM_RLS),np.zeros(NUM_RLS),np.zeros(NUM_RLS),
    b1,b2,b3,b4 = np.zeros(NUM_RLS),np.zeros(NUM_RLS),np.zeros(NUM_RLS),np.zeros(NUM_RLS),
    
    for j in range(NUM_DATA): # 0,...,502-1 (2^9-10)
        t.append(          time_curve_rls[ j: j+NUM_RLS]) # j : j+10     
        r.append(      yawrate_i_filt_rls[ j: j+NUM_RLS]) # j : j+10     
        y.append(      yawrate_a_filt_rls[ j: j+NUM_RLS]) # j : j+10     
        #yH_rls.append( yawrate_a_filt_rls[ j: j+NUM_RLS]) # j : j+10    

    else:
        t_train  = np.array(     t).reshape( NUM_DATA, NUM_RLS) # 10 x 502 
        r_train  = np.array(     r).reshape( NUM_DATA, NUM_RLS) 
        y_train  = np.array(     y).reshape( NUM_DATA, NUM_RLS) 
        #yH_train = np.array(yH_rls).reshape( NUM_DATA, NUM_RLS) 
    
    for k in range(NUM_RLS): # 0,...,501 (NUM_RLS=502)
        Yn   = y_train[NUM_DATA-1,k]
        trZn = np.array([
            -y_train[3,k], 
            -y_train[2,k], 
            -y_train[1,k], 
            -y_train[0,k],
            r_train[3,k],  
            r_train[2,k],  
            r_train[1,k],  
            r_train[0,k]
        ])
        Zn = np.reshape(trZn, (8, 1))
        Theta, Pn = rls(Theta, Pn, 1.0, Yn, Zn)
        a1[k],a2[k],a3[k],a4[k] = Theta[0],Theta[1],Theta[2],Theta[3]
        b1[k],b2[k],b3[k],b4[k] = Theta[4],Theta[5],Theta[6],Theta[7]
    
    Hnum = [   b1[NUM_RLS-1], b2[NUM_RLS-1], b3[NUM_RLS-1], b4[NUM_RLS-1]]
    Hden = [1, a1[NUM_RLS-1], a2[NUM_RLS-1], a3[NUM_RLS-1], a4[NUM_RLS-1]]
     
    H = tf(Hnum, Hden, 1/NUM_RLS)
    sys = tf2ss(H);
    CA = np.dot(sys.C,sys.A)
    CAA = np.dot(CA,sys.A)
    CAAA = np.dot(CAA,sys.A)
    C_CA_inv = np.linalg.inv(np.concatenate([sys.C, CA, CAA, CAAA], 0))
    CB,CAB,CAAB = np.dot(sys.C,sys.B),np.dot(CA,sys.B),np.dot(CAA,sys.B)

#    global input_omega
#    input_omega = "omega"
    if input_omega == "omega":
#        global step
#        step = 2
        for j in range(NUM_RLS-5):
            y0_Du  = np.matrix(y_train[0,j]-np.dot(sys.D, r_train[0,j]))
            y1_Du1 = y_train[1,j]-np.dot(sys.D, r_train[1,j])-CB*r_train[0,j]
            y2_Du2 = y_train[2,j]-np.dot(sys.D, r_train[2,j])-CB*r_train[1,j]-CAB*r_train[0,j]
            y3_Du3 = y_train[3,j]-np.dot(sys.D, r_train[3,j])-CB*r_train[2,j]-CAB*r_train[1,j]-CAAB*r_train[0,j]
            
            y0_y1_y2 = np.concatenate([y0_Du, y1_Du1, y2_Du2,y3_Du3], 0)
            x0_x1_x2 = np.dot(C_CA_inv, y0_y1_y2)
            
            #r_input = r_train[:,j].reshape(-1)  
            #r_input = r_train[:,j+4].reshape(-1) 
            r_input = np.append(r_train[:,j].reshape(-1),r_train[0:step-1,j+5].reshape(-1)) 
            y_input = y_train[:,j].reshape(-1)  
            #yH_local, tH, xH = lsim(sys, r_input, X0=x0_x1_x2)
            #r_input = r_train[0,j:j+5].reshape(-1)  
            yH_local, tH, xH = lsim(sys, r_input, X0=x0_x1_x2)
            yH_rls.append(yH_local[-1])
        #y_pred = yH_rls[5-step:]
        y_pred = yH_rls[:]
    else:
        j = 0

    #for j in range(NUM_RLS):
        y0_Du  = np.matrix(y_train[0,j]-np.dot(sys.D, r_train[0,j]))
        y1_Du1 = y_train[1,j]-np.dot(sys.D, r_train[1,j])-CB*r_train[0,j]
        y2_Du2 = y_train[2,j]-np.dot(sys.D, r_train[2,j])-CB*r_train[1,j]-CAB*r_train[0,j]
        y3_Du3 = y_train[3,j]-np.dot(sys.D, r_train[3,j])-CB*r_train[2,j]-CAB*r_train[1,j]-CAAB*r_train[0,j]
        
        y0_y1_y2 = np.concatenate([y0_Du, y1_Du1, y2_Du2,y3_Du3], 0)
        x0_x1_x2 = np.dot(C_CA_inv, y0_y1_y2)
        
        r_input = r_train[-1,:].reshape(-1)  
        #y_input = y_train[:,j].reshape(-1)  
        yH_rls, tH, xH = lsim(sys, r_input, X0 = x0_x1_x2)
        #yH_rls.append(yH_local[-1])
        y_pred = yH_rls[:]

    show_data = False
    if show_data:
        plt.plot(t_train[-1,:], yH_rls[5:],    label="yH"      )
        plt.plot(t_train[-1,:], r_train[-1,:], label="r_input" )
        plt.plot(t_train[-1,:], y_train[-1,:], label="y_input" )
        plt.legend()
        plt.show()

        save_data_example = True
        if save_data_example == True:
            csv_df = pd.DataFrame(
                            data={'time':   pd.Series(t_train[-1,:]), 
                                  'y_pred': pd.Series(y_pred), 
                                  'r_train': pd.Series(r_train[-1,:]), 
                                  'y_train': pd.Series(y_train[-1,:]), 
                                  })     
            csv_path = "../result/csv/rls/dataset_example_1step_"+str(datetime.datetime.utcnow().date())+".csv"
            #header_flg = True if curve_cnt==1 else False
            csv_df.to_csv(csv_path,mode='a', header=True) 
        sys.exit()

    return yH_rls[5:], y_train[-1,:], r_train[-1,:], a1,a2,a3,a4, b1,b2,b3,b4

def main():
    # load dataset
    time_str, yawrate_i, yawrate_a, angular_velocity, curvature, steering_angle, dev_center,\
    time_eeg, alpha_local, beta_local, beta_alpha_local = load_dataset()

    alpha_ave, beta_ave, beta_alpha_ave = [],[],[]
    time_curve_train = []
    alpha_curve,beta_curve,beta_alpha_curve = [],[],[]
    yawrate_i_filt_train,yawrate_a_filt_train = [],[]
    a1_train,a2_train,a3_train,a4_train = [],[],[],[]
    b1_train,b2_train,b3_train,b4_train = [],[],[],[]
    rmse_all = []
    
    N       = len(time_str)             # number of samples 
    N_curve = 2**9                      # number of samples for 1 curve 
    dt      = time_str[1]-time_str[0]   # sampling cycle [s] for vehile data
    dt_eeg  = 10 * 0.0001               # sampling cycle [s] for EEG data 
    fs      = 1/dt                      # sampling frequency

    num_test_dataset = 30
    num_train_dateset = 100

    NUM_DATA = 5 
    NUM_RLS = N_curve - NUM_DATA  

    eeg_low_th  = 0.4
    eeg_high_th = 1.0
    eeg_high_th_str = str(int(eeg_high_th*1000))

    freq_th = 5
    rmse_min = 1
    
    save_fig_fft = False
    if save_fig_fft == True: 
        fig, axes = plt.subplots(4, 1, tight_layout=True)

    curve_focus_train, curve_focus_test, curve_drow, in_curve_flg, in_curve_cnt, curve_cnt\
     = curve_detection(
        time_str, 
        dt, 
        N_curve, 
        yawrate_i, 
        yawrate_a, 
        fs,
        alpha_local, 
        beta_local, 
        beta_alpha_local,
        eeg_high_th, 
        eeg_low_th, 
        num_train_dateset
    )
    curve_focus_train_int = list(map(int,curve_focus_train))
    curve_focus_test_int  = list(map(int,curve_focus_test))
    curve_drow_int        = list(map(int,curve_drow))
    print_stdio("train_focus")
    #for i in tqdm(curve_focus_train_int):
    for i in curve_focus_train_int:
        curve_cnt += 1
        idx_start = i-int(3/dt)
        idx_end   = i-int(3/dt)+N_curve
        alpha_ave, beta_ave,beta_alpha_ave = trim_mean(
            alpha_local, 
            beta_local, 
            beta_alpha_local,
            time_str, 
            idx_start, 
            idx_end
        ) 
        time_curve, yawrate_i_curve, yawrate_a_curve = trim_idx(
            time_str,
            yawrate_i,
            yawrate_a,idx_start,idx_end)
    
        # FFT
        Amp_i, Amp_a, freq = calc_fft(yawrate_i_curve, yawrate_a_curve, N_curve, dt)
    
        # LPF
        yawrate_a_filt, yawrate_i_filt = filtering_process("LPF", yawrate_a, yawrate_i,\
                                                                         idx_start, idx_end, freq_th, fs)
    
        time_curve_train     = np.append(    time_curve_train,     time_curve)
        yawrate_i_filt_train = np.append(yawrate_i_filt_train, yawrate_i_filt)
        yawrate_a_filt_train = np.append(yawrate_a_filt_train, yawrate_a_filt)
    
        if save_fig_fft == True: 
            draw_fig_fft(time_str, yawrate_i, yawrate_a, dt, i, N_curve, Amp_i, Amp_a)
            Amp_i_filt, Amp_a_filt, freq = calc_fft(yawrate_i_filt, yawrate_a_filt, N_curve, dt)
            draw_fig_fft_filt(time_str, time_eeg, yawrate_a_filt, yawrate_i_filt,\
                                                   idx_start, idx_end, freq, Amp_i_filt, Amp_a_filt,\
                                                   dt, i, N_curve)

        yH,y_train,r_input, a1,a2,a3,a4, b1,b2,b3,b4 = [],[],[],[],[],[],[],[],[],[],[]
        yH,y_train,r_input, a1,a2,a3,a4, b1,b2,b3,b4\
            = calc_rls(NUM_RLS, NUM_DATA, time_curve, yawrate_i_filt, yawrate_a_filt)
    
        save_dataset_csv_flg = False
        if save_dataset_csv_flg == True: save_dataset_csv(time_curve, r_input, y_train, yH, curve_cnt) 
    
        save_fig_pzmap_test = False
        if save_fig_pzmap_test == True: draw_pzmap(t_train[NUM_DATA-1],
                                                   r_train[NUM_DATA-1].reshape(-1),
                                                   y_train[NUM_DATA-1].reshape(-1),
                                                   yH[NUM_DATA-1],
                                                   beta_alpha_ave_pzmap,
                                                   a1,a2,a3,a4,b1,b2,b3,b4)
        
        rmse_temp = np.nanmean(np.abs(yH-y_train[NUM_DATA-1].reshape(-1)))
        rmse_all.append(rmse_temp)
    
        if rmse_temp < rmse_min:

            rmse_min = rmse_temp

            a1_train_best, a2_train_best, a3_train_best, a4_train_best, \
            b1_train_best, b2_train_best, b3_train_best, b4_train_best = \
            a1[NUM_RLS-1], a2[NUM_RLS-1], a3[NUM_RLS-1], a4[NUM_RLS-1], \
            b1[NUM_RLS-1], b2[NUM_RLS-1], b3[NUM_RLS-1], b4[NUM_RLS-1]
    
        if max(abs(yH[NUM_DATA-1]-y_train[NUM_DATA-1].reshape(-1))) < 0.9:
            a1_train = np.append(a1_train, a1[NUM_RLS-1])
            a2_train = np.append(a2_train, a2[NUM_RLS-1])
            a3_train = np.append(a3_train, a3[NUM_RLS-1])
            a4_train = np.append(a4_train, a4[NUM_RLS-1])
            b1_train = np.append(b1_train, b1[NUM_RLS-1])
            b2_train = np.append(b2_train, b2[NUM_RLS-1])
            b3_train = np.append(b3_train, b3[NUM_RLS-1])
            b4_train = np.append(b4_train, b4[NUM_RLS-1])
        
        save_system_csv_flg = False
        if save_system_csv_flg == True: 
            save_system_csv(time_str[i-int(3/dt)],
                            a1[NUM_RLS-1],a2[NUM_RLS-1],a3[NUM_RLS-1],a4[NUM_RLS-1],
                            b1[NUM_RLS-1],b2[NUM_RLS-1],b3[NUM_RLS-1],b4[NUM_RLS-1],
                            yH, y_train.reshape(-1), curve_cnt, header_flg)
    
    save_focus_rmse = True
    if save_focus_rmse == True:
        csv_df = pd.DataFrame(
                        data={'rmse_pred': pd.Series(rmse_all), 
                              })     
        csv_path = "../result/csv/rls/rmse_pred_"+str(datetime.datetime.utcnow().date())+".csv"
        #header_flg = True if curve_cnt==1 else False
        csv_df.to_csv(csv_path,mode='a', header=True) 

    a1_train_ave, a2_train_ave, a3_train_ave, a4_train_ave, \
    b1_train_ave, b2_train_ave, b3_train_ave, b4_train_ave \
        = np.nanmean(a1_train), np.nanmean(a2_train), np.nanmean(a3_train), np.nanmean(a4_train),\
          np.nanmean(b1_train), np.nanmean(b2_train), np.nanmean(b3_train), np.nanmean(b4_train)
    
    curve_cnt = 0 
    print_stdio("test_drow_focus")

    #for i in tqdm(curve_drow_int + curve_focus_test_int):
    for i in curve_drow_int + curve_focus_test_int:
        curve_cnt += 1
        idx_start = i-int(3/dt)
        idx_end   = i-int(3/dt)+N_curve
    
        alpha_ave,beta_ave,beta_alpha_ave = trim_mean(alpha_local, beta_local, beta_alpha_local,\
                                                      time_str, idx_start, idx_end) 
        time_curve,yawrate_i_curve,yawrate_a_curve = trim_idx(time_str,yawrate_i,yawrate_a,idx_start,idx_end)
    
        # FFT
        Amp_i, Amp_a, freq = calc_fft(yawrate_i, yawrate_a, N_curve, dt)
    
        # LPF (original data)        
        yawrate_a_filt, yawrate_i_filt = filtering_process("LPF", yawrate_a, yawrate_i,\
                                                                         idx_start, idx_end, freq_th, fs)

        if save_fig_fft == True: 
            draw_fig_fft(time_str, yawrate_i, yawrate_a, dt, i, N_curve, Amp_i, Amp_a)
            Amp_i_filt, Amp_a_filt, freq = calc_fft(yawrate_i_filt, yawrate_a_filt, N_curve, dt)
            draw_fig_fft_filt(time_str, time_eeg, yawrate_a_filt, yawrate_i_filt,\
                                              idx_start, idx_end, freq, amp_i_filt, amp_a_filt,\
                                              dt, i, n_curve)
    
        alpha_ave,beta_ave,beta_alpha_ave = append_mean_trim(alpha_ave,beta_ave,beta_alpha_ave,
                                                             alpha_local,beta_local,beta_alpha_local,
                                                             time_str,idx_start,idx_end)                
        # RLS
        yH,y_train,r_input, a1,a2,a3,a4, b1,b2,b3,b4 = \
        calc_rls(NUM_RLS, NUM_DATA, time_curve, yawrate_i_filt, yawrate_a_filt)
    
        if save_dataset_csv_flg == True: save_dataset_csv(time_curve, r_input, yawrate_a_filt, yH, curve_cnt) 
        if save_fig_pzmap_test == True: draw_pzmap(t_train[NUM_DATA-1],
                                                   r_train[NUM_DATA-1].reshape(-1),
                                                   y_train[NUM_DATA-1].reshape(-1),
                                                   yH[NUM_DATA-1],
                                                   beta_alpha_ave_pzmap,
                                                   a1,a2,a3,a4,b1,b2,b3,b4)

        drowsiness_flg = 1 if i in curve_drow_int else 0
        save_rmse_csv_flg = True
        if save_rmse_csv_flg == True: 
            if input_omega == "omega":
                save_rmse_csv(idx_start,y_train[:-10],yH,freq_th,eeg_high_th_str,curve_cnt,time_str,drowsiness_flg,input_omega,step) 
            else:
                save_rmse_csv(idx_start,y_train[:-5],yH,freq_th,eeg_high_th_str,curve_cnt,time_str,drowsiness_flg,input_omega,0) 

if __name__ == '__main__':

    global input_omega
    global step

    for input_omega in ["omega","omega_hat"]:
        if input_omega == "omega":
            for step in range(5):
                main()
        else:
            main()

