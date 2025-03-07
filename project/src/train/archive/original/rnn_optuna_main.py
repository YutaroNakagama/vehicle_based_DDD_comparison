
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

import pylab
#import control
#import statsmodels.api as sm
import scipy as sp
import scipy.stats as st
import sklearn.metrics as mt
import keras
import optuna

import datetime
import time
import sys

#from control.matlab import * 
from matplotlib import animation as ani
from pandas import read_csv
from random import gauss
from copy import copy 
from dataclasses import dataclass
from typing import List, Optional, Tuple
from statistics import stdev, variance, median
from numpy.typing import NDArray
#from tqdm import tqdm

#from statsmodels.tsa.ar_model import AR
from sklearn.metrics import roc_curve, auc, mean_squared_error, roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Input, SimpleRNN, LSTM, GRU, RepeatVector, Activation
#from tensorflow.python.keras.layers.core import Dense, Lambda
#from keras.layers.core import Dense, Lambda
from keras.layers import Dense, Lambda
from keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import Adam

from scipy import signal
#from scipy.stats import spearmanr
from scipy.interpolate import interp1d

# local lib
from local_lib.preprocess import calc_fft, butter_lowpass,butter_lowpass_filter,\
                                 filtering_process, calc_rmse,\
                                 curve_check, load_dataset, curve_detection,\
                                 save_dataset_csv, save_rmse_csv,\
                                 trim_mean, append_mean_trim, trim_idx,\
                                 make_train_dataset

plt.style.use('ggplot')
plt.rcParams['axes.grid'] = True  # make grid
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams['mathtext.fontset'] = 'stix'  # The setting of math font
plt.rcParams["legend.fancybox"] = False # 丸角
plt.rcParams["legend.framealpha"] = 1 # 透明度の指定、0で塗りつぶしなし
plt.rcParams["legend.edgecolor"] = 'black' # edgeの色を変更
plt.rcParams["legend.handlelength"] = 2 # 凡例の線の長さを調節
plt.rcParams["legend.labelspacing"] = 0. # 垂直方向（縦）の距離の各凡例の距離
plt.rcParams["legend.handletextpad"] = 0.5 # 凡例の線と文字の距離の長さ
plt.rcParams["legend.markerscale"] = 2 # 点がある場合のmarker scale
plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
plt.rcParams['lines.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ

def rnn_predict(NUM_DATA_local, yawrate_i_filt_local, yawrate_a_filt_local, NUM_RNN_local, model_local):

    y_test = np.array(yawrate_a_filt_local[0:NUM_RNN]).reshape(-1)
    x_test = np.array(yawrate_a_filt_local[0:NUM_RNN]).reshape(-1)

    for j in (range(NUM_DATA_local)):
        x0_test = np.array(yawrate_i_filt_local[j:j+NUM_RNN]).reshape(1, NUM_RNN, 1) 
        if input_omega == "omega_hat": 
            x1_test = np.array(y_test[j:j+NUM_RNN]).reshape(1, NUM_RNN, 1) 
        else:
            x1_test = np.array(yawrate_a_filt_local[j:j+NUM_RNN]).reshape(1, NUM_RNN, 1) 
        x_test_input = np.concatenate([x0_test, x1_test],2)

        #y_pred = model_local.predict(x_test_input,verbose=0)
        fit_start = time.time()
        y_pred = model_local(x_test_input, training=False).numpy()
        elapsed_time = time.time() - fit_start
        #print(f"fit elapsed time: {elapsed_time}")

        if input_omega == "omega_hat": 
            y_test = np.concatenate([y_test, y_pred[0,NUM_RNN-1,0].reshape(-1)],0) 
        else:
            y_test = np.concatenate([y_test, y_pred[0,-step-1,0].reshape(-1)],0) 
        x_test = np.concatenate([x_test, x1_test[0,NUM_RNN-1,0].reshape(-1)],0) 

    return x_test, y_test

def draw_fig_rnn(time_curve, yawrate_i_filt, yawrate_a_filt, y_test,\
                 rmse_local, rmse_local_cnt, time_str, i,\
                 rnn_type, freq_th, eeg_high_th_str, NUM_DIM, n_batch, data_type):
    plt.plot(time_curve, yawrate_i_filt, c='k', label="ideal")
    plt.plot(time_curve, yawrate_a_filt, c='g', label="actual")
    plt.plot(time_curve, y_test, c='r', label="pred")
    rmse_ave = rmse_local/rmse_local_cnt if rmse_local_cnt != 0 else 0
    plt.text(min(time_curve),max(y_test),rmse_ave)
    plt.legend()
    begin_time = str(int(1000*time_str[i-int(3/dt)])) 
    plt.savefig("fig/"+rnn_type+"_i_"+str(i)+"_train_result_fc_"\
                +str(freq_th)+"_eegth_"+eeg_high_th_str+"_"+str(NUM_DIM)+"_"\
                +str(n_batch)+"_"+begin_time+"_"+data_type+".png", 
                format="png", dpi=1600)
    plt.cla()

def train_focused_data(NUM_DIM,rnn_type,n_batch,freq_th):

    yawrate_i_filt_train, yawrate_a_filt_train = [],[]
    rmse_all = []
    alpha_ave, beta_ave, beta_alpha_ave = [],[],[]

    first_model = True
    rmse_local_min = 0.5

    print("train focused data")

    #for i in tqdm(curve_focus_train_int):
    for i in curve_focus_train_int:

        idx_start = i-int(3/dt)
        idx_end   = i-int(3/dt)+N_curve
        alpha_ave,beta_ave,beta_alpha_ave = trim_mean(alpha_local, beta_local, beta_alpha_local,\
                                                      time_str, idx_start, idx_end) 

        time_curve,yawrate_i_curve,yawrate_a_curve = trim_idx(time_str,yawrate_i,yawrate_a,idx_start,idx_end)
    
        #FFT
        Amp_i, Amp_a, freq = calc_fft(yawrate_i_curve, yawrate_a_curve, N_curve, dt)
        if save_fig_fft == True: draw_fig_fft(time_str, yawrate_i, yawrate_a, dt, i, N_curve, Amp_i, Amp_a)

        # LPF (original data)        
        yawrate_a_filt_signed,yawrate_i_filt_signed = filtering_process("LPF", yawrate_a, yawrate_i,\
                                                                        i, i+N_curve, freq_th, fs)

        yawrate_i_pos_abs,yawrate_i_neg_abs = abs(max(yawrate_i_filt_signed)),abs(min(yawrate_i_filt_signed))
        curve_sign = 1 if yawrate_i_pos_abs > yawrate_i_neg_abs else -1 
        yawrate_a_filt, yawrate_i_filt = yawrate_a_filt_signed*curve_sign+1, yawrate_i_filt_signed*curve_sign+1
    
        #NUM_RNN  = int(fs) 
        model_gen = True
        if model_gen == True:
        #    for NUM_DIM, rnn_type, n_batch in itertools.product([10],["RNN"],[16]):
            NUM_DATA = len(yawrate_i_filt) - NUM_RNN
            x0_train,x1_train,x_train,y_train = make_train_dataset(NUM_DATA, NUM_RNN,\
                                                                   yawrate_i_filt, yawrate_a_filt)
            #first_model = False
            model_test = False
            if first_model == True:
                model = Sequential()
                if rnn_type == "RNN":
                    model.add(SimpleRNN(NUM_DIM, input_shape=(NUM_RNN,2), return_sequences=True))
                elif rnn_type == "LSTM":
                    model.add(LSTM(NUM_DIM, input_shape=(NUM_RNN,2), return_sequences=True))
                elif rnn_type == "GRU":
                    model.add(GRU(NUM_DIM, input_shape=(NUM_RNN,2), return_sequences=True))
        
                #model.add(Dense(1, activation="relu"))  
                model.add(Dense(1)) 
                model.compile(loss="mean_squared_error", optimizer="Adam")
                model.summary()
                early_stopping =  EarlyStopping(
                                                monitor='loss',
                                                min_delta=0.00001,
                                                patience=10,
                                                )

                fit_start = time.time() 
                history = model.fit(x_train, y_train,\
                                    epochs=n_epoch,\
                                    batch_size=n_batch,\
                                    callbacks=[early_stopping],\
                                    verbose=False)

                model.save('../result/model/model_i_'+str(i)+'_fc_'\
                           +str(freq_th)+'_eegth_'+eeg_high_th_str+'_'\
                           +rnn_type+'_'+str(NUM_DIM)+'_'+str(n_batch)+'_'+current_date+'.h5')

                elapsed_time = time.time() - fit_start
                print(f"fit elapsed time: {elapsed_time}")

                first_model = False
                best_model = i

            elif model_test == True:
                model = load_model('../result/model/model_i_2815_fc_5_eegth_1000.0_LSTM_10_16_2024_3_16.h5')
                
            else: 
                model = load_model('../result/model/model_i_'+str(best_model)+'_fc_'\
                                   +str(freq_th)+'_eegth_'+eeg_high_th_str+'_'\
                                   +rnn_type+'_'+str(NUM_DIM)+'_'+str(n_batch)+'_'+current_date+'.h5')

                early_stopping =  EarlyStopping(
                                                monitor='loss',
                                                min_delta=0.00001,
                                                patience=10,
                                                )

                history = model.fit(x_train, y_train,\
                                    epochs=n_epoch,\
                                    batch_size=n_batch,\
                                    callbacks=[early_stopping],\
                                    verbose=0)

                model_latest = '../result/model/model_i_'+str(i)+'_fc_'\
                                +str(freq_th)+'_eegth_'+eeg_high_th_str+'_'\
                                +rnn_type+'_'+str(NUM_DIM)+'_'+str(n_batch)+'_'\
                                +current_date+'.h5'
                model.save(model_latest)


            x_test, y_test = rnn_predict(NUM_DATA, yawrate_i_filt, yawrate_a_filt, NUM_RNN, model)
            
            save_eg_csv = True
            if save_eg_csv:
                csv_df = pd.DataFrame(
                                data={'time': pd.Series(time_curve), 
                                      'input': pd.Series(yawrate_i_curve),
                                      'output': pd.Series(yawrate_a_curve),
                                      'input_filt': pd.Series(yawrate_i_filt-1),
                                      'output_filt': pd.Series(yawrate_a_filt-1),
                                      'output_filt_pred': pd.Series(y_test-1),
                                      'rmse': pd.Series(abs(y_test-yawrate_a_filt))
                                      })     
                csv_path = "../result/csv/rnn/predict_data_"+str(step)+"step_"+str(freq_th)+"_eegth_"+eeg_high_th_str+"_"\
                            +rnn_type+"_"+str(NUM_DIM)+"_"+str(n_batch)+"_"+current_time+"_"+input_omega+".csv"
                csv_df.to_csv(csv_path,mode='a', header=True)#False)#True) 

            if save_fig_loss_flg == True: 
                loss = history.history['loss']
                save_loss_fig(loss, i, freq_th, eeg_high_th_str,\
                                                        rnn_type, NUM_DIM, n_batch) 

            show_fig = False
            if show_fig == True:
                plt.plot(time_curve, yawrate_i_filt-1, label="input_filt")
                plt.plot(time_curve, yawrate_a_filt-1, label="output_filt")
                plt.plot(time_curve, y_test-1,         label="prediction")
                #plt.text(min(time_curve), min(y_test-1), csv_path)
                plt.legend()
                plt.show()
                sys.exit()

            rmse_local, rmse_local_cnt = calc_rmse(NUM_DATA, y_test, yawrate_a_filt)
            rmse_local = np.nanmean(np.abs(yawrate_a_filt-y_test))
            rmse_all.append(rmse_local)

            # save rnn fig
            if save_fig_rnn == True:
                draw_fig_rnn(time_curve, yawrate_i_filt, yawrate_a_filt, y_test,\
                             rmse_local, rmse_local_cnt, time_str, i,\
                             rnn_type, freq_th, eeg_high_th_str, NUM_DIM, n_batch, "train")

            # update best model
            if rmse_local < rmse_local_min: rmse_local_min, best_model = rmse_local, i

        else:
            best_model = 2815
            # model_i_2815_fc_5_eegth_1000.0_LSTM_5_16_2024_3_24.h5

        yawrate_i_filt_train = np.append(yawrate_i_filt_train, yawrate_i_filt)
        yawrate_a_filt_train = np.append(yawrate_a_filt_train, yawrate_a_filt)
        Amp_i, Amp_a, freq = calc_fft(yawrate_i_filt, yawrate_a_filt, N_curve, dt)
        if save_fig_fft == True: draw_fig_fft(time_str, yawrate_i_filt, yawrate_a_filt,\
                                              dt, i, N_curve, Amp_i, Amp_a)

    save_rmse_pred = False
    if save_rmse_pred == True:
        csv_df = pd.DataFrame(
                        data={
                              'rmse': pd.Series(rmse_all)
                              })     
        csv_path = "../result/csv/rnn/rmse_pred_"+current_date+".csv"
        csv_df.to_csv(csv_path,mode='a', header=True)#False)#True) 

    return best_model

def test_datasets(best_model,NUM_DIM,rnn_type,n_batch,freq_th):

    header_flg = True
    y_true,y_score = [],[]
    print("test focus data")

    #for i in tqdm(curve_focus_test_int + curve_drow_int):
    for i in curve_focus_test_int + curve_drow_int:
        index_start = i-int(3/dt)
        index_end   = i-int(3/dt)+N_curve
        time_curve       = time_str[index_start:index_end]
        yawrate_i_curve,yawrate_a_curve = yawrate_i[index_start:index_end], yawrate_a[index_start:index_end]
      
        Amp_i, Amp_a, freq = calc_fft(yawrate_i_curve, yawrate_a_curve, N_curve, dt)

        if save_fig_fft == True: draw_fig_fft(time_str, yawrate_i, yawrate_a, dt, i, N_curve, Amp_i, Amp_a)
      
        # LPF (original data)        
        yawrate_a_filt_signed, yawrate_i_filt_signed = filtering_process("LPF", yawrate_a, yawrate_i,\
                                                                         index_start, index_end, freq_th, fs)
      
        yawrate_i_pos_abs, yawrate_i_neg_abs = abs(max(yawrate_i_filt_signed)), abs(min(yawrate_i_filt_signed))
        curve_sign = 1 if yawrate_i_pos_abs > yawrate_i_neg_abs else -1
        yawrate_a_filt,yawrate_i_filt = yawrate_a_filt_signed * curve_sign, yawrate_i_filt_signed * curve_sign
      
        Amp_i_filt, Amp_a_filt, freq = calc_fft(yawrate_i_filt, yawrate_a_filt, N_curve, dt)
        if save_fig_fft == True: draw_fig_fft_filt(time_str, time_eeg, yawrate_a_filt, yawrate_i_filt,\
                                              index_start, index_end, freq, Amp_i_filt, Amp_a_filt,\
                                              dt, i, N_curve)
      
        NUM_RNN = int(fs) 
        NUM_DATA = len(yawrate_i_filt) - NUM_RNN#n_sample#len(x_sin) - NUM_RNN # 今回は40(=50-10)
        
        x0_vali,x1_vali,x_vali,y_vali = make_train_dataset(NUM_DATA, NUM_RNN, yawrate_i_filt, yawrate_a_filt)
      
        model = load_model('../result/model/model_i_'+str(best_model)+'_fc_'\
                   +str(freq_th)+'_eegth_'+eeg_high_th_str+'_'\
                   +rnn_type+'_'+str(NUM_DIM)+'_'+str(n_batch)+'_'+current_date+'.h5')
      
        y_pred, y_test = rnn_predict(NUM_DATA, yawrate_i_filt+1, yawrate_a_filt+1, NUM_RNN, model)
      
        rmse_local,rmse_local_cnt = calc_rmse(NUM_DATA, y_test, yawrate_a_filt)
      
        drowsiness_flg = 1 if i in curve_drow_int else 0
        data_type = "drow" if i in curve_drow_int else "focus"
      
        if save_fig_rnn == True:
            draw_fig_rnn(time_curve, yawrate_i_filt, yawrate_a_filt, \
                         y_test, rmse_local, rmse_local_cnt, time_str, i,\
                         rnn_type, freq_th, eeg_high_th_str, NUM_DIM, n_batch, data_type)

        rmse_ave = rmse_local/rmse_local_cnt if rmse_local_cnt!=0 else 0

        csv_df = pd.DataFrame(
                        data={'train_start': pd.Series(int(1000*time_str[i-int(3/dt)])), 
                              'rmse': pd.Series(rmse_ave),
                              'drowsiness':pd.Series(drowsiness_flg)
                              })     

        y_true.append(drowsiness_flg)
        y_score.append(rmse_ave)

        save_rmse_csv_roc = False
        if save_rmse_csv_roc:
            csv_path = "../result/csv/rnn/rmse_"+str(step)+"step_fc_"+str(freq_th)+"_eegth_"+eeg_high_th_str+"_"\
                        +rnn_type+"_"+str(NUM_DIM)+"_"+str(n_batch)+"_"+current_date+"_"+input_omega+".csv"
            csv_df.to_csv(csv_path,mode='a', header=header_flg)#False)#True) 
            header_flg = False

    save_roc_fig_flg = False
    if save_roc_fig_flg == True:
        print(roc_auc_score(y_true, y_score))
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, marker='o')
        plt.text(0,1,roc_auc_score(y_true, y_score))
        plt.xlabel('FPR: False positive rate')
        plt.ylabel('TPR: True positive rate')
        roc_path = "./fig/roc_"+str(freq_th)+"_eegth_"+eeg_high_th_str+"_"\
                    +rnn_type+"_"+str(NUM_DIM)+"_"+str(n_batch)+"_"+current_date+".png"
        plt.savefig(roc_path)
        plt.cla()

    return roc_auc_score(y_true, y_score)

def objective(trial):

    # set hyper-parameters
    NUM_DIM  = trial.suggest_categorical("NUM_DIM",[5])
    rnn_type = trial.suggest_categorical("rnn_type",["LSTM"])
    n_batch  = trial.suggest_categorical("n_batch",[16])
    freq_th  = trial.suggest_int("freq_th",5,5)

    # training 
    best_model = train_focused_data(NUM_DIM, rnn_type, n_batch, freq_th)

    # velidation
    auc_score  = test_datasets(best_model,NUM_DIM, rnn_type, n_batch, freq_th)

    return auc_score
                
if __name__ == '__main__':

    # get current time
    t1 = time.time()
    d_t = datetime.datetime.today()
    current_date = str(d_t.year)+"_"+str(d_t.month)+"_"+str(d_t.day)
    current_time = str(d_t.year)+"_"+str(d_t.month)+"_"+str(d_t.day)+"_"+str(d_t.hour)+"_"+str(d_t.minute)+"_"+str(d_t.second)

    # load dataset
    time_str, yawrate_i, yawrate_a, angular_velocity, curvature, steering_angle, dev_center,\
    time_eeg, alpha_local, beta_local, beta_alpha_local = load_dataset()
    
    #N       = len(time_str)            # number of samples
    dt      = time_str[1]-time_str[0]   # sampling time [s]
    #dt_eeg  = 10 * 0.0001 #time_eeg[1]-time_eeg[0]         # サンプリング周期 [s]
    fs      = 1/dt                      # sampling freqency [1/s] 
    N_curve = int(fs*10)                # number of samples per curve 
    NUM_RNN = int(fs)                   # number of input samples per 1 rnn 

    n_epoch = 20
    
    save_fig_rnn      = False
    save_fig_loss_flg = False
    save_fig_fft      = False
    if save_fig_fft == True: fig, axes = plt.subplots(4, 1, tight_layout=True)
    
    # EEG threshold
    eeg_low_th        = 0.4
    eeg_high_th       = 1.0
    eeg_high_th_str   = str(1000*eeg_high_th)
    
    # curve detection
    num_train_dateset = 5
    curve_cnt         = 0
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

    curve_focus_train_int = list(map(int,curve_focus_train[:]))
    curve_focus_test_int  = list(map(int,curve_focus_test[:]))
    curve_drow_int        = list(map(int,curve_drow[:]))
    
    # call optuna function
    study = optuna.create_study(direction='maximize')

    global input_omega
    global step

    for input_omega in ["omega_hat","omega"]:
        if input_omega == "omega":
            for step in range(5):
                study.optimize(objective, n_trials=1)
                print("step",step)
        else:
            step=0
            study.optimize(objective, n_trials=1)
            print("step",step)

    # get optuna result
    print("Best_parameter : " , study.best_params)
    print("Accuracy       : " , study.best_value)

    elapsed_time =time.time() - t1
    print(f"elapsed time: {elapsed_time}")

    current_date = str(datetime.datetime.utcnow().date())

    fig=optuna.visualization.plot_contour(study)
    plt.savefig("../result/visualisation/fig/plot_contour_"+current_date+".pdf")
    plt.savefig("../result/visualisation/fig/plot_contour_"+current_date+".png")

    fig=optuna.visualization.plot_param_importances(study)
    plt.savefig("../result/visualisation/fig/plot_param_importances_"+current_date+".pdf")
    plt.savefig("../result/visualisation/fig/plot_param_importances_"+current_date+".png")

    fig=optuna.visualization.plot_optimization_history(study)
    plt.savefig("../result/visualisation/fig/plot_optimization_history_"+current_date+".pdf")
    plt.savefig("../result/visualisation/fig/plot_optimization_history_"+current_date+".png")

    fig=optuna.visualization.plot_parallel_coordinate(study)
    plt.savefig("../result/visualisation/fig/plot_parallel_coordinate_"+current_date+".pdf")
    plt.savefig("../result/visualisation/fig/plot_parallel_coordinate_"+current_date+".png")

    '''
    # set hyper-parameters
    NUM_DIM  = 10
    rnn_type = "LSTM"
    n_batch  = 16
    freq_th  = 5
    
    # training 
    best_model = train_focused_data(NUM_DIM, rnn_type, n_batch, freq_th)
    
    # velidation
    auc_score  = test_datasets(best_model,NUM_DIM, rnn_type, n_batch, freq_th)
    '''
     
