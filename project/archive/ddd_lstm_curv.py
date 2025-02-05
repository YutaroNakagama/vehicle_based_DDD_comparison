import warnings
warnings.filterwarnings("ignore")

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint
import shutil

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import (
    roc_curve, 
    classification_report,
    auc, 
    roc_auc_score, 
    precision_recall_curve, 
    confusion_matrix, 
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    )

from tqdm import tqdm
from scipy.signal import butter,filtfilt

# class in other files
from module.preprocess import Preprocess
from module.dataset import Datasets
from module.train import PredictSimpleFormulaNet, Train 

if __name__ == '__main__':
    start = time.time()

    beta_lo_thld = 0.3 
    beta_hi_thld = 0.60

    # load dataset
    dataset = Datasets()
    df_veh, curv_time = dataset.load_veh(output_type="curve", filt=True)
    df_veh_eeg = dataset.load_eeg(data_type="curve",curv_time = curv_time)

    # split dataset by time index
    df_veh_eeg["beta_ave"] = (df_veh_eeg["beta_af7"] + df_veh_eeg["beta_af8"])/2 
    awake = df_veh_eeg[df_veh_eeg["beta_ave"] >= beta_hi_thld]
    drow  = df_veh_eeg[df_veh_eeg["beta_ave"] <  beta_lo_thld]
    #awake, drow = dataset.make_dataset(threshold = beta_thld) 
    #test_input,test_labels = dataset.make_dataset(test_idx) 
    #plt.hist(df_veh_eeg["beta_ave"])
    #plt.show()

    awake_yaw = awake[["yaw_rate_i","yaw_rate_a"]].values.astype('float32') 
    drow_yaw  = drow[["yaw_rate_i","yaw_rate_a"]].values.astype('float32') 
    awake_cnt = awake[["curv_cnt"]].values.astype('int') 
    drow_cnt  = drow[["curv_cnt"]].values.astype('int') 

    print("number of normal/abnormal datasets")
    print("awake (normal):    ",len(awake[["curv_cnt"]].drop_duplicates()))
    print("drowsy (abnormal): ", len(drow[["curv_cnt"]].drop_duplicates()))
    #awake[["curv_cnt","beta_ave"]].drop_duplicates().to_csv("awake_cnt.csv")
    #drow[["curv_cnt","beta_ave"]].drop_duplicates().to_csv("drow_cnt.csv")

    # dataset parameters
    sequence_length = 30
    awake_num = len(awake_yaw) - sequence_length
    drow_num  = len(drow_yaw) - sequence_length
    t_start = 0

    # model pram
    input_size = 2
    output_size = 1
    hidden_size = 10
    num_layer = 5 
    batch_first = True

    # train pram
    lr = 0.1 * (10 ** -3)
    epochs = 300
    batch_size = 64
    test_size = 0.95 # % of test datasets
    
    # initialize train class
    train = Train(input_size, output_size, hidden_size, num_layer, batch_first, lr)

    # create datasest from timeseries
    # awake
    awake_inputs, awake_labels, awake_times = train.make_dataset(awake_num, sequence_length, t_start, awake_yaw,awake["yaw_rate_a"].diff().max())
    _, awake_cnt_labels, _ = train.make_dataset(awake_num, sequence_length, t_start, awake_cnt, 100)
    # drow
    drow_inputs, drow_labels, drow_times = train.make_dataset(drow_num, sequence_length, t_start, drow_yaw, drow["yaw_rate_a"].diff().max())
    _, drow_cnt_labels, _ = train.make_dataset(drow_num, sequence_length, t_start, drow_cnt, 100)

    # split dataset into train & test (only awake datasets)
    awake_train_inputs, awake_test_inputs, awake_train_labels, awake_test_labels\
        = train_test_split(awake_inputs, awake_labels, test_size=test_size, shuffle=False)
    awake_train_times, awake_test_times\
        = train_test_split(awake_times, test_size=test_size, shuffle=False)
    awake_train_cnt, awake_test_cnt\
        = train_test_split(awake_cnt_labels, test_size=test_size, shuffle=False)

    # model training 
    train.train(awake_train_inputs, awake_train_labels, awake_test_inputs, awake_test_labels,\
    #train.train(awake_test_inputs, awake_test_labels, awake_test_inputs, awake_test_labels,\
                epochs, batch_size, sequence_length, input_size, plot=True)

    # check prediction result
    epsss_ave_drow = train.pred_result(drow_inputs, drow_labels, drow_times,\
                                       drow_cnt_labels, sequence_length, input_size, plot=True)
    epsss_ave_awake = train.pred_result(awake_test_inputs, awake_test_labels, awake_test_times,\
                                        awake_test_cnt, sequence_length, input_size, plot=True)

    print("number of normal/abnormal datasets for test")
    print("awake  (normal):   ", len(epsss_ave_awake[:,0]))
    print("drowsy (abnormal): ", len(epsss_ave_drow[:,0]))
    plt.hist(epsss_ave_awake[:,0], alpha=0.5, bins=10, label="awake")
    plt.hist(epsss_ave_drow[:,0],  alpha=0.5, bins=10, label="drow")
    plt.legend()
    plt.savefig('beta_ave_dist_'+str(hidden_size)+'_'+str(num_layer)+'.pdf')
    plt.show()

    epsss_ave_df_awake = pd.DataFrame(epsss_ave_awake)
    epsss_ave_df_drow  = pd.DataFrame(epsss_ave_drow)
    epsss_ave_df = pd.concat([epsss_ave_df_awake, epsss_ave_df_drow])
    epsss_ave_df.columns = ['epss','cnt']

    df_beta_ave = pd.DataFrame([], columns=["index","beta_ave"], index=range(len(epsss_ave_df['cnt'].astype('int').tolist())))

    index = 0
    for i in tqdm(epsss_ave_df['cnt'].astype('int').tolist()):
        df_beta_ave.iloc[index] = df_veh_eeg["beta_ave"][df_veh_eeg["curv_cnt"]==i].drop_duplicates().reset_index().copy()
        index += 1

    #df_beta_ave.to_csv("df_beta_ave.csv")
    roc_df = pd.concat([epsss_ave_df.reset_index(),df_beta_ave.reset_index()], axis=1)
    #roc_df = roc_df.drop(columns=roc_df.columns[[0, 3]])
    roc_df.to_csv("roc.csv")
    #roc_df.columns = ["epss","cnt","beta_ave"]
        
    y_true = roc_df["beta_ave"] <= beta_lo_thld 
    y_score = roc_df["epss"]

    fpr, tpr, threshold = roc_curve(y_true, y_score)
    roc_auc_value = roc_auc_score(y_true, y_score)
    print("roc_auc_value: ", roc_auc_value)

    # f1 Value
    precision, recall, threshold_from_pr = precision_recall_curve(y_true, probas_pred = y_score)
    print("precision:   ", precision)
    print("recall:  ", recall)
    a = 2* precision * recall
    b = precision + recall
    f1 = np.divide(a,b,out=np.zeros_like(a), where=b!=0)
    #print("f1", f1)

    idx_opt = np.argmax(f1)
    threshold_opt = threshold_from_pr[idx_opt] #Confusion Matrix
    idx_opt_from_pr = np.where(threshold == threshold_opt) # ROC Curve

    y_pred = y_score > threshold_opt 

    print("confusion_matrix: \n",confusion_matrix(np.logical_not(y_true),y_pred))
    print("f1:        ",f1_score(y_true,y_pred))
    print("acc:       ",accuracy_score(y_true, y_pred))
    print("precision: ",precision_score(y_true, y_pred))
    print("recall:    ",recall_score(y_true, y_pred))

    d = classification_report(y_true, y_pred, output_dict=True)
    pprint.pprint(d)

    plt.plot(fpr, tpr, marker='o')
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.title("hidden_size: "+str(hidden_size)+" num_layer: "+str(num_layer)+" AUC: "+str(roc_auc_value))
    plt.grid()
    plt.savefig('roc_curve_'+str(hidden_size)+'_'+str(num_layer)+'.pdf')
    
    plt.show()

    print("elapse time", time.time()-start)
