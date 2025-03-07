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

from tqdm import tqdm_notebook as tqdm
from scipy.signal import butter,filtfilt

# class in other files
from module.preprocess import Preprocess
from module.dataset import Datasets
from module.train import PredictSimpleFormulaNet, Train 

if __name__ == '__main__':
    start = time.time()

    dataset_type  = "../../dataset/UAH-DRIVESET-v1/"
    #dataset       = "D1/20151111135612-13km-D1-DROWSY-SECONDARY/"
    #dataset       = "D2/20151120164606-16km-D2-DROWSY-SECONDARY/"
    #dataset       = "D3/20151126132013-17km-D3-DROWSY-SECONDARY/"
    #dataset       = "D4/20151203175637-17km-D4-DROWSY-SECONDARY/"
    #dataset       = "D5/20151211170502-16km-D5-DROWSY-SECONDARY/"
    dataset       = "D6/20151221113846-16km-D6-DROWSY-SECONDARY/"
    dataset_dir   = dataset_type + dataset
    
    fp_sem_online = dataset_dir + "SEMANTIC_ONLINE.txt" # dt=1s
    fp_raw_acc    = dataset_dir + "RAW_ACCELEROMETERS.txt" # 10Hz dt=0.01s
    fp_raw_gps    = dataset_dir + "RAW_GPS.txt" # dt=1s
    fp_raw_lane   = dataset_dir + "PROC_LANE_DETECTION.txt" # 30Hz
    fp_raw_osm    = dataset_dir + "PROC_OPENSTREETMAP_DATA.txt" # 1Hz
    
    df_sem_online = pd.read_csv(fp_sem_online, delimiter=" ", header=None)
    df_raw_acc    = pd.read_csv(fp_raw_acc,    delimiter=" ", header=None)
    df_raw_gps    = pd.read_csv(fp_raw_gps,    delimiter=" ", header=None)
    df_raw_lane   = pd.read_csv(fp_raw_lane,   delimiter=" ", header=None)
    df_raw_osm    = pd.read_csv(fp_raw_osm,    delimiter=" ", header=None)
    
    df_sem_online.columns = ['time','gps lat','gps lon',
                              'score total win','score acc win',  'score brake win',  'score turn win', 
                              'score weav win', 'score drift win','score overspd win','score car-follow win', 
                              'ratio norm win','ratio drow win','ratio agg win','ratio dist win',
                              'score total','score acc',  'score brake',    'score turn', 
                              'score weav', 'score drift','score overspeed','score car-follow', 
                              'ratio norm','ratio drow','ratio agg','ratio dist',
                              'other']
    
    df_raw_acc.columns = ['time','act bool',
                          'X acc','Y acc','Z acc',
                          'X acc KF','Y acc KF','Z acc KF',
                          'roll','pitch','yaw (deg)',
                          'other']
    
    df_raw_acc['yawrate'] = df_raw_acc['yaw (deg)'].diff() / 0.01
    
    df_raw_gps.columns = ['time','speed',
                          'lat','lon','alt',
                          'vert accu','hori accu','course','discourse',
                          'pos state','lanex dist state','lanex hist',
                          'others']
    
    df_raw_lane.columns = ['time','dev from centre','ang from curv (deg)','road width','lane det state']
    df_raw_lane['yaw diff rate'] = df_raw_lane['ang from curv (deg)'].diff() / (1/30)
    
    df_raw_lane['time'] = df_raw_lane['time'].round(2)
    data_list = pd.DataFrame(np.arange(min(df_raw_lane['time']),max(df_raw_lane['time']),0.01), columns=["time"])
    data_list = pd.merge(data_list,df_raw_lane, on="time", how="outer")
    data_list = data_list.sort_values('time').drop_duplicates(keep='first').reset_index(drop=True)

    df_raw_acc['time'] = df_raw_acc['time'].round(2)
    data_list = pd.merge(data_list, df_raw_acc, on="time", how="outer")
    data_list = data_list.sort_values('time').drop_duplicates(keep='first').reset_index(drop=True)

    df_sem_online['time'] = df_sem_online['time'].round(2)
    data_list = pd.merge(data_list, df_sem_online, on="time", how="outer")
    data_list = data_list.sort_values('time').drop_duplicates(keep='first').reset_index(drop=True)

    data_list = data_list[['time','yaw (deg)','ang from curv (deg)','ratio norm','ratio drow']] 

    data_list['yaw (deg)'].interpolate(method="cubic",inplace = True)
    data_list['ang from curv (deg)'].interpolate(method="cubic",inplace = True)
    data_list['ratio norm'].interpolate(method="cubic",inplace = True)
    data_list['ratio drow'].interpolate(method="cubic",inplace = True)
    
    data_list = data_list.dropna()
    data_list['ang from curv (deg)'] = data_list['ang from curv (deg)'].rolling(2000).mean()
    data_list['yaw curv (deg)']      = data_list['yaw (deg)'] - data_list['ang from curv (deg)']
    data_list['yaw diff rate']       = data_list['ang from curv (deg)'].diff() / (1/10)
    data_list['yaw rate']            = data_list['yaw (deg)'].diff().rolling(20).mean()  / (1/10)
    data_list['curv yaw rate']       = data_list['yaw curv (deg)'].diff().rolling(20).mean()  / (1/10)
    data_list = data_list.loc[::10]
    data_list = data_list.dropna()

    data_list.to_csv("test.csv")
    
    fig, axes = plt.subplots(4, 1, tight_layout=True)
    df_sem_online.plot(x="time", y=['ratio norm','ratio drow','ratio agg'], ax=axes[0])
    df_raw_lane.plot(  x="time", y=['dev from centre','ang from curv (deg)','lane det state'], ax=axes[1])
    data_list.plot(    x="time", y=['yaw (deg)', 'ang from curv (deg)', 'yaw curv (deg)'], ax=axes[2])
    data_list.plot(    x="time", y=['yaw rate','curv yaw rate'], ax=axes[3])
    axes[1].set_ylim((-2,2))
    plt.show()

    awake = data_list[data_list['ratio norm'] >= data_list['ratio drow']]
    drow  = data_list[data_list['ratio norm'] <  data_list['ratio drow']]

    awake_yaw = awake[["yaw rate","curv yaw rate"]].values.astype('float32') 
    drow_yaw  = drow[["yaw rate","curv yaw rate"]].values.astype('float32') 

    # dataset parameters
    sequence_length = 15
    awake_num = len(awake_yaw) - sequence_length
    drow_num  = len(drow_yaw)  - sequence_length
    t_start = 0

    # model pram
    input_size  = 2
    output_size = 1
    hidden_size = 20
    num_layer   = 3
    batch_first = True

    # train pram
    lr         = 0.01
    epochs     = 20
    batch_size = 32
    test_size  = .2 # % of test datasets
    
    # initialize train class
    train = Train(input_size, output_size, hidden_size, num_layer, batch_first, lr)

    # create datasest from timeseries
    awake_inputs, awake_labels, awake_times = train.make_dataset(awake_num, sequence_length, t_start, awake_yaw, 100)
    drow_inputs,  drow_labels,  drow_times  = train.make_dataset(drow_num,  sequence_length, t_start, drow_yaw,  100)

    # split dataset into train & test (only awake datasets)
    awake_train_inputs, awake_test_inputs, awake_train_labels, awake_test_labels\
        = train_test_split(awake_inputs, awake_labels, test_size=test_size, shuffle=False)
    awake_train_times, awake_test_times\
        = train_test_split(awake_times, test_size=test_size, shuffle=False)

    # model training 
    train.train(awake_train_inputs, awake_train_labels, awake_test_inputs, awake_test_labels,\
                epochs, batch_size, sequence_length, input_size, plot=True)

    # check prediction result
    epsss_ave_awake = train.pred_result_UAH(awake_test_inputs, awake_test_labels, awake_test_times,\
                                            sequence_length,   input_size,        plot=True)
    epsss_ave_drow  = train.pred_result_UAH(drow_inputs,       drow_labels, drow_times,\
                                            sequence_length,   input_size, plot=True)

    epsss_ave_df_awake          = pd.DataFrame(epsss_ave_awake, columns=['epss'])
    epsss_ave_df_awake['label'] = pd.DataFrame(np.zeros_like(epsss_ave_awake))
    epsss_ave_df_drow           = pd.DataFrame(epsss_ave_drow, columns=['epss'])
    epsss_ave_df_drow['label']  = pd.DataFrame(np.ones_like(epsss_ave_drow))
    epsss_ave_df = pd.concat([epsss_ave_df_awake, epsss_ave_df_drow])

    df_beta_ave = pd.DataFrame()
    y_true      = epsss_ave_df["label"]
    y_score     = epsss_ave_df["epss"]

    fpr, tpr, threshold = roc_curve(y_true, y_score)
    roc_auc_value       = roc_auc_score(y_true, y_score)

    # f1 Value
    precision, recall, threshold_from_pr = precision_recall_curve(y_true, probas_pred = y_score)
    a  = 2* precision * recall
    b = precision + recall
    f1 = np.divide(a,b,out=np.zeros_like(a), where=b!=0)

    idx_opt         = np.argmax(f1)
    threshold_opt   = threshold_from_pr[idx_opt] #Confusion Matrix
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
    plt.grid()
    #plt.savefig('sklearn_roc_curve.pdf')
    plt.show()

    print("elapse time", time.time()-start)
