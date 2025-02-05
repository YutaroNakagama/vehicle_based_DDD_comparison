import warnings
warnings.filterwarnings("ignore")

import os
import sys
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

    # load dataset
    dataset = Datasets()
    #df_veh, _ = dataset.load_veh(output_type="full", filt=True)
    #df_eeg = dataset.load_eeg(data_type="full")
    veh_eeg_all, df_veh_eeg = dataset.load_veh_eeg_ppb(data_type="full")
    # ---------------------------------------------------------------------- #
    # dataset parameters
    sequence_length = 5
    veh_eeg_num = len(veh_eeg_all) - sequence_length
    t_start = 0
    
    # model pram
    input_size = 1
    output_size = 1
    hidden_size = 10
    num_layer = 2
    batch_first = True

    # train pram
    lr = 0.01
    epochs = 50 
    batch_size = 32
    test_size = 0.50 # % of test datasets (train_val (1-size) vs. test (size))
    vali_size = 0.5 # % of vali datasets (train (1-size) vs. vali (size))

    # beta thld
    beta_lo_thld = 3 
    beta_hi_thld = 3

    # ---------------------------------------------------------------------- #

    # initialize train class
    train_normal = Train(input_size, output_size, hidden_size, num_layer, batch_first, lr)
    train_abnormal = Train(input_size, output_size, hidden_size, num_layer, batch_first, lr)
    
    # create datasest from timeseries
    veh_eeg_inputs, veh_eeg_labels, veh_eeg_times = train_normal.make_dataset(veh_eeg_num, sequence_length, t_start, veh_eeg_all)
    _,_,_ = train_abnormal.make_dataset(veh_eeg_num, sequence_length, t_start, veh_eeg_all)
    
    print("type(veh_eeg_labels)", type(veh_eeg_labels))
    print("veh_eeg_labels[1,1] ", veh_eeg_labels[1,1])
    print("veh_eeg_labels[1,:] ", veh_eeg_labels[1,:])

    drow_inputs = np.empty((1, veh_eeg_inputs.shape[1], veh_eeg_inputs.shape[2]))
    drow_labels = np.empty((1, veh_eeg_labels.shape[1]))
    drow_times  = np.empty((1))
    awake_inputs = np.empty((1, veh_eeg_inputs.shape[1], veh_eeg_inputs.shape[2]))
    awake_labels = np.empty((1, veh_eeg_labels.shape[1]))
    awake_times  = np.empty((1))

    drow_inputs = veh_eeg_inputs[veh_eeg_labels[:,1] < beta_lo_thld,:,:1]
    drow_labels = veh_eeg_labels[veh_eeg_labels[:,1] < beta_lo_thld,:1]
    drow_times  = veh_eeg_times[veh_eeg_labels[:,1] < beta_lo_thld]
    drow_outputs = veh_eeg_inputs[veh_eeg_labels[:,1] < beta_lo_thld,:,1]
    awake_inputs = veh_eeg_inputs[veh_eeg_labels[:,1] > beta_hi_thld,:,:1]
    awake_labels = veh_eeg_labels[veh_eeg_labels[:,1] > beta_hi_thld,:1]
    awake_times  = veh_eeg_times[veh_eeg_labels[:,1] > beta_hi_thld]
    awake_outputs = veh_eeg_inputs[veh_eeg_labels[:,1] > beta_hi_thld,:,1]

    print("number of normal/abnormal datasets")
    print("awake  (normal)  : ", len(awake_times))
    print("drowsy (abnormal): ", len(drow_times))
    
    # split dataset into train & test (only awake datasets)

    awake_train_val_inputs, awake_test_inputs, awake_train_val_labels, awake_test_labels\
        = train_test_split(awake_inputs, awake_labels, test_size=test_size, shuffle=False)
    awake_train_inputs, awake_vali_inputs, awake_train_labels, awake_vali_labels\
        = train_test_split(awake_train_val_inputs, awake_train_val_labels, test_size=vali_size, shuffle=False)

    awake_train_val_times, awake_test_times\
        = train_test_split(awake_times, test_size=test_size, shuffle=False)
    awake_train_times, awake_vali_times\
        = train_test_split(awake_train_val_times, test_size=vali_size, shuffle=False)
    
    awake_train_val_outputs, awake_test_outputs\
        = train_test_split(awake_outputs, test_size=test_size, shuffle=False)
    awake_train_outputs, awake_vali_outputs\
        = train_test_split(awake_train_val_outputs, test_size=vali_size, shuffle=False)
    
    drow_train_val_inputs, drow_test_inputs, drow_train_val_labels, drow_test_labels\
        = train_test_split(drow_inputs, drow_labels, test_size=test_size, shuffle=False)
    drow_train_inputs, drow_vali_inputs, drow_train_labels, drow_vali_labels\
        = train_test_split(drow_train_val_inputs, drow_train_val_labels, test_size=vali_size, shuffle=False)

    drow_train_val_times, drow_test_times\
        = train_test_split(drow_times, test_size=test_size, shuffle=False)
    drow_train_times, drow_vali_times\
        = train_test_split(drow_train_val_times, test_size=vali_size, shuffle=False)
    
    drow_train_val_outputs, drow_test_outputs\
        = train_test_split(drow_outputs, test_size=test_size, shuffle=False)
    drow_train_outputs, drow_vali_outputs\
        = train_test_split(drow_train_val_outputs, test_size=vali_size, shuffle=False)
    
    X_train = np.concatenate([awake_train_inputs, drow_train_inputs])
    #y_train = np.concatenate([awake_train_outputs, drow_train_outputs])
    y_train = np.concatenate([np.zeros_like(awake_train_times),np.ones_like(drow_train_times)])
    X_test  = np.concatenate([awake_test_inputs, drow_test_inputs])
    #y_test  = np.concatenate([awake_train_outputs, drow_train_outputs])
    y_test  = np.concatenate([np.zeros_like(awake_test_times),np.ones_like(drow_test_times)])

    from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
    classifier = KNeighborsTimeSeriesClassifier(distance="euclidean")
    print(classifier) 
    classifier.fit(X_train, y_train) 
    y_pred = classifier.predict(X_test)

    y_true  = y_test
    y_score = y_pred

    fpr, tpr, threshold = roc_curve(y_true, y_score)
    roc_auc_value = roc_auc_score(y_true, y_score)
    print("roc_auc_value: ", roc_auc_value)

    # f1 Value
    precision, recall, threshold_from_pr = precision_recall_curve(y_true, probas_pred = y_score)
    #print("precision", precision)
    #print("recall", recall)
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

    plt.plot(fpr, tpr, marker='.')
    plt.title('AUC:{:.4f}'.format(roc_auc_value))
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    plt.savefig('roc_len_'+str(sequence_length)+\
                '_hid-si_'+str(hidden_size)+\
                '_layer_'+str(num_layer)+\
                '_test_size_'+str(test_size)+\
                '_train_vali.pdf')
    plt.show()

    sys.exit()

    print("model training") 
    #train.train(awake_train_inputs, awake_train_labels, awake_test_inputs, awake_test_labels,\
    #train.train(awake_train_inputs, awake_train_labels, awake_vali_inputs, awake_vali_labels,\
    train_normal.train(awake_test_inputs, awake_test_labels, awake_test_inputs, awake_test_labels,\
                epochs, batch_size, sequence_length, input_size, plot=True)
    train_abnormal.train(drow_test_inputs, drow_test_labels, drow_test_inputs, drow_test_labels,\
                epochs, batch_size, sequence_length, input_size, plot=True)

    print("number of normal/abnormal datasets for train")
    print("awake  (normal):   ", len(awake_test_labels))
    print("drowsy (abnormal): ", len(drow_test_labels))
    print("number of normal/abnormal datasets for vali")
    print("awake  (normal):   ", len(awake_test_labels))
    print("drowsy (abnormal): ", len(drow_test_labels))
    
    # check prediction result
    awake_test_cnt, drow_test_cnt = [],[]
    awake_vali_cnt, drow_vali_cnt = [],[]
    awake_train_cnt, drow_train_cnt = [],[]
    print("awake")
    #epsss_ave_awake = train.pred_result(awake_train_inputs, awake_train_labels, awake_train_times,\
    #                                    awake_train_cnt, sequence_length, input_size, plot=True, sample_type="full")
    epsss_ave_awake_normal = train_normal.pred_result(awake_test_inputs, awake_test_labels, awake_test_times,\
                                        awake_test_cnt, sequence_length, input_size, plot=True, sample_type="full")
    epsss_ave_awake_abnormal = train_abnormal.pred_result(awake_test_inputs, awake_test_labels, awake_test_times,\
                                        awake_train_cnt, sequence_length, input_size, plot=True, sample_type="full")
    epsss_ave_awake = epsss_ave_awake_normal / epsss_ave_awake_abnormal

    np.savetxt("epsss_ave_awake_normal.csv",epsss_ave_awake_normal,delimiter=",")
    np.savetxt("epsss_ave_awake_abnormal.csv",epsss_ave_awake_abnormal,delimiter=",")
    np.savetxt("epsss_ave_awake.csv",epsss_ave_awake,delimiter=",")

    #sys.exit()
    print("drow")
    epsss_ave_drow_normal = train_normal.pred_result(drow_test_inputs, drow_test_labels, drow_test_times,\
                                       drow_test_cnt, sequence_length, input_size, plot=True, sample_type="full")
    epsss_ave_drow_abnormal = train_abnormal.pred_result(drow_test_inputs, drow_test_labels, drow_test_times,\
                                       drow_test_cnt, sequence_length, input_size, plot=True, sample_type="full")
    epsss_ave_drow = epsss_ave_drow_normal / epsss_ave_drow_abnormal

    np.savetxt("epsss_ave_drow_normal.csv",epsss_ave_drow_normal,delimiter=",")
    np.savetxt("epsss_ave_drow_abnormal.csv",epsss_ave_drow_abnormal,delimiter=",")
    np.savetxt("epsss_ave_drow.csv",epsss_ave_drow,delimiter=",")
    
    print("number of normal/abnormal datasets for test")
    print("awake  (normal):   ", len(epsss_ave_awake[:,0]))
    print("drowsy (abnormal): ", len(epsss_ave_drow[:,0]))

    plt.hist(epsss_ave_awake[:,0], alpha=0.5, label="awake")
    plt.hist(epsss_ave_drow[:,0],  alpha=0.5, label="drow")
    plt.legend()
    plt.show()
    
    epsss_ave_df_awake = pd.DataFrame(epsss_ave_awake[:,0], columns=["epss"])
    epsss_ave_df_awake["drowsiness"] = 0
    epsss_ave_df_drow  = pd.DataFrame(data=epsss_ave_drow[:,0], columns=["epss"])
    epsss_ave_df_drow["drowsiness"] = 1
    epsss_ave_df = pd.concat([epsss_ave_df_awake, epsss_ave_df_drow])
    #epsss_ave_df.columns = ['epss','cnt']
    
    y_true  = epsss_ave_df["drowsiness"]
    y_score = epsss_ave_df["epss"]

    fpr, tpr, threshold = roc_curve(y_true, y_score)
    roc_auc_value = roc_auc_score(y_true, y_score)
    print("roc_auc_value: ", roc_auc_value)

    # f1 Value
    precision, recall, threshold_from_pr = precision_recall_curve(y_true, probas_pred = y_score)
    #print("precision", precision)
    #print("recall", recall)
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

    plt.plot(fpr, tpr, marker='.')
    plt.title('AUC:{:.4f}'.format(roc_auc_value))
    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    plt.savefig('roc_len_'+str(sequence_length)+\
                '_hid-si_'+str(hidden_size)+\
                '_layer_'+str(num_layer)+\
                '_test_size_'+str(test_size)+\
                '_train_vali.pdf')
    plt.show()

    print("elapse time", time.time()-start)
