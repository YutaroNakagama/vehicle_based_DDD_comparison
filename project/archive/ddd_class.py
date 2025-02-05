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
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.interval_based import RandomIntervalSpectralEnsemble
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.dictionary_based import BOSSEnsemble 
from sktime.classification.interval_based import RandomIntervalSpectralEnsemble
from sktime.classification.shapelet_based import ShapeletTransformClassifier 
from sktime.classification.deep_learning.cnn import CNNClassifier
from sktime.classification.deep_learning  import LSTMFCNClassifier
from sktime.classification.deep_learning.resnet import ResNetClassifier
from sktime.classification.deep_learning.rnn import SimpleRNNClassifier 

# class in other files
from module.preprocess import Preprocess
from module.dataset import Datasets
from module.train import PredictSimpleFormulaNet, Train 

if __name__ == '__main__':
    start = time.time()

    beta_thld = 0.7
    # load dataset
    dataset = Datasets()
    df_filt, curv_time = dataset.load_veh(output_type="curve", filt=True)
    #df_filt.to_csv("df_filt.csv")
    df_eeg, curv_time_drow = dataset.load_eeg(data_type='curve',curv_time = curv_time)
    #curv_time_drow.to_csv("beta_ave.csv")

    # split dataset by time index
    #train_idx,test_idx = curv_time[:2],curv_time[2:]
    awake, drow = dataset.make_dataset(threshold = beta_thld) 
    #test_input,test_labels = dataset.make_dataset(test_idx) 
    
    awake_yaw = awake[["yaw_rate_a"]].values.astype('float32') 
    drow_yaw = drow[["yaw_rate_a"]].values.astype('float32') 
    awake_cnt = awake[["curv_cnt"]].values.astype('int') 
    drow_cnt = drow[["curv_cnt"]].values.astype('int') 

    # dataset parameters
    sequence_length = 20
    awake_num = len(awake_yaw) - sequence_length
    drow_num = len(drow_yaw) - sequence_length
    t_start = 0

    # model pram
    input_size = 2
    output_size = 1
    hidden_size = 10
    batch_first = True

    # train pram
    lr = 0.001
    epochs = 15
    batch_size = 128
    test_size = 0.6
    
    # initialize train class
    train = Train(input_size, output_size, hidden_size, batch_first, lr)

    # create datasest from timeseries
    # awake
    awake_inputs, awake_labels, awake_times\
        = train.make_dataset(awake_num, sequence_length, t_start, awake_yaw)
    # awake
    _, awake_cnt_labels, _\
        = train.make_dataset(awake_num, sequence_length, t_start, awake_cnt)
    # drow
    drow_inputs, drow_labels, drow_times\
        = train.make_dataset(drow_num, sequence_length, t_start, drow_yaw)
    # awake
    _, drow_cnt_labels, _\
        = train.make_dataset(drow_num, sequence_length, t_start, drow_cnt)

    # split dataset into train & test
    # awake
    awake_train_inputs, awake_test_inputs, awake_train_labels, awake_test_labels\
        = train_test_split(awake_inputs, awake_labels, test_size=test_size, shuffle=False)
    awake_train_times, awake_test_times\
        = train_test_split(awake_times, test_size=test_size, shuffle=False)
    awake_train_cnt, awake_test_cnt\
        = train_test_split(awake_cnt_labels, test_size=test_size, shuffle=False)
    # drowe
    drow_train_inputs, drow_test_inputs, drow_train_labels, drow_test_labels\
        = train_test_split(drow_inputs, drow_labels, test_size=test_size, shuffle=False)
    drow_train_times, drow_test_times\
        = train_test_split(drow_times, test_size=test_size, shuffle=False)
    drow_train_cnt, drow_test_cnt\
        = train_test_split(drow_cnt_labels, test_size=test_size, shuffle=False)

    X_train, X_test = pd.DataFrame(), pd.DataFrame()
    y_train, y_test = pd.DataFrame(), pd.DataFrame()

    for i in np.unique(awake_train_cnt).astype('int').tolist():
        X_train = pd.concat([X_train, df_filt["yaw_rate_a"][df_filt["curv_cnt"]==i]]) 
        y_train = pd.concat([y_train, curv_time_drow.loc[[i]]]) 
    for i in np.unique(drow_train_cnt).astype('int').tolist():
        X_train = pd.concat([X_train, df_filt["yaw_rate_a"][df_filt["curv_cnt"]==i]]) 
        y_train = pd.concat([y_train, curv_time_drow.loc[[i]]]) 
    for i in np.unique(awake_test_cnt).astype('int').tolist():
        X_test = pd.concat([X_test, df_filt["yaw_rate_a"][df_filt["curv_cnt"]==i]]) 
        y_test = pd.concat([y_test, curv_time_drow.loc[[i]]]) 
    for i in np.unique(drow_test_cnt).astype('int').tolist():
        X_test = pd.concat([X_test, df_filt["yaw_rate_a"][df_filt["curv_cnt"]==i]]) 
        y_test = pd.concat([y_test, curv_time_drow.loc[[i]]]) 

    y_train = np.array(y_train["beta_ave"] > beta_thld) 
    y_test = np.array(y_test["beta_ave"] > beta_thld) 
    X_train = X_train.values.reshape(len(y_train),int(len(X_train)/len(y_train))) 
    X_test = X_test.values.reshape(len(y_test),int(len(X_test)/len(y_test)))
    X_test, X_pred, y_test, y_pred = train_test_split(X_test,y_test,train_size=0.5)
    y_true=(y_test==False)
    
    print("# ------------------------------------------------------ #")
    print("KNeighborsTimeSeriesClassifier")
    class_start = time.time()
    classifier = KNeighborsTimeSeriesClassifier(distance="euclidean")
    classifier.fit(X_train, y_train)
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    y_pred=(y_test_pred==False)
    print("confusion_matrix: \n",confusion_matrix(y_test,y_test_pred))
    print("f1:          ",f1_score(y_true=(y_test==False),y_pred=(y_test_pred==False)))
    print("acc:         ",accuracy_score(y_true, y_pred))
    print("precision:   ",precision_score(y_true, y_pred))
    print("recall:      ",recall_score(y_true, y_pred))
    print("elapse time  ",time.time()-class_start)

    print("# ------------------------------------------------------ #")
    print("RandomIntervalSpectralEnsemble")
    class_start = time.time()
    classifier = RandomIntervalSpectralEnsemble()
    classifier.fit(X_train, y_train)
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    y_pred=(y_test_pred==False)
    print("confusion_matrix: ",confusion_matrix(y_test,y_test_pred))
    print("f1:          ",f1_score(y_true=(y_test==False),y_pred=(y_test_pred==False)))
    print("acc:         ",accuracy_score(y_true, y_pred))
    print("precision:   ",precision_score(y_true, y_pred))
    print("recall:      ",recall_score(y_true, y_pred))
    print("elapse time  ",time.time()-class_start)

    print("# ------------------------------------------------------ #")
    print("TimeSeriesForestClassifier")
    class_start = time.time()
    classifier = TimeSeriesForestClassifier()
    classifier.fit(X_train, y_train)
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    y_pred=(y_test_pred==False)
    print("confusion_matrix: ",confusion_matrix(y_test,y_test_pred))
    print("f1:          ",f1_score(y_true=(y_test==False),y_pred=(y_test_pred==False)))
    print("acc:         ",accuracy_score(y_true, y_pred))
    print("precision:   ",precision_score(y_true, y_pred))
    print("recall:      ",recall_score(y_true, y_pred))
    print("elapse time  ",time.time()-class_start)

    print("# ------------------------------------------------------ #")
    print("BOSSEnsemble")
    class_start = time.time()
    classifier = BOSSEnsemble()
    classifier.fit(X_train, y_train)
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    y_pred=(y_test_pred==False)
    print("confusion_matrix: ",confusion_matrix(y_test,y_test_pred))
    print("f1:          ",f1_score(y_true=(y_test==False),y_pred=(y_test_pred==False)))
    print("acc:         ",accuracy_score(y_true, y_pred))
    print("precision:   ",precision_score(y_true, y_pred))
    print("recall:      ",recall_score(y_true, y_pred))
    print("elapse time  ",time.time()-class_start)

    print("# ------------------------------------------------------ #")
    print("RandomIntervalSpectralEnsemble")
    class_start = time.time()
    classifier = RandomIntervalSpectralEnsemble()
    classifier.fit(X_train, y_train)
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    y_pred=(y_test_pred==False)
    print("confusion_matrix: ",confusion_matrix(y_test,y_test_pred))
    print("f1:          ",f1_score(y_true=(y_test==False),y_pred=(y_test_pred==False)))
    print("acc:         ",accuracy_score(y_true, y_pred))
    print("precision:   ",precision_score(y_true, y_pred))
    print("recall:      ",recall_score(y_true, y_pred))
    print("elapse time  ",time.time()-class_start)

    print("# ------------------------------------------------------ #")
    print("ShapeletTransformClassifier")
    class_start = time.time()
    classifier = ShapeletTransformClassifier()
    classifier.fit(X_train, y_train)
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    y_pred=(y_test_pred==False)
    print("confusion_matrix: ",confusion_matrix(y_test,y_test_pred))
    print("f1:          ",f1_score(y_true=(y_test==False),y_pred=(y_test_pred==False)))
    print("acc:         ",accuracy_score(y_true, y_pred))
    print("precision:   ",precision_score(y_true, y_pred))
    print("recall:      ",recall_score(y_true, y_pred))
    print("elapse time  ",time.time()-class_start)

    print("# ------------------------------------------------------ #")
    print("CNNClassifier")
    class_start = time.time()
    classifier = CNNClassifier()
    classifier.fit(X_train, y_train)
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    y_pred=(y_test_pred==False)
    print("confusion_matrix: ",confusion_matrix(y_test,y_test_pred))
    print("f1:          ",f1_score(y_true=(y_test==False),y_pred=(y_test_pred==False)))
    print("acc:         ",accuracy_score(y_true, y_pred))
    print("precision:   ",precision_score(y_true, y_pred))
    print("recall:      ",recall_score(y_true, y_pred))
    print("elapse time  ",time.time()-class_start)

    print("# ------------------------------------------------------ #")
    print("LSTMFCNClassifier")
    class_start = time.time()
    classifier = LSTMFCNClassifier()
    classifier.fit(X_train, y_train)
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    y_pred=(y_test_pred==False)
    print("confusion_matrix: ",confusion_matrix(y_test,y_test_pred))
    print("f1:          ",f1_score(y_true=(y_test==False),y_pred=(y_test_pred==False)))
    print("acc:         ",accuracy_score(y_true, y_pred))
    print("precision:   ",precision_score(y_true, y_pred))
    print("recall:      ",recall_score(y_true, y_pred))
    print("elapse time  ",time.time()-class_start)

    print("# ------------------------------------------------------ #")
    print("ResNetClassifier")
    class_start = time.time()
    classifier = ResNetClassifier()
    classifier.fit(X_train, y_train)
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    y_pred=(y_test_pred==False)
    print("confusion_matrix: ",confusion_matrix(y_test,y_test_pred))
    print("f1:          ",f1_score(y_true=(y_test==False),y_pred=(y_test_pred==False)))
    print("acc:         ",accuracy_score(y_true, y_pred))
    print("precision:   ",precision_score(y_true, y_pred))
    print("recall:      ",recall_score(y_true, y_pred))
    print("elapse time  ",time.time()-class_start)

    print("# ------------------------------------------------------ #")
    print("SimpleRNNClassifier")
    class_start = time.time()
    classifier = SimpleRNNClassifier()
    classifier.fit(X_train, y_train)
    y_train_pred = classifier.predict(X_train)
    y_test_pred = classifier.predict(X_test)

    y_pred=(y_test_pred==False)
    print("confusion_matrix: ",confusion_matrix(y_test,y_test_pred))
    print("f1:          ",f1_score(y_true=(y_test==False),y_pred=(y_test_pred==False)))
    print("acc:         ",accuracy_score(y_true, y_pred))
    print("precision:   ",precision_score(y_true, y_pred))
    print("recall:      ",recall_score(y_true, y_pred))
    print("elapse time  ",time.time()-class_start)

    print("# ------------------------------------------------------ #")

    print("elapse time", time.time()-start)
