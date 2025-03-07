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

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

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
    #veh_eeg_all, df_veh_eeg = dataset.load_veh_eeg_ppb(data_type="full")
    veh_eeg_all = dataset.load_veh_eeg_Aref(data_type="full")
    veh_eeg_all = veh_eeg_all.replace([np.inf, -np.inf], np.nan)
    veh_eeg_all = veh_eeg_all.dropna(how='any') 
    print(veh_eeg_all.head())
    X = veh_eeg_all[["engy_str_no_road","str_vel_dir_zcr","kurtosis_str_vel","samen_str_no_road","samen_str_vel"]]
    Y = veh_eeg_all[["drowsy"]]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, random_state=0)

    print("training",len(y_train))
    print(type(y_train))
    print("True" ,(y_train==True).sum().sum())
    print("False",(y_train==False).sum().sum())
    print("test",len(y_test))
    print("True" ,(y_test==True).sum().sum())
    print("False",(y_test==False).sum().sum())

    #Import svm model
    from sklearn import svm

    #Create a svm Classifier
    clf = svm.SVC(kernel='rbf')

    #Train the model using the training sets
    clf.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics

    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:",metrics.precision_score(y_test, y_pred))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:",metrics.recall_score(y_test, y_pred))

    print("confusion matrix:\n",metrics.confusion_matrix(y_test, y_pred))

    from sklearn.metrics import RocCurveDisplay, roc_curve

    y_score = clf.decision_function(X_test)

    print("AUC:",metrics.roc_auc_score(y_test, y_score))
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf.classes_[1])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.show()

    from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve

    prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=clf.classes_[1])
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
    plt.show()

    #    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    #    roc_display.plot(ax=ax1)
    #    pr_display.plot(ax=ax2)
    #    plt.show()

    X = veh_eeg_all[["engy_str_no_road","str_vel_dir_zcr","kurtosis_str_vel","samen_str_no_road","samen_str_vel"]]
    T = veh_eeg_all[["drowsy"]]
    
    # ここではテストデータは使わないが、形式上学習用とテスト用に分けておく
    X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=0.2, stratify=T, random_state=0)
    
    pipe = make_pipeline(StandardScaler(),
                         SVC()) # パラメータはデフォルト
    # learning_curve関数で交差検証（k=10）
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipe,
                                                            X = X_train, y = T_train,
                                                            train_sizes=np.linspace(0.1, 1.0, 10), # 与えたデータセットの何割を使用するかを指定
                                                            cv=10, n_jobs=1)
    # 学習曲線の描画
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(8,6))
    plt.plot(train_sizes, train_mean, marker='o', label='Train accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.2)
    plt.plot(train_sizes, test_mean, marker='s', linestyle='--', label='Validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.2)
    plt.grid()
    plt.title('Learning curve', fontsize=16)
    plt.xlabel('Number of training data sizes', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    #plt.ylim([0.5, 1.05])
    plt.show()

    # SVMのパラメータgammaを変化させる
    param_range = [1e-5,1e-4,1e-3,1e-2,1e-1,1,1e1,1e2,1e3]
    # validation_curve関数で交差検証
    train_scores, test_scores = validation_curve(estimator=pipe,
                                                 X=X_train, y=T_train,
                                                 param_name='svc__gamma',
                                                 param_range=param_range, cv=10)
    
    # 検証曲線の描画
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.figure(figsize=(8,6))
    plt.plot(param_range, train_mean, marker='o', label='Train accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.2)
    plt.plot(param_range, test_mean, marker='s', linestyle='--', label='Validation accuracy')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.2)
    plt.grid()
    plt.xscale('log')
    plt.title('Validation curve(gamma)', fontsize=16)
    plt.xlabel('Parameter gamma', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    #plt.ylim([0.5, 1.05])
    plt.show()

    print("elapse time", time.time()-start)
