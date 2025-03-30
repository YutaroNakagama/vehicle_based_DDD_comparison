import os
import sys
import pickle
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from pyswarm import pso

from src.config import SUBJECT_LIST_PATH, PROCESS_CSV_PATH, MODEL_PKL_PATH
from src.utils.io.loaders import read_subject_list, get_model_type, load_subject_csvs
from src.utils.io.split import data_split
from src.utils.domain_generalization.domain_mixup import generate_domain_labels, domain_mixup
from src.utils.domain_generalization.coral import coral
from src.utils.domain_generalization.vae_augment import vae_augmentation
from src.models.feature_selection.index import calculate_feature_indices
from src.models.feature_selection.anfis import calculate_id
from src.models.architectures.helpers import get_classifier
from src.models.architectures.lstm import lstm_train
from src.models.architectures.SvmA import SvmA_train
from src.models.architectures.common import common_train

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_pipeline(model, use_domain_mixup=False, use_coral=False, use_vae=False):
    subject_list = read_subject_list()
    model_type = get_model_type(model)
    data = load_subject_csvs(subject_list, model_type, add_subject_id=True)
    X_train, X_val, X_test, y_train, y_val, y_test = data_split(data)


    if use_domain_mixup:
        domain_labels_train = generate_domain_labels(subject_list, X_train)
        X_train_aug, y_train_aug = domain_mixup(X_train, y_train, domain_labels_train)
        X_train, y_train = X_train_aug, y_train_aug


    if use_coral:
        # Here, for demonstration purposes, X_val is used as target domain
        X_train_numeric = X_train.select_dtypes(include=[np.number]).values
        X_val_numeric = X_val.select_dtypes(include=[np.number]).values

        X_train_aligned = coral(X_train_numeric, X_val_numeric)
        
        # Reconstruct DataFrame after CORAL
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train[numeric_cols] = X_train_aligned


    if use_vae:
        X_train = vae_augmentation(X_train, augment_ratio=0.3)


    if model == 'Lstm':
        lstm_train(X_train, y_train, model)
    elif model == 'SvmA':
        feature_indices = calculate_feature_indices(X_train, y_train)
        SvmA_train(X_train, X_test, y_train, y_test, feature_indices, model)
    else:
        X_train_for_fs = X_train.drop(columns=["subject_id"], errors='ignore')
        X_test_for_fs = X_test.drop(columns=["subject_id"], errors='ignore')

        X_train_for_fs = X_train_for_fs.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

        feature_indices = calculate_feature_indices(X_train_for_fs, y_train)
        clf = get_classifier(model)
        common_train(X_train_for_fs, X_test_for_fs, y_train, y_test, feature_indices, model, model_type, clf)

