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
from src.utils.io.loaders import read_subject_list
from src.utils.io.split import data_split
from src.models.feature_selection.index import calculate_feature_indices
from src.models.feature_selection.anfis import calculate_id
from src.models.architectures.lstm import lstm_train
from src.models.architectures.SvmA import SvmA_train
from src.models.domain_generalization.data_aug import generate_domain_labels, domain_mixup

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_and_combine_data(subject_list, model_type):
    dfs = []
    for subject in subject_list:
        subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
        file_name = f'processed_{subject_id}_{version}.csv'
        file_path = f'{PROCESS_CSV_PATH}/{model_type}/{file_name}'
        try:
            df = pd.read_csv(file_path)
            df['subject_id'] = subject_id
            dfs.append(df)
            logging.info(f"Loaded data for {subject_id}_{version}")
        except FileNotFoundError:
            logging.warning(f"File not found: {file_name}")

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def optimize_classifier(name, clf, X_train, X_test, y_train, y_test, feature_indices, model, model_type):

    def objective_function(params):
        threshold = params[0]
        ids = calculate_id(feature_indices, params[1:])
        selected_indices = np.where(ids > threshold)[0]
        if len(selected_indices) == 0:
            return 1e6

        selected_features = X_train.columns[selected_indices]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[selected_features])
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(scaler.transform(X_test[selected_features]))

        return mean_squared_error(y_test, y_pred)

    lb = [0.5] + [0.1] * (len(feature_indices) - 1)
    ub = [0.9] + [1.0] * (len(feature_indices) - 1)
    optimized_params, _ = pso(objective_function, lb, ub, swarmsize=20, maxiter=10, debug=False)

    ids = calculate_id(feature_indices, optimized_params)
    selected_features = X_train.columns[np.where(ids > optimized_params[0])[0]]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[selected_features])
    X_test_scaled = scaler.transform(X_test[selected_features])

    clf.fit(X_train_scaled, y_train)

    with open(f"{MODEL_PKL_PATH}/{model_type}/{model}.pkl", "wb") as f:
        pickle.dump(clf, f)

    np.save(f"{MODEL_PKL_PATH}/{model_type}/{model}_feat.npy", selected_features)

    y_pred = clf.predict(X_test_scaled)
    roc_auc = 0
    if hasattr(clf, "predict_proba"):
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

    logging.info(f"{name} trained with ROC AUC: {roc_auc:.2f}")
    logging.info(classification_report(y_test, y_pred))


def train_pipeline(model, domain_generalize=True):
    subject_list = read_subject_list()
    model_type = model if model in {"SvmW", "SvmA", "Lstm"} else "common"

    data = load_and_combine_data(subject_list, model_type)
    X_train, X_val, X_test, y_train, y_val, y_test = data_split(data)

    if domain_generalize == True:
        domain_labels_train = generate_domain_labels(subject_list, X_train)
        X_train_aug, y_train_aug = domain_mixup(X_train, y_train, domain_labels_train)
        X_train, y_train = X_train_aug, y_train_aug

    if model == 'Lstm':
        lstm_train(X_train, y_train, model)
    elif model == 'SvmA':
        feature_indices = calculate_feature_indices(X_train, y_train)
        SvmA_train(X_train, X_test, y_train, y_test, feature_indices, model)
    else:
        X_train_for_fs = X_train.drop(columns=["subject_id"], errors='ignore')
        X_test_for_fs = X_test.drop(columns=["subject_id"], errors='ignore')
        
        feature_indices = calculate_feature_indices(X_train_for_fs, y_train)
        
        classifiers = {
            "RF": RandomForestClassifier(random_state=42),
            "SvmW": SVC(kernel="rbf", probability=True, random_state=42),
            #"Decision Tree": DecisionTreeClassifier(random_state=42),
            #"AdaBoost": AdaBoostClassifier(random_state=42),
            #"Gradient Boosting": GradientBoostingClassifier(random_state=42),
            #"XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
            #"LightGBM": lgb.LGBMClassifier(random_state=42),
            #"CatBoost": CatBoostClassifier(verbose=0, random_state=42),
            #"Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            #"Perceptron": Perceptron(max_iter=1000, random_state=42),
            #"SVM (Linear Kernel)": SVC(kernel="linear", probability=True, random_state=42),
            #"SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, random_state=42),
            #"K-Nearest Neighbors": KNeighborsClassifier(),
            #"MLP (Neural Network)": MLPClassifier(max_iter=500, random_state=42),
        }

        clf = classifiers.get(model)
        if clf is None:
            logging.error(f"Model '{model}' not recognized.")
            return

        optimize_classifier(model, clf, X_train_for_fs, X_test_for_fs, y_train, y_test, feature_indices, model, model_type)

