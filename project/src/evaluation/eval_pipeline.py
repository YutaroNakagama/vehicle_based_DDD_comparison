import os
import pickle
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import classification_report, roc_curve, auc, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import SUBJECT_LIST_PATH, PROCESS_CSV_PATH, MODEL_PKL_PATH
from src.utils.io.loaders import read_subject_list
from src.models.feature_selection.index import calculate_feature_indices
from src.evaluation.models.lstm import lstm_eval
from src.evaluation.models.SvmA import SvmA_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_and_combine_files(subject_list, model_type):
    dfs = []
    for subject in subject_list:
        subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
        file_name = f'processed_{subject_id}_{version}.csv'
        path = os.path.join(PROCESS_CSV_PATH, model_type, file_name)
        try:
            df = pd.read_csv(path)
            dfs.append(df)
            logging.info(f"Loaded: {file_name}")
        except FileNotFoundError:
            logging.warning(f"File not found: {file_name}")

    return pd.concat(dfs, ignore_index=True)


def prepare_data(df):
    df = df[df["KSS_Theta_Alpha_Beta"].isin([1, 2, 8, 9])]
    X = df.iloc[:, 1:46].dropna()
    y = df.loc[X.index, "KSS_Theta_Alpha_Beta"].replace({1: 0, 2: 0, 8: 1, 9: 1})
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_model(clf, X_train, X_test, y_train, y_test, features):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[features])
    X_test_scaled = scaler.transform(X_test[features])

    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    roc_auc = 0
    if hasattr(clf, "predict_proba"):
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

    mse = mean_squared_error(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return mse, report, roc_auc


def eval_pipeline(model):
    subject_list = read_subject_list()
    model_type = model if model in {"SvmW", "SvmA", "Lstm"} else "common"

    combined_data = load_and_combine_files(subject_list, model_type)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(combined_data)

    if model == 'Lstm':
        lstm_eval(X_test, y_test, model_type)

    elif model == 'SvmA':
        feature_indices = calculate_feature_indices(X_train, y_train)
        SvmA_eval(X_test, y_test, model)

    else:
        model_filename = f"{MODEL_PKL_PATH}/{model_type}/{model}.pkl"
        features_filename = f"{MODEL_PKL_PATH}/{model_type}/{model}_feat.npy"

        with open(model_filename, "rb") as f:
            clf = pickle.load(f)
        
        selected_features = np.load(features_filename, allow_pickle=True)
        mse, report, roc_auc = evaluate_model(
            clf, X_train, X_test, y_train, y_test, selected_features
        )

        logging.info(f"Model: {model}")
        logging.info(f"MSE: {mse:.4f}")
        logging.info(f"ROC AUC: {roc_auc:.4f}")
        logging.info(f"Classification Report:\n{report}")


