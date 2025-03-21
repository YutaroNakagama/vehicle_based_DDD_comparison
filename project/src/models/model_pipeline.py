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
from src.models.feature_selection.index import calculate_feature_indices
from src.models.feature_selection.anfis import calculate_id
from src.models.architectures.lstm import lstm_train
from src.models.architectures.SvmA import SvmA_train

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_and_combine_data(subject_list, model_type):
    dfs = []
    for subject in subject_list:
        subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
        file_name = f'processed_{subject_id}_{version}.csv'
        file_path = f'{PROCESS_CSV_PATH}/{model_type}/{file_name}'
        try:
            df = pd.read_csv(file_path)
            dfs.append(df)
            logging.info(f"Loaded data for {subject_id}_{version}")
        except FileNotFoundError:
            logging.warning(f"File not found: {file_name}")

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def prepare_data(df):
    filtered_data = df[df["KSS_Theta_Alpha_Beta"].isin([1, 2, 8, 9])]
    X = filtered_data.iloc[:, 1:46].dropna()
    y = filtered_data.loc[X.index, "KSS_Theta_Alpha_Beta"].replace({1: 0, 2: 0, 8: 1, 9: 1})
    return train_test_split(X, y, test_size=0.3, random_state=42)


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


def train_pipeline(model):
    subject_list = read_subject_list()
    model_type = model if model in {"SvmW", "SvmA", "Lstm"} else "common"

    data = load_and_combine_data(subject_list, model_type)
    X_train, X_test, y_train, y_test = prepare_data(data)

    if model == 'Lstm':
        lstm_train(X_train, y_train, model)
    elif model == 'SvmA':
        feature_indices = calculate_feature_indices(X_train, y_train)
        SvmA_train(X_train, X_test, y_train, y_test, feature_indices, model)
    else:
        feature_indices = calculate_feature_indices(X_train, y_train)
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

        optimize_classifier(model, clf, X_train, X_test, y_train, y_test, feature_indices, model, model_type)

