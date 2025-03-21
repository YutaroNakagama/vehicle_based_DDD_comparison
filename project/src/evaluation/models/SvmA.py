import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve,
    precision_score, recall_score, f1_score,
)

from src.config import MODEL_PKL_PATH


def load_model_and_features(model):
    model_path = f'{MODEL_PKL_PATH}/{model}/svm_model_final.pkl'
    features_path = f'{MODEL_PKL_PATH}/{model}/selected_features_train.pkl'

    svm_model = joblib.load(model_path)
    selected_features = joblib.load(features_path).columns.tolist()

    return svm_model, selected_features


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    return accuracy, precision, recall, f1


def evaluate_model(svm_model, X_test, y_test):
    y_pred = svm_model.predict(X_test)

    accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision}")
    print(f"Recall    : {recall}")
    print(f"F1 Score  : {f1}")
    print("\nConfusion Matrix:\n", conf_matrix)


def SvmA_eval(X_test, y_test, model):
    svm_model, selected_features = load_model_and_features(model)
    X_selected_test = X_test[selected_features]

    evaluate_model(svm_model, X_selected_test, y_test)

