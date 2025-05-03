"""Evaluation pipeline for trained models in driver drowsiness detection.

This script selects the appropriate evaluation routine based on the model type:
- LSTM (deep learning)
- SVM-ANFIS (optimized with PSO)
- Common classifiers (e.g., Random Forest, SvmW)

The evaluation includes data loading, preprocessing, and performance logging.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import logging
import joblib
from sklearn.metrics import classification_report, roc_curve, auc, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from src.evaluation.models.lstm import AttentionLayer 

from src.config import SUBJECT_LIST_PATH, PROCESS_CSV_PATH, MODEL_PKL_PATH
from src.utils.io.loaders import read_subject_list, get_model_type, load_subject_csvs
from src.utils.io.split import data_split
from src.models.feature_selection.index import calculate_feature_indices
from src.evaluation.models.lstm import lstm_eval
from src.evaluation.models.SvmA import SvmA_eval
from src.evaluation.models.common import common_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def eval_pipeline(model: str, tag: str = None, sample_size: int = None, seed: int = 42) -> None:
    """Evaluate the specified trained model using appropriate method.

    The evaluation routine is selected based on the `model` name.
    For SvmA and LSTM models, custom evaluators are used.
    For other models, a generic evaluation method is used.

    Args:
        model (str): Model name (e.g., 'Lstm', 'SvmA', 'RF', 'SvmW').

    Returns:
        None
    """

    # Load subject list and determine model type
    subject_list = read_subject_list()
    if sample_size is not None:
        rng = np.random.default_rng(seed)
        subject_list = rng.choice(subject_list, size=sample_size, replace=False).tolist()
        logging.info(f"Evaluating on {sample_size} subjects: {subject_list}")

    model_type = get_model_type(model)

    # Load preprocessed feature data
    combined_data = load_subject_csvs(subject_list, model_type, add_subject_id=False)
    X_train, X_val, X_test, y_train, y_val, y_test = data_split(combined_data)

    # Load trained model
    if model == "SvmA":
        model_file = "svm_model_final.pkl"
    elif model == "Lstm":
        model_file = "lstm_model_fold1.keras"
    else:
        model_file = f"{model}{f'_{tag}' if tag else ''}.pkl"
    model_path = os.path.join(MODEL_PKL_PATH, model_type, model_file)

    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return

    # Load model based on type
    if model == "Lstm":
        clf = load_model(model_path, custom_objects={"AttentionLayer": AttentionLayer})
    elif model == "SvmA":
        clf = joblib.load(model_path)
    else:
        with open(model_path, "rb") as f:
            clf = pickle.load(f)

    # call evaluation functions
    if model == 'Lstm':
        scaler_path = os.path.join(MODEL_PKL_PATH, model_type, f"scaler_fold1{f'_{tag}' if tag else ''}.pkl")
        if not os.path.exists(scaler_path):
            logging.error(f"Scaler file not found: {scaler_path}")
            return
        scaler = joblib.load(scaler_path)

        result = lstm_eval(X_test, y_test, model_type, clf, scaler)

    elif model == 'SvmA':
        feature_file = f"selected_features_train{f'_{tag}' if tag else ''}.pkl"
        feature_path = os.path.join(MODEL_PKL_PATH, model_type, feature_file)
        if not os.path.exists(feature_path):
            logging.error(f"Feature file not found: {feature_path}")
            return
        with open(feature_path, "rb") as ff:
            selected_features = pickle.load(ff)
        result = SvmA_eval(X_test, y_test, model, clf, selected_features)

    else:
        result = common_eval(X_train, X_test, y_train, y_test, model, model_type, clf)

    # After each evaluation call (e.g., result = common_eval(...))
    results_dir = os.path.join("results", model_type)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"metrics_{model}_{tag}.json" if tag else f"metrics_{model}.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
