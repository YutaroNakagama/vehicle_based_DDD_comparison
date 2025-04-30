"""Evaluation pipeline for trained models in driver drowsiness detection.

This script selects the appropriate evaluation routine based on the model type:
- LSTM (deep learning)
- SVM-ANFIS (optimized with PSO)
- Common classifiers (e.g., Random Forest, SvmW)

The evaluation includes data loading, preprocessing, and performance logging.
"""

import os
import pickle
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import classification_report, roc_curve, auc, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import SUBJECT_LIST_PATH, PROCESS_CSV_PATH, MODEL_PKL_PATH
from src.utils.io.loaders import read_subject_list, get_model_type, load_subject_csvs
from src.utils.io.split import data_split
from src.models.feature_selection.index import calculate_feature_indices
from src.evaluation.models.lstm import lstm_eval
from src.evaluation.models.SvmA import SvmA_eval
from src.evaluation.models.common import common_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def eval_pipeline(model: str) -> None:
    """Evaluate the specified trained model using appropriate method.

    The evaluation routine is selected based on the `model` name.
    For SvmA and LSTM models, custom evaluators are used.
    For other models, a generic evaluation method is used.

    Args:
        model (str): Model name (e.g., 'Lstm', 'SvmA', 'RF', 'SvmW').

    Returns:
        None
    """
    subject_list = read_subject_list()
    model_type = get_model_type(model)
    combined_data = load_subject_csvs(subject_list, model_type, add_subject_id=False)
    X_train, X_val, X_test, y_train, y_val, y_test = data_split(combined_data)

    if model == 'Lstm':
        lstm_eval(X_test, y_test, model_type)

    elif model == 'SvmA':
        feature_indices = calculate_feature_indices(X_train, y_train)
        SvmA_eval(X_test, y_test, model)

    else:
        common_eval(X_train, X_test, y_train, y_test, model, model_type)

