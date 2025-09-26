"""Evaluation pipeline for trained models in driver drowsiness detection.

This module provides the `eval_pipeline` function, which orchestrates the evaluation of various
trained driver drowsiness detection (DDD) models. It handles data loading, preprocessing
(including subject-wise splitting if specified), model loading, and dispatches to
model-specific evaluation routines. The evaluation results, including performance metrics
and metadata, are logged and saved to a JSON file.

It supports different model types:
- LSTM (deep learning models)
- SVM-ANFIS (optimized with PSO)
- Common classifiers (e.g., Random Forest, SvmW, etc.)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import logging
import joblib
import datetime
from sklearn.metrics import classification_report, roc_curve, auc, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from src.evaluation.models.lstm import AttentionLayer 

from src.config import SUBJECT_LIST_PATH, PROCESS_CSV_PATH, MODEL_PKL_PATH
from src.utils.io.loaders import read_subject_list, read_subject_list_fold, get_model_type, load_subject_csvs
from src.utils.io.split import data_split, data_split_by_subject
from src.models.feature_selection.index import calculate_feature_indices
from src.evaluation.models.lstm import lstm_eval
from src.evaluation.models.SvmA import SvmA_eval
from src.evaluation.models.common import common_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def eval_pipeline(
    model: str, 
    tag: str = None, 
    sample_size: int = None, 
    seed: int = 42, 
    fold: int = 0, 
    subject_wise_split: bool = False
    ) -> None:
    """
    Evaluate the specified trained model using the appropriate method.

    This function orchestrates the evaluation process for a given model.
    It loads the necessary data and model, applies any required preprocessing
    (such as subject-wise splitting), and then calls the relevant evaluation
    function based on the model type. Finally, it saves the evaluation results.

    Parameters
    ----------
    model : str
        The name of the model to evaluate (e.g., ``'Lstm'``, ``'SvmA'``, ``'RF'``, ``'SvmW'``).
    tag : str, optional
        Optional tag to identify specific model variants or experiment runs.
    sample_size : int, optional
        Number of subjects to sample for evaluation. If ``None``, use all subjects.
    seed : int, default=42
        Random seed for reproducibility of subject sampling.
    fold : int, default=0
        Fold index for k-fold cross-validation. ``0`` means no specific fold is used.
    subject_wise_split : bool, default=False
        If True, data splitting ensures subjects are not mixed across train/test sets.

    Returns
    -------
    None
        Evaluation metrics are saved to a JSON file in the results directory.

    Notes
    -----
    - Supports the following model types:
      * Lstm (deep learning with attention)
      * SvmA (SVM with ANFIS feature selection)
      * Classical ML models (e.g., RF, SvmW, etc.)
    """

    logging.info(f"Evaluation split mode: {'subject-wise' if subject_wise_split else 'sample-wise'}")

    # Load subject list and determine model type
    if fold == 0:
        subject_list = read_subject_list()
    else:
        subject_list = read_subject_list_fold(fold)

    if sample_size is not None:
        rng = np.random.default_rng(seed)
        if sample_size > len(subject_list):
            logging.error(f"Sample size ({sample_size}) is larger than available subject count ({len(subject_list)}).")
            return
        subject_list = rng.choice(subject_list, size=sample_size, replace=False).tolist()
        logging.info(f"Evaluating on {sample_size} subjects: {subject_list}")

    model_type = get_model_type(model)

    # Load preprocessed feature data
    #combined_data = load_subject_csvs(subject_list, model_type, add_subject_id=True if subject_wise_split else False)
    combined_data, feature_columns = load_subject_csvs(
        subject_list, model_type, add_subject_id=True if subject_wise_split else False
    )
    
    if subject_wise_split:
        X_train, X_val, X_test, y_train, y_val, y_test = data_split_by_subject(combined_data, subject_list, seed)
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = data_split(combined_data)

        # --- fallback for empty validation/test (to allow evaluation run) ---
        if (X_test is None or len(X_test) == 0) and (X_train is not None and len(X_train) > 0):
            logging.warning("Test split is empty. Falling back to using the entire dataset as test.")
            X_test, y_test = X_train, y_train
            X_train, y_train = None, None
            X_val, y_val = None, None

    if y_test.nunique() < 2:
        logging.warning(f"Test labels are not binary or contain only one class. Found: {y_test.value_counts().to_dict()}")

    if X_test.shape[0] == 0:
        logging.error("X_test is empty. Check your subject/sample configuration or preprocessing.")
        return

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

    # Call evaluation functions based on model type
    if model == 'Lstm':
        scaler_path = os.path.join(MODEL_PKL_PATH, model_type, f"scaler_fold1{f'_{tag}' if tag else ''}.pkl")
        if not os.path.exists(scaler_path):
            logging.error(f"Scaler file not found: {scaler_path}")
            return
        scaler = joblib.load(scaler_path)

        result = lstm_eval(X_test, y_test, model_type, clf, scaler)

    elif model == 'SvmA':
        feature_file = f"selected_features_train_{model}{f'_{tag}' if tag else ''}.pkl"
        feature_path = os.path.join(MODEL_PKL_PATH, model_type, feature_file)
        if not os.path.exists(feature_path):
            logging.error(f"Feature file not found: {feature_path}")
            return
        with open(feature_path, "rb") as ff:
            selected_features = pickle.load(ff)
        result = SvmA_eval(X_test, y_test, model, clf, selected_features)

    else:
        feature_file = f"selected_features_train_{model}{f'_{tag}' if tag else ''}.pkl"
        feature_path = os.path.join(MODEL_PKL_PATH, model_type, feature_file)
        if not os.path.exists(feature_path):
            logging.error(f"Feature file not found: {feature_path}")
            return

        with open(feature_path, "rb") as ff:
            selected_features = pickle.load(ff)
        if not isinstance(selected_features, list):
            selected_features = list(selected_features)

        scaler_path = os.path.join(MODEL_PKL_PATH, model_type, f"scaler_{model}{f'_{tag}' if tag else ''}.pkl")
        if not os.path.exists(scaler_path):
            logging.error(f"Scaler file not found: {scaler_path}")
            return
        scaler = joblib.load(scaler_path)

        # Before feature selection, ensure columns are aligned and consistent
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
        X_test = X_test.loc[:, ~X_test.columns.duplicated()]

        # Drop unnecessary columns and ensure only numeric columns are used
        X_train = X_train.drop(columns=["subject_id"], errors='ignore')
        X_test = X_test.drop(columns=["subject_id"], errors='ignore')

        X_train = X_train.select_dtypes(include=[np.number])
        X_test = X_test.select_dtypes(include=[np.number])

        # Align both datasets to only common columns
        common_columns = X_train.columns.intersection(X_test.columns)
        X_train = X_train[common_columns]
        X_test = X_test[common_columns]

        # Check for missing features
        missing = [f for f in selected_features if f not in X_test.columns]
        if missing:
            logging.error(f"Missing features in X_test: {missing}")
            return

        X_train = X_train.loc[:, selected_features]
        X_test = X_test.loc[:, selected_features]
        X_test = scaler.transform(X_test)
        
        logging.info(f"X_train shape after feature alignment: {X_train.shape}")
        logging.info(f"X_test shape after feature alignment: {X_test.shape}")

        logging.info(f"Available columns in X_test: {list(X_test.columns) if hasattr(X_test, 'columns') else 'array'}")
        logging.info(f"Selected features: {selected_features}")

        result = common_eval(X_test, y_test, model, model_type, clf)

    # Add evaluation metadata
    result["sample_size"] = sample_size if sample_size is not None else len(subject_list)
    result["subject_list"] = subject_list
    
    # Add selected features if available
    if model in ["SvmA", "RF", "SvmW"]:  # Add other classical models here if needed
        result["selected_features"] = selected_features

    # After each evaluation call (e.g., result = common_eval(...))
    results_dir = os.path.join("results", model_type)
    os.makedirs(results_dir, exist_ok=True)

    # generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 

    # save results
    results_filename = f"metrics_{model}_{tag}_{timestamp}.json" if tag else f"metrics_{model}_{timestamp}.json"
    results_path = os.path.join(results_dir, results_filename)
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    logging.info(f"Saved evaluation metrics to {results_path}")
