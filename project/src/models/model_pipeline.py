"""Model training pipeline for driver drowsiness detection.

This module defines the main function `train_pipeline()` which:
- Loads preprocessed data for selected model type.
- Splits data into training/validation/test sets.
- Applies optional domain generalization techniques (Domain Mixup, CORAL, VAE).
- Trains model-specific architectures (LSTM, SVM, RF, etc.).
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from pyswarm import pso

from src.config import SUBJECT_LIST_PATH, PROCESS_CSV_PATH, MODEL_PKL_PATH, TOP_K_FEATURES
from src.utils.io.loaders import read_subject_list, get_model_type, load_subject_csvs
from src.utils.io.split import data_split, data_split_by_subject  
from src.utils.domain_generalization.domain_mixup import generate_domain_labels, domain_mixup
from src.utils.domain_generalization.coral import coral
from src.utils.domain_generalization.vae_augment import vae_augmentation
from src.models.feature_selection.index import calculate_feature_indices
from src.models.feature_selection.anfis import calculate_id
from src.models.feature_selection.rf_importance import select_top_features_by_importance 
from src.models.architectures.helpers import get_classifier
from src.models.architectures.lstm import lstm_train
from src.models.architectures.SvmA import SvmA_train
from src.models.architectures.common import common_train

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_pipeline(
    model: str,
    use_domain_mixup: bool = False,
    use_coral: bool = False,
    use_vae: bool = False,
    sample_size: int = None,
    seed: int = 42,
    tag: str = None, 
    subject_wise_split: bool = False, 
    feature_selection_method: str = "rf"  
) -> None:
    """Train a machine learning model for drowsiness detection.

    This function loads the processed feature data, applies optional
    domain generalization methods (Domain Mixup, CORAL, VAE), and
    dispatches training to model-specific training functions (LSTM, SVM, etc.).

    Args:
        model (str): The name of the model to train ('Lstm', 'SvmA', 'RF', etc.).
        use_domain_mixup (bool): If True, apply domain mixup augmentation.
        use_coral (bool): If True, apply CORAL alignment between source and target domains.
        use_vae (bool): If True, apply VAE-based data augmentation.
        feature_selection_method (str): Feature selection method ('rf' or 'mi').

    Returns:
        None
    """
    subject_list = read_subject_list()

    if sample_size is not None:
        rng = np.random.default_rng(seed)
        subject_list = rng.choice(subject_list, size=sample_size, replace=False).tolist()
        logging.info(f"Using {sample_size} subjects: {subject_list}")

    model_type = get_model_type(model)
    logging.info(f"Model type: {model_type}")

    data = load_subject_csvs(subject_list, model_type, add_subject_id=True)

    if subject_wise_split:
        X_train, X_val, X_test, y_train, y_val, y_test = data_split_by_subject(data, subject_list, seed)
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = data_split(data)

    # Check for label diversity in training set
    if y_train.nunique() < 2:
        logging.error(f"Training labels are not binary. Found: {y_train.value_counts().to_dict()}")
        return
    
    # Check for empty splits
    if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
        logging.error("Train/val/test split has empty set. Try increasing sample_size or reviewing KSS filtering.")
        return

    if use_domain_mixup:
        domain_labels_train = generate_domain_labels(subject_list, X_train)
        X_train_aug, y_train_aug = domain_mixup(X_train, y_train, domain_labels_train)
        X_train, y_train = X_train_aug, y_train_aug

    if use_coral:
        # CORAL: align X_train to X_val (target domain)
        X_train_numeric = X_train.select_dtypes(include=[np.number]).values
        X_val_numeric = X_val.select_dtypes(include=[np.number]).values

        X_train_aligned = coral(X_train_numeric, X_val_numeric)

        # Replace original numeric columns
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train[numeric_cols] = X_train_aligned

    if use_vae:
        X_train = vae_augmentation(X_train, augment_ratio=0.3)

    # function call
    if model == 'Lstm':
        lstm_train(X_train, y_train, model)

    elif model == 'SvmA':
        X_train_for_fs = X_train.drop(columns=["subject_id"], errors='ignore')
        X_val_for_fs = X_val.drop(columns=["subject_id"], errors='ignore')
        X_train_for_fs = X_train_for_fs.select_dtypes(include=[np.number])
        X_val_for_fs = X_val_for_fs.select_dtypes(include=[np.number])
    
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        feature_indices = calculate_feature_indices(X_train_for_fs, y_train)
        SvmA_train(X_train_for_fs, X_val_for_fs, y_train, y_val, feature_indices, model)

    else:
        X_train_for_fs = X_train.drop(columns=["subject_id"], errors='ignore')
        X_val_for_fs = X_val.drop(columns=["subject_id"], errors='ignore')

        X_train_for_fs = X_train_for_fs.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)

        logging.info(f"y_train unique: {y_train.unique()}, counts: {y_train.value_counts().to_dict()}")

        if feature_selection_method == "mi":
            selector = SelectKBest(score_func=mutual_info_classif, k=TOP_K_FEATURES)
            selector.fit(X_train_for_fs, y_train)
            selected_mask = selector.get_support()
            selected_features = X_train_for_fs.columns[selected_mask].tolist()
            logging.info(f"Selected features (mutual_info): {selected_features}")
        
        elif feature_selection_method == "rf":
            selected_features = select_top_features_by_importance(X_train_for_fs, y_train, top_k=TOP_K_FEATURES)
            logging.info(f"Selected features (RF importance): {selected_features}")
        
        else:
            raise ValueError(f"Unknown feature_selection_method: {feature_selection_method}")
    
        clf = get_classifier(model)

        suffix = ""
        if tag:
            suffix += f"_{tag}"
        if use_coral:
            suffix += "_coral"
        if use_domain_mixup:
            suffix += "_mixup"
        if use_vae:
            suffix += "_vae"

        common_train(
            X_train_for_fs[selected_features],
            X_val_for_fs[selected_features],
            y_train, y_val,
            selected_features, 
            model, model_type, clf,
            suffix=suffix
        )
    
