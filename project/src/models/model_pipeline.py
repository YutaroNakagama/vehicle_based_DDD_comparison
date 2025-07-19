"""
Model Training Pipeline for Driver Drowsiness Detection.

This module orchestrates the entire model training workflow for the Driver Drowsiness Detection (DDD) system.
It provides the `train_pipeline` function, which serves as the central entry point for:

- Loading preprocessed data for the selected model type.
- Splitting data into training, validation, and test sets, with options for subject-wise splitting and k-fold cross-validation.
- Applying optional domain generalization techniques such as Domain Mixup, CORAL (Correlation Alignment), and VAE-based augmentation.
- Performing feature selection using various methods (e.g., Random Forest importance, Mutual Information, ANOVA F-test).
- Training model-specific architectures (e.g., LSTM, SVM, various tree-based classifiers) based on the configuration.
- Saving trained models and relevant artifacts.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, mean_squared_error
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif  
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from pyswarm import pso

from src.config import SUBJECT_LIST_PATH, PROCESS_CSV_PATH, MODEL_PKL_PATH, TOP_K_FEATURES
from src.utils.io.loaders import read_subject_list, read_train_subject_list_fold, get_model_type, load_subject_csvs
from src.utils.io.split import data_split, data_split_by_subject, data_time_split_by_subject 
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

def remove_outliers_zscore(df, cols, threshold=5.0):
    """
    Removes rows from a DataFrame where any specified column's value has a Z-score
    exceeding a given threshold. This is used for outlier removal.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cols (list): A list of column names to check for outliers.
        threshold (float): The Z-score threshold. Rows with Z-scores above this
                           threshold in any of the specified columns will be removed.

    Returns:
        pd.DataFrame: The DataFrame with outliers removed.
    """
    z = np.abs(zscore(df[cols], nan_policy='omit'))
    mask = (z < threshold).all(axis=1)
    return df[mask]

def save_feature_histograms(df, feature_columns, outdir="feature_hist_svg"):
    """
    Generates and saves histograms for specified feature columns as SVG files.
    Each histogram is saved as a separate SVG file in the specified output directory.

    Args:
        df (pd.DataFrame): DataFrame containing the features.
        feature_columns (list): A list of column names (features) for which to generate histograms.
        outdir (str): The directory where the SVG histogram files will be saved.
                      Defaults to "feature_hist_svg".
    """
    os.makedirs(outdir, exist_ok=True)
    for col in feature_columns:
        plt.figure(figsize=(6, 4))
        df[col].hist(bins=50)
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{col}.svg"), format="svg")
        plt.close()


def train_pipeline(
    model_name: str,
    use_domain_mixup: bool = False,
    use_coral: bool = False,
    use_vae: bool = False,
    sample_size: int = None,
    seed: int = 42,
    fold: int = None,
    n_folds: int = None,
    tag: str = None, 
    subject_wise_split: bool = False, 
    feature_selection_method: str = "rf", 
    data_leak: bool = False,
    subject_split_strategy: str = "random",
    target_subjects: list = [],
    train_subjects: list = [],
    val_subjects: list = [],
    test_subjects: list = [],
    general_subjects: list = [],
    finetune_setting=None,   
) -> None:
    """Train a machine learning model for drowsiness detection.

    This function loads the processed feature data, applies optional
    domain generalization methods (Domain Mixup, CORAL, VAE), performs
    feature selection, and dispatches training to model-specific training
    functions (LSTM, SVM, etc.).

    Args:
        model (str): The name of the model to train (e.g., 'Lstm', 'SvmA', 'RF').
        use_domain_mixup (bool): If True, apply domain mixup augmentation to the training data.
        use_coral (bool): If True, apply CORAL (Correlation Alignment) for domain adaptation.
        use_vae (bool): If True, apply VAE-based data augmentation to the training data.
        sample_size (int, optional): If specified, subsamples the given number of subjects.
        seed (int): Random seed for reproducibility of data splitting and sampling. Defaults to 42.
        fold (int, optional): The current fold number if performing k-fold cross-validation.
        n_folds (int, optional): The total number of folds for cross-validation.
        tag (str, optional): An optional tag to append to the saved model filename for identification.
        subject_wise_split (bool): If True, ensures that subjects are not split across train/validation/test sets.
        feature_selection_method (str): The method to use for feature selection. Options include
                                        'rf' (Random Forest importance), 'mi' (Mutual Information),
                                        and 'anova' (ANOVA F-test). Defaults to "rf".
        data_leak (bool): If True, intentionally allows data leakage during feature selection or scaling
                          (e.g., fitting scaler on both train and validation data). Use for ablation studies.
                          Defaults to False.

    Returns:
        None
    """

    # 1. Load subject list
    subject_list = read_subject_list()

    # 2. Subsample subjects if sample_size is specified
    if sample_size is not None:
        rng = np.random.default_rng(seed)
        subject_list = rng.choice(subject_list, size=sample_size, replace=False).tolist()
        logging.info(f"Using {sample_size} subjects: {subject_list}")

    model_type = get_model_type(model_name)
    logging.info(f"Model type: {model_type}")

    # 3. Data Splitting based on strategy
    if subject_split_strategy == "single_subject_data_split":
        if not target_subjects or len(target_subjects) != 1:
            logging.error("`single_subject_data_split` strategy requires exactly one subject in `--target_subjects`.")
            return
        
        logging.info(f"Performing single subject data split for: {target_subjects[0]}")
        data, _ = load_subject_csvs(target_subjects, model_type, add_subject_id=True)
        X_train, X_val, X_test, y_train, y_val, y_test = data_split(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=seed)

    elif subject_split_strategy == "isolate_target_subjects":
        if not target_subjects:
            logging.error("`isolate_target_subjects` strategy requires `--target_subjects` to be set.")
            return
        
        logging.info(f"Isolating target subjects: {target_subjects}")
        # Split target subjects: 80% train, 10% val, 10% test
        train_subjects, temp_subjects = train_test_split(target_subjects, test_size=0.2, random_state=seed)
        val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=seed)
        
        use_subjects = target_subjects
        data, _ = load_subject_csvs(use_subjects, model_type, add_subject_id=True)
        X_train, X_val, X_test, y_train, y_val, y_test = data_split_by_subject(
            data, train_subjects, seed, val_subjects=val_subjects, test_subjects=test_subjects
        )

    elif subject_split_strategy == "finetune_target_subjects":
        if not target_subjects:
            logging.error("`finetune_target_subjects` strategy requires `--target_subjects` to be set.")
            return

        logging.info(f"Finetuning with target subjects: {target_subjects}")
        
        # Split target subjects for validation and testing (if more than one target subject)
        if len(target_subjects) > 1:
            target_train_subjects, temp_subjects = train_test_split(target_subjects, test_size=0.2, random_state=seed)
            val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=seed)
        else: # Only one target subject, use data_split for within-subject split
            target_train_subjects = target_subjects # The single subject is the 'target_train_subject'
            # Load data for the single target subject to split it
            single_subject_data, _ = load_subject_csvs(target_subjects, model_type, add_subject_id=True)
            X_single_train, X_single_val, X_single_test, y_single_train, y_single_val, y_single_test = data_split(single_subject_data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=seed)
            # These will be used later to construct the final X_val, y_val, X_test, y_test

            val_subjects = [] # No separate validation subject list
            test_subjects = [] # No separate test subject list

        # Use all other subjects for the main training set
        if general_subjects:
            general_subjects_list = general_subjects
        else:
            general_subjects_list = [s for s in subject_list if s not in target_subjects]
        
        # Load all data (general subjects + target subjects)
        use_subjects = list(set(general_subjects_list + target_subjects)) # Ensure unique subjects
        data, _ = load_subject_csvs(use_subjects, model_type, add_subject_id=True)

        if len(target_subjects) > 1:
            train_subjects = general_subjects_list + target_train_subjects
            X_train, X_val, X_test, y_train, y_val, y_test = data_split_by_subject(
                data, train_subjects, seed, val_subjects=val_subjects, test_subjects=test_subjects
            )
        else: # Single target subject case
            # Combine general subjects data with the single target subject's training data
            X_general_train, _, _, y_general_train, _, _ = data_split_by_subject(
                data, general_subjects_list, seed, val_subjects=[], test_subjects=[]
            )
            X_train = pd.concat([X_general_train, X_single_train], ignore_index=True)
            y_train = pd.concat([y_general_train, y_single_train], ignore_index=True)
            X_val = X_single_val
            y_val = y_single_val
            X_test = X_single_test
            y_test = y_single_test

            logging.info(f"X_train shape after finetune_target_subjects (single subject): {X_train.shape}")
            logging.info(f"X_val   shape after finetune_target_subjects (single subject): {X_val.shape}")
            logging.info(f"X_test  shape after finetune_target_subjects (single subject): {X_test.shape}")

    elif subject_split_strategy == "subject_time_split":
        if target_subjects:
            data, _ = load_subject_csvs(target_subjects, model_type, add_subject_id=True)
        else:
            data, _ = load_subject_csvs(subject_list, model_type, add_subject_id=True)
        X_train, X_val, X_test, y_train, y_val, y_test = data_time_split_by_subject(
            data, subject_col="subject_id", time_col="Timestamp"
        )

    elif subject_wise_split and fold and fold > 0:
        # Existing logic for cross-validation
        n_splits = n_folds
        subject_array = np.array(subject_list)
        gkf = GroupKFold(n_splits=n_splits)

        splits = list(gkf.split(subject_array, groups=subject_array))
        train_idx, test_idx = splits[fold - 1]
        train_subjects = subject_array[train_idx].tolist()
        test_subjects = subject_array[test_idx].tolist()

        rng = np.random.default_rng(seed)
        n_train = int(len(train_subjects) * 0.9)
        shuffled = rng.permutation(train_subjects)
        actual_train_subjects = shuffled[:n_train].tolist()
        val_subjects = shuffled[n_train:].tolist()

        use_subjects = actual_train_subjects + val_subjects + test_subjects
        data, _ = load_subject_csvs(use_subjects, model_type, add_subject_id=True)
        X_train, X_val, X_test, y_train, y_val, y_test = data_split_by_subject(
            data,
            actual_train_subjects,
            seed,
            val_subjects=val_subjects,
            test_subjects=test_subjects
        )
        logging.info(f"X_train shape after subject-wise data split: {X_train.shape}")
        logging.info(f"X_val   shape after subject-wise data split: {X_val.shape}")
        logging.info(f"X_test  shape after subject-wise data split: {X_test.shape}")
    else:
        # Default random split
        data, feature_columns = load_subject_csvs(subject_list, model_type, add_subject_id=True)
        X_train, X_val, X_test, y_train, y_val, y_test = data_split(data, random_state=seed)
        logging.info(f"X_train shape after random data split: {X_train.shape}")
        logging.info(f"X_val   shape after random data split: {X_val.shape}")
        logging.info(f"X_test  shape after random data split: {X_test.shape}")

    # 4. Data Validation: Check for empty splits or non-binary labels
    if y_train.nunique() < 2:
        logging.error(f"Training labels are not binary. Found: {y_train.value_counts().to_dict()}")
        return
    
    # Check for empty splits
    if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
        logging.error("Train/val/test split has empty set. Try increasing sample_size or reviewing KSS filtering.")
        return

    # 5. Domain Generalization Techniques
    if use_domain_mixup:
        domain_labels_train = generate_domain_labels(subject_list, X_train)
        X_train_aug, y_train_aug = domain_mixup(X_train, y_train, domain_labels_train)
        X_train, y_train = X_train_aug, y_train_aug
        logging.info(f"Applied Domain Mixup. New X_train shape: {X_train.shape}")

    # Model-specific training dispatch
    if model_name == 'Lstm':
        lstm_train(X_train, y_train, model)
        logging.info("LSTM model training initiated.")

    elif model_name == 'SvmA':
        X_train_for_fs = X_train.drop(columns=["subject_id"], errors='ignore')
        X_val_for_fs = X_val.drop(columns=["subject_id"], errors='ignore')
        X_train_for_fs = X_train_for_fs.select_dtypes(include=[np.number])
        X_val_for_fs = X_val_for_fs.select_dtypes(include=[np.number])
    
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        feature_indices = calculate_feature_indices(X_train_for_fs, y_train)
        SvmA_train(X_train_for_fs, X_val_for_fs, y_train, y_val, feature_indices, model_name)
        logging.info("SvmA model training initiated with internal feature selection.")

    else:
        X_train_for_fs = X_train.drop(columns=["subject_id"], errors='ignore')
        X_val_for_fs = X_val.drop(columns=["subject_id"], errors='ignore')

        X_train_for_fs = X_train_for_fs.reset_index(drop=True)
        X_val_for_fs = X_val_for_fs.reset_index(drop=True)

        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        logging.info(f"y_train unique: {y_train.unique()}, counts: {y_train.value_counts().to_dict()}")

        # 7. Feature Selection
        selected_features = []
        if feature_selection_method == "mi":
            # Mutual Information based feature selection
            selector = SelectKBest(score_func=mutual_info_classif, k=TOP_K_FEATURES)
            selector.fit(X_train_for_fs, y_train)
            selected_mask = selector.get_support()
            selected_features = X_train_for_fs.columns[selected_mask].tolist()
            logging.info(f"Selected features (mutual_info): {selected_features}")

        elif feature_selection_method == "anova":  
            # ANOVA F-test based feature selection
            selector = SelectKBest(score_func=f_classif, k=TOP_K_FEATURES)
            if data_leak == True:
                # Fit selector on combined train and validation data if data_leak is True
                selector.fit(pd.concat([X_train_for_fs, X_val_for_fs]), pd.concat([y_train, y_val]))  
                selected_mask = selector.get_support()
                selected_features = pd.concat([X_train_for_fs, X_val_for_fs]).columns[selected_mask].tolist()
            else:
                # Fit selector only on training data
                selector.fit(X_train_for_fs, y_train)
                selected_mask = selector.get_support()
                selected_features = X_train_for_fs.columns[selected_mask].tolist()
            logging.info(f"Selected features (ANOVA F-test): {selected_features}")
#            data_clean = remove_outliers_zscore(data, selected_features, threshold=5.0)
#            save_feature_histograms(data, selected_features, outdir="./data/log/")
        
        elif feature_selection_method == "rf":
            # Random Forest importance based feature selection
            selected_features = select_top_features_by_importance(X_train_for_fs, y_train, top_k=TOP_K_FEATURES)
            logging.info(f"Selected features (RF importance): {selected_features}")
        
        else:
            raise ValueError(f"Unknown feature_selection_method: {feature_selection_method}")
    
        # 8. Data Scaling
        if data_leak:
            # Fit scaler on combined train and validation data if data_leak is True
            scaler = StandardScaler()
            scaler.fit(pd.concat([X_train_for_fs[selected_features], X_val_for_fs[selected_features]]))
            logging.info("Scaler was fit using both X_train and X_val (data_leak=True).")
        else:
            # Fit scaler only on training data
            scaler = StandardScaler()
            scaler.fit(X_train_for_fs[selected_features])
            logging.info("Scaler was fit using only X_train (standard procedure).")

        X_train_scaled = scaler.transform(X_train_for_fs[selected_features])
        X_val_scaled = scaler.transform(X_val_for_fs[selected_features])

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=selected_features)

        # 9. Get Classifier and Train
        clf = get_classifier(model_name)

        # Construct suffix for model saving based on applied techniques
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
            X_train_scaled,    
            X_val_scaled,      
            y_train, y_val,
            selected_features, 
            model_name, model_type, clf,
            scaler=scaler,    
            suffix=suffix,
            data_leak=data_leak,
        )
