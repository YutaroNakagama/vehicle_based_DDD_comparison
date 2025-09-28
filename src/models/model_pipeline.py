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
import json
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif  
from sklearn.preprocessing import StandardScaler

from src.config import SUBJECT_LIST_PATH, PROCESS_CSV_PATH, MODEL_PKL_PATH, TOP_K_FEATURES
from src.utils.io.loaders import read_subject_list, read_train_subject_list_fold, get_model_type, load_subject_csvs
from src.utils.io.split import data_split, data_split_by_subject, data_time_split_by_subject, time_stratified_three_way_split 
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
    Remove rows from a DataFrame where any specified column's Z-score exceeds the threshold.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    cols : list of str
        Column names to check for outliers.
    threshold : float, default=5.0
        Z-score threshold. Rows with Z-scores above this value are removed.

    Returns
    -------
    pandas.DataFrame
        DataFrame with outlier rows removed.
    """
    z = np.abs(zscore(df[cols], nan_policy='omit'))
    mask = (z < threshold).all(axis=1)
    return df[mask]

def save_feature_histograms(df, feature_columns, outdir="feature_hist_svg"):
    """
    Generate and save histograms for the specified feature columns as SVG files.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing feature data.
    feature_columns : list of str
        List of column names for which to generate histograms.
    outdir : str, default="feature_hist_svg"
        Output directory to save the histogram SVG files.

    Returns
    -------
    None
        SVG files are saved to the specified directory.
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

def _prepare_df_with_label_and_features(df: pd.DataFrame):
    """
    Filter by KSS labels, add a binary ``label`` column, and return feature column names.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing KSS labels and feature columns.

    Returns
    -------
    tuple
        A tuple containing:

        - ``df`` : pandas.DataFrame
            Filtered DataFrame with an added ``label`` column.
        - ``feature_columns`` : list of str
            List of feature column names including ``subject_id`` if present.
    """
    from src.config import KSS_BIN_LABELS, KSS_LABEL_MAP
    df = df[df["KSS_Theta_Alpha_Beta"].isin(KSS_BIN_LABELS)].copy()
    df["label"] = df["KSS_Theta_Alpha_Beta"].replace(KSS_LABEL_MAP)
    start_col = "Steering_Range"
    end_col = "LaneOffset_AAA"
    feature_columns = df.loc[:, start_col:end_col].columns.tolist()
    if "subject_id" in df.columns:
        feature_columns.append("subject_id")
    return df, feature_columns

def _log_split_ratios(y_tr: pd.Series, y_va: pd.Series, y_te: pd.Series, tag: str = ""):
    """
    Log the sample counts, positive counts, and positive ratios for train, validation, and test splits.

    Parameters
    ----------
    y_tr : pandas.Series
        Training labels.
    y_va : pandas.Series
        Validation labels.
    y_te : pandas.Series
        Test labels.
    tag : str, optional
        Optional tag to include in log messages.

    Returns
    -------
    None
        Logs summary statistics for each split.
    """
    def _r(y):
        n = int(y.shape[0])
        p = int(y.sum()) if n else 0
        r = p / n if n else float("nan")
        return n, p, r
    n_tr, p_tr, r_tr = _r(y_tr)
    n_va, p_va, r_va = _r(y_va)
    n_te, p_te, r_te = _r(y_te)
    n_all = n_tr + n_va + n_te
    p_all = p_tr + p_va + p_te
    r_all = (p_all / n_all) if n_all else float("nan")

    logging.info("[split%s] global  : n=%d, pos=%d, pos_ratio=%.3f", f":{tag}" if tag else "", n_all, p_all, r_all)
    logging.info("[split%s] train   : n=%d, pos=%d, pos_ratio=%.3f, |Δ|=%.3f", f":{tag}" if tag else "", n_tr, p_tr, r_tr, abs(r_tr - r_all) if n_tr else float("nan"))
    logging.info("[split%s] val     : n=%d, pos=%d, pos_ratio=%.3f, |Δ|=%.3f", f":{tag}" if tag else "", n_va, p_va, r_va, abs(r_va - r_all) if n_va else float("nan"))
    logging.info("[split%s] test    : n=%d, pos=%d, pos_ratio=%.3f, |Δ|=%.3f", f":{tag}" if tag else "", n_te, p_te, r_te, abs(r_te - r_all) if n_te else float("nan"))

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
    save_pretrain: str = None,
    balance_labels: bool = False,
    balance_method: str = "undersample",
    time_stratify_labels: bool = False,
    time_stratify_tolerance: float = 0.02,
    time_stratify_window: float = 0.10,
    time_stratify_min_chunk: int = 100,
    mode: str = None,
) -> None:
    """
    Train a machine learning model for driver drowsiness detection.

    This function loads processed feature data, applies optional domain
    generalization techniques (Domain Mixup, CORAL, VAE), performs feature
    selection, and trains a model based on the specified configuration.

    Parameters
    ----------
    model_name : str
        The model to train (e.g., ``'Lstm'``, ``'SvmA'``, ``'RF'``).
    use_domain_mixup : bool, default=False
        Apply domain mixup augmentation.
    use_coral : bool, default=False
        Apply CORAL (Correlation Alignment) domain adaptation.
    use_vae : bool, default=False
        Apply VAE-based augmentation.
    sample_size : int, optional
        Number of subjects to subsample. If ``None``, use all subjects.
    seed : int, default=42
        Random seed for reproducibility.
    fold : int, optional
        Fold index for cross-validation.
    n_folds : int, optional
        Total number of folds for cross-validation.
    tag : str, optional
        Identifier appended to saved model artifacts.
    subject_wise_split : bool, default=False
        If True, ensures subjects are not mixed across splits.
    feature_selection_method : {"rf", "mi", "anova"}, default="rf"
        Feature selection method:
        - ``"rf"`` : Random Forest importance
        - ``"mi"`` : Mutual Information
        - ``"anova"`` : ANOVA F-test
    data_leak : bool, default=False
        Allow intentional data leakage (for ablation studies).
    subject_split_strategy : str, default="random"
        Strategy for subject splitting (e.g., ``"random"``, ``"finetune_target_subjects"``).
    target_subjects : list of str, optional
        List of subjects to treat as targets.
    train_subjects : list of str, optional
        Explicit subject list for training.
    val_subjects : list of str, optional
        Explicit subject list for validation.
    test_subjects : list of str, optional
        Explicit subject list for testing.
    general_subjects : list of str, optional
        List of general subjects for pretraining or domain generalization.
    finetune_setting : str, optional
        Path to pretrained feature/scaler configuration for fine-tuning.
    save_pretrain : str, optional
        Path to save pretraining artifacts (features/scaler).
    balance_labels : bool, default=False
        Whether to balance class distribution.
    balance_method : str, default="undersample"
        Strategy for label balancing.
    time_stratify_labels : bool, default=False
        If True, apply time-stratified label splitting.
    time_stratify_tolerance : float, default=0.02
        Allowed tolerance for time-based stratification.
    time_stratify_window : float, default=0.10
        Window proportion for time-based stratification.
    time_stratify_min_chunk : int, default=100
        Minimum chunk size for time-based stratification.

    Returns
    -------
    NOTE:
    -----
    - Evaluation is no longer performed here.
    - Use `eval_pipeline` for evaluation.
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

        # Use all other subjects for the main training set
        if general_subjects:
            general_subjects_list = general_subjects
        else:
            general_subjects_list = [s for s in subject_list if s not in target_subjects]

        # --- (helper) Build identical target splits for both FineTune and EvalOnly ---
        def _build_target_splits_df(dataframe):
            """
            Build consistent target splits for both fine-tuning and evaluation-only modes.
            
            Validation and test splits are created identically across modes to ensure
            fair comparison. In evaluation-only mode, the training split is ignored.
            
            Parameters
            ----------
            dataframe : pandas.DataFrame
                Input DataFrame containing subject, timestamp, and label columns.
            
            Returns
            -------
            tuple
                A tuple containing:
            
                - ``X_tr`` : pandas.DataFrame
                    Training feature matrix.
                - ``X_va`` : pandas.DataFrame
                    Validation feature matrix.
                - ``X_te`` : pandas.DataFrame
                    Test feature matrix.
                - ``y_tr`` : pandas.Series
                    Training labels.
                - ``y_va`` : pandas.Series
                    Validation labels.
                - ``y_te`` : pandas.Series
                    Test labels.
            """
            if time_stratify_labels:
                df_lab, feature_columns = _prepare_df_with_label_and_features(dataframe)
                sort_keys = ("subject_id", "Timestamp")
                idx_tr, idx_va, idx_te = time_stratified_three_way_split(
                    df_lab, label_col="label", sort_keys=sort_keys,
                    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                    tolerance=time_stratify_tolerance, window_prop=time_stratify_window,
                    min_chunk=time_stratify_min_chunk,
                )
                X_tr = df_lab.loc[idx_tr, feature_columns]
                X_va = df_lab.loc[idx_va, feature_columns]
                X_te = df_lab.loc[idx_te, feature_columns]
                y_tr = df_lab.loc[idx_tr, "label"]
                y_va = df_lab.loc[idx_va, "label"]
                y_te = df_lab.loc[idx_te, "label"]
            else:
                X_tr, X_va, X_te, y_tr, y_va, y_te = data_time_split_by_subject(
                    dataframe, subject_col="subject_id", time_col="Timestamp"
                )
            # Drop subject_id only at the very end so both paths are identical
            X_tr = X_tr.drop(columns=["subject_id"], errors="ignore")
            X_va = X_va.drop(columns=["subject_id"], errors="ignore")
            X_te = X_te.drop(columns=["subject_id"], errors="ignore")
            return X_tr, X_va, X_te, y_tr, y_va, y_te

        # Load general-subjects data once (used for pretrain artifacts and/or pretrain model)
        general_data, _ = load_subject_csvs(general_subjects_list, model_type, add_subject_id=True)

        # --- Step 1: Save pretrain artifacts (features/scaler) on general_subjects ---
        if save_pretrain:
            if not os.path.dirname(save_pretrain):
                save_pretrain = os.path.join("model/common", save_pretrain)

            # 2. random separation
            X_gtrain, _, _, y_gtrain, _, _ = data_split(general_data, random_state=seed)
            X_gtrain_for_fs = X_gtrain.drop(columns=["subject_id"], errors='ignore')
    
            # 3. Feature Selection (e.g. RF importance)
            selected_features = select_top_features_by_importance(X_gtrain_for_fs, y_gtrain, top_k=TOP_K_FEATURES)
    
            # 4. Scaler
            scaler = StandardScaler()
            scaler.fit(X_gtrain_for_fs[selected_features])
    
            # 6. Save
            pretrain_dict = {
                "selected_features": selected_features,
                "scaler": scaler,
            }

            try:
                n_scale = getattr(scaler, "n_features_in_", None)
                if n_scale is not None and n_scale != len(selected_features):
                    logging.warning("[EvalOnly] scaler.n_features_in_=%d != len(selected_features)=%d",
                                    n_scale, len(selected_features))
            except Exception:
                pass

            with open(save_pretrain, "wb") as f:
                pickle.dump(pretrain_dict, f)
            logging.info(f"Saved pretrain setting (features/scaler) to {save_pretrain}")

        # --- Step 2-B: (Default) Fine-tuning path using finetune_setting ---
        if finetune_setting:
            if not os.path.dirname(finetune_setting):
                finetune_setting = os.path.join("model/common", finetune_setting)
            with open(finetune_setting, "rb") as f:
                pretrain_dict = pickle.load(f)
            selected_features = pretrain_dict["selected_features"]
            scaler = pretrain_dict["scaler"]
            logging.info(f"Loaded pretrain setting from {finetune_setting}: features={selected_features}")

            # Use the SAME split builder so eval-only and fine-tune share val/test
            data, _ = load_subject_csvs(target_subjects, model_type, add_subject_id=True)
            X_train, X_val, X_test, y_train, y_val, y_test = _build_target_splits_df(data)

            _log_split_ratios(y_train, y_val, y_test, tag="finetune_setting")

            # --- Feature selection and scaling ---

            clf = get_classifier(model_name)
            suffix = ""
            if tag:
                suffix += f"_{tag}_finetune"
            elif target_subjects:
                safe_targets = "-".join(map(str, target_subjects))[:40]
                suffix += f"_targets-{safe_targets}_finetune"

            common_train(
                X_train, X_val, X_test, y_train, y_val, y_test,
                selected_features,
                model_name, model_type, clf,
                scaler=scaler, suffix=suffix, data_leak=data_leak,
            )
            return  
        
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

        if time_stratify_labels:
            df_lab, feature_columns = _prepare_df_with_label_and_features(data)
            sort_keys = ("subject_id", "Timestamp") # Adjust according to actual dataset column names 
            idx_tr, idx_va, idx_te = time_stratified_three_way_split(
                df_lab,
                label_col="label",
                sort_keys=sort_keys,
                train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                tolerance=time_stratify_tolerance,
                window_prop=time_stratify_window,
                min_chunk=time_stratify_min_chunk,
            )
            # Exclude label column from features
            X_train = df_lab.loc[idx_tr, feature_columns].drop(columns=["subject_id"], errors="ignore")
            X_val   = df_lab.loc[idx_va, feature_columns].drop(columns=["subject_id"], errors="ignore")
            X_test  = df_lab.loc[idx_te, feature_columns].drop(columns=["subject_id"], errors="ignore")
            y_train = df_lab.loc[idx_tr, "label"]
            y_val   = df_lab.loc[idx_va, "label"]
            y_test  = df_lab.loc[idx_te, "label"]
        else:
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

    _log_split_ratios(y_train, y_val, y_test, tag=f"{subject_split_strategy}|time_stratify={time_stratify_labels}")

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
        lstm_train(X_train, y_train, model_name)
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

        X_test_for_fs = X_test.drop(columns=["subject_id"], errors='ignore')
        X_test_for_fs = X_test_for_fs.reset_index(drop=True)

        # 9. Get Classifier and Train
        clf = get_classifier(model_name)
        model_key = model_name  
        
        # Construct suffix for model saving based on applied techniques
        suffix = ""
        if mode:
            suffix += f"_{mode}"
        if tag:
            suffix += f"_{tag}"
        elif target_subjects:
            safe_targets = "-".join(map(str, target_subjects))[:40]
            suffix += f"_targets-{safe_targets}"
        elif os.getenv("PBS_ARRAY_INDEX"):
            suffix += f"_group{os.getenv('PBS_ARRAY_INDEX')}"
        
        if use_coral:
            suffix += "_coral"
        if use_domain_mixup:
            suffix += "_mixup"
        if use_vae:
            suffix += "_vae"

        best_clf, scaler, best_threshold, feature_meta, results = common_train(
            X_train_for_fs, X_val_for_fs, X_test_for_fs,
            y_train, y_val, y_test,
            selected_features,
            model_name, model_type, clf,
            scaler=scaler,
            suffix=suffix,
            data_leak=data_leak,
        )

        # ===== Save artifacts =====
        out_model_dir = f"models/{model_type}"
        out_result_dir = f"results/{model_type}/{model_key}"
        os.makedirs(out_model_dir, exist_ok=True)
        os.makedirs(out_result_dir, exist_ok=True)

        # Save model and scaler
        with open(f"{out_model_dir}/{model_key}{suffix}.pkl", "wb") as f:
            pickle.dump(best_clf, f)
        with open(f"{out_model_dir}/scaler_{model_key}{suffix}.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open(f"{out_model_dir}/feature_meta_{model_key}{suffix}.json", "w") as f:
            json.dump(feature_meta, f, indent=2)
        
        # Save selected features
        with open(f"{out_model_dir}/selected_features_{model_key}{suffix}.pkl", "wb") as f:
            pickle.dump(selected_features, f)
        
        # Save threshold
        if best_threshold is not None:
            thr_meta = {
                "model": model_key,
                "threshold": float(best_threshold),
                "metric": "F1-optimal",
            }
            with open(f"{out_result_dir}/threshold_{model_key}{suffix}.json", "w") as f:
                json.dump(thr_meta, f, indent=2)

        # Save metrics (training-time only)
        if results:
            rows = []
            for split, m in results.items():
                rows.append({
                    "phase": "training",   
                    "split": split,
                    **m
                })

            df_results = pd.DataFrame(rows)
            df_results.to_csv(f"{out_result_dir}/trainmetrics_{model_key}{suffix}.csv", index=False)

            with open(f"{out_result_dir}/trainmetrics_{model_key}{suffix}.json", "w") as f:
                json.dump(rows, f, indent=2)

            logging.info(f"Training-time metrics saved to "
                         f"{out_result_dir}/trainmetrics_{model_key}{suffix}.csv/json")

        logging.info(f"Artifacts saved under {out_model_dir} and {out_result_dir}")

    logging.info(f"[DONE] Training complete for model={model_name}, tag={tag}")
