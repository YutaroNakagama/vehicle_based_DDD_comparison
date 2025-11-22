"""Model Training Pipeline for Driver Drowsiness Detection (DDD).

This module orchestrates the model training workflow (data loading, splitting,
feature selection, model training, artifact saving). It exposes :func:`train_pipeline`
to train models for drowsiness detection.

Notes
-----
- Delegates all major stages to focused helper modules for testability.
- Supports multiple experiment modes: pooled, target_only, source_only, joint_train.
- Compatible with HPC job submission via scripts/python/train.py.

Functions
---------
train_pipeline(model_name, mode, ...) -> None
    Train a model with specified configuration and save artifacts.
"""

import logging
from typing import List, Optional

from src.config import TOP_K_FEATURES
from src.models.train_stages import (
    prepare_suffix_with_jobid,
    load_and_filter_data,
    prepare_source_only_splits,
)
from src.utils.io.split_helpers import split_data, log_split_ratios
from src.utils.io.feature_utils import normalize_feature_names
from src.utils.io.preprocessing import clean_feature_dataframe, align_train_val_test_columns
from src.utils.io.savers import save_artifacts
from src.models.feature_selection.feature_helpers import select_features_and_scale
from src.models.architectures.train_helpers import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_pipeline(
    model_name: str,
    mode: Optional[str] = None,
    subject_split_strategy: str = "random",
    target_subjects: Optional[List[str]] = None,
    subject_wise_split: bool = False,
    seed: int = 42,
    tag: Optional[str] = None,
    time_stratify_labels: bool = False,
    time_stratify_tolerance: float = 0.02,
    time_stratify_window: float = 0.10,
    time_stratify_min_chunk: int = 100,
    use_oversampling: bool = False,
    oversample_method: str = "smote",
    *,
    feature_selection_method: str = "rf",
    data_leak: bool = False,
) -> None:
    """Train a model for driver drowsiness detection.

    Parameters
    ----------
    model_name : {"RF", "SvmA", "Lstm"}
        Model architecture to train.
    mode : {"pooled", "target_only", "source_only", "joint_train"}, optional
        Experimental mode for domain generalization.
    subject_split_strategy : {"random", "subject_time_split"}, default="random"
        Strategy for splitting subjects into train/val/test.
    target_subjects : list of str, optional
        Target subject IDs (used in target_only/source_only modes).
    subject_wise_split : bool, default=False
        Kept for CLI compatibility; unused.
    seed : int, default=42
        Random seed for reproducibility.
    tag : str, optional
        Experiment tag (e.g., "rank_dtw_mean_high").
    time_stratify_labels : bool, default=False
        Enable time-stratified splitting with class ratio tolerance.
    time_stratify_tolerance : float, default=0.02
        Allowed deviation in positive class ratio per split.
    time_stratify_window : float, default=0.10
        Window (fraction of N) for boundary search.
    time_stratify_min_chunk : int, default=100
        Minimum rows per split.
    use_oversampling : bool, default=False
        Apply oversampling to minority class in training data.
    oversample_method : {"smote", "adasyn", "borderline"}, default="smote"
        Oversampling method to use.
    feature_selection_method : {"rf", "mi", "anova"}, default="rf"
        Feature selection method.
    data_leak : bool, default=False
        If True, fit scaler on train+val (legacy compatibility).

    Returns
    -------
    None
        Artifacts are saved to models/<model_name>/ directory.
    """
    # Stage 1: Build suffix with job ID
    suffix = prepare_suffix_with_jobid(mode, tag)
    logging.info(f"[START] model={model_name} | mode={mode} | tag={tag} | suffix={suffix}")

    # Stage 2: Load and filter data based on mode
    data, subject_list, target_subjects_resolved = load_and_filter_data(
        model_name=model_name,
        mode=mode,
        tag=tag,
        target_subjects=target_subjects,
    )

    # Stage 3: Split data into train/val/test
    if mode == "source_only":
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_source_only_splits(
            model_name=model_name,
            tag=tag,
            seed=seed,
            target_subjects=target_subjects_resolved or target_subjects or [],
            time_stratify_labels=time_stratify_labels,
            time_stratify_tolerance=time_stratify_tolerance,
            time_stratify_window=time_stratify_window,
            time_stratify_min_chunk=time_stratify_min_chunk,
        )
    elif mode == "target_only":
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            subject_split_strategy="subject_time_split",
            subject_list=subject_list,
            target_subjects=target_subjects_resolved or target_subjects or [],
            model_name=model_name,
            seed=seed,
            time_stratify_labels=time_stratify_labels,
            time_stratify_tolerance=time_stratify_tolerance,
            time_stratify_window=time_stratify_window,
            time_stratify_min_chunk=time_stratify_min_chunk,
        )
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            subject_split_strategy=subject_split_strategy,
            subject_list=subject_list,
            target_subjects=target_subjects or [],
            model_name=model_name,
            seed=seed,
            time_stratify_labels=time_stratify_labels,
            time_stratify_tolerance=time_stratify_tolerance,
            time_stratify_window=time_stratify_window,
            time_stratify_min_chunk=time_stratify_min_chunk,
        )

    log_split_ratios(y_train, y_val, y_test, tag=f"{subject_split_strategy}|time_stratify={time_stratify_labels}")

    # Sanity checks
    if y_train.nunique() < 2:
        logging.error("Training labels are not binary. Stats: %s", y_train.value_counts().to_dict())
        return
    if min(len(X_train), len(X_val), len(X_test)) == 0:
        logging.error("One of the splits is empty. Review subject/time filtering.")
        return

    # Stage 4: Preprocess features
    X_train.columns = normalize_feature_names(X_train.columns)
    X_val.columns = normalize_feature_names(X_val.columns)
    if X_test is not None:
        X_test.columns = normalize_feature_names(X_test.columns)

    X_train = clean_feature_dataframe(X_train, drop_subject_id=True, drop_eeg=True, numeric_only=True)
    X_val = clean_feature_dataframe(X_val, drop_subject_id=True, drop_eeg=True, numeric_only=True)
    if X_test is not None:
        X_test = clean_feature_dataframe(X_test, drop_subject_id=True, drop_eeg=True, numeric_only=True)

    X_train, X_val, X_test = align_train_val_test_columns(X_train, X_val, X_test)

    # Stage 5: Feature selection & scaling
    selected_features, scaler, X_train_fs, X_val_fs, X_test_fs = select_features_and_scale(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        feature_selection_method=feature_selection_method,
        top_k=TOP_K_FEATURES,
        data_leak=data_leak,
    )

    selected_features = normalize_feature_names(selected_features)
    logging.info(f"[TRAIN] Selected {len(selected_features)} features.")

    feature_meta = {
        "selected_features": selected_features,
        "feature_source": model_name,
    }

    # Stage 6: Early checkpoint (save scaler & features)
    try:
        save_artifacts(
            model_name=model_name,
            suffix=suffix,
            best_clf=None,
            scaler=scaler,
            selected_features=selected_features,
            feature_meta=feature_meta,
        )
        logging.info("[CHECKPOINT] Early checkpoint saved (scaler & selected_features).")
    except Exception as e:
        logging.warning(f"[CHECKPOINT] Early checkpoint failed: {e}")

    # Stage 7: Train the model
    best_clf = None
    best_threshold = None
    results = {}
    try:
        best_clf, scaler, best_threshold, feature_meta, results = train_model(
            model_name=model_name,
            X_train_fs=X_train_fs,
            X_val_fs=X_val_fs,
            X_test_fs=X_test_fs,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            selected_features=selected_features,
            scaler=scaler,
            suffix=suffix,
            use_oversampling=use_oversampling,
            oversample_method=oversample_method,
        )
    except KeyboardInterrupt:
        logging.error("[TRAIN] Interrupted (KeyboardInterrupt). Will persist current checkpoint.")
    except Exception as e:
        logging.error(f"[TRAIN] Exception during training: {e}. Will persist current checkpoint.")
    finally:
        # Stage 8: Always save final/partial artifacts
        try:
            save_artifacts(
                model_name=model_name,
                suffix=suffix,
                best_clf=best_clf,
                scaler=scaler,
                selected_features=selected_features,
                feature_meta=feature_meta,
                best_threshold=best_threshold,
            )
            logging.info("[CHECKPOINT] Final artifacts saved.")
        except Exception as e:
            logging.error(f"[CHECKPOINT] Failed to save artifacts: {e}")

    logging.info(f"[DONE] Training complete for {model_name}{suffix}")
