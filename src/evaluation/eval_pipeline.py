"""Evaluation Pipeline for Driver Drowsiness Detection (DDD).

This module coordinates the model evaluation process by delegating
data loading, feature alignment, model/scaler retrieval, and result saving
to utility functions under ``src/utils/io``.

The design philosophy is to keep this file minimal and focused solely on
the orchestration of the evaluation sequence, while all I/O and pre/post
processing are implemented in the reusable utility layer.

"""

import datetime
import json
import logging
import os
from typing import Optional

from src.utils.io.loaders import load_subjects_and_data, load_model_and_scaler
from src.utils.io.split_helpers import prepare_data_split
from src.utils.io.feature_utils import align_and_normalize_features
from src.utils.io.savers import save_eval_results
from src.evaluation.models import lstm_eval, SvmA_eval, common_eval


def eval_pipeline(
    model: str,
    mode: str,
    tag: Optional[str] = None,
    sample_size: Optional[int] = None,
    seed: int = 42,
    fold: int = 0,
    subject_wise_split: bool = False,
    jobid: Optional[str] = None,
    target_file: str = None,
    **kwargs,
) -> None:
    """Run the unified evaluation pipeline for a trained DDD model.

    This function performs model evaluation by orchestrating the following steps:

    1. Load subject list and combined dataset for evaluation.
    2. Split the dataset according to the evaluation configuration.
    3. Load the trained model, scaler, and selected feature set.
    4. Normalize and align feature names between train/eval stages.
    5. Perform evaluation using model-specific evaluation functions.
    6. Save evaluation metrics and metadata in JSON and CSV formats.

    Parameters
    ----------
    model : str
        Name of the trained model to evaluate (e.g., ``"RF"``, ``"SvmA"``, ``"Lstm"``).
    mode : str
        Experiment mode (e.g., ``"pooled"``, ``"only_target"``, ``"only_general"``, ``"finetune"``).
        Used to determine suffix and directory naming for evaluation outputs.
    tag : str, optional
        Additional experiment tag or variant identifier (default: ``None``).
    sample_size : int, optional
        Number of subjects to evaluate. If ``None``, all available subjects are used.
    seed : int, default=42
        Random seed used for reproducibility in sampling and splitting.
    fold : int, default=0
        Fold index for cross-validation. ``0`` indicates no fold-based split.
    subject_wise_split : bool, default=False
        Whether to perform subject-wise data splitting to prevent subject leakage.
    jobid : str, optional
        PBS job ID under which the model artifacts were saved. If not provided,
        the most recent job directory is automatically detected (if available).

    Returns
    -------
    None
        The function performs evaluation, logs metrics, and saves outputs to disk.
        Evaluation artifacts are stored under:
        ``results/evaluation/<model>/metrics_<model>_<mode>[_<tag>].json|csv``.

    Notes
    -----
    - All model/scaler/feature loading is handled by ``src.utils.io.loaders``.
    - Feature name normalization ensures consistency between train and eval pipelines.
    - Each evaluation run logs performance metrics such as AUC and F1-score.

    Examples
    --------
    >>> eval_pipeline(
    ...     model="SvmA",
    ...     mode="pooled",
    ...     tag="erm",
    ...     sample_size=20,
    ...     seed=123,
    ...     subject_wise_split=True
    ... )
    [EVAL] Start SvmA (pooled) | subject_split=True
    [EVAL DONE] SvmA | n=20 | AUC=0.812 | F1=0.791
    """
    logging.info(f"[EVAL] Start {model} ({mode}) | subject_split={subject_wise_split}")

    # Step 1: Load subjects and dataset
    subjects, model_type, data = load_subjects_and_data(
        model, fold, sample_size, seed, subject_wise_split
    )

    # Step 2: Prepare split for evaluation
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data_split(
        data, subjects, seed, subject_wise_split
    )

    # ------------------------------------------------------------------
    # Step 2.5: Filter test data according to mode
    # ------------------------------------------------------------------
    if mode in ["source_only", "target_only"]:
        # Load target subject list (from --target_file or default path)
        import pandas as pd
        target_file = kwargs.get("target_file", None)
        if target_file and os.path.exists(target_file):
            target_subjects = [s.strip() for s in open(target_file).read().splitlines() if s.strip()]
        else:
            # fallback: look for default group file in results/domain_analysis
            default_rank_file = "results/domain_analysis/distance/rank_names.txt"
            if os.path.exists(default_rank_file):
                with open(default_rank_file) as f:
                    target_subjects = [s.strip() for s in f.read().splitlines() if s.strip()]
            else:
                target_subjects = []
                logging.warning("[EVAL] No target subject list found; evaluating all subjects.")

        # Extract subject_id column if available
        if "subject_id" in data.columns:
            subj_col = data["subject_id"]
        else:
            subj_col = None

        if subj_col is not None and len(target_subjects) > 0:
            if mode == "target_only":
                mask = subj_col.isin(target_subjects)
                logging.info(f"[EVAL] Restricting evaluation to {mask.sum()} target samples.")
            else:  # source_only
                mask = ~subj_col.isin(target_subjects)
                logging.info(f"[EVAL] Restricting evaluation to {mask.sum()} source samples.")

            X_test = X_test.loc[mask].reset_index(drop=True)
            y_test = y_test.loc[mask].reset_index(drop=True)
        else:
            logging.warning("[EVAL] subject_id or target list not found — evaluating all samples.")

    # Step 3: Load model, scaler, and features
    # --- Resolve jobid if not provided ---
    if jobid is None:
        # Try environment variable FIXED_JOBID (from launch script)
        jobid = os.getenv("FIXED_JOBID")

        # If not set, try reading from latest_job.txt
        if not jobid:
            latest_path = f"models/{model}/latest_job.txt"
            if os.path.exists(latest_path):
                with open(latest_path, "r") as f:
                    jobid = f.readline().strip()
                logging.info(f"[EVAL] Loaded latest jobid from {latest_path}: {jobid}")
            else:
                # Fallback to PBS_JOBID or 'local'
                jobid = os.getenv("PBS_JOBID", "local")
                logging.warning(f"[EVAL] No latest_job.txt found; fallback to current jobid={jobid}")

    # --- Load model/scaler/features from resolved jobid ---
    clf, scaler, features = load_model_and_scaler(model, model_type, mode, tag, fold, jobid)
    if clf is None:
        logging.error(f"[EVAL] Model or scaler could not be loaded for jobid={jobid}. Evaluation aborted.")
        return
    else:
        logging.info(f"[EVAL] Successfully loaded model and scaler for jobid={jobid}")

    # ------------------------------------------------------------------
    # Step 4: Remove EEG features (not used during training)
    # ------------------------------------------------------------------
    eeg_cols = [c for c in X_test.columns if c.startswith("Channel_")]
    if eeg_cols:
        logging.info(f"[EVAL] Dropping {len(eeg_cols)} EEG-related columns (unused in training).")
        X_test = X_test.drop(columns=eeg_cols, errors="ignore")

    # Drop other known non-feature columns if they appear
    X_test = X_test.drop(columns=["subject_id"], errors="ignore")

    # Remove duplicated columns and keep numeric only
    X_test = X_test.loc[:, ~X_test.columns.duplicated()]
    import numpy as np
    X_test = X_test.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    logging.info(f"[EVAL] X_test ready for scaling (n_features={X_test.shape[1]})")

    # Step 4: Align and normalize features BEFORE scaling
    X_test, features = align_and_normalize_features(X_test, features)

    # --- Ensure column order matches the training phase ---
    if hasattr(scaler, "feature_names_in_"):
        X_test = X_test[scaler.feature_names_in_]

    # --- Transform after alignment ---
    X_test = scaler.transform(X_test)

    logging.info(f"[EVAL] Successfully transformed X_test (shape={X_test.shape})")

    # Step 5: Model-specific evaluation
    if model == "Lstm":
        result = lstm_eval(X_test, y_test, model_type, clf, scaler)
    elif model == "SvmA":
        result = SvmA_eval(X_test, y_test, model, model_type, clf)
    else:
        result = common_eval(X_test, y_test, model, model_type, clf)

    # Step 6: Save results with metadata
    result.update(
        {
            "subject_list": subjects,
            "mode": mode,
            "tag": tag,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Extract distance/level from tag (e.g., "rank_dtw_mean_high")
            "distance": (tag.split("_")[1] if tag and tag.startswith("rank_") else "unknown"),
            "level": (tag.split("_")[-1] if tag and tag.startswith("rank_") else "unknown"),
        }
    )

    save_eval_results(
        results=result,
        model_name=model,
        mode=mode,
        job_id=jobid,
        out_dir="results/evaluation"
    )

    logging.info(
        f"[EVAL DONE] {model} | n={len(subjects)} | AUC={result.get('auc')} | F1={result.get('f1')}"
    )

