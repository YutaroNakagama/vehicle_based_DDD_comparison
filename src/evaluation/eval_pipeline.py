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

    # Step 3: Load model, scaler, and features
    clf, scaler, features = load_model_and_scaler(model, model_type, mode, tag, fold, jobid)
    if clf is None:
        logging.error("[EVAL] Model or scaler could not be loaded. Evaluation aborted.")
        return

    # Step 4: Align and normalize features
    X_test, features = align_and_normalize_features(X_test, features)
    X_test = scaler.transform(X_test)

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
        }
    )

    save_eval_results(
        results=result,
        model_name=model,
        mode=mode,
        job_id=jobid or os.environ.get("PBS_JOBID", "local"),
        out_dir="results/evaluation"
    )

    logging.info(
        f"[EVAL DONE] {model} | n={len(subjects)} | AUC={result.get('auc')} | F1={result.get('f1')}"
    )

