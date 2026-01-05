"""Evaluation Pipeline for Driver Drowsiness Detection (DDD).

This module orchestrates the model evaluation workflow (data loading, model loading,
feature transformation, prediction, metric calculation). It exposes :func:`eval_pipeline`
to evaluate trained models.

Notes
-----
- Delegates all major stages to focused helper modules for testability.
- Automatically resolves job IDs from model files or latest_job.txt.
- Supports threshold optimization on validation set for F2 score.

Functions
---------
eval_pipeline(model, mode, ...) -> None
    Evaluate a trained model and save metrics to results directory.
"""

import datetime
import logging
import os
from typing import Optional

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)

from src.evaluation.eval_stages import (
    resolve_jobid_for_evaluation,
    extract_metadata_from_tag,
    extract_full_metadata_from_tag,
)
from src.utils.io.loaders import load_subjects_and_data, load_model_and_scaler
from src.utils.io.savers import save_eval_results
from src.utils.io.preprocessing import prepare_evaluation_features
from src.utils.io.target_resolution import resolve_target_subjects_from_tag
from src.utils.io.split_helpers import split_data, log_split_ratios
from src.evaluation.threshold import (
    load_or_optimize_threshold,
    extract_jobid_components,
)
from src.evaluation.models import lstm_eval, SvmA_eval, common_eval
from src import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
    threshold: Optional[float] = None,
    **kwargs,
) -> None:
    """Evaluate a trained DDD model and save metrics.

    Parameters
    ----------
    model : str
        Model name (e.g., "RF", "SvmA", "Lstm").
    mode : str
        Experiment mode (e.g., "pooled", "target_only", "source_only").
    tag : str, optional
        Experiment tag (e.g., "rank_dtw_mean_high").
    sample_size : int, optional
        Number of subjects to evaluate (None = all).
    seed : int, default=42
        Random seed for reproducibility.
    fold : int, default=0
        Fold index for cross-validation.
    subject_wise_split : bool, default=False
        Whether to perform subject-wise data splitting.
    jobid : str, optional
        Job ID for loading model artifacts (auto-resolved if None).
    target_file : str, optional
        Path to target subjects file.
    threshold : float, optional
        Custom prediction threshold (0.0-1.0). If None, uses default or optimized threshold.

    Returns
    -------
    None
        Evaluation metrics are saved to results/evaluation/<model>/ directory.
    """
    logging.info(f"[EVAL] Start {model} ({mode}) | tag={tag} | threshold={threshold}")

    # Stage 1: Load subjects and dataset
    subjects, model_name, data = load_subjects_and_data(
        model, fold, sample_size, seed, subject_wise_split
    )

    # Stage 2: Load CLI target subjects from file if provided
    cli_target_subjects = None
    if target_file and os.path.exists(target_file):
        with open(target_file, 'r') as f:
            cli_target_subjects = [line.strip() for line in f if line.strip()]
        logging.info(f"[EVAL] Loaded {len(cli_target_subjects)} target subjects from {target_file}")

    # Stage 3: Resolve target subjects for mode-based filtering
    target_subjects = resolve_target_subjects_from_tag(
        tag=tag,
        mode=mode,
        cli_target_subjects=cli_target_subjects
    )

    # Stage 4: Split data for evaluation
    if mode in ["source_only", "target_only"] and len(target_subjects) > 0:
        X_t_tr, X_val, X_test, y_t_tr, y_val, y_test = split_data(
            subject_split_strategy="subject_time_split",
            subject_list=subjects,
            target_subjects=target_subjects,
            model_name=model_name,
            seed=seed,
            time_stratify_labels=False,
            time_stratify_tolerance=0.02,
            time_stratify_window=0.10,
            time_stratify_min_chunk=100,
        )
        log_split_ratios(y_t_tr, y_val, y_test, tag=f"eval|target_timewise|mode={mode}|tag={tag}")
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            subject_split_strategy="random",
            subject_list=subjects,
            target_subjects=[],
            model_name=model_name,
            seed=seed,
            time_stratify_labels=False,
            time_stratify_tolerance=0.1,
            time_stratify_window=5,
            time_stratify_min_chunk=30,
        )
        log_split_ratios(y_train, y_val, y_test, tag=f"eval|random|mode={mode}|tag={tag}")

    # Stage 5: Resolve job ID and load model artifacts
    jobid, model_path = resolve_jobid_for_evaluation(model, mode, tag, jobid)
    
    clf, scaler, features = load_model_and_scaler(model, mode, tag, fold, jobid)
    if clf is None:
        logging.error(f"[EVAL] Model could not be loaded for jobid={jobid}. Aborted.")
        return
    
    logging.info(f"[EVAL] Loaded model for jobid={jobid}")

    # Stage 6: Prepare evaluation features
    X_test_prepared = prepare_evaluation_features(X_test, scaler, features)
    logging.info(f"[EVAL] Transformed X_test shape={X_test_prepared.shape}")
    
    X_val_prepared = None
    if 'X_val' in locals() and X_val is not None:
        X_val_prepared = prepare_evaluation_features(X_val, scaler, features)
        logging.info(f"[EVAL] Transformed X_val shape={X_val_prepared.shape}")

    # Stage 7: Model-specific evaluation
    if model == "Lstm":
        result = lstm_eval(X_test_prepared, y_test, model_name, clf, scaler)
    elif model == "SvmA":
        result = SvmA_eval(X_test_prepared, y_test, model_name, clf, features)
    else:
        result = common_eval(X_test_prepared, y_test, model_name, clf)

    # Stage 8: Load or optimize threshold (use CLI threshold if provided)
    try:
        base_jobid, run_idx = extract_jobid_components(jobid, model_path=model_path)
        fold_idx = int(fold) if isinstance(fold, int) else 0
        
        # Use CLI threshold if provided, otherwise load/optimize
        if threshold is not None:
            thr = threshold
            logging.info(f"[EVAL] Using CLI-specified threshold={thr:.3f}")
        else:
            thr = load_or_optimize_threshold(
                model=model,
                mode=mode,
                tag=tag,
                base_jobid=base_jobid,
                run_idx=run_idx,
                fold_idx=fold_idx,
                clf=clf,
                X_val=X_val_prepared if 'X_val_prepared' in locals() else None,
                y_val=y_val if 'y_val' in locals() else None,
            )

        # Apply threshold to test set
        if thr is not None:
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_test_prepared)[:, 1]
            elif hasattr(clf, "decision_function"):
                proba = clf.decision_function(X_test_prepared)
                p_min, p_max = float(proba.min()), float(proba.max())
                proba = (proba - p_min) / (p_max - p_min + 1e-12)
            else:
                proba = None

            if proba is not None:
                yhat_thr = (proba >= thr).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_test, yhat_thr).ravel()
                spec_thr = float(tn / (tn + fp)) if (tn + fp) > 0 else None
                
                result.update({
                    "thr": thr,
                    "acc_thr": float(accuracy_score(y_test, yhat_thr)),
                    "prec_thr": float(precision_score(y_test, yhat_thr, zero_division=0)),
                    "recall_thr": float(recall_score(y_test, yhat_thr, zero_division=0)),
                    "f1_thr": float(f1_score(y_test, yhat_thr, zero_division=0)),
                    "f2_thr": float(fbeta_score(y_test, yhat_thr, beta=2, zero_division=0)),
                    "specificity_thr": spec_thr,
                })
                logging.info(f"[EVAL] Applied threshold={thr:.3f} on TEST")
    except Exception as e:
        logging.warning(f"[EVAL] Threshold logic failed: {e}")

    # Stage 8: Extract positive-class metrics
    try:
        cr = result.get("classification_report", {}) or {}
        candidate_pos_keys = ["1", "1.0", "True", "pos", "positive"]
        pos_block = None
        for k in candidate_pos_keys:
            if k in cr and isinstance(cr[k], dict):
                pos_block = cr[k]
                break

        if pos_block:
            result.setdefault("precision_pos", float(pos_block.get("precision", 0.0)))
            result.setdefault("recall_pos", float(pos_block.get("recall", 0.0)))
            result.setdefault("f1_pos", float(pos_block.get("f1-score", 0.0)))
    except Exception as e:
        logging.warning(f"[EVAL] Post-metric normalization failed: {e}")

    # Stage 9: Add metadata and save results
    distance, level = extract_metadata_from_tag(tag)
    full_metadata = extract_full_metadata_from_tag(tag)
    
    # Add threshold suffix to tag for filename if custom threshold was specified
    save_tag = tag
    if threshold is not None:
        # e.g., "baseline_s42" -> "baseline_s42_th05" for threshold=0.5
        th_suffix = f"_th{int(threshold * 100):02d}"
        save_tag = f"{tag}{th_suffix}"
    
    result.update({
        "subject_list": subjects,
        "mode": mode,
        "tag": save_tag,  # Use save_tag with threshold suffix
        "original_tag": tag,  # Keep original tag for reference
        "custom_threshold": threshold,  # Record the custom threshold used
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "distance": distance,
        "level": level,
        "ranking_method": full_metadata.get("ranking_method", "unknown"),
        "distance_metric": full_metadata.get("distance_metric", "unknown"),
        "jobid_idx": f"{base_jobid}_{run_idx}_{fold_idx}",
    })

    save_eval_results(
        results=result,
        model_name=model,
        mode=mode,
        job_id=jobid,
        out_dir=cfg.RESULTS_EVALUATION_PATH
    )

    logging.info(f"[EVAL DONE] {model} | n={len(subjects)} | AUC={result.get('auc')} | F1={result.get('f1')}")
