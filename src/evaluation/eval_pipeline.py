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
import glob
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
)

from src.utils.io.loaders import load_subjects_and_data, load_model_and_scaler
from src.utils.io.savers import save_eval_results
from src import config as cfg
from src.evaluation.models import lstm_eval, SvmA_eval, common_eval
from src.utils.io.split_helpers import split_data, log_split_ratios


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

    target_subjects = []

    # Step 1: Load subjects and dataset
    subjects, model_name, data = load_subjects_and_data(
        model, fold, sample_size, seed, subject_wise_split
    )

    # --------------------------------------------------------------
    # Step 1.5: Restrict evaluation to target group for tagged runs
    # --------------------------------------------------------------
    default_rank_file = os.path.join(
        cfg.RESULTS_DOMAIN_ANALYSIS_PATH, "distance", cfg.RANK_NAMES_FILENAME
    )
    target_subjects = []
    if (tag and os.path.exists(default_rank_file)):
        with open(default_rank_file) as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]
        tag_key = tag.replace("rank_", "")
        match = [x for x in lines if os.path.basename(x).startswith(tag_key)]
        if match:
            group_file = os.path.normpath(match[0])
            if os.path.exists(group_file):
                with open(group_file) as g:
                    target_subjects = [s.strip() for s in g.readlines() if s.strip()]
                logging.info(f"[EVAL] Loaded target group list ({len(target_subjects)}) from {group_file}")
                logging.info(f"[EVAL] Mode={mode} | target_subjects={len(target_subjects)}")
            else:
                logging.warning(f"[EVAL] Target group file not found: {group_file}")
        else:
            logging.warning(f"[EVAL] No matching group found for tag={tag}")
    else:
        logging.info("[EVAL] No tag or rank_names.txt not found; evaluating all subjects.")

    if mode in ["source_only", "target_only"] and len(target_subjects) > 0:
        X_t_tr, X_val, X_test, y_t_tr, y_val, y_test = split_data(
            subject_split_strategy="subject_time_split",
            subject_list=subjects,                      # ignored when target_subjects provided
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
            time_stratify_labels=None,
            time_stratify_tolerance=0.1,
            time_stratify_window=5,
            time_stratify_min_chunk=30,
        )
        log_split_ratios(y_train, y_val, y_test, tag=f"eval|random|mode={mode}|tag={tag}")

    # Step 3: Load model, scaler, and features
    # --- Resolve jobid if not provided ---
    if jobid is None:
        # Try environment variable FIXED_JOBID (from launch script)
        jobid = os.getenv("FIXED_JOBID")

        # If not set, try reading from latest_job.txt
        if not jobid:
            latest_path = f"models/{model}/{cfg.LATEST_JOB_FILENAME}"
            if os.path.exists(latest_path):
                with open(latest_path, "r") as f:
                    jobid = f.readline().strip()
                logging.info(f"[EVAL] Loaded latest jobid from {latest_path}: {jobid}")
            else:
                # Fallback to PBS_JOBID or 'local'
                jobid = os.getenv("PBS_JOBID", "local")
                logging.warning(f"[EVAL] No latest_job.txt found; fallback to current jobid={jobid}")

    # --- NEW: prefer jobid corresponding to the same mode/tag ---
    # Example: use models/RF/<jobid>/<jobid>*/RF_target_only_rank_dtw_mean_high*.pkl
    model_root = f"models/{model}"
    if os.path.exists(model_root):
        import glob
        # Remove redundant 'rank_' prefix from tag to prevent double match
        tag_key = tag.replace("rank_", "") if tag else ""
        pattern = f"{model_root}/**/{model}_{mode}_rank_*{tag_key}*.pkl"
        matches = [m for m in glob.glob(pattern, recursive=True) if f"{model}_{mode}_" in os.path.basename(m)]
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            model_path = matches[0]
            jobid = model_path.split("/")[3]
            logging.info(f"[EVAL] Auto-detected model file for mode={mode}: {model_path}")
            # Skip fallback if found
            latest_model_found = True
        else:
            latest_model_found = False
            logging.warning(f"[EVAL] No specific model found for mode={mode}; will fallback to latest_job.txt.")

    # --- fallback only if nothing found ---
    if not locals().get("latest_model_found", False):
        model_path = None
        jobid_path = f"{model_root}/{cfg.LATEST_JOB_FILENAME}"
        if os.path.exists(jobid_path):
            with open(jobid_path) as f:
                jobid = f.read().strip()
            logging.info(f"[EVAL] Loaded latest jobid from {jobid_path}: {jobid}")
        else:
            jobid = os.getenv("PBS_JOBID", "local")
            logging.warning(f"[EVAL] No model file or latest_job.txt found, using default jobid={jobid}")

    # --- Load model/scaler/features from resolved jobid ---
    clf, scaler, features = load_model_and_scaler(model, mode, tag, fold, jobid)
    if clf is None:
        logging.error(f"[EVAL] Model or scaler could not be loaded for jobid={jobid}. Evaluation aborted.")
        return
    else:
        logging.info(f"[EVAL] Successfully loaded model and scaler for jobid={jobid}")

    # keywords for EEG columns (removed during evaluation)
    eeg_keywords = ["Channel_", "EEG", "Theta", "Alpha", "Beta", "Gamma", "Delta"]

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [c for c in df.columns if any(k in c for k in eeg_keywords)]
        if drop_cols:
            logging.info(f"[EVAL] Dropping {len(drop_cols)} EEG columns (e.g., {drop_cols[:5]})")
            df = df.drop(columns=drop_cols)
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.drop(columns=["subject_id"], errors="ignore")
        df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        def safe_align_columns(df_in: pd.DataFrame, expected_cols):
            extra_cols = [c for c in df_in.columns if c not in expected_cols]
            missing_cols = [c for c in expected_cols if c not in df_in.columns]
            if extra_cols:
                logging.info(f"[EVAL] Dropping {len(extra_cols)} extra columns (e.g., {extra_cols[:5]})")
                df_in = df_in.drop(columns=extra_cols, errors="ignore")
            if missing_cols:
                logging.warning(f"[EVAL] {len(missing_cols)} missing columns filled with 0.0 (e.g., {missing_cols[:5]})")
                for c in missing_cols:
                    df_in[c] = 0.0
            return df_in.reindex(columns=expected_cols)

        if features is not None and len(features) > 0:
            df = safe_align_columns(df, features)
        elif hasattr(scaler, "feature_names_in_"):
            df = safe_align_columns(df, list(scaler.feature_names_in_))

        df = df.clip(lower=-1_000_000.0, upper=1_000_000.0, axis=1)

        if hasattr(scaler, "feature_names_in_"):
            expected_scaler_cols = list(scaler.feature_names_in_)
            missing_for_scaler = [c for c in expected_scaler_cols if c not in df.columns]
            extra_for_scaler = [c for c in df.columns if c not in expected_scaler_cols]
            if extra_for_scaler:
                logging.warning(f"[EVAL] Dropping {len(extra_for_scaler)} columns unseen at fit time (e.g., {extra_for_scaler[:5]})")
                df = df.drop(columns=extra_for_scaler, errors="ignore")
            if missing_for_scaler:
                logging.warning(f"[EVAL] {len(missing_for_scaler)} columns seen at fit time but now missing (e.g., {missing_for_scaler[:5]}). Filling with 0.0.")
                for c in missing_for_scaler:
                    df[c] = 0.0
            df = df.reindex(columns=expected_scaler_cols)
        return pd.DataFrame(scaler.transform(df), index=df.index)

    X_test = _prep(X_test)
    logging.info(f"[EVAL] Transformed X_test shape={X_test.shape}")
    X_val_prepared = None
    if 'X_val' in locals():
        X_val_prepared = _prep(X_val)
        logging.info(f"[EVAL] Transformed X_val shape={X_val_prepared.shape}")

    # Step 5: Model-specific evaluation
    if model == "Lstm":
        result = lstm_eval(X_test, y_test, model_name, clf, scaler)
    elif model == "SvmA":
        result = SvmA_eval(X_test, y_test, model_name, clf, features)
    else:
        result = common_eval(X_test, y_test, model, model_name, clf)

    try:
        if tag and tag.startswith("rank_"):
            parts = tag.split("_")  # ["rank", "dtw", "mean", "high"]
            distance_key = "_".join(parts[1:-1]) if len(parts) >= 3 else "unknown"  # "dtw_mean"
            level = parts[-1] if len(parts) >= 2 else "unknown"                      # "high"
        else:
            distance_key, level = "unknown", "unknown"

        base_jobid, run_idx = None, None
        if 'model_path' in locals() and model_path:
            # .../models/RF/14209090/14209090[1]/RF_...
            mroot = model_path.split("/")  # ["models","RF","14209090","14209090[1]",...]
            if len(mroot) >= 4:
                base_jobid = mroot[2]
                subdir = mroot[3]  # "14209090[1]"
                import re
                mm = re.match(r"^(\d+)\[(\d+)\]$", subdir)
                if mm:
                    base_jobid = mm.group(1)
                    run_idx = mm.group(2)
        if base_jobid is None:
            import re
            mm = re.match(r"^(\d+)(?:\[(\d+)\])?$", str(jobid))
            if mm:
                base_jobid = mm.group(1)
                run_idx = mm.group(2) or "1"
        if base_jobid is None:
            base_jobid = str(jobid).replace("[","").replace("]","")
        if run_idx is None:
            run_idx = "1"

        fold_idx = int(fold) if isinstance(fold, int) else 0
        jobid_idx = f"{base_jobid}_{run_idx}"  # e.g., "14209090_1"

        pattern = (
            f"models/{model}/{base_jobid}/{base_jobid}[{run_idx}]/"
            f"threshold_{model}_{mode}_rank_{distance_key}_{level}_{jobid_idx}_{fold_idx}.json"
        )
        cand = glob.glob(pattern, recursive=True)

        threshold_path = None
        if cand and os.path.exists(cand[0]):
            cand.sort(key=os.path.getmtime, reverse=True)
            threshold_path = cand[0]
            with open(threshold_path, "r") as f:
                meta = json.load(f)
            thr = float(meta.get("threshold", 0.5))
            logging.info(f"[EVAL] Found strict-matched threshold file: {threshold_path} (thr={thr:.3f})")
        else:
            logging.info(f"[EVAL] No strict-matched threshold. Searching F2-opt on VAL (jobid_idx={jobid_idx})...")
            if X_val_prepared is None or 'y_val' not in locals():
                logging.warning("[EVAL] Validation split not available; cannot optimize threshold. Skipping.")
                thr = None
            else:
                # Prob on VAL
                if hasattr(clf, "predict_proba"):
                    p_val = clf.predict_proba(X_val_prepared)[:, 1]
                elif hasattr(clf, "decision_function"):
                    p_val = clf.decision_function(X_val_prepared)
                    p_min, p_max = float(np.min(p_val)), float(np.max(p_val))
                    p_val = (p_val - p_min) / (p_max - p_min + 1e-12)
                else:
                    p_val = None

                thr = None
                if p_val is not None:
                    thrs = np.linspace(0, 1, 1001)
                    best, best_f2 = 0.5, -1.0
                    yv = y_val.astype(int)
                    for t in thrs:
                        yhat = (p_val >= t).astype(int)
                        f2 = fbeta_score(yv, yhat, beta=2, zero_division=0)
                        if f2 > best_f2:
                            best_f2, best = f2, t
                    thr = float(best)
                    logging.info(f"[EVAL] Optimized F2 on VAL: thr={thr:.3f}, F2={best_f2:.4f}")

                    candidate_subdir = os.path.join("models", model, base_jobid, f"{base_jobid}[{run_idx}]")
                    target_dir = candidate_subdir if os.path.isdir(candidate_subdir) else os.path.join("models", model, base_jobid)
                    os.makedirs(target_dir, exist_ok=True)
                    expected_name = (
                        f"threshold_{model}_{mode}_rank_{distance_key}_{level}_{jobid_idx}_{fold_idx}.json"
                    )
                    threshold_path = os.path.join(target_dir, expected_name)
                    with open(threshold_path, "w") as f:
                        json.dump({"threshold": thr}, f, indent=2)
                    logging.info(f"[EVAL] Saved new threshold file → {threshold_path}")
                else:
                    logging.warning("[EVAL] Classifier has no probability-like scores; cannot optimize threshold.")
                    thr = None

        if thr is not None:
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_test)[:, 1]
            elif hasattr(clf, "decision_function"):
                proba = clf.decision_function(X_test)
                p_min, p_max = float(np.min(proba)), float(np.max(proba))
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
                    "_proba_len": int(len(proba)),
                    "_threshold_file": threshold_path,
                })
                logging.info(f"[EVAL] Applied threshold={thr:.3f} on TEST (file={threshold_path})")
            else:
                logging.info("[EVAL] Classifier has no probabilities; threshold application skipped.")
        else:
            logging.info("[EVAL] No threshold available; using default metrics only.")
    except Exception as e:
        logging.warning(f"[EVAL] Threshold logic failed: {e}")

    # ----------------------------------------------------------------------
    # Add explicit positive-class and macro/weighted metrics to the result.
    # This prevents ambiguity in downstream summary scripts and plots.
    # ----------------------------------------------------------------------
    try:
        cr = result.get("classification_report", {}) or {}

        # Helper to locate the positive class block regardless of key format.
        # Possible keys seen in historical runs: "1", "1.0", "True", "pos", "positive"
        candidate_pos_keys = ["1", "1.0", "True", "pos", "positive"]
        pos_block = None
        for k in candidate_pos_keys:
            if k in cr and isinstance(cr[k], dict):
                pos_block = cr[k]
                break

        macro_block = cr.get("macro avg", {})
        weighted_block = cr.get("weighted avg", {})

        # Positive-class (binary, pos_label=1) metrics
        if pos_block:
            result.setdefault("precision_pos", float(pos_block.get("precision", 0.0)))
            result.setdefault("recall_pos",    float(pos_block.get("recall", 0.0)))
            result.setdefault("f1_pos",        float(pos_block.get("f1-score", 0.0)))
        else:
            # If the report did not include a positive block (should be rare),
            # keep explicit zeros to avoid accidental fallbacks to macro/weighted.
            result.setdefault("precision_pos", 0.0)
            result.setdefault("recall_pos",    0.0)
            result.setdefault("f1_pos",        0.0)

        # Macro / weighted metrics (kept only as clearly named references)
        if macro_block:
            result.setdefault("precision_macro", float(macro_block.get("precision", 0.0)))
            result.setdefault("recall_macro",    float(macro_block.get("recall", 0.0)))
            result.setdefault("f1_macro",        float(macro_block.get("f1-score", 0.0)))
        if weighted_block:
            result.setdefault("precision_weighted", float(weighted_block.get("precision", 0.0)))
            result.setdefault("recall_weighted",    float(weighted_block.get("recall", 0.0)))
            result.setdefault("f1_weighted",        float(weighted_block.get("f1-score", 0.0)))

        # Backward-compatibility aliases for thresholded metrics:
        # Keep original keys but also expose explicit *_thr_pos.
        if "prec_thr" in result and "precision_thr_pos" not in result:
            result["precision_thr_pos"] = float(result.get("prec_thr", 0.0))
        if "recall_thr" in result and "recall_thr_pos" not in result:
            result["recall_thr_pos"] = float(result.get("recall_thr", 0.0))
        if "f1_thr" in result and "f1_thr_pos" not in result:
            result["f1_thr_pos"] = float(result.get("f1_thr", 0.0))
        if "f2_thr" in result and "f2_thr_pos" not in result:
            result["f2_thr_pos"] = float(result.get("f2_thr", 0.0))

        # If top-level generic keys are present and ambiguous, do NOT rewrite them here.
        # Downstream summarizer will now consume explicit *_pos keys.
    except Exception as e:
        logging.warning(f"[EVAL] Post-metric normalization failed: {e}")


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
            "jobid_idx": f"{base_jobid}_{run_idx}_{int(fold) if isinstance(fold,int) else 0}",
        }
    )

    save_eval_results(
        results=result,
        model_name=model,
        mode=mode,
        job_id=os.getenv("PBS_JOBID", jobid),  # Prefer evaluation jobid if available
        out_dir=cfg.RESULTS_EVALUATION_PATH
    )

    logging.info(
        f"[EVAL DONE] {model} | n={len(subjects)} | AUC={result.get('auc')} | F1={result.get('f1')}"
    )

