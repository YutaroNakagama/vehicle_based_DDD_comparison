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

    target_subjects = []

    # Step 1: Load subjects and dataset
    subjects, model_type, data = load_subjects_and_data(
        model, fold, sample_size, seed, subject_wise_split
    )

    # --------------------------------------------------------------
    # Step 1.5: Restrict evaluation to target group for tagged runs
    # --------------------------------------------------------------
    default_rank_file = "results/domain_analysis/distance/rank_names.txt"
    target_subjects = []
    if tag and os.path.exists(default_rank_file):
        with open(default_rank_file) as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]
        tag_key = tag.replace("rank_", "")
        match = [x for x in lines if tag_key in os.path.basename(x)]
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

    # Step 2: Prepare split for evaluation
    # --- Use same split logic as training (split_data) ---
    from src.utils.io.split_helpers import split_data, log_split_ratios
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        subject_split_strategy="random",
        subject_list=subjects,
        target_subjects=target_subjects if mode in ["source_only", "target_only"] else [],
        model_type=model_type,
        seed=seed,
        # Time stratification parameters (same defaults as training)
        time_stratify_labels=None,
        time_stratify_tolerance=0.1,
        time_stratify_window=5,
        time_stratify_min_chunk=30,
    )
    log_split_ratios(
        y_train, y_val, y_test,
        tag=f"eval|mode={mode}|tag={tag}"
    )

    # ------------------------------------------------------------------
    # Step 2.5: For source_only/target_only, filter test data to target group
    # ------------------------------------------------------------------
    if mode in ["source_only", "target_only"] and len(target_subjects) > 0:
        if "subject_id" in X_test.columns:
            subj_col = X_test["subject_id"]
        elif "subject_id" in data.columns:
            subj_col = data.loc[X_test.index, "subject_id"]
        else:
            subj_col = None

        if subj_col is not None:
            mask = subj_col.isin(target_subjects)
            logging.info(
                f"[EVAL] (Unified) Restricting evaluation to {mask.sum()} target samples "
                f"(mode={mode}, tag={tag})"
            )
            X_test = X_test.loc[mask].reset_index(drop=True)
            y_test = y_test.loc[mask].reset_index(drop=True)
        else:
            logging.warning("[EVAL] subject_id not found; evaluating all samples.")

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
        jobid_path = f"{model_root}/latest_job.txt"
        if os.path.exists(jobid_path):
            with open(jobid_path) as f:
                jobid = f.read().strip()
            logging.info(f"[EVAL] Loaded latest jobid from {jobid_path}: {jobid}")
        else:
            jobid = os.getenv("PBS_JOBID", "local")
            logging.warning(f"[EVAL] No model file or latest_job.txt found, using default jobid={jobid}")

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
    eeg_keywords = ["Channel_", "EEG", "Theta", "Alpha", "Beta", "Gamma", "Delta"]
    drop_cols = [c for c in X_test.columns if any(k in c for k in eeg_keywords)]
    if drop_cols:
        logging.info(f"[EVAL] Dropping {len(drop_cols)} EEG-related columns (e.g., {drop_cols[:5]})")
        X_test = X_test.drop(columns=drop_cols)

    # Drop unnecessary or duplicated columns
    X_test = X_test.loc[:, ~X_test.columns.duplicated()]
    X_test = X_test.drop(columns=["subject_id"], errors="ignore")
    import numpy as np
    X_test = X_test.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    logging.info(f"[EVAL] X_test ready for scaling (n_features={X_test.shape[1]})")

    # ------------------------------------------------------------------
    # Step 4.5: Safe feature alignment to prevent KeyError
    # ------------------------------------------------------------------
    import pandas as pd

    def safe_align_columns(df: pd.DataFrame, expected_cols):
        """Align df to expected_cols (fill missing, drop extras)."""
        extra_cols = [c for c in df.columns if c not in expected_cols]
        missing_cols = [c for c in expected_cols if c not in df.columns]

        if extra_cols:
            logging.info(f"[EVAL] Dropping {len(extra_cols)} extra columns (e.g., {extra_cols[:5]})")
            df = df.drop(columns=extra_cols, errors="ignore")

        if missing_cols:
            logging.warning(f"[EVAL] {len(missing_cols)} missing columns filled with 0.0 "
                            f"(e.g., {missing_cols[:5]})")
            for c in missing_cols:
                df[c] = 0.0

        # Reorder to expected order
        df = df.reindex(columns=expected_cols)
        return df

    # Align with features saved during training
    if features is not None and len(features) > 0:
        X_test = safe_align_columns(X_test, features)
    elif hasattr(scaler, "feature_names_in_"):
        X_test = safe_align_columns(X_test, list(scaler.feature_names_in_))

    # Step 4.6: Clip extreme values before scaling
    clip_val = 1_000_000.0
    X_test = X_test.clip(lower=-clip_val, upper=clip_val, axis=1)

    # Step 4.7: Transform after alignment
    # --- Final sanity alignment for scikit-learn scalers ---
    if hasattr(scaler, "feature_names_in_"):
        expected_scaler_cols = list(scaler.feature_names_in_)
        missing_for_scaler = [c for c in expected_scaler_cols if c not in X_test.columns]
        extra_for_scaler = [c for c in X_test.columns if c not in expected_scaler_cols]

        if extra_for_scaler:
            logging.warning(f"[EVAL] Dropping {len(extra_for_scaler)} columns unseen at fit time "
                            f"(e.g., {extra_for_scaler[:5]})")
            X_test = X_test.drop(columns=extra_for_scaler, errors="ignore")

        if missing_for_scaler:
            logging.warning(f"[EVAL] {len(missing_for_scaler)} columns seen at fit time but now missing "
                            f"(e.g., {missing_for_scaler[:5]}). Filling with 0.0.")
            for c in missing_for_scaler:
                X_test[c] = 0.0

        X_test = X_test.reindex(columns=expected_scaler_cols)

    X_test = scaler.transform(X_test)
    logging.info(f"[EVAL] Successfully transformed X_test (shape={X_test.shape})")

    # Step 5: Model-specific evaluation
    if model == "Lstm":
        result = lstm_eval(X_test, y_test, model_type, clf, scaler)
    elif model == "SvmA":
        result = SvmA_eval(X_test, y_test, model, model_type, clf)
    else:
        result = common_eval(X_test, y_test, model, model_type, clf)

    # --- (New) Save probability histogram if available ---
    try:
        probs = result.get("y_pred_proba", None)
        if probs is not None and len(probs) > 0:
            import matplotlib.pyplot as plt
            out_job_dir = os.path.join("results", "evaluation", model, os.getenv("PBS_JOBID", jobid))
            os.makedirs(out_job_dir, exist_ok=True)
            fname_tag = (tag or "all").replace("/", "_")
            hist_path = os.path.join(out_job_dir, f"proba_hist_{model}_{mode}_{fname_tag}.png")

            plt.figure(figsize=(6,4))
            plt.hist(probs, bins=50)
            plt.xlabel("Predicted probability (positive class)")
            plt.ylabel("Count")
            plt.title(f"RF Probabilities on Test | mode={mode} tag={tag}")
            plt.tight_layout()
            plt.savefig(hist_path, dpi=150)
            plt.close()

            # Also search best F1 threshold for quick reference
            from sklearn.metrics import precision_recall_curve
            import numpy as np
            prec, rec, thr = precision_recall_curve(y_test, np.array(probs))
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            if thr.size > 0:
                best_idx = int(np.nanargmax(f1))
                # precision_recall_curve returns thresholds size = len(prec)-1
                best_thr = float(thr[min(best_idx, len(thr)-1)])
                best_f1 = float(np.nanmax(f1))
                result["best_threshold_f1"] = best_thr
                result["best_f1"] = best_f1

            # --- NEW: Apply best threshold for predicted labels ---
            y_pred_thr = (np.array(probs) >= best_thr).astype(int)
            from sklearn.metrics import classification_report, confusion_matrix

            report_thr = classification_report(y_test, y_pred_thr, output_dict=True)
            conf_thr = confusion_matrix(y_test, y_pred_thr)

            # store threshold-specific evaluation
            result["classification_report_best_thr"] = report_thr
            result["confusion_matrix_best_thr"] = conf_thr.tolist()

            pos_rate_pred_thr = float(y_pred_thr.mean())
            result["pred_pos_rate_best_thr"] = pos_rate_pred_thr

            logging.info(
                f"[EVAL] Applied dynamic threshold={best_thr:.3f} | "
                f"PosRate={pos_rate_pred_thr:.3%} | "
                f"F1@best={best_f1:.3f}"
            )

            result["proba_hist_path"] = hist_path
    except Exception as e:
        logging.warning(f"[EVAL] Failed to write probability histogram: {e}")

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
        job_id=os.getenv("PBS_JOBID", jobid),  # Prefer evaluation jobid if available
        out_dir="results/evaluation"
    )

    logging.info(
        f"[EVAL DONE] {model} | n={len(subjects)} | AUC={result.get('auc')} | F1={result.get('f1')}"
    )

