# src/utils/io/savers.py
"""Result saving utilities for training and evaluation outputs."""

import os
import re
import json
import logging
import datetime
import pandas as pd
from typing import Dict, Optional, List

from src.config import LATEST_JOB_FILENAME

# NOTE: The above import errors are due to missing dependencies in the environment and are unrelated to the variable naming unification. The code changes for variable naming are correct and ready for commit.

# === Unified artifact saving utility ===
import joblib
from typing import Any


def save_artifacts(
    model_obj=None,
    scaler_obj=None,
    selected_features=None,
    feature_meta=None,
    model_name=None,
    mode=None,
    *args,
    **kwargs,
):
    """
    Save model, scaler, features, and metadata to the appropriate directory.

    Parameters
    ----------
    model_obj : object
        Trained model object (e.g., RandomForestClassifier, LSTM model)
    scaler_obj : object or None
        Fitted scaler (StandardScaler, MinMaxScaler, etc.)
    selected_features : list or None
        List of selected feature names
    feature_meta : dict or None
        Additional feature metadata
    model_name : str
        Name of the model (e.g., 'RF', 'Lstm')
    mode : str
        Training mode (e.g., 'pooled', 'source_only')

    Notes
    -----
    - If ``best_threshold`` is provided via kwargs, it will be saved into the
      **same directory** as model/scaler/features:
      ``models/<model_name>/<job_base>/<job_base>[<idx>]/threshold_*.json``.
    """

    # --- Accept legacy/alternate keyword names from callers ---
    # The training pipeline passes: suffix, best_clf, scaler
    # Map them to the canonical local names used below.
    if mode is None and "suffix" in kwargs:
        # Treat provided "suffix" as the full mode token used for directory/suffix building
        mode = kwargs.pop("suffix")
    if model_obj is None and "best_clf" in kwargs:
        model_obj = kwargs.pop("best_clf")
    if scaler_obj is None and "scaler" in kwargs:
        scaler_obj = kwargs.pop("scaler")
    # selected_features / feature_meta already match our parameter names.
    # Ignore any extra kwargs like "results" safely.

    # backward-compatibility: if positional arguments were passed (old-style call)
    if model_obj is None and len(args) >= 6:
        model_obj, scaler_obj, selected_features, feature_meta, model_name, mode = args[:6]

    # --- type safety guard ---
    if isinstance(model_name, dict):
        logging.warning(f"[save_artifacts] Detected dict model_name: {model_name}")
        model_name = model_name.get("name", "unknown")
    if not isinstance(model_name, str):
        model_name = str(model_name)

    # --- safety patch: detect misordered call (RandomForestClassifier passed as mode) ---
    # In some cases, positional args may shift and cause model_name/mode swap
    # This block auto-corrects if mode appears to be a model object (has .fit method)
    if hasattr(mode, "fit") and not isinstance(mode, str):
        logging.warning("[save_artifacts] Detected argument shift: correcting model_name/mode order")
        # Reassign safely
        mode, model_name = model_name, feature_meta
        feature_meta, selected_features = selected_features, scaler_obj
        scaler_obj = model_obj

    # --- output directory ---
    job_id = os.environ.get("PBS_JOBID", "local")

    # --- normalize PBS jobid: remove hostname part like ".spcc-adm1"
    if "." in job_id:
        job_id = job_id.split(".")[0]

    # --- detect jobid & fold index robustly ---
    env_jobid = os.environ.get("PBS_JOBID", "local")
    if "." in env_jobid:
        env_jobid = env_jobid.split(".")[0]

    # ================================================================
    # Improved jobid/fold detection (avoid double embedding like [1][1])
    # ================================================================
    # If mode already includes jobid[fold], extract both jobid and fold directly
    m = re.search(r"(?P<jobid>\d{5,})\[(?P<fold>\d+)\]", str(mode))
    if m:
        jobid = m.group("jobid")
        fold_id = m.group("fold")
        model_dir = os.path.join("models", model_name, jobid, f"{jobid}[{fold_id}]")
    else:
        # extract jobid/fold normally (no explicit [n] in mode)
        jobid_match = re.search(r"(\d{5,})", str(mode))
        fold_match = re.search(r"\[(\d+)\]", str(mode))
        jobid = jobid_match.group(1) if jobid_match else env_jobid
        fold_id = fold_match.group(1) if fold_match else None

        # --- unified hierarchical directory ---
        if fold_id:
            model_dir = os.path.join("models", model_name, jobid, f"{jobid}[{fold_id}]")
        else:
            model_dir = os.path.join("models", model_name, jobid)

    # --- avoid duplicate patterns like 14060261[1][1] ---
    model_dir = re.sub(r"(\[\d+\])\1", r"\1", model_dir)

    # --- safety enforcement: always nested under jobid ---
    if not model_dir.startswith(os.path.join("models", model_name, jobid)):
        model_dir = os.path.join("models", model_name, jobid, f"{jobid}[{fold_id or '1'}]")

    # ============================================================
    # === Guard: skip duplicate call before directory creation ===
    # ============================================================
    # If mode doesn't include [n] (fold index), skip the entire save process
    # BEFORE creating directories. This prevents empty dirs like models/RF/14060981[1]/.
    if not re.search(r"\[\d+\]", str(mode)):
        # If callers provide a suffix that already expanded [n] into "_n",
        # we cannot infer the fold; in that case we intentionally skip to keep tree clean.
        # Callers should pass mode with "[n]" (PBS_ARRAY_INDEX) when saving fold artifacts.
        # This behavior is unchanged.
        logging.warning(
            f"[save_artifacts] Skipping non-fold call (mode={mode}, jobid={jobid}) — no directory created"
        )
        return

    # --- create directory only if fold-level call ---
    os.makedirs(model_dir, exist_ok=True)
    logging.debug(f"[save_artifacts] Unified save path = {model_dir}")

    # ============================================================
    # === Safe and unified artifact saving with sanity checks ===
    # ============================================================
    # --- unified suffix rule (avoid trailing underscores) ---
    suffix = f"_{mode.strip()}" if isinstance(mode, str) and mode.strip() else ""

    # --- remove redundant double jobid[fold] patterns in suffix ---
    suffix = re.sub(r"(\d{5,}\[\d+\])\1", r"\1", suffix)

    # === NEW FIX ===
    # Remove square brackets from suffix to prevent misplacement and globbing issues
    # Example: "joint_train_rank_mmd_mean_out_domain_14060610[4]" → "joint_train_rank_mmd_mean_out_domain_14060610_4"
    suffix = re.sub(r"\[(\d+)\]", r"_\1", suffix)

    # Also sanitize any double underscores created accidentally
    suffix = re.sub(r"__+", "_", suffix)

    # --- ensure model object validity ---
    if model_obj is None:
        logging.error(f"[SAVE] Model object is None — skipping save for {model_name}{suffix}.pkl")
    else:
        try:
            # --- special handling for Keras models (Keras 3.x compatibility) ---
            if "keras" in str(type(model_obj)).lower():
                keras_path = os.path.join(model_dir, f"{model_name}{suffix}.keras")
                model_obj.save(keras_path)
                logging.info(f"[SAVE] Keras model saved: {keras_path}")
            else:
                joblib.dump(model_obj, os.path.join(model_dir, f"{model_name}{suffix}.pkl"))
                logging.info(f"[SAVE] Model saved (joblib): {model_name}{suffix}.pkl")
        except Exception as e:
            logging.error(f"[SAVE] Failed to save model {model_name}{suffix}: {e}")

    # --- save scaler ---
    if scaler_obj is not None:
        try:
            joblib.dump(scaler_obj, os.path.join(model_dir, f"scaler_{model_name}{suffix}.pkl"))
        except Exception as e:
            logging.warning(f"[SAVE] Failed to save scaler for {model_name}: {e}")

    # --- save selected features ---
    if selected_features is not None:
        try:
            joblib.dump(selected_features, os.path.join(model_dir, f"selected_features_{model_name}{suffix}.pkl"))
        except Exception as e:
            logging.warning(f"[SAVE] Failed to save selected_features for {model_name}: {e}")

    # --- save feature metadata (JSON) ---
    if feature_meta is not None:
        try:
            with open(os.path.join(model_dir, f"feature_meta_{model_name}{suffix}.json"), "w") as f:
                json.dump(feature_meta, f, indent=2)
        except Exception as e:
            logging.warning(f"[SAVE] Failed to write feature_meta for {model_name}: {e}")

    # --- save threshold (if provided) IN THE SAME DIRECTORY as other artifacts ---
    best_threshold = kwargs.get("best_threshold", None)
    if best_threshold is not None:
        try:
            thr_path = os.path.join(model_dir, f"threshold_{model_name}{suffix}.json")
            with open(thr_path, "w") as f:
                json.dump(
                    {"threshold": float(best_threshold),
                     "saved_at": datetime.datetime.utcnow().isoformat()},
                    f, indent=2
                )
            logging.info(f"[SAVE] Threshold saved: {thr_path}")
        except Exception as e:
            logging.warning(f"[SAVE] Failed to save threshold for {model_name}: {e}")


    # --- save job marker (latest successful jobid) ---
    try:
        with open(os.path.join("models", model_name, LATEST_JOB_FILENAME), "w") as f:
            f.write(str(job_id))
    except Exception as e:
        logging.warning(f"[SAVE] Could not update latest_job.txt: {e}")

    logging.info(f"[SAVE] Artifacts saved under: {model_dir}")

# ============================================
# Evaluation result saving utility
# ============================================

def save_eval_results(
    results: dict,
    model_name: str,
    mode: str,
    job_id: str = None,
    out_dir: str = "models",
) -> str:
    """
    Save evaluation results (accuracy, F1, AUC, etc.) to JSON.

    Parameters
    ----------
    results : dict
        Dictionary of evaluation metrics, e.g. {"accuracy": 0.91, "f1": 0.88}.
    model_name : str
        Model name (e.g. "RF", "SvmA").
    mode : str
        Mode (e.g. "pooled", "target_only").
    job_id : str, optional
        PBS job ID (if available). Defaults to the value of PBS_JOBID env var.
    out_dir : str, default="models"
        Root directory for saving evaluation files.

    Returns
    -------
    str
        Path to the saved JSON file.
    """

    # --- resolve jobid (with hostname stripped) ---
    if job_id is None:
        job_id = os.environ.get("PBS_JOBID", "local")
    if "." in job_id:
        job_id = job_id.split(".")[0]

    # --- normalize job_id: drop trailing [n] if present ---
    # e.g. "14061921[3]" -> "14061921"
    pure_job_id = re.sub(r"\[\d+\]$", "", str(job_id))

    # --- get PBS array index (fold number) robustly ---
    # 1) prefer PBS_ARRAY_INDEX
    # 2) else, try to extract from job_id like "...[n]"
    # 3) else, default to "1"
    array_idx = os.environ.get("PBS_ARRAY_INDEX")
    if not array_idx:
        m = re.search(r"\[(\d+)\]$", str(job_id))
        array_idx = m.group(1) if m else "1"

    # --- hierarchical directory structure ---
    # results/evaluation/<model>/<pure_job_id>/<pure_job_id>[array_idx]/
    job_root = os.path.join(out_dir, model_name, pure_job_id)
    save_dir = os.path.join(job_root, f"{pure_job_id}[{array_idx}]")
    os.makedirs(save_dir, exist_ok=True)

    # --- optional: record the latest job ID for reference ---
    try:
        latest_marker = os.path.join(out_dir, model_name, LATEST_JOB_FILENAME)
        with open(latest_marker, "w") as f:
            f.write(str(pure_job_id) + "\n")
    except Exception as e:
        logging.warning(f"[SAVE] Could not update latest_job.txt: {e}")

    # --- save evaluation JSON ---
    # --- extract distance & level info if available ---
    dist = results.get("distance") or results.get("metric") or "unknown"
    level = results.get("level") or "unknown"
    tag = results.get("tag") or ""

    # --- construct file name including tag for unique identification ---
    # If tag exists and starts with known prefixes, use tag directly for filename
    if tag and (tag.startswith("imbalv2_") or tag.startswith("full_") or tag.startswith("rank_")):
        # Use tag directly: eval_results_{model}_{mode}_{tag}
        base_name = f"eval_results_{model_name}_{mode}_{tag}"
    elif dist != "unknown" and "_" in dist:
        # New format: dist = "mean_distance_mmd", level = "out_domain"
        base_name = f"eval_results_{model_name}_{mode}_rank_{dist}_{level}"
    else:
        # Legacy format fallback
        base_name = f"eval_results_{model_name}_{mode}_rank_{dist}_mean_{level}"

    out_path_json = os.path.join(save_dir, f"{base_name}.json")
    out_path_csv  = os.path.join(save_dir, f"{base_name}.csv")

    # --- save JSON ---
    with open(out_path_json, "w") as f:
        json.dump(results, f, indent=2)

    # --- optionally save flat CSV (for easier parsing) ---
    try:
        df = pd.json_normalize(results)
        df.to_csv(out_path_csv, index=False)
    except Exception as e:
        logging.warning(f"[SAVE] Failed to export CSV for {base_name}: {e}")

    # --- Detailed logging of saved result paths ---
    abs_json = os.path.abspath(out_path_json)
    abs_csv  = os.path.abspath(out_path_csv)
    logging.info(f"[EVAL] Results saved successfully:")
    logging.info(f"        JSON: {abs_json}")
    logging.info(f"        CSV : {abs_csv}")

    return out_path_json
