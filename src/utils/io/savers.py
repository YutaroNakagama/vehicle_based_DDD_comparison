# src/utils/io/savers.py
"""Result saving utilities for training and evaluation outputs."""

import os
import json
import logging
import datetime
import pandas as pd
from typing import Dict, Optional, List


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
    scaler : object or None
        Fitted scaler (StandardScaler, MinMaxScaler, etc.)
    selected_features : list or None
        List of selected feature names
    feature_meta : dict or None
        Additional feature metadata
    model_name : str
        Name of the model (e.g., 'RF', 'Lstm')
    mode : str
        Training mode (e.g., 'pooled', 'source_only')
    """

    import os
    import joblib
    import json

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

    model_dir = os.path.join("models", model_name, str(job_id))
    os.makedirs(model_dir, exist_ok=True)

    # ============================================================
    # === Safe and unified artifact saving with sanity checks ===
    # ============================================================
    # --- unified suffix rule (avoid trailing underscores) ---
    suffix = f"_{mode.strip()}" if isinstance(mode, str) and mode.strip() else ""

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

    # --- save job marker (latest successful jobid) ---
    try:
        with open(os.path.join("models", model_name, "latest_job.txt"), "w") as f:
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
    import json
    import os

    # --- resolve jobid (with hostname stripped) ---
    if job_id is None:
        job_id = os.environ.get("PBS_JOBID", "local")
    if "." in job_id:
        job_id = job_id.split(".")[0]

    save_dir = os.path.join(out_dir, model_name, str(job_id))
    os.makedirs(save_dir, exist_ok=True)

    out_path = os.path.join(save_dir, f"eval_results_{model_name}_{mode}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"[SAVE] Evaluation results -> {out_path}")
    return out_path

