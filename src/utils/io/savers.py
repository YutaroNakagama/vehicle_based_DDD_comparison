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

    # --- save artifacts ---
    suffix = f"_{mode}" if mode else ""
    joblib.dump(model_obj, os.path.join(model_dir, f"{model_name}{suffix}.pkl"))

    if scaler_obj is not None:
        joblib.dump(scaler_obj, os.path.join(model_dir, f"scaler_{model_name}{suffix}.pkl"))

    if selected_features is not None:
        joblib.dump(selected_features, os.path.join(model_dir, f"selected_features_{model_name}{suffix}.pkl"))

    if feature_meta is not None:
        with open(os.path.join(model_dir, f"feature_meta_{model_name}{suffix}.json"), "w") as f:
            json.dump(feature_meta, f, indent=2)

    # Save job marker
    with open(os.path.join("models", model_name, "latest_job.txt"), "w") as f:
        f.write(str(job_id))

    print(f"[SAVE] Artifacts saved under: {model_dir}")
