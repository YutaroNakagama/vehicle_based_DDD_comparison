"""Shared model artifact loading utilities.

Consolidates repeated pickle/joblib model loading patterns used across 
evaluation and training pipelines. Handles legacy paths and file naming conventions.

Unified naming: all functions use `model_name` consistently.
"""

import os
import logging
import pickle
import joblib
from typing import Optional, Tuple, Any, List


def load_pickle_artifact(
    file_path: str,
    artifact_name: str = "artifact",
    use_joblib: bool = False,
) -> Optional[Any]:
    """Load a single pickle or joblib artifact from disk.

    Parameters
    ----------
    file_path : str
        Absolute path to the pickle file.
    artifact_name : str, default="artifact"
        Descriptive name for logging purposes.
    use_joblib : bool, default=False
        If True, use joblib.load instead of pickle.load.

    Returns
    -------
    object or None
        Loaded artifact, or None if file not found or loading failed.
    """
    if not os.path.exists(file_path):
        logging.warning(f"[LOAD] {artifact_name} not found: {file_path}")
        return None

    try:
        if use_joblib:
            obj = joblib.load(file_path)
        else:
            with open(file_path, "rb") as f:
                obj = pickle.load(f)
        logging.info(f"[LOAD] Successfully loaded {artifact_name}: {file_path}")
        return obj
    except Exception as e:
        logging.error(f"[LOAD] Failed to load {artifact_name} from {file_path}: {e}")
        return None


def load_model_artifacts(
    model_name: str,
    base_dir: str,
    suffix: str = "",
    mode: Optional[str] = None,
    use_joblib: bool = True,
) -> Tuple[Optional[Any], Optional[Any], Optional[List[str]]]:
    """Load model, scaler, and selected features from standard directory structure.

    Handles common naming patterns:
    - {model_name}{suffix}.pkl
    - scaler_{model_name}{suffix}.pkl
    - selected_features_{model_name}{suffix}.pkl

    Parameters
    ----------
    model_name : str
        Model identifier (e.g., "RF", "XGBoost", "Lstm").
    base_dir : str
        Base directory where artifacts are stored.
    suffix : str, default=""
        Optional suffix appended to filenames (e.g., "_target_only").
    mode : str, optional
        Mode identifier used in some naming conventions (e.g., "target_only").
    use_joblib : bool, default=True
        If True, use joblib for loading (recommended for sklearn models).

    Returns
    -------
    tuple of (model, scaler, features)
        - model : trained classifier or None
        - scaler : StandardScaler or None
        - features : list of str or None (selected feature names)
    """
    # Construct file paths
    if mode:
        model_file = os.path.join(base_dir, f"{model_name}_{mode}{suffix}.pkl")
        scaler_file = os.path.join(base_dir, f"scaler_{model_name}_{mode}{suffix}.pkl")
        feature_file = os.path.join(base_dir, f"selected_features_{model_name}_{mode}{suffix}.pkl")
    else:
        model_file = os.path.join(base_dir, f"{model_name}{suffix}.pkl")
        scaler_file = os.path.join(base_dir, f"scaler_{model_name}{suffix}.pkl")
        feature_file = os.path.join(base_dir, f"selected_features_{model_name}{suffix}.pkl")

    # Load artifacts
    model = load_pickle_artifact(model_file, "model", use_joblib=use_joblib)
    scaler = load_pickle_artifact(scaler_file, "scaler", use_joblib=use_joblib)
    features = load_pickle_artifact(feature_file, "selected_features", use_joblib=False)

    if model is None:
        logging.warning(f"[LOAD] Model loading failed for {model_name}")
    if scaler is None:
        logging.warning(f"[LOAD] Scaler not found; evaluation may require manual scaling")
    if features is None:
        logging.warning(f"[LOAD] Selected features not found; using all features")

    return model, scaler, features


def save_pickle_artifact(
    obj: Any,
    file_path: str,
    artifact_name: str = "artifact",
    use_joblib: bool = False,
) -> bool:
    """Save a single artifact as pickle or joblib.

    Parameters
    ----------
    obj : object
        Python object to save.
    file_path : str
        Absolute path where the file will be saved.
    artifact_name : str, default="artifact"
        Descriptive name for logging purposes.
    use_joblib : bool, default=False
        If True, use joblib.dump instead of pickle.dump.

    Returns
    -------
    bool
        True if save succeeded, False otherwise.
    """
    if obj is None:
        logging.warning(f"[SAVE] {artifact_name} is None — skipping save to {file_path}")
        return False

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        if use_joblib:
            joblib.dump(obj, file_path)
        else:
            with open(file_path, "wb") as f:
                pickle.dump(obj, f)
        logging.info(f"[SAVE] Successfully saved {artifact_name}: {file_path}")
        return True
    except Exception as e:
        logging.error(f"[SAVE] Failed to save {artifact_name} to {file_path}: {e}")
        return False
