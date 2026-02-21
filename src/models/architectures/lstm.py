"""LSTM-based neural network model with attention mechanism.

This module defines:
- A custom Keras AttentionLayer
- A Bidirectional LSTM model with attention
- A k-fold training routine with early stopping and model saving
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Bidirectional, LSTM, Layer, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from src.config import MODEL_PKL_PATH, LSTM_SEGMENT_TIMESTEPS
from src.utils.io.savers import save_artifacts
from src.models.sampling.oversampling import apply_oversampling

logger = logging.getLogger(__name__)


def configure_gpu():
    """Configure GPU memory growth to avoid allocating all GPU memory at once.

    When a GPU is available, enables memory growth so TensorFlow allocates
    memory incrementally rather than grabbing the entire GPU memory.
    Falls back gracefully to CPU if no GPU is detected.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU detected: %s — memory growth enabled", gpus)
        except RuntimeError as e:
            logger.warning("GPU config error (must be set before init): %s", e)
    else:
        logger.info("No GPU detected — using CPU")


class AttentionLayer(Layer):
    """Bahdanau-style additive attention with 48-unit hidden projection.

    Projects each timestep through a Dense(48) layer, then computes
    a scalar alignment score per timestep via a learned context vector.
    Produces a weighted sum (context vector) over all timesteps.

    Reference: Wang et al. (2022), Section 3.3 — 48 memory neurons.

    Parameters
    ----------
    units : int, default 48
        Dimensionality of the attention hidden projection.
    **kwargs : dict
        Additional keyword arguments passed to the base Layer class.
    """

    def __init__(self, units: int = 48, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        """Initialize weights for attention projection and context vector."""
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.GlorotUniform(),
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(self.units,),
            initializer="zeros",
        )
        self.v = self.add_weight(
            name="att_context",
            shape=(self.units, 1),
            initializer=tf.keras.initializers.GlorotUniform(),
        )
        super().build(input_shape)

    def call(self, x):
        """Forward pass: apply attention to input sequence.

        Parameters
        ----------
        x : tensor
            LSTM output tensor of shape (batch, timesteps, features).

        Returns
        -------
        tensor
            Aggregated context vector (batch, features).
        """
        # Hidden projection: (batch, timesteps, units)
        e = tf.keras.backend.tanh(
            tf.keras.backend.dot(x, self.W) + self.b
        )
        # Alignment scores: (batch, timesteps, 1)
        score = tf.keras.backend.dot(e, self.v)
        a = tf.keras.backend.softmax(score, axis=1)
        # Weighted sum: (batch, features)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def create_segments(
    X: np.ndarray,
    y: np.ndarray,
    seg_len: int = LSTM_SEGMENT_TIMESTEPS,
) -> tuple:
    """Create fixed-length non-overlapping segments from 2-D feature data.

    Wang et al. (2022) divides time-series into 10-second segments, each
    containing 100 data points.  This function replicates that approach
    for the LSTM input.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Scaled feature matrix (2-D).
    y : np.ndarray, shape (n_samples,)
        Labels aligned with *X*.
    seg_len : int, default=LSTM_SEGMENT_TIMESTEPS
        Number of timesteps per segment (100).

    Returns
    -------
    X_seg : np.ndarray, shape (n_segments, seg_len, n_features)
    y_seg : np.ndarray, shape (n_segments,)
        Label of the last timestep in each segment.
    """
    n = len(X)
    if n < seg_len:
        # Not enough data — pad with zeros and return single segment
        pad = np.zeros((seg_len - n, X.shape[1]), dtype=X.dtype)
        X_padded = np.vstack([pad, X])
        return X_padded[np.newaxis, :, :], y[-1:]

    n_segments = n // seg_len
    X_seg = np.array(
        [X[i * seg_len : (i + 1) * seg_len] for i in range(n_segments)]
    )
    # Label = last timestep in each segment
    y_seg = np.array([y[(i + 1) * seg_len - 1] for i in range(n_segments)])
    return X_seg, y_seg


def build_lstm_model(input_shape: tuple) -> Model:
    """
    Construct the Bidirectional LSTM model with attention.

    Parameters
    ----------
    input_shape : tuple
        Shape of the input tensor in the form ``(timesteps, features)``.

    Returns
    -------
    tensorflow.keras.Model
        Compiled LSTM model with attention mechanism.
    """
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(36, return_sequences=True))(inputs)
    x = AttentionLayer()(x)  # (batch, features)
    x = Dense(20, activation='relu')(x)
    # Flatten is technically not needed here, but keep for consistency
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                   loss='binary_crossentropy',
                   metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    return model

def lstm_train(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
    X_test: pd.DataFrame = None,
    y_test: pd.Series = None,
    n_splits: int = 5,
    epochs: int = 100,
    batch_size: int = 32,
    use_oversampling: bool = False,
    oversample_method: str = "smote",
    target_ratio: float = 0.33,
) -> tuple:
    """
    Train a Bidirectional LSTM model with an attention mechanism using stratified k-fold cross-validation.

    This function performs preprocessing, model construction, training, evaluation, and saving of
    an LSTM model using a stratified k-fold strategy. It selects numeric features, handles missing or infinite
    values, and standardizes input before model training. Each fold's model and scaler are saved.

    Parameters
    ----------
    X : pandas.DataFrame
        Input feature matrix. Only numeric columns are used.
    y : pandas.Series
        Binary class labels (0 or 1) corresponding to ``X``.
    model_name : str
        Name used for saving the model and scaler (e.g., ``"Lstm"``) under ``MODEL_PKL_PATH``.
    X_val : pandas.DataFrame, optional
        Validation feature matrix (not used in k-fold but kept for interface consistency).
    y_val : pandas.Series, optional
        Validation labels.
    X_test : pandas.DataFrame, optional
        Test feature matrix for final evaluation.
    y_test : pandas.Series, optional
        Test labels for final evaluation.
    n_splits : int, default=5
        Number of folds for Stratified K-Fold cross-validation.
    epochs : int, default=100
        Number of training epochs for each fold.
    batch_size : int, default=32
        Batch size used during training.
    use_oversampling : bool, default=False
        Whether to apply oversampling to the training data.
    oversample_method : str, default="smote"
        Oversampling method (e.g., "smote", "undersample_rus").
        Applied per-fold on the training segments after reshaping 3D→2D.
    target_ratio : float, default=0.33
        Target minority/majority ratio for oversampling.

    Returns
    -------
    tuple
        (best_model, scaler, selected_features, results_dict)
    """
    import joblib
    from scipy.stats import ttest_ind
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

    # Configure GPU if available (safe to call multiple times)
    configure_gpu()

    # Convert to DataFrame if numpy array
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    if X_val is not None and isinstance(X_val, np.ndarray):
        X_val = pd.DataFrame(X_val)
    if y_val is not None and isinstance(y_val, np.ndarray):
        y_val = pd.Series(y_val)
    if X_test is not None and isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)
    if y_test is not None and isinstance(y_test, np.ndarray):
        y_test = pd.Series(y_test)

    # --- Exclude EEG features (Wang et al. 2022 uses vehicle dynamics only) ---
    # Keep only the 15 smooth_std_pe vehicle dynamics features.
    # EEG band power columns match the pattern "Channel_*".
    _vehicle_feature_pattern = r'^(speed|long_acc|lat_acc|lane_offset|steering_speed)_(std_dev|pred_error|mean)$'
    vehicle_cols = [c for c in X.columns if pd.Series([c]).str.match(_vehicle_feature_pattern).iloc[0]]
    if vehicle_cols:
        logging.info(
            f"[LSTM] Selecting {len(vehicle_cols)} vehicle dynamics features "
            f"(excluding {len(X.columns) - len(vehicle_cols)} non-vehicle columns)"
        )
        X = X[vehicle_cols]
        if X_val is not None:
            X_val = X_val[[c for c in vehicle_cols if c in X_val.columns]]
        if X_test is not None:
            X_test = X_test[[c for c in vehicle_cols if c in X_test.columns]]
    else:
        logging.warning("[LSTM] No vehicle dynamics columns matched — using all numeric columns.")

    # Extract numeric columns
    X_numeric = X.select_dtypes(include=[np.number])
    
    # Convert inf/-inf to NaN
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN and align corresponding labels
    X_numeric = X_numeric.dropna()
    y_cleaned = y.loc[X_numeric.index]

    # --- t-test feature selection (Wang et al. 2022, Section 4.2) ---
    # Remove features that do NOT significantly differ between classes (p > 0.05).
    # In the paper, steering wheel rate mean and SD were not significant.
    selected_cols = []
    dropped_cols = []
    for col in X_numeric.columns:
        class_0 = X_numeric.loc[y_cleaned == 0, col].dropna()
        class_1 = X_numeric.loc[y_cleaned == 1, col].dropna()
        if len(class_0) > 1 and len(class_1) > 1:
            _, p_value = ttest_ind(class_0, class_1, equal_var=False)
            if p_value < 0.05:
                selected_cols.append(col)
            else:
                dropped_cols.append((col, p_value))
        else:
            selected_cols.append(col)  # keep if insufficient data for test

    if dropped_cols:
        logging.info(
            f"[LSTM] t-test feature selection: dropped {len(dropped_cols)} features "
            f"(p > 0.05): {[(c, f'{p:.4f}') for c, p in dropped_cols]}"
        )
    logging.info(f"[LSTM] Retained {len(selected_cols)} features after t-test: {selected_cols}")
    X_numeric = X_numeric[selected_cols]
    
    # Clip values to fit within float32 range
    X_numeric = X_numeric.clip(lower=np.finfo(np.float32).min, upper=np.finfo(np.float32).max)
    
    # Convert to float32
    X_array = X_numeric.astype(np.float32).values

    # --- Scale ALL training data once (shared across folds) ---
    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(X_array)

    # --- Create fixed-length segments (Wang et al. 2022: 10-sec, 100 timesteps) ---
    seg_len = LSTM_SEGMENT_TIMESTEPS
    X_seq_all, y_seq_all = create_segments(X_scaled_all, y_cleaned.values, seg_len)
    logging.info(
        f"[LSTM] Created {len(X_seq_all)} segments "
        f"(seg_len={seg_len}) from {len(X_array)} samples"
    )

    # --- Compute class weights for imbalanced data ---
    # When oversampling is applied, the data distribution changes,
    # so class_weight is computed on the original (pre-oversampling) segments.
    # However, when oversampling IS used, we skip class_weight in model.fit()
    # to avoid double-adjusting for imbalance.
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_seq_all)
    cw = compute_class_weight('balanced', classes=classes, y=y_seq_all)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, cw)}
    logging.info(f"[LSTM] Class weights: {class_weight_dict}")
    # Only use class_weight during training when NOT applying oversampling
    fit_class_weight = None if use_oversampling else class_weight_dict

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_f1_scores = []
    fold_histories = []  # Store training history for convergence visualization
    best_f1 = -1.0
    best_model = None
    best_fold = 1

    for fold_no, (train_idx, test_idx) in enumerate(kf.split(X_seq_all, y_seq_all), start=1):
        X_train_fold = X_seq_all[train_idx]
        X_test_fold = X_seq_all[test_idx]
        y_train_fold = y_seq_all[train_idx]
        y_test_fold = y_seq_all[test_idx]

        # --- Apply oversampling/undersampling to training fold ---
        if use_oversampling:
            n_seg, n_ts, n_feat = X_train_fold.shape
            logging.info(
                f"[LSTM] Fold {fold_no}: applying {oversample_method} "
                f"(ratio={target_ratio}) to {n_seg} training segments"
            )
            # Reshape 3D (segments, timesteps, features) → 2D (segments, timesteps*features)
            X_flat = X_train_fold.reshape(n_seg, n_ts * n_feat)
            X_flat_df = pd.DataFrame(X_flat)
            y_series = pd.Series(y_train_fold)

            X_resampled, y_resampled = apply_oversampling(
                X_flat_df, y_series,
                method=oversample_method,
                target_ratio=target_ratio,
                random_state=42 + fold_no,  # Vary seed per fold
            )
            # Reshape back to 3D
            X_train_fold = X_resampled.values.reshape(-1, n_ts, n_feat).astype(np.float32)
            y_train_fold = y_resampled.values
            logging.info(
                f"[LSTM] Fold {fold_no}: after {oversample_method}: "
                f"{n_seg} → {len(X_train_fold)} segments, "
                f"class dist: {np.bincount(y_train_fold.astype(int))}"
            )

        model = build_lstm_model((seg_len, X_train_fold.shape[2]))
        early_stopping = EarlyStopping(
            monitor='val_auc', mode='max', patience=5,
            restore_best_weights=True,
        )

        logging.info(f'--- Fold {fold_no} Training Start ---')
        history = model.fit(
            X_train_fold, y_train_fold,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1,
            callbacks=[early_stopping],
            class_weight=fit_class_weight,
        )
        
        # Save training history for convergence visualization
        history_dict = {
            'loss': history.history.get('loss', []),
            'val_loss': history.history.get('val_loss', []),
            'accuracy': history.history.get('accuracy', []),
            'val_accuracy': history.history.get('val_accuracy', []),
            'auc': history.history.get('auc', []),
            'val_auc': history.history.get('val_auc', []),
            'epochs': list(range(1, len(history.history.get('loss', [])) + 1))
        }
        fold_histories.append({'fold': fold_no, 'history': history_dict})

        scores = model.evaluate(X_test_fold, y_test_fold, verbose=0)
        # scores: [loss, accuracy, auc]
        fold_acc = scores[1]
        fold_auc = scores[2] if len(scores) > 2 else 0.0

        # Compute F1 for positive class on fold test set
        y_prob_fold = model.predict(X_test_fold, verbose=0).flatten()
        y_pred_fold = (y_prob_fold > 0.5).astype(int)
        fold_f1 = f1_score(y_test_fold, y_pred_fold, zero_division=0)
        fold_f1_scores.append(fold_f1)
        logging.info(f'Fold {fold_no} - F1(pos): {fold_f1:.4f}, AUC: {fold_auc:.4f}')

        # Keep track of the best model (highest F1 on positive class)
        if fold_f1 > best_f1:
            best_f1 = fold_f1
            best_model = model
            best_fold = fold_no

        # Save model with proper PBS jobid format
        # Get PBS job ID and array index for proper directory structure
        pbs_jobid = os.environ.get("PBS_JOBID", "local")
        if "." in pbs_jobid:
            pbs_jobid = pbs_jobid.split(".")[0]
        pbs_array_idx = os.environ.get("PBS_ARRAY_INDEX", "1")
        
        # Construct mode with jobid[array_idx] format for save_artifacts
        save_mode = f"fold{fold_no}_{pbs_jobid}[{pbs_array_idx}]"
        
        save_artifacts(
            model_obj=model,
            scaler_obj=scaler,
            selected_features=X_numeric.columns.tolist(),
            feature_meta=None,
            model_name=model_name,
            mode=save_mode
        )
        logging.info(f"Artifacts for fold {fold_no} saved successfully (via unified saver).")
        logging.info(f'Fold {fold_no} - Loss: {scores[0]}, Accuracy: {scores[1]}')

    logging.info(f'\nF1 scores per fold: {fold_f1_scores}')
    logging.info(f'Average F1: {np.mean(fold_f1_scores):.4f}')
    
    # Save training histories for convergence visualization
    import json
    from pathlib import Path
    pbs_jobid_hist = os.environ.get("PBS_JOBID", "local")
    if "." in pbs_jobid_hist:
        pbs_jobid_hist = pbs_jobid_hist.split(".")[0]
    pbs_array_idx_hist = os.environ.get("PBS_ARRAY_INDEX", "1")
    history_filename = f"training_history_{model_name}_{pbs_jobid_hist}_{pbs_array_idx_hist}.json"
    history_path = Path(MODEL_PKL_PATH) / model_name / f"{pbs_jobid_hist}" / f"{pbs_jobid_hist}[{pbs_array_idx_hist}]" / history_filename
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(fold_histories, f, indent=2)
    logging.info(f"Training history saved to {history_path}")
    
    # --- Compute evaluation metrics using the best model ---
    # Use only the features selected by t-test (stored in selected_cols)
    def compute_metrics(model_obj, scaler_obj, X_data, y_data, dataset_name):
        # Convert to DataFrame if numpy array
        if isinstance(X_data, np.ndarray):
            X_data = pd.DataFrame(X_data)
        if isinstance(y_data, np.ndarray):
            y_data = pd.Series(y_data)
        
        # Use only the t-test selected vehicle dynamics columns
        available_cols = [c for c in selected_cols if c in X_data.columns]
        X_num = X_data[available_cols] if available_cols else X_data.select_dtypes(include=[np.number])
        X_num = X_num.replace([np.inf, -np.inf], np.nan).dropna()
        y_aligned = y_data.loc[X_num.index]
        X_num = X_num.clip(lower=np.finfo(np.float32).min, upper=np.finfo(np.float32).max)
        X_arr = X_num.astype(np.float32).values
        
        # Scale and create segments
        X_scaled = scaler_obj.transform(X_arr)
        X_seq, y_seq = create_segments(X_scaled, y_aligned.values, seg_len)
        
        # Predict
        y_prob = model_obj.predict(X_seq, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        y_true = y_seq
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='binary', zero_division=0)
        rec = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        conf = confusion_matrix(y_true, y_pred)
        
        metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix": conf.tolist(),
        }
        
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["auc_pr"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = None
            metrics["auc_pr"] = None
        
        logging.info(f"{dataset_name} Accuracy: {acc}")
        logging.info(f"{dataset_name} Precision: {prec}")
        logging.info(f"{dataset_name} Recall: {rec}")
        logging.info(f"{dataset_name} F1 Score: {f1}")
        
        return metrics
    
    results = {
        "cv_f1_scores": fold_f1_scores,
        "cv_mean_f1": float(np.mean(fold_f1_scores)),
        "best_fold": best_fold,
        "class_weight": class_weight_dict,
    }
    
    # Compute train metrics
    results["train"] = compute_metrics(best_model, scaler, X, y, "Training")
    
    # Compute validation metrics if provided
    if X_val is not None and y_val is not None:
        results["val"] = compute_metrics(best_model, scaler, X_val, y_val, "Validation")
    
    # Compute test metrics if provided
    if X_test is not None and y_test is not None:
        results["test"] = compute_metrics(best_model, scaler, X_test, y_test, "Test")
    
    return best_model, scaler, X_numeric.columns.tolist(), results

