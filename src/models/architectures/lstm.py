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
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Bidirectional, LSTM, Layer, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from src.config import MODEL_PKL_PATH
from src.utils.io.savers import save_artifacts


class AttentionLayer(Layer):
    """
    Custom attention layer for sequence input.

    Applies learned weights over LSTM outputs to focus on important time steps.

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments passed to the base Layer class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        """Initialize weights and biases for attention."""
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer=tf.keras.initializers.GlorotUniform())
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
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
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)


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
    x = Bidirectional(LSTM(50, return_sequences=True))(inputs)
    x = AttentionLayer()(x)  # (batch, features)
    x = Dense(20, activation='relu')(x)
    # Flatten is technically not needed here, but keep for consistency
    x = Flatten()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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
    epochs: int = 10,
    batch_size: int = 16
) -> tuple:
    """
    Train a Bidirectional LSTM model with an attention mechanism using k-fold cross-validation.

    This function performs preprocessing, model construction, training, evaluation, and saving of
    an LSTM model using a k-fold strategy. It selects numeric features, handles missing or infinite
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
        Number of folds for K-Fold cross-validation.
    epochs : int, default=10
        Number of training epochs for each fold.
    batch_size : int, default=16
        Batch size used during training.

    Returns
    -------
    tuple
        (best_model, scaler, selected_features, results_dict)
    """
    import joblib
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

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

    # Extract numeric columns
    X_numeric = X.select_dtypes(include=[np.number])
    
    # Convert inf/-inf to NaN
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN and align corresponding labels
    X_numeric = X_numeric.dropna()
    y_cleaned = y.loc[X_numeric.index]
    
    # Clip values to fit within float32 range
    X_numeric = X_numeric.clip(lower=np.finfo(np.float32).min, upper=np.finfo(np.float32).max)
    
    # Convert to float32
    X_array = X_numeric.astype(np.float32).values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    best_accuracy = 0.0
    best_model = None
    best_scaler = None
    best_fold = 1

    for fold_no, (train_idx, test_idx) in enumerate(kf.split(X_array), start=1):
        # Standardize (fit per fold)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_array[train_idx])
        X_scaled_test = scaler.transform(X_array[test_idx])

        # Reshape for LSTM
        X_train = np.expand_dims(X_scaled, axis=1)
        X_test = np.expand_dims(X_scaled_test, axis=1)
        y_train = y_cleaned.values[train_idx]
        y_test = y_cleaned.values[test_idx]

        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        logging.info(f'--- Fold {fold_no} Training Start ---')
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1,
            callbacks=[early_stopping]
        )

        scores = model.evaluate(X_test, y_test, verbose=0)
        accuracies.append(scores[1])
        
        # Keep track of the best model (highest validation accuracy)
        if fold_no == 1 or scores[1] > best_accuracy:
            best_accuracy = scores[1]
            best_model = model
            best_scaler = scaler
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

    logging.info(f'\nScores per fold: {accuracies}')
    logging.info(f'Average accuracy: {np.mean(accuracies)}')
    
    # --- Compute evaluation metrics using the best model ---
    def compute_metrics(model_obj, scaler_obj, X_data, y_data, dataset_name):
        # Preprocess data
        X_num = X_data.select_dtypes(include=[np.number])
        X_num = X_num.replace([np.inf, -np.inf], np.nan).dropna()
        y_aligned = y_data.loc[X_num.index]
        X_num = X_num.clip(lower=np.finfo(np.float32).min, upper=np.finfo(np.float32).max)
        X_arr = X_num.astype(np.float32).values
        
        # Scale and reshape
        X_scaled = scaler_obj.transform(X_arr)
        X_reshaped = np.expand_dims(X_scaled, axis=1)
        
        # Predict
        y_prob = model_obj.predict(X_reshaped, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        y_true = y_aligned.values
        
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
        "cv_accuracies": accuracies,
        "cv_mean_accuracy": float(np.mean(accuracies)),
        "best_fold": best_fold,
    }
    
    # Compute train metrics
    results["train"] = compute_metrics(best_model, best_scaler, X, y, "Training")
    
    # Compute validation metrics if provided
    if X_val is not None and y_val is not None:
        results["val"] = compute_metrics(best_model, best_scaler, X_val, y_val, "Validation")
    
    # Compute test metrics if provided
    if X_test is not None and y_test is not None:
        results["test"] = compute_metrics(best_model, best_scaler, X_test, y_test, "Test")
    
    return best_model, best_scaler, X_numeric.columns.tolist(), results

