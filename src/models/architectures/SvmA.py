"""SVM model with ANFIS-based feature weighting and PSO optimization.

This module implements:
- ANFIS-style feature importance calculation
- Feature selection using learned importance
- PSO-based hyperparameter and feature weighting optimization
- SVM training and evaluation

Trained models and selected features are saved using `joblib`.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
import logging
from scipy import stats
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif
from pyswarm import pso

from src.config import MODEL_PKL_PATH
from src.utils.io.savers import save_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def compute_feature_indices(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Compute ANFIS-style feature indices for each feature.
    
    The four indices are:
    1. Fisher Index: Between-class variance / Within-class variance
    2. Correlation Index: Pearson correlation with target (absolute value)
    3. T-test Index: t-statistic between class 0 and class 1 (normalized)
    4. Mutual Information Index: MI between feature and target
    
    All indices are normalized to [0, 1] range.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (n_samples, n_features)
    y : pd.Series
        Binary labels (0 or 1)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with shape (n_features, 4) containing normalized indices
        Columns: ['Fisher_Index', 'Correlation_Index', 'T-test_Index', 'Mutual_Information_Index']
    """
    n_features = X.shape[1]
    feature_names = X.columns.tolist()
    
    # Initialize arrays
    fisher_idx = np.zeros(n_features)
    corr_idx = np.zeros(n_features)
    ttest_idx = np.zeros(n_features)
    mi_idx = np.zeros(n_features)
    
    # Split by class
    X_class0 = X[y == 0]
    X_class1 = X[y == 1]
    
    n0, n1 = len(X_class0), len(X_class1)
    
    if n0 == 0 or n1 == 0:
        logging.warning("[ANFIS] One class has no samples, returning zero indices")
        return pd.DataFrame({
            'Fisher_Index': fisher_idx,
            'Correlation_Index': corr_idx,
            'T-test_Index': ttest_idx,
            'Mutual_Information_Index': mi_idx
        }, index=feature_names)
    
    for i, col in enumerate(feature_names):
        x = X[col].values
        x0 = X_class0[col].values
        x1 = X_class1[col].values
        
        # 1. Fisher Index: (mu1 - mu0)^2 / (var0 + var1)
        mu0, mu1 = np.mean(x0), np.mean(x1)
        var0, var1 = np.var(x0) + 1e-10, np.var(x1) + 1e-10
        fisher_idx[i] = (mu1 - mu0) ** 2 / (var0 + var1)
        
        # 2. Correlation Index: |pearson correlation with y|
        corr, _ = stats.pearsonr(x, y.values)
        corr_idx[i] = abs(corr) if not np.isnan(corr) else 0.0
        
        # 3. T-test Index: |t-statistic| (Welch's t-test)
        t_stat, _ = stats.ttest_ind(x0, x1, equal_var=False)
        ttest_idx[i] = abs(t_stat) if not np.isnan(t_stat) else 0.0
    
    # 4. Mutual Information (computed in batch)
    try:
        mi_idx = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    except Exception as e:
        logging.warning(f"[ANFIS] MI computation failed: {e}, using zeros")
        mi_idx = np.zeros(n_features)
    
    # Normalize all indices to [0, 1]
    def normalize(arr):
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val < 1e-10:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    fisher_idx = normalize(fisher_idx)
    corr_idx = normalize(corr_idx)  # Already in [0, 1] but normalize for consistency
    ttest_idx = normalize(ttest_idx)
    mi_idx = normalize(mi_idx)
    
    indices_df = pd.DataFrame({
        'Fisher_Index': fisher_idx,
        'Correlation_Index': corr_idx,
        'T-test_Index': ttest_idx,
        'Mutual_Information_Index': mi_idx
    }, index=feature_names)
    
    logging.info(f"[ANFIS] Computed feature indices for {n_features} features")
    logging.info(f"[ANFIS] Index stats - Fisher: [{fisher_idx.min():.3f}, {fisher_idx.max():.3f}], "
                 f"Corr: [{corr_idx.min():.3f}, {corr_idx.max():.3f}], "
                 f"T-test: [{ttest_idx.min():.3f}, {ttest_idx.max():.3f}], "
                 f"MI: [{mi_idx.min():.3f}, {mi_idx.max():.3f}]")
    
    return indices_df


def calculate_importance_degree(params: list, indices_df: pd.DataFrame) -> np.ndarray:
    """
    Compute importance degree per feature using ANFIS-style weighted indices.

    Parameters
    ----------
    params : list of float
        Weight parameters for each index in the order:
        [Fisher, Correlation, T-test, Mutual Information].
    indices_df : pandas.DataFrame
        DataFrame containing feature index scores.

    Returns
    -------
    numpy.ndarray
        Importance degree array, where values are:
        - 1   : High importance
        - 0.5 : Medium importance
        - 0   : Low importance
    """
    # --- Defensive conversion (must come first) ---
    if not isinstance(indices_df, pd.DataFrame):
        try:
            # Handle list/array/dict gracefully
            if isinstance(indices_df, (list, np.ndarray)):
                df = pd.DataFrame(indices_df)
                ncols = df.shape[1] if df.ndim > 1 else 1
                # Assign column names safely
                if ncols == 4:
                    df.columns = ["Fisher_Index", "Correlation_Index", "T-test_Index", "Mutual_Information_Index"]
                else:
                    # If only 1 column → copy it 4 times
                    if ncols == 1:
                        df = pd.concat([df] * 4, axis=1)
                    elif ncols > 4:
                        df = df.iloc[:, :4]
                    else:
                        df = pd.concat([df] * (4 // ncols), axis=1)
                    df.columns = ["Fisher_Index", "Correlation_Index", "T-test_Index", "Mutual_Information_Index"]
                indices_df = df
                logging.warning(f"indices_df auto-converted from {type(indices_df)} with shape {indices_df.shape}")
            elif isinstance(indices_df, dict):
                indices_df = pd.DataFrame(indices_df)
                missing = set(["Fisher_Index", "Correlation_Index", "T-test_Index", "Mutual_Information_Index"]) - set(indices_df.columns)
                for m in missing:
                    indices_df[m] = 0.0
                indices_df = indices_df[["Fisher_Index", "Correlation_Index", "T-test_Index", "Mutual_Information_Index"]]
                logging.warning("indices_df auto-built from dict keys.")
            else:
                raise TypeError(f"indices_df must be DataFrame, list, array, or dict — got {type(indices_df)}")

        except Exception as e:
            raise TypeError(
                f"indices_df could not be converted properly. Original type: {type(indices_df)}"
            ) from e

    # Ensure expected columns exist
    expected = ["Fisher_Index", "Correlation_Index", "T-test_Index", "Mutual_Information_Index"]
    for col in expected:
        if col not in indices_df.columns:
            indices_df[col] = 0.0

    # --- Ensure numeric dtype for all index columns ---
    for col in expected:
        if col in indices_df.columns:
            indices_df[col] = pd.to_numeric(indices_df[col], errors="coerce").fillna(0.0)
    # --- Compute weighted scores ---
    weighted_scores = (
        indices_df["Fisher_Index"] * params[0] +
        indices_df["Correlation_Index"] * params[1] +
        indices_df["T-test_Index"] * params[2] +
        indices_df["Mutual_Information_Index"] * params[3]
    )
    return np.where(weighted_scores > 0.75, 1, np.where(weighted_scores > 0.4, 0.5, 0))


def select_features(features_df: pd.DataFrame, importance_degree: np.ndarray) -> pd.DataFrame:
    """
    Select features based on importance threshold.

    Parameters
    ----------
    features_df : pandas.DataFrame
        Input feature matrix.
    importance_degree : numpy.ndarray
        Importance levels per feature (0, 0.5, or 1).

    Returns
    -------
    pandas.DataFrame
        Filtered DataFrame containing only features with
        importance equal to 1.
    """
    return features_df.loc[:, importance_degree == 1]


def optimize_svm_anfis(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    indices_df: pd.DataFrame
) -> tuple[list[float], list[dict]]:
    """
    Optimize ANFIS feature weights and SVM hyperparameters using PSO.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix.
    y_train : pandas.Series
        Training labels.
    X_val : pandas.DataFrame
        Validation feature matrix.
    y_val : pandas.Series
        Validation labels.
    indices_df : pandas.DataFrame
        Feature importance index values.

    Returns
    -------
    tuple
        - list of float: Optimal parameters [w1, w2, w3, w4, C, gamma].
        - list of dict: PSO optimization history for convergence visualization.
    """
    # Track PSO optimization history
    pso_history = []
    eval_count = [0]  # Use list to allow mutation in nested function
    
    def objective(params):
        eval_count[0] += 1
        importance_degree = calculate_importance_degree(params[:4], indices_df)
        X_train_sel = select_features(X_train, importance_degree)
        X_val_sel = select_features(X_val, importance_degree)

        if X_train_sel.shape[1] == 0:
            fitness = 1.0  # Penalty for empty selection
        else:
            model = SVC(kernel='rbf', C=params[4], gamma=params[5])
            model.fit(X_train_sel, y_train)
            accuracy = accuracy_score(y_val, model.predict(X_val_sel))
            fitness = -accuracy
        
        # Record this evaluation
        pso_history.append({
            'evaluation': eval_count[0],
            'params': {
                'w1': float(params[0]),
                'w2': float(params[1]),
                'w3': float(params[2]),
                'w4': float(params[3]),
                'C': float(params[4]),
                'gamma': float(params[5])
            },
            'fitness': float(fitness),
            'accuracy': float(-fitness) if fitness != 1.0 else 0.0
        })
        
        return fitness

    lb = [0, 0, 0, 0, 0.1, 0.001]
    ub = [1, 1, 1, 1, 10, 1]

    # Increased swarmsize and maxiter for better exploration
    # Original: swarmsize=3, maxiter=3 (12 evaluations) - insufficient
    # Improved: swarmsize=10, maxiter=20 (200 evaluations)
    optimal_params, _ = pso(objective, lb, ub, swarmsize=10, maxiter=20)
    return optimal_params, pso_history


def evaluate_model(model: SVC, X: pd.DataFrame, y: pd.Series, dataset_name: str) -> None:
    """
    Log classification metrics for a given dataset.

    Parameters
    ----------
    model : sklearn.svm.SVC
        Trained SVM model.
    X : pandas.DataFrame
        Input feature matrix.
    y : pandas.Series
        Ground truth labels.
    dataset_name : str
        Label used in log messages (e.g., ``"Training"``).

    Returns
    -------
    None
    """
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average=None)
    recall = recall_score(y, y_pred, average=None)
    f1 = f1_score(y, y_pred, average=None)
    conf_matrix = confusion_matrix(y, y_pred)

    logging.info(f"{dataset_name} Accuracy: {accuracy}")
    logging.info(f"{dataset_name} Precision: {precision}")
    logging.info(f"{dataset_name} Recall: {recall}")
    logging.info(f"{dataset_name} F1 Score: {f1}")
    logging.info(f"{dataset_name} Confusion Matrix:\n{conf_matrix}")


def SvmA_train(
    X_train: pd.DataFrame, X_val: pd.DataFrame,
    y_train: pd.Series, y_val: pd.Series,
    indices_df: pd.DataFrame, model: str,
    X_test: pd.DataFrame = None, y_test: pd.Series = None,
) -> tuple:
    """
    Train SVM using ANFIS-based feature weighting and PSO optimization.

    This function selects features, trains an SVM with optimized
    hyperparameters, and saves the model and selected features.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix.
    X_val : pandas.DataFrame
        Validation feature matrix.
    y_train : pandas.Series
        Training labels.
    y_val : pandas.Series
        Validation labels.
    indices_df : pandas.DataFrame
        Feature selection index scores.
    model : str
        Model name used for saving artifacts.
    X_test : pandas.DataFrame, optional
        Test feature matrix for final evaluation.
    y_test : pandas.Series, optional
        Test labels for final evaluation.

    Returns
    -------
    tuple
        (model, scaler, selected_features, results_dict)
    """
    logging.info("Starting SVM-ANFIS optimization...")

    optimal_params, pso_history = optimize_svm_anfis(X_train, y_train, X_val, y_val, indices_df)
    best_anfis_params, best_C, best_gamma = optimal_params[:4], optimal_params[4], optimal_params[5]
    
    # Save PSO optimization history for convergence visualization
    import json
    from pathlib import Path
    pbs_jobid_pso = os.environ.get("PBS_JOBID", "local")
    if "." in pbs_jobid_pso:
        pbs_jobid_pso = pbs_jobid_pso.split(".")[0]
    pbs_array_idx_pso = os.environ.get("PBS_ARRAY_INDEX", "1")
    history_dir = Path(MODEL_PKL_PATH) / model / f"{pbs_jobid_pso}" / f"{pbs_jobid_pso}[{pbs_array_idx_pso}]"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_filename = f"pso_history_{model}_{pbs_jobid_pso}_{pbs_array_idx_pso}.json"
    with open(history_dir / history_filename, 'w') as f:
        json.dump(pso_history, f, indent=2)
    logging.info(f"PSO optimization history saved to {history_dir / history_filename}")

    importance_degree = calculate_importance_degree(best_anfis_params, indices_df)
    X_train_sel = select_features(X_train, importance_degree)
    X_val_sel = select_features(X_val, importance_degree)

    # --- Safeguard: fallback if no features selected ---
    if X_train_sel.shape[1] == 0:
        logging.warning("[WARN] No features selected by ANFIS importance → using all input features instead.")
        X_train_sel = X_train.copy()
        X_val_sel = X_val.copy()

    svm_final = SVC(kernel='rbf', C=best_C, gamma=best_gamma, probability=True)
    svm_final.fit(X_train_sel, y_train)

    model_dir = f"{MODEL_PKL_PATH}/{model}"
    os.makedirs(model_dir, exist_ok=True)
    
    # --- Unified saving rules ---
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score
    dummy_scaler = StandardScaler().fit(X_train_sel)

    # Get PBS job ID and array index for proper directory structure
    pbs_jobid = os.environ.get("PBS_JOBID", "local")
    if "." in pbs_jobid:
        pbs_jobid = pbs_jobid.split(".")[0]
    pbs_array_idx = os.environ.get("PBS_ARRAY_INDEX", "1")
    
    # Construct mode with jobid[array_idx] format for save_artifacts
    save_mode = f"train_{pbs_jobid}[{pbs_array_idx}]"

    save_artifacts(
        model_obj=svm_final,
        scaler_obj=dummy_scaler,
        selected_features=X_train_sel.columns.tolist(),
        feature_meta=None,
        model_name=model,
        mode=save_mode
    )
    logging.info("Model and features saved successfully (via unified saver).")

    # --- Compute evaluation metrics ---
    def compute_metrics(model_obj, X, y, dataset_name):
        y_pred = model_obj.predict(X)
        y_prob = model_obj.predict_proba(X)[:, 1] if hasattr(model_obj, 'predict_proba') else None
        
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='binary', zero_division=0)
        rec = recall_score(y, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y, y_pred, average='binary', zero_division=0)
        conf = confusion_matrix(y, y_pred)
        
        metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix": conf.tolist(),
        }
        
        if y_prob is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y, y_prob))
                metrics["auc_pr"] = float(average_precision_score(y, y_prob))
            except Exception:
                metrics["roc_auc"] = None
                metrics["auc_pr"] = None
        
        logging.info(f"{dataset_name} Accuracy: {acc}")
        logging.info(f"{dataset_name} Precision: {prec}")
        logging.info(f"{dataset_name} Recall: {rec}")
        logging.info(f"{dataset_name} F1 Score: {f1}")
        logging.info(f"{dataset_name} Confusion Matrix:\n{conf}")
        
        return metrics
    
    results = {}
    results["train"] = compute_metrics(svm_final, X_train_sel, y_train, "Training")
    results["val"] = compute_metrics(svm_final, X_val_sel, y_val, "Validation")
    
    # Evaluate on test set if provided
    if X_test is not None and y_test is not None:
        X_test_sel = select_features(X_test, importance_degree)
        if X_test_sel.shape[1] == 0:
            X_test_sel = X_test.copy()
        results["test"] = compute_metrics(svm_final, X_test_sel, y_test, "Test")

    logging.info(f"Optimal ANFIS Parameters: {best_anfis_params}")
    logging.info(f"Optimal SVM Parameters (C, gamma): ({best_C}, {best_gamma})")
    
    # Return model, scaler, selected features, and results
    return svm_final, dummy_scaler, X_train_sel.columns.tolist(), results

