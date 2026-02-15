"""SVM model with ANFIS-based feature weighting and PSO optimization.

This module implements the method from:
    Arefnezhad, S., Samiee, S., Eichberger, A., & Nahvi, A. (2019).
    Driver Drowsiness Detection Based on Steering Wheel Data Applying
    Adaptive Neuro-Fuzzy Feature Selection. Sensors, 19(4), 943.

Components:
- ANFIS (Adaptive Neuro-Fuzzy Inference System) with Gaussian MFs
  and Takagi-Sugeno rules for feature importance computation
- PSO optimization of ANFIS membership function parameters
- Separate SVM hyperparameter optimization via grid search
- Feature selection using ANFIS importance degree (threshold 0.5)

Trained models and selected features are saved using ``joblib``.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import itertools
import numpy as np
import pandas as pd
import joblib
import logging
from scipy import stats
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from pyswarm import pso

from src.config import MODEL_PKL_PATH

# ---------------------------------------------------------------------------
# SvmA-specific KSS mapping (Arefnezhad et al. 2019)
# Paper: KSS 1-6 → Alert (0), KSS 8-9 → Drowsy (1), KSS 7 → excluded
# ---------------------------------------------------------------------------
SVMA_KSS_BIN_LABELS = [1, 2, 3, 4, 5, 6, 8, 9]
SVMA_KSS_LABEL_MAP = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 8: 1, 9: 1}
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


# ============================================================================
# ANFIS (Adaptive Neuro-Fuzzy Inference System)
# ============================================================================

class ANFIS:
    """Takagi-Sugeno ANFIS for computing feature importance degree.

    Implements the fuzzy inference system described in Arefnezhad et al.
    (2019, Sensors 19(4), 943):

    * **4 inputs** – Fisher, Correlation, T-test, Mutual-Information indices
      (all normalised to [0, 1]).
    * **3 Gaussian MFs per input** – Low (L), Medium (M), High (H).
    * **81 rules** (3^4 combinations) with singleton consequences
      α_l ∈ {0, 0.5, 1}.
    * **Defuzzification** – weighted-average (Takagi-Sugeno first-order
      with constant consequent).

    PSO optimises the 24 MF parameters (centre *c* and spread *s* for
    each of the 12 Gaussian MFs).  Rule consequences are derived from
    a deterministic mapping of the input-level combination.
    """

    N_INPUTS = 4
    N_MFS    = 3       # Low, Medium, High
    N_RULES  = 81      # 3 ** 4
    N_PARAMS = 24      # 4 inputs × 3 MFs × 2 (c, s)

    # PSO bounds ----------------------------------------------------------
    #   centre c ∈ [0, 1]   (indices are normalised)
    #   spread s ∈ [0.01, 0.5]
    LB = []
    UB = []
    for _i in range(N_INPUTS):
        for _j in range(N_MFS):
            LB.extend([0.0, 0.01])
            UB.extend([1.0, 0.50])

    def __init__(self):
        self._rule_combos = list(
            itertools.product(range(self.N_MFS), repeat=self.N_INPUTS)
        )
        self._rule_consequences = self._init_rule_consequences()

    # -- Rule consequences ------------------------------------------------
    def _init_rule_consequences(self) -> np.ndarray:
        """Deterministic singleton consequences for the 81 rules.

        Each input MF level is coded as L=0, M=1, H=2.  The sum of the
        four levels (range 0 … 8) is mapped to:

        * sum ≥ 6  →  α = 1.0  (high importance)
        * 3 ≤ sum < 6  →  α = 0.5  (medium importance)
        * sum < 3  →  α = 0.0  (low importance)
        """
        alpha = np.zeros(self.N_RULES)
        for r, combo in enumerate(self._rule_combos):
            level_sum = sum(combo)
            if level_sum >= 6:
                alpha[r] = 1.0
            elif level_sum >= 3:
                alpha[r] = 0.5
            # else 0.0
        return alpha

    # -- Gaussian MF ------------------------------------------------------
    @staticmethod
    def gaussian_mf(x: np.ndarray, c: float, s: float) -> np.ndarray:
        """Gaussian membership function  exp(-0.5 * ((x-c)/s)^2)."""
        return np.exp(-0.5 * ((x - c) / (abs(s) + 1e-10)) ** 2)

    # -- Core inference ---------------------------------------------------
    def compute_importance_degree(
        self,
        indices_values: np.ndarray,
        mf_params: np.ndarray,
    ) -> np.ndarray:
        """Compute continuous importance degree for every feature.

        Parameters
        ----------
        indices_values : ndarray, shape (n_features, 4)
            Normalised filter-index values per feature.
        mf_params : ndarray, shape (24,)
            MF parameters ``[c_L0, s_L0, c_M0, s_M0, c_H0, s_H0,
            c_L1, s_L1, …, c_H3, s_H3]``.

        Returns
        -------
        ndarray, shape (n_features,)
            Importance degree ∈ [0, 1].
        """
        n_feat = indices_values.shape[0]

        # 1) Fuzzification – compute MF values  (n_feat, 4, 3)
        mf_vals = np.zeros((n_feat, self.N_INPUTS, self.N_MFS))
        for inp in range(self.N_INPUTS):
            for mf in range(self.N_MFS):
                idx = inp * 6 + mf * 2
                c = mf_params[idx]
                s = mf_params[idx + 1]
                mf_vals[:, inp, mf] = self.gaussian_mf(
                    indices_values[:, inp], c, s
                )

        # 2) Firing strengths – product across inputs  (n_feat, 81)
        firing = np.ones((n_feat, self.N_RULES))
        for r, combo in enumerate(self._rule_combos):
            for inp_idx, mf_idx in enumerate(combo):
                firing[:, r] *= mf_vals[:, inp_idx, mf_idx]

        # 3) Takagi-Sugeno defuzzification – weighted average
        denom = firing.sum(axis=1) + 1e-10
        importance = (firing @ self._rule_consequences) / denom
        return importance


def select_features(
    features_df: pd.DataFrame,
    importance_degree: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Select features whose ANFIS importance degree exceeds *threshold*.

    Per Arefnezhad et al. (2019), a threshold of **0.5** on the
    continuous importance degree is used.

    Parameters
    ----------
    features_df : pandas.DataFrame
        Full feature matrix  (n_samples, n_features).
    importance_degree : numpy.ndarray
        Continuous importance degree per feature (length = n_features).
    threshold : float, default 0.5
        Selection threshold.

    Returns
    -------
    pandas.DataFrame
        Subset of columns with importance > threshold.
    """
    mask = importance_degree > threshold
    if not mask.any():
        logging.warning(
            "[ANFIS] No features above threshold %.2f → using all features",
            threshold,
        )
        return features_df.copy()
    return features_df.loc[:, mask]


def _grid_search_svm(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
) -> tuple:
    """Grid-search SVM *C* and *gamma* on the already-selected features.

    Returns
    -------
    tuple
        (best_C, best_gamma, best_f1)
    """
    best_f1 = -1.0
    best_C, best_gamma = 1.0, 0.1
    for C in [0.1, 1.0, 10.0, 100.0]:
        for gamma in [0.001, 0.01, 0.1, 1.0]:
            svm = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced')
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_val)
            f1_val = f1_score(y_val, y_pred, average='binary', zero_division=0)
            if f1_val > best_f1:
                best_f1 = f1_val
                best_C, best_gamma = C, gamma
    logging.info(
        "[SvmA] SVM grid-search best: C=%.3f gamma=%.4f F1=%.4f",
        best_C, best_gamma, best_f1,
    )
    return best_C, best_gamma, best_f1


def optimize_svm_anfis(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    indices_df: pd.DataFrame,
) -> tuple:
    """Optimise ANFIS MF parameters via PSO, then grid-search SVM params.

    **Stage 1 – PSO** optimises the 24 Gaussian-MF parameters (centre
    and spread for 4 inputs × 3 MFs).  During PSO, a fixed-parameter
    SVM (C=1, gamma='scale') evaluates each candidate feature subset.

    **Stage 2 – Grid search** finds the best SVM (C, gamma) on the
    feature subset selected by the optimised ANFIS.

    Parameters
    ----------
    X_train, X_val : DataFrame
        Training / validation feature matrices.
    y_train, y_val : Series
        Training / validation labels.
    indices_df : DataFrame
        Per-feature filter indices (n_features × 4).

    Returns
    -------
    tuple
        (optimal_mf_params, best_C, best_gamma, pso_history, anfis)
    """
    anfis = ANFIS()
    indices_values = indices_df.values  # (n_features, 4)

    pso_history = []
    eval_count = [0]

    def objective(mf_params):
        eval_count[0] += 1
        importance = anfis.compute_importance_degree(indices_values, mf_params)
        mask = importance > 0.5
        n_selected = int(mask.sum())

        if n_selected == 0:
            pso_history.append({
                'evaluation': eval_count[0],
                'n_selected': 0, 'accuracy': 0.0, 'fitness': 1.0,
            })
            return 1.0  # penalty

        X_tr = X_train.iloc[:, mask]
        X_va = X_val.iloc[:, mask]

        # Fixed SVM during PSO (paper: PSO trains ANFIS only)
        # Paper Eq.(11): objective = 0.5 * (y_hat - y)^2 → equivalent to
        # maximising classification accuracy (Arefnezhad et al. 2019)
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')
        svm.fit(X_tr, y_train)
        y_pred = svm.predict(X_va)
        acc_val = accuracy_score(y_val, y_pred)
        fitness = -acc_val

        pso_history.append({
            'evaluation': eval_count[0],
            'n_selected': n_selected,
            'accuracy': float(acc_val),
            'fitness': float(fitness),
        })
        return fitness

    # PSO: 24 MF parameters (Table 3 of Arefnezhad et al. 2019)
    optimal_mf_params, _ = pso(
        objective, ANFIS.LB, ANFIS.UB,
        swarmsize=50, omega=0.95, phip=2.0, phig=2.0,
        maxiter=100,
    )

    # Stage 2 – grid-search SVM (C, gamma) on selected features
    importance = anfis.compute_importance_degree(indices_values, optimal_mf_params)
    X_train_sel = select_features(X_train, importance)
    X_val_sel = select_features(X_val, importance)
    best_C, best_gamma, _ = _grid_search_svm(
        X_train_sel, y_train, X_val_sel, y_val,
    )

    return optimal_mf_params, best_C, best_gamma, pso_history, anfis


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
    use_oversampling: bool = False,
    oversample_method: str = "smote",
    target_ratio: float = 0.33,
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
    use_oversampling : bool, optional
        Whether to apply oversampling (reserved for future use).
    oversample_method : str, optional
        Oversampling method name (reserved for future use).
    target_ratio : float, optional
        Target minority ratio (reserved for future use).

    Returns
    -------
    tuple
        (model, scaler, selected_features, results_dict)
    """
    logging.info("Starting SVM-ANFIS optimization...")

    # ----- Min-max normalization [0, 1] (Arefnezhad et al. 2019, Sec.2.2) -----
    # "Extracted features have been normalized between 0 and 1 using their
    #  minimum and maximum values."
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_normed = pd.DataFrame(
        minmax_scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index,
    )
    X_val_normed = pd.DataFrame(
        minmax_scaler.transform(X_val),
        columns=X_val.columns, index=X_val.index,
    )
    X_test_normed = None
    if X_test is not None:
        X_test_normed = pd.DataFrame(
            minmax_scaler.transform(X_test),
            columns=X_test.columns, index=X_test.index,
        )
    logging.info("[SvmA] Applied min-max [0,1] normalization (paper Sec.2.2)")

    optimal_mf_params, best_C, best_gamma, pso_history, anfis = optimize_svm_anfis(
        X_train_normed, y_train, X_val_normed, y_val, indices_df,
    )

    # Save PSO optimization history + MF parameters for reproducibility
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
        json.dump({
            'pso_history': pso_history,
            'optimal_mf_params': optimal_mf_params.tolist(),
            'best_C': best_C,
            'best_gamma': best_gamma,
        }, f, indent=2)
    logging.info(f"PSO optimization history saved to {history_dir / history_filename}")

    # Feature selection via optimised ANFIS
    indices_values = indices_df.values
    importance_degree = anfis.compute_importance_degree(indices_values, optimal_mf_params)
    n_selected = int((importance_degree > 0.5).sum())
    logging.info(f"[ANFIS] Selected {n_selected}/{len(importance_degree)} features (threshold=0.5)")
    X_train_sel = select_features(X_train_normed, importance_degree)
    X_val_sel = select_features(X_val_normed, importance_degree)

    # --- Safeguard: fallback if no features selected ---
    if X_train_sel.shape[1] == 0:
        logging.warning("[WARN] No features selected by ANFIS importance → using all input features instead.")
        X_train_sel = X_train_normed.copy()
        X_val_sel = X_val_normed.copy()

    svm_final = SVC(kernel='rbf', C=best_C, gamma=best_gamma, probability=True,
                     class_weight='balanced')
    svm_final.fit(X_train_sel, y_train)

    model_dir = f"{MODEL_PKL_PATH}/{model}"
    os.makedirs(model_dir, exist_ok=True)
    
    # --- Unified saving rules ---
    from sklearn.metrics import roc_auc_score, average_precision_score
    # Save a MinMaxScaler fit on the *selected* columns of the original
    # (un-normalised) training data.  During evaluation the pipeline calls
    # prepare_evaluation_features() which aligns to `selected_features`,
    # then applies scaler.transform() → identical [0,1] normalisation.
    selected_cols = X_train_sel.columns.tolist()
    scaler_to_save = MinMaxScaler(feature_range=(0, 1)).fit(
        X_train[selected_cols]
    )

    # Get PBS job ID and array index for proper directory structure
    pbs_jobid = os.environ.get("PBS_JOBID", "local")
    if "." in pbs_jobid:
        pbs_jobid = pbs_jobid.split(".")[0]
    pbs_array_idx = os.environ.get("PBS_ARRAY_INDEX", "1")
    
    # Construct mode with jobid[array_idx] format for save_artifacts
    save_mode = f"train_{pbs_jobid}[{pbs_array_idx}]"

    save_artifacts(
        model_obj=svm_final,
        scaler_obj=scaler_to_save,
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
    if X_test_normed is not None and y_test is not None:
        X_test_sel = select_features(X_test_normed, importance_degree)
        if X_test_sel.shape[1] == 0:
            X_test_sel = X_test_normed.copy()
        results["test"] = compute_metrics(svm_final, X_test_sel, y_test, "Test")

    logging.info(f"Optimal ANFIS MF params (24): {optimal_mf_params.tolist()}")
    logging.info(f"Optimal SVM Parameters (C, gamma): ({best_C}, {best_gamma})")
    
    # Return model, scaler, selected features, and results
    return svm_final, scaler_to_save, X_train_sel.columns.tolist(), results

