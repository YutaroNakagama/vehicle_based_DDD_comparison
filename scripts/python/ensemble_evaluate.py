"""Ensemble Evaluation Script for Imbalanced Data.

This script combines predictions from multiple trained models to improve
classification performance on imbalanced datasets. It supports various
ensemble strategies including majority voting and probability averaging.

Supported ensemble strategies:
  - majority_vote: Predict positive if majority of models predict positive
  - any_positive: Predict positive if ANY model predicts positive (high recall)
  - prob_average: Average predicted probabilities and apply threshold
  - weighted_prob: Weighted average based on model F2 scores

Examples
--------
Ensemble of all 6 imbalance methods:
    python ensemble_evaluate.py --models RF BalancedRF EasyEnsemble \\
        --tags smote_tomek smote_enn baseline smote_rus BalancedRF EasyEnsemble \\
        --mode pooled --strategy prob_average --threshold 0.5

Quick ensemble with top performers:
    python ensemble_evaluate.py --models RF EasyEnsemble --tags smote_rus EasyEnsemble \\
        --mode pooled --strategy prob_average --threshold 0.45
"""

import sys
import os
import argparse
import logging
import json
import datetime
import numpy as np

# Add project root to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from src.utils.io.loaders import load_subjects_and_data, load_model_and_scaler
from src.utils.io.preprocessing import prepare_evaluation_features
from src.utils.io.split_helpers import split_data
from src.evaluation.eval_stages import resolve_jobid_for_evaluation
from src import config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_model_predictions(
    model: str,
    mode: str,
    tag: str,
    X_test: np.ndarray,
    jobid: str = None,
    fold: int = 0,
) -> tuple:
    """Load a model and get its predictions on test data.
    
    Parameters
    ----------
    model : str
        Model name (e.g., "RF", "BalancedRF", "EasyEnsemble").
    mode : str
        Experiment mode.
    tag : str
        Model tag.
    X_test : np.ndarray
        Test features.
    jobid : str, optional
        Explicit job ID.
    fold : int, default=0
        Fold index.
    
    Returns
    -------
    tuple
        (predictions, probabilities, clf) or (None, None, None) if loading fails.
    """
    try:
        # Resolve job ID
        resolved_jobid, model_path = resolve_jobid_for_evaluation(model, mode, tag, jobid)
        
        # Load model and scaler
        clf, scaler, features = load_model_and_scaler(model, mode, tag, fold, resolved_jobid)
        
        if clf is None:
            logging.warning(f"[ENSEMBLE] Could not load {model}/{tag}")
            return None, None, None
        
        # Prepare features
        X_prepared = prepare_evaluation_features(X_test, scaler, features)
        
        # Get predictions
        y_pred = clf.predict(X_prepared)
        
        # Get probabilities
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_prepared)[:, 1]
        elif hasattr(clf, "decision_function"):
            scores = clf.decision_function(X_prepared)
            # Normalize to 0-1
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        else:
            y_proba = y_pred.astype(float)
        
        logging.info(f"[ENSEMBLE] Loaded {model}/{tag} (jobid={resolved_jobid})")
        return y_pred, y_proba, clf
        
    except Exception as e:
        logging.warning(f"[ENSEMBLE] Failed to load {model}/{tag}: {e}")
        return None, None, None


def ensemble_predictions(
    all_preds: list,
    all_probas: list,
    strategy: str = "prob_average",
    threshold: float = 0.5,
    weights: list = None,
) -> tuple:
    """Combine predictions from multiple models.
    
    Parameters
    ----------
    all_preds : list of np.ndarray
        List of prediction arrays from each model.
    all_probas : list of np.ndarray
        List of probability arrays from each model.
    strategy : str
        Ensemble strategy: "majority_vote", "any_positive", "prob_average", "weighted_prob".
    threshold : float
        Threshold for probability-based strategies.
    weights : list, optional
        Weights for weighted_prob strategy.
    
    Returns
    -------
    tuple
        (ensemble_predictions, ensemble_probabilities)
    """
    n_models = len(all_preds)
    n_samples = len(all_preds[0])
    
    if strategy == "majority_vote":
        # Predict positive if majority agrees
        vote_sum = np.sum(all_preds, axis=0)
        y_ensemble = (vote_sum > n_models / 2).astype(int)
        y_proba_ensemble = vote_sum / n_models
        
    elif strategy == "any_positive":
        # Predict positive if ANY model predicts positive (max recall)
        y_ensemble = np.max(all_preds, axis=0).astype(int)
        y_proba_ensemble = np.max(all_probas, axis=0)
        
    elif strategy == "prob_average":
        # Average probabilities and apply threshold
        y_proba_ensemble = np.mean(all_probas, axis=0)
        y_ensemble = (y_proba_ensemble >= threshold).astype(int)
        
    elif strategy == "weighted_prob":
        # Weighted average of probabilities
        if weights is None:
            weights = np.ones(n_models) / n_models
        else:
            weights = np.array(weights) / np.sum(weights)  # Normalize
        
        y_proba_ensemble = np.average(all_probas, axis=0, weights=weights)
        y_ensemble = (y_proba_ensemble >= threshold).astype(int)
        
    else:
        raise ValueError(f"Unknown ensemble strategy: {strategy}")
    
    return y_ensemble, y_proba_ensemble


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute comprehensive evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray
        Predicted probabilities.
    
    Returns
    -------
    dict
        Dictionary of metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    
    # AUC metrics
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_proba))
    except:
        metrics["auroc"] = None
    
    try:
        metrics["auprc"] = float(average_precision_score(y_true, y_proba))
    except:
        metrics["auprc"] = None
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble evaluation for imbalanced classification."
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model names (e.g., RF BalancedRF EasyEnsemble).",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        required=True,
        help="List of tags corresponding to each model.",
    )
    parser.add_argument(
        "--mode",
        choices=["pooled", "target_only", "source_only"],
        default="pooled",
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--strategy",
        choices=["majority_vote", "any_positive", "prob_average", "weighted_prob"],
        default="prob_average",
        help="Ensemble strategy.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prediction threshold for probability-based strategies.",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=None,
        help="Weights for weighted_prob strategy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--jobids",
        nargs="*",
        default=None,
        help="List of job IDs corresponding to each model. If not provided, auto-resolve.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results.",
    )
    
    args = parser.parse_args()
    
    if len(args.models) != len(args.tags):
        logging.error("Number of models must match number of tags!")
        sys.exit(1)
    
    if args.jobids and len(args.jobids) != len(args.models):
        logging.error("Number of jobids must match number of models!")
        sys.exit(1)
    
    logging.info("=" * 80)
    logging.info("[ENSEMBLE] Starting Ensemble Evaluation")
    logging.info(f"[ENSEMBLE] Models: {args.models}")
    logging.info(f"[ENSEMBLE] Tags: {args.tags}")
    logging.info(f"[ENSEMBLE] Strategy: {args.strategy}, Threshold: {args.threshold}")
    logging.info("=" * 80)
    
    # Load data (use first model's data loading)
    subjects, model_name, data = load_subjects_and_data(
        args.models[0], fold=0, sample_size=None, seed=args.seed, subject_wise_split=False
    )
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        subject_split_strategy="random",
        subject_list=subjects,
        target_subjects=[],
        model_name=model_name,
        seed=args.seed,
        time_stratify_labels=False,
        time_stratify_tolerance=0.1,
        time_stratify_window=5,
        time_stratify_min_chunk=30,
    )
    
    logging.info(f"[ENSEMBLE] Test set: {len(y_test)} samples, {np.sum(y_test)} positives ({100*np.mean(y_test):.1f}%)")
    
    # Load predictions from each model
    all_preds = []
    all_probas = []
    loaded_models = []
    
    for i, (model, tag) in enumerate(zip(args.models, args.tags)):
        jobid = args.jobids[i] if args.jobids else None
        y_pred, y_proba, clf = load_model_predictions(
            model=model,
            mode=args.mode,
            tag=tag,
            X_test=X_test,
            jobid=jobid,
            fold=0,
        )
        
        if y_pred is not None:
            all_preds.append(y_pred)
            all_probas.append(y_proba)
            loaded_models.append(f"{model}/{tag}")
            
            # Print individual model metrics
            metrics = compute_metrics(y_test, y_pred, y_proba)
            logging.info(
                f"  [{model}/{tag}] Recall={metrics['recall']:.3f}, "
                f"Precision={metrics['precision']:.3f}, F2={metrics['f2']:.3f}"
            )
    
    if len(all_preds) < 2:
        logging.error("[ENSEMBLE] Need at least 2 models for ensemble!")
        sys.exit(1)
    
    logging.info(f"[ENSEMBLE] Successfully loaded {len(all_preds)} models")
    
    # Combine predictions
    y_ensemble, y_proba_ensemble = ensemble_predictions(
        all_preds=all_preds,
        all_probas=all_probas,
        strategy=args.strategy,
        threshold=args.threshold,
        weights=args.weights,
    )
    
    # Compute ensemble metrics
    metrics = compute_metrics(y_test, y_ensemble, y_proba_ensemble)
    
    # Print results
    print("\n" + "=" * 80)
    print("🏆 ENSEMBLE RESULTS")
    print("=" * 80)
    print(f"Strategy: {args.strategy}")
    print(f"Threshold: {args.threshold}")
    print(f"Models: {', '.join(loaded_models)}")
    print("-" * 80)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1 Score:    {metrics['f1']:.4f}")
    print(f"F2 Score:    {metrics['f2']:.4f}")
    print(f"AUROC:       {metrics['auroc']:.4f}" if metrics['auroc'] else "AUROC:       N/A")
    print(f"AUPRC:       {metrics['auprc']:.4f}" if metrics['auprc'] else "AUPRC:       N/A")
    print("-" * 80)
    print(f"Confusion Matrix: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
    print("=" * 80)
    
    # Save results
    output_dir = args.output_dir or os.path.join(cfg.RESULTS_EVALUATION_PATH, "ensemble")
    os.makedirs(output_dir, exist_ok=True)
    
    result = {
        "strategy": args.strategy,
        "threshold": args.threshold,
        "models": loaded_models,
        "weights": args.weights,
        "metrics": metrics,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_test_samples": len(y_test),
        "n_positive": int(np.sum(y_test)),
    }
    
    output_file = os.path.join(
        output_dir,
        f"ensemble_{args.strategy}_{len(loaded_models)}models_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    
    logging.info(f"[ENSEMBLE] Results saved to {output_file}")


if __name__ == "__main__":
    main()
