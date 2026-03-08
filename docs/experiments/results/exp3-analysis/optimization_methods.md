# Prior Research Hyperparameter Optimization Summary

## Optimization Methods

| Model | Method | Parameters Tuned | Notes |
|-------|--------|-----------------|-------|
| SvmW | Optuna (TPE) | C (regularization) | 50 trials, RBF kernel |
| SvmA | PSO (Particle Swarm) | ANFIS + SVM params | No Optuna trials available |
| Lstm | K-Fold CV | None (fixed architecture) | 5-fold, early stopping |

## SvmW Optuna Details
- Objective: Maximize CV F1 score
- Search space: C ∈ [0.001, 10] (log scale)
- Best C values found: ~0.006 (very low regularization)
- Convergence: Gradual improvement over 50 trials

## Notes
- SvmA uses PSO for joint ANFIS-SVM optimization (not Optuna)
- Lstm uses fixed architecture with early stopping (no hyperparameter search)
- Only SvmW convergence plots are available from Optuna
