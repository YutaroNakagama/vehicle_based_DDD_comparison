# Prior Research Hyperparameter Optimization Summary

## Optimization Methods

| Model | Method | Parameters Tuned | Trials | Notes |
|-------|--------|-----------------|--------|-------|
| SvmW | Optuna (TPE) | C (regularization) | **100** | RBF kernel, log-scale C |
| SvmA | PSO (Particle Swarm) | ANFIS + SVM params | n/a | No Optuna trials available |
| Lstm | K-Fold CV | None (fixed architecture) | n/a | 5-fold, early stopping |

## SvmW Optuna Details
- Objective: Maximize CV F1 score
- Search space: C ∈ [0.001, 10] (log scale)
- Best C values found: ~0.006 (very low regularization)
- Convergence: Gradual improvement; 100 trials is the canonical setting and is
  held constant across the 15-seed expansion (2026-04-30 decision) so all rows
  in the final table compare under the same optimization budget.

## Notes
- SvmA uses PSO for joint ANFIS-SVM optimization (not Optuna). With
  subject-wise SMOTE the per-job runtime tail can reach 8–15 h on the
  cluster's CPU nodes; this dominates the 15-seed expansion ETA. The
  per-trial cost is fixed (no early stopping), so increasing concurrency
  is the only knob — see `operations_log.md` for QOS caps.
- Lstm uses fixed architecture with early stopping (no hyperparameter
  search); per-job time on GPU is ~5 min, on CPU ~2 h.
- Only SvmW convergence plots are available from Optuna.
