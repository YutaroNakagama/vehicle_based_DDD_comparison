# exp3 c1 — Progress Summary (recorded AUROC × method × mode, and processing speed)

**Last updated: 2026-08-02.** Operational progress tracker for the exp3 "c1" grid. AUROC values are
the recorded metrics; the model-characteristic analysis is in
[c1_recorded_value_analysis.md](c1_recorded_value_analysis.md). The two **Within** modes are shown
here for progress completeness but are **retired** from the reported analysis (see
[within_retirement_plan.md](within_retirement_plan.md)); the deployable regimes are Pooled and Mixed.

## 1. Recorded AUROC by method × evaluation mode

Mean ± SD (n). All cells complete except RF-nofs Mixed (ETA below).

| Method | Pooled-SW-SMOTE | Within-in | Within-out | Mixed-in | Mixed-out |
|---|---|---|---|---|---|
| **RF (fs)** | 0.795 ± 0.052 (15) | 0.746 ± 0.089 (24) | 0.778 ± 0.108 (24) | 0.719 ± 0.085 (24) | 0.749 ± 0.104 (24) |
| **RF (nofs)** | 0.870 ± 0.026 (5) | 0.895 ± 0.053 (15) | 0.931 ± 0.044 (15) | 0.841 ± 0.082 (13)\* | 0.906 ± 0.086 (13)\* |
| **SvmW** | 0.694 ± 0.018 (6) | 0.800 ± 0.012 (8) | 0.759 ± 0.013 (8) | 0.742 ± 0.012 (8) | 0.771 ± 0.016 (8) |
| **SvmA** | 0.538 ± 0.042 (6) | 0.583 ± 0.026 (11) | 0.574 ± 0.064 (11) | 0.530 ± 0.026 (11) | 0.597 ± 0.022 (11) |
| **Lstm** | 0.513 ± 0.006 (6) | 0.779 ± 0.007 (15) | 0.763 ± 0.012 (15) | 0.782 ± 0.009 (15) | 0.779 ± 0.009 (15) |

\*RF-nofs Mixed cells are at **n=13/15**. They are already CI-adequate (required n ≈ 11–12 for a 95 %
CI half-width ≤ 0.05); the last 2 seeds/cell only tighten the interval. Lstm targets the DRT
`event_label` (a different construct from the KSS label the other four use), so its absolute level is
not directly commensurable.

**Completion status / ETA (2026-08-02):**

| Method | Status |
|---|---|
| RF (fs) | ✅ complete (all modes) |
| SvmW | ✅ complete (Within/Mixed 8/8, Pooled-SW-SMOTE 6/6) |
| SvmA | ✅ complete (Within/Mixed 11/11, Pooled-SW-SMOTE 6/6) |
| Lstm | ✅ complete (all modes) |
| RF (nofs) | Pooled/Within ✅; **Mixed-in 13/15, Mixed-out 13/15 — in progress** |

- **Only remaining work:** RF-nofs Mixed, 2 seeds/cell. Each RF-nofs cell is ~15–38 h (165-feature
  Optuna search, CPU, 6 workers) → **ETA ≈ 2026-08-03**. The analysis is already CI-adequate without
  them.

![Recorded AUROC by method × evaluation mode](figures/c1_recorded/fig_progress_auroc_by_mode.png)

*Recorded AUROC (mean ± SD, clipped to [0,1]) by method across the five SW-SMOTE modes. RF-nofs is
highest but most dispersed; in the deployable Mixed modes the top-10 RF-fs is 4th of 5 (SvmW and Lstm
above it); SvmA sits near the 0.5 line throughout.*

## 2. Processing speed (added as an evaluation metric per advisor's suggestion)

Two cost axes are reported: **training/tuning cost** (the price of building each detector) and
**inference latency** (the deployment-relevant per-window prediction speed).

### 2a. Training + tuning wall-time per cell (measured from the c1 run logs)

Each cell is one seed × mode fit including its hyperparameter search (Optuna for RF/SvmW/Lstm, PSO
for SvmA).

| Method | Learner / tuning | Median | Range | Notes |
|---|---|---|---|---|
| **RF (fs)** | RandomForest + Optuna, top-10 | **0.7 h** | 0.4–1.9 h | cheapest |
| **Lstm** | Bi-LSTM (GPU) | **0.8 h** | 0.4–1.5 h | GPU |
| **SvmA** | ANFIS/PSO + cuML-SVM (GPU) | **2.6 h** | 1.2–5.9 h | within/mixed; the Pooled SW-SMOTE arm is ~9–12.5 h (57 k-row subject-wise SMOTE) |
| **SvmW** | GHM-wavelet + SVM + Optuna | **17.4 h** | 6.3–101.8 h | wavelet-packet energy per window |
| **RF (nofs)** | RandomForest + Optuna, all 165 | **23.2 h** | 7.4–68.0 h | most expensive; 165-feature search |

**RF-nofs and SvmW are ~20–30× more expensive to train than RF-fs / Lstm.** RF-nofs's cost is the
165-feature Optuna search; SvmW's is the multiwavelet feature construction plus SVM tuning.

### 2b. Inference latency (per-window prediction, CPU single-thread, model only)

300-tree RF and RBF-SVM on the recorded feature sets (feature extraction excluded):

| Model | µs / window | windows / s |
|---|---|---|
| RF (top-10 features) | **24** | ~42,000 |
| RF (165 features) | 33 | ~30,000 |
| RBF-SVM (36 steering feats, 1750 SVs) | 110 | ~9,000 |

RF predicts a window in tens of microseconds; the RBF-SVM is ~4–5× slower (kernel evaluation over
~1750 support vectors) but still ~9 k windows/s. Lstm is a GPU sequence model (sub-millisecond per
window on GPU) and is not directly comparable to these CPU timings.

### 2c. Deployment reading

- **Inference is not a bottleneck for any method.** A KSS/DRT window spans seconds, so even the
  slowest learner (RBF-SVM, ≈0.1 ms) runs thousands of times faster than real time. On the model
  side, processing speed does **not** discriminate the methods at deployment.
- **The real cost separation is at build time** (RF-nofs and SvmW ≈ 20–30× RF-fs / Lstm) and in
  **feature extraction**: SvmW's GHM multiwavelet packet energy and SvmA's per-window statistical /
  entropy features add preprocessing that the RF-fs top-10 and Lstm (raw sequence) largely avoid. For
  a fielded detector this favours the cheaper RF-fs / Lstm pipelines when accuracy is comparable.

**Caveats.** Training times are wall-clock from a shared, non-isolated machine (GPU/CPU contention),
so they are order-of-magnitude, not benchmark-grade. Inference timings are model-only (exclude
feature extraction) and CPU single-thread; Lstm is GPU. A rigorous throughput benchmark on isolated
hardware is a possible follow-up.
