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

### 2b. Model-only inference latency (per-window prediction)

Model prediction time with the features already computed (300-tree RF and RBF-SVM on CPU
single-thread; Lstm is the BiLSTM-36 + attention forward pass measured with TF):

| Model | µs / window | windows / s |
|---|---|---|
| RF (top-10 features) | **24** | ~42,000 |
| RF (165 features) | 33 | ~30,000 |
| RBF-SVM (36 steering feats, 1750 SVs) | 110 | ~9,000 |
| **Lstm** (BiLSTM-36 + attention, seq 100×12) | **316** | ~3,200 |

Model-only, every method predicts a window in tens-to-hundreds of microseconds — trivially
real-time. But this excludes the feature extraction, which turns out to dominate (2c–2d).

### 2c. Feature-extraction cost per window (the dominant term)

Measured on a representative 300-sample window with the pipeline's own extractors:

| Feature family | ms / window / signal | used by |
|---|---|---|
| statistical + entropy (22 feats: Sample/Shannon entropy, Katz FD, moments, spectral) | **8.1** | SvmA, RF |
| GHM multiwavelet packet (8 band energies) | **3.4** | SvmW, RF |

The statistical/entropy family is expensive because **Sample Entropy is O(n²)** in the window
length. This per-window preprocessing, not the model, is the real cost.

### 2d. End-to-end inference latency (feature extraction + model)

**Why feature extraction counts as inference cost, not build cost.** At *training* time the features
are extracted once over the training set — an offline, one-off cost folded into 2a. But a fielded
real-time detector receives raw steering / lane signals and must extract each **new** window's
features before the model can score it, so per-window extraction *recurs at inference, once per
prediction*. It would be a build-only cost only if every future window could be pre-extracted in
batch, which a streaming detector cannot do. The experiment measured 2b (model-only) because the
features were pre-computed offline into the processed CSVs; **2d is the latency a deployed detector
actually pays per window.** (If a deployment precomputes or incrementally maintains features, its
latency falls back toward 2b.)

Composed from the per-signal extraction cost × the signals each method actually uses, plus its
model inference:

| Method | features extracted | **end-to-end ms / window** |
|---|---|---|
| **RF (fs & nofs)** | full set (~5 signals statistical + 3 wavelet) | **~51** |
| **SvmA** | 2 steering signals × statistical/entropy | **~16** |
| **SvmW** | 1 steering-wheel signal × GHM wavelet | **~3.5** |
| **Lstm** | light std/mean/pred-error (no entropy) + BiLSTM | **~0.8** |

**This inverts the model-only ranking.** RF has the *fastest model* but the *slowest end-to-end*,
because it must compute the whole 165-feature set — including the O(n²) entropy features — before it
predicts, and RF-fs pays the same extraction cost as RF-nofs (feature selection happens after
extraction). Lstm, whose features are light and whose sequence model is only 0.3 ms, is the fastest
end-to-end.

![Processing speed comparison](figures/c1_recorded/fig_processing_speed.png)

*Left: build cost (training + tuning) per cell, log scale — RF-nofs / SvmW ≈ 20–30× RF-fs / Lstm.
Right: per-window inference latency (feature extraction + model), log scale — extraction dominates,
and RF is heaviest end-to-end despite the fastest model.*

### 2e. Deployment reading — accuracy × speed trade-off

Reading the two cost axes against Mixed-regime accuracy (the deployable regime; RF-nofs Mixed is
provisional at n=13):

| Method | Mixed AUROC (rank) | Build (h/cell) | End-to-end infer (ms/window) |
|---|---|---|---|
| **RF (nofs)** | **0.87 (1st)**\* | 23.2 | ~51 |
| **Lstm** | 0.78 (2nd) | **0.8** | **~0.8** |
| **SvmW** | 0.76 (3rd) | 17.4 | ~3.5 |
| **RF (fs)** | 0.73 (4th) | 0.7 | ~51 |
| **SvmA** | 0.56 (5th) | 2.6 | ~16 |

- **No method is ruled out for real-time use.** A KSS/DRT window spans seconds, so even the heaviest
  end-to-end pipeline (RF, ~51 ms) is ~200× faster than real time.
- **Lstm is the cost-efficient (Pareto) choice** — 2nd-highest Mixed AUROC at the **lowest build cost
  and the lowest inference latency** of all five.
- **RF-nofs buys the top accuracy at the highest cost on both axes** (23 h to build, ~51 ms/window to
  run).
- **RF-fs is dominated:** despite the cheapest build, it still pays the full ~51 ms extraction (it
  computes all 165 features before selecting the top-10) yet is only 4th in Mixed accuracy — a small
  model does not buy a fast detector.
- **SvmW** trades a heavy build for light inference; **SvmA** is cheap-ish to run but lowest accuracy.
- **Net:** processing speed favours **Lstm / SvmW at inference** and **RF-fs / Lstm at build time**;
  RF's recorded accuracy edge (especially RF-nofs) comes at the highest cost on *both* axes.

**Caveats.** Training times are wall-clock from a shared, non-isolated machine (GPU/CPU contention),
so they are order-of-magnitude. Model timings are CPU single-thread (Lstm via TF). Feature-extraction
and end-to-end figures use a representative 300-sample window and the stated per-method signal
counts, so they are estimates; a rigorous end-to-end benchmark on isolated hardware with the exact
per-method signal set is a possible follow-up.
