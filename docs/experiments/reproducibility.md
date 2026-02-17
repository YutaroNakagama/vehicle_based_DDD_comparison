# Experiments and Reproducibility

This guide explains how to reproduce the experiments from our research.

## Experiment Overview

The repository supports three main experiment types:

| # | Experiment | Model | Description |
|---|-----------|-------|-------------|
| 1 | **Imbalance Analysis** | RF | Compare oversampling/undersampling strategies |
| 2 | **Domain Shift** | RF | Cross-domain generalization with split2 grouping |
| 3 | **Prior Research Replication** | SvmA, SvmW, Lstm | Replicate baselines with domain split2 |

## Experiment 1: Imbalance Analysis

Compare imbalance handling methods in isolation (pooled training without domain generalization).

**詳細な実験条件一覧:** [conditions/01-imbalance-conditions.md](conditions/01-imbalance-conditions.md)

**Launcher:** `scripts/hpc/launchers/launch_imbalance.sh`

```bash
# 8 methods × 2 seeds = 16 experiments
bash scripts/hpc/launchers/launch_imbalance.sh
```

| Method | Description |
|--------|-------------|
| Baseline | RF with class_weight only |
| SMOTE 0.1 / 0.5 | SMOTE with sampling_ratio |
| SW-SMOTE 0.1 / 0.5 | Subject-wise SMOTE |
| RUS 0.1 / 0.5 | Random Under-Sampling |
| Balanced RF | BalancedRandomForestClassifier |

**Key Parameters:**
- Model: RF
- Seeds: 42, 123
- Optuna trials: 100
- Optimization metric: F2 score

## Experiment 2: Domain Shift (RF, Split2)

Evaluate cross-domain generalization using 2-group domain splitting.

**Launchers:**
- Cross/Within-domain: `scripts/hpc/launchers/launch_paper_domain_split2.sh`
- Multi-domain: `scripts/hpc/launchers/launch_exp2_mixed.sh`

```bash
# Cross/Within-domain (source_only + target_only): 192 jobs
bash scripts/hpc/launchers/launch_paper_domain_split2.sh --dry-run
bash scripts/hpc/launchers/launch_paper_domain_split2.sh

# Multi-domain (mixed): 96 jobs
bash scripts/hpc/launchers/launch_exp2_mixed.sh --dry-run
bash scripts/hpc/launchers/launch_exp2_mixed.sh
```

**Experiment Matrix:**
**詳細な実験条件一覧:** [conditions/02-domain-conditions.md](conditions/02-domain-conditions.md)

| Parameter | Values |
|-----------|--------|
| Distance metrics | mmd, dtw, wasserstein (3) |
| Domain groups | in_domain (44), out_domain (43) |
| Training modes | source_only, target_only, mixed (3) |
| Seeds | 42, 123 (2) |
| Imbalance methods | baseline, smote_plain, smote (=sw_smote), undersample, balanced_rf |
| Target ratios | 0.1, 0.5 (for ratio-based methods) |

**Conditions per job count (8 jobs per distance×domain×mode×seed combo):**
- baseline (1)
- smote_plain × 2 ratios (2)
- smote (subject-wise SMOTE) × 2 ratios (2)
- undersample × 2 ratios (2)
- balanced_rf (1)

**Training Mode Descriptions:**

| Mode | Description | Training Data | Evaluation Data |
|------|-------------|---------------|------------------|
| source_only | Cross-domain | Opposite domain | Target domain |
| target_only | Within-domain | Same domain | Same domain |
| mixed | Multi-domain | All 87 subjects (pooled) | Target domain |

**Total:** 3 × 2 × 3 × 2 × 8 conditions = **288 jobs** (192 cross/within + 96 mixed)

### Condition 命名規則

実験コード内の `CONDITION` パラメータとタグ名・実際の処理の対応:

| CONDITION | タグ prefix | 処理内容 |
|-----------|-------------|----------|
| `baseline` | `baseline_domain_*` | 不均衡対策なし（class_weight のみ） |
| `smote_plain` | `smote_plain_*` | グローバル SMOTE（全被験者プール後に適用） |
| `smote` | `imbalv3_*` | Subject-wise SMOTE（被験者単位で SMOTE 適用） |
| `undersample` | `undersample_rus_*` | Random Under-Sampling |
| `balanced_rf` | `balanced_rf_*` | BalancedRandomForestClassifier（RF のみ） |

## Experiment 3: Prior Research Replication (Split2)

Replicate prior research baselines with domain split2 grouping.

**詳細な実験条件一覧:** [conditions/03-prior-research-conditions.md](conditions/03-prior-research-conditions.md)

**Launchers:**
- Cross/Within-domain: `scripts/hpc/launchers/launch_prior_research_split2.sh`
- Multi-domain: `scripts/hpc/launchers/launch_prior_research_mixed.sh`

```bash
# Cross/Within-domain (source_only + target_only): 504 jobs
bash scripts/hpc/launchers/launch_prior_research_split2.sh --dry-run
bash scripts/hpc/launchers/launch_prior_research_split2.sh

# Multi-domain (mixed): 252 jobs
bash scripts/hpc/launchers/launch_prior_research_mixed.sh --dry-run
bash scripts/hpc/launchers/launch_prior_research_mixed.sh
```

**Experiment Matrix:**

| Parameter | Values |
|-----------|--------|
| Models | SvmW, SvmA, Lstm (3) |
| Distance metrics | mmd, dtw, wasserstein (3) |
| Domain groups | in_domain, out_domain (2) |
| Training modes | source_only, target_only, mixed (3) |
| Seeds | 42, 123 (2) |
| Imbalance methods | model-dependent (see below) |

**Methods per model (all models share the same 4 conditions):**
- SvmW: baseline, smote_plain, smote (=sw_smote), undersample (4 methods → 7 jobs/combo)
- SvmA: baseline, smote_plain, smote (=sw_smote), undersample (4 methods → 7 jobs/combo)
- Lstm: baseline, smote_plain, smote (=sw_smote), undersample (4 methods → 7 jobs/combo)

> **Note:** `balanced_rf` は RF 専用のため exp3 では使用しない。

**Total:** 3 × 2 × 3 × 2 × 7 × 3 models = **756 jobs** (504 cross/within + 252 mixed)

**HPC Resources per model:**

| Model | CPUs | Memory | Walltime (source/target) | Walltime (mixed) |
|-------|------|--------|--------------------------|-------------------|
| SvmW | 8 | 16 GB | 12h (SMOTE: 24h) | 16h (SMOTE: 24h) |
| SvmA | 8 | 32 GB | 24h (SMOTE: 48h) | 30h (SMOTE: 48h) |
| Lstm | 4 | 32 GB | 16h (SMOTE: 24h) | 20h (SMOTE: 24h) |

> **Walltime 注記:** SMOTE 系条件（smote_plain, smote）は Optuna 100 trial の各 trial で
> SMOTE を実行するため、baseline/undersample より大幅に時間がかかる。
> 実験中に判明したため、walltime を増加して再投入した経緯がある。

### デーモンによる自動投入

実験3は全 504 ジョブ（cross/within-domain）をキュー上限内で順次投入するため、
モデルごとにデーモンプロセスを使用している。デーモンは残りジョブリストから
未投入分を取り出し、キュー空き状況に応じて自動的に `qsub` する。

```bash
# デーモン起動例（SvmW 用）
nohup bash scripts/hpc/launchers/auto_resub_svmw.sh &

# デーモン状態確認
ps aux | grep auto_resub
```

デーモンスクリプトはモデルごとに以下が存在する:
- `scripts/hpc/launchers/auto_resub_svmw.sh`
- `scripts/hpc/launchers/auto_resub_svma.sh`
- `scripts/hpc/launchers/auto_resub_lstm.sh`

## HPC Batch Execution

### Submit Jobs

```bash
# Single job
qsub -N job_name -l select=1:ncpus=4:mem=8gb -l walltime=08:00:00 -q SINGLE \
    -v CONDITION=baseline,MODE=source_only,DISTANCE=mmd,DOMAIN=out_domain,SEED=42 \
    scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh

# Bulk submission across multiple queues
bash scripts/hpc/launchers/massive_submit.sh
```

### Monitor Jobs

```bash
# Check queue status
qstat -u $USER

# Count running/queued jobs
qstat -u $USER | grep -c " R "
qstat -u $USER | grep -c " Q "

# View job output
tail -50 scripts/hpc/logs/train/*.OU
```

### Check Results

```bash
# Count result files by model
find results/outputs/evaluation/RF -name "*.json" | wc -l
find results/outputs/evaluation/SvmA -name "*.json" | wc -l

# Check for errors in recent logs
grep -l "Traceback\|ERROR\|KeyError" scripts/hpc/logs/train/*.OU | tail -5
```

## Results Structure

```
results/
├── analysis/
│   └── domain/
│       ├── distance/           # Distance matrices and rankings
│       └── summary/            # Aggregated analysis
├── outputs/
│   ├── evaluation/             # Test metrics (JSON/CSV)
│   │   ├── RF/
│   │   ├── BalancedRF/
│   │   ├── SvmA/
│   │   ├── SvmW/
│   │   └── Lstm/
│   └── training/               # Training metrics
└── README.md
```

## Saved Artifacts

Each experiment saves (model-dependent):

| File | Description | Models |
|------|-------------|--------|
| `{model}_{mode}.pkl` | Trained model (sklearn) | RF, SvmW, SvmA |
| `{model}_*.keras` / `.h5` | Trained model (Keras) | Lstm |
| `scaler_{mode}.pkl` | Feature scaler | All |
| `features_{mode}.json` | Selected features | All |
| `eval_results_*.json` | Evaluation metrics | All |
| `training_history_*.json` | Training history | Lstm |
| `optuna_study_*.pkl` | Optuna study object | RF, SvmW |

**Expected artifact counts per successful job:**

| Model | eval JSON | Model files | Other |
|-------|-----------|-------------|-------|
| RF / BalancedRF | 1 | 4 pkl | scaler, features, study |
| SvmW | 1 | 4 pkl | scaler, features, study |
| SvmA | 1 | 4 pkl | scaler, features |
| Lstm | 1 | ~20 (keras + pkl) | history JSON |

## Reproducibility Checklist

✅ **Fixed random seeds** — Set in config and CLI (`--seed 42`)

✅ **Versioned dependencies** — `requirements.txt` with pinned versions

✅ **Documented preprocessing** — Feature extraction parameters in `src/config.py`

✅ **Saved hyperparameters** — Optuna study objects preserved

✅ **Git versioning** — All code changes tracked

✅ **Thread control** — HPC jobs set `OMP_NUM_THREADS=1` etc. for determinism

✅ **CPU mode** — TensorFlow forced to CPU (`CUDA_VISIBLE_DEVICES=""`)

---

## Related Documents

- [Developer Guide](../architecture/developer_guide.md) — Repository architecture
- [Domain Generalization Pipeline](../architecture/domain_generalization.md) — Domain analysis details
- [Imbalance Methods](../reference/imbalance_methods.md) — Method descriptions
- [Experiment Results](results/) — Historical experiment results
