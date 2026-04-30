# Experiment 3 Analysis Index

Operational and analytical notes for the prior-research replication
experiment (SvmW / SvmA / Lstm × distance × domain × 15 seeds, with both
`domain_train` and `mixed` evaluation modes).

| Doc | What it covers |
|---|---|
| [operations_log.md](operations_log.md) | HPC re-submission history, every known issue and its resolution, current submitter / auto-resubmit pipeline, per-queue QOS caps, throughput observations. **Read first when something looks wrong.** |
| [optimization_methods.md](optimization_methods.md) | Per-model hyperparameter optimization scheme (SvmW Optuna, SvmA PSO, Lstm fixed architecture). Records the n_trials choice for the 15-seed expansion. |

## Quick Reference

- **Conditions:** [`../../conditions/03-prior-research-conditions.md`](../../conditions/03-prior-research-conditions.md)
- **Top-level results doc:** [`../03-prior-research-results.md`](../03-prior-research-results.md)
- **Models architecture:** [`../../architecture/prior_research.md`](../../architecture/prior_research.md)

## Where the Pipeline Lives

| Layer | Path |
|---|---|
| PBS wrapper (CPU) | [`scripts/hpc/jobs/train/pbs_prior_research_unified.sh`](../../../../scripts/hpc/jobs/train/pbs_prior_research_unified.sh) |
| PBS wrapper (GPU, dt) | [`scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh`](../../../../scripts/hpc/jobs/train/pbs_prior_research_unified_gpu.sh) |
| PBS wrapper (CPU, mixed) | [`scripts/hpc/jobs/train/pbs_prior_research_split2.sh`](../../../../scripts/hpc/jobs/train/pbs_prior_research_split2.sh) |
| PBS wrapper (GPU, mixed) | [`scripts/hpc/jobs/train/pbs_prior_research_split2_gpu.sh`](../../../../scripts/hpc/jobs/train/pbs_prior_research_split2_gpu.sh) |
| Submitters | [`scripts/hpc/launchers/submit_*.sh`](../../../../scripts/hpc/launchers/) |
| Auto-resubmit loops | [`scripts/hpc/launchers/auto_resubmit_*.sh`](../../../../scripts/hpc/launchers/) |
| Eval-only retry helper | [`scripts/hpc/launchers/eval_retry_inflight_svma.sh`](../../../../scripts/hpc/launchers/eval_retry_inflight_svma.sh) |
| Metric collection / plots | [`scripts/python/analysis/domain/collect_split2_prior_research_metrics.py`](../../../../scripts/python/analysis/domain/collect_split2_prior_research_metrics.py) |

## Job-Name Decoder

`<MM>_<CC>_<dl><DM>_<TT>_r<R>_s<SEED>` where:

- `MM`: `Sv` SvmA-dt · `Sw` SvmW-mx · `Sa` SvmA-mx · `Ls` Lstm-GPU · `Lc` Lstm-CPU-hedge · `Lm` Lstm-mx-CPU
- `CC`: `ba` baseline · `iv` smote (subjectwise) · `sm` smote_plain · `un` undersample_rus
- `dl`: distance — `m` mmd · `d` dtw · `w` wasserstein
- `DM`: domain — `i` in_domain · `o` out_domain
- `TT`: `dt` domain_train · `mx` mixed
- `r<R>`: target ratio (omitted for baseline / mx baseline)
- `s<SEED>`: random seed (15 canonical seeds — see conditions doc)

## Known-Issues Quick List

Sorted by detection date; details + fix linked in [operations_log.md](operations_log.md).

1. Lstm `seq_len` typo (resolved old-split2 only)
2. SMOTE walltime exceeded (walltimes increased)
3. Eval filename overwrite — `eval_type` suffix added
4. Old split2 source/target double-training (replaced by `domain_train`)
5. Lstm cross-eval silent failures (re-submitted)
6. CUDA libs — `hpc_sdk/22.2` removed (LD_LIBRARY_PATH manual fix)
7. PBS_JOBID format breaks save/load path (made purely numeric)
8. CUDA path differs login vs compute node (use `module load cuda/12.8u1`)
