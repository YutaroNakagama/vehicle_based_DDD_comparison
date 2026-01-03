# Domain Analysis HPC Job Scripts

PBS job scripts for domain-based driver drowsiness detection experiments.

## Scripts Overview

| Script | Description |
|--------|-------------|
| `train_domain_ranking.sh` | Train models using domain-ranked subject splits |
| `eval_domain_ranking.sh` | Evaluate trained models on test data |

## Usage

### Training with Domain Ranking

```bash
# Submit training job
qsub -v RANKING=knn,DOMAIN_LEVEL=out_domain,MODE=source_only train_domain_ranking.sh
```

**Environment Variables:**
- `RANKING`: Ranking method (knn, lof, mean)
- `DOMAIN_LEVEL`: Domain level (in_domain, out_domain)
- `MODE`: Training mode (source_only, target_only, pooled)
- `SEED`: Random seed (default: 42)

### Evaluation

```bash
# Submit evaluation job (after training completes)
qsub -v TRAIN_JOB_ID=12345678 eval_domain_ranking_v3.sh
```

## Resource Requirements

| Script | CPUs | Memory | Walltime | Queue |
|--------|------|--------|----------|-------|
| train_domain_ranking | 4-8 | 8-16GB | 10-12h | SINGLE/DEFAULT |
| eval_domain_ranking | 4 | 8GB | 2h | SINGLE |

## Output Locations

- **Models:** `models/<ModelName>/<JobID>/`
- **Evaluation Results:** `results/outputs/training/<ModelName>/<JobID>/`
- **Logs:** `scripts/hpc/logs/`

## Related Launchers

See `scripts/hpc/launchers/` for automated job submission scripts.
