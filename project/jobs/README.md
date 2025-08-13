
# jobs/ directory

This directory contains PBS batch job scripts and output logs for cluster processing.

- Typical scripts:
  - `pbs_distance_pipeline.sh`: end-to-end distance → correlation pipeline
  - `pbs_only10_6groups.sh`, `pbs_finetune_6groups.sh`: array jobs for only10 / finetune
  - `pbs_data_preprocess.sh`: multi-process preprocessing
- Output logs (e.g., `*.o12345`, `*.e12345`) are generated per job submission.

## Notes
- Submit jobs **from `project/`** so that `PBS_O_WORKDIR` points to the project root.
- Scripts use `misc/requirements.txt` if available; otherwise they install minimal deps.
- Hidden parallelism is suppressed via `OMP_NUM_THREADS=1`, etc.

## How to use

1. Edit the target `.sh` (e.g., `pbs_distance_pipeline.sh`) if you need to change paths or environment names.
2. Submit with PBS:
   ```bash
   qsub jobs/pbs_distance_pipeline.sh
   # or array examples:
   qsub -J 1-6 jobs/pbs_only10_6groups.sh
   qsub -J 1-6 jobs/pbs_finetune_6groups.sh
   ```
3. Check logs under `project/jobs/` after execution.
