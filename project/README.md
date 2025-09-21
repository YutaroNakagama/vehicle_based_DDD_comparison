# jobs/ directory

This directory contains **PBS batch job scripts** and corresponding output logs for cluster processing.  
These scripts automate preprocessing, training, evaluation, and analysis on HPC environments.

---

## Structure
```

project/jobs/
├── pbs\_distance\_pipeline.sh    # End-to-end distance → correlation analysis
├── pbs\_only10\_6groups.sh       # Array job: isolate 10 subjects
├── pbs\_finetune\_6groups.sh     # Array job: pretrain + finetune strategy
├── pbs\_data\_preprocess.sh      # Multi-process preprocessing
├── ...                         # Other PBS scripts
└── \*.oXXXX / \*.eXXXX           # Output logs (stdout/stderr)

````

---

## Notes
- Always submit jobs **from `project/`** so that `PBS_O_WORKDIR` points to the correct project root.
- Scripts typically rely on `misc/requirements.txt` for dependencies.  
- Hidden parallelism is suppressed using:
  ```bash
  export OMP_NUM_THREADS=1
````

---

## How to use

### 1. Submit a single job

```bash
qsub jobs/pbs_distance_pipeline.sh
```

### 2. Submit array jobs

```bash
qsub -J 1-6 jobs/pbs_only10_6groups.sh
qsub -J 1-6 jobs/pbs_finetune_6groups.sh
```

### 3. Check job logs

```bash
ls jobs/*.o*
tail -n 50 jobs/12345.o
```

---

## Output

* Logs (`*.oXXXX`, `*.eXXXX`) are stored under `project/jobs/`
* Results are stored under:

  * `results/` → evaluation metrics, distance matrices, CSVs
  * `models/` → trained models, scalers, feature metadata
  * `figures/` → visualizations (plots, heatmaps, diagrams)
  * `logs/` → runtime logs

---

## HPC Environment

These scripts assume a **general PBS/Torque environment** with:

* Job submission via `qsub`
* Resource specification with `#PBS -l select=...`
* Array jobs via `qsub -J`

If your cluster differs (e.g., SLURM), adapt the submission lines accordingly.

