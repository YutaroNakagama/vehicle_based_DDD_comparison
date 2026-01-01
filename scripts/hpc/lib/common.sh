#!/bin/bash
# common.sh
# Common functions and environment setup for domain analysis HPC jobs

# Setup conda environment and Python paths
setup_environment() {
  local project_root="${1:-/home/s2240011/git/ddd/vehicle_based_DDD_comparison}"
  
  export PATH=~/conda/bin:$PATH
  source ~/conda/etc/profile.d/conda.sh || true
  conda activate python310 || { echo "[ERROR] conda env failed"; exit 1; }
  
  export PYTHONPATH="${project_root}:${PYTHONPATH:-}"
  export MPLBACKEND=Agg
  export PYTHONNOUSERSITE=1
}

# Setup thread limits for parallel processing
setup_thread_limits() {
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1
  export BLIS_NUM_THREADS=1
  export NUMBA_NUM_THREADS=1
}

# Log job information
log_job_info() {
  echo "[INFO] Job started at $(date)"
  echo "[INFO] Job ID: ${PBS_JOBID:-unknown}"
  echo "[INFO] Array Index: ${PBS_ARRAY_INDEX:-N/A}"
  echo "[INFO] Working directory: $(pwd)"
}

# Log job completion
log_job_complete() {
  echo "[INFO] Job finished at $(date)"
}
