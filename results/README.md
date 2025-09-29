# results/

This directory stores all experiment results.  
The structure follows a consistent policy to ensure reproducibility, clarity, and reusability.

## Directory Structure

- **metrics/**  
  Per-job raw performance metrics (CSV/JSON). Organized by job ID.  
  Example: `metrics/13924040/metrics_RF.csv`

- **predictions/**  
  Per-job prediction outputs (ROC, PR, CM; CSV + PNG).  
  Example: `predictions/13924040/roc_test_RF.png`

- **distances/**  
  Per-job distance matrices and visualizations (DTW, MMD, Wasserstein).  
  Example: `distances/13924040/dtw_matrix.npy`

- **ranks/**  
  Per-job ranking results (mean, std, top10, top20).  
  Example: `ranks/13924040/top10/wasserstein_mean_high.txt`

- **tables/**  
  Global summary and comparison tables (CSV only).  
  - Single-job summaries include job ID.  
    Example: `summary_27cases_13924040.csv`  
  - Multi-job comparisons include date.  
    Example: `compare_40cases_aucwide_20250929.csv`  
  - Global summaries also include date.  
    Example: `summary_all_metrics_20250929.csv`

- **figures/**  
  Global visualization results (PNG only).  
  - 1:1 correspondence with files in `tables/` (same basename, different extension).  
  - Example:  
    - `tables/compare_40cases_aucwide_20250929.csv`  
    - `figures/compare_40cases_aucwide_20250929.png`

## Naming Conventions

- **Single-job summary:**  
  `summary_<target>_<jobID>.csv / .png`

- **Multi-job comparison:**  
  `compare_<target>_<analysisType>_<date>.csv / .png`

- **Global summary:**  
  `summary_all_<target>_<date>.csv / .png`

## Policy

- **tables/** → numbers (CSV only)  
- **figures/** → visualizations (PNG only)  
- **metrics/, predictions/, distances/, ranks/** → per-job results, organized by job ID  
- No PDF/SVG or other formats are generated — **PNG is the single standard**  
- Job results are never overwritten: each run is placed in its own job ID folder

