# Vehicle-Based Driver Drowsiness Detection: Class Imbalance and Domain Shift

Code accompanying the paper

> Y. Nakagama, D. Ishii, and K. Yoshizoe,
> **"Class Imbalance and Domain Shift in Vehicle Dynamics-Based Driver Drowsiness Detection,"**
> manuscript under review, 2026. <!-- TODO: replace with journal + DOI link once published -->

The repository benchmarks lightweight driver drowsiness detection (DDD) models built on
**vehicle-dynamics features** (steering, lateral/longitudinal motion) and studies how
**class rebalancing** (SMOTE, random under-sampling, subject-wise SMOTE, balanced RF) and
**cross-subject domain shift** (MMD / Wasserstein / DTW subject distances, in-/out-of-domain
target groups, pooled / source-only / target-only training modes) affect detection
performance. Factor effects are quantified with Sobol sensitivity indices and
non-parametric tests.

| | |
|---|---|
| Code | this repository (MIT License) |
| Derived data & per-run results | Zenodo, DOI: `10.5281/zenodo.24384763` (reserved; resolves once the record is published) (CC BY 4.0) |
| Upstream raw dataset | Aygun et al. (2024), Harvard Dataverse, [doi:10.7910/DVN/HMZ5RG](https://doi.org/10.7910/DVN/HMZ5RG) (CC0 1.0) |

---

## Repository layout

```
config/subjects/        Subject lists and target-group definitions
src/                    Library code
  data_pipeline/          Pre-processing and feature extraction (EEG, vehicle dynamics)
  models/                 Model definitions and training pipelines
  evaluation/             Evaluation framework
  analysis/               Subject distance metrics, correlation, summaries
  utils/                  Shared helpers
scripts/python/         Entry points
  preprocess/preprocess.py
  train/train.py
  evaluation/evaluate.py
  analysis/domain/        Paper tables, Sobol analysis and journal figures
  prepare_zenodo_bundle.py  Builds the Zenodo archive from an explicit allowlist
scripts/hpc/            PBS job scripts used on the JAIST HPC cluster (site-specific)
results/analysis/exp2_domain_shift/figures/
  csv/split2/             Aggregated per-condition metrics used in the paper
  svg/split2/journal_v2/  Final journal figures (PDF)
data/, models/          Not tracked (see data/README.md, models/README.md)
tests/                  Unit tests
DATADICTIONARY.md       Column dictionary and file-naming grammar of the Zenodo bundle
```

## Installation

The experiments of the paper were run with Python 3.10.11; the versions of the main
libraries used for them are pinned in `requirements-lock.txt`.

```bash
git clone https://github.com/YutaroNakagama/vehicle_based_DDD_comparison.git
cd vehicle_based_DDD_comparison
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt -r requirements.txt
```

Run the tests with `pytest`.

## Data

### Option A — use the published derived data (recommended)

Download the bundle from Zenodo (DOI above), verify it, and copy its trees into the
repository root:

```bash
curl -L -o ddd-comparison-bundle-v1.0.0.zip \
  "https://zenodo.org/records/24384763/files/ddd-comparison-bundle-v1.0.0.zip?download=1"
unzip ddd-comparison-bundle-v1.0.0.zip
(cd ddd-comparison-bundle-v1.0.0 && sha256sum -c MANIFEST.sha256)   # verify every file
cp -r ddd-comparison-bundle-v1.0.0/data ddd-comparison-bundle-v1.0.0/results .
```

This populates `data/processed/common/` (feature tables), `results/outputs/evaluation/RF/`
(the evaluation records of the 1,512 evaluations) and `results/analysis/exp2_domain_shift/`
(metric tables, distance matrices, In/Out subject lists and figures).
See `DATADICTIONARY.md` for the meaning of every column and file name.

Trained models and per-run training records are **not** distributed (loading untrusted
pickles is unsafe, and the evaluation records already contain everything reported in the
paper). Re-train the models with the steps below if needed.

### Option B — start from the raw dataset

```bash
curl -L -O -J "https://dataverse.harvard.edu/api/access/dataset/:persistentId/?persistentId=doi:10.7910/DVN/HMZ5RG"
```

Place the extracted files under `data/external/` (see `data/README.md`), then run the
pre-processing step.

## Reproducing the paper

1. **Pre-processing** (raw data only)

   ```bash
   python scripts/python/preprocess/preprocess.py --model common
   ```

2. **Training and evaluation** — one run per factor cell and seed
   (126 cells × 12 seeds = 1,512 runs, RF). The runs were executed on the JAIST
   HAKUSAN cluster (PBS); each run is one job of
   `scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh`
   (see [docs/experiments/reproducibility.md](docs/experiments/reproducibility.md)).
   The job scripts are specific to that cluster: replace the placeholders `hpcuser`
   (account name), `/home/hpcuser/...` (project path), `user@example.com`
   (notification address) and the queue names before use.

3. **Aggregation, statistics and figures** — works directly on the CSVs tracked in
   this repository (or on the Zenodo bundle). Output goes to
   `results/analysis/exp2_domain_shift/figures/`.

   ```bash
   python scripts/python/analysis/domain/sensitivity_analysis_exp2.py
   python scripts/python/analysis/domain/compute_paper_table_may2026.py
   python scripts/python/analysis/domain/paper_statistics.py
   python scripts/python/analysis/domain/plot_journal_figures_v2.py
   python scripts/python/analysis/domain/plot_ofat_supplement.py
   ```

### Scripts for reproducing the paper

| Step | Script | Produces |
|---|---|---|
| Pre-processing | `scripts/python/preprocess/preprocess.py` | Feature tables in `data/processed/common/` |
| Subject grouping | `scripts/python/analysis/domain/regenerate_split2_normalized.py` | Distance matrices and In/Out subject lists |
| Training | `scripts/python/train/train.py` | Per-run training records and models |
| Evaluation | `scripts/python/evaluation/evaluate.py` | Per-run evaluation records |
| Cluster jobs | `scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh`, `scripts/hpc/launchers/launch_paper_domain_split2.sh`, `scripts/hpc/launchers/launch_exp2_mixed.sh`, `scripts/hpc/launchers/gen_exp2_seed12_joblist.py`, `scripts/hpc/launchers/launch_exp2_seed12_fill.sh` | The 1,512 runs |
| Metric tables | `scripts/python/analysis/domain/regenerate_all_conditions_may2026.py` | `figures/csv/split2/*_NEW_may2026/*.csv`, from the evaluation records |
| Performance table | `scripts/python/analysis/domain/compute_paper_table_may2026.py` | Performance per rebalancing method and training mode |
| Sobol indices | `scripts/python/analysis/domain/sensitivity_analysis_exp2.py` | Total and first-order indices with bootstrap intervals (input of the Sobol-index figure) |
| Other statistics | `scripts/python/analysis/domain/paper_statistics.py` | S<sub>{M,R}</sub> with its interval, M–R dominance ratio, Cliff's δ, Spearman's ρ (M and D), seed convergence |
| Figures | `scripts/python/analysis/domain/plot_journal_figures_v2.py` | Sobol-index figure and seed-convergence figure |
| Figures | `scripts/python/analysis/domain/plot_ofat_supplement.py` | Profile plots of the four factors |
| Feature statistics | `scripts/python/analysis/domain/vehicle_feature_distance_analysis.py` | Intraclass correlation and effective dimensionality of the features (needs `data/processed/common/`) |

## Citation

If you use this code or data, please cite the paper and the Zenodo record
(see also [`CITATION.cff`](CITATION.cff)), together with the upstream dataset:

```bibtex
@misc{nakagama2026ddd_data,
  author    = {Nakagama, Yutaro and Ishii, Daisuke and Yoshizoe, Kazuki},
  title     = {Driver Drowsiness Detection Comparison Dataset},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.24384763}
}

@misc{aygun2024dataset,
  author    = {Aygun, Ayca and Blaney, Giles and Haga, Zachary and McWilliams, Thomas
               and Mertens, Julia and de Ruiter, Jan Peter and Ward, Nathan and Scheutz, Matthias},
  title     = {Multi-modal data acquisition platform for behavioral evaluation},
  year      = {2024},
  publisher = {Harvard Dataverse},
  doi       = {10.7910/DVN/HMZ5RG}
}
```

## License

Source code: MIT. Data and figures: CC BY 4.0. Details and third-party notices in
[`LICENSE`](LICENSE).

## Acknowledgements

This work was supported by JST CREST Grant Number JPMJCR23M1 and JSPS KAKENHI Grant
Number 23K11969. Experiments were run on the HAKUSAN supercomputer at JAIST.
