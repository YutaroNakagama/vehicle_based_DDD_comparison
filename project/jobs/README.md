# jobs/ directory

This directory contains **PBS job scripts** and related logs for running experiments on HPC clusters.

---

## Example Job Scripts

Representative PBS job scripts are provided as `*.sh.example`:

- `pbs_preprocess.sh.example` → Preprocessing step
- `pbs_train.sh.example` → Model training
- `pbs_finetune.sh.example` → Fine-tuning
- `pbs_evaluate.sh.example` → Evaluation
- `pbs_analyze.sh.example` → Analysis (distances, correlations)

### Usage

Copy an example script and rename it to `.sh`:

```bash
cp jobs/pbs_train.sh.example jobs/pbs_train.sh
````

Then edit:

* **Environment** (conda env, modules, etc.)
* **Paths** (dataset, output directories)
* **PBS options** (queue, nodes, memory, walltime)

### Notes

* Custom variants (e.g., `pbs_rank_10.sh`, `pbs_only10_6groups.sh`) should be created locally and will **not** be tracked by Git.
* Logs under `jobs/log/` are ignored automatically.
* Only representative `.example` scripts are version-controlled to keep the repository clean and reproducible.

## Notes on Job Scripts

- Only representative PBS job scripts are version-controlled as `.example` files:
  - `pbs_preprocess.sh.example`
  - `pbs_train.sh.example`
  - `pbs_finetune.sh.example`
  - `pbs_evaluate.sh.example`
  - `pbs_analyze.sh.example`

- To create an actual job script, copy and edit:
  ```bash
  cp pbs_train.sh.example pbs_train.sh
  vi pbs_train.sh
````

* Variations (e.g., rank-based, only10 groups) should be managed locally and are excluded from version control.

