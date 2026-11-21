# Scripts Overview

Executable scripts of the project: Python entry points (thin wrappers around `src/`)
and PBS job scripts for the cluster. The scripts needed to reproduce the paper are
listed in the top-level `README.md` ("Scripts for reproducing the paper").

```
scripts/
├── python/
│   ├── preprocess/      preprocess.py — feature extraction
│   ├── train/           train.py — model training
│   ├── evaluation/      evaluate.py — evaluation of a trained model
│   ├── analysis/        post-hoc analysis
│   │   ├── domain/        metric tables, Sobol analysis, statistics and figures of the paper
│   │   └── distance/      inter-subject distance analysis
│   ├── visualization/   plotting helpers
│   └── prepare_zenodo_bundle.py — builds the Zenodo archive from an explicit allowlist
├── hpc/                 PBS job scripts for the JAIST cluster (site-specific)
│   ├── jobs/            job scripts by task (preprocess, train, evaluate, imbalance, domain_analysis)
│   ├── launchers/       batch submission scripts
│   └── lib/             shared shell functions
└── local/               helpers for running the pipeline on a workstation
```

Example:

```bash
python scripts/python/train/train.py --model RF --subject_wise_split
```

The PBS scripts activate the cluster's conda environment, set the threading variables
and call the Python entry points. Replace the account name, project path,
notification address and queue names before using them on another cluster.
