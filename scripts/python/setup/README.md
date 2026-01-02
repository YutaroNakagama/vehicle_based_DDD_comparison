# Setup and Validation Scripts

This directory contains one-time setup scripts and validation utilities for the Driver Drowsiness Detection project.

## Setup Scripts

### `generate_target_groups.py`
Generates target subject groups from the master subject list.
- **Input**: `config/subjects/subject_list.txt`
- **Output**: `config/subjects/target_groups.txt`
- **Purpose**: Creates groups of 10 subjects each for domain generalization experiments

**Usage**:
```bash
python scripts/python/setup/generate_target_groups.py
```

### `generate_pretrain_group.py`
Generates the pretrain (general) subject list by excluding target subjects.
- **Input**: 
  - `config/subjects/subject_list.txt`
  - `config/subjects/target_groups.txt`
- **Output**: `config/subjects/general_subjects.txt`
- **Purpose**: Creates source domain subject list (complement of first target group)

**Usage**:
```bash
python scripts/python/setup/generate_pretrain_group.py
```

## Validation Scripts

### `check_feature_columns.py`
Validates feature consistency across processed subject CSV files.
- **Purpose**: Ensures all target subjects have consistent feature columns
- **Input**: Processed CSV files in `data/processed/common/`
- **Output**: Console report of feature mismatches

**Usage**:
```bash
python scripts/python/setup/check_feature_columns.py
```

### `check_only_modes.py`
Compares evaluation metrics between different training modes.
- **Purpose**: Validates consistency between "eval-only" and "only-target" training modes
- **Input**: Metric CSV files in `model/common/`
- **Output**: Console comparison of accuracy, precision, recall, F1, AUC, AP

**Usage**:
```bash
python scripts/python/setup/check_only_modes.py
```

### `make_ranked_groups.py`
Generates ranked subject groups based on domain distance matrices.
- **Purpose**: Creates out_domain/mid_domain/in_domain subject groups using MMD, Wasserstein, DTW distances
- **Input**: Distance matrices in `results/analysis/domain/distance/subject-wise/`
- **Output**: Group files in `results/analysis/domain/distance/subject-wise/ranks/`

**Usage**:
```bash
python scripts/python/setup/make_ranked_groups.py --target_size 29
python scripts/python/setup/make_ranked_groups.py --target_size 10  # smaller groups
```

## Notes

- These scripts are typically run once during initial setup or after data preprocessing
- Validation scripts help ensure data quality before training experiments
- Group generation scripts create the subject partitioning used in domain generalization studies
