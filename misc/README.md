# misc/ directory

This directory contains **experiment configurations and ad-hoc analysis scripts**.  
They are not part of the main pipeline, but support dataset preparation, group definition, and result inspection.

---

## Files

### Subject and Group Definitions
- **subject_list.txt** → Master list of all subjects in the dataset  
- **general_subjects.txt** → Subjects used for general training (pretrain groups)  
- **target_groups.txt** → Target subject groups for evaluation/finetuning  

### Analysis Scripts (`analysis/`)
One-off scripts for aggregating results and plotting summary metrics.

### Experiment Configs (`config/`)
- **subject_list.txt**, **target_groups.txt**, **general_subjects.txt**  
- **rank_names_*.txt** files  
- **filelist.txt** → File index (used for dataset integrity checks or preprocessing pipeline input)  
- **requirements.txt** → Python dependencies for preprocessing, training, and evaluation  

---

## Notes
- These scripts/files are mainly **helpers** for reproducibility of experiments.  
- The subject/group lists in `misc/config/` are referenced by preprocessing and training scripts under `bin/`.  
- When adding new subjects or groups, update the relevant list file here.  

---

## Tips
- Keep subject IDs consistent with dataset file naming (e.g., `S0101_1`).  
- If you run large-scale experiments, consider versioning `misc/` so that group splits are reproducible.  
```

