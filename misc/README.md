# 📄 `project/misc/README.md`

```markdown
# misc/ directory

This directory contains **utility scripts, configuration files, and subject lists** that support experiments.  
They are not part of the main pipeline but are useful for dataset preparation, group definition, and analysis.

---

## Files

### Subject and Group Definitions
- **subject_list.txt** → Master list of all subjects in the dataset  
- **general_subjects.txt** → Subjects used for general training (pretrain groups)  
- **target_groups.txt** → Target subject groups for evaluation/finetuning  

### Utility Scripts
- **make_pretrain_group.py** → Generate subject groups for pretraining experiments  
- **make_target_groups.py** → Generate subject groups for target experiments  
- **run.sh** → Example script for launching jobs or preprocessing  
- **unzip.sh** → Helper script to unzip dataset archives into the correct directory  

### Experiment Configs
- **requirements.txt** → Python dependencies for preprocessing, training, and evaluation  
- **filelist.txt** → File index (used for dataset integrity checks or preprocessing pipeline input)  

---

## Notes
- These scripts/files are mainly **helpers** for reproducibility of experiments.  
- The subject/group lists are referenced in:
  - `project/bin/preprocess.py`
  - `project/bin/train.py`
  - `project/bin/analyze.py`
- When adding new subjects or groups, update the relevant list file here.  

---

## Tips
- Keep subject IDs consistent with dataset file naming (e.g., `S0101_1`).  
- If you run large-scale experiments, consider versioning `misc/` so that group splits are reproducible.  
