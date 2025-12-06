# config/ directory

This directory contains **experiment configurations and subject group definitions**.  
They support dataset preparation, group definition, and experiment reproducibility.

---

## Files

### Subject and Group Definitions (subjects/ subdirectory)
- **subjects/subject_list.txt** → Master list of all subjects in the dataset  
- **subjects/general_subjects.txt** → Subjects used for general training (pretrain groups)  
- **subjects/target_groups.txt** → Target subject groups for evaluation/finetuning  

### Experiment Configs
- **../requirements.txt** → Python dependencies (located at project root)  

---

## Notes
- These files are **helpers** for reproducibility of experiments.  
- The subject/group lists in `config/subjects/` are referenced by preprocessing and training scripts.  
- When adding new subjects or groups, update the relevant list file in the subjects/ subdirectory.  

---

## Tips
- Keep subject IDs consistent with dataset file naming (e.g., `S0101_1`).  
- If you run large-scale experiments, consider versioning `config/` so that group splits are reproducible.  
```

