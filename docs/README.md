# docs/ directory

This directory contains the **Sphinx documentation** for the Vehicle-Based DDD Comparison project.

---

## Build Instructions

### 1. Install dependencies
```bash
cd docs
pip install -r requirements.txt
````

### 2. Build HTML documentation

```bash
make html
```

The generated documentation will be available at:

```
docs/_build/html/index.html
```

---

## Content

* **analysis.rst** → Analysis tools (distance, correlation, summaries)
* **data\_pipeline.rst** → Preprocessing pipelines & feature extraction
* **evaluation.rst** → Evaluation framework
* **models.rst** → Model architectures & pipelines
* **utils.rst** → Utility modules (I/O, visualization, domain generalization)
* **bin/** → Command-line entry points (`preprocess`, `train`, `evaluate`, etc.)

---

## Notes

* Use `autodoc` to automatically extract docstrings from `project/src/`.
* Style can be customized in `_static/` and `conf.py`.
* To add new modules, update `index.rst` and rebuild.

````

---

# 📄 `misc/README.md`（任意）

```markdown
# misc/ directory

This directory contains utility scripts and configuration files used in experiments.

---

## Files
- **make_pretrain_group.py** → Generate pretraining subject groups  
- **make_target_groups.py** → Generate target subject lists  
- **run.sh** → Example shell script for launching jobs  
- **unzip.sh** → Utility to unpack dataset archives  
- **filelist.txt** → File listing for processed datasets  
- **subject_list.txt** → Master list of subjects  
- **target_groups.txt** → Target subject group definitions  
- **requirements.txt** → Runtime dependencies for training & evaluation  

---

## Notes
These scripts are not part of the main pipelines but help organize datasets and experiments.
