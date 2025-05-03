# Vehicle-Based DDD Comparison

## Overview
This repository focuses on benchmarking **lightweight Driver Drowsiness Detection (DDD) models** using **vehicle-based features**. The goal is to compare the efficiency and performance of various lightweight ML models.

## Dataset
We utilize the **open dataset** from _Multi-modal Data Acquisition Platform for Behavioral Evaluation_ (Aygun et al., 2024). The dataset contains:
- Vehicle-based features (steering, acceleration, etc.)
- EEG signals
- Physiological data (heart rate, GSR, etc.)
- Labels for driver drowsiness states

**DOI:** [10.7910/DVN/HMZ5RG](https://doi.org/10.7910/DVN/HMZ5RG)
<!-- need to explain how to download complete dataset via API -->

<!--
### Preprocessing
- Vehicle-based signals are filtered and normalized.
- EEG data is transformed into meaningful frequency components.
- Labels are categorized into drowsiness levels based on predefined thresholds.

## ⚙️ Setup
To set up the environment, install the required dependencies:

```sh
pip install -r requirements.txt
```

Recommended Python version: **3.8+**
-->

## Model Comparison
We compare different lightweight models optimized for efficiency:

<!--
| Model | Pruning | Quantization | Params | Accuracy |
|--------|---------|-------------|--------|----------|
| Baseline DNN | ❌ | ❌ | 1M | 85% |
| Pruned DNN | ✅ | ❌ | 500K | 84% |
| Quantized DNN | ❌ | ✅ | 250K | 83% |
| Pruned + Quantized | ✅ | ✅ | 200K | 82% |
-->

### Compared DDD Approaches
We compare our models against the following established DDD methods. 
1. **Zhao et al. (2009)** - Detecting driver’s drowsiness using multiwavelet packet energy spectrum. [DOI: 10.1109/CISP.2009.5301253](http://dx.doi.org/10.1109/CISP.2009.5301253)
2. **Arefnezhad et al. (2019)** - Driver drowsiness detection using steering wheel data and adaptive neuro-fuzzy feature selection. [DOI: 10.3390/s19040943](https://doi.org/10.3390/s19040943)
3. **Wang et al. (2022)** - Driver distraction detection based on vehicle dynamics using naturalistic driving data. [DOI: 10.1016/j.trc.2022.103561](http://dx.doi.org/10.1016/j.trc.2022.103561)

## Usage
### Install dependencies
Make sure you have Python 3.10 installed. Then, install the required packages:
```sh
pip install -r requirements.txt
```

### Prepare Dataset
setup the public dataset as follows:
```sh
├───dataset
│   └───mdapbe
│       ├───subject_list.txt
│       └───physio
│           ├───S0101
│           │   ├───SIMlsl_S0101_1.mat
│           │   ├───SIMlsl_S0101_2.mat
│           │   ├───EEG_S0101_1.mat
│           │   ├───EEG_S0101_2.mat
│           │   ├───      :
│           │  
│           ├───S0103
│           ├───S0105
│           │     :
│           │     :
│           ├───S0211
│           ├───S0212
│           └───S0213
└───vehicle_based_DDD_comparison
    └───project
        ├───bin
        ├───data
        │   ├───interim
        │   └───processed
        ├───model
        └───src
```

### Step 1: Preprocess Data

```sh
cd project
python bin/preprocess.py --model [common | SvmA | SvmW | Lstm] [--jittering]
```

* `--model`: Required. Specifies the preprocessing logic for the selected model.
* `--jittering`: Optional. Applies jittering-based data augmentation.

---

### Step 2: Train a Model

```sh
python bin/train.py \
    --model [RF | SvmA | SvmW | Lstm] \
    [--domain_mixup] \
    [--coral] \
    [--vae] \
    [--sample_size N] \
    [--seed N] \
    [--tag TAG] \
    [--subject_wise_split] \
    [--feature_selection rf|mi]
```

* `--model`: Required. Specifies the model architecture to train.
* `--domain_mixup`: Optional. Enables Domain Mixup interpolation.
* `--coral`: Optional. Applies CORAL for domain alignment.
* `--vae`: Optional. Enables VAE-based feature augmentation.
* `--sample_size`: Optional. Randomly samples N subjects from the dataset.
* `--seed`: Optional. Sets the random seed (default: 42).
* `--tag`: Optional. Adds a suffix tag to identify model variants.
* `--subject_wise_split`: Optional. Uses subject-wise splitting to avoid leakage.
* `--feature_selection`: Optional. Chooses feature selection method: `rf` (default) or `mi`.

---

### Step 3: Evaluate a Model

```sh
python bin/evaluate.py \
    --model [RF | SvmA | SvmW | Lstm] \
    [--tag TAG] \
    [--sample_size N] \
    [--seed N] \
    [--subject_wise_split]
```

* `--model`: Required. Specifies which trained model to evaluate.
* `--tag`: Optional. Used to identify specific model variants.
* `--sample_size`: Optional. Limits the number of subjects to evaluate.
* `--seed`: Optional. Sets the random seed for consistent sampling (default: 42).
* `--subject_wise_split`: Optional. Enables subject-wise evaluation.

<!--
## Results & Visualization
The model performance is evaluated using **AUC, accuracy, and inference time**. Results are plotted for better interpretability.

```sh
python scripts/plot_results.py
```
## 📜 License
This project is released under the **MIT License**.

## Credits
- Open dataset by Aygun et al. (2024) [DOI: 10.7910/DVN/HMZ5RG]
- Developed for research on efficient DDD models
- Compared with existing DDD methods by Arefnezhad et al. (2019), Zhao et al. (2009), and Wang et al. (2022)
-->

