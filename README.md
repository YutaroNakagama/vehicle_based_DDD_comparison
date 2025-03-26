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

### Go to Project directory
```sh
cd project
```

### Prepare data
```sh
python bin/preprocess.py --model [model name: SvmW/SvmA/Lstm/common]
```

### Training a Model
```sh
python bin/train.py --model [model name: SvmW/SvmA/Lstm/RF]
```

### Evaluating a Model
```sh
python bin/evaluate.py --model [model name: SvmW/SvmA/Lstm/RF]
```

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

