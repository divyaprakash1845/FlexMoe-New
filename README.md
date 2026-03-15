## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/divyaprakash1845/FlexMoE-modified
cd FlexMoE-modified

```

### 2. Install libraries

```bash
pip install -q "numpy<2.0.0" "setuptools<70.0.0" wheel ninja scikit-learn pandas scipy mne torch

```

### 3. Compile the FastMoE Engine 

```bash
cd /content
rm -rf fastmoe
git clone https://github.com/laekov/fastmoe.git
cd fastmoe
CUDA_HOME=/usr/local/cuda USE_NCCL=0 python setup.py install

```

---

## 🚀 Execution Steps

### Step 1: EEG Artifact Removal (MATLAB)

Cleans raw `.edf` files using Independent Component Analysis (ICA).

* **Action:** Open MATLAB and navigate to the `FlexMoE-modified` folder.
* **Update Paths:** Open `clean_raw_eeg.m` and set `eegpath` (your local EEGLAB folder) and `rootDir` (your `raw_data` folder).
* **Command:** click **Run**.

### Step 2: Multi-Modal Fusion (Python)

```bash
cd /content/FlexMoE-modified
python preprocess.py

```

### Step 3: Mixture-of-Experts Training (Python)

```bash
python train.py

```

---

## 📁 Required Folder Structure

```text
Workspace/
├── raw_data/                 <-- Raw lab folders (7873, etc.)
├── fastmoe/                  <-- (Compiled in Setup Step 3)
└── FlexMoE-modified/        <-- (This Repository)
    ├── clean_raw_eeg.m
    ├── preprocess.py
    ├── dataset.py
    ├── moe_module.py        
    ├── model.py
    ├── train.py
    └── README.md

```

---
