# SAFIR: Prior-Informed Active Reconstruction of Drug Combination Dose–Response Landscapes under Limited Experimental Budgets

This repository provides the implementation of SAFIR, a prior-informed active framework for predicting drug combination dose–response landscapes under limited experimental budgets.

SAFIR integrates drug molecular structures represented by graphlet fingerprints, cell-line functional dependency profiles from CRISPR Gene Effect data, dose information, single-agent responses, and sparse target-combination observations within a reference–residual architecture.

By combining transferable prior information with newly acquired combination responses, SAFIR progressively improves predictions of unmeasured dose positions and dynamically selects informative measurements through uncertainty-guided acquisition.

---

# Repository Structure

```text
SAFIR/
├── README.md
├── requirements.txt
├── code.ipynb
├── data/
│   ├── O'Neil/
│   │   ├── O'Neil.csv
│   │   └── Graphlet_features_6_raw.csv
│   ├── NCI-ALMANAC/
│   │   ├── NCI_ALMANAC_3x5_SAFIR_input_Graphlet_CRISPR.csv
│   │   └── Graphlet_features_6_raw.csv
│   ├── DrugComb/
│   │   ├── DrugComb_6x6_SAFIR_input_Graphlet_CRISPR.csv
│   │   ├── DrugComb_10x10_SAFIR_input_CRISPR8_block.csv
│   │   ├── 6x6/
│   │   │   └── Graphlet_features_6_raw.csv
│   │   └── 10x10/
│   │       └── Graphlet_features_6_raw.csv
│   └── Shared/
│       └── Model.csv
└── results/
```

---

# Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended environment:

- Python >= 3.9
- PyTorch
- pandas
- numpy
- scikit-learn
- scipy

---

# Data Preparation

## Drug Molecular Features

Drug structures are represented using graphlet fingerprints.

Each dataset contains an independent graphlet vocabulary because datasets are modeled separately.

Required file:

```text
Graphlet_features_6_raw.csv
```

Graphlet features are standardized during training using statistics estimated only from training drugs.

---

## Cell-Line Functional Features

SAFIR uses CRISPR Gene Effect profiles from the DepMap project.

Required files:

```text
data/Shared/

├── Model.csv
└── CRISPRGeneEffect.csv
```

`Model.csv` is included in this repository.

Due to the large file size, `CRISPRGeneEffect.csv` is not included in this repository.

The file can be downloaded from the official DepMap data portal:

https://depmap.org/portal/data_page/

After downloading, place the file as:

```text
SAFIR/
└── data/
    └── Shared/
        ├── Model.csv
        └── CRISPRGeneEffect.csv
```

The preprocessing workflow is:

```text
cell-line name
        ↓
Model.csv alias matching
        ↓
DepMap ModelID
        ↓
CRISPRGeneEffect.csv
        ↓
training-fold-only gene selection
        ↓
Top 1000 CRISPR features
        ↓
normalization
```

For each cross-validation fold, CRISPR feature selection and preprocessing are performed using training cells only.

---

# Running the O'Neil Five-Fold Experiment

The main reproducibility notebook is:

```text
code.ipynb
```

The workflow includes:

1. Loading drug combination landscapes.
2. Matching cell lines to DepMap ModelIDs.
3. Constructing Graphlet and CRISPR features.
4. Performing five-fold matrix-level cross-validation.
5. Training SAFIR under sparse observation states.
6. Evaluating reconstruction performance.

---

# Evaluation

Performance is evaluated using:

- RMSE
- MAE
- Pearson correlation coefficient (PCC)
- Spearman correlation coefficient

Results are summarized as:

```text
mean ± standard deviation
```

across five held-out test folds.

---

# Active Acquisition

SAFIR evaluates sequential dose selection through uncertainty-guided acquisition.

The acquisition process:

```text
Initial sparse observations
        ↓
Landscape prediction
        ↓
Ensemble uncertainty estimation
        ↓
Select uncertain dose positions
        ↓
Reveal new responses
        ↓
Update observation state
        ↓
Reconstruct remaining landscape
```

Model parameters remain fixed during acquisition; only the observation state is updated.

---

# Supported Datasets

| Dataset | Dose grid |
|---|---|
| O'Neil | 4 × 4 |
| NCI-ALMANAC | 3 × 5 |
| DrugComb 6×6 | 6 × 6 |
| DrugComb 10×10 | 9 × 9 non-zero combination region |

Each dataset is processed independently.

---

# Reproducibility Notes

- Complete dose-response matrices are treated as hidden ground truth during active acquisition simulation.
- Cross-validation splitting is performed at the complete drug pair–cell line matrix level.
- Dose positions from the same landscape are kept in the same partition.
- Feature preprocessing is fitted only on training data.
- Drug structure features and CRISPR features are not pooled across datasets.

---

# Citation

If you use this code, please cite:

```text
SAFIR: Prior-Informed Active Reconstruction of Drug Combination
Dose–Response Landscapes under Limited Experimental Budgets
```
