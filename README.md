# SAFIR

Official code and data package for the manuscript:

**SAFIR: Prior-Informed Active Reconstruction of Drug Combination Dose-Response Landscapes under Limited Experimental Budgets**

Wei Xia, Huanhuan Du, Yayu Tian, and Xuan He

SAFIR reconstructs two-dimensional drug-combination dose-response landscapes from transferable drug/cell/dose/single-agent information and sparse target-matrix observations. The released O'Neil workflow uses a single-agent-informed reference, an observation-conditioned residual correction, Random Mask-K training, training-only drug-swap augmentation, and ensemble-disagreement acquisition.

## Repository layout

```text
SAFIR/
├── README.md
├── requirements.txt
├── requirements_full.txt
├── environment.yml
├── CITATION.cff
├── check_data.py
├── run_oneil.py
├── src/safir/
│   ├── config.py       # paths, hyperparameters, active-acquisition settings
│   ├── utils.py        # reproducibility, name matching, metrics, masks
│   ├── data.py         # data loading, 17-D point features, 4x4 blocks, fixed split
│   ├── model.py        # released SAFIR reference-residual network
│   ├── losses.py       # reconstruction + local-difference + residual losses
│   ├── evaluation.py   # fixed-budget prediction and summary metrics
│   └── training.py     # ensemble training and uncertainty-guided acquisition
├── data/
│   ├── O'Neil/
│   ├── NCI-ALMANAC/
│   └── DrugComb/
├── notebooks/
│   └── SAFIR_original_O'Neil.ipynb
└── results/
```

The original all-in-one notebook is retained under `notebooks/` for provenance. The executable code in `src/safir/` is split into the minimal functional modules listed above.

## O'Neil file rename

The uploaded file `processed_combination_response_r095_clean(2).csv` has been renamed exactly as requested to:

```text
data/O'Neil/O'Neil.csv
```

All released code paths now reference this new name.

## Environment and installation

The experiments were run in the `safir` environment with **Python 3.10.20**. The principal package versions used by the released modular workflow were:

- NumPy 2.2.6
- pandas 2.3.3
- scikit-learn 1.7.2
- PyTorch 2.7.1 (the original environment used the CUDA 11.8 build, `2.7.1+cu118`)

For the minimal released workflow:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

Alternatively, a Conda environment can be created from:

```bash
conda env create -f environment.yml
conda activate safir
```

`requirements_full.txt` records the full package environment exported from the original SAFIR Jupyter environment. A machine-specific editable `minervachem` path was omitted because it is not portable and is not imported by the released modular O'Neil workflow.

A CUDA-enabled PyTorch installation can be used for GPU training. If CUDA is unavailable, the code automatically falls back to CPU.

## Data check

Before training, run:

```bash
python check_data.py
```

All four O'Neil inputs required by the released workflow are included in `data/O'Neil/`. `check_data.py` verifies their presence and basic table structure before training.

## Run the released O'Neil workflow

```bash
python run_oneil.py
```

Outputs are written to:

```text
results/oneil_main/
```

and final summaries are also written to:

```text
results/final_fixedK_summary.csv
results/final_active_summary.csv
```

## Released O'Neil settings

The modular code preserves the settings used by the supplied notebook:

- grid size: 4 x 4
- fixed split file: `ONEIL_4x4_block_split_70_10_20_seed42.csv`
- Random Mask-K training schedule: `[0, 2, 4, 4, 6, 6, 8, 8, 10, 12]`
- fixed-K evaluation: `K = 0, 2, 4, 6, 8, 10, 12, 14, 16`
- optimizer: AdamW
- learning rate: `5e-4`
- weight decay: `1e-5`
- maximum epochs: `120`
- batch size: `512`
- early-stopping patience: `20`
- drug-swap augmentation probability: `0.5`
- ensemble seeds: `42, 2024, 3407`
- active acquisition: uncertainty only
- initial active observations: random 2 positions
- acquisition batch size: 2

## Data folders

### `data/O'Neil/`

Contains the renamed O'Neil response table, standardized graphlet features, train-only CRISPR features, and the fixed split reference. See `data/O'Neil/README.md`.

### `data/NCI-ALMANAC/`

Contains the supplied NCI-ALMANAC response/input and graphlet/CRISPR-related files. These files are included as study data; the modular executable in this package is the O'Neil workflow reconstructed from the supplied main notebook.

### `data/DrugComb/`

Contains the supplied DrugComb 6x6 and 10x10 files. Large CSV files are gzip-compressed so every repository file stays comfortably below common GitHub web-upload limits. Pandas can read these `.csv.gz` files directly with `pd.read_csv(...)`.

## Reproducibility note

This repository separates **code cleanup** from **method changes**. The released model implementation was simplified structurally by removing branches that were inactive in the supplied main configuration. A numerical equivalence check was performed against the original notebook implementation for the active architecture; with identical weights and inputs, the compared model outputs were identical.

No scientific input was synthesized. The released O'Neil workflow now includes the exact standardized graphlet feature table supplied for the experiment.

## Citation

Please cite the manuscript if you use this code. A machine-readable citation template is provided in `CITATION.cff`.
