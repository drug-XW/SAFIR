# O'Neil data

Files in this directory:

- `O'Neil.csv` - renamed from the supplied `processed_combination_response_r095_clean(2).csv`; point-level combination-response table used by the main O'Neil code.
- `Graphlet_features_6_standardized.csv` - standardized graphlet feature table used by the released O'Neil workflow.
- `CRISPR_TopVar1000_TrainOnly_cell_features.csv` - train-only selected/processed CRISPR cell features supplied with the experiment.
- `ONEIL_4x4_block_split_70_10_20_seed42.csv` - fixed train/validation/test split reference supplied with the experiment.

The released paths in `src/safir/config.py` are:

```python
COMBO_FILE = ONEIL_DIR / "O'Neil.csv"
DRUG_FEATURE_FILE = ONEIL_DIR / "Graphlet_features_6_standardized.csv"
CELL_FEATURE_FILE = ONEIL_DIR / "CRISPR_TopVar1000_TrainOnly_cell_features.csv"
SPLIT_REFERENCE_FILE = ONEIL_DIR / "ONEIL_4x4_block_split_70_10_20_seed42.csv"
```

Run `python check_data.py` from the repository root to verify these files before training.
