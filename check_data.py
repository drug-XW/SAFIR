#!/usr/bin/env python
"""Quick integrity check for repository data files."""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from safir.config import COMBO_FILE, DRUG_FEATURE_FILE, CELL_FEATURE_FILE, SPLIT_REFERENCE_FILE

files = {
    "O'Neil.csv": COMBO_FILE,
    "Graphlet_features_6_standardized.csv": DRUG_FEATURE_FILE,
    "CRISPR_TopVar1000_TrainOnly_cell_features.csv": CELL_FEATURE_FILE,
    "ONEIL_4x4_block_split_70_10_20_seed42.csv": SPLIT_REFERENCE_FILE,
}

for label, path in files.items():
    if path.exists():
        try:
            df = pd.read_csv(path, nrows=3)
            print(f"[OK] {label}: {path} ({len(df.columns)} columns)")
        except Exception as exc:
            print(f"[ERROR] {label}: cannot read {path}: {exc}")
    else:
        print(f"[MISSING] {label}: {path}")
