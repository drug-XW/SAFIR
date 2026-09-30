#!/usr/bin/env python
"""Run the released O'Neil SAFIR experiment.

This entry point uses repository-relative paths and writes outputs to
results/oneil_main/.
"""
import sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from safir.config import (
    COMBO_FILE, DRUG_FEATURE_FILE, CELL_FEATURE_FILE, SPLIT_REFERENCE_FILE,
    RESULT_DIR, RESULT_ROOT_DIR, RANDOM_STATE, MODEL_MODE_NAME, MODEL_CONFIG,
)
from safir.utils import seed_everything
from safir.data import (
    load_raw_data, preprocess_combo_points, build_surface_blocks,
    load_fixed_no_leakage_block_split,
)
from safir.training import train_one_setting_active


def validate_required_files():
    required = {
        "O'Neil response table": COMBO_FILE,
        "O'Neil graphlet features": DRUG_FEATURE_FILE,
        "train-only CRISPR features": CELL_FEATURE_FILE,
        "fixed train/val/test split": SPLIT_REFERENCE_FILE,
    }
    missing = [(label, path) for label, path in required.items() if not Path(path).exists()]
    if missing:
        msg = ["Required input file(s) are missing:"]
        msg += [f"  - {label}: {path}" for label, path in missing]
        msg.append(
            "\nThe provided upload set did not contain the O'Neil file "
            "Graphlet_features_6_standardized.csv. Add it to data/O'Neil/ "
            "before running the full experiment."
        )
        raise FileNotFoundError("\n".join(msg))


def main():
    validate_required_files()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    seed_everything(RANDOM_STATE)

    combo_df, drug_features, cell_features = load_raw_data()
    point_df = preprocess_combo_points(combo_df, drug_features, cell_features)
    blocks, _ = build_surface_blocks(point_df, drug_features, cell_features)
    train_blocks, val_blocks, test_blocks = load_fixed_no_leakage_block_split(
        blocks, SPLIT_REFERENCE_FILE
    )

    _, fixed_summary, _, active_summary, _ = train_one_setting_active(
        train_blocks,
        val_blocks,
        test_blocks,
        setting_name="block_random_split",
        model_mode_name=MODEL_MODE_NAME,
        model_mode_config=MODEL_CONFIG,
    )

    fixed_summary = fixed_summary.copy()
    fixed_summary["setting"] = "block_random_split"
    fixed_summary["model_mode"] = MODEL_MODE_NAME
    active_summary = active_summary.copy()
    active_summary["setting"] = "block_random_split"
    active_summary["model_mode"] = MODEL_MODE_NAME

    fixed_summary.to_csv(
        RESULT_ROOT_DIR / "final_fixedK_summary.csv", index=False, encoding="utf-8-sig"
    )
    active_summary.to_csv(
        RESULT_ROOT_DIR / "final_active_summary.csv", index=False, encoding="utf-8-sig"
    )

    print("\nFinished. Results are in:", RESULT_ROOT_DIR)


if __name__ == "__main__":
    main()
