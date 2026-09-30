"""Configuration for the released O'Neil SAFIR workflow.

All paths are repository-relative so the code can be cloned and run without
editing machine-specific absolute paths.
"""
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ONEIL_DIR = DATA_DIR / "O'Neil"
RESULT_ROOT_DIR = REPO_ROOT / "results"
RESULT_DIR = RESULT_ROOT_DIR / "oneil_main"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# User-requested rename: the original processed combination-response file is
# released as data/O'Neil/O'Neil.csv.
COMBO_FILE = ONEIL_DIR / "O'Neil.csv"
DRUG_FEATURE_FILE = ONEIL_DIR / "Graphlet_features_6_standardized.csv"
CELL_FEATURE_FILE = ONEIL_DIR / "CRISPR_TopVar1000_TrainOnly_cell_features.csv"
ACTIVE_CELL_FEATURE_FILE = CELL_FEATURE_FILE
SPLIT_REFERENCE_FILE = ONEIL_DIR / "ONEIL_4x4_block_split_70_10_20_seed42.csv"

RANDOM_STATE = 42
DEFAULT_TRAIN_OBSERVED_K_LIST = [0, 2, 4, 4, 6, 6, 8, 8, 10, 12]
TRAIN_OBSERVED_K_LIST = list(DEFAULT_TRAIN_OBSERVED_K_LIST)
EVAL_OBSERVED_K_LIST = [0, 2, 4, 6, 8, 10, 12, 14, 16]
EVAL_MASK_REPEATS = 3

GRID_SIZE_A = 4
GRID_SIZE_B = 4
N_GRID = GRID_SIZE_A * GRID_SIZE_B
REQUIRE_COMPLETE_4X4 = True

NUM_EPOCHS = 120
BATCH_SIZE = 512
LR = 5e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 20
MIN_DELTA = 1e-6
GRAD_CLIP_NORM = 5.0
NUM_WORKERS = 0

DRUG_SWAP_AUGMENTATION = True
DRUG_SWAP_PROB = 0.50

LAMBDA_OBS_RECON = 0.50
LAMBDA_DELTA = 0.12
LAMBDA_RESIDUAL_CONSISTENCY = 0.10

DRUG_EMB_DIM = 128
CELL_EMB_DIM = 128
GLOBAL_DIM = 256
OBS_EMB_DIM = 64
HIDDEN_DIM = 256
DROPOUT = 0.10
LOCAL_OBS_EMB_DIM = 32

SAVE_MATCHED_POINTS = True
SAVE_BLOCK_TABLE = True
SAVE_PREDICTIONS = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIVE_ENABLE = True
ACTIVE_ENSEMBLE_SIZE = 3
ACTIVE_ENSEMBLE_SEEDS = [42, 2024, 3407]
ACTIVE_INITIAL_SEED_MODE = "random2"
ACTIVE_BATCH_SIZE = 2
ACTIVE_TOTAL_BUDGETS = [2, 4, 6, 8, 10, 12]
ACTIVE_REPEATS = 3
ACTIVE_STRATEGIES = ["uncertainty"]
MAXIMIZE_RESPONSE = True
SAVE_ACTIVE_TRAJECTORY = True

MODEL_MODE_NAME = "ordered_fusion_swapaug"
MODEL_CONFIG = {
    "desc": (
        "SAFIR with ordered global fusion, training-only drug-swap augmentation, "
        "single-agent-informed reference, observation-conditioned residual correction, "
        "CRISPR features, Random Mask-K training, and uncertainty-only acquisition."
    ),
    "use_grid_gnn": False,
    "use_local_obs": True,
    "use_residual_anchor": True,
    "cell_feature_type": "CRISPR_TopVar1000_TrainOnly",
    "train_observed_k_list": list(DEFAULT_TRAIN_OBSERVED_K_LIST),
    "active_strategies": ["uncertainty"],
    "main_experiment_type": "ordered_global_fusion_drug_swap_augmentation",
}
