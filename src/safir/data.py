"""Data loading, matching, 4x4 surface construction, and the fixed O'Neil split."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .config import *
from .utils import (
    build_normalized_index_map, find_column, make_block_key, make_observed_mask,
    normalize_match_name,
)

def load_drug_features(path):
    df = pd.read_csv(path)

    name_col = find_column(df, ["name", "drug", "drug_name", "Drug", "DrugName"])
    if name_col is None:
        # 如果没有 name 列，默认第一列是药物名
        name_col = df.columns[0]

    drop_cols = []
    for c in df.columns:
        if c == name_col:
            continue
        if c.lower() in ["smile", "smiles", "mol", "molecule"]:
            drop_cols.append(c)

    feature_cols = [c for c in df.columns if c not in [name_col] + drop_cols]
    feat_df = df[[name_col] + feature_cols].copy()
    feat_df = feat_df.set_index(name_col)

    # 只保留数值列
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce")
    feat_df = feat_df.fillna(0.0).astype(np.float32)

    return feat_df


def load_cell_features(path):
    df = pd.read_csv(path)

    cell_col = find_column(df, ["cell_line", "Cell_Line", "cell", "Cell", "CELL_LINE"])
    if cell_col is None:
        cell_col = df.columns[0]

    feat_df = df.set_index(cell_col)
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce")
    feat_df = feat_df.fillna(0.0).astype(np.float32)

    return feat_df


def load_raw_data():
    combo_df = pd.read_csv(COMBO_FILE)
    drug_features = load_drug_features(DRUG_FEATURE_FILE)
    cell_features = load_cell_features(ACTIVE_CELL_FEATURE_FILE)

    print("\n========== Raw data ==========")
    print("Combo shape:", combo_df.shape)
    print("Combo columns:", combo_df.columns.tolist())
    print("Drug numeric feature shape:", drug_features.shape)
    print("Cell numeric feature shape:", cell_features.shape)

    return combo_df, drug_features, cell_features


def preprocess_combo_points(combo_df, drug_features, cell_features):
    drug1_col = find_column(combo_df, ["drugA_name", "drug1_name", "drug_row", "Drug1", "drug1"])
    drug2_col = find_column(combo_df, ["drugB_name", "drug2_name", "drug_col", "Drug2", "drug2"])
    cell_col = find_column(combo_df, ["cell_line", "cell", "CellLine", "CELL_LINE"])

    conc1_col = find_column(combo_df, ["drugA_conc", "drug1_conc", "conc1", "dose1", "Drug1.Conc"])
    conc2_col = find_column(combo_df, ["drugB_conc", "drug2_conc", "conc2", "dose2", "Drug2.Conc"])

    single1_col = find_column(combo_df, ["single_resp_1", "single_response_1", "singleA", "YA"])
    single2_col = find_column(combo_df, ["single_resp_2", "single_response_2", "singleB", "YB"])

    target_col = find_column(combo_df, ["target", "response", "Y", "X/X0", "inhibition"])

    required = [drug1_col, drug2_col, cell_col, conc1_col, conc2_col, single1_col, single2_col, target_col]
    if any(c is None for c in required):
        raise ValueError(
            "找不到必要列，请检查组合数据列名。当前列：\n"
            + str(combo_df.columns.tolist())
        )

    drug_key_to_raw = build_normalized_index_map(drug_features.index, label="drug_features")
    cell_key_to_raw = build_normalized_index_map(cell_features.index, label="cell_features")

    combo_drugs = sorted(
        set(combo_df[drug1_col].dropna().astype(str)) |
        set(combo_df[drug2_col].dropna().astype(str))
    )
    exact_drug_match = len(set(combo_drugs) & set(map(str, drug_features.index)))
    norm_drug_match = sum(1 for d in combo_drugs if normalize_match_name(d) in drug_key_to_raw)

    print("\n========== Drug name matching check ==========")
    print("Combo unique drugs:", len(combo_drugs))
    print("Drug feature rows:", len(drug_features))
    print("Exact matched combo drugs:", exact_drug_match)
    print("Normalized matched combo drugs:", norm_drug_match)

    norm_missing_drugs = [d for d in combo_drugs if normalize_match_name(d) not in drug_key_to_raw]
    if len(norm_missing_drugs) > 0:
        print("[Warning] Still missing drugs after normalized matching:")
        print(norm_missing_drugs)
    else:
        print("All combo drugs matched after normalized matching.")

    rows = []
    skipped = {"drugA": 0, "drugB": 0, "cell": 0, "error": 0}

    for _, row in combo_df.iterrows():
        try:
            drugA = str(row[drug1_col]).strip()
            drugB = str(row[drug2_col]).strip()
            cell = str(row[cell_col]).strip()

            drugA_key = normalize_match_name(drugA)
            drugB_key = normalize_match_name(drugB)
            cell_key = normalize_match_name(cell)

            if drugA_key not in drug_key_to_raw:
                skipped["drugA"] += 1
                continue
            if drugB_key not in drug_key_to_raw:
                skipped["drugB"] += 1
                continue
            if cell_key not in cell_key_to_raw:
                skipped["cell"] += 1
                continue

            drugA_feature_name = drug_key_to_raw[drugA_key]
            drugB_feature_name = drug_key_to_raw[drugB_key]
            cell_feature_name = cell_key_to_raw[cell_key]

            drugA_conc = float(row[conc1_col])
            drugB_conc = float(row[conc2_col])
            single_resp_1 = float(row[single1_col])
            single_resp_2 = float(row[single2_col])
            target = float(row[target_col])

            if not np.isfinite([drugA_conc, drugB_conc, single_resp_1, single_resp_2, target]).all():
                skipped["error"] += 1
                continue

            rows.append({
                "drugA_name": drugA,
                "drugB_name": drugB,
                "cell_line": cell,
                "drugA_feature_name": drugA_feature_name,
                "drugB_feature_name": drugB_feature_name,
                "cell_feature_name": cell_feature_name,
                "drugA_conc": drugA_conc,
                "drugB_conc": drugB_conc,
                "single_resp_1": single_resp_1,
                "single_resp_2": single_resp_2,
                "target": target,
            })
        except Exception:
            skipped["error"] += 1

    data_df = pd.DataFrame(rows)

    print("\n========== Matched point-level data ==========")
    print("Valid point samples:", len(data_df))
    print("Valid cell lines:", data_df["cell_line"].nunique() if len(data_df) else 0)
    print("Valid drugA unique:", data_df["drugA_name"].nunique() if len(data_df) else 0)
    print("Valid drugB unique:", data_df["drugB_name"].nunique() if len(data_df) else 0)
    print("Skipped:", skipped)

    if SAVE_MATCHED_POINTS:
        out_path = RESULT_DIR / "matched_point_level_data.csv"
        data_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print("Saved matched point-level data:", out_path)

    return data_df


def build_grid_features_from_axes(doseA_levels, doseB_levels, ya, yb):
    """
    Build the exact 17-D point-wise grid features used by SAFIR.

    Keeping this logic in one helper is important for drug-swap augmentation:
    after swapping A/B we REBUILD the grid features from the swapped dose and
    single-agent axes rather than trying to permute/sign-flip individual columns.
    """
    doseA_levels = np.asarray(doseA_levels, dtype=np.float32)
    doseB_levels = np.asarray(doseB_levels, dtype=np.float32)
    ya = np.asarray(ya, dtype=np.float32)
    yb = np.asarray(yb, dtype=np.float32)

    if len(doseA_levels) != GRID_SIZE_A or len(doseB_levels) != GRID_SIZE_B:
        raise ValueError(
            f"Expected {GRID_SIZE_A}x{GRID_SIZE_B} dose axes, got "
            f"{len(doseA_levels)}x{len(doseB_levels)}"
        )

    logA = np.log10(doseA_levels + 1e-12)
    logB = np.log10(doseB_levels + 1e-12)
    logA_norm = (logA - logA.mean()) / (logA.std() + 1e-6)
    logB_norm = (logB - logB.mean()) / (logB.std() + 1e-6)

    grid_feats = []
    for i in range(GRID_SIZE_A):
        for j in range(GRID_SIZE_B):
            a = logA_norm[i]
            b = logB_norm[j]
            yai = ya[i]
            ybj = yb[j]

            bliss_like = yai + ybj - yai * ybj
            max_single = max(yai, ybj)
            min_single = min(yai, ybj)

            grid_feats.append([
                a,
                b,
                a * b,
                abs(a - b),
                a + b,
                yai,
                ybj,
                yai * ybj,
                abs(yai - ybj),
                yai + ybj,
                bliss_like,
                max_single,
                min_single,
                yai - ybj,
                (yai - ybj) ** 2,
                a ** 2,
                b ** 2,
            ])

    return np.asarray(grid_feats, dtype=np.float32)


def transpose_flat_grid(x):
    """Transpose a flattened GRID_SIZE_A x GRID_SIZE_B array.

    This code is for the O'Neil 4x4 experiment, so the transposed shape is the
    same.  An explicit guard prevents silently using it for a non-square grid.
    """
    if GRID_SIZE_A != GRID_SIZE_B:
        raise ValueError(
            "Drug-swap augmentation in this O'Neil script assumes a square grid. "
            f"Current grid is {GRID_SIZE_A}x{GRID_SIZE_B}."
        )
    arr = np.asarray(x)
    return arr.reshape(GRID_SIZE_A, GRID_SIZE_B).T.reshape(-1).copy()


def build_surface_blocks(point_df, drug_features, cell_features):
    """
    一个 block = 一个 drugA-drugB-cell_line 的 4x4 剂量响应矩阵。
    输出 list[dict]，每个 dict 包含：
    - drug/cell 特征
    - doseA/doseB
    - YA[4], YB[4]
    - Y[4,4]
    - grid_features[16, 10]
    """
    blocks = []
    skipped = {
        "not_4x4": 0,
        "missing_points": 0,
        "feature_error": 0,
    }

    group_cols = ["drugA_name", "drugB_name", "cell_line"]
    grouped = point_df.groupby(group_cols, sort=False)

    for (drugA, drugB, cell), g in grouped:
        try:
            doseA_levels = sorted(g["drugA_conc"].unique())
            doseB_levels = sorted(g["drugB_conc"].unique())

            if REQUIRE_COMPLETE_4X4:
                if len(doseA_levels) != GRID_SIZE_A or len(doseB_levels) != GRID_SIZE_B:
                    skipped["not_4x4"] += 1
                    continue

            # pivot target
            y_mat = g.pivot_table(
                index="drugA_conc",
                columns="drugB_conc",
                values="target",
                aggfunc="mean"
            ).reindex(index=doseA_levels, columns=doseB_levels)

            if y_mat.isna().any().any():
                skipped["missing_points"] += 1
                continue

            # 单药响应向量
            ya = (
                g.groupby("drugA_conc")["single_resp_1"]
                .mean()
                .reindex(doseA_levels)
                .values
                .astype(np.float32)
            )
            yb = (
                g.groupby("drugB_conc")["single_resp_2"]
                .mean()
                .reindex(doseB_levels)
                .values
                .astype(np.float32)
            )

            if np.any(~np.isfinite(ya)) or np.any(~np.isfinite(yb)):
                skipped["missing_points"] += 1
                continue

            # feature names should be constant inside block
            drugA_feature_name = g["drugA_feature_name"].iloc[0]
            drugB_feature_name = g["drugB_feature_name"].iloc[0]
            cell_feature_name = g["cell_feature_name"].iloc[0]

            drugA_feat = drug_features.loc[drugA_feature_name].values.astype(np.float32)
            drugB_feat = drug_features.loc[drugB_feature_name].values.astype(np.float32)
            cell_feat = cell_features.loc[cell_feature_name].values.astype(np.float32)

            y = y_mat.values.astype(np.float32)

            # Build the canonical A/B-oriented point-wise dose/single-agent features.
            grid_feats = build_grid_features_from_axes(
                doseA_levels=doseA_levels,
                doseB_levels=doseB_levels,
                ya=ya,
                yb=yb,
            )

            blocks.append({
                "block_id": len(blocks),
                "block_key": make_block_key(drugA, drugB, cell),
                "drugA_name": drugA,
                "drugB_name": drugB,
                "cell_line": cell,
                "drugA_feature_name": drugA_feature_name,
                "drugB_feature_name": drugB_feature_name,
                "cell_feature_name": cell_feature_name,
                "doseA_levels": np.asarray(doseA_levels, dtype=np.float32),
                "doseB_levels": np.asarray(doseB_levels, dtype=np.float32),
                "YA": ya,
                "YB": yb,
                "Y": y,
                "grid_features": grid_feats,
                "drugA_feat": drugA_feat,
                "drugB_feat": drugB_feat,
                "cell_feat": cell_feat,
            })

        except Exception:
            skipped["feature_error"] += 1

    print("\n========== Block-level surface data ==========")
    print("Valid 4x4 blocks:", len(blocks))
    print("Skipped:", skipped)

    if len(blocks) == 0:
        raise RuntimeError("没有构建出任何 4x4 block，请检查数据。")

    block_summary = []
    for b in blocks:
        block_summary.append({
            "block_id": b["block_id"],
            "block_key": b["block_key"],
            "drugA_name": b["drugA_name"],
            "drugB_name": b["drugB_name"],
            "cell_line": b["cell_line"],
            "target_min": float(np.min(b["Y"])),
            "target_max": float(np.max(b["Y"])),
            "target_range": float(np.max(b["Y"]) - np.min(b["Y"])),
            "doseA_levels": "|".join([f"{x:g}" for x in b["doseA_levels"]]),
            "doseB_levels": "|".join([f"{x:g}" for x in b["doseB_levels"]]),
        })

    block_summary_df = pd.DataFrame(block_summary)

    print("Cell lines:", block_summary_df["cell_line"].nunique())
    print("Drug A unique:", block_summary_df["drugA_name"].nunique())
    print("Drug B unique:", block_summary_df["drugB_name"].nunique())
    print("Median target range:", block_summary_df["target_range"].median())

    if SAVE_BLOCK_TABLE:
        out_path = RESULT_DIR / "surface_block_summary.csv"
        block_summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print("Saved block summary:", out_path)

    return blocks, block_summary_df


class SurfaceBlockDataset(Dataset):
    def __init__(
        self,
        blocks,
        observed_k_list,
        training=True,
        random_state=42,
        drug_swap_augmentation=False,
        drug_swap_prob=0.50,
    ):
        self.blocks = blocks
        self.observed_k_list = list(observed_k_list)
        self.training = bool(training)
        self.drug_swap_augmentation = bool(drug_swap_augmentation)
        self.drug_swap_prob = float(drug_swap_prob)

        if not (0.0 <= self.drug_swap_prob <= 1.0):
            raise ValueError(f"drug_swap_prob must be in [0,1], got {self.drug_swap_prob}")
        if self.drug_swap_augmentation and GRID_SIZE_A != GRID_SIZE_B:
            raise ValueError(
                "This O'Neil drug-swap augmentation implementation requires a square grid; "
                f"got {GRID_SIZE_A}x{GRID_SIZE_B}."
            )

        # Preserve the original mask-K random stream for fair comparison with the
        # no-augmentation model.  Swap decisions use a separate RNG.
        self.rng = np.random.default_rng(random_state)
        self.swap_rng = np.random.default_rng(int(random_state) + 104729)

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        b = self.blocks[idx]

        # Random Mask-K is generated exactly as in the original model.
        if self.training:
            k = int(self.rng.choice(self.observed_k_list))
            obs_mask = make_observed_mask(N_GRID, k, self.rng)
        else:
            # Evaluation functions construct their own masks; dataset default is K=0.
            k = 0
            obs_mask = np.zeros(N_GRID, dtype=np.float32)

        # Drug-swap augmentation is TRAINING-ONLY and uses an independent RNG.
        # The same physical experiment is re-expressed as (B,A):
        #   drugA <-> drugB
        #   doseA <-> doseB
        #   YA <-> YB
        #   Y -> Y^T
        #   observation/loss masks -> transpose
        do_swap = (
            self.training
            and self.drug_swap_augmentation
            and (self.swap_rng.random() < self.drug_swap_prob)
        )

        if do_swap:
            drugA_feat = b["drugB_feat"].astype(np.float32)
            drugB_feat = b["drugA_feat"].astype(np.float32)
            YA = b["YB"].astype(np.float32)
            YB = b["YA"].astype(np.float32)
            Y = b["Y"].T.copy().astype(np.float32)

            # Rebuild orientation-dependent grid features after swapping axes.
            grid_features = build_grid_features_from_axes(
                doseA_levels=b["doseB_levels"],
                doseB_levels=b["doseA_levels"],
                ya=YA,
                yb=YB,
            )

            # Preserve the same physical context points after A/B exchange.
            obs_mask = transpose_flat_grid(obs_mask).astype(np.float32)
            loss_mask = transpose_flat_grid(
                b.get("loss_mask", np.ones(N_GRID, dtype=np.float32))
            ).astype(np.float32)
            eval_dose_mask = transpose_flat_grid(
                b.get("eval_dose_mask", np.zeros(N_GRID, dtype=np.float32))
            ).astype(np.float32)
        else:
            drugA_feat = b["drugA_feat"].astype(np.float32)
            drugB_feat = b["drugB_feat"].astype(np.float32)
            YA = b["YA"].astype(np.float32)
            YB = b["YB"].astype(np.float32)
            Y = b["Y"].astype(np.float32)
            grid_features = b["grid_features"].astype(np.float32)
            loss_mask = b.get(
                "loss_mask", np.ones(N_GRID, dtype=np.float32)
            ).astype(np.float32)
            eval_dose_mask = b.get(
                "eval_dose_mask", np.zeros(N_GRID, dtype=np.float32)
            ).astype(np.float32)

        y_flat = Y.reshape(-1).astype(np.float32)
        obs_values = y_flat * obs_mask

        item = {
            "block_id": np.int64(b["block_id"]),
            "drugA_feat": drugA_feat,
            "drugB_feat": drugB_feat,
            "cell_feat": b["cell_feat"].astype(np.float32),
            "grid_features": grid_features,
            "YA": YA,
            "YB": YB,
            "Y": Y,
            "Y_flat": y_flat,
            "obs_mask": obs_mask.astype(np.float32),
            "obs_values": obs_values.astype(np.float32),
            "loss_mask": loss_mask,
            "eval_dose_mask": eval_dose_mask,
            "observed_k": np.int64(k),
            "drug_swapped": np.int64(1 if do_swap else 0),
        }
        return item


def collate_fn(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if k in ["block_id", "observed_k", "drug_swapped"]:
            out[k] = torch.tensor(vals, dtype=torch.long)
        else:
            out[k] = torch.tensor(np.stack(vals), dtype=torch.float32)
    return out


def make_loader(
    blocks, observed_k_list, training=True, shuffle=True,
    drug_swap_augmentation=False, drug_swap_prob=DRUG_SWAP_PROB,
):
    ds = SurfaceBlockDataset(
        blocks=blocks,
        observed_k_list=observed_k_list,
        training=training,
        random_state=RANDOM_STATE,
        drug_swap_augmentation=drug_swap_augmentation,
        drug_swap_prob=drug_swap_prob,
    )
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE.type == "cuda" else False,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        collate_fn=collate_fn,
    )


def load_fixed_no_leakage_block_split(blocks, split_reference_file=SPLIT_REFERENCE_FILE):
    """
    读取上游 CRISPR 预处理阶段已经固定好的 70/10/20 block split。
    绝不在这里重新随机划分；否则会破坏 train-only Top1000 的数据隔离。
    """
    if not Path(split_reference_file).exists():
        raise FileNotFoundError(
            f"固定 split 文件不存在：{split_reference_file}\n"
            "Please provide the released fixed split file in data/O'Neil/."
        )

    split_df = pd.read_csv(split_reference_file)
    required = {"block_key", "split"}
    if not required.issubset(split_df.columns):
        raise ValueError(f"split 文件必须包含 {required}，当前列：{split_df.columns.tolist()}")

    split_df["block_key"] = split_df["block_key"].astype(str)
    split_df["split"] = split_df["split"].astype(str).str.lower().replace({"validation": "val"})
    if split_df["block_key"].duplicated().any():
        dup = split_df.loc[split_df["block_key"].duplicated(False), "block_key"].head(10).tolist()
        raise RuntimeError(f"split reference 中 block_key 重复：{dup}")

    split_map = dict(zip(split_df["block_key"], split_df["split"]))
    current_keys = {str(b["block_key"]) for b in blocks}
    ref_keys = set(split_map)

    missing_in_reference = sorted(current_keys - ref_keys)
    missing_in_current = sorted(ref_keys - current_keys)
    if missing_in_reference or missing_in_current:
        raise RuntimeError(
            "当前 SAFIR blocks 与生成 CRISPR 时的 blocks 不一致。\n"
            f"当前有但 split 无：{missing_in_reference[:10]}\n"
            f"split 有但当前无：{missing_in_current[:10]}\n"
            "请确保组合数据、Graphlet 文件和筛选条件与上游预处理 Notebook 相同。"
        )

    train_blocks = [b for b in blocks if split_map[b["block_key"]] == "train"]
    val_blocks = [b for b in blocks if split_map[b["block_key"]] == "val"]
    test_blocks = [b for b in blocks if split_map[b["block_key"]] == "test"]

    assigned = len(train_blocks) + len(val_blocks) + len(test_blocks)
    if assigned != len(blocks):
        unknown = sorted({split_map[b["block_key"]] for b in blocks} - {"train", "val", "test"})
        raise RuntimeError(f"存在未知 split 标签：{unknown}")
    if not train_blocks or not val_blocks or not test_blocks:
        raise RuntimeError("Train/Val/Test 任一集合为空。")

    # 保存本次实际使用的审计表
    audit_rows = []
    for b in blocks:
        audit_rows.append({
            "block_id_runtime": b["block_id"],
            "block_key": b["block_key"],
            "drugA_name": b["drugA_name"],
            "drugB_name": b["drugB_name"],
            "cell_line": b["cell_line"],
            "split": split_map[b["block_key"]],
        })
    pd.DataFrame(audit_rows).to_csv(
        RESULT_DIR / "block_split_reference_reused_from_CRISPR_preprocessing.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== Fixed no-leakage block split ==========")
    print("Train blocks:", len(train_blocks))
    print("Val blocks:", len(val_blocks))
    print("Test blocks:", len(test_blocks))
    print("Split source:", split_reference_file)
    return train_blocks, val_blocks, test_blocks
