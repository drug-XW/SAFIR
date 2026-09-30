"""General utilities, reproducibility helpers, and evaluation primitives."""
import random
import re
import numpy as np
import pandas as pd
import torch

from .config import GRID_SIZE_A, GRID_SIZE_B

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def normalize_match_name(x):
    """
    药物名/细胞系名标准化匹配：
    - 大写
    - 去掉空格、横杠、下划线、点、斜杠、括号等符号

    例：
    MK-2206 <-> MK2206
    SN-38 <-> SN38
    Bortezomib <-> BORTEZOMIB
    """
    if pd.isna(x):
        return ""
    x = str(x).strip().upper()
    x = re.sub(r"[\s\-_./()]+", "", x)
    return x


def make_block_key(drug_a, drug_b, cell):
    """与上游无泄露 CRISPR 预处理 Notebook 完全相同的 block key。"""
    return "||".join([
        normalize_match_name(drug_a),
        normalize_match_name(drug_b),
        normalize_match_name(cell),
    ])


def build_normalized_index_map(index_values, label="feature"):
    key_to_raw = {}
    duplicated = []

    for raw in index_values:
        key = normalize_match_name(raw)
        if key == "":
            continue
        if key in key_to_raw:
            duplicated.append((key, key_to_raw[key], raw))
        else:
            key_to_raw[key] = raw

    if len(duplicated) > 0:
        print(f"[Warning] {label} has duplicated normalized keys. Keep first occurrence.")
        print("Examples:", duplicated[:10])

    return key_to_raw


def find_column(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def safe_pcc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def concordance_index(y_true, y_pred):
    """
    C-index：衡量预测排序和真实排序的一致性。
    这里每个 4x4 block 只有 16 个点，所以 O(n^2) 计算不会慢。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = 0
    h_sum = 0.0

    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] == y_true[j]:
                continue
            n += 1
            true_order = y_true[i] - y_true[j]
            pred_order = y_pred[i] - y_pred[j]

            if pred_order == 0:
                h_sum += 0.5
            elif true_order * pred_order > 0:
                h_sum += 1.0

    if n == 0:
        return np.nan

    return h_sum / n


def adjacent_delta_corr(y_true_surface, y_pred_surface):
    yt = np.asarray(y_true_surface, dtype=float)
    yp = np.asarray(y_pred_surface, dtype=float)

    true_deltas = []
    pred_deltas = []

    true_deltas.extend((yt[1:, :] - yt[:-1, :]).ravel().tolist())
    true_deltas.extend((yt[:, 1:] - yt[:, :-1]).ravel().tolist())

    pred_deltas.extend((yp[1:, :] - yp[:-1, :]).ravel().tolist())
    pred_deltas.extend((yp[:, 1:] - yp[:, :-1]).ravel().tolist())

    return safe_pcc(true_deltas, pred_deltas)


def topk_hit_metrics(y_true_surface, y_pred_surface, k=3):
    yt = np.asarray(y_true_surface).reshape(-1)
    yp = np.asarray(y_pred_surface).reshape(-1)

    true_best = int(np.argmax(yt))
    pred_order = np.argsort(-yp)

    top1_hit = int(pred_order[0] == true_best)
    topk_hit = int(true_best in pred_order[:k])

    true_best_ij = np.unravel_index(true_best, (GRID_SIZE_A, GRID_SIZE_B))
    pred_best_ij = np.unravel_index(int(pred_order[0]), (GRID_SIZE_A, GRID_SIZE_B))
    best_dose_distance = float(
        abs(true_best_ij[0] - pred_best_ij[0]) + abs(true_best_ij[1] - pred_best_ij[1])
    )

    return {
        "top1_hit": top1_hit,
        f"top{k}_hit": topk_hit,
        "best_dose_distance": best_dose_distance,
        "true_best_flat": true_best,
        "pred_best_flat": int(pred_order[0]),
    }


def make_observed_mask(n_grid, k, rng):
    mask = np.zeros(n_grid, dtype=np.float32)
    if k <= 0:
        return mask
    k = min(k, n_grid)
    idx = rng.choice(n_grid, size=k, replace=False)
    mask[idx] = 1.0
    return mask


def surface_smoothness_np(surface):
    s = np.asarray(surface, dtype=float)
    vals = []
    vals.append(np.mean((s[1:, :] - s[:-1, :]) ** 2))
    vals.append(np.mean((s[:, 1:] - s[:, :-1]) ** 2))
    return float(np.mean(vals))
