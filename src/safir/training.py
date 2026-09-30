"""Ensemble training, uncertainty-guided acquisition, and result export."""
import copy
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader

from .config import *
from .utils import seed_everything, make_observed_mask, safe_pcc, concordance_index, adjacent_delta_corr
from .data import SurfaceBlockDataset, collate_fn
from .model import SAFIRDoseSurfaceCompletionNet
from .losses import compute_loss
from .evaluation import summarize_metrics, evaluate_val_loss


def make_loader_with_seed(
    blocks, observed_k_list, training=True, shuffle=True, seed=42,
    drug_swap_augmentation=False, drug_swap_prob=DRUG_SWAP_PROB,
):
    """Loader with independent mask/swap RNG streams for each ensemble seed."""
    ds = SurfaceBlockDataset(
        blocks=blocks,
        observed_k_list=observed_k_list,
        training=training,
        random_state=seed,
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


def block_to_batch_with_mask(block, obs_mask):
    """把单个 block + 当前观测 mask 打包成模型输入。"""
    y_flat = block["Y"].reshape(-1).astype(np.float32)
    obs_mask = np.asarray(obs_mask, dtype=np.float32)
    obs_values = y_flat * obs_mask
    loss_mask = block.get("loss_mask", np.ones(N_GRID, dtype=np.float32)).astype(np.float32)
    eval_dose_mask = block.get("eval_dose_mask", np.zeros(N_GRID, dtype=np.float32)).astype(np.float32)
    batch = {
        "block_id": torch.tensor([block["block_id"]], dtype=torch.long, device=DEVICE),
        "drugA_feat": torch.tensor(block["drugA_feat"][None, :], dtype=torch.float32, device=DEVICE),
        "drugB_feat": torch.tensor(block["drugB_feat"][None, :], dtype=torch.float32, device=DEVICE),
        "cell_feat": torch.tensor(block["cell_feat"][None, :], dtype=torch.float32, device=DEVICE),
        "grid_features": torch.tensor(block["grid_features"][None, :, :], dtype=torch.float32, device=DEVICE),
        "YA": torch.tensor(block["YA"][None, :], dtype=torch.float32, device=DEVICE),
        "YB": torch.tensor(block["YB"][None, :], dtype=torch.float32, device=DEVICE),
        "Y": torch.tensor(block["Y"][None, :, :], dtype=torch.float32, device=DEVICE),
        "Y_flat": torch.tensor(y_flat[None, :], dtype=torch.float32, device=DEVICE),
        "obs_mask": torch.tensor(obs_mask[None, :], dtype=torch.float32, device=DEVICE),
        "obs_values": torch.tensor(obs_values[None, :], dtype=torch.float32, device=DEVICE),
        "loss_mask": torch.tensor(loss_mask[None, :], dtype=torch.float32, device=DEVICE),
        "eval_dose_mask": torch.tensor(eval_dose_mask[None, :], dtype=torch.float32, device=DEVICE),
        "observed_k": torch.tensor([int(obs_mask.sum())], dtype=torch.long, device=DEVICE),
    }
    return batch


@torch.no_grad()
def ensemble_predict_flat(models, block, obs_mask):
    """返回 ensemble mean/std，用于主动选择与评估。"""
    preds = []
    batch = block_to_batch_with_mask(block, obs_mask)
    for model in models:
        model.eval()
        out = model(batch)
        pred = out["pred_flat"][0].detach().cpu().numpy().astype(np.float32)
        preds.append(pred)
    preds = np.stack(preds, axis=0)  # [E, 16]
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)
    return pred_mean, pred_std, preds



@torch.no_grad()
def ensemble_predict_components_flat(models, block, obs_mask):
    """
    返回 active selection 需要的 ensemble 预测分量。

    pred_*    : 完整剂量响应矩阵预测。
    epsilon_* : 模型残差交互项 epsilon，来自 Pred = Base + epsilon。
    synergy_* : 基于 Bliss-like 单药预期的预测协同近似，synergy = pred - bliss_like。

    epsilon/base/synergy-related components are retained only for downstream diagnostics
    and do not participate in uncertainty acquisition.
    """
    preds = []
    epsilons = []
    bases = []
    batch = block_to_batch_with_mask(block, obs_mask)
    for model in models:
        model.eval()
        out = model(batch)
        preds.append(out["pred_flat"][0].detach().cpu().numpy().astype(np.float32))
        epsilons.append(out["epsilon_flat"][0].detach().cpu().numpy().astype(np.float32))
        bases.append(out["base_surface"][0].detach().cpu().numpy().reshape(-1).astype(np.float32))

    preds = np.stack(preds, axis=0)
    epsilons = np.stack(epsilons, axis=0)
    bases = np.stack(bases, axis=0)

    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)
    epsilon_mean = epsilons.mean(axis=0)
    epsilon_std = epsilons.std(axis=0)
    base_mean = bases.mean(axis=0)
    base_std = bases.std(axis=0)

    bliss = np.asarray(block["grid_features"][:, 10], dtype=np.float32)
    synergy_preds = preds - bliss[None, :]
    synergy_mean = synergy_preds.mean(axis=0)
    synergy_std = synergy_preds.std(axis=0)

    return {
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "preds": preds,
        "epsilon_mean": epsilon_mean,
        "epsilon_std": epsilon_std,
        "epsilons": epsilons,
        "base_mean": base_mean,
        "base_std": base_std,
        "synergy_mean": synergy_mean,
        "synergy_std": synergy_std,
        "synergy_preds": synergy_preds,
        "bliss": bliss,
    }

def initial_observed_mask(mode, rng):
    """构造初始 seed 点。"""
    mask = np.zeros(N_GRID, dtype=np.float32)
    if mode is None or mode == "none":
        return mask
    if mode == "random2":
        idx = rng.choice(N_GRID, size=min(2, N_GRID), replace=False)
        mask[idx] = 1.0
        return mask
    if mode == "corners4":
        idx = [0, GRID_SIZE_B - 1, (GRID_SIZE_A - 1) * GRID_SIZE_B, N_GRID - 1]
        mask[idx] = 1.0
        return mask
    if mode == "center4":
        # 4x4 中心区域: (1,1),(1,2),(2,1),(2,2)
        idx = [1 * GRID_SIZE_B + 1, 1 * GRID_SIZE_B + 2, 2 * GRID_SIZE_B + 1, 2 * GRID_SIZE_B + 2]
        mask[idx] = 1.0
        return mask
    raise ValueError(f"Unknown ACTIVE_INITIAL_SEED_MODE: {mode}")











def surface_shape_diagnostics(y_true_surface, y_pred_surface):
    """
    曲面形态诊断指标，用于评估补全矩阵拟合成曲面时是否保留相邻剂量变化趋势和局部弯曲形态。
    这些指标仍然基于 4x4 离散矩阵，不等同于真实连续剂量曲面验证。
    """
    yt = np.asarray(y_true_surface, dtype=float).reshape(GRID_SIZE_A, GRID_SIZE_B)
    yp = np.asarray(y_pred_surface, dtype=float).reshape(GRID_SIZE_A, GRID_SIZE_B)

    true_grad = []
    pred_grad = []
    true_grad.extend((yt[1:, :] - yt[:-1, :]).reshape(-1).tolist())
    true_grad.extend((yt[:, 1:] - yt[:, :-1]).reshape(-1).tolist())
    pred_grad.extend((yp[1:, :] - yp[:-1, :]).reshape(-1).tolist())
    pred_grad.extend((yp[:, 1:] - yp[:, :-1]).reshape(-1).tolist())
    true_grad = np.asarray(true_grad, dtype=float)
    pred_grad = np.asarray(pred_grad, dtype=float)

    grad_rmse = float(np.sqrt(mean_squared_error(true_grad, pred_grad))) if len(true_grad) > 0 else np.nan
    grad_mae = float(mean_absolute_error(true_grad, pred_grad)) if len(true_grad) > 0 else np.nan
    grad_corr = safe_pcc(true_grad, pred_grad)

    def laplacian_like(arr):
        vals = []
        for i in range(GRID_SIZE_A):
            for j in range(GRID_SIZE_B):
                neigh = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < GRID_SIZE_A and 0 <= nj < GRID_SIZE_B:
                        neigh.append(arr[ni, nj])
                if len(neigh) > 0:
                    vals.append(arr[i, j] - np.mean(neigh))
                else:
                    vals.append(0.0)
        return np.asarray(vals, dtype=float)

    true_lap = laplacian_like(yt)
    pred_lap = laplacian_like(yp)
    lap_rmse = float(np.sqrt(mean_squared_error(true_lap, pred_lap)))
    lap_mae = float(mean_absolute_error(true_lap, pred_lap))
    lap_corr = safe_pcc(true_lap, pred_lap)

    return {
        "surface_gradient_rmse": grad_rmse,
        "surface_gradient_mae": grad_mae,
        "surface_gradient_corr": grad_corr,
        "surface_laplacian_rmse": lap_rmse,
        "surface_laplacian_mae": lap_mae,
        "surface_laplacian_corr": lap_corr,
    }



def select_next_points(
    strategy,
    pred_mean,
    pred_std,
    obs_mask,
    batch_size,
    rng,
    epsilon_mean=None,
    epsilon_std=None,
    synergy_mean=None,
    synergy_std=None,
):
    """Uncertainty-only acquisition: select unobserved positions with largest ensemble std."""
    if strategy != "uncertainty":
        raise ValueError(
            f"This notebook is uncertainty-only, but got strategy={strategy!r}. "
            "Set ACTIVE_STRATEGIES = ['uncertainty']."
        )

    unobs_idx = np.where(obs_mask < 0.5)[0]
    if len(unobs_idx) == 0:
        return []

    batch_size = min(int(batch_size), len(unobs_idx))
    score = np.asarray(pred_std, dtype=float).copy()
    score[obs_mask > 0.5] = -np.inf
    return np.argsort(-score)[:batch_size].tolist()


def true_top_indices(y_flat, k=3):
    if MAXIMIZE_RESPONSE:
        return np.argsort(-y_flat)[:k]
    return np.argsort(y_flat)[:k]


def flat_to_ij(idx):
    return int(idx // GRID_SIZE_B), int(idx % GRID_SIZE_B)


def _top_region_metrics(true_score, pred_score, obs_mask, prefix, larger_is_better=True, topk=3):
    """用于高响应区域/协同热点区域的通用 top-k 评估。"""
    true_score = np.asarray(true_score, dtype=float)
    pred_score = np.asarray(pred_score, dtype=float)
    obs_mask = np.asarray(obs_mask, dtype=np.float32)

    if larger_is_better:
        true_order = np.argsort(-true_score)
        pred_order = np.argsort(-pred_score)
    else:
        true_order = np.argsort(true_score)
        pred_order = np.argsort(pred_score)

    true_best = int(true_order[0])
    pred_best = int(pred_order[0])
    true_topk = set(true_order[:topk].tolist())
    pred_topk = set(pred_order[:topk].tolist())

    ti, tj = flat_to_ij(true_best)
    pi, pj = flat_to_ij(pred_best)
    true_best_value = float(true_score[true_best])
    pred_best_true_value = float(true_score[pred_best])
    regret = float(true_best_value - pred_best_true_value) if larger_is_better else float(pred_best_true_value - true_best_value)

    observed_idx = np.where(obs_mask > 0.5)[0]
    if len(observed_idx) > 0:
        observed_set = set(observed_idx.tolist())
        observed_top1_hit = float(true_best in observed_set)
        observed_top3_hit = float(len(true_topk & observed_set) > 0)
        if larger_is_better:
            obs_best_idx = int(observed_idx[np.argmax(true_score[observed_idx])])
            obs_best_value = float(np.max(true_score[observed_idx]))
            observed_regret = float(true_best_value - obs_best_value)
        else:
            obs_best_idx = int(observed_idx[np.argmin(true_score[observed_idx])])
            obs_best_value = float(np.min(true_score[observed_idx]))
            observed_regret = float(obs_best_value - true_best_value)
        oi, oj = flat_to_ij(obs_best_idx)
        observed_best_distance = float(abs(ti - oi) + abs(tj - oj))
    else:
        observed_top1_hit = 0.0
        observed_top3_hit = 0.0
        obs_best_idx = -1
        obs_best_value = np.nan
        observed_regret = np.nan
        observed_best_distance = np.nan

    # top-25% 区域重叠：4x4 矩阵默认取前 4 个点。
    q = max(1, int(np.ceil(0.25 * len(true_score))))
    true_hot = set(true_order[:q].tolist())
    pred_hot = set(pred_order[:q].tolist())
    obs_hot = set(observed_idx.tolist()) if len(observed_idx) > 0 else set()
    pred_jaccard = float(len(true_hot & pred_hot) / max(1, len(true_hot | pred_hot)))
    observed_recall = float(len(true_hot & obs_hot) / max(1, len(true_hot)))

    return {
        f"{prefix}_pred_top1_hit": float(pred_best == true_best),
        f"{prefix}_pred_top3_hit": float(len(true_topk & pred_topk) > 0),
        f"{prefix}_pred_best_distance": float(abs(ti - pi) + abs(tj - pj)),
        f"{prefix}_pred_regret": regret,
        f"{prefix}_observed_top1_hit": observed_top1_hit,
        f"{prefix}_observed_top3_hit": observed_top3_hit,
        f"{prefix}_observed_best_distance": observed_best_distance,
        f"{prefix}_observed_regret": observed_regret,
        f"{prefix}_true_best_flat": true_best,
        f"{prefix}_pred_best_flat": pred_best,
        f"{prefix}_observed_best_flat": obs_best_idx,
        f"{prefix}_true_best_value": true_best_value,
        f"{prefix}_pred_best_true_value": pred_best_true_value,
        f"{prefix}_observed_best_true_value": obs_best_value,
        f"{prefix}_top25_jaccard_pred": pred_jaccard,
        f"{prefix}_top25_recall_observed": observed_recall,
    }


def evaluate_active_state(
    block,
    pred_mean,
    pred_std,
    obs_mask,
    epsilon_mean=None,
    epsilon_std=None,
    synergy_mean=None,
    synergy_std=None,
):
    """评估当前 active state：矩阵恢复、高响应区域覆盖、协同热点区域覆盖。"""
    y_flat = block["Y"].reshape(-1).astype(float)
    pred_flat = np.asarray(pred_mean, dtype=float)
    pred_std = np.asarray(pred_std, dtype=float)
    obs_mask = np.asarray(obs_mask, dtype=np.float32)
    miss_mask = 1.0 - obs_mask

    # 曲面级指标
    rmse_all = float(np.sqrt(mean_squared_error(y_flat, pred_flat)))
    mae_all = float(mean_absolute_error(y_flat, pred_flat))
    pcc_all = safe_pcc(y_flat, pred_flat)
    cindex_all = concordance_index(y_flat, pred_flat)

    if miss_mask.sum() > 0:
        miss_bool = miss_mask.astype(bool)
        rmse_missing = float(np.sqrt(mean_squared_error(y_flat[miss_bool], pred_flat[miss_bool])))
        mae_missing = float(mean_absolute_error(y_flat[miss_bool], pred_flat[miss_bool]))
        pcc_missing = safe_pcc(y_flat[miss_bool], pred_flat[miss_bool])
    else:
        rmse_missing = np.nan
        mae_missing = np.nan
        pcc_missing = np.nan

    # 高响应区域指标：沿用原 pred_top/observed_top 命名，方便和旧结果兼容。
    response_metrics = _top_region_metrics(
        true_score=y_flat,
        pred_score=pred_flat,
        obs_mask=obs_mask,
        prefix="response",
        larger_is_better=MAXIMIZE_RESPONSE,
        topk=3,
    )

    # 旧列名兼容。
    pred_top1_hit = response_metrics["response_pred_top1_hit"]
    pred_top3_hit = response_metrics["response_pred_top3_hit"]
    pred_best_distance = response_metrics["response_pred_best_distance"]
    pred_regret = response_metrics["response_pred_regret"]
    observed_top1_hit = response_metrics["response_observed_top1_hit"]
    observed_top3_hit = response_metrics["response_observed_top3_hit"]
    observed_best_distance = response_metrics["response_observed_best_distance"]
    observed_regret = response_metrics["response_observed_regret"]
    true_best = response_metrics["response_true_best_flat"]
    pred_best = response_metrics["response_pred_best_flat"]
    obs_best_idx = response_metrics["response_observed_best_flat"]
    true_best_value = response_metrics["response_true_best_value"]
    pred_best_true_value = response_metrics["response_pred_best_true_value"]
    obs_best_value = response_metrics["response_observed_best_true_value"]

    pred_surface = pred_flat.reshape(GRID_SIZE_A, GRID_SIZE_B)
    surface_shape_metrics = surface_shape_diagnostics(block["Y"], pred_surface)

    # 协同/残差热点指标：true synergy 用 Bliss-like 单药预期作为参照。
    bliss = np.asarray(block["grid_features"][:, 10], dtype=float)
    true_synergy = y_flat - bliss
    if synergy_mean is None:
        pred_synergy = pred_flat - bliss
    else:
        pred_synergy = np.asarray(synergy_mean, dtype=float)
    synergy_metrics = _top_region_metrics(
        true_score=true_synergy,
        pred_score=pred_synergy,
        obs_mask=obs_mask,
        prefix="synergy",
        larger_is_better=True,
        topk=3,
    )

    # residual/evidence summary
    epsilon_mean_arr = np.asarray(epsilon_mean, dtype=float) if epsilon_mean is not None else np.full_like(pred_flat, np.nan, dtype=float)
    epsilon_std_arr = np.asarray(epsilon_std, dtype=float) if epsilon_std is not None else np.full_like(pred_flat, np.nan, dtype=float)
    synergy_std_arr = np.asarray(synergy_std, dtype=float) if synergy_std is not None else np.full_like(pred_flat, np.nan, dtype=float)

    if miss_mask.sum() > 0:
        miss_bool = miss_mask.astype(bool)
        mean_epsilon_unobserved = float(np.nanmean(epsilon_mean_arr[miss_bool]))
        mean_positive_epsilon_unobserved = float(np.nanmean(np.maximum(epsilon_mean_arr[miss_bool], 0.0)))
        mean_abs_epsilon_unobserved = float(np.nanmean(np.abs(epsilon_mean_arr[miss_bool])))
        mean_synergy_unobserved = float(np.nanmean(pred_synergy[miss_bool]))
        mean_positive_synergy_unobserved = float(np.nanmean(np.maximum(pred_synergy[miss_bool], 0.0)))
        mean_synergy_uncertainty_unobserved = float(np.nanmean(synergy_std_arr[miss_bool]))
    else:
        mean_epsilon_unobserved = 0.0
        mean_positive_epsilon_unobserved = 0.0
        mean_abs_epsilon_unobserved = 0.0
        mean_synergy_unobserved = 0.0
        mean_positive_synergy_unobserved = 0.0
        mean_synergy_uncertainty_unobserved = 0.0

    # dose-point cold-start 专用指标：本版只跑 block_random_split，保留字段兼容旧代码。
    eval_dose_mask = np.asarray(block.get("eval_dose_mask", np.zeros(N_GRID, dtype=np.float32)), dtype=np.float32)
    if eval_dose_mask.sum() > 0:
        held_bool = eval_dose_mask.astype(bool)
        rmse_heldout_dose = float(np.sqrt(mean_squared_error(y_flat[held_bool], pred_flat[held_bool])))
        mae_heldout_dose = float(mean_absolute_error(y_flat[held_bool], pred_flat[held_bool]))
        pcc_heldout_dose = safe_pcc(y_flat[held_bool], pred_flat[held_bool])
    else:
        rmse_heldout_dose = np.nan
        mae_heldout_dose = np.nan
        pcc_heldout_dose = np.nan

    out = {
        "rmse_all": rmse_all,
        "mae_all": mae_all,
        "pcc_all": pcc_all,
        "c_index_all": cindex_all,
        "rmse_missing": rmse_missing,
        "mae_missing": mae_missing,
        "pcc_missing": pcc_missing,
        "adjacent_delta_corr": adjacent_delta_corr(block["Y"], pred_surface),
        "pred_top1_hit": pred_top1_hit,
        "pred_top3_hit": pred_top3_hit,
        "pred_best_distance": pred_best_distance,
        "pred_regret": pred_regret,
        "observed_top1_hit": observed_top1_hit,
        "observed_top3_hit": observed_top3_hit,
        "observed_best_distance": observed_best_distance,
        "observed_regret": observed_regret,
        "true_best_flat": true_best,
        "pred_best_flat": pred_best,
        "observed_best_flat": obs_best_idx,
        "true_best_value": true_best_value,
        "pred_best_true_value": pred_best_true_value,
        "observed_best_true_value": obs_best_value,
        "mean_uncertainty_unobserved": float(np.mean(pred_std[miss_mask.astype(bool)])) if miss_mask.sum() > 0 else 0.0,
        "mean_epsilon_unobserved": mean_epsilon_unobserved,
        "mean_positive_epsilon_unobserved": mean_positive_epsilon_unobserved,
        "mean_abs_epsilon_unobserved": mean_abs_epsilon_unobserved,
        "mean_synergy_unobserved": mean_synergy_unobserved,
        "mean_positive_synergy_unobserved": mean_positive_synergy_unobserved,
        "mean_synergy_uncertainty_unobserved": mean_synergy_uncertainty_unobserved,
        "rmse_heldout_dose": rmse_heldout_dose,
        "mae_heldout_dose": mae_heldout_dose,
        "pcc_heldout_dose": pcc_heldout_dose,
    }
    out.update(response_metrics)
    out.update(synergy_metrics)
    out.update(surface_shape_metrics)
    return out

def simulate_active_one_block(models, block, strategy, repeat_seed, total_budgets):
    """
    在单个 block 上模拟主动剂量点选择。
    注意：这里的“更新模型”是 amortized update：揭示新点后更新 obs_mask/obs_values，
    再重新前向传播得到新的曲面预测和不确定性；不在每个测试 block 上重新训练网络参数。
    """
    rng = np.random.default_rng(repeat_seed)
    y_flat = block["Y"].reshape(-1).astype(np.float32)

    obs_mask = initial_observed_mask(ACTIVE_INITIAL_SEED_MODE, rng)
    selected_history = np.where(obs_mask > 0.5)[0].tolist()
    rows = []
    traj_rows = []

    max_budget = int(max(total_budgets))
    round_id = 0

    while int(obs_mask.sum()) < max_budget:
        comp = ensemble_predict_components_flat(models, block, obs_mask)
        need = min(ACTIVE_BATCH_SIZE, max_budget - int(obs_mask.sum()))
        next_points = select_next_points(
            strategy=strategy,
            pred_mean=comp["pred_mean"],
            pred_std=comp["pred_std"],
            obs_mask=obs_mask,
            batch_size=need,
            rng=rng,
            epsilon_mean=comp["epsilon_mean"],
            epsilon_std=comp["epsilon_std"],
            synergy_mean=comp["synergy_mean"],
            synergy_std=comp["synergy_std"],
        )
        if len(next_points) == 0:
            break

        round_id += 1
        for idx in next_points:
            obs_mask[idx] = 1.0
            selected_history.append(int(idx))
            if SAVE_ACTIVE_TRAJECTORY:
                i, j = flat_to_ij(idx)
                traj_rows.append({
                    "block_id": block["block_id"],
                    "drugA_name": block["drugA_name"],
                    "drugB_name": block["drugB_name"],
                    "cell_line": block["cell_line"],
                    "strategy": strategy,
                    "repeat_seed": repeat_seed,
                    "round": round_id,
                    "selected_flat": int(idx),
                    "selected_i": i,
                    "selected_j": j,
                    "doseA": float(block["doseA_levels"][i]),
                    "doseB": float(block["doseB_levels"][j]),
                    "true_response": float(y_flat[idx]),
                    "pred_response_before_measurement": float(comp["pred_mean"][idx]),
                    "pred_uncertainty_before_measurement": float(comp["pred_std"][idx]),
                    "pred_epsilon_before_measurement": float(comp["epsilon_mean"][idx]),
                    "pred_synergy_before_measurement": float(comp["synergy_mean"][idx]),
                    "true_bliss_expected": float(comp["bliss"][idx]),
                    "true_bliss_synergy": float(y_flat[idx] - comp["bliss"][idx]),
                    "observed_count_after": int(obs_mask.sum()),
                })

        current_budget = int(obs_mask.sum())
        if current_budget in total_budgets:
            comp = ensemble_predict_components_flat(models, block, obs_mask)
            met = evaluate_active_state(
                block=block,
                pred_mean=comp["pred_mean"],
                pred_std=comp["pred_std"],
                obs_mask=obs_mask,
                epsilon_mean=comp["epsilon_mean"],
                epsilon_std=comp["epsilon_std"],
                synergy_mean=comp["synergy_mean"],
                synergy_std=comp["synergy_std"],
            )
            row = {
                "block_id": block["block_id"],
                "drugA_name": block["drugA_name"],
                "drugB_name": block["drugB_name"],
                "cell_line": block["cell_line"],
                "strategy": strategy,
                "repeat_seed": repeat_seed,
                "budget": current_budget,
                "round": round_id,
                "initial_seed_mode": ACTIVE_INITIAL_SEED_MODE,
                "active_batch_size": ACTIVE_BATCH_SIZE,
                "selected_history": ";".join(map(str, selected_history)),
            }
            row.update(met)
            rows.append(row)

    # 如果初始 seed 本身就等于某些 budget，也补记一次。
    initial_k = int(initial_observed_mask(ACTIVE_INITIAL_SEED_MODE, np.random.default_rng(repeat_seed)).sum())
    recorded_budgets = set([r["budget"] for r in rows])
    if initial_k in total_budgets and initial_k not in recorded_budgets:
        init_mask = initial_observed_mask(ACTIVE_INITIAL_SEED_MODE, np.random.default_rng(repeat_seed))
        comp = ensemble_predict_components_flat(models, block, init_mask)
        met = evaluate_active_state(
            block=block,
            pred_mean=comp["pred_mean"],
            pred_std=comp["pred_std"],
            obs_mask=init_mask,
            epsilon_mean=comp["epsilon_mean"],
            epsilon_std=comp["epsilon_std"],
            synergy_mean=comp["synergy_mean"],
            synergy_std=comp["synergy_std"],
        )
        row = {
            "block_id": block["block_id"],
            "drugA_name": block["drugA_name"],
            "drugB_name": block["drugB_name"],
            "cell_line": block["cell_line"],
            "strategy": strategy,
            "repeat_seed": repeat_seed,
            "budget": initial_k,
            "round": 0,
            "initial_seed_mode": ACTIVE_INITIAL_SEED_MODE,
            "active_batch_size": ACTIVE_BATCH_SIZE,
            "selected_history": ";".join(map(str, np.where(init_mask > 0.5)[0].tolist())),
        }
        row.update(met)
        rows.insert(0, row)

    return rows, traj_rows

def run_active_simulation(models, test_blocks, setting_dir, setting_name):
    """在整个测试集上运行 retrospective active-learning simulation。"""
    active_rows = []
    traj_rows = []
    budgets = set(map(int, ACTIVE_TOTAL_BUDGETS))

    print("\n========== SAFIR-Active simulation ==========")
    print("Strategies:", ACTIVE_STRATEGIES)
    print("Initial seed:", ACTIVE_INITIAL_SEED_MODE)
    print("Batch size:", ACTIVE_BATCH_SIZE)
    print("Budgets:", ACTIVE_TOTAL_BUDGETS)
    print("Repeats:", ACTIVE_REPEATS)

    for strategy in ACTIVE_STRATEGIES:
        print(f"Running active strategy: {strategy}")
        for r in range(ACTIVE_REPEATS):
            # uncertainty-only：不同 repeat 使用不同 random2 初始观测，
            # 同一 repeat 内每个 block 的初始随机种子可复现。
            repeat_seed_base = RANDOM_STATE + 10000 + r * 100000
            for bi, block in enumerate(test_blocks):
                paired_seed = repeat_seed_base + bi
                rows, traj = simulate_active_one_block(
                    models=models,
                    block=block,
                    strategy=strategy,
                    repeat_seed=paired_seed,
                    total_budgets=budgets,
                )
                active_rows.extend(rows)
                traj_rows.extend(traj)

    active_df = pd.DataFrame(active_rows)
    active_path = setting_dir / f"{setting_name}_active_dose_selection_metrics.csv"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    active_df.to_csv(active_path, index=False, encoding="utf-8-sig")

    if SAVE_ACTIVE_TRAJECTORY:
        traj_df = pd.DataFrame(traj_rows)
        traj_path = setting_dir / f"{setting_name}_active_dose_selection_trajectory.csv"
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_df.to_csv(traj_path, index=False, encoding="utf-8-sig")
    else:
        traj_df = None

    summary_df = summarize_active_metrics(active_df)
    summary_path = setting_dir / f"{setting_name}_active_dose_selection_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("Saved active metrics:", active_path)
    print("Saved active summary:", summary_path)
    print(summary_df)
    return active_df, summary_df, traj_df


def summarize_active_metrics(active_df):
    group_cols = ["strategy", "budget"]
    metric_cols = [
        # 矩阵补全质量
        "rmse_all", "mae_all", "pcc_all", "c_index_all",
        "rmse_missing", "mae_missing", "pcc_missing", "adjacent_delta_corr",
        "surface_gradient_rmse", "surface_gradient_mae", "surface_gradient_corr",
        "surface_laplacian_rmse", "surface_laplacian_mae", "surface_laplacian_corr",
        "mean_uncertainty_unobserved",
        # 高响应区域发现
        "pred_top1_hit", "pred_top3_hit", "pred_best_distance", "pred_regret",
        "observed_top1_hit", "observed_top3_hit", "observed_best_distance", "observed_regret",
        "response_pred_top1_hit", "response_pred_top3_hit", "response_pred_best_distance", "response_pred_regret",
        "response_observed_top1_hit", "response_observed_top3_hit", "response_observed_best_distance", "response_observed_regret",
        "response_top25_jaccard_pred", "response_top25_recall_observed",
        # 残差/协同热点发现
        "synergy_pred_top1_hit", "synergy_pred_top3_hit", "synergy_pred_best_distance", "synergy_pred_regret",
        "synergy_observed_top1_hit", "synergy_observed_top3_hit", "synergy_observed_best_distance", "synergy_observed_regret",
        "synergy_top25_jaccard_pred", "synergy_top25_recall_observed",
        "mean_epsilon_unobserved", "mean_positive_epsilon_unobserved", "mean_abs_epsilon_unobserved",
        "mean_synergy_unobserved", "mean_positive_synergy_unobserved", "mean_synergy_uncertainty_unobserved",
        # 兼容旧 dose-point cold-start 字段，本版 block_random_split 下为 NaN。
        "rmse_heldout_dose", "mae_heldout_dose", "pcc_heldout_dose",
    ]
    metric_cols = [c for c in metric_cols if c in active_df.columns]
    agg = active_df.groupby(group_cols)[metric_cols].mean().reset_index()
    agg = agg.sort_values(["budget", "strategy"]).reset_index(drop=True)
    return agg

@torch.no_grad()
def predict_blocks_with_mask_ensemble(models, blocks, observed_k=0, mask_repeat=1, seed=42):
    """固定 K 的 ensemble 评估，用于保留原先 K=0/2/... 的补全表。"""
    rng = np.random.default_rng(seed)
    rows = []
    for repeat in range(mask_repeat):
        for b in blocks:
            y_flat = b["Y"].reshape(-1).astype(np.float32)
            obs_mask = make_observed_mask(N_GRID, observed_k, rng)
            pred_mean, pred_std, _ = ensemble_predict_flat(models, b, obs_mask)
            met = evaluate_active_state(b, pred_mean, pred_std, obs_mask)
            row = {
                "repeat": repeat,
                "block_id": b["block_id"],
                "drugA_name": b["drugA_name"],
                "drugB_name": b["drugB_name"],
                "cell_line": b["cell_line"],
                "observed_k": observed_k,
            }
            # 兼容原 summary 字段名
            row.update({
                "rmse_all": met["rmse_all"],
                "mae_all": met["mae_all"],
                "pcc_all": met["pcc_all"],
                "r2_all": float(r2_score(y_flat, pred_mean)) if np.std(y_flat) > 1e-12 else np.nan,
                "c_index_all": met["c_index_all"],
                "rmse_missing": met["rmse_missing"],
                "mae_missing": met["mae_missing"],
                "pcc_missing": met["pcc_missing"],
                "c_index_missing": np.nan,
                "adjacent_delta_corr": met["adjacent_delta_corr"],
                "top1_hit": met["pred_top1_hit"],
                "top3_hit": met["pred_top3_hit"],
                "best_dose_distance": met["pred_best_distance"],
                "true_best_flat": met["true_best_flat"],
                "pred_best_flat": met["pred_best_flat"],
                "mean_uncertainty_unobserved": met["mean_uncertainty_unobserved"],
                "rmse_heldout_dose": met.get("rmse_heldout_dose", np.nan),
                "mae_heldout_dose": met.get("mae_heldout_dose", np.nan),
                "pcc_heldout_dose": met.get("pcc_heldout_dose", np.nan),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def train_single_model_for_ensemble(tr_blocks, val_blocks, setting_dir, model_mode_config, seed, ensemble_id):
    """训练一个 SAFIR-Surface 预测引擎，作为 ensemble 成员。"""
    seed_everything(seed)
    train_loader = make_loader_with_seed(
        tr_blocks,
        TRAIN_OBSERVED_K_LIST,
        training=True,
        shuffle=True,
        seed=seed,
        drug_swap_augmentation=DRUG_SWAP_AUGMENTATION,
        drug_swap_prob=DRUG_SWAP_PROB,
    )
    # Validation keeps the original A/B orientation. It still samples Random Mask-K
    # exactly as before, but does NOT receive drug-swap augmentation.
    val_loader = make_loader_with_seed(
        val_blocks,
        TRAIN_OBSERVED_K_LIST,
        training=True,
        shuffle=False,
        seed=seed + 17,
        drug_swap_augmentation=False,
        drug_swap_prob=0.0,
    )

    drug_dim = tr_blocks[0]["drugA_feat"].shape[0]
    cell_dim = tr_blocks[0]["cell_feat"].shape[0]
    grid_feat_dim = tr_blocks[0]["grid_features"].shape[1]

    model = SAFIRDoseSurfaceCompletionNet(
        drug_dim=drug_dim,
        cell_dim=cell_dim,
        grid_feat_dim=grid_feat_dim,
        use_grid_gnn=model_mode_config["use_grid_gnn"],
        use_local_obs=model_mode_config["use_local_obs"],
        use_residual_anchor=model_mode_config["use_residual_anchor"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val = float("inf")
    best_epoch = None
    best_state = None
    epochs_no_improve = 0
    history = []

    print(f"\n--- Train ensemble member {ensemble_id} | seed={seed} ---")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(batch)
            loss, loss_dict = compute_loss(out, batch)
            loss.backward()
            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            epoch_losses.append(loss_dict)

        train_loss = float(np.mean([x["loss_total"] for x in epoch_losses]))
        train_recon = float(np.mean([x["loss_reconstruction"] for x in epoch_losses]))
        train_missing = float(np.mean([x["loss_missing"] for x in epoch_losses]))
        train_obs = float(np.mean([x["loss_obs"] for x in epoch_losses]))
        train_delta = float(np.mean([x["loss_delta"] for x in epoch_losses]))
        train_residual = float(np.mean([x["loss_residual_consistency"] for x in epoch_losses]))
        val_loss = evaluate_val_loss(model, val_loader)
        scheduler.step(val_loss)

        history.append({
            "ensemble_id": ensemble_id,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_reconstruction_loss": train_recon,
            "train_missing_loss": train_missing,
            "train_observed_loss": train_obs,
            "train_delta_loss": train_delta,
            "train_residual_consistency_loss": train_residual,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        })

        print(
            f"Ensemble {ensemble_id} | Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"Train {train_loss:.5f} | Recon {train_recon:.5f} | "
            f"Delta {train_delta:.5f} | Residual {train_residual:.5f} | Val {val_loss:.5f}"
        )

        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save(best_state, setting_dir / f"best_model_ensemble{ensemble_id}_seed{seed}.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping ensemble {ensemble_id} at epoch {epoch}. Best epoch={best_epoch}, val={best_val:.6f}")
            break

    pd.DataFrame(history).to_csv(
        setting_dir / f"training_history_ensemble{ensemble_id}_seed{seed}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"ensemble_id": ensemble_id, "seed": seed, "best_epoch": best_epoch, "best_val_loss": best_val}


def train_one_setting_active(train_blocks, val_blocks, test_blocks, setting_name, model_mode_name, model_mode_config):
    """训练 ensemble 预测引擎，并进行固定 K 评估 + 主动剂量点选择模拟。"""
    setting_dir = RESULT_DIR / setting_name
    setting_dir.mkdir(parents=True, exist_ok=True)

    # Train/Val/Test 已在 CRISPR 特征拟合之前固定；这里禁止重新划分。
    tr_blocks = list(train_blocks)
    val_blocks = list(val_blocks)

    print(f"\n========== Start SAFIR-Active training: {setting_name} ==========")
    print("Train blocks:", len(tr_blocks))
    print("Val blocks:", len(val_blocks))
    print("Test blocks:", len(test_blocks))
    print("Train observed K list:", TRAIN_OBSERVED_K_LIST)
    print("Drug-swap augmentation:", DRUG_SWAP_AUGMENTATION, "prob=", DRUG_SWAP_PROB)
    print("Active ensemble size:", ACTIVE_ENSEMBLE_SIZE)

    models = []
    ensemble_info = []
    seeds = ACTIVE_ENSEMBLE_SEEDS[:ACTIVE_ENSEMBLE_SIZE]
    for eid, seed in enumerate(seeds):
        model, info = train_single_model_for_ensemble(
            tr_blocks=tr_blocks,
            val_blocks=val_blocks,
            setting_dir=setting_dir,
            model_mode_config=model_mode_config,
            seed=int(seed),
            ensemble_id=eid,
        )
        models.append(model)
        ensemble_info.append(info)

    pd.DataFrame(ensemble_info).to_csv(setting_dir / "ensemble_training_summary.csv", index=False, encoding="utf-8-sig")

    # 固定 K 评估：保留原 SAFIR-Surface 补全结果，用 ensemble mean 做预测。
    fixed_k_metrics = []
    for k in EVAL_OBSERVED_K_LIST:
        repeat = EVAL_MASK_REPEATS if k > 0 else 1
        mdf = predict_blocks_with_mask_ensemble(
            models=models,
            blocks=test_blocks,
            observed_k=k,
            mask_repeat=repeat,
            seed=RANDOM_STATE + 2000 + k,
        )
        fixed_k_metrics.append(mdf)
    fixed_k_metrics_df = pd.concat(fixed_k_metrics, axis=0, ignore_index=True)
    fixed_k_summary_df = summarize_metrics(fixed_k_metrics_df)
    fixed_k_metrics_df.to_csv(setting_dir / f"{setting_name}_ensemble_fixedK_metrics.csv", index=False, encoding="utf-8-sig")
    fixed_k_summary_df.to_csv(setting_dir / f"{setting_name}_ensemble_fixedK_summary.csv", index=False, encoding="utf-8-sig")

    # 主动剂量点选择模拟：这是升级后的核心实验。
    if ACTIVE_ENABLE:
        active_df, active_summary_df, traj_df = run_active_simulation(
            models=models,
            test_blocks=test_blocks,
            setting_dir=setting_dir,
            setting_name=setting_name,
        )
    else:
        active_df = pd.DataFrame()
        active_summary_df = pd.DataFrame()

    config = {
        "setting_name": setting_name,
        "model_mode_name": model_mode_name,
        "model_mode_desc": model_mode_config.get("desc", ""),
        "use_grid_gnn": model_mode_config["use_grid_gnn"],
        "use_local_obs": model_mode_config["use_local_obs"],
        "use_residual_anchor": model_mode_config["use_residual_anchor"],
        "cell_feature_type": model_mode_config["cell_feature_type"],
        "main_experiment_type": model_mode_config.get("main_experiment_type", "unknown"),
        "active_cell_feature_file": str(ACTIVE_CELL_FEATURE_FILE),
        "split_reference_file": str(SPLIT_REFERENCE_FILE),
        "crispr_preprocessor_file": None,
        "leakage_control": "split first; Top1000/median/scaler fit on train cells only; val/test transform only; test excluded from model selection",
        "train_blocks": len(tr_blocks),
        "val_blocks": len(val_blocks),
        "test_blocks": len(test_blocks),
        "train_observed_k_list": TRAIN_OBSERVED_K_LIST,
        "eval_observed_k_list": EVAL_OBSERVED_K_LIST,
        "global_fusion": "ordered_concat_[hA,hB,hA*hB,abs(hA-hB),hC]",
        "drug_swap_augmentation": DRUG_SWAP_AUGMENTATION,
        "drug_swap_probability": DRUG_SWAP_PROB,
        "drug_swap_scope": "training_only; validation/test/fixed-K/active evaluation remain original orientation",
        "active_enable": ACTIVE_ENABLE,
        "active_ensemble_size": ACTIVE_ENSEMBLE_SIZE,
        "active_ensemble_seeds": seeds,
        "active_initial_seed_mode": ACTIVE_INITIAL_SEED_MODE,
        "active_batch_size": ACTIVE_BATCH_SIZE,
        "active_total_budgets": ACTIVE_TOTAL_BUDGETS,
        "active_repeats": ACTIVE_REPEATS,
        "active_strategies": ACTIVE_STRATEGIES,
        "maximize_response": MAXIMIZE_RESPONSE,
        "num_epochs_max": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "lambda_obs_recon": LAMBDA_OBS_RECON,
        "lambda_delta": LAMBDA_DELTA,
        "lambda_residual_consistency": LAMBDA_RESIDUAL_CONSISTENCY,
        "loss_objective": "reconstruction + lambda_delta*delta + lambda_residual*residual_consistency",
        "model_name": "SAFIR-Active-OrderedFusion-DrugSwapAug-UncertaintyOnly-SimplifiedLoss",
        "important_note": "Active update means updating obs_mask/obs_values with newly revealed dose points and recomputing ensemble predictions; network parameters are not refit inside each test block.",
    }
    with open(setting_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n========== Finished SAFIR-Active setting: {setting_name} ==========")
    print("Fixed-K summary:")
    print(fixed_k_summary_df)
    if ACTIVE_ENABLE:
        print("Active summary:")
        print(active_summary_df)

    return models, fixed_k_summary_df, fixed_k_metrics_df, active_summary_df, active_df
