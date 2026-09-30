"""Fixed-budget evaluation utilities."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .config import *
from .utils import (make_observed_mask, safe_pcc, concordance_index, adjacent_delta_corr,
                    surface_smoothness_np, topk_hit_metrics)
from .losses import compute_loss

def predict_blocks_with_mask(model, blocks, observed_k=0, mask_repeat=1, seed=42):
    model.eval()
    rng = np.random.default_rng(seed)

    records = []
    metric_rows = []

    for repeat in range(mask_repeat):
        for b in blocks:
            y_flat = b["Y"].reshape(-1).astype(np.float32)
            obs_mask = make_observed_mask(N_GRID, observed_k, rng)
            obs_values = y_flat * obs_mask

            batch = {
                "block_id": torch.tensor([b["block_id"]], dtype=torch.long, device=DEVICE),
                "drugA_feat": torch.tensor(b["drugA_feat"][None, :], dtype=torch.float32, device=DEVICE),
                "drugB_feat": torch.tensor(b["drugB_feat"][None, :], dtype=torch.float32, device=DEVICE),
                "cell_feat": torch.tensor(b["cell_feat"][None, :], dtype=torch.float32, device=DEVICE),
                "grid_features": torch.tensor(b["grid_features"][None, :, :], dtype=torch.float32, device=DEVICE),
                "YA": torch.tensor(b["YA"][None, :], dtype=torch.float32, device=DEVICE),
                "YB": torch.tensor(b["YB"][None, :], dtype=torch.float32, device=DEVICE),
                "Y": torch.tensor(b["Y"][None, :, :], dtype=torch.float32, device=DEVICE),
                "Y_flat": torch.tensor(y_flat[None, :], dtype=torch.float32, device=DEVICE),
                "obs_mask": torch.tensor(obs_mask[None, :], dtype=torch.float32, device=DEVICE),
                "obs_values": torch.tensor(obs_values[None, :], dtype=torch.float32, device=DEVICE),
                "loss_mask": torch.tensor(b.get("loss_mask", np.ones(N_GRID, dtype=np.float32))[None, :], dtype=torch.float32, device=DEVICE),
                "eval_dose_mask": torch.tensor(b.get("eval_dose_mask", np.zeros(N_GRID, dtype=np.float32))[None, :], dtype=torch.float32, device=DEVICE),
                "observed_k": torch.tensor([observed_k], dtype=torch.long, device=DEVICE),
            }

            out = model(batch)

            pred = out["pred_surface"][0].detach().cpu().numpy()
            eps = out["epsilon_surface"][0].detach().cpu().numpy()
            alpha = out["alpha_surface"][0].detach().cpu().numpy()
            beta = out["beta_surface"][0].detach().cpu().numpy()
            base = out["base_surface"][0].detach().cpu().numpy()
            y = b["Y"]

            miss_mask = 1.0 - obs_mask
            all_true = y.reshape(-1)
            all_pred = pred.reshape(-1)

            if miss_mask.sum() > 0:
                miss_true = all_true[miss_mask.astype(bool)]
                miss_pred = all_pred[miss_mask.astype(bool)]
                rmse_missing = float(np.sqrt(mean_squared_error(miss_true, miss_pred)))
                mae_missing = float(mean_absolute_error(miss_true, miss_pred))
                pcc_missing = safe_pcc(miss_true, miss_pred)
                c_index_missing = concordance_index(miss_true, miss_pred)
            else:
                rmse_missing = np.nan
                mae_missing = np.nan
                pcc_missing = np.nan
                c_index_missing = np.nan

            hit = topk_hit_metrics(y, pred, k=3)

            metric_rows.append({
                "repeat": repeat,
                "block_id": b["block_id"],
                "drugA_name": b["drugA_name"],
                "drugB_name": b["drugB_name"],
                "cell_line": b["cell_line"],
                "observed_k": observed_k,
                "rmse_all": float(np.sqrt(mean_squared_error(all_true, all_pred))),
                "mae_all": float(mean_absolute_error(all_true, all_pred)),
                "pcc_all": safe_pcc(all_true, all_pred),
                "r2_all": float(r2_score(all_true, all_pred)) if np.std(all_true) > 1e-12 else np.nan,
                "c_index_all": concordance_index(all_true, all_pred),
                "rmse_missing": rmse_missing,
                "mae_missing": mae_missing,
                "pcc_missing": pcc_missing,
                "c_index_missing": c_index_missing,
                "adjacent_delta_corr": adjacent_delta_corr(y, pred),
                "true_surface_smoothness": surface_smoothness_np(y),
                "pred_surface_smoothness": surface_smoothness_np(pred),
                "epsilon_surface_smoothness": surface_smoothness_np(eps),
                "top1_hit": hit["top1_hit"],
                "top3_hit": hit["top3_hit"],
                "best_dose_distance": hit["best_dose_distance"],
                "true_best_flat": hit["true_best_flat"],
                "pred_best_flat": hit["pred_best_flat"],
            })

            if SAVE_PREDICTIONS:
                for i in range(GRID_SIZE_A):
                    for j in range(GRID_SIZE_B):
                        flat_idx = i * GRID_SIZE_B + j
                        records.append({
                            "repeat": repeat,
                            "block_id": b["block_id"],
                            "drugA_name": b["drugA_name"],
                            "drugB_name": b["drugB_name"],
                            "cell_line": b["cell_line"],
                            "observed_k": observed_k,
                            "doseA_index": i,
                            "doseB_index": j,
                            "doseA": float(b["doseA_levels"][i]),
                            "doseB": float(b["doseB_levels"][j]),
                            "is_observed_input": int(obs_mask[flat_idx] > 0.5),
                            "single_resp_1": float(b["YA"][i]),
                            "single_resp_2": float(b["YB"][j]),
                            "target": float(y[i, j]),
                            "prediction": float(pred[i, j]),
                            "base": float(base[i, j]),
                            "epsilon": float(eps[i, j]),
                            "alpha": float(alpha[i, j]),
                            "beta": float(beta[i, j]),
                        })

    metrics_df = pd.DataFrame(metric_rows)
    pred_df = pd.DataFrame(records) if SAVE_PREDICTIONS else None

    return metrics_df, pred_df


def summarize_metrics(metrics_df):
    agg_spec = {
        "rmse_all": "mean",
        "mae_all": "mean",
        "pcc_all": "mean",
        "r2_all": "mean",
        "c_index_all": "mean",
        "rmse_missing": "mean",
        "mae_missing": "mean",
        "pcc_missing": "mean",
        "c_index_missing": "mean",
        "adjacent_delta_corr": "mean",
        "top1_hit": "mean",
        "top3_hit": "mean",
        "best_dose_distance": "mean",
        "rmse_heldout_dose": "mean",
        "mae_heldout_dose": "mean",
        "pcc_heldout_dose": "mean",
    }
    agg_spec = {k: v for k, v in agg_spec.items() if k in metrics_df.columns}
    agg = metrics_df.groupby("observed_k").agg(agg_spec).reset_index()
    return agg


def evaluate_val_loss(model, val_loader):
    model.eval()
    losses = []

    for batch in val_loader:
        batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = model(batch)
        loss, _ = compute_loss(out, batch)
        losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses))
