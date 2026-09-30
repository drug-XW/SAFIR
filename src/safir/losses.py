"""Training objective for SAFIR."""
import torch
import torch.nn.functional as F

from .config import LAMBDA_OBS_RECON, LAMBDA_DELTA, LAMBDA_RESIDUAL_CONSISTENCY

def delta_loss(pred_surface, true_surface, loss_mask=None):
    """
    相邻剂量变化趋势约束。
    若未来使用 dose holdout，只有相邻两个点都允许参与训练时才计算该差分，
    防止 held-out true Y 通过 delta loss 间接进入梯度。
    当前 block-random split 下 loss_mask 全1，因此不改变主实验定义。
    """
    dpa = pred_surface[:, 1:, :] - pred_surface[:, :-1, :]
    dta = true_surface[:, 1:, :] - true_surface[:, :-1, :]
    dpb = pred_surface[:, :, 1:] - pred_surface[:, :, :-1]
    dtb = true_surface[:, :, 1:] - true_surface[:, :, :-1]

    if loss_mask is None:
        return (F.mse_loss(dpa, dta) + F.mse_loss(dpb, dtb)) / 2.0

    mask = loss_mask.reshape_as(true_surface)
    ma = mask[:, 1:, :] * mask[:, :-1, :]
    mb = mask[:, :, 1:] * mask[:, :, :-1]

    la = (((dpa - dta) ** 2) * ma).sum() / ma.sum().clamp(min=1.0)
    lb = (((dpb - dtb) ** 2) * mb).sum() / mb.sum().clamp(min=1.0)
    return (la + lb) / 2.0


def compute_loss(outputs, batch):
    """
    Simplified SAFIR objective:
        L = L_reconstruction + lambda_delta * L_delta
            + lambda_residual * L_residual_consistency

    where
        L_reconstruction = L_missing + lambda_obs * L_observed.

    Removed from the original objective:
        - smoothness regularization
        - epsilon L2 regularization
    """
    pred = outputs["pred_flat"]
    y = batch["Y_flat"]
    obs_mask = batch["obs_mask"]
    loss_mask = batch.get("loss_mask", torch.ones_like(obs_mask))

    miss_mask = (1.0 - obs_mask) * loss_mask
    obs_loss_mask = obs_mask * loss_mask

    # 1) Reconstruction loss: missing positions are the primary target;
    #    observed positions receive a lower-weight consistency term.
    miss_count = miss_mask.sum().clamp(min=1.0)
    obs_count = obs_loss_mask.sum().clamp(min=1.0)

    loss_missing = (((pred - y) ** 2) * miss_mask).sum() / miss_count
    loss_obs = (((pred - y) ** 2) * obs_loss_mask).sum() / obs_count
    loss_reconstruction = loss_missing + LAMBDA_OBS_RECON * loss_obs

    # 2) Landscape local-difference consistency along the two dose axes.
    loss_delta = delta_loss(
        outputs["pred_surface"],
        batch["Y"],
        loss_mask=loss_mask,
    )

    # 3) Observation-guided residual consistency.
    #    epsilon should approximate Y - Base at observed positions.
    #    Base is detached so this auxiliary supervision only updates the residual branch.
    base_flat = outputs["base_surface"].reshape_as(y)
    if outputs["epsilon_flat"].requires_grad:
        true_obs_residual = (y - base_flat.detach()) * obs_loss_mask
        loss_residual_consistency = (
            ((outputs["epsilon_flat"] - true_obs_residual) ** 2) * obs_loss_mask
        ).sum() / obs_count
    else:
        loss_residual_consistency = pred.new_tensor(0.0)

    total = (
        loss_reconstruction
        + LAMBDA_DELTA * loss_delta
        + LAMBDA_RESIDUAL_CONSISTENCY * loss_residual_consistency
    )

    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_reconstruction": float(loss_reconstruction.detach().cpu()),
        "loss_missing": float(loss_missing.detach().cpu()),
        "loss_obs": float(loss_obs.detach().cpu()),
        "loss_delta": float(loss_delta.detach().cpu()),
        "loss_residual_consistency": float(loss_residual_consistency.detach().cpu()),
    }
