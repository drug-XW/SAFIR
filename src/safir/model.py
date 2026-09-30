"""SAFIR model used in the released O'Neil experiment.

This file contains only the architecture exercised by the released main model:
ordered drug fusion + global/local sparse-observation encoders +
single-agent-informed reference + gated residual correction.
"""
import torch
import torch.nn as nn

from .config import (
    DRUG_EMB_DIM, CELL_EMB_DIM, GLOBAL_DIM, OBS_EMB_DIM, HIDDEN_DIM,
    DROPOUT, LOCAL_OBS_EMB_DIM, N_GRID, GRID_SIZE_A, GRID_SIZE_B,
)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.0, use_bn=True):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SAFIRDoseSurfaceCompletionNet(nn.Module):
    """Fixed SAFIR architecture used by the released O'Neil workflow."""

    def __init__(self, drug_dim, cell_dim, grid_feat_dim,
                 use_grid_gnn=False, use_local_obs=True, use_residual_anchor=True):
        super().__init__()
        if use_grid_gnn:
            raise ValueError("The released main model does not use Grid-GNN message passing.")
        if not use_local_obs:
            raise ValueError("The released main model requires local observation injection.")
        if not use_residual_anchor:
            raise ValueError("The released main model requires the reference-residual architecture.")

        self.drug_encoder = MLP(
            in_dim=drug_dim, hidden_dims=[512, 256], out_dim=DRUG_EMB_DIM,
            dropout=DROPOUT, use_bn=True,
        )
        self.cell_encoder = MLP(
            in_dim=cell_dim, hidden_dims=[512, 256], out_dim=CELL_EMB_DIM,
            dropout=DROPOUT, use_bn=True,
        )
        self.global_encoder = MLP(
            in_dim=DRUG_EMB_DIM * 4 + CELL_EMB_DIM,
            hidden_dims=[384], out_dim=GLOBAL_DIM,
            dropout=DROPOUT, use_bn=True,
        )
        self.obs_encoder = MLP(
            in_dim=N_GRID * 3 + 5, hidden_dims=[128], out_dim=OBS_EMB_DIM,
            dropout=DROPOUT, use_bn=True,
        )
        self.local_obs_encoder = nn.Sequential(
            nn.Linear(4, LOCAL_OBS_EMB_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(LOCAL_OBS_EMB_DIM, LOCAL_OBS_EMB_DIM),
            nn.ReLU(),
        )

        base_node_in_dim = GLOBAL_DIM + grid_feat_dim
        self.base_node_projector = nn.Sequential(
            nn.Linear(base_node_in_dim, HIDDEN_DIM), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
        )

        # No Grid-GNN in the released model, so the global observation embedding
        # is available directly to each dose-position residual predictor.
        eps_node_in_dim = GLOBAL_DIM + OBS_EMB_DIM + grid_feat_dim + LOCAL_OBS_EMB_DIM
        self.eps_node_projector = nn.Sequential(
            nn.Linear(eps_node_in_dim, HIDDEN_DIM), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
        )

        self.ab_head = nn.Linear(HIDDEN_DIM, 2)
        self.epsilon_head = nn.Linear(HIDDEN_DIM, 1)
        self.epsilon_gate = nn.Sequential(
            nn.Linear(OBS_EMB_DIM + GLOBAL_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def encode_observed_points(self, grid_features, obs_values, obs_mask):
        bliss_grid = grid_features[:, :, 10]
        obs_residual = (obs_values - bliss_grid) * obs_mask

        obs_count_raw = obs_mask.sum(dim=1, keepdim=True)
        obs_count = obs_count_raw.clamp(min=1.0)
        obs_mean = (obs_values * obs_mask).sum(dim=1, keepdim=True) / obs_count

        very_neg = torch.full_like(obs_values, -1e9)
        very_pos = torch.full_like(obs_values, 1e9)
        obs_max = torch.where(obs_mask > 0.5, obs_values, very_neg).max(dim=1, keepdim=True).values
        obs_min = torch.where(obs_mask > 0.5, obs_values, very_pos).min(dim=1, keepdim=True).values
        obs_max = torch.where(obs_count_raw > 0, obs_max, torch.zeros_like(obs_max))
        obs_min = torch.where(obs_count_raw > 0, obs_min, torch.zeros_like(obs_min))

        obs_var = (((obs_values - obs_mean) ** 2) * obs_mask).sum(dim=1, keepdim=True) / obs_count
        obs_std = torch.sqrt(obs_var + 1e-8)
        obs_ratio = obs_mask.mean(dim=1, keepdim=True)

        obs_summary = torch.cat([obs_mean, obs_max, obs_min, obs_std, obs_ratio], dim=1)
        obs_input = torch.cat([obs_values, obs_mask, obs_residual, obs_summary], dim=1)
        h_obs = self.obs_encoder(obs_input)

        local_obs_input = torch.stack([
            obs_mask,
            obs_values * obs_mask,
            obs_residual,
            torch.abs(obs_residual),
        ], dim=-1)
        batch_size, n_grid, _ = local_obs_input.shape
        h_local_obs = self.local_obs_encoder(
            local_obs_input.reshape(batch_size * n_grid, -1)
        ).reshape(batch_size, n_grid, -1)
        return h_obs, h_local_obs, obs_residual

    def forward(self, batch):
        drug_a = batch["drugA_feat"]
        drug_b = batch["drugB_feat"]
        cell = batch["cell_feat"]
        grid_features = batch["grid_features"]
        obs_values = batch["obs_values"]
        obs_mask = batch["obs_mask"]

        batch_size = drug_a.size(0)
        h_a = self.drug_encoder(drug_a)
        h_b = self.drug_encoder(drug_b)
        h_c = self.cell_encoder(cell)

        ordered_pair = torch.cat([
            h_a, h_b, h_a * h_b, torch.abs(h_a - h_b), h_c,
        ], dim=1)
        h_global = self.global_encoder(ordered_pair)
        h_obs, h_local_obs, obs_residual = self.encode_observed_points(
            grid_features, obs_values, obs_mask
        )

        h_global_rep = h_global.unsqueeze(1).expand(batch_size, N_GRID, -1)
        h_obs_rep = h_obs.unsqueeze(1).expand(batch_size, N_GRID, -1)

        base_node_input = torch.cat([h_global_rep, grid_features], dim=-1)
        h_base = self.base_node_projector(
            base_node_input.reshape(batch_size * N_GRID, -1)
        ).reshape(batch_size, N_GRID, HIDDEN_DIM)

        eps_node_input = torch.cat([
            h_global_rep, h_obs_rep, grid_features, h_local_obs,
        ], dim=-1)
        h_eps = self.eps_node_projector(
            eps_node_input.reshape(batch_size * N_GRID, -1)
        ).reshape(batch_size, N_GRID, HIDDEN_DIM)

        base_hidden = h_base.reshape(batch_size * N_GRID, HIDDEN_DIM)
        eps_hidden = h_eps.reshape(batch_size * N_GRID, HIDDEN_DIM)

        ab = torch.softmax(self.ab_head(base_hidden).reshape(batch_size, N_GRID, 2), dim=-1)
        alpha = ab[:, :, 0]
        beta = ab[:, :, 1]

        epsilon_base = self.epsilon_head(eps_hidden).reshape(batch_size, N_GRID)
        eps_gate = self.epsilon_gate(torch.cat([h_global, h_obs], dim=1))
        epsilon = epsilon_base * (0.5 + 1.5 * eps_gate)

        y_a_grid = grid_features[:, :, 5]
        y_b_grid = grid_features[:, :, 6]
        base = alpha * y_a_grid + beta * y_b_grid
        pred = base + epsilon

        return {
            "pred_flat": pred,
            "pred_surface": pred.reshape(batch_size, GRID_SIZE_A, GRID_SIZE_B),
            "epsilon_flat": epsilon,
            "epsilon_surface": epsilon.reshape(batch_size, GRID_SIZE_A, GRID_SIZE_B),
            "epsilon_base_surface": epsilon_base.reshape(batch_size, GRID_SIZE_A, GRID_SIZE_B),
            "alpha_surface": alpha.reshape(batch_size, GRID_SIZE_A, GRID_SIZE_B),
            "beta_surface": beta.reshape(batch_size, GRID_SIZE_A, GRID_SIZE_B),
            "base_surface": base.reshape(batch_size, GRID_SIZE_A, GRID_SIZE_B),
            "obs_residual_surface": obs_residual.reshape(batch_size, GRID_SIZE_A, GRID_SIZE_B),
        }
