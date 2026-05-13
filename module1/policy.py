"""Full schedule policy: GIN encoder + reduced-basis decoder + Gaussian policy.

Provides two interfaces:
    Training   — ``forward()`` / ``sample_schedule()`` on PyG Batches
                  (returns logprobs, values, entropy for PPO / REINFORCE).
    Inference  — ``make_schedule(nx.Graph) -> GlobalSchedule``
                  (numpy-backed, compatible with the ScheduleModel contract).
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch_geometric.data import Batch

from config import ProjectConfig, compute_blockade_omega
from schedules import GlobalSchedule
from module1.base import ScheduleModel
from module1.featurize import graph_to_pyg, K_PE_DEFAULT
from module1.encoder import GINEncoder
from module1.heads import (
    AnalyticOmega,
    DeltaHead,
    FourierDeltaHead,
    FourierOmegaHead,
    OmegaHead,
)

GRAPH_FEAT_DIM = 4  # (n_nodes_norm, n_edges_norm, lambda_2, density)


class SchedulePolicy(nn.Module, ScheduleModel):
    """GIN → reduced-basis policy network for Rydberg MIS schedules.

    Inherits from both ``nn.Module`` (for PyTorch training) and
    ``ScheduleModel`` (so evaluation code can treat it interchangeably
    with simpler baselines like ``FixedScheduleBaseline``).

    Parameters
    ----------
    config : ProjectConfig
        Full project config (controls, udg, backend).
    hidden_dim : int
        Hidden width for the GIN layers, decoder MLPs, and value head.
    n_gnn_layers : int
        Number of GIN message-passing layers.
    k_pe : int
        Number of Laplacian eigenvector features per node.
    init_log_std : float
        Initial value for the per-parameter log-std of the Gaussian policy.
    """

    def __init__(
        self,
        config: ProjectConfig,
        hidden_dim: int = 64,
        n_gnn_layers: int = 3,
        k_pe: int = K_PE_DEFAULT,
        init_log_std: float = -1.0,
    ):
        nn.Module.__init__(self)
        self.config = config
        self.k_pe = k_pe
        node_feat_dim = 3 + k_pe  # degree + clustering + triangles + PE

        self.encoder = GINEncoder(node_feat_dim, hidden_dim, n_gnn_layers)
        embed_dim = self.encoder.out_dim + GRAPH_FEAT_DIM

        controls = config.controls
        arch = controls.architecture

        if controls.learn_omega:
            if arch == 1:
                self.omega_head: nn.Module = OmegaHead(embed_dim, controls)
            elif arch == 2:
                self.omega_head = FourierOmegaHead(embed_dim, controls)
            else:
                raise ValueError(f"Unknown architecture {arch}")
        else:
            hw = config.hardware
            R_b_um = config.udg.radius * config.udg.spacing
            omega_peak = compute_blockade_omega(hw.C6, R_b_um, hw.omega_max)
            T_us = controls.T * 1e6
            self.omega_head = AnalyticOmega(
                embed_dim,
                N_t=controls.N_t,
                omega_peak=omega_peak,
                t_ramp_frac=hw.t_ramp / T_us,
                t_onset_frac=hw.t_onset / T_us,
            )

        if arch == 1:
            self.delta_head: nn.Module = DeltaHead(embed_dim, controls)
        elif arch == 2:
            self.delta_head = FourierDeltaHead(embed_dim, controls)
        else:
            raise ValueError(f"Unknown architecture {arch}")

        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        n_action_params = self.omega_head.n_params + self.delta_head.n_params
        self.log_std = nn.Parameter(torch.full((n_action_params,), init_log_std))

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def encode(self, batch: Batch) -> torch.Tensor:
        """Encode a PyG batch to graph-level embeddings."""
        h = self.encoder(batch.x, batch.edge_index, batch.batch)
        return torch.cat([h, batch.graph_feats], dim=-1)

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        """Deterministic forward: mean schedule parameters + value estimate."""
        embed = self.encode(batch)
        omega_params, omega = self.omega_head(embed)
        delta_params, delta = self.delta_head(embed)
        value = self.value_head(embed).squeeze(-1)
        mean_params = torch.cat([omega_params, delta_params], dim=-1)
        return {
            "mean_params": mean_params,
            "omega": omega,
            "delta": delta,
            "value": value,
            "embed": embed,
        }

    # ------------------------------------------------------------------
    # Stochastic policy interface (for REINFORCE / PPO)
    # ------------------------------------------------------------------

    def action_dist(self, mean_params: torch.Tensor) -> Normal:
        """Diagonal Gaussian over the latent action parameters."""
        std = self.log_std.exp().expand_as(mean_params)
        return Normal(mean_params, std)

    def sample_schedule(
        self, batch: Batch, deterministic: bool = False
    ) -> dict[str, torch.Tensor]:
        """Sample schedules for a batch of graphs.

        Returns omega, delta tensors plus logprob, value, and entropy
        for policy-gradient training.
        """
        out = self.forward(batch)
        dist = self.action_dist(out["mean_params"])
        sampled = out["mean_params"] if deterministic else dist.rsample()
        logprob = dist.log_prob(sampled).sum(dim=-1)

        n_omega = self.omega_head.n_params
        omega_p, delta_p = sampled[:, :n_omega], sampled[:, n_omega:]
        if n_omega > 0:
            omega = self.omega_head.reconstruct(omega_p)
        else:
            _, omega = self.omega_head(out["embed"])
        delta = self.delta_head.reconstruct(delta_p)

        return {
            "omega": omega,
            "delta": delta,
            "sampled_params": sampled,
            "logprob": logprob,
            "value": out["value"],
            "entropy": dist.entropy().sum(dim=-1),
        }

    # ------------------------------------------------------------------
    # ScheduleModel-compatible inference interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def make_schedule(self, g: nx.Graph) -> GlobalSchedule:
        """Produce a deterministic schedule for a single graph.

        This satisfies the same contract as ``ScheduleModel.make_schedule``
        so that downstream code (Module 2 backends, Module 3 orchestration)
        can treat this model interchangeably with the simpler baselines.
        """
        data = graph_to_pyg(g, k_pe=self.k_pe)
        batch = Batch.from_data_list([data])
        device = next(self.parameters()).device
        batch = batch.to(device)

        out = self.sample_schedule(batch, deterministic=True)
        return GlobalSchedule(
            omega=out["omega"][0].cpu().numpy().astype(np.float64),
            delta=out["delta"][0].cpu().numpy().astype(np.float64),
            dt=self.config.controls.dt,
            param_kind=self.config.controls.param_kind,
        )
