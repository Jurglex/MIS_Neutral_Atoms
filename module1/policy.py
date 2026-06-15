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
    MonotoneDeltaHead,
    MultiplicativeOmegaHead,
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
        hw = config.hardware
        R_b_um = config.udg.radius * config.udg.spacing

        if controls.learn_omega:
            if arch == 1:
                self.omega_head: nn.Module = OmegaHead(embed_dim, controls)
            elif arch == 2:
                self.omega_head = FourierOmegaHead(embed_dim, controls)
            elif arch == 3:
                self.omega_head = MultiplicativeOmegaHead(
                    embed_dim, controls, hw, R_b_um,
                )
            else:
                raise ValueError(f"Unknown architecture {arch}")
        else:
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
        elif arch == 3:
            self.delta_head = MonotoneDeltaHead(embed_dim, controls)
        else:
            raise ValueError(f"Unknown architecture {arch}")

        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        n_action_params = self.omega_head.n_params + self.delta_head.n_params
        self.log_std = nn.Parameter(torch.full((n_action_params,), init_log_std))

        # ── warm-start / residual parameterization ──────────────────
        # Architecture 3 has the residual structure baked into its heads —
        # warm_start is ignored to avoid double-counting.
        self._warm_start = controls.warm_start and arch != 3
        if self._warm_start:
            self._init_warm_start_buffers(config)

        # ── residual α schedule (trust-region-style budget) ─────────
        # alpha=0 ⇒ exactly baseline, alpha=1 ⇒ unmodified head output.
        # Applied to both Arch 1/2 warm-start corrections AND to the
        # parameter magnitudes for Arch 3 (so initial exploration is small).
        self.register_buffer(
            "residual_alpha",
            torch.tensor(float(controls.residual_alpha_start)),
        )

    def _init_warm_start_buffers(self, config: ProjectConfig) -> None:
        """Precompute the baseline schedule and neutral head profiles.

        When warm_start is enabled, head outputs are interpreted as
        *corrections* relative to the fixed adiabatic baseline:

            final(t) = baseline(t) + [head(t) − neutral(t)]

        At random initialization the head outputs ≈ neutral, so the
        corrections start near zero and the model produces ≈ baseline.
        """
        ctrl = config.controls
        hw = config.hardware
        N_t = ctrl.N_t

        # Baseline omega: trapezoidal envelope (same as FixedScheduleBaseline)
        R_b_um = config.udg.radius * config.udg.spacing
        omega_peak = compute_blockade_omega(hw.C6, R_b_um, hw.omega_max)
        T_us = ctrl.T * 1e6
        t = torch.linspace(0.0, 1.0, N_t)
        r = max(hw.t_ramp / T_us, 1e-9)
        o = hw.t_onset / T_us
        rise = ((t - o) / r).clamp(0.0, 1.0)
        fall = ((1.0 - o - t) / r).clamp(0.0, 1.0)
        baseline_omega = omega_peak * torch.min(rise, fall)

        # Baseline delta: linear sweep from delta_min to delta_max
        baseline_delta = torch.linspace(
            ctrl.delta_min, ctrl.delta_max, N_t
        )

        self.register_buffer("_baseline_omega", baseline_omega)
        self.register_buffer("_baseline_delta", baseline_delta)

        # Neutral profiles: what the heads output for zero-valued params.
        # At random init the MLP outputs are small (~0), so head_output ≈
        # reconstruct(zeros).  Subtracting this makes the correction ≈ 0.
        with torch.no_grad():
            if self.omega_head.n_params > 0:
                z_omega = torch.zeros(1, self.omega_head.n_params)
                self.register_buffer(
                    "_neutral_omega",
                    self.omega_head.reconstruct(z_omega).squeeze(0),
                )
            else:
                self.register_buffer("_neutral_omega", torch.zeros(N_t))

            z_delta = torch.zeros(1, self.delta_head.n_params)
            self.register_buffer(
                "_neutral_delta",
                self.delta_head.reconstruct(z_delta).squeeze(0),
            )

    def _apply_warm_start(
        self, omega: torch.Tensor, delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert head outputs from absolute values to baseline + α·correction.

        Parameters
        ----------
        omega, delta : (B, N_t) tensors from the head reconstructions.

        Returns
        -------
        Corrected (omega, delta) tensors, clamped to physical bounds.

        Notes
        -----
        The correction is scaled by ``self.residual_alpha``:

            final(t) = baseline(t) + α · [head(t) − neutral(t)]

        With ``α`` small early in training the policy is constrained to a
        narrow neighborhood of the baseline, then the neighborhood expands
        as ``α`` is annealed by the orchestrator.
        """
        ctrl = self.config.controls
        alpha = self.residual_alpha

        if self.omega_head.n_params > 0:
            omega_corr = omega - self._neutral_omega.unsqueeze(0)
            omega = (
                self._baseline_omega.unsqueeze(0) + alpha * omega_corr
            ).clamp(0.0, ctrl.omega_max)

        delta_corr = delta - self._neutral_delta.unsqueeze(0)
        delta = (
            self._baseline_delta.unsqueeze(0) + alpha * delta_corr
        ).clamp(ctrl.delta_min, ctrl.delta_max)
        return omega, delta

    def set_residual_alpha(self, alpha: float) -> None:
        """Update the residual scaling factor.  Called by the orchestrator
        once per training step from the configured warmup schedule."""
        self.residual_alpha.fill_(float(alpha))

    def current_alpha(self, step: int) -> float:
        """Compute the warmup-scheduled α for the given training step.

        Linear interpolation from ``residual_alpha_start`` →
        ``residual_alpha_end`` over ``residual_alpha_warmup_steps``.
        """
        ctrl = self.config.controls
        warmup = max(ctrl.residual_alpha_warmup_steps, 0)
        if warmup <= 0:
            return float(ctrl.residual_alpha_end)
        frac = min(max(step / float(warmup), 0.0), 1.0)
        return (
            float(ctrl.residual_alpha_start)
            + frac
            * (float(ctrl.residual_alpha_end) - float(ctrl.residual_alpha_start))
        )

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def encode(self, batch: Batch) -> torch.Tensor:
        """Encode a PyG batch to graph-level embeddings."""
        h = self.encoder(batch.x, batch.edge_index, batch.batch)
        return torch.cat([h, batch.graph_feats], dim=-1)

    def _arch3_alpha_scale(self) -> torch.Tensor:
        """Scaling factor on Arch 3 head parameters before reconstruction.

        Architecture 3 heads are constructed so that ``params = 0`` already
        produces the analytic baseline (zero-initialized linear layers +
        baseline-equivalent reconstruction).  Scaling the params by α
        shrinks the deviation from baseline by the same factor, giving the
        same trust-region behavior the warm-start path provides for
        Arch 1/2.
        """
        return self.residual_alpha

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        """Deterministic forward: mean schedule parameters + value estimate."""
        embed = self.encode(batch)
        omega_params, omega_full = self.omega_head(embed)
        delta_params, _delta_full = self.delta_head(embed)

        arch = self.config.controls.architecture
        if arch == 3:
            alpha = self._arch3_alpha_scale()
            if self.omega_head.n_params > 0:
                omega = self.omega_head.reconstruct(alpha * omega_params)
            else:
                omega = omega_full
            delta = self.delta_head.reconstruct(alpha * delta_params)
        else:
            omega, delta = omega_full, _delta_full
            if self._warm_start:
                omega, delta = self._apply_warm_start(omega, delta)

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

        arch = self.config.controls.architecture
        if arch == 3:
            alpha = self._arch3_alpha_scale()
            if n_omega > 0:
                omega = self.omega_head.reconstruct(alpha * omega_p)
            else:
                _, omega = self.omega_head(out["embed"])
            delta = self.delta_head.reconstruct(alpha * delta_p)
        else:
            if n_omega > 0:
                omega = self.omega_head.reconstruct(omega_p)
            else:
                _, omega = self.omega_head(out["embed"])
            delta = self.delta_head.reconstruct(delta_p)
            if self._warm_start:
                omega, delta = self._apply_warm_start(omega, delta)

        return {
            "omega": omega,
            "delta": delta,
            "sampled_params": sampled,
            "logprob": logprob,
            "value": out["value"],
            "entropy": dist.entropy().sum(dim=-1),
        }

    def log_prob_for(
        self, batch: Batch, sampled_params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log-prob, entropy, and value for previously-sampled params.

        Used by PPO to re-evaluate stored rollouts under the current policy.
        """
        out = self.forward(batch)
        dist = self.action_dist(out["mean_params"])
        logprob = dist.log_prob(sampled_params).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return logprob, entropy, out["value"]

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
