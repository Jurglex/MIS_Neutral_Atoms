"""GIN graph encoder with concat(mean, max, sum) pooling."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool


class GINEncoder(nn.Module):
    """Graph Isomorphism Network encoder.

    Each layer applies a 2-layer MLP inside a GINConv with a learnable
    epsilon.  The final graph-level embedding is the concatenation of
    mean-, max-, and sum-pooled node representations (3 * hidden_dim).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList()
        d = in_dim
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(GINConv(mlp, train_eps=True))
            d = hidden_dim
        self.out_dim = hidden_dim * 3

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        for conv in self.layers:
            x = F.relu(conv(x, edge_index))
        return torch.cat(
            [
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
                global_add_pool(x, batch),
            ],
            dim=-1,
        )
