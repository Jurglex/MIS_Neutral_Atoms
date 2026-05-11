from __future__ import annotations

import networkx as nx
import numpy as np
import math
from functools import lru_cache

from config import UDGConfig


def generate_square_lattice_udg(cfg: UDGConfig) -> tuple[nx.Graph, dict[int, tuple[float, float]]]:
    """Generate a unit disk graph (UDG) on an nx-by-ny square lattice with dropout.

    Nodes are placed on a grid with spacing `cfg.spacing`. Each node is
    independently dropped with probability `cfg.dropout_rate` using RNG seed
    `cfg.seed`. Edges connect retained nodes whose Euclidean distance is <=
    `cfg.radius` (in the same units as spacing).

    Returns
    -------
    G : nx.Graph
        Graph with nodes relabeled to contiguous integers 0..N-1 after dropout.
    pos : dict[int, tuple[float, float]]
        Positions for each node in `G` in the same units as spacing.
    """
    rng = np.random.default_rng(int(cfg.seed))

    # Grid coordinates (i,j) with physical positions (x,y)
    coords: list[tuple[int, int]] = [(i, j) for i in range(cfg.nx) for j in range(cfg.ny)]
    positions_raw: dict[tuple[int, int], tuple[float, float]] = {
        (i, j): (float(i) * cfg.spacing, float(j) * cfg.spacing) for (i, j) in coords
    }

    # Deterministic dropout with a fixed count: drop exactly round(p * N) sites
    N_total = len(coords)
    p = float(cfg.dropout_rate)
    k_drop = int(round(p * N_total))
    k_drop = max(0, min(N_total, k_drop))

    if k_drop == 0:
        kept_coords = coords
    elif k_drop == N_total:
        kept_coords = []
    else:
        drop_indices = rng.choice(N_total, size=k_drop, replace=False)
        drop_set = {coords[i] for i in np.asarray(drop_indices).tolist()}
        kept_coords = [c for c in coords if c not in drop_set]

    G_raw = nx.Graph()

    # Efficient edge construction using integer grid offsets within the radius
    r = float(cfg.radius)
    s = float(cfg.spacing)
    kept_set = set(kept_coords)

    max_cells = int(math.ceil(r / s))
    r_over_s = r / s

    @lru_cache(maxsize=128)
    def _neighbor_offsets(max_cells_arg: int, r_over_s_arg: float) -> tuple[tuple[int, int], ...]:
        offsets_local: list[tuple[int, int]] = []
        c2 = r_over_s_arg * r_over_s_arg
        for dy in range(-max_cells_arg, max_cells_arg + 1):
            for dx in range(-max_cells_arg, max_cells_arg + 1):
                if dx == 0 and dy == 0:
                    continue
                # Only consider one half-plane to add each undirected edge once
                if not (dy > 0 or (dy == 0 and dx > 0)):
                    continue
                if (dx * dx + dy * dy) <= c2 + 1e-12:
                    offsets_local.append((dx, dy))
        return tuple(offsets_local)

    offsets = _neighbor_offsets(max_cells, r_over_s)

    # Build edges list and add in bulk
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (i, j) in kept_coords:
        for (dx, dy) in offsets:
            nb = (i + dx, j + dy)
            if nb in kept_set:
                edges.append(((i, j), nb))

    G_raw.add_nodes_from(kept_coords)
    if edges:
        G_raw.add_edges_from(edges)

    # Relabel to contiguous ints 0..N-1
    mapping: dict[tuple[int, int], int] = {c: k for k, c in enumerate(sorted(G_raw.nodes()))}
    G = nx.relabel_nodes(G_raw, mapping)
    pos: dict[int, tuple[float, float]] = {mapping[c]: positions_raw[c] for c in G_raw.nodes()}

    return G, pos
