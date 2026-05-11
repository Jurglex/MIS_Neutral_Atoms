"""Graph featurization: convert networkx graphs to PyG Data objects."""
from __future__ import annotations

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

K_PE_DEFAULT = 4
MAX_NODES_NORM = 64.0
MAX_EDGES_NORM = 256.0


def laplacian_positional_encoding(G: nx.Graph, k: int = K_PE_DEFAULT) -> np.ndarray:
    """Top-k non-trivial eigenvectors of the normalized Laplacian.

    Returns an (n, k) array.  For graphs with fewer than k+1 nodes the
    missing columns are zero-padded.  Sign ambiguity is canonicalized by
    forcing the column-sum to be non-negative.
    """
    L = nx.normalized_laplacian_matrix(G).astype(float).todense()
    _eigvals, eigvecs = np.linalg.eigh(L)
    pe = np.asarray(eigvecs[:, 1 : 1 + k])
    if pe.shape[1] < k:
        pe = np.pad(pe, ((0, 0), (0, k - pe.shape[1])))
    signs = np.sign(pe.sum(axis=0) + 1e-9)
    pe = pe * signs
    return pe.astype(np.float32)


def algebraic_connectivity(G: nx.Graph) -> float:
    """Lambda-2 of the normalized Laplacian (0.0 for disconnected or trivial graphs)."""
    if G.number_of_nodes() < 2 or not nx.is_connected(G):
        return 0.0
    L = nx.normalized_laplacian_matrix(G).astype(float).todense()
    eigvals = np.linalg.eigvalsh(L)
    return float(eigvals[1])


def graph_to_pyg(G: nx.Graph, k_pe: int = K_PE_DEFAULT) -> Data:
    """Convert a networkx graph to a PyG ``Data`` object with structural features.

    Node features  (dim = 2 + k_pe):
        [degree_normalized, clustering_coeff, laplacian_pe_0 … laplacian_pe_{k-1}]

    Graph-level features  (dim = 3, stored as ``data.graph_feats``):
        [n_nodes / 64, n_edges / 256, lambda_2]
    """
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}

    deg = np.array([G.degree(v) for v in nodes], dtype=np.float32) / max(n - 1, 1)
    clust = np.array([nx.clustering(G, v) for v in nodes], dtype=np.float32)
    pe = laplacian_positional_encoding(G, k=k_pe)

    x = np.concatenate([deg[:, None], clust[:, None], pe], axis=1).astype(np.float32)

    edges = []
    for u, v in G.edges():
        edges.append([idx[u], idx[v]])
        edges.append([idx[v], idx[u]])
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.zeros((2, 0), dtype=torch.long)
    )

    graph_feats = torch.tensor(
        [n / MAX_NODES_NORM, G.number_of_edges() / MAX_EDGES_NORM, algebraic_connectivity(G)],
        dtype=torch.float32,
    )

    data = Data(x=torch.from_numpy(x), edge_index=edge_index)
    data.graph_feats = graph_feats.unsqueeze(0)  # (1, 3) for batching
    return data
