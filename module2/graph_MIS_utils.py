import numpy as np
import random
import networkx as nx

# from scipy.optimize import minimize
import matplotlib.pyplot as plt

import itertools
import math

############################################################
# Graph functions
############################################################

def get_lattice_UDG(L: int, Rb_a: float, a: float, dropout_rate):
    """Square lattice of 'L × L' nodes (atoms); edges if distance ≤ radius., lattice constant 'a' is in um, 'Rb/a' in units of spacing

    Returns
    -------
    adj : (N,N) np.ndarray[int8]
        Adjacency matrix (0/1, symmetric, no self-loops).
    positions : (N,2) np.ndarray[float]
        Cartesian coordinates of each node (atom).
    """
    positions = np.array(list(itertools.product(range(L), repeat = 2)))
    indices_to_remove = np.random.choice(len(positions), math.floor(dropout_rate * len(positions)), replace=False)
    positions_ = np.delete(positions, indices_to_remove, axis = 0)
    positions_ = a * positions_

    N = len(positions_)
    adj = np.zeros((N, N), dtype = np.int8)
    r2 = (Rb_a * a) ** 2
    for i in range(N):
        for j in range(i + 1, N):
            if np.sum((positions_[i] - positions_[j]) ** 2) <= r2:
                adj[i,j] = adj[j,i] = 1
    return adj, positions_


def infer_spacing(coords, keepmask):
    n_tot = keepmask.size
    side  = int(round(np.sqrt(n_tot)))          # original lattice dimension
    if side * side != n_tot:
        raise ValueError("keepmask length is not a perfect square")

    # indices (0 … side²-1) of the sites that survived dropout
    flat_idx = np.nonzero(keepmask)[0]

    for idx, (x, y) in zip(flat_idx, coords):
        r, c = divmod(idx, side)                # original integer grid coords
        if r > 0:                               # use first row-offset that isn't 0
            return x / r
        if c > 0:                               # otherwise use a col-offset
            return y / c

    raise ValueError("At least two non-collinear atoms must remain.")

# def graph_to_atom_arrangement(adj, coords, keepmask):
#     """
#     from bloqade.analog import rydberg_h, piecewise_linear, piecewise_constant, waveform, cast, var, start
#     """
#     geom = start.add_position(coords)
#     return geom

def graph_to_nx(adj, coords, keepmask):
    return nx.from_numpy_array(adj)

def construct_graph(positions, Rb_a, a):
    """ 
    Returns
    -------
    G : nx.graph.Graph() object
    """ 
    # get_edges = lambda positions : [(i,j) for i,x in enumerate(positions) for j,y in enumerate(positions) if np.linalg.norm(np.array(x)-np.array(y))<=Rb and i<j]
    N = len(positions)
    nx_edges = [(i,j) for i,x in enumerate(positions) for j,y in enumerate(positions) if np.linalg.norm(np.array(x)-np.array(y)) <= Rb_a * a and i<j]
    nx_positions = {i:p for i,p in enumerate(positions)}
    G = nx.graph.Graph()
    G.add_nodes_from([i for i in range(N)])
    G.add_edges_from(nx_edges)
    return G, nx_positions

# adj, positions = get_lattice_UDG(4, 1.5, 4.0e-6, 0.2)
# G, nx_positions = construct_graph(positions)
# nx.draw(G, pos = nx_positions)
# plt.show()

def get_all_MIS(G):
    """
    Brute force method for finding all the MIS solutions
    """
    if len(G.nodes)<=1:
        return [set(G.nodes)]
    MISset=[]
    for i,node in enumerate(G.nodes):
        G_=G.copy()
        G_.remove_node(node)
        for n in G.neighbors(node):
            G_.remove_node(n)
        for j in range(i):
            if j in G_.nodes:
                G_.remove_node(j)
        misnew=get_all_MIS(G_)
        for s in get_all_MIS(G_):
            A=s
            A.add(node)
            if A not in MISset:
                MISset+=[A]
    N=max([len(i) for i in MISset])
    return [a for a in MISset if len(a)==N]

# MIS = nx.algorithms.approximation.maximum_independent_set(G)
# cardinality = len(MIS)

def draw_MIS(G, nx_positions, MIS):
    """
    if approx:
    MIS = nx.algorithms.approximation.maximum_independent_set(G)
    if exact:
    set_of_MIS = getallMIS(G)
    """
    colors=["red" if n in MIS else "blue" for n in G.nodes]
    nx.draw(G,pos = nx_positions,node_color = colors)

# draw_MIS(G, nx_positions, get_all_MIS(G)[5])


############################################################
# Post processing functions
############################################################

def check_independent_set(bitstring, G, excited_char='r'):
    """Check if the excited atoms in *bitstring* form an independent set.

    Parameters
    ----------
    bitstring : str
        Measurement outcome, one character per atom.
    G : nx.Graph
        The interaction graph.
    excited_char : str
        Character that marks an excited / Rydberg atom (Braket AHS uses ``'r'``).
    """
    for i in range(len(bitstring)):
        for j in range(i + 1, len(bitstring)):
            if bitstring[i] == excited_char and bitstring[j] == excited_char and (i, j) in G.edges:
                return False
    return True

def regroup_by_ones_counts(bitstring_counts):
    regrouped = {}
    for bitstring, count in bitstring_counts.items():
        num_ones = bitstring.count('r')
        if num_ones not in regrouped:
            regrouped[num_ones] = 0
        regrouped[num_ones] += count
    return regrouped

def regroup_by_ones_dict(bitstring_counts, len_MIS):
    regrouped = {}
    for bitstring, count in bitstring_counts.items():
        num_ones = bitstring.count('r')
        if num_ones not in regrouped:
            regrouped[num_ones] = []
        regrouped[num_ones].append(bitstring)
    return regrouped.get(len_MIS, [])

# MIS=nx.algorithms.approximation.maximum_independent_set(G)
# regroup_by_ones_dict(counts, len(MIS)), len(regroup_by_ones_dict(counts, len(MIS)))
# Recall counts = result_full_loaded.get_counts()
# Histogram functions go inside

def find_MIS_probability(G, counts, nshots, verbose=True):
    """Estimate p_MIS from Braket measurement counts.

    Returns ``(p_mis, mis_bitstrings, mis_node_tuples)``.  If no shots
    match the target cardinality, returns ``(0.0, [], [])``.
    """
    MIS = nx.algorithms.approximation.maximum_independent_set(G)
    candidates = regroup_by_ones_dict(counts, len(MIS))

    qMIS_bitstrings = []
    qMIS_tuples = []
    for bitstring in candidates:
        if check_independent_set(bitstring, G):
            qMIS_bitstrings.append(bitstring)
            qMIS_tuples.append(tuple(i for i, c in enumerate(bitstring) if c == 'r'))

    total_counts = sum(counts[bs] for bs in qMIS_bitstrings)
    p_mis = total_counts / max(nshots, 1)

    if verbose:
        print(f'MIS probability : {p_mis * 100:.1f} %')

    return p_mis, qMIS_bitstrings, qMIS_tuples