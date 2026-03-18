"""
matrices.py — Adjacency, Degree & Laplacian Matrix Construction
================================================================
Author: Kwakye Ishmael

Constructs the core graph matrices used in network analysis:

    A  — Weighted Adjacency Matrix
    D  — Degree (strength) Matrix
    L  — Laplacian Matrix  L = D - A

Graph Model
-----------
The WiFi network is modelled as a **weighted bipartite graph**:
    - One set of nodes: Buildings  {B1, B2, B3, ...}
    - Other set: Access Points     {AP_A, AP_B, ...}
    - Edge weights: signal strengths (from distances.py)

Buildings never connect directly to other buildings, and APs never
connect directly to other APs — only cross-connections exist.
"""

import numpy as np
from typing import Dict, List, Tuple


def build_adjacency_matrix(
    signal_matrix: np.ndarray,
    building_names: List[str],
    ap_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a full symmetric adjacency matrix A for the bipartite graph.

    The signal_matrix has shape (n_buildings, n_aps). We expand this into
    a square (n_buildings + n_aps) × (n_buildings + n_aps) matrix where:

        A[i, j] = signal strength  if i is a building and j is an AP
        A[j, i] = signal strength  (symmetric)
        A[i, i] = 0                (no self-loops)
        A[i, j] = 0                if both nodes are same type

    Parameters
    ----------
    signal_matrix : np.ndarray, shape (n_buildings, n_aps)
        Entry [i, j] is the signal strength between building i and AP j.
    building_names : list of str
    ap_names : list of str

    Returns
    -------
    A : np.ndarray, shape (n_nodes, n_nodes)
        Full symmetric weighted adjacency matrix.
    node_names : list of str
        Ordered list of all node names (buildings first, then APs).

    Example
    -------
    For 3 buildings and 2 APs, A is 5×5 with the structure:

        [ 0   0   0  | s₁₁  s₁₂ ]   ← B1
        [ 0   0   0  | s₂₁  s₂₂ ]   ← B2
        [ 0   0   0  | s₃₁  s₃₂ ]   ← B3
        [s₁₁ s₂₁ s₃₁|  0    0  ]   ← AP_A
        [s₁₂ s₂₂ s₃₂|  0    0  ]   ← AP_B
    """
    n_b = len(building_names)
    n_ap = len(ap_names)
    n = n_b + n_ap

    node_names = building_names + ap_names
    A = np.zeros((n, n))

    # Fill off-diagonal blocks
    A[:n_b, n_b:] = signal_matrix          # buildings → APs
    A[n_b:, :n_b] = signal_matrix.T        # APs → buildings (symmetric)

    return A, node_names


def build_degree_matrix(A: np.ndarray) -> np.ndarray:
    """
    Build the diagonal degree (strength) matrix D from adjacency matrix A.

    For weighted graphs, the degree of node i is its *strength*:
        d_i = Σ_j A[i, j]   (sum of all edge weights incident to i)

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Weighted adjacency matrix.

    Returns
    -------
    D : np.ndarray, shape (n, n)
        Diagonal matrix where D[i, i] = sum of row i of A.

    Notes
    -----
    The degree represents the total signal connectivity of a node.
    A high-degree building is well-covered; a high-degree AP is heavily loaded.
    """
    degrees = A.sum(axis=1)
    D = np.diag(degrees)
    return D


def build_laplacian(A: np.ndarray) -> np.ndarray:
    """
    Build the graph Laplacian matrix L = D - A.

    Properties of L
    ---------------
    1. Symmetric: L = Lᵀ
    2. Positive semi-definite: all eigenvalues λᵢ ≥ 0
    3. Singular: rows and columns each sum to zero (det(L) = 0)
    4. Smallest eigenvalue λ₁ = 0 always
    5. Number of zero eigenvalues = number of connected components

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        Weighted adjacency matrix.

    Returns
    -------
    L : np.ndarray, shape (n, n)
        Graph Laplacian.

    Verification
    ------------
    >>> A = np.array([[0,1],[1,0]], dtype=float)
    >>> L = build_laplacian(A)
    >>> np.allclose(L.sum(axis=1), 0)
    True
    """
    D = build_degree_matrix(A)
    L = D - A
    return L


def matrix_summary(
    A: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    node_names: List[str],
) -> str:
    """
    Return a formatted text summary of all three matrices with node labels.

    Parameters
    ----------
    A, D, L : np.ndarray
        Adjacency, Degree, and Laplacian matrices.
    node_names : list of str
        Labels for rows/columns.

    Returns
    -------
    str
        Human-readable table showing all three matrices.
    """
    def _fmt_matrix(name, M, names):
        header = f"\n{'='*60}\n{name}\n{'='*60}\n"
        col_header = f"{'':>8}" + "".join(f"{n:>8}" for n in names) + "\n"
        rows = ""
        for i, row_name in enumerate(names):
            row = f"{row_name:>8}" + "".join(f"{M[i,j]:>8.2f}" for j in range(len(names)))
            rows += row + "\n"
        return header + col_header + rows

    return (
        _fmt_matrix("Adjacency Matrix A", A, node_names)
        + _fmt_matrix("Degree Matrix D", D, node_names)
        + _fmt_matrix("Laplacian Matrix L = D - A", L, node_names)
    )
