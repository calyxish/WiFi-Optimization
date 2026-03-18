"""
spectral.py — Eigenvalue Analysis & Spectral Connectivity
==========================================================
Author: Kwame Adjei Amoah

Computes and interprets the eigenspectrum of the Laplacian matrix to
assess network connectivity.

Key Concepts
------------
The **Fiedler value** (λ₂, the second-smallest eigenvalue of L) is the
primary measure of *algebraic connectivity*:

    λ₂ = 0      → graph is disconnected (at least 2 components)
    λ₂ > 0      → graph is connected
    λ₂ large    → graph is robustly connected (hard to disconnect)

This makes λ₂ the ideal metric for comparing AP configurations:
the placement that maximises λ₂ gives the best-connected network.

Reference: Fiedler, M. (1973). Algebraic connectivity of graphs.
           Czechoslovak Mathematical Journal, 23(2), 298–305.
"""

import numpy as np
from scipy.linalg import eigh
from typing import Dict, List, Tuple


def compute_eigenvalues(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the eigenvalues and eigenvectors of a symmetric Laplacian matrix.

    Uses scipy.linalg.eigh which exploits symmetry for numerical stability
    and guarantees real eigenvalues (appropriate for positive semi-definite L).

    Parameters
    ----------
    L : np.ndarray, shape (n, n)
        Graph Laplacian matrix (symmetric, positive semi-definite).

    Returns
    -------
    eigenvalues : np.ndarray, shape (n,)
        Eigenvalues sorted in ascending order: λ₁ ≤ λ₂ ≤ ... ≤ λₙ.
        λ₁ is always 0 (or numerically ~0) for a valid Laplacian.
    eigenvectors : np.ndarray, shape (n, n)
        Column i is the eigenvector corresponding to eigenvalues[i].
        The Fiedler vector is eigenvectors[:, 1].
    """
    eigenvalues, eigenvectors = eigh(L)
    return eigenvalues, eigenvectors


def fiedler_value(L: np.ndarray) -> float:
    """
    Extract the Fiedler value λ₂ from the Laplacian.

    The Fiedler value is the second-smallest eigenvalue of L and serves
    as the measure of algebraic connectivity (network robustness).

    Parameters
    ----------
    L : np.ndarray, shape (n, n)
        Graph Laplacian matrix.

    Returns
    -------
    float
        λ₂ — the Fiedler value.

    Example
    -------
    >>> L = np.array([[ 1, -1], [-1,  1]], dtype=float)
    >>> round(fiedler_value(L), 4)
    2.0
    """
    eigenvalues, _ = compute_eigenvalues(L)
    return float(eigenvalues[1])


def fiedler_vector(L: np.ndarray) -> np.ndarray:
    """
    Return the Fiedler vector (eigenvector corresponding to λ₂).

    The Fiedler vector partitions the graph into two communities based on
    the sign of each component:
        - Positive entries → one cluster
        - Negative entries → another cluster

    Parameters
    ----------
    L : np.ndarray, shape (n, n)

    Returns
    -------
    np.ndarray, shape (n,)
        The Fiedler vector.
    """
    _, eigenvectors = compute_eigenvalues(L)
    return eigenvectors[:, 1]


def count_connected_components(L: np.ndarray, tol: float = 1e-8) -> int:
    """
    Count the number of connected components by counting near-zero eigenvalues.

    The algebraic multiplicity of eigenvalue 0 in the Laplacian spectrum
    equals the number of connected components in the graph.

    Parameters
    ----------
    L : np.ndarray
    tol : float
        Tolerance for treating an eigenvalue as zero (default 1e-8).

    Returns
    -------
    int
        Number of connected components.
    """
    eigenvalues, _ = compute_eigenvalues(L)
    return int(np.sum(eigenvalues < tol))


def connectivity_report(
    L: np.ndarray,
    node_names: List[str],
    label: str = "Network",
) -> Dict:
    """
    Generate a full connectivity report for a network configuration.

    Parameters
    ----------
    L : np.ndarray
        Laplacian matrix.
    node_names : list of str
        Labels for the nodes.
    label : str
        A human-readable name for this configuration (e.g. "Before AP_C").

    Returns
    -------
    report : dict with keys:
        - label         : str
        - eigenvalues   : np.ndarray  (all λᵢ)
        - fiedler_value : float       (λ₂)
        - fiedler_vector: np.ndarray
        - n_components  : int
        - is_connected  : bool
        - summary       : str         (formatted text)
    """
    eigenvalues, eigenvectors = compute_eigenvalues(L)
    lam2 = float(eigenvalues[1])
    n_comp = int(np.sum(eigenvalues < 1e-8))
    fvec = eigenvectors[:, 1]

    # Fiedler vector partition
    partition_A = [node_names[i] for i in range(len(node_names)) if fvec[i] >= 0]
    partition_B = [node_names[i] for i in range(len(node_names)) if fvec[i] < 0]

    summary = (
        f"\n{'='*55}\n"
        f"  Connectivity Report: {label}\n"
        f"{'='*55}\n"
        f"  Nodes               : {len(node_names)}\n"
        f"  Connected components: {n_comp}\n"
        f"  Is connected        : {'YES ✓' if n_comp == 1 else 'NO ✗'}\n"
        f"\n  Eigenspectrum (sorted):\n"
        + "".join(f"    λ{i+1} = {v:>8.4f}\n" for i, v in enumerate(eigenvalues))
        + f"\n  Fiedler value λ₂    : {lam2:.4f}\n"
        f"  (Higher = better connected)\n"
        f"\n  Fiedler Partition:\n"
        f"    Group (+) : {partition_A}\n"
        f"    Group (-) : {partition_B}\n"
        f"{'='*55}\n"
    )

    return {
        "label": label,
        "eigenvalues": eigenvalues,
        "fiedler_value": lam2,
        "fiedler_vector": fvec,
        "n_components": n_comp,
        "is_connected": n_comp == 1,
        "summary": summary,
    }
