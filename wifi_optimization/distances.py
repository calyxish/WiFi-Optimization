"""
distances.py — Distance & Signal Strength Calculations
=======================================================
Author: Adwoa Pokua

Computes Euclidean distances between nodes (buildings and access points)
and converts them to signal strengths using the inverse square law.

Physics Background
------------------
WiFi signal power decays with the square of distance (free-space path loss):

    Signal = S₀ / d²

where S₀ is a reference power constant (default 10,000) and d is the
Euclidean distance between the transmitter and receiver.
"""

import numpy as np
from typing import Dict, List, Tuple


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Compute the Euclidean (straight-line) distance between two 2D points.

    Parameters
    ----------
    p1 : tuple (x1, y1)
    p2 : tuple (x2, y2)

    Returns
    -------
    float
        Distance in the same units as the coordinates (e.g. metres).

    Example
    -------
    >>> euclidean_distance((0, 0), (3, 4))
    5.0
    """
    x1, y1 = p1
    x2, y2 = p2
    return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def signal_strength(distance: float, S0: float = 10_000.0, min_dist: float = 1.0) -> float:
    """
    Compute WiFi signal strength using the inverse square law.

        Signal = S₀ / max(d, min_dist)²

    Parameters
    ----------
    distance : float
        Distance between node and access point (metres).
    S0 : float, optional
        Reference signal power constant (default 10,000).
    min_dist : float, optional
        Minimum distance floor to prevent division by zero (default 1 m).

    Returns
    -------
    float
        Signal strength (arbitrary units, higher = stronger).

    Example
    -------
    >>> signal_strength(25)
    16.0
    >>> signal_strength(100)
    1.0
    """
    d = max(distance, min_dist)
    return S0 / (d ** 2)


def build_distance_matrix(
    buildings: Dict[str, Tuple[float, float]],
    access_points: Dict[str, Tuple[float, float]],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build a distance matrix between all buildings and access points.

    Parameters
    ----------
    buildings : dict
        Mapping of building name → (x, y) coordinates.
    access_points : dict
        Mapping of AP name → (x, y) coordinates.

    Returns
    -------
    D : np.ndarray, shape (n_buildings, n_aps)
        D[i, j] = Euclidean distance from building i to AP j.
    building_names : list of str
    ap_names : list of str
    """
    building_names = list(buildings.keys())
    ap_names = list(access_points.keys())

    D = np.zeros((len(building_names), len(ap_names)))
    for i, bname in enumerate(building_names):
        for j, aname in enumerate(ap_names):
            D[i, j] = euclidean_distance(buildings[bname], access_points[aname])

    return D, building_names, ap_names


def build_signal_matrix(
    buildings: Dict[str, Tuple[float, float]],
    access_points: Dict[str, Tuple[float, float]],
    S0: float = 10_000.0,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build a signal strength matrix and apply a threshold to determine
    which building–AP connections are active.

    Parameters
    ----------
    buildings : dict
        Mapping of building name → (x, y) coordinates.
    access_points : dict
        Mapping of AP name → (x, y) coordinates.
    S0 : float, optional
        Reference power constant (default 10,000).
    threshold : float, optional
        Minimum signal strength to consider a connection active (default 0.5).

    Returns
    -------
    S : np.ndarray, shape (n_buildings, n_aps)
        S[i, j] = signal strength from building i to AP j (0 if below threshold).
    building_names : list of str
    ap_names : list of str

    Notes
    -----
    The threshold prevents very weak, unreliable signals from being modelled
    as real connections in the graph.
    """
    dist_matrix, building_names, ap_names = build_distance_matrix(buildings, access_points)

    S = np.zeros_like(dist_matrix)
    for i in range(dist_matrix.shape[0]):
        for j in range(dist_matrix.shape[1]):
            s = signal_strength(dist_matrix[i, j], S0=S0)
            S[i, j] = s if s >= threshold else 0.0

    return S, building_names, ap_names
