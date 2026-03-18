"""
optimizer.py — Access Point Placement Optimizer
================================================
Author: Kofi Sintim

Uses the Fiedler value (λ₂) as an objective function to evaluate and
rank potential AP placement locations.

Strategy: Greedy Search
-----------------------
For each candidate AP position:
    1. Add it to the existing network
    2. Recompute signal strengths, adjacency matrix, and Laplacian
    3. Extract λ₂
    4. Record the result

The candidate that maximises λ₂ is selected as the optimal placement.

This is a *greedy* strategy — it selects the single best next AP without
considering future placements. For multi-AP optimisation, a branch-and-bound
or exhaustive search could be used instead.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .distances import build_signal_matrix
from .matrices import build_adjacency_matrix, build_laplacian
from .spectral import fiedler_value, connectivity_report


def evaluate_ap_placement(
    buildings: Dict[str, Tuple[float, float]],
    existing_aps: Dict[str, Tuple[float, float]],
    candidate_name: str,
    candidate_coords: Tuple[float, float],
    S0: float = 10_000.0,
    threshold: float = 0.5,
) -> Dict:
    """
    Evaluate the effect of adding one candidate AP to the existing network.

    Parameters
    ----------
    buildings : dict
        {name: (x, y)} for all buildings.
    existing_aps : dict
        {name: (x, y)} for currently installed APs.
    candidate_name : str
        Label for the candidate AP (e.g. "AP_C").
    candidate_coords : tuple (x, y)
        Proposed location for the new AP.
    S0 : float
        Reference signal power constant.
    threshold : float
        Minimum signal to form an active connection.

    Returns
    -------
    dict with keys:
        - name          : str    candidate AP name
        - coords        : tuple
        - fiedler_before: float  λ₂ without this AP
        - fiedler_after : float  λ₂ with this AP added
        - improvement   : float  absolute increase in λ₂
        - pct_change    : float  percentage improvement
        - report_before : dict   full connectivity report before
        - report_after  : dict   full connectivity report after
    """
    # --- BEFORE: existing network only ---
    S_before, b_names, ap_names_before = build_signal_matrix(
        buildings, existing_aps, S0=S0, threshold=threshold
    )
    A_before, nodes_before = build_adjacency_matrix(S_before, b_names, ap_names_before)
    L_before = build_laplacian(A_before)
    lam2_before = fiedler_value(L_before)
    rep_before = connectivity_report(L_before, nodes_before, label="Before " + candidate_name)

    # --- AFTER: add the candidate AP ---
    augmented_aps = {**existing_aps, candidate_name: candidate_coords}
    S_after, _, ap_names_after = build_signal_matrix(
        buildings, augmented_aps, S0=S0, threshold=threshold
    )
    A_after, nodes_after = build_adjacency_matrix(S_after, b_names, ap_names_after)
    L_after = build_laplacian(A_after)
    lam2_after = fiedler_value(L_after)
    rep_after = connectivity_report(L_after, nodes_after, label="After " + candidate_name)

    improvement = lam2_after - lam2_before
    pct_change = (improvement / lam2_before * 100) if lam2_before > 1e-10 else float("inf")

    return {
        "name": candidate_name,
        "coords": candidate_coords,
        "fiedler_before": lam2_before,
        "fiedler_after": lam2_after,
        "improvement": improvement,
        "pct_change": pct_change,
        "report_before": rep_before,
        "report_after": rep_after,
    }


def find_best_placement(
    buildings: Dict[str, Tuple[float, float]],
    existing_aps: Dict[str, Tuple[float, float]],
    candidates: Dict[str, Tuple[float, float]],
    S0: float = 10_000.0,
    threshold: float = 0.5,
    verbose: bool = True,
) -> Tuple[str, Dict]:
    """
    Evaluate all candidate AP locations and return the best one.

    Ranks candidates by the improvement in Fiedler value (λ₂) they produce.

    Parameters
    ----------
    buildings : dict
    existing_aps : dict
    candidates : dict
        {name: (x, y)} for all candidate positions to evaluate.
    S0 : float
    threshold : float
    verbose : bool
        If True, print a ranking table to stdout.

    Returns
    -------
    best_name : str
        Name of the candidate AP with the highest λ₂ improvement.
    results : dict
        {candidate_name: evaluation_dict} for all candidates.

    Example
    -------
    >>> best, results = find_best_placement(buildings, existing_aps, candidates)
    >>> print(f"Best location: {best}")
    Best location: AP_C
    """
    results = {}
    for name, coords in candidates.items():
        results[name] = evaluate_ap_placement(
            buildings, existing_aps, name, coords, S0=S0, threshold=threshold
        )

    # Sort by improvement descending
    ranked = sorted(results.items(), key=lambda x: x[1]["improvement"], reverse=True)

    if verbose:
        print("\n" + "=" * 60)
        print("  AP PLACEMENT RANKING (by Fiedler value improvement)")
        print("=" * 60)
        print(f"  {'Candidate':<12} {'λ₂ Before':>10} {'λ₂ After':>10} {'Δλ₂':>8} {'% Change':>10}")
        print("-" * 60)
        for name, r in ranked:
            print(
                f"  {name:<12} {r['fiedler_before']:>10.4f} {r['fiedler_after']:>10.4f} "
                f"{r['improvement']:>8.4f} {r['pct_change']:>9.1f}%"
            )
        print("=" * 60)
        best_name = ranked[0][0]
        print(f"\n  ✓ RECOMMENDED: Add {best_name} at {results[best_name]['coords']}")
        print(f"    Connectivity improves by {results[best_name]['pct_change']:.1f}%\n")

    best_name = ranked[0][0]
    return best_name, results


def grid_search(
    buildings: Dict[str, Tuple[float, float]],
    existing_aps: Dict[str, Tuple[float, float]],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    grid_size: int = 20,
    S0: float = 10_000.0,
    threshold: float = 0.5,
) -> Tuple[Tuple[float, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep a grid of candidate AP positions and record λ₂ for each.

    Useful for generating a heatmap of connectivity improvement
    across the campus area.

    Parameters
    ----------
    buildings : dict
    existing_aps : dict
    x_range : tuple (x_min, x_max)
    y_range : tuple (y_min, y_max)
    grid_size : int
        Number of points along each axis (default 20 → 400 evaluations).
    S0, threshold : float

    Returns
    -------
    best_coords : tuple (x, y)
        Grid point with highest λ₂.
    xs, ys : np.ndarray
        Meshgrid coordinates.
    lam2_grid : np.ndarray, shape (grid_size, grid_size)
        Fiedler value at each grid point.
    """
    xs = np.linspace(x_range[0], x_range[1], grid_size)
    ys = np.linspace(y_range[0], y_range[1], grid_size)
    lam2_grid = np.zeros((grid_size, grid_size))

    best_val = -np.inf
    best_coords = (xs[0], ys[0])

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            aug = {**existing_aps, "_candidate": (x, y)}
            S, b_names, ap_names = build_signal_matrix(buildings, aug, S0=S0, threshold=threshold)
            A, _ = build_adjacency_matrix(S, b_names, ap_names)
            L = build_laplacian(A)
            lam2 = fiedler_value(L)
            lam2_grid[i, j] = lam2
            if lam2 > best_val:
                best_val = lam2
                best_coords = (x, y)

    return best_coords, xs, ys, lam2_grid
