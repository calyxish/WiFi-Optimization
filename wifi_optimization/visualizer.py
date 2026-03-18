"""
visualizer.py — Network Visualisation
======================================
Author: Melchi

Provides publication-quality plots for the WiFi network analysis,
including graph layouts, eigenvalue spectra, signal heatmaps, and
before/after connectivity comparisons.

All functions return Matplotlib Figure objects so they can be
displayed in Jupyter notebooks or saved to disk.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from typing import Dict, List, Optional, Tuple


# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "building":    "#2196F3",   # blue
    "ap_existing": "#4CAF50",   # green
    "ap_new":      "#FF9800",   # orange
    "edge_strong": "#1565C0",
    "edge_weak":   "#BBDEFB",
    "background":  "#FAFAFA",
    "grid":        "#E0E0E0",
}


def plot_network(
    buildings: Dict[str, Tuple[float, float]],
    access_points: Dict[str, Tuple[float, float]],
    adjacency: np.ndarray,
    node_names: List[str],
    title: str = "WiFi Network Graph",
    new_aps: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    min_edge_weight: float = 0.5,
) -> plt.Figure:
    """
    Draw the WiFi network as a graph with weighted edges.

    Nodes are positioned at their real campus coordinates. Edge thickness
    and colour intensity reflect signal strength (edge weight in A).

    Parameters
    ----------
    buildings : dict  {name: (x, y)}
    access_points : dict  {name: (x, y)}
    adjacency : np.ndarray  full symmetric adjacency matrix
    node_names : list of str  ordered list matching adjacency rows/cols
    title : str
    new_aps : list of str, optional  names of newly added APs (coloured orange)
    ax : plt.Axes, optional  if provided, draw onto this axes
    min_edge_weight : float  edges below this weight are not drawn

    Returns
    -------
    fig : plt.Figure
    """
    new_aps = new_aps or []
    fig, ax = (plt.subplots(figsize=(9, 7)) if ax is None else (ax.get_figure(), ax))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    # Build position dict from all nodes
    coords = {**buildings, **access_points}

    n = len(node_names)
    max_weight = adjacency.max() if adjacency.max() > 0 else 1.0

    # Draw edges
    for i in range(n):
        for j in range(i + 1, n):
            w = adjacency[i, j]
            if w < min_edge_weight:
                continue
            xi, yi = coords[node_names[i]]
            xj, yj = coords[node_names[j]]
            alpha = 0.3 + 0.7 * (w / max_weight)
            lw = 0.5 + 3.5 * (w / max_weight)
            ax.plot([xi, xj], [yi, yj], color=COLORS["edge_strong"],
                    alpha=alpha, linewidth=lw, zorder=1)
            # Label edge weight at midpoint
            mx, my = (xi + xj) / 2, (yi + yj) / 2
            ax.text(mx, my, f"{w:.2f}", fontsize=6.5, color="#555",
                    ha="center", va="center", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6, ec="none"))

    # Draw nodes
    for name, (x, y) in coords.items():
        if name in buildings:
            color = COLORS["building"]
            marker = "s"
            size = 200
        elif name in new_aps:
            color = COLORS["ap_new"]
            marker = "^"
            size = 220
        else:
            color = COLORS["ap_existing"]
            marker = "^"
            size = 180

        ax.scatter(x, y, s=size, c=color, marker=marker,
                   zorder=4, edgecolors="white", linewidths=1.5)
        ax.text(x, y + 5, name, fontsize=9, ha="center",
                va="bottom", fontweight="bold", color="#222", zorder=5)

    # Legend
    legend_handles = [
        mpatches.Patch(color=COLORS["building"],    label="Building"),
        mpatches.Patch(color=COLORS["ap_existing"], label="Existing AP"),
    ]
    if new_aps:
        legend_handles.append(mpatches.Patch(color=COLORS["ap_new"], label="New AP"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("x (metres)", fontsize=10)
    ax.set_ylabel("y (metres)", fontsize=10)
    ax.grid(True, color=COLORS["grid"], linestyle="--", alpha=0.6)
    ax.set_xlim(-20, 130)
    ax.set_ylim(-20, 110)
    plt.tight_layout()
    return fig


def plot_eigenvalue_comparison(
    eigenvalues_before: np.ndarray,
    eigenvalues_after: np.ndarray,
    label_before: str = "Before AP_C",
    label_after: str = "After AP_C",
) -> plt.Figure:
    """
    Side-by-side stem plots comparing Laplacian eigenspectra before and after
    adding a new AP. Highlights λ₂ (Fiedler value) in both cases.

    Parameters
    ----------
    eigenvalues_before, eigenvalues_after : np.ndarray
    label_before, label_after : str

    Returns
    -------
    fig : plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    fig.patch.set_facecolor(COLORS["background"])

    for ax, eigs, label, color in zip(
        axes,
        [eigenvalues_before, eigenvalues_after],
        [label_before, label_after],
        ["#1565C0", "#2E7D32"],
    ):
        ax.set_facecolor(COLORS["background"])
        x = np.arange(1, len(eigs) + 1)
        markerline, stemlines, baseline = ax.stem(
            x, eigs, linefmt=color, markerfmt="o", basefmt="gray"
        )
        plt.setp(markerline, color=color, markersize=9)
        plt.setp(stemlines, linewidth=2)

        # Highlight λ₂
        ax.scatter([2], [eigs[1]], s=160, c="#FF5722", zorder=5,
                   label=f"λ₂ = {eigs[1]:.4f}")
        ax.axhline(eigs[1], color="#FF5722", linestyle=":", alpha=0.5)

        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Eigenvalue index", fontsize=10)
        ax.set_ylabel("λ value", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"λ{i}" for i in x], fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(True, color=COLORS["grid"], linestyle="--", alpha=0.5)

    fig.suptitle("Laplacian Eigenspectrum: Connectivity Comparison",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_signal_heatmap(
    buildings: Dict[str, Tuple[float, float]],
    access_points: Dict[str, Tuple[float, float]],
    signal_matrix: np.ndarray,
    building_names: List[str],
    ap_names: List[str],
    title: str = "Signal Strength Matrix",
) -> plt.Figure:
    """
    Display the signal strength matrix as an annotated heatmap.

    Parameters
    ----------
    buildings, access_points : dict
    signal_matrix : np.ndarray, shape (n_buildings, n_aps)
    building_names, ap_names : list of str
    title : str

    Returns
    -------
    fig : plt.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    im = ax.imshow(signal_matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(ap_names)))
    ax.set_yticks(np.arange(len(building_names)))
    ax.set_xticklabels(ap_names, fontsize=10)
    ax.set_yticklabels(building_names, fontsize=10)
    ax.set_xlabel("Access Points", fontsize=11)
    ax.set_ylabel("Buildings", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    # Annotate each cell
    for i in range(len(building_names)):
        for j in range(len(ap_names)):
            val = signal_matrix[i, j]
            text_color = "white" if val > signal_matrix.max() * 0.6 else "#333"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Signal Strength (S₀/d²)")
    plt.tight_layout()
    return fig


def plot_fiedler_bar(results: Dict, baseline_lam2: float) -> plt.Figure:
    """
    Bar chart comparing Fiedler values across all candidate AP placements.

    Parameters
    ----------
    results : dict  {candidate_name: evaluation_dict}  from optimizer
    baseline_lam2 : float  λ₂ of the network without any new AP

    Returns
    -------
    fig : plt.Figure
    """
    names = list(results.keys())
    lam2_vals = [r["fiedler_after"] for r in results.values()]
    improvements = [r["pct_change"] for r in results.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    bar_colors = ["#FF9800" if v == max(lam2_vals) else "#90CAF9" for v in lam2_vals]
    bars = ax.bar(names, lam2_vals, color=bar_colors, edgecolor="white",
                  linewidth=1.2, zorder=3)

    # Baseline line
    ax.axhline(baseline_lam2, color="#E53935", linestyle="--", linewidth=1.8,
               label=f"Baseline λ₂ = {baseline_lam2:.2f}", zorder=4)

    # Annotate bars
    for bar, pct in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"+{pct:.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#333")

    ax.set_xlabel("Candidate AP", fontsize=11)
    ax.set_ylabel("Fiedler Value λ₂", fontsize=11)
    ax.set_title("Connectivity Improvement by AP Placement", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", color=COLORS["grid"], linestyle="--", alpha=0.6, zorder=0)
    ax.set_ylim(0, max(lam2_vals) * 1.2)
    plt.tight_layout()
    return fig


def plot_grid_heatmap(
    xs: np.ndarray,
    ys: np.ndarray,
    lam2_grid: np.ndarray,
    buildings: Dict[str, Tuple[float, float]],
    existing_aps: Dict[str, Tuple[float, float]],
    best_coords: Tuple[float, float],
    title: str = "Connectivity Heatmap (λ₂ by AP Location)",
) -> plt.Figure:
    """
    Visualise how λ₂ varies across the campus as a 2D heatmap,
    overlaid with building and AP locations.

    Parameters
    ----------
    xs, ys : np.ndarray  grid coordinates from grid_search()
    lam2_grid : np.ndarray  shape (grid_size, grid_size)
    buildings, existing_aps : dict
    best_coords : tuple (x, y)
    title : str

    Returns
    -------
    fig : plt.Figure
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(COLORS["background"])

    extent = [xs.min(), xs.max(), ys.min(), ys.max()]
    im = ax.imshow(lam2_grid, origin="lower", extent=extent,
                   cmap="RdYlGn", aspect="auto", alpha=0.85)
    plt.colorbar(im, ax=ax, label="Fiedler Value λ₂")

    # Overlay buildings
    for name, (x, y) in buildings.items():
        ax.scatter(x, y, s=200, c=COLORS["building"], marker="s",
                   zorder=5, edgecolors="white", linewidths=1.5)
        ax.text(x, y + 3, name, fontsize=9, ha="center", fontweight="bold", color="white")

    # Overlay existing APs
    for name, (x, y) in existing_aps.items():
        ax.scatter(x, y, s=180, c=COLORS["ap_existing"], marker="^",
                   zorder=5, edgecolors="white", linewidths=1.5)
        ax.text(x, y + 3, name, fontsize=9, ha="center", fontweight="bold", color="white")

    # Best placement star
    bx, by = best_coords
    ax.scatter(bx, by, s=350, c="#FF5722", marker="*", zorder=6,
               edgecolors="white", linewidths=1.5, label=f"Best: ({bx:.0f}, {by:.0f})")
    ax.legend(fontsize=10, loc="upper right")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("x (metres)", fontsize=10)
    ax.set_ylabel("y (metres)", fontsize=10)
    ax.grid(False)
    plt.tight_layout()
    return fig
