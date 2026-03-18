"""
WiFi Optimization — Graph Theory & Linear Algebra
==================================================
A modular Python package for modeling and optimizing WiFi access point
placement using adjacency matrices, Laplacian matrices, and spectral
graph theory (Fiedler value / algebraic connectivity).

Authors
-------
- Kwakye Ishmael   — matrices.py
- Kwame Adjei Amoah — spectral.py
- Adwoa Pokua      — distances.py
- Kofi Sintim      — optimizer.py
- Melchi           — visualizer.py & notebook

License: MIT
"""

from .distances import euclidean_distance, signal_strength, build_signal_matrix
from .matrices import build_adjacency_matrix, build_degree_matrix, build_laplacian
from .spectral import compute_eigenvalues, fiedler_value, connectivity_report
from .optimizer import evaluate_ap_placement, find_best_placement
from .visualizer import plot_network, plot_eigenvalue_comparison, plot_signal_heatmap

__version__ = "1.0.0"
__all__ = [
    "euclidean_distance",
    "signal_strength",
    "build_signal_matrix",
    "build_adjacency_matrix",
    "build_degree_matrix",
    "build_laplacian",
    "compute_eigenvalues",
    "fiedler_value",
    "connectivity_report",
    "evaluate_ap_placement",
    "find_best_placement",
    "plot_network",
    "plot_eigenvalue_comparison",
    "plot_signal_heatmap",
]
