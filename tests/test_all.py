"""
tests/test_all.py — Unit Tests for WiFi Optimization Package
=============================================================
Run with:  pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wifi_optimization.distances import euclidean_distance, signal_strength, build_signal_matrix
from wifi_optimization.matrices import build_adjacency_matrix, build_degree_matrix, build_laplacian
from wifi_optimization.spectral import fiedler_value, count_connected_components, connectivity_report
from wifi_optimization.optimizer import evaluate_ap_placement, find_best_placement


# ── Fixtures ──────────────────────────────────────────────────────────────────

BUILDINGS = {
    "Library": (0, 0),
    "JQB":     (100, 0),
    "Hostel":  (50, 87),
}

EXISTING_APS = {
    "AP_A": (25, 0),
    "AP_B": (75, 43),
}

CANDIDATES = {
    "AP_C": (50, 0),
}


# ── distances.py ──────────────────────────────────────────────────────────────

class TestDistances:
    def test_euclidean_basic(self):
        assert euclidean_distance((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_euclidean_same_point(self):
        assert euclidean_distance((5, 5), (5, 5)) == 0.0

    def test_euclidean_horizontal(self):
        assert euclidean_distance((0, 0), (100, 0)) == pytest.approx(100.0)

    def test_signal_strength_known(self):
        # d=25 → S = 10000/625 = 16
        assert signal_strength(25, S0=10_000) == pytest.approx(16.0)

    def test_signal_strength_inverse_square(self):
        s1 = signal_strength(10)
        s2 = signal_strength(20)
        assert s1 == pytest.approx(4 * s2, rel=1e-5)

    def test_signal_strength_min_dist(self):
        # Should not blow up at d=0
        s = signal_strength(0, S0=10_000, min_dist=1.0)
        assert s == pytest.approx(10_000.0)

    def test_signal_matrix_shape(self):
        S, b_names, ap_names = build_signal_matrix(BUILDINGS, EXISTING_APS)
        assert S.shape == (len(BUILDINGS), len(EXISTING_APS))

    def test_signal_matrix_nonnegative(self):
        S, _, _ = build_signal_matrix(BUILDINGS, EXISTING_APS)
        assert (S >= 0).all()


# ── matrices.py ───────────────────────────────────────────────────────────────

class TestMatrices:
    def setup_method(self):
        S, b_names, ap_names = build_signal_matrix(BUILDINGS, EXISTING_APS)
        self.S = S
        self.b_names = b_names
        self.ap_names = ap_names
        self.A, self.node_names = build_adjacency_matrix(S, b_names, ap_names)
        self.D = build_degree_matrix(self.A)
        self.L = build_laplacian(self.A)

    def test_adjacency_shape(self):
        n = len(BUILDINGS) + len(EXISTING_APS)
        assert self.A.shape == (n, n)

    def test_adjacency_symmetric(self):
        assert np.allclose(self.A, self.A.T)

    def test_adjacency_no_self_loops(self):
        assert np.allclose(np.diag(self.A), 0)

    def test_degree_diagonal(self):
        off_diag = self.D - np.diag(np.diag(self.D))
        assert np.allclose(off_diag, 0)

    def test_degree_positive(self):
        assert (np.diag(self.D) > 0).all()

    def test_laplacian_row_sums_zero(self):
        row_sums = self.L.sum(axis=1)
        assert np.allclose(row_sums, 0, atol=1e-10)

    def test_laplacian_symmetric(self):
        assert np.allclose(self.L, self.L.T)

    def test_laplacian_equals_D_minus_A(self):
        assert np.allclose(self.L, self.D - self.A)


# ── spectral.py ───────────────────────────────────────────────────────────────

class TestSpectral:
    def setup_method(self):
        S, b_names, ap_names = build_signal_matrix(BUILDINGS, EXISTING_APS)
        A, self.node_names = build_adjacency_matrix(S, b_names, ap_names)
        self.L = build_laplacian(A)

    def test_smallest_eigenvalue_is_zero(self):
        from wifi_optimization.spectral import compute_eigenvalues
        eigs, _ = compute_eigenvalues(self.L)
        assert eigs[0] == pytest.approx(0.0, abs=1e-8)

    def test_all_eigenvalues_nonneg(self):
        from wifi_optimization.spectral import compute_eigenvalues
        eigs, _ = compute_eigenvalues(self.L)
        assert (eigs >= -1e-9).all()

    def test_fiedler_value_positive(self):
        lam2 = fiedler_value(self.L)
        assert lam2 > 0

    def test_connected_components_one(self):
        n_comp = count_connected_components(self.L)
        assert n_comp == 1

    def test_disconnected_graph(self):
        # Two isolated nodes → L = [[0,0],[0,0]], two components
        L_disc = np.zeros((2, 2))
        assert count_connected_components(L_disc) == 2

    def test_connectivity_report_keys(self):
        report = connectivity_report(self.L, self.node_names)
        for key in ["label", "eigenvalues", "fiedler_value", "fiedler_vector",
                    "n_components", "is_connected", "summary"]:
            assert key in report

    def test_connected_report_is_connected(self):
        report = connectivity_report(self.L, self.node_names)
        assert report["is_connected"] is True


# ── optimizer.py ──────────────────────────────────────────────────────────────

class TestOptimizer:
    def test_evaluate_ap_placement_improves(self):
        result = evaluate_ap_placement(
            BUILDINGS, EXISTING_APS, "AP_C", (50, 0)
        )
        assert result["fiedler_after"] > result["fiedler_before"]
        assert result["improvement"] > 0
        assert result["pct_change"] > 0

    def test_find_best_placement_returns_best(self):
        best_name, results = find_best_placement(
            BUILDINGS, EXISTING_APS, CANDIDATES, verbose=False
        )
        assert best_name == "AP_C"
        assert "AP_C" in results

    def test_evaluate_result_keys(self):
        result = evaluate_ap_placement(
            BUILDINGS, EXISTING_APS, "AP_C", (50, 0)
        )
        for key in ["name", "coords", "fiedler_before", "fiedler_after",
                    "improvement", "pct_change"]:
            assert key in result
