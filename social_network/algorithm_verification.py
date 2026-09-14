"""
Algorithm Verification for WCS-GNCCP

This script verifies:

1. Known-permutation graph matching
2. Relabeling invariance
3. Numerical gradient correctness
4. Sinkhorn projection feasibility

The purpose is algorithmic validation before running large
real-world social-network experiments.
"""

import numpy as np
import networkx as nx

from src.baseline_gnccp import run_baseline_gnccp
from src.proposed_gnccp import (
    run_proposed_gnccp,
    compute_wcs_gradient,
    compute_wcs_objective,
    sinkhorn_project,
)


# ============================================================
# Utility functions
# ============================================================

def graph_to_adjacency(G):
    """
    Convert a NetworkX graph into a weighted adjacency matrix.

    Nodes are ordered numerically.
    """
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(
        G,
        nodelist=nodes,
        weight="weight",
        dtype=float
    )
    return A, nodes


def graph_features(G):
    """
    Compute invariant structural node features.

    Features:
        1. unweighted degree
        2. weighted degree
        3. clustering coefficient
    """

    nodes = sorted(G.nodes())

    degree = dict(G.degree())

    weighted_degree = dict(G.degree(weight="weight"))

    clustering = nx.clustering(
        G,
        weight="weight"
    )

    features = []

    for node in nodes:
        features.append([
            float(degree[node]),
            float(weighted_degree[node]),
            float(clustering[node])
        ])

    return np.asarray(features, dtype=float)


def normalize_features(F):
    """
    Column-wise min-max normalization.
    """

    F = F.copy()

    minimum = F.min(axis=0)
    maximum = F.max(axis=0)

    denominator = maximum - minimum

    denominator[denominator == 0] = 1.0

    return (F - minimum) / denominator


def build_cost_matrix(FG, FH):
    """
    Euclidean node-feature cost matrix.

    C[i,j] = distance between node i in G
             and node j in H.
    """

    FG = normalize_features(FG)
    FH = normalize_features(FH)

    M = FG.shape[0]
    N = FH.shape[0]

    C = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            C[i, j] = np.linalg.norm(
                FG[i] - FH[j]
            )

    # Min-max normalization
    cmin = C.min()
    cmax = C.max()

    if cmax > cmin:
        C = (C - cmin) / (cmax - cmin)

    return C


def extract_matching(X):
    """
    Convert continuous/discrete matching matrix into
    row -> column assignments.
    """

    return np.argmax(X, axis=1)


def matching_accuracy(predicted, expected):
    """
    Percentage of correctly recovered correspondences.
    """

    predicted = np.asarray(predicted)
    expected = np.asarray(expected)

    return (
        np.mean(predicted == expected) * 100.0
    )


# ============================================================
# Construct a distinctive weighted social-style graph
# ============================================================

def create_test_graph():
    """
    Construct a small weighted graph with a distinctive structure.

    The graph is intentionally not random so that the verification
    is reproducible.
    """

    G = nx.Graph()

    # Add six users
    G.add_nodes_from(range(6))

    # Weighted social interactions
    edges = [
        (0, 1, 5.0),
        (0, 2, 2.0),
        (1, 2, 4.0),
        (1, 3, 3.0),
        (2, 4, 6.0),
        (3, 4, 2.0),
        (3, 5, 5.0),
        (4, 5, 3.0),
    ]

    G.add_weighted_edges_from(edges)

    return G


# ============================================================
# Create a relabelled copy
# ============================================================

def create_relabelled_graph(G, permutation):
    """
    Relabel G using:

        original node i -> new node permutation[i]

    Therefore, the expected correspondence from G to H is:

        i -> permutation[i]
    """

    mapping = {
        i: int(permutation[i])
        for i in G.nodes()
    }

    H = nx.relabel_nodes(
        G,
        mapping,
        copy=True
    )

    return H


# ============================================================
# Test 1: Known permutation
# ============================================================

def test_known_permutation():

    print("\n" + "=" * 70)
    print("TEST 1: KNOWN-PERMUTATION GRAPH MATCHING")
    print("=" * 70)

    G = create_test_graph()

    permutation = np.array([
        3,
        5,
        0,
        4,
        1,
        2
    ])

    H = create_relabelled_graph(
        G,
        permutation
    )

    AG, nodes_G = graph_to_adjacency(G)
    AH, nodes_H = graph_to_adjacency(H)

    FG = graph_features(G)
    FH = graph_features(H)

    C = build_cost_matrix(
        FG,
        FH
    )

    expected = permutation.copy()

    print("\nExpected correspondence:")
    for i in range(len(expected)):
        print(
            f"  G node {i} -> H node {expected[i]}"
        )

    print("\nRunning baseline WCS-GNCCP...")

    baseline_result = run_baseline_gnccp(
        AG,
        AH,
        C,
        matching_size=6
    )

    baseline_X = baseline_result[0]

    baseline_matching = extract_matching(
        baseline_X
    )

    baseline_accuracy = matching_accuracy(
        baseline_matching,
        expected
    )

    print(
        f"Baseline accuracy: "
        f"{baseline_accuracy:.2f}%"
    )

    print("\nRunning proposed WCS-GNCCP...")

    proposed_result = run_proposed_gnccp(
        AG,
        AH,
        C,
        matching_size=6
    )

    proposed_X = proposed_result[0]

    proposed_matching = extract_matching(
        proposed_X
    )

    proposed_accuracy = matching_accuracy(
        proposed_matching,
        expected
    )

    print(
        f"Proposed accuracy: "
        f"{proposed_accuracy:.2f}%"
    )

    print("\nBaseline predicted:")
    print(baseline_matching)

    print("\nProposed predicted:")
    print(proposed_matching)

    print("\nExpected:")
    print(expected)

    return {
        "G": G,
        "H": H,
        "AG": AG,
        "AH": AH,
        "C": C,
        "expected": expected,
        "baseline_matching": baseline_matching,
        "proposed_matching": proposed_matching,
        "baseline_accuracy": baseline_accuracy,
        "proposed_accuracy": proposed_accuracy,
    }


# ============================================================
# Test 2: Relabeling invariance
# ============================================================

def test_relabeling_invariance():

    print("\n" + "=" * 70)
    print("TEST 2: RELABELING INVARIANCE")
    print("=" * 70)

    G = create_test_graph()

    # First permutation
    permutation_1 = np.array([
        3,
        5,
        0,
        4,
        1,
        2
    ])

    H1 = create_relabelled_graph(
        G,
        permutation_1
    )

    AG, _ = graph_to_adjacency(G)
    AH1, _ = graph_to_adjacency(H1)

    FG = graph_features(G)
    FH1 = graph_features(H1)

    C1 = build_cost_matrix(
        FG,
        FH1
    )

    result_1 = run_proposed_gnccp(
        AG,
        AH1,
        C1,
        matching_size=6
    )

    X1 = result_1[0]

    matching_1 = extract_matching(X1)

    accuracy_1 = matching_accuracy(
        matching_1,
        permutation_1
    )

    print("\nOriginal relabeling:")
    print("Expected:", permutation_1)
    print("Found:   ", matching_1)
    print(
        f"Accuracy: {accuracy_1:.2f}%"
    )

    # --------------------------------------------------------
    # Apply another relabeling to H
    # --------------------------------------------------------

    second_permutation = np.array([
        2,
        0,
        5,
        1,
        4,
        3
    ])

    H2 = nx.relabel_nodes(
        H1,
        {
            i: int(second_permutation[i])
            for i in H1.nodes()
        },
        copy=True
    )

    AH2, _ = graph_to_adjacency(H2)

    FH2 = graph_features(H2)

    C2 = build_cost_matrix(
        FG,
        FH2
    )

    result_2 = run_proposed_gnccp(
        AG,
        AH2,
        C2,
        matching_size=6
    )

    X2 = result_2[0]

    matching_2 = extract_matching(X2)

    # --------------------------------------------------------
    # Expected mapping after two relabelings
    # --------------------------------------------------------

    expected_2 = np.zeros(6, dtype=int)

    for i in range(6):
        intermediate = permutation_1[i]
        expected_2[i] = second_permutation[
            intermediate
        ]

    accuracy_2 = matching_accuracy(
        matching_2,
        expected_2
    )

    print("\nAfter additional relabeling:")
    print("Expected:", expected_2)
    print("Found:   ", matching_2)

    print(
        f"Accuracy: {accuracy_2:.2f}%"
    )

    # --------------------------------------------------------
    # Objective-value consistency
    # --------------------------------------------------------

    objective_1 = result_1[1]

    objective_2 = result_2[1]

    print("\nObjective comparison:")
    print(
        f"First labeling : {objective_1:.10f}"
    )
    print(
        f"Second labeling: {objective_2:.10f}"
    )

    objective_difference = abs(
        objective_1 - objective_2
    )

    print(
        f"Absolute difference: "
        f"{objective_difference:.10e}"
    )

    invariance_pass = (
        objective_difference < 1e-5
    )

    print(
        "\nRelabeling invariance:",
        "PASS" if invariance_pass else "CHECK"
    )

    return {
        "accuracy_before": accuracy_1,
        "accuracy_after": accuracy_2,
        "objective_difference": objective_difference,
        "pass": invariance_pass,
    }


# ============================================================
# Test 3: Numerical gradient verification
# ============================================================

def test_gradient():

    print("\n" + "=" * 70)
    print("TEST 3: NUMERICAL GRADIENT VERIFICATION")
    print("=" * 70)

    G = create_test_graph()

    permutation = np.array([
        3,
        5,
        0,
        4,
        1,
        2
    ])

    H = create_relabelled_graph(
        G,
        permutation
    )

    AG, _ = graph_to_adjacency(G)
    AH, _ = graph_to_adjacency(H)

    FG = graph_features(G)
    FH = graph_features(H)

    C = build_cost_matrix(
        FG,
        FH
    )

    M = AG.shape[0]
    N = AH.shape[0]

    rng = np.random.default_rng(42)

    # Small random continuous matrix
    X = rng.random(
        (M, N)
    )

    # Keep X numerically well-scaled
    X = X / X.sum()

    alpha = 0.5

    analytical = compute_wcs_gradient(
        X,
        AG,
        AH,
        C,
        alpha
    )

    numerical = np.zeros_like(X)

    epsilon = 1e-6

    # Central finite differences
    for i in range(M):

        for j in range(N):

            X_plus = X.copy()
            X_minus = X.copy()

            X_plus[i, j] += epsilon
            X_minus[i, j] -= epsilon

            f_plus = compute_wcs_objective(
                X_plus,
                AG,
                AH,
                C,
                alpha
            )

            f_minus = compute_wcs_objective(
                X_minus,
                AG,
                AH,
                C,
                alpha
            )

            numerical[i, j] = (
                f_plus - f_minus
            ) / (2.0 * epsilon)

    absolute_error = np.linalg.norm(
        analytical - numerical
    )

    relative_error = (
        absolute_error /
        max(
            1.0,
            np.linalg.norm(numerical)
        )
    )

    print(
        f"Absolute gradient error: "
        f"{absolute_error:.10e}"
    )

    print(
        f"Relative gradient error: "
        f"{relative_error:.10e}"
    )

    passed = relative_error < 1e-5

    print(
        "\nGradient verification:",
        "PASS" if passed else "FAIL"
    )

    return {
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "pass": passed,
    }


# ============================================================
# Test 4: Sinkhorn projection
# ============================================================

def test_sinkhorn_projection():

    print("\n" + "=" * 70)
    print("TEST 4: SINKHORN PROJECTION")
    print("=" * 70)

    rng = np.random.default_rng(123)

    M = 6
    N = 6
    L = 6

    X = rng.random(
        (M, N)
    )

    projected = sinkhorn_project(
    X,
    L,
    iterations=1000
)

    row_sums = projected.sum(axis=1)
    col_sums = projected.sum(axis=0)

    target_row = L / M
    target_col = L / N

    row_error = np.max(
        np.abs(
            row_sums - target_row
        )
    )

    col_error = np.max(
        np.abs(
            col_sums - target_col
        )
    )

    print("\nProjected matrix:")
    print(
        np.round(projected, 4)
    )

    print("\nRow sums:")
    print(
        np.round(row_sums, 6)
    )

    print("\nColumn sums:")
    print(
        np.round(col_sums, 6)
    )

    print(
        f"\nMaximum row-sum error: "
        f"{row_error:.10e}"
    )

    print(
        f"Maximum column-sum error: "
        f"{col_error:.10e}"
    )

    passed = (
        row_error < 1e-5
        and col_error < 1e-5
    )

    print(
        "\nSinkhorn feasibility:",
        "PASS" if passed else "CHECK"
    )

    return {
        "row_error": row_error,
        "column_error": col_error,
        "pass": passed,
    }


# ============================================================
# Main verification routine
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("WCS-GNCCP ALGORITHM VERIFICATION")
    print("=" * 70)

    results = {}

    # --------------------------------------------------------
    # 1. Known permutation
    # --------------------------------------------------------

    known_result = test_known_permutation()

    results["known_permutation"] = known_result

    # --------------------------------------------------------
    # 2. Relabeling invariance
    # --------------------------------------------------------

    invariance_result = test_relabeling_invariance()

    results["relabeling_invariance"] = (
        invariance_result
    )

    # --------------------------------------------------------
    # 3. Gradient
    # --------------------------------------------------------

    gradient_result = test_gradient()

    results["gradient"] = gradient_result

    # --------------------------------------------------------
    # 4. Sinkhorn
    # --------------------------------------------------------

    sinkhorn_result = test_sinkhorn_projection()

    results["sinkhorn"] = sinkhorn_result

    # ========================================================
    # Final summary
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 70)

    print(
        "\nKnown-permutation baseline accuracy: "
        f"{known_result['baseline_accuracy']:.2f}%"
    )

    print(
        "Known-permutation proposed accuracy: "
        f"{known_result['proposed_accuracy']:.2f}%"
    )

    print(
        "\nRelabeling invariance: "
        f"{'PASS' if invariance_result['pass'] else 'CHECK'}"
    )

    print(
        "Gradient verification: "
        f"{'PASS' if gradient_result['pass'] else 'FAIL'}"
    )

    print(
        "Sinkhorn feasibility: "
        f"{'PASS' if sinkhorn_result['pass'] else 'CHECK'}"
    )

    # --------------------------------------------------------
    # Overall status
    # --------------------------------------------------------

    overall_pass = (
        invariance_result["pass"]
        and gradient_result["pass"]
        and sinkhorn_result["pass"]
    )

    print("\n" + "=" * 70)

    if overall_pass:
        print("OVERALL STATUS: PASS")
        print(
            "The implementation passes the core "
            "algorithmic verification tests."
        )
    else:
        print("OVERALL STATUS: CHECK")
        print(
            "At least one verification test requires "
            "inspection before final experiments."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()