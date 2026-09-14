"""
Weisfeiler-Lehman Optimal Assignment (WL-OA) Kernel
-----------------------------------------------------

Computes a WL-based optimal-assignment similarity between
two graphs.

The method:
    1. Generates WL labels for every node.
    2. Represents each node using its WL label.
    3. Computes a node-to-node similarity matrix.
    4. Solves an optimal assignment problem.
    5. Normalizes the resulting assignment score.

This gives a graph similarity based on explicit node
assignment, making it conceptually closer to graph matching
than the standard WL subtree kernel.
"""

import math
import networkx as nx

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


# ============================================================
# INITIAL LABELS
# ============================================================

def _initial_labels(G):
    """
    Initial node labels based on weighted degree.
    """

    labels = {}

    for node in G.nodes():

        weighted_degree = G.degree(
            node,
            weight="weight"
        )

        labels[node] = (
            "deg:"
            + str(
                round(
                    float(weighted_degree),
                    6
                )
            )
        )

    return labels


# ============================================================
# WL REFINEMENT
# ============================================================

def _refine_labels(G, labels):
    """
    Perform one WL refinement step.
    """

    signatures = {}

    for node in G.nodes():

        neighborhood = []

        for neighbor in G.neighbors(node):

            weight = float(
                G[node][neighbor].get(
                    "weight",
                    1.0
                )
            )

            neighborhood.append(
                (
                    labels[neighbor],
                    round(weight, 6)
                )
            )

        neighborhood.sort(
            key=str
        )

        signatures[node] = (
            labels[node],
            tuple(neighborhood)
        )

    # --------------------------------------------------------
    # Compress signatures
    # --------------------------------------------------------

    unique_signatures = {}

    labels_new = {}

    next_label = 0

    for node in G.nodes():

        signature = signatures[node]

        if signature not in unique_signatures:

            unique_signatures[
                signature
            ] = next_label

            next_label += 1

        labels_new[node] = (
            unique_signatures[
                signature
            ]
        )

    return labels_new


# ============================================================
# WL LABEL HISTORY
# ============================================================

def _wl_label_history(G, h=3):
    """
    Generate WL labels for every iteration.

    Returns:
        dictionary:
            node -> list of labels
    """

    labels = _initial_labels(G)

    history = {
        node: [labels[node]]
        for node in G.nodes()
    }

    for _ in range(h):

        labels = _refine_labels(
            G,
            labels
        )

        for node in G.nodes():

            history[node].append(
                labels[node]
            )

    return history


# ============================================================
# NODE SIMILARITY
# ============================================================

def _node_similarity(
    history_G,
    history_H,
    node_G,
    node_H
):
    """
    Calculate similarity between two nodes based on their
    WL label histories.

    The score is the fraction of WL levels at which the
    two nodes have matching labels.
    """

    labels_G = history_G[node_G]
    labels_H = history_H[node_H]

    levels = min(
        len(labels_G),
        len(labels_H)
    )

    if levels == 0:
        return 0.0

    matches = 0

    for i in range(levels):

        if labels_G[i] == labels_H[i]:

            matches += 1

    return matches / levels


# ============================================================
# WL-OA KERNEL
# ============================================================

def wl_oa_kernel_similarity(
    G,
    H,
    h=3
):
    """
    Compute WL Optimal Assignment similarity.

    Requires scipy for the Hungarian assignment.
    """

    if linear_sum_assignment is None:

        raise ImportError(
            "WL-OA requires scipy. "
            "Install it using: pip install scipy"
        )

    nodes_G = list(
        G.nodes()
    )

    nodes_H = list(
        H.nodes()
    )

    # --------------------------------------------------------
    # WL histories
    # --------------------------------------------------------

    history_G = _wl_label_history(
        G,
        h=h
    )

    history_H = _wl_label_history(
        H,
        h=h
    )

    # --------------------------------------------------------
    # Build similarity matrix
    # --------------------------------------------------------

    similarity_matrix = []

    for node_G in nodes_G:

        row = []

        for node_H in nodes_H:

            score = _node_similarity(
                history_G,
                history_H,
                node_G,
                node_H
            )

            row.append(score)

        similarity_matrix.append(row)

    # --------------------------------------------------------
    # Convert similarity to cost
    # --------------------------------------------------------

    cost_matrix = [
        [
            1.0 - value
            for value in row
        ]
        for row in similarity_matrix
    ]

    # --------------------------------------------------------
    # Hungarian optimal assignment
    # --------------------------------------------------------

    row_ind, col_ind = (
        linear_sum_assignment(
            cost_matrix
        )
    )

    assignment_score = 0.0

    for r, c in zip(
        row_ind,
        col_ind
    ):

        assignment_score += (
            similarity_matrix[r][c]
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    denominator = min(
        len(nodes_G),
        len(nodes_H)
    )

    if denominator == 0:
        return 0.0

    similarity = (
        assignment_score /
        denominator
    )

    similarity = max(
        0.0,
        min(
            1.0,
            similarity
        )
    )

    return float(
        similarity
    )


# ============================================================
# BENCHMARK WRAPPER
# ============================================================

def run_wl_oa_kernel(
    G,
    H,
    h=3
):
    """
    Wrapper used by the benchmark.
    """

    similarity = (
        wl_oa_kernel_similarity(
            G,
            H,
            h=h
        )
    )

    return {
        "method": "WL-OA",
        "similarity": similarity,
        "WL Iterations": h
    }


# ============================================================
# SANITY TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print(
        "WL OPTIMAL ASSIGNMENT KERNEL TEST"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Graph 1
    # --------------------------------------------------------

    G = nx.Graph()

    G.add_edge(
        0,
        1,
        weight=2.0
    )

    G.add_edge(
        1,
        2,
        weight=1.0
    )

    G.add_edge(
        0,
        2,
        weight=3.0
    )

    # --------------------------------------------------------
    # Graph 2
    # --------------------------------------------------------

    H = nx.Graph()

    H.add_edge(
        10,
        11,
        weight=2.0
    )

    H.add_edge(
        11,
        12,
        weight=1.0
    )

    H.add_edge(
        10,
        12,
        weight=3.0
    )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    result = run_wl_oa_kernel(
        G,
        H,
        h=3
    )

    print(
        f"Similarity: "
        f"{result['similarity']:.6f}"
    )

    print(
        "\nExpected similarity: 1.000000"
    )