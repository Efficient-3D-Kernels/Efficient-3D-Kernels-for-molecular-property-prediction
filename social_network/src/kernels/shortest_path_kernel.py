"""
Shortest-Path Graph Kernel
--------------------------

Computes a normalized shortest-path kernel similarity between
two NetworkX graphs.

The kernel compares the distributions of shortest-path lengths
between node pairs.

For weighted graphs, edge weights are used as distances.

Output:
    similarity : normalized similarity in [0, 1]
"""

import math
import networkx as nx
from collections import Counter


# ============================================================
# SHORTEST-PATH FEATURE EXTRACTION
# ============================================================

def _shortest_path_features(G):
    """
    Extract shortest-path length features from a graph.

    For every pair of reachable nodes, the shortest-path distance
    is computed and stored as a histogram.

    Edge weights are treated as distances.
    """

    features = Counter()

    # All-pairs shortest paths using edge weights
    shortest_paths = nx.all_pairs_dijkstra_path_length(
        G,
        weight="weight"
    )

    for source, distances in shortest_paths:

        for target, distance in distances.items():

            # Ignore self-distance
            if source == target:
                continue

            # Avoid counting every pair twice in an undirected graph
            if source < target:

                # Round floating point values so that
                # numerically equivalent distances are grouped.
                distance = round(float(distance), 6)

                features[distance] += 1

    return features


# ============================================================
# KERNEL CALCULATION
# ============================================================

def _dot_product(counter_a, counter_b):
    """
    Sparse-vector dot product.
    """

    common_keys = (
        set(counter_a.keys())
        &
        set(counter_b.keys())
    )

    return sum(
        counter_a[key] * counter_b[key]
        for key in common_keys
    )


def shortest_path_kernel_similarity(G, H):
    """
    Compute normalized shortest-path kernel similarity.

    The result is approximately in [0, 1].

    A value near 1 means that the two graphs have very
    similar shortest-path distributions.
    """

    features_G = _shortest_path_features(G)
    features_H = _shortest_path_features(H)

    numerator = _dot_product(
        features_G,
        features_H
    )

    norm_G = _dot_product(
        features_G,
        features_G
    )

    norm_H = _dot_product(
        features_H,
        features_H
    )

    if norm_G == 0 or norm_H == 0:
        return 0.0

    similarity = numerator / math.sqrt(
        norm_G * norm_H
    )

    # Numerical protection
    similarity = max(
        0.0,
        min(1.0, similarity)
    )

    return float(similarity)


# ============================================================
# BENCHMARK WRAPPER
# ============================================================

def run_shortest_path_kernel(G, H):
    """
    Wrapper used by the benchmark script.
    """

    similarity = shortest_path_kernel_similarity(
        G,
        H
    )

    return {
        "method": "Shortest Path",
        "similarity": similarity
    }


# ============================================================
# SANITY TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SHORTEST-PATH KERNEL TEST")
    print("=" * 60)

    # Graph 1
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
        2,
        3,
        weight=3.0
    )

    # Graph 2
    # Same structure and same weights,
    # but different node IDs.
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
        12,
        13,
        weight=3.0
    )

    result = run_shortest_path_kernel(
        G,
        H
    )

    print(
        f"Similarity: "
        f"{result['similarity']:.6f}"
    )

    print("\nExpected similarity: 1.000000")