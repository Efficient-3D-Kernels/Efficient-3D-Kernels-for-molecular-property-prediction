"""
Graphlet Kernel
---------------

Computes a graphlet-based similarity between two NetworkX graphs.

The implementation counts connected graphlets of size 3:
    1. 3-node path
    2. 3-node triangle

For weighted graphs, edge weights are incorporated into the
graphlet signature through the total weight of the graphlet.

The final similarity is cosine similarity between the
graphlet feature vectors.
"""

import math
import networkx as nx
from collections import Counter


# ============================================================
# GRAPHLET FEATURE EXTRACTION
# ============================================================

def _graphlet_features(G):
    """
    Extract 3-node graphlet features.

    For every combination of three nodes, determine whether
    the induced subgraph is:

        - a 3-node path
        - a triangle

    The total edge weight is also included in the signature.
    """

    features = Counter()

    nodes = list(G.nodes())

    n = len(nodes)

    # --------------------------------------------------------
    # Examine every 3-node combination
    # --------------------------------------------------------

    for i in range(n):

        u = nodes[i]

        for j in range(i + 1, n):

            v = nodes[j]

            for k in range(j + 1, n):

                w = nodes[k]

                selected = [u, v, w]

                # Induced subgraph
                subgraph = G.subgraph(
                    selected
                )

                number_of_edges = (
                    subgraph.number_of_edges()
                )

                # ------------------------------------------------
                # Ignore disconnected 3-node subgraphs
                # ------------------------------------------------

                if number_of_edges < 2:
                    continue

                # ------------------------------------------------
                # Calculate total weight
                # ------------------------------------------------

                total_weight = 0.0

                for a, b in subgraph.edges():

                    weight = float(
                        subgraph[a][b].get(
                            "weight",
                            1.0
                        )
                    )

                    total_weight += weight

                total_weight = round(
                    total_weight,
                    6
                )

                # ------------------------------------------------
                # Triangle
                # ------------------------------------------------

                if number_of_edges == 3:

                    features[
                        (
                            "triangle",
                            total_weight
                        )
                    ] += 1

                # ------------------------------------------------
                # 3-node path
                # ------------------------------------------------

                elif number_of_edges == 2:

                    features[
                        (
                            "path3",
                            total_weight
                        )
                    ] += 1

    return features


# ============================================================
# VECTOR OPERATIONS
# ============================================================

def _dot_product(
    counter_a,
    counter_b
):
    """
    Sparse-vector dot product.
    """

    common_keys = (
        set(counter_a.keys())
        &
        set(counter_b.keys())
    )

    return sum(
        counter_a[key] *
        counter_b[key]
        for key in common_keys
    )


# ============================================================
# GRAPHLET KERNEL
# ============================================================

def graphlet_kernel_similarity(
    G,
    H
):
    """
    Compute normalized graphlet similarity.

    Returns:
        float approximately in [0, 1]
    """

    features_G = _graphlet_features(
        G
    )

    features_H = _graphlet_features(
        H
    )

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

    similarity = (
        numerator /
        math.sqrt(
            norm_G *
            norm_H
        )
    )

    similarity = max(
        0.0,
        min(1.0, similarity)
    )

    return float(
        similarity
    )


# ============================================================
# BENCHMARK WRAPPER
# ============================================================

def run_graphlet_kernel(
    G,
    H
):
    """
    Wrapper used by the benchmark.
    """

    similarity = (
        graphlet_kernel_similarity(
            G,
            H
        )
    )

    return {
        "method": "Graphlet",
        "similarity": similarity
    }


# ============================================================
# SANITY TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GRAPHLET KERNEL TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Graph 1: triangle
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
    # Graph 2: same triangle with different node IDs
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

    result = run_graphlet_kernel(
        G,
        H
    )

    print(
        f"Similarity: "
        f"{result['similarity']:.6f}"
    )

    print(
        "\nExpected similarity: 1.000000"
    )